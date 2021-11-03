import numpy as np
import torch
from models.model import VGG_Deblur
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from opts.test_opt import args
import os
import cv2
import torch.nn as nn
from utils import index

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

norm_val=(2**8)-1
src_mean = 0
test_path = './dpdd_datasets/'

def creat_list(path):
    gt_list = []
    iml_list = []
    imr_list = []
    r_flow_list = []
    l_flow_list = []
    iml_path = path + 'dpdd_16bit/test/test_l/source/'
    imr_path = path + 'dpdd_16bit/test/test_r/source/'
    gt_path = path + 'dpdd_16bit/test/test_c/target/'

    for _,_,fnames in sorted(os.walk(gt_path)):
        for fname in fnames:
            gt_list.append(gt_path+fname)
            iml_list.append(iml_path+fname)
            imr_list.append(imr_path+fname)

    return gt_list,iml_list,imr_list

to_tensor = transforms.ToTensor()            

class TestDataset(Dataset):
    def __init__(self,gt_list,iml_list,imr_list):
        self.gt_list = gt_list
        self.iml_list = iml_list
        self.imr_list = imr_list

    def __getitem__(self, index):
        
        gt = cv2.imread(self.gt_list[index])
        while gt is None:
            gt = cv2.imread(self.gt_list[index])
        
        im_l = cv2.imread(self.iml_list[index])
        while im_l is None:
            im_l = cv2.imread(self.iml_list[index])
        
        im_r = cv2.imread(self.imr_list[index])
        while im_r is None:
            im_r = cv2.imread(self.imr_list[index])
            
        # sha0 = gt.shape[0]//2
        # sha1 = gt.shape[1]//2
        # im_l = cv2.resize(im_l, (sha1, sha0), cv2.INTER_CUBIC)
        # im_r = cv2.resize(im_r, (sha1, sha0), cv2.INTER_CUBIC)
        # gt = cv2.resize(gt, (sha1, sha0), cv2.INTER_CUBIC)
        
        gt = np.float32((gt-src_mean)/norm_val)
        im_l = np.float32((im_l-src_mean)/norm_val)
        im_r = np.float32((im_r-src_mean)/norm_val)
        
        gt=torch.from_numpy(gt).permute(2,0,1)
        im_l=torch.from_numpy(im_l).permute(2,0,1)
        im_r=torch.from_numpy(im_r).permute(2,0,1)
        
        return gt, im_l, im_r
    
    def __len__(self):
        return len(self.gt_list)


def test_dataset(test_loader,save_path=None):
    ssim_sum = 0
    mse_sum = 0
    psnr_sum = 0
    mae_sum = 0
    
    for j, (gt, im_l, im_r) in enumerate(test_loader):
        gt = gt.cuda()
        im_l = im_l.cuda()
        im_r = im_r.cuda()
        with torch.no_grad():   
            im_r.requires_grad_(False)            
            im_l.requires_grad_(False) 
            
            output = deblur_model(im_l,im_r)
            
            output = np.uint8(output[0,...].permute(1,2,0).cpu().detach().numpy()*255)
            gt = np.uint8(gt[0,...].permute(1,2,0).cpu().detach().numpy()*255)
            
            if save_path:
                if not os.path.exists(save_path):
                    os.mkdir(save_path) 
                cv2.imwrite("%s/%s_l.png"%(save_path,j),(im_l[0,...].permute(1,2,0).cpu().detach().numpy()*norm_val).astype(np.uint8))  
                cv2.imwrite("%s/%s_r.png"%(save_path,j),(im_r[0,...].permute(1,2,0).cpu().detach().numpy()*norm_val).astype(np.uint8)) 
                cv2.imwrite("%s/%s_gt.png"%(save_path,j),gt)
                cv2.imwrite("%s/%s_output.png"%(save_path,j),output)
            
            mse, psnr, ssim = index.MSE_PSNR_SSIM(np.float32(gt)/255.0, np.float32(output)/255.0)
            mae = index.MAE(np.float32(gt)/255.0, np.float32(output)/255.0)
            ssim_sum += ssim
            mse_sum += mse
            psnr_sum += psnr
            mae_sum += mae
            print('SSIM:',ssim,'PSNR:',psnr,'MSE:',mse,'MAE:',mae)
                
    print(len(test_loader),'SSIM:',ssim_sum/len(test_loader),'PSNR:',psnr_sum/len(test_loader),'MSE:',mse_sum/len(test_loader),'MAE:',mae_sum/len(test_loader))
    return len(test_loader),ssim_sum/len(test_loader),psnr_sum/len(test_loader),mse_sum/len(test_loader),mae_sum/len(test_loader)

def add(num_,ssim_sum_,psnr_sum_,lmse_sum_,ncc_sum_,
        num_test,ssim_sum_test,psnr_sum_test,lmse_sum_test,ncc_sum_test):
    return num_+num_test,ssim_sum_+ssim_sum_test,psnr_sum_+psnr_sum_test,lmse_sum_+lmse_sum_test,ncc_sum_+ncc_sum_test

gt_list,iml_list,imr_list = creat_list(test_path)
test_dpdd_dataset = TestDataset(gt_list,iml_list,imr_list)
test_loader_dpdd = torch.utils.data.DataLoader(dataset=test_dpdd_dataset,batch_size=1,shuffle=False,num_workers=args.num_workers)

deblur_model = VGG_Deblur()
deblur_model = nn.DataParallel(deblur_model)
deblur_model.cuda()
deblur_model.eval()

def test_state(state_dict):
    deblur_model.load_state_dict(state_dict)
    del(state_dict)
    save_path = None

    if args.save_result_path == True:
        save_path = './result'
    num, ssim_av,psnr_av,mse_av,mae_av = test_dataset(test_loader_dpdd,save_path)

    return ssim_av,psnr_av,mse_av,mae_av




    


