import cv2
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from utils.utils import syn
import torchvision.transforms as transforms  

to_tensor = transforms.ToTensor()    
norm_val=(2**8)-1
src_mean = 0

def Crop_img(img,ran):    
    return img[ran:ran+512,ran:ran+512]

class MyDataset(Dataset):
    def __init__(self,im_gt,im_r,im_l,crop=Crop_img):
       
        self.im_gt_list=im_gt
        self.im_r_list=im_r
        self.im_l_list=im_l
        self.crop = Crop_img
        
    def __getitem__(self, index):

        img_gt = cv2.imread(self.im_gt_list[index])
        while img_gt is None:
            img_gt = cv2.imread(self.im_gt_list[index])
        
        img_l = cv2.imread(self.im_l_list[index])
        while img_l is None:
            img_l = cv2.imread(self.im_l_list[index])
        
        img_r = cv2.imread(self.im_r_list[index])
        while img_r is None:
            img_r = cv2.imread(self.im_r_list[index])      
        
        img_gt = np.float32((img_gt-src_mean)/norm_val)
        img_l = np.float32((img_l-src_mean)/norm_val)
        img_r = np.float32((img_r-src_mean)/norm_val)
        
        new = int(np.random.randint(512,640)/2)*2        
        img_gt = cv2.resize(img_gt, (new, new), cv2.INTER_CUBIC)
        img_r = cv2.resize(img_r, (new, new), cv2.INTER_CUBIC)
        img_l = cv2.resize(img_l, (new, new), cv2.INTER_CUBIC)
        
        random_crop = int(np.random.randint(0,new-511))
        img_gt = self.crop(img_gt,random_crop)
        img_r = self.crop(img_r, random_crop)
        img_l = self.crop(img_l, random_crop)   
        
        magic = np.random.random()
            
        if magic < 0.5:
            img_gt = cv2.flip(img_gt,0)
            img_l = cv2.flip(img_l,0)
            img_r = cv2.flip(img_r,0)
            
        img_gt=torch.from_numpy(img_gt).permute(2,0,1)
        img_l=torch.from_numpy(img_l).permute(2,0,1)
        img_r=torch.from_numpy(img_r).permute(2,0,1)
    
        return img_gt,img_l,img_r

    def __len__(self):
        return len(self.im_gt_list)
        
        
        
        
        
        