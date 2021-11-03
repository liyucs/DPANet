#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 09:35:56 2019

@author: li
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model import VGG_Deblur
from dataset import MyDataset
from test_model.test_model import test_state
from tensorboardX import SummaryWriter
import numpy as np
from utils.losses import CharbonnierLoss
import cv2

EPS=1e-12
writer=SummaryWriter('summary')
norm_val=(2**8)-1
src_mean=0

class Solver:
    def __init__(self,args):
        self.args = args
        self.start_epoch=0
        self.global_step = 0

    def prepare_data(self,train_path):
        im_r_list=[]
        im_l_list=[]
        im_gt_list=[]

        path_gt= train_path+"/dpdd_16bit/train/train_c/target/"
        path_l = train_path+"/dpdd_16bit/train/train_l/source/"
        path_r = train_path+"/dpdd_16bit/train/train_r/source/"
           
        for _,_,fnames in sorted(os.walk(path_gt)):
            for fname in fnames:
                    
                im_r_list.append(path_r+fname)
                im_l_list.append(path_l+fname)
                im_gt_list.append(path_gt+fname)
                
        return im_gt_list,im_r_list,im_l_list
    
    def train_model(self):
        self.deblur_model = VGG_Deblur()
        self.deblur_model = nn.DataParallel(self.deblur_model)
        self.deblur_model.cuda()
        self.lr = self.args.lr
        self.deblur_opt = torch.optim.Adam(self.deblur_model.parameters(),lr=self.lr,)
        
        lr_=[]
        lr_.append(self.lr) #initial learning rate
        for i in range(int(self.args.num_epochs/self.args.lr_decay)):
            lr_.append(lr_[i]*0.5)

        #optional resume from a checkpoit
        if self.args.resume_file:
            if os.path.isfile(self.args.resume_file):
                print("loading checkpoint'{}'".format(self.args.resume_file))
                checkpoint = torch.load(self.args.resume_file)
                self.start_epoch = checkpoint['epoch']
                self.global_step = checkpoint['global_step']
                self.deblur_model.load_state_dict(checkpoint['G_state_dict'])
                self.deblur_opt.load_state_dict(checkpoint['G_opt'])
                del(checkpoint)
                print("'{}' loaded".format(self.args.resume_file,self.args.start_epoch))
            else:
                print("no checkpoint found at '{}'".format(self.args.resume_file))
                return 1

        # torch.backends.cudnn.benchmark = True

        im_gt,im_r,im_l=self.prepare_data(self.args.data_path) 
        train_dpdd_dataset = MyDataset(im_gt,im_r,im_l)
        train_dpdd_loader = torch.utils.data.DataLoader(dataset=train_dpdd_dataset,batch_size=self.args.batch_size,shuffle=True,
                    num_workers=self.args.load_workers)

        # train the model
        best_psnr = best_ssim = 0
        
        for epoch in range(self.start_epoch, self.args.num_epochs):
            
            self.lr = lr_[int(epoch/self.args.lr_decay)]
            for param_group in self.deblur_opt.param_groups:
                param_group['lr'] = self.lr
                                
            G_loss_avg = self.train_epoch(train_dpdd_loader,epoch)
            
            if epoch % self.args.save_model_freq == 0:
                state = {
                    'epoch': epoch + 1,
                    'global_step': self.global_step,
                    'G_state_dict': self.deblur_model.state_dict(),
                    'G_opt': self.deblur_opt.state_dict(),
                }
                
                ssim,psnr,mse,mae = test_state(state['G_state_dict'])
                if not os.path.exists('./checkpoint'):
                    os.mkdir('checkpoint')    
                
                if epoch % 10 == 0:
                    torch.save(state, './checkpoint/epoch_{:0>3}_G_{:.3f}_P_{:.3f}.pth'.format(epoch,G_loss_avg,psnr))
                    
                if psnr > best_psnr or ssim > best_ssim:
                    print('Saving checkpoint, psnr: {} ssim: {}'.format(psnr,ssim))
                    if psnr > best_psnr:
                        best_psnr = psnr
                        torch.save(state, './checkpoint/epoch_{:0>3}_G_{:.3f}_P_{:.3f}.pth'.format(epoch,G_loss_avg,psnr))
                    else:
                        best_ssim = ssim
                        torch.save(state, './checkpoint/epoch_{:0>3}_G_{:.3f}_S_{:.3f}.pth'.format(epoch,G_loss_avg,ssim))


    def train_epoch(self,train_dpdd_loader,epoch):
        self.deblur_model.train()                         
        G_loss_sum=0
        loss_fn = CharbonnierLoss()
        print("The num of training images:",len(train_dpdd_loader))
        for index, (im_gt,im_l,im_r) in enumerate(train_dpdd_loader):
               
            im_gt = im_gt.cuda()
            im_r = im_r.cuda()
            im_l = im_l.cuda()
            output = self.deblur_model(im_l,im_r)      

            char_loss = loss_fn(output, im_gt)  
            loss = char_loss 
            print('char_loss: {0}\step: {1}\tepoch: {2}\tlr: {3}'.format(loss.item(),index,epoch,self.lr))  
            
            self.deblur_opt.zero_grad()
            G_loss_sum += loss.item()
            loss = loss.cuda(non_blocking=True)
            torch.cuda.empty_cache()
            loss.backward()
            self.deblur_opt.step()
            
        self.global_step+=1
        return G_loss_sum/len(train_dpdd_loader)
