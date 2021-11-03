#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 09:56:18 2019

@author: li
"""

import os
import argparse
from solver.solver import Solver

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

parser = argparse.ArgumentParser('Perceptual Reflection Removel')
parser.add_argument('--data_path',default="./dpdd_datasets",help="synthetic data")
parser.add_argument('--save_model_freq',default=1,type=int,help="frequency to save model")
parser.add_argument('--test_model_freq',default=1,type=int,help="frequency to test model")
parser.add_argument('--print_freq',type=int,default=10000,help='print frequency (default: 10)')
parser.add_argument('--resume_file',default='',help="resume file path")
parser.add_argument('--lr',default=2e-5,type=float,help="learning rate")
parser.add_argument('--load_workers',default=8,type=int,help="number of workers to load data")      
parser.add_argument('--lr_decay',default=60,type=int,help="learning rate")
parser.add_argument('--batch_size',default=4,type=int,help="batch size")
parser.add_argument('--start_epoch',type=int,default=0,help="start epoch of training")
parser.add_argument('--num_epochs',type=int,default=150,help="total epoch of training")
parser.add_argument('--is_training',default=True,help="training or testing")

def main():
	if not os.path.exists('./summary'):
		os.mkdir('summary')
	args = parser.parse_args()
	solver=Solver(args) 
	solver.train_model()

if __name__=='__main__':
	main()
