#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 09:13:04 2020

@author: li
"""

import argparse

parser = argparse.ArgumentParser('test')
parser.add_argument('--checkpoint_folder',default='',help="path to checkpoint folder, use when find best ckpt")
parser.add_argument('--checkpoint_path',
                    default="./checkpoint/final.pth",help="path to checkpoint")
parser.add_argument('--save_result_path',default=True,help="if save result")
parser.add_argument('--num_workers',default=4,help="num_workers")
args = parser.parse_args()
