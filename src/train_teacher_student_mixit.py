#!/usr/bin/env python

# Created on 2021/01
# Author: Jisi Zhang

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import AudioDataLoader, AudioDataset
from data_dynamic import DynamicAudioDataset
#from solver import Solver
from solver_teacher_student_mixit import Solver

from conv_tasnet_mixture_consistency import ConvTasNet


parser = argparse.ArgumentParser(
    "Fully-Convolutional Time-domain Audio Separation Network (Conv-TasNet) "
    "with Teacher-student Mixture Invariant Training")
# General config
# Task related
parser.add_argument('--train_dir', type=str, default=None,
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--valid_dir', type=str, default=None,
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--sample_rate', default=8000, type=int,
                    help='Sample rate')
parser.add_argument('--segment', default=4, type=float,
                    help='Segment length (seconds)')
parser.add_argument('--cv_maxlen', default=8, type=float,
                    help='max audio length (seconds) in cv, to avoid OOM issue.')
# Network architecture
parser.add_argument('--N', default=256, type=int,
                    help='Number of filters in autoencoder')
parser.add_argument('--S', default=30, type=int,
                    help='Number of filters in spatial encoder')
parser.add_argument('--L', default=20, type=int,
                    help='Length of the filters in samples (40=5ms at 8kHZ)')
parser.add_argument('--B', default=256, type=int,
                    help='Number of channels in bottleneck 1 × 1-conv block')
parser.add_argument('--H', default=512, type=int,
                    help='Number of channels in convolutional blocks')
parser.add_argument('--P', default=3, type=int,
                    help='Kernel size in convolutional blocks')
parser.add_argument('--X', default=8, type=int,
                    help='Number of convolutional blocks in each repeat')
parser.add_argument('--R', default=4, type=int,
                    help='Number of repeats')
parser.add_argument('--C', default=2, type=int,
                    help='Number of speakers')
parser.add_argument('--norm_type', default='gLN', type=str,
                    choices=['gLN', 'cLN', 'BN'], help='Layer norm type')
parser.add_argument('--causal', type=int, default=0,
                    help='Causal (1) or noncausal(0) training')
parser.add_argument('--mask_nonlinear', default='relu', type=str,
                    choices=['relu', 'softmax', 'tanh'], help='non-linear to generate mask')
# Training config
parser.add_argument('--use_cuda', type=int, default=1,
                    help='Whether use GPU')
parser.add_argument('--epochs', default=30, type=int,
                    help='Number of maximum epochs')
parser.add_argument('--half_lr', dest='half_lr', default=0, type=int,
                    help='Halving learning rate when get small improvement')
parser.add_argument('--early_stop', dest='early_stop', default=0, type=int,
                    help='Early stop training when no improvement for 10 epochs')
parser.add_argument('--max_norm', default=5, type=float,
                    help='Gradient norm threshold to clip')
# minibatch
parser.add_argument('--shuffle', default=0, type=int,
                    help='reshuffle the data at every epoch')
parser.add_argument('--batch_size', default=128, type=int,
                    help='Batch size')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers to generate minibatch')
# optimizer
parser.add_argument('--optimizer', default='adam', type=str,
                    choices=['sgd', 'adam'],
                    help='Optimizer (support sgd and adam now)')
parser.add_argument('--lr', default=1e-3, type=float,
                    help='Init learning rate')
parser.add_argument('--momentum', default=0.0, type=float,
                    help='Momentum for optimizer')
parser.add_argument('--l2', default=0.0, type=float,
                    help='weight decay (L2 penalty)')
# save and load model
parser.add_argument('--save_folder', default='exp/temp',
                    help='Location to save epoch models')
parser.add_argument('--checkpoint', dest='checkpoint', default=0, type=int,
                    help='Enables checkpoint saving of model')
parser.add_argument('--continue_from', default='',
                    help='Continue from checkpoint model')
parser.add_argument('--model_path', default='final.pth.tar',
                    help='Location to save best validation model')
# logging
parser.add_argument('--print_freq', default=10, type=int,
                    help='Frequency of printing training infomation')
parser.add_argument('--visdom', dest='visdom', type=int, default=0,
                    help='Turn on visdom graphing')
parser.add_argument('--visdom_epoch', dest='visdom_epoch', type=int, default=0,
                    help='Turn on visdom graphing each epoch')
parser.add_argument('--visdom_id', default='TasNet training',
                    help='Identifier for visdom run')
parser.add_argument('--cfg', default='SincNet_WSJ_8k.cfg',
                    help='Speaker embedding model config')


def main(args):
    # Construct Solver
    # data
#    tr_dataset = DynamicAudioDataset(args.train_dir,
#                              sample_rate=args.sample_rate, segment=args.segment)
#    tr_loader = DataLoader(tr_dataset, batch_size=args.batch_size,
#                           shuffle=args.shuffle,
#                           num_workers=args.num_workers)
    tr_dataset = AudioDataset(args.train_dir, args.batch_size,
                              sample_rate=args.sample_rate, segment=args.segment)
    tr_loader = AudioDataLoader(tr_dataset, batch_size=1,
                                shuffle=args.shuffle,
                                num_workers=args.num_workers)
    cv_dataset = AudioDataset(args.valid_dir, batch_size=1,  # 1 -> use less GPU memory to do cv
                              sample_rate=args.sample_rate,
                              segment=-1, cv_maxlen=args.cv_maxlen)  # -1 -> use full audio
    cv_loader = AudioDataLoader(cv_dataset, batch_size=1,
                                num_workers=args.num_workers)
    data = {'tr_loader': tr_loader, 'cv_loader': cv_loader}
    
    
    
    # model
    ''' Multi-channel model initialization'''
#    model = ConvTasNet(args.N, args.S, args.L, args.B, args.H, args.P, args.X, args.R,
#                       args.C, norm_type=args.norm_type, causal=args.causal,
#                       mask_nonlinear=args.mask_nonlinear)
    ''' Single-channel model initialization'''
#    model = SeparationNet(args.N, args.L, args.B, args.H, args.P, args.X, args.R,
#                       args.C, D, norm_type=args.norm_type, causal=args.causal,
#                       mask_nonlinear=args.mask_nonlinear)
    teacher_model = ConvTasNet(args.N, args.L, args.B, args.H, args.P, args.X, args.R,
                       4, norm_type=args.norm_type, causal=args.causal,
                       mask_nonlinear=args.mask_nonlinear)
    
    teacher_model.load_state_dict(torch.load(args.continue_from)['state_dict'])
    
    student_model = ConvTasNet(args.N, args.L, args.B, args.H, args.P, args.X, args.R,
                       4, norm_type=args.norm_type, causal=args.causal,
                       mask_nonlinear=args.mask_nonlinear)
    
    print(student_model)
    if args.use_cuda:
        teacher_model = torch.nn.DataParallel(teacher_model)
        teacher_model.cuda()
        student_model = torch.nn.DataParallel(student_model)
        student_model.cuda()
    # optimizer
    if args.optimizer == 'sgd':
        teacher_optimizer = torch.optim.SGD(teacher_model.parameters(),
                                     lr=args.lr,
                                     momentum=args.momentum,
                                     weight_decay=args.l2)
        student_optimizer = torch.optim.SGD(student_model.parameters(),
                                     lr=args.lr,
                                     momentum=args.momentum,
                                     weight_decay=args.l2)
    elif args.optimizer == 'adam':
        teacher_optimizer = torch.optim.Adam(teacher_model.parameters(),
                                      lr=float(args.lr),
                                      weight_decay=args.l2)
        student_optimizer = torch.optim.Adam(student_model.parameters(),
                                      lr=float(args.lr),
                                      weight_decay=args.l2)
    else:
        print("Not support optimizer")
        return
    

    # solver
    solver = Solver(data, teacher_model, student_model, teacher_optimizer, student_optimizer, args)

    solver.train()


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)

