# -*- coding: utf-8 -*-
"""
Created on Sat Feb  6 17:30:51 2021

Teacher-Sudent with Mixture Invariant Training pretraining

@author: Jisi
"""

import os
import time

import torch
import numpy as np

from pit_criterion import cal_loss_pit
from mixit_criterion import cal_loss_mixit


class Solver(object):
    
    def __init__(self, data, teacher_model, student_model, 
                 teacher_optimizer, student_optimizer, args):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.teacher_optimizer = teacher_optimizer
        self.student_optimizer = student_optimizer

        # Training config
        self.use_cuda = args.use_cuda
        self.epochs = args.epochs
        self.half_lr = args.half_lr
        self.early_stop = args.early_stop
        self.max_norm = args.max_norm
        # save and load model
        self.save_folder = args.save_folder
        self.checkpoint = args.checkpoint
        self.continue_from = args.continue_from
        self.model_path = args.model_path
        # logging
        self.print_freq = args.print_freq
        # visualizing loss using visdom
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)
        self.visdom = args.visdom
        self.visdom_epoch = args.visdom_epoch
        self.visdom_id = args.visdom_id
        if self.visdom:
            from visdom import Visdom
            self.vis = Visdom(env=self.visdom_id)
            self.vis_opts = dict(title=self.visdom_id,
                                 ylabel='Loss', xlabel='Epoch',
                                 legend=['train loss', 'cv loss'])
            self.vis_window = None
            self.vis_epochs = torch.arange(1, self.epochs + 1)

        self._reset()

    def _reset(self):
        # Reset
        if self.continue_from:
            print('Loading checkpoint model %s' % self.continue_from)
            package = torch.load(self.continue_from)
            self.teacher_model.module.load_state_dict(package['state_dict'])
            self.teacher_optimizer.load_state_dict(package['optim_dict'])
            #self.start_epoch = int(package.get('epoch', 1))
            #self.tr_loss[:self.start_epoch] = package['tr_loss'][:self.start_epoch]
            #self.cv_loss[:self.start_epoch] = package['cv_loss'][:self.start_epoch]
        else:
            self.start_epoch = 0
        self.start_epoch = 0
        # Create save folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.prev_val_loss = float("inf")
        self.best_val_loss = float("inf")
        self.halving = False
        self.val_no_impv = 0

    def train(self):
        # Train model multi-epoches
        for epoch in range(self.start_epoch, self.epochs):
            # Train one epoch
            print("Training...")
            self.teacher_model.eval()
            self.student_model.train()  # Turn on BatchNorm & Dropout
            start = time.time()
            tr_avg_loss = self._run_one_epoch(epoch)
            print('-' * 85)
            print('Train Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Train Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, tr_avg_loss))
            print('-' * 85)

            # Save model each epoch
            if self.checkpoint:
                file_path = os.path.join(
                    self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))
                torch.save(self.student_model.module.serialize(self.student_model.module,
                                                       self.student_optimizer, epoch + 1,
                                                       tr_loss=self.tr_loss,
                                                       cv_loss=self.cv_loss),
                           file_path)
                print('Saving checkpoint model to %s' % file_path)

            # Cross validation
            print('Cross validation...')
            self.student_model.eval()  # Turn off Batchnorm & Dropout
            val_loss = self._run_one_epoch(epoch, cross_valid=True)
            print('-' * 85)
            print('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                  'Valid Loss {2:.3f}'.format(
                      epoch + 1, time.time() - start, val_loss))
            print('-' * 85)

            # Adjust learning rate (halving)
            if self.half_lr:
                if val_loss >= self.prev_val_loss:
                    self.val_no_impv += 1
                    if self.val_no_impv >= 3:
                        self.halving = True
                    if self.val_no_impv >= 10 and self.early_stop:
                        print("No imporvement for 10 epochs, early stopping.")
                        break
                else:
                    self.val_no_impv = 0
            if self.halving:
                optim_state = self.student_optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] = \
                    optim_state['param_groups'][0]['lr'] / 2.0
                self.student_optimizer.load_state_dict(optim_state)
                print('Learning rate adjusted to: {lr:.6f}'.format(
                    lr=optim_state['param_groups'][0]['lr']))
                self.halving = False
            self.prev_val_loss = val_loss

            # Save the best model
            self.tr_loss[epoch] = tr_avg_loss
            self.cv_loss[epoch] = val_loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                file_path = os.path.join(self.save_folder, self.model_path)
                torch.save(self.student_model.module.serialize(self.student_model.module,
                                                       self.student_optimizer, epoch + 1,
                                                       tr_loss=self.tr_loss,
                                                       cv_loss=self.cv_loss),
                           file_path)
                print("Find better validated model, saving to %s" % file_path)

            # visualizing loss using visdom
            if self.visdom:
                x_axis = self.vis_epochs[0:epoch + 1]
                y_axis = torch.stack(
                    (self.tr_loss[0:epoch + 1], self.cv_loss[0:epoch + 1]), dim=1)
                if self.vis_window is None:
                    self.vis_window = self.vis.line(
                        X=x_axis,
                        Y=y_axis,
                        opts=self.vis_opts,
                    )
                else:
                    self.vis.line(
                        X=x_axis.unsqueeze(0).expand(y_axis.size(
                            1), x_axis.size(0)).transpose(0, 1),  # Visdom fix
                        Y=y_axis,
                        win=self.vis_window,
                        update='replace',
                    )

    def _run_one_epoch(self, epoch, cross_valid=False):
        start = time.time()
        total_loss = 0

        data_loader = self.tr_loader if not cross_valid else self.cv_loader

        # visualizing loss using visdom
        if self.visdom_epoch and not cross_valid:
            vis_opts_epoch = dict(title=self.visdom_id + " epoch " + str(epoch),
                                  ylabel='Loss', xlabel='Epoch')
            vis_window_epoch = None
            vis_iters = torch.arange(1, len(data_loader) + 1)
            vis_iters_loss = torch.Tensor(len(data_loader))

        for i, (data) in enumerate(data_loader):
            padded_mixture, mixture_lengths, padded_source, filenames = data
            if self.use_cuda:
                padded_mixture = padded_mixture.cuda()
                mixture_lengths = mixture_lengths.cuda()
                padded_source = padded_source.cuda()
            estimate_teacher_source = self.teacher_model(padded_mixture) # [B, M, T]
            
            """
            Teacher-Student with Mixture Invariant Training Pre-training @ Jisi 2021/02/06
            """
            B, M, T = estimate_teacher_source.size()
            _, indxs = torch.sort(torch.sqrt(torch.mean(estimate_teacher_source ** 2, dim=-1)), descending=True)
            indxs = indxs[:, :2]  # we know a-priori that in eval there are 2 sources,
            # we thus discard the estimates with lower energy.
            estimate_teacher_source = estimate_teacher_source.gather(1, indxs.unsqueeze(-1).repeat(1, 1, estimate_teacher_source.shape[-1]))
            
            estimate_student_source = self.student_model(padded_mixture) # [B, M, T]
            if not cross_valid:
                loss, estimate_source, reorder_estimate_source = cal_loss_pit(estimate_teacher_source,
                                                                              estimate_student_source,
                                                                              mixture_lengths,
                                                                              cross_valid)
            else:
                loss, estimate_source, reorder_estimate_source = cal_loss_pit(padded_source, 
                                                                              estimate_student_source, 
                                                                              mixture_lengths,
                                                                              cross_valid)
            
            
            
            if not cross_valid:
                self.student_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(),
                                               self.max_norm)
                self.student_optimizer.step()

            total_loss += loss.item()

            if i % self.print_freq == 0:
                print('Epoch {0} | Iter {1} | Average Loss {2:.3f} | '
                      'Current Loss {3:.6f} | {4:.1f} ms/batch'.format(
                          epoch + 1, i + 1, total_loss / (i + 1),
                          loss.item(), 1000 * (time.time() - start) / (i + 1)),
                      flush=True)

            # visualizing loss using visdom
            if self.visdom_epoch and not cross_valid:
                vis_iters_loss[i] = loss.item()
                if i % self.print_freq == 0:
                    x_axis = vis_iters[:i+1]
                    y_axis = vis_iters_loss[:i+1]
                    if vis_window_epoch is None:
                        vis_window_epoch = self.vis.line(X=x_axis, Y=y_axis,
                                                         opts=vis_opts_epoch)
                    else:
                        self.vis.line(X=x_axis, Y=y_axis, win=vis_window_epoch,
                                      update='replace')

        return total_loss / (i + 1)