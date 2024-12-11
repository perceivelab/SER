# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from datasets import get_dataset

from models.utils.cl2branches import CLModel2Branches
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, add_saliency_args, ArgumentParser


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Experience Replay.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_saliency_args(parser)

    return parser


class ErACESER(CLModel2Branches):
    NAME = 'er_ace_ser'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform):
        super(ErACESER, self).__init__(backbone, loss, args, transform)
        
        self.seen_so_far = torch.tensor([]).long().to(self.device)
        self.num_classes = get_dataset(args).N_TASKS * get_dataset(args).N_CLASSES_PER_TASK



    def observe(self, inputs, labels, not_aug_inputs):
        
        if self.args.saliency_frozen:
            assert not self.saliency_net.training
        present = labels.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        assert isinstance(inputs, list)
        if not self.args.saliency_frozen:
            self.saliency_opt.zero_grad()

        imgs, sal_map = inputs
        #Saliency Prediction task
        sal_pred, sal_features = self.saliency_net(imgs)
        sal_features = [sal_f.detach() for sal_f in sal_features]

        sal_loss = self.sal_criterion(sal_pred, sal_map) * self.sal_coeff
        if not self.args.saliency_frozen:
            sal_loss.backward()
            self.saliency_opt.step()

        logits = self.forward_mnp(imgs, sal_features)
        mask = torch.zeros_like(logits)
        mask[:, present] = 1

        self.opt.zero_grad()
        if self.seen_so_far.max() < (self.num_classes - 1):
            mask[:, self.seen_so_far.max():] = 1

        if self.task > 0:
            logits = logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)

        loss = self.loss(logits, labels)
        loss_re = torch.tensor(0.)

        if self.task > 0:
            # sample from buffer
            saliency_status = self.saliency_net.training
            self.saliency_net.eval()
            buf_inputs, buf_labels = self.buffer.get_data(self.args.minibatch_size, transform=None)

            with torch.no_grad():
                _, sal_features = self.saliency_net(buf_inputs)

            buf_outputs = self.forward_mnp(buf_inputs, sal_features)

            loss_re = self.loss(buf_outputs, buf_labels)
            self.saliency_net.train(saliency_status)

        loss += loss_re

        loss.backward()
        self.opt.step()

        self.buffer.add_data(examples=imgs,
                             labels=labels)

        return [loss.item(), sal_loss.item()]
