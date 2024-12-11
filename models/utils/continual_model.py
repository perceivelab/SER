# Copyright 2020-present, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Davide Abati, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
from argparse import Namespace
from contextlib import suppress
from typing import List

import torch
import torch.nn as nn
from torch.optim import SGD

from utils.conf import get_device
from utils.magic import persistent_locals
from datasets import get_dataset
from torchvision import transforms
with suppress(ImportError):
    import wandb


class ContinualModel(nn.Module):
    """
    Continual learning model.
    """
    NAME: str
    COMPATIBILITY: List[str]

    def __init__(self, backbone: nn.Module, loss: nn.Module,
                 args: Namespace, transform: nn.Module) -> None:
        super(ContinualModel, self).__init__()


        ds = get_dataset(args)
        self.cpt = ds.N_CLASSES_PER_TASK
        self.n_task = ds.N_TASKS
        self.num_classes = self.n_task * self.cpt
    
        self.train_transform = ds.TRANSFORM
        self.test_transform = ds.TEST_TRANSFORM if hasattr(ds, 'TEST_TRANSFORM') else transforms.Compose(
            [transforms.ToTensor(), ds.get_normalization_transform()])

        self.net = backbone
        self.loss = loss
        self.args = args
        self.transform = transform
        self.opt = SGD(self.net.parameters(), lr=self.args.lr)
        #self.device = get_device()
        self.device = self.args.device

        if not self.NAME or not self.COMPATIBILITY:
            raise NotImplementedError('Please specify the name and the compatibility of the model.')
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes a forward pass.
        :param x: batch of inputs
        :param task_label: some models require the task label
        :return: the result of the computation
        """
        return self.net(x)

    def meta_observe(self, *args, **kwargs):
        if 'wandb' not in sys.modules and not self.args.nowand:
            pl = persistent_locals(self.observe)
            ret = pl(*args, **kwargs)
            self.autolog_wandb(pl.locals)
        else:
            ret = self.observe(*args, **kwargs)
        return ret

    def observe(self, inputs: torch.Tensor, labels: torch.Tensor,
                not_aug_inputs: torch.Tensor) -> float:
        """
        Compute a training step over a given batch of examples.
        :param inputs: batch of examples
        :param labels: ground-truth labels
        :param kwargs: some methods could require additional parameters
        :return: the value of the loss function
        """
        raise NotImplementedError

    def autolog_wandb(self, locals):
        """
        All variables starting with "_wandb_" or "loss" in the observe function
        are automatically logged to wandb upon return if wandb is installed.
        """
        if not self.args.nowand and not self.args.debug_mode:
            wandb.log({k: (v.item() if isinstance(v, torch.Tensor) and v.dim() == 0 else v)
                      for k, v in locals.items() if k.startswith('_wandb_') or k.startswith('loss')})

    def load_cp(self):    
        ckpt = torch.load(self.args.sal_ckpt, map_location=torch.device('cpu'))
        if 'state_dict' in ckpt.keys():
            print('Load ckpt from classification')
            ckpt= ckpt['state_dict']
        elif 'model_state_dict' in ckpt.keys():
            print('Load ckpt from saliency prediction')
            ckpt= ckpt['model_state_dict']
            ckpt = {k.replace('cnn.', ''):v for k,v in ckpt.items() if 'cnn' in k and 'post_cnn' not in k}
    
        #skip these steps if you want to load classification layer
        if 'linear.weight' in ckpt.keys():
            del ckpt['linear.weight'] 
        if 'linear.bias' in ckpt.keys():
            del ckpt['linear.bias']
        if 'classifier.weight' in ckpt.keys():
            del ckpt['classifier.weight']
        if 'classifier.bias' in ckpt.keys():
            del ckpt['classifier.bias']
        
        miss_keys, unex_keys = self.net.load_state_dict(ckpt, strict=False)
        assert set(miss_keys) == set(['classifier.weight', 'classifier.bias'])
        assert len(unex_keys) == 0 
        print('Done!')

    @torch.no_grad()
    def apply_transform(self, inputs, transform, device=None, add_pil_transforms=True):
        tr = transforms.Compose([transforms.ToPILImage()] + transform.transforms) if add_pil_transforms else transform
        device = self.device if device is None else device
        if len(inputs.shape) == 3:
            return tr(inputs)
        return torch.stack([tr(inp) for inp in inputs.cpu()], dim=0).to(device)

    @torch.no_grad()
    def aug_batch(self, not_aug_inputs, device=None):
        """
        Full train transform 
        """
        return self.apply_transform(not_aug_inputs, self.train_transform, device=device)

    @torch.no_grad()
    def test_data_aug(self, inputs, device=None):
        """
        Test transform
        """
        return self.apply_transform(inputs, self.test_transform, device=device)
    

    def reset_classifier(self):
        self.net.classifier = torch.nn.Linear(
                self.net.classifier.in_features, self.net.num_classes).to(self.device)
        self.reset_opt()

    def reset_opt(self):
        self.opt = get_dataset(self.args).get_optimizer(self.net.parameters(), self.args)
        return self.opt