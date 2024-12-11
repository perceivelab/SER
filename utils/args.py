# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from datasets import NAMES as DATASET_NAMES
from models import get_all_models
from utils.conf import base_path_dataset as base_path


def add_experiment_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the models.
    :param parser: the parser instance
    """
    parser.add_argument('--dataset', type=str, required=True,
                        choices=DATASET_NAMES,
                        help='Which dataset to perform experiments on.')
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())

    parser.add_argument('--lr', type=float, required=True,
                        help='Learning rate.')

    parser.add_argument('--optim_wd', type=float, default=0.,
                        help='optimizer weight decay.')
    parser.add_argument('--optim_mom', type=float, default=0.,
                        help='optimizer momentum.')
    parser.add_argument('--optim_nesterov', type=int, default=0,
                        help='optimizer nesterov momentum.')

    parser.add_argument('--n_epochs', type=int,
                        help='Batch size.')
    parser.add_argument('--batch_size', type=int,
                        help='Batch size.')

    parser.add_argument('--distributed', type=str, default='no', choices=['no', 'dp', 'ddp'])

    parser.add_argument('--dataset_subset', type=float, default=1.)

    


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument('--seed', type=int, default=None,
                        help='The random seed.')
    parser.add_argument('--notes', type=str, default=None,
                        help='Notes for this run.')

    parser.add_argument('--non_verbose', default=0, choices=[0, 1], type=int, help='Make progress bars non verbose')
    parser.add_argument('--disable_log', default=0, choices=[0, 1], type=int, help='Enable csv logging')

    parser.add_argument('--validation', default=0, choices=[0, 1], type=int,
                        help='Test on the validation set')
    parser.add_argument('--ignore_other_metrics', default=0, choices=[0, 1], type=int,
                        help='disable additional metrics')
    parser.add_argument('--debug_mode', type=int, default=0, help='Run only a few forward steps per epoch')
    parser.add_argument('--nowand', default=0, choices=[0, 1], type=int, help='Inhibit wandb logging')
    parser.add_argument('--wandb_entity', type=str, help='Wandb entity')
    parser.add_argument('--wandb_project', type=str, help='Wandb project name')
    parser.add_argument('--device', type=str, default='cuda:0')

    parser.add_argument('--savecheck', action='store_true',
                        help='Whether to save checkpoint')
    parser.add_argument('--sal_ckpt', type=str, default=None)
    parser.add_argument('--class_ckpt', type=str, default=None)
    parser.add_argument('--dataset_path', type=str, default=base_path())
    parser.add_argument('--run_idx', type=int, default=0, help='fake run index for wandb')


def add_rehearsal_args(parser: ArgumentParser) -> None:
    """
    Adds the arguments used by all the rehearsal-based methods
    :param parser: the parser instance
    """
    parser.add_argument('--buffer_size', type=int, required=True,
                        help='The size of the memory buffer.')
    parser.add_argument('--minibatch_size', type=int,
                        help='The batch size of the memory buffer.')


def add_saliency_args(parser: ArgumentParser) -> None:
        #Saliency net paramaters;
        parser.add_argument('--sal_lr', type=float, default=0.04)
        parser.add_argument('--sal_cnn_lr_factor', type=float, default=0.1)
        parser.add_argument('--sal_cnn_weight_decay', type=float, default=1e-5)
        parser.add_argument('--sal_momentum', type=float, default=0.9)
        parser.add_argument('--sal_weight_decay', type=float, default=1e-4)
        parser.add_argument('--sal_opt', type=str, default='SGD', choices=['SGD', 'Adam', 'RMSprop'])
        parser.add_argument('--sal_scheduler', type=str, default='None', choices = ['None','ExponentialLR'])
        parser.add_argument('--sal_lr_gamma', type=float, default=0.999)
        parser.add_argument('--sal_n_gaussians', type=int, default=0)
        parser.add_argument('--sal_kld_weight', type=float, default=1.0,
                            help='KL_div loss penalty')
        parser.add_argument('--sal_cc_weight', type=float, default=-0.1,
                            help='CC loss penalty')
        parser.add_argument('--sal_coeff', type=float, default=1.,
                            help= 'saliency loss coefficient')
        parser.add_argument('--backbone_pretrained', action='store_true')
        parser.add_argument('--mnp', choices=['aggregate', 'multiply', 'dwseparable'], required=True)
        parser.add_argument('--mnp_blocks', type=int, nargs=5, default=[1,1,1,1,1])
        parser.add_argument('--saliency_frozen', action='store_true')
