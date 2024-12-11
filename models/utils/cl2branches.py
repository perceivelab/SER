

import torch
from torch.nn import functional as F

from models.utils.continual_model import ContinualModel
from utils.buffer import Buffer
from datasets import get_dataset

from models.aux.unisal import UNISAL
from utils.mnp import *
from utils.saliency_metrics import KLDLoss
from copy import deepcopy



class CLModel2Branches(ContinualModel):
    
    def __init__(self, backbone, loss, args, transform) -> None:
        super(CLModel2Branches, self).__init__(backbone, loss, args, transform)
        # saliency_net and net have the same backbone architecture
        self.saliency_net = UNISAL(backbone=deepcopy(backbone), n_gaussians=self.args.sal_n_gaussians)
        self.saliency_opt = self.get_saliency_opt() if not self.args.saliency_frozen else None
        self.saliency_scheduler = self.get_saliency_scheduler() if not self.args.saliency_frozen else None
        if self.args.sal_ckpt != None or self.args.class_ckpt != None:
            self.load_cp()
        if hasattr(self.args, 'buffer_size') and self.args.buffer_size > 0:
            self.buffer = Buffer(self.args.buffer_size, self.device)
        self.sal_criterion = KLDLoss(self.device)
        self.sal_coeff = self.args.sal_coeff

        self.task = 0

        self.mnp = self.get_mnp(self.args.mnp)
        self.mnp_blocks = [bool(x) for x in args.mnp_blocks]

        ds = get_dataset(args)
        self.cpt = ds.N_CLASSES_PER_TASK
        self.n_tasks = ds.N_TASKS
        self.num_classes = self.n_tasks * self.cpt
  

    def load_cp(self):
        if self.args.sal_ckpt != None:
            print('Load saliency model ckpt...')
            print(self.args.sal_ckpt)
            ckpt = torch.load(self.args.sal_ckpt, map_location='cpu')
            if 'model_state_dict' in ckpt.keys():
                print('Pretraining from saliency prediction')
                #ckpt from saliency prediction
                ckpt = ckpt['model_state_dict']
                if len(ckpt.keys()) > len(self.saliency_net.state_dict().keys()):
                    rem_layers = set(ckpt.keys()) - set(self.saliency_net.state_dict().keys())
                    for l in rem_layers:
                        del ckpt[l]
                assert len(ckpt.keys()) == len(self.saliency_net.state_dict().keys())
                self.saliency_net.load_state_dict(ckpt)
            elif 'state_dict' in ckpt.keys():
                #ckpt from classification
                print('Pretraining from classifaction')
                ckpt = ckpt['state_dict']
                for k in ['linear.weight', 'linear.bias', 'classifier.weight', 'classifier.bias']:
                    if k in ckpt.keys():
                        del ckpt[k]
                miss_keys, unex_keys = self.saliency_net.cnn.load_state_dict(ckpt)
                assert len(unex_keys) == len(miss_keys) == 0
            elif ckpt.keys() == self.saliency_net.state_dict().keys():
                self.saliency_net.load_state_dict(ckpt)
            else:
                raise NotImplementedError('load_cp saliency_net error')
            print('Done!')

        if self.args.backbone_pretrained: 
            print('Initialize backbone as Saliency Encoder...')
            ckpt = {k.replace('cnn.', ''):v for k,v in ckpt.items() if 'cnn' in k and 'post_cnn' not in k}
            miss_keys, unex_keys = self.net.load_state_dict(ckpt, strict=False)
            assert set(miss_keys) == set(['classifier.weight', 'classifier.bias'])
            assert len(unex_keys) == 0
            print('Done!')
        
        if self.args.class_ckpt != None:
            print('Load backbone ckpt from classification..')
            print(self.args.class_ckpt)
            ckpt = torch.load(self.args.class_ckpt, map_location='cpu')
            if 'state_dict' in ckpt.keys():
                ckpt = ckpt['state_dict']
                for l in ['linear.weight', 'linear.bias', 'classifier.weight', 'classifier.bias']:
                    if l in ckpt.keys(): 
                        del ckpt[l]
                miss_keys, unex_keys = self.net.load_state_dict(ckpt, strict=False)
                assert set(miss_keys) == set(['classifier.weight', 'classifier.bias'])
                assert len(unex_keys) == 0
            elif ckpt.keys() == self.net.state_dict().keys():
                self.net.load_state_dict(ckpt, strict=False)
            else:
                raise NotImplementedError('load_cp net error')
            print('Done!')


    def get_saliency_scheduler(self):
        if self.args.sal_scheduler == 'None':
            return None
        elif self.args.sal_scheduler == 'ExponentialLR':
            return torch.optim.lr_scheduler.ExponentialLR(
                self.saliency_opt, gamma=self.args.sal_lr_gamma, last_epoch=-1
            )
        raise ValueError('Unknown {} saliency model scheduler.'.format(self.args.sal_scheduler))

    def get_saliency_opt(self):
        if self.args.sal_opt == 'SGD':
            return torch.optim.SGD(self.get_sal_model_parameter_groups(), lr=self.args.sal_lr,
                momentum=self.args.sal_momentum, weight_decay=self.args.sal_weight_decay)
        elif self.args.sal_opt == 'Adam':
            return torch.optim.Adam(self.get_sal_model_parameter_groups(), lr=self.args.sal_lr,
                weight_decay=self.args.sal_weight_decay)
        elif self.args.sal_opt == 'RMSprop':
            return torch.optim.RMSprop(self.get_sal_model_parameter_groups(), lr=self.args.sal_lr,
                weight_decay=self.args.sal_weight_decay, momentum=self.args.sal_momentum)
        raise ValueError('Unknown {} saliency model optimizer.'.format(self.args.sal_opt))
    
    def get_sal_model_parameter_groups(self):
        """
        Get parameter groups.
        Output CNN parameters separately with reduced LR and weight decay.
        """
        def parameters_except_cnn():
            parameters = []
            adaptation = []
            for name, module in self.saliency_net.named_children():
                if name == 'cnn':
                    continue
                elif 'adaptation' in name:
                    adaptation += list(module.parameters())
                else:
                    parameters += list(module.parameters())
            return parameters, adaptation

        parameters, adaptation = parameters_except_cnn()

        for name, this_parameter in self.saliency_net.named_parameters():
            if 'gaussian' in name:
                parameters.append(this_parameter)

        return [
            {'params': parameters + adaptation},
            {'params': self.saliency_net.cnn.parameters(),
             'lr': self.args.sal_lr * self.args.sal_cnn_lr_factor,
             'weight_decay': self.args.sal_cnn_weight_decay,
             },
        ]

    def get_mnp(self, mnp_type: str) -> MNP:
        if mnp_type == 'aggregate':
            return MNPAggregate
        elif mnp_type == 'multiply':
            return MNPMultiply
        elif mnp_type == 'dwseparable':
            return MNPSeparable

        raise NotImplementedError("Unknonw mnp type: {}".format(mnp_type))
            
    
    def forward(self, x) -> torch.Tensor:
        # x: (img, sal_map)
        imgs, _ = x
        sal_pred, sal_features = self.saliency_net(imgs)
        outputs = self.forward_mnp(imgs, sal_features)
        return sal_pred, outputs
    
    def to(self, device): 
        super().to(device)
        self.net = self.net.to(device)
        self.saliency_net.to(device)

    def begin_task(self, dataset):
        
        saliency_status = self.saliency_net.training
        if self.task == 0:
            self.net.set_return_prerelu(self.NAME != 'dualnet_2branches')
            self.saliency_net.cnn.set_return_prerelu(self.NAME != 'dualnet_2branches')

            self.net.eval()
            self.saliency_net.eval()
            
            inputs, _, _ = iter(dataset.train_loader).next()
            assert isinstance(inputs, list) 
            imgs, _ = inputs 
            imgs = imgs.to(self.device)
            with torch.no_grad():
                _, feats = self.net(imgs, 'full')
                _, sal_feats = self.saliency_net(imgs, 'full')
            
            assert len(feats) == len(sal_feats) == len(self.mnp_blocks)
            assert all([feats[i].shape == sal_feats[i].shape for i in range(len(sal_feats))])

            for i, (x, sal_x, mnp_blck) in enumerate(zip(feats, sal_feats, self.mnp_blocks)):
                if mnp_blck:
                    setattr(self.net, f"adapter_{i}", self.mnp(
                        s_feature=x,
                        t_feature=sal_x
                    ).to(self.device))
                    self.opt.add_param_group({
                        'params':self.net.__getattr__(f"adapter_{i}").parameters()})
            
            self.net.train()
        
        if self.args.saliency_frozen:
            for param in self.saliency_net.parameters():
                param.requires_grad = False
        self.saliency_net.train(saliency_status)
    
    def end_task(self, dataset):
        self.task +=1
    
    def forward_mnp(self, imgs: torch.Tensor, t_feature: list, returnt: str='out'):
        # block_0
        out_0 = self.net.bn1(self.net.conv1(imgs))
        if hasattr(self.net, 'adapter_0'):
            out_0 = self.net.adapter_0(out_0, t_feature[0])
        out_0 = F.relu(out_0)
        #if hasattr(self.net, 'maxpool'):
        if self.net.enable_maxpool:
            out_0 = self.net.maxpool(out_0)
        
        f_block = out_0
        for i in range(1, 5):
            f_block = self.net.__getattr__(f"layer{i}")(f_block)
            if hasattr(self.net, f"adapter_{i}"):
                f_block = self.net.__getattr__(f"adapter_{i}")( 
                    self.net.__getattr__(f"layer{i}")[-1].prerelu,
                    t_feature[i]
                )

        feature = F.avg_pool2d(f_block, f_block.shape[2])
        feature = feature.view(feature.size(0), -1)
        if returnt == 'features':
            return feature
        
        out = self.net.classifier(feature)    
        if returnt == 'out':
            return out

        raise NotImplementedError(f"Unknown return type: {returnt}")