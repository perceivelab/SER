

import torch
from torch.nn import functional as F

from models.utils.cl2branches import CLModel2Branches
from utils.args import add_management_args, add_experiment_args, add_rehearsal_args, add_saliency_args, ArgumentParser

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual learning via'
                                        ' Dark Experience Replay++.')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    add_saliency_args(parser)
    
    parser.add_argument('--alpha', type=float, required=True,
                        help='Penalty weight.')
    parser.add_argument('--beta', type=float, required=True,
                        help='Penalty weight.')
    
    return parser

class DerppSER(CLModel2Branches):
    
    NAME = 'derpp_ser'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, backbone, loss, args, transform) -> None:
        super(DerppSER, self).__init__(backbone, loss, args, transform)


    def observe(self, inputs, labels, not_aug_inputs):

        if self.args.saliency_frozen:
            assert not self.saliency_net.training
        self.opt.zero_grad()

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

        outputs = self.forward_mnp(imgs, sal_features)
        loss = self.loss(outputs, labels)

        if hasattr(self, 'buffer') and not self.buffer.is_empty():
            saliency_status = self.saliency_net.training
            self.saliency_net.eval()
            buf_inputs, _, buf_logits = self.buffer.get_data(self.args.minibatch_size, transform=None)

            with torch.no_grad():
                _, sal_features = self.saliency_net(buf_inputs)

            buf_outputs = self.forward_mnp(buf_inputs, sal_features)
            
            buf_mse_loss = F.mse_loss(buf_outputs, buf_logits)
            loss+= self.args.alpha * buf_mse_loss

            buf_inputs, buf_labels, _ = self.buffer.get_data(self.args.minibatch_size, transform=None)
            with torch.no_grad():
                _, sal_features = self.saliency_net(buf_inputs)
            
            buf_outputs = self.forward_mnp(buf_inputs, sal_features)

            buffer_ce_loss = self.loss(buf_outputs, buf_labels)
            loss+= self.args.beta * buffer_ce_loss
            
            self.saliency_net.train(saliency_status)

        
        loss.backward()
        self.opt.step()

        if hasattr(self, 'buffer'):
            self.buffer.add_data(examples=imgs,
                             labels = labels,
                             logits = outputs.data)
        return [loss.item(), sal_loss.item()]
