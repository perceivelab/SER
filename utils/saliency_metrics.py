import torch
from torch.nn import functional as F
import sys

def kld_loss(pred, target):
    loss = F.kl_div(pred, target, reduction='none')
    loss = loss.sum(-1).sum(-1)
    return loss

def corr_coeff(pred, target):
    size = pred.size()
    new_size = (-1, size[-1] * size[-2])
    pred = pred.reshape(new_size)
    target = target.reshape(new_size)

    cc = []
    for x, y in zip(torch.unbind(pred, 0), torch.unbind(target, 0)):
        xm, ym = x - x.mean(), y - y.mean()
        r_num = torch.mean(xm * ym)
        r_den = torch.sqrt(
            torch.mean(torch.pow(xm, 2)) * torch.mean(torch.pow(ym, 2)))
        r = r_num / r_den
        cc.append(r.item())
    
    return torch.tensor(cc)

def similarity(s_map: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.min(s_map.squeeze(1), gt.squeeze(1)), dim=(1,2)).unsqueeze(1)


def _pointwise_loss(lambd, input, target, size_average=True, reduce=True):
    d = lambd(input, target)
    if not reduce:
        return d
    return torch.mean(d) if size_average else torch.sum(d)

class KLDLoss(torch.nn.Module):
    def __init__(self, dev='cpu'):
        super(KLDLoss, self).__init__()
        self.dev = dev

    def KLD(self, inp, trg):
        assert inp.size(0)==trg.size(0), "Size of distributions doesn't match"
        batch_size = inp.size(0)
        kld_tensor = torch.empty(batch_size)

        for k in range(batch_size):
            i = inp[k] / torch.sum(inp[k])
            t = trg[k] / torch.sum(trg[k])  
            eps = sys.float_info.epsilon
            kld_tensor[k] = torch.sum(t*torch.log(eps+torch.div(t,(i+eps))))
        return kld_tensor.to(self.dev)
    
    def forward(self, inp, trg):
        return _pointwise_loss(lambda a, b: self.KLD(a, b), inp, trg)


import numpy as np


def normalize_map(s_map):
    # normalize the salience map (as done in MIT code)
    if np.max(s_map) == 0:
        return s_map
    norm_s_map = (s_map - np.min(s_map))/((np.max(s_map)-np.min(s_map))*1.0)
    return norm_s_map


def similarity(sal_map,tgt):
    # here gt is not discretized nor normalized
    sal_map = sal_map.cpu().numpy().squeeze(1)
    tgt = tgt.cpu().numpy().squeeze(1)
    scores = []
    for k in range(sal_map.shape[0]):
        s_map = normalize_map(sal_map[k])
        gt = normalize_map(tgt[k])
        s_map = s_map/(np.sum(s_map)*1.0)
        gt = gt/(np.sum(gt)*1.0)
        sim = np.sum(np.minimum(gt, s_map))
        
        scores.append(sim)
    
    return torch.tensor(scores)


def kldiv(inp, tgt):
    kld_tensor = torch.empty(inp.shape[0])
    for k in range(inp.shape[0]):
        i = inp[k] / torch.sum(inp[k])
        t = tgt[k] / torch.sum(tgt[k])
        kld_tensor[k] = torch.sum(t * torch.log(sys.float_info.epsilon + torch.div(t, (i+ sys.float_info.epsilon))))
    return kld_tensor



def compute_saliency_metrics(sal_preds: torch.Tensor, sal_maps: torch.Tensor, metrics: list) -> list: 
    scores = []

    for this_metric in metrics:
        if this_metric == 'kld':
            kld = kldiv(sal_preds, sal_maps)
            scores.append(kld)
        if this_metric == 'cc':
            cc = corr_coeff(sal_preds, sal_maps)
            scores.append(cc)
        if this_metric == 'sim':
            sim = similarity(sal_preds, sal_maps)
            scores.append(sim)

    return scores
