import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from datasets import get_dataset

def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Continual Prototype Evolution')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    parser.add_argument('--hidden_dim', type=int, default=256,
                        help='Dimensionality of prototypes')
    parser.add_argument('--num_batches', type=int, default=5,
                        help='Number of inner cycles for online CL')
    parser.add_argument('--loss_T', type=float, default=0.05,
                        help='loss temperature')
    parser.add_argument('--p_momentum', type=float, default=0.9,
                        help='momentum for prototype updates')
    parser.add_argument('--reset_backbone', type=int, required=False, choices=[0, 1], default=0,
                        help='Reset stream backbone after pretrain loading?')
    return parser

class PPPloss(nn.Module):
    def __init__(self, T=1):
        """
        :param margin: margin on distance between pos vs neg samples (see TripletMarginLoss)
        :param dist: distance function 2 vectors (e.g. L2-norm, CosineSimilarity,...)
        """
        super().__init__()
        self.T = T
        self.margin = 1
        

    

    def forward(self, x_metric, y, p_x, p_y):
        """
        - \sum_{i in B^c} log(Pc) - \sum_{i in B^c}  \sum_{k \ne c} log( (1 - Pk))

        Note:
        log(Exp(y)) makes y always positive, which is required for our loss.
        """
        if torch.isnan(x_metric).any():
            print("CAUTION: skipping NaN batch!!!")
            return torch.tensor(0)
        assert len(x_metric.shape) == 2, "Should only have batch and metric dimension."
        bs = x_metric.size(0)

        pos, neg = True, True

        # Init
        loss = None
        y_unique = torch.unique(y).squeeze()
        neg = False if len(y_unique.size()) == 0 else neg # If only from the same class, there is no neg term
        y_unique = y_unique.view(-1)
        
        for label_idx in range(y_unique.size(0)):  # [summation over i]
            c = y_unique[label_idx]

            # Select from batch
            xc_idxs = (y == c).nonzero().squeeze(dim=1)
            xc = x_metric.index_select(0, xc_idxs)

            xk_idxs = (y != c).nonzero().squeeze(dim=1)
            xk = x_metric.index_select(0, xk_idxs)

            p_idx = (p_y == c).nonzero().squeeze(dim=1)
            pc = p_x[p_idx].detach()
            pk = torch.cat([p_x[:p_idx], p_x[p_idx + 1:]]).detach()  # Other class prototypes
            

            lnL_pos = self.attractor(pc, pk, xc) if pos else 0  # Pos
            lnL_neg = self.repellor(pc, pk, xc, xk) if neg else 0  # Neg

            # Pos + Neg
            Loss_c = -lnL_pos - lnL_neg  # - \sum_{i in B^c} log(Pc) - \sum_{i in B^c}  \sum_{k \ne c} log( (1 - Pk))
            

            # Update loss
            loss = Loss_c if loss is None else loss + Loss_c

            # Checks
            assert lnL_pos <= 0
            assert lnL_neg <= 0
            assert loss >= 0 and loss < 1e10
        return loss / bs, -lnL_pos/bs, -lnL_neg/bs  # Make independent batch size

    def repellor(self, pc, pk, xc, xk):
        # Gather per other-class samples
        union_c = torch.cat([xc, pc])
        union_ck = torch.cat([union_c, pk]) #.clone().detach()
        c_split = union_c.shape[0]
        
        neg_Lterms = torch.mm(union_ck, xk.t()).div_(self.T).exp_()  # Last row is with own prototype
        pk_terms = neg_Lterms[c_split:].sum(dim=0).unsqueeze(0)  # For normalization
        pc_terms = neg_Lterms[:c_split]
        Pneg = pc_terms / (pc_terms + pk_terms)

        expPneg = (Pneg[:-1] + Pneg[-1].unsqueeze(0)) / 2  # Expectation pseudo/prototype
        lnPneg_k = expPneg.mul_(-1).add_(1).log_()  # log( (1 - Pk))
        lnPneg = lnPneg_k.sum()  # Sum over (pseudo-prototypes), and instances
        assert -10e10 < lnPneg <= 0
        return lnPneg

    def attractor(self, pc, pk, xc):
        # Union: Current class batch-instances, prototype, memory
        pos_union_l = [xc.clone()]
        pos_len = xc.shape[0]
        
        pos_union_l.append(pc)

        pos_union = torch.cat(pos_union_l)
        all_pos_union = torch.cat([pos_union, pk]).clone().detach()  # Include all other-class prototypes p_k
        pk_offset = pos_union.shape[0]  # from when starts p_k

        # Resulting distance columns are per-instance loss terms (don't include self => diagonal)
        pos_Lterms = torch.mm(all_pos_union, xc.t()).div_(self.T).exp_()  # .fill_diagonal_(0)
        
        mask = torch.eye(*pos_Lterms.shape).bool().to(pc.device)
        pos_Lterms = pos_Lterms.masked_fill(mask, 0)  # Fill with zeros

        Lc_pos = pos_Lterms[:pk_offset]
        Lk_pos = pos_Lterms[pk_offset:].sum(dim=0)  # sum column dist to pk's

        # Divide each of the terms by itself+ Lk term to get probability
        Pc_pos = Lc_pos / (Lc_pos + Lk_pos)
        expPc_pos = Pc_pos.sum(0) / (pos_len)  # Don't count self in
        lnL_pos = expPc_pos.log_().sum()

        # Sum instance loss-terms (per-column), divide by pk distances as well
        assert lnL_pos <= 0
        return lnL_pos

class CoPE3(ContinualModel):
    NAME = 'cope3'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super().__init__(backbone, loss, args, transform)
        
        self.net.num_classes = self.args.hidden_dim
        self.reset_classifier()

        self.buffer = Buffer(self.args.buffer_size, self.device, mode='balancoir')
        self.seen_so_far = torch.tensor([]).long().to(self.device)
        self.num_classes = get_dataset(args).N_TASKS * get_dataset(args).N_CLASSES_PER_TASK
        self.task = 0

        self.loss = PPPloss(T=self.args.loss_T)
        self.eye = torch.eye(self.num_classes).to(self.device)

        self.proto_shape = self.args.hidden_dim
        self.protos_x = torch.empty(0, self.proto_shape).to(self.device)
        self.protos_y = torch.empty(0).long().to(self.device)
        self.tmp_protx = torch.empty(0, self.proto_shape).to(self.device)
        self.tmp_protcnt = torch.empty(0).long().to(self.device)

        if self.args.sal_ckpt != None:
            self.load_cp()

    def begin_task(self, dataset):
        return 0

    def end_task(self, dataset):
        self.task += 1

    def to(self, device):
        super().to(device)
        self.seen_so_far = self.seen_so_far.to(device)
        
    def init_protos(self, new_classes):
        for c in new_classes:
            print('Initializing prototype for class {}'.format(c))
            p = torch.nn.functional.normalize(torch.empty((1, self.proto_shape)).uniform_(0, 1), p=2, dim=1).detach().to(self.device)
            self.protos_x = torch.cat([self.protos_x, p], dim=0)
            self.protos_y = torch.cat([self.protos_y, torch.tensor([c]).long().to(self.device)])
            self.tmp_protx = torch.cat([self.tmp_protx, torch.zeros_like(p)], dim=0)
            self.tmp_protcnt = torch.cat([self.tmp_protcnt, torch.zeros(1).long().to(self.device)])
    
    def accumulate_protos(self, f, y):
        ''' Accumulate prototype values for each item in a batch '''
        for c in torch.unique(y):
            p_tmp_batch = f[c == y].sum(dim=0)
            index = (self.protos_y == c).nonzero().squeeze(1)[0]
            self.tmp_protx[index] += p_tmp_batch.detach()
            self.tmp_protcnt[index] += len(p_tmp_batch)

    def update_protos(self):
        for c in self.protos_y:
            proto_ind = (self.protos_y == c).nonzero().squeeze(1)[0]
            if self.tmp_protcnt[proto_ind] > 0:
                # Momentum Update
                incr_p = self.tmp_protx[proto_ind] / self.tmp_protcnt[proto_ind]
                old_p = self.protos_x[proto_ind].clone()
                new_p = self.args.p_momentum * old_p + (1 - self.args.p_momentum) * incr_p
                new_p = torch.nn.functional.normalize(new_p, p=2,dim=0)
                # Update
                self.protos_x[proto_ind] = new_p.detach()
                assert not torch.isnan(self.protos_x).any()
                # Reset counters
                self.tmp_protx[proto_ind] *= 0
                self.tmp_protcnt[proto_ind] = 0

                # summarize update
                # d = (new_p - old_p).pow_(2).sum().sqrt_()
                # print("Class {} p-update: L2 delta={:.4f}".format(c, float(d.mean().item())))
    
        

    def forward(self, x):
        """ Deployment forward. Find closest prototype for each sample. """
        # nearest neighbor
        nd = self.proto_shape
        ns = x.size(0)

        if not len(self.seen_so_far):
            # no exemplar in memory yet, output uniform distr. over all classes
            out = torch.Tensor(ns, self.n_classes).fill_(1.0 / self.n_classes)
            if self.gpu:
                out = out.cuda()
            return out
            
        means = torch.ones(len(self.seen_so_far), nd).to(x.device) * -float('inf')
        means[self.protos_y] = self.protos_x # Class idx gets allocated its prototype

        preds = self.net(x)
        preds = F.normalize(preds, p=2, dim=1)  # L2-embedding normalization
        # Predict to nearest
        sims = []
        for sample_idx in range(ns):  # Per class
            simlr = torch.mm(means, preds[sample_idx].view(-1, preds[sample_idx].shape[-1]).t())  # Dot product
            sims.append(simlr.T)

        sims = torch.cat(sims, dim=0)
        if sims.shape[1] < self.num_classes:
            sims = torch.cat([sims, torch.ones(sims.shape[0], self.num_classes - sims.shape[1]).to(sims.device) * -float('inf')], dim=1)

        return sims  # return 1-of-C code, ns x nc

    def observe(self, inputs, labels, not_aug_inputs, epoch=None):
        labels = labels.long()
        present = labels.unique()
        new_classes = present[~torch.isin(present, self.seen_so_far)]
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        self.opt.zero_grad()
        overall_loss = 0
        for i in range(self.args.num_batches):
            update = i == self.args.num_batches - 1
            all_inputs, all_labels = inputs, labels
            if not self.buffer.is_empty():
                buf_inputs, buf_labels = self.buffer.get_data(
                    self.args.minibatch_size, transform=self.transform)
                all_inputs = torch.cat([all_inputs, buf_inputs], dim=0)
                all_labels = torch.cat([all_labels, buf_labels], dim=0)
            all_latents = self.net(all_inputs)
            all_latents = F.normalize(all_latents, p=2)
            if i == 0 and len(new_classes):
                self.init_protos(new_classes)
            
            # Accumulate prototypes update at each inner iteration
            # (only for buffer items -.-)
            if not self.buffer.is_empty():
                self.accumulate_protos(all_latents[len(labels):], buf_labels)
            
            if len(self.protos_x) > 1:
                loss, loss_pos, loss_neg = self.loss(all_latents, all_labels, self.protos_x, self.protos_y)
            else:
                loss, loss_pos, loss_neg = torch.tensor(0.).to(self.device), torch.tensor(0.).to(self.device), torch.tensor(0.).to(self.device)

            if loss.requires_grad:
                loss.backward()
                self.opt.step()

            self.autolog_wandb(locals())
            overall_loss += loss.item()

            if update:
                # Accumulate prototypes for stream examples
                self.accumulate_protos(all_latents[:len(labels)], labels)
                # And Update
                self.update_protos()
                self.buffer.add_data(examples=not_aug_inputs,
                                labels=labels)
            
        return overall_loss / self.args.num_batches 