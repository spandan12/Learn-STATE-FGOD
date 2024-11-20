#!/usr/bin/env python3
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np
from torch.autograd import Variable
from ..utils import logging
logger = logging.get_logger("visual_prompt")


class SigmoidLoss(nn.Module):
    def __init__(self, cfg=None):
        super(SigmoidLoss, self).__init__()

    def is_single(self):
        return True

    def is_local(self):
        return False

    def multi_hot(self, labels: torch.Tensor, nb_classes: int) -> torch.Tensor:
        labels = labels.unsqueeze(1)  # (batch_size, 1)
        target = torch.zeros(
            labels.size(0), nb_classes, device=labels.device
        ).scatter_(1, labels, 1.)
        # (batch_size, num_classes)
        return target

    def loss(
        self, logits, targets, per_cls_weights,
        multihot_targets: Optional[bool] = False
    ):
        # targets: 1d-tensor of integer
        # Only support single label at this moment
        # if len(targets.shape) != 2:
        num_classes = logits.shape[1]
        targets = self.multi_hot(targets, num_classes)

        loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none")
        # logger.info(f"loss shape: {loss.shape}")
        weight = torch.tensor(
            per_cls_weights, device=logits.device
        ).unsqueeze(0)
        # logger.info(f"weight shape: {weight.shape}")
        loss = torch.mul(loss.to(torch.float32), weight.to(torch.float32))
        return torch.sum(loss) / targets.shape[0]

    def forward(
        self, pred_logits, targets, per_cls_weights, multihot_targets=False
    ):
        loss = self.loss(
            pred_logits, targets,  per_cls_weights, multihot_targets)
        return loss


class SoftmaxLoss(SigmoidLoss):
    def __init__(self, cfg=None):
        super(SoftmaxLoss, self).__init__()

    def loss(self, logits, targets, per_cls_weights, kwargs):
        weight = torch.tensor(
            per_cls_weights, device=logits.device
        )
        #DONE: Change remove weight
        loss = F.cross_entropy(logits, targets, reduction="none")

        return torch.sum(loss) / targets.shape[0]

class STATELOSS(nn.Module):
    def __init__(self, cfg, T):
        super(STATELOSS, self).__init__()
        self.T = T
        self.global_states = torch.from_numpy(T["global_states"])
        self.is_ancestors = torch.from_numpy(1 *  np.invert(self.T['is_ancestor_mat'][ self.T['root']:]))
        
        self.kld_u_const = math.log(len(T["wnids"]))
        self.relevant = [torch.from_numpy(rel) for rel in T["relevant"]]
        self.labels_relevant = torch.from_numpy(T["labels_relevant"].astype(np.uint8))
        ch_slice = T["ch_slice"]
        self.class_weight = torch.ones(ch_slice[-1]) / ch_slice[-1]
        self.cfg = cfg

    def is_local(self):
        return False

    def forward(self, input, target, cls_weights):  # input = Variable(logits), target = labels
        loss = Variable(torch.zeros(1).cuda())
        if self.cfg.SOLVER.LOO == 0:
            new_input = torch.transpose(torch.matmul(self.global_states.float(), torch.transpose(input.float(), 0, 1)), 0, 1) + 1
            effective_evidence = new_input[:, : len(self.T["wnids"])]
        
        # novel loss
        target_novel = self.labels_relevant[target]
        for i, rel in enumerate(self.relevant):
            if target_novel[:, i].any():
                relevant_loc = target_novel[:, i].nonzero().view(-1)
                
                # )
                if self.cfg.SOLVER.LOO == 0:
                    loss += (
                        -F.log_softmax(effective_evidence[relevant_loc][:, rel], dim=1)[
                            :, 0
                        ].mean()
                        * self.class_weight[i]
                    )
                else:
                    loss += (
                        -F.log_softmax(input[relevant_loc][:, rel], dim=1)[
                            :, 0
                        ].mean()
                        * self.class_weight[i]
                    )
        
        loss *= self.cfg.NOVELWEIGHT
        logger.info(f"novel loss: {loss[0]}")
        
        # log_probs = F.log_softmax(effective_evidence, dim=1)
        # loss +=  F.nll_loss(log_probs, Variable(target))
        
        
        if self.cfg.SOLVER.LOO == 0:
            known_loss = self.cfg.KNOWNWEIGHT * F.cross_entropy(effective_evidence, target, reduction="none").mean()
            loss+= known_loss
            logger.info(f"known loss: {known_loss}")
            # loss += self.cfg.KNOWNWEIGHT * F.cross_entropy(effective_evidence, target, reduction="none").mean()
        else:
            loss += F.cross_entropy(input, target, reduction="none").mean()
        
        return loss[0]

    def cuda(self, device=None):
        super(STATELOSS, self).cuda(device)
        self.relevant = [rel.cuda(device) for rel in self.relevant]
        self.labels_relevant = self.labels_relevant.cuda(device)
        self.global_states = self.global_states.cuda(device)
        self.is_ancestors = self.is_ancestors.cuda(device)
        return self

LOSS = {
    "softmax": SoftmaxLoss,
}


def build_loss(cfg, taxonomy):
    loss_name = cfg.SOLVER.LOSS
    assert loss_name in LOSS, \
        f'loss name {loss_name} is not supported'
    
    # loss_fn = LOSS[loss_name]
    loss_fn = STATELOSS
    
    if not loss_fn:
        return None
    else:
        # return loss_fn(cfg).cuda()
        return loss_fn(cfg, taxonomy).cuda()
