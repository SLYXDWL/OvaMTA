import torch.nn as nn
import torch
import torch.nn.functional as F


from utils.registry import LOSSES

def smooth_l1_loss(pred, target, beta=1.0, reduction='mean'):
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    diff = torch.abs(pred - target)
    loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                       diff - 0.5 * beta)
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.sum() / pred.numel()
    elif reduction_enum == 2:
        return loss.sum()


def weighted_smoothl1(pred, target, weight, beta=1.0, avg_factor=None):
    if avg_factor is None:
        avg_factor = torch.sum(weight > 0).float().item() / 4 + 1e-6
        # print(avg_factor)
    loss = smooth_l1_loss(pred, target, beta, reduction='none')
    # print( loss ,weight)
    return torch.sum(loss * weight)[None] / avg_factor

@LOSSES.register_module
class SmoothL1Loss(nn.Module):

    def __init__(self, beta=1.0, loss_weight=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight, *args, **kwargs):
        target = F.one_hot(target, pred.size(-1))
        # print(target.shape)
        loss_bbox = self.loss_weight * weighted_smoothl1(
            pred, target, weight, beta=self.beta, *args, **kwargs)
        return loss_bbox
