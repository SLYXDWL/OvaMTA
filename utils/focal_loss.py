import torch
import torch.nn as nn
import torch.nn.functional as F

class Focal_loss(nn.Module):

    def __init__(self,weight = [0.2,0.3,0.2,0.3],gamma = 2):
        super(Focal_loss,self).__init__()
        self.gamma = gamma
        self.weight = torch.tensor(weight)

    def forward(self, preds, labels):
        weight = self.weight[labels]
        log_softmax = torch.log_softmax(preds,dim = 1)
        logpt = torch.gather(log_softmax,dim=1,index=labels.view(-1,1))
        logpt = logpt.view(-1)
        ce_loss = -logpt
        pt = torch.exp(logpt)
        focal_loss = weight*((1-pt)**self.gamma)*ce_loss

        return torch.mean(focal_loss)

class FocalLoss(torch.nn.Module):

    def __init__(self, weight, gamma=2, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.weight = weight

    def forward(self, input, target):
        # y = one_hot(target, input.size(-1))
        y = F.one_hot(target, input.size(-1))
        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit)  # cross entropy
        loss = loss * (1 - logit) ** self.gamma  # focal loss
        # print(loss)
        # # print(loss.type)

        loss = torch.mul(loss, self.weight)

        # return loss.sum()
        return loss.sum() / input.size(0)