import torch
import torch.nn as nn

class CharbonnierLoss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, x, y, **kwargs):
        diff = x - y
        loss = self.loss_weight * torch.sqrt(diff * diff + self.eps)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss 