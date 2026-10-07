import torch
import torch.nn as nn

class UncertaintyLoss(nn.Module):
    def __init__(self, loss_type='smooth_l1'):
        super().__init__()
        self.loss_type = loss_type
        if loss_type == 'l1':
            self.criterion = nn.L1Loss()
        elif loss_type == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

    def forward(self, pred, target):
        return self.criterion(pred, target)
