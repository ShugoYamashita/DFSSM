import torch
from torch import nn as nn
from torch.nn import functional as F
import numpy as np

from basicsr.models.losses.loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']

@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')

class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', **kwargs):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)

class L1FreqLoss(nn.Module):
    """
    L1 (Mean Absolute Error, MAE) loss combining spatial-domain error and FFT-based error.

    Args:
        loss_weight (float): Weight assigned to the frequency domain loss. Default: 0.01.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """
    def __init__(self, loss_weight=0.01, reduction='mean', **kwargs):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f"Unsupported reduction mode: {reduction}. "
                             f"Valid options are: ['none', 'mean', 'sum']")

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.l1_loss = L1Loss(loss_weight, reduction)

    def forward(self, pred, target):
        # Compute the L1 loss in the frequency domain using the FFT difference
        diff = torch.fft.rfft2(pred) - torch.fft.rfft2(target)
        freq_loss = torch.mean(torch.abs(diff))

        # Combine frequency and spatial L1 losses
        return self.loss_weight * freq_loss + self.l1_loss(pred, target)
