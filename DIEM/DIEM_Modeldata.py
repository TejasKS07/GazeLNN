import numpy as np
import pandas as pd
import cv2 as cv

import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl

from ncps.torch import CfCCell
from fractions import Fraction

import os
import random
import time
from tqdm import tqdm
from typing import Tuple, List, Dict, Optional

import numpy as np
import scipy.io as sio
from PIL import Image
from scipy.ndimage import gaussian_filter

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms    
"""
KL-DTW Loss for scanpath prediction, as used in GazeLNN / tSPM-Net.

The loss has two components:
    1. KL Divergence  — compares predicted vs ground truth heatmap at each
                        fixation step (spatial accuracy)
    2. DTW            — aligns predicted and ground truth sequences optimally
                        before computing the KL, handling timing differences
                        (temporal alignment)

The final loss is the DTW-aligned sum of per-step KL divergences,
averaged over the batch and masked so padded steps don't contribute.

Usage:
    criterion = KLDTWLoss()
    loss = criterion(predictions, targets, mask)

    predictions : (B, T, H, W)  — model output heatmaps (after softmax)
    targets     : (B, T, H, W)  — ground truth heatmaps (from dataset)
    mask        : (B, T)        — True = real fixation, False = padding
"""
# ---------------------------------------------------------------------------
# Per-step KL cost matrix  (B, T_pred, T_gt)
# ---------------------------------------------------------------------------

def _cost_matrix(
    pred:   torch.Tensor,   # (B, T, H, W)
    target: torch.Tensor,   # (B, T, H, W)
) -> torch.Tensor:
    """
    Compute pairwise KL divergence between every predicted step i and every
    ground truth step j, giving a cost matrix of shape (B, T_pred, T_gt).

    cost[b, i, j] = KL(target[b,j] || pred[b,i])
    """
    B, T, H, W = pred.shape
    eps = 1e-8

    # flatten spatial dims for broadcasting: (B, T, H*W)
    p = pred.view(B, T, -1).clamp(min=eps)
    t = target.view(B, T, -1).clamp(min=eps)

    # normalise
    p = p / p.sum(dim=-1, keepdim=True)
    t = t / t.sum(dim=-1, keepdim=True)

    # (B, T_pred, 1, H*W) vs (B, 1, T_gt, H*W)
    p_exp = p.unsqueeze(2)           # (B, T, 1, H*W)
    t_exp = t.unsqueeze(1)           # (B, 1, T, H*W)

    # KL(t || p) = sum(t * (log t - log p))
    cost = (t_exp * (t_exp.log() - p_exp.log())).sum(dim=-1)  # (B, T, T)

    return cost   # (B, T_pred, T_gt)


# Soft-DTW  (differentiable DTW via log-sum-exp smoothing)

def soft_dtw(cost_matrix: torch.Tensor, gamma: float = 0.1) -> torch.Tensor:
    """
    Soft-DTW: a fully differentiable approximation to DTW.

    Uses the soft-min (log-sum-exp) recurrence instead of hard min, so
    gradients flow back through the alignment path.

    Args:
        cost_matrix : (B, T, T) pairwise cost between pred and target steps
        gamma       : smoothing temperature.
                      gamma → 0  recovers hard DTW
                      gamma → ∞  makes all paths equally likely

    Returns:
        dtw_loss : (B,) soft-DTW distance for each item in the batch
    """
    B, T_pred, T_gt = cost_matrix.shape
    device = cost_matrix.device

    # initialise DP table with +inf
    # R[b, i, j] = soft-DTW distance up to (i, j)
    R = torch.full((B, T_pred + 1, T_gt + 1), fill_value=1e9, device=device)
    R[:, 0, 0] = 0.0

    for i in range(1, T_pred + 1):
        for j in range(1, T_gt + 1):
            c = cost_matrix[:, i - 1, j - 1]   # (B,)

            # three possible predecessor cells
            r0 = R[:, i - 1, j    ]   # came from above
            r1 = R[:, i    , j - 1]   # came from left
            r2 = R[:, i - 1, j - 1]   # came from diagonal

            # soft-min via log-sum-exp trick
            # softmin(a, b, c) = -gamma * log(exp(-a/γ) + exp(-b/γ) + exp(-c/γ))
            stacked = torch.stack([r0, r1, r2], dim=-1) / gamma        # (B, 3)
            soft_min = -gamma * torch.logsumexp(-stacked, dim=-1)      # (B,)

            R[:, i, j] = c + soft_min

    return R[:, T_pred, T_gt]   # (B,)


# ---------------------------------------------------------------------------
# Combined KL-DTW Loss
# ---------------------------------------------------------------------------

class KLDTWLoss(nn.Module):
    """
    KL-DTW loss combining spatial KL divergence with temporal DTW alignment.

    Args:
        gamma : soft-DTW smoothing temperature (default 0.1)
        eps   : numerical stability constant for KL divergence
    """

    def __init__(self, gamma: float = 0.1, eps: float = 1e-8):
        super().__init__()
        self.gamma = gamma
        self.eps   = eps

    def forward(
        self,
        predictions: torch.Tensor,   # (B, T, H, W)
        targets:     torch.Tensor,   # (B, T, H, W)
        mask:        torch.Tensor,   # (B, T) bool — True = real fixation
    ) -> torch.Tensor:
        """
        Args:
            predictions : (B, T, H, W) — model predicted heatmaps
            targets     : (B, T, H, W) — ground truth heatmaps
            mask        : (B, T)       — False on padded steps

        Returns:
            scalar loss averaged over the batch
        """
        B, T, H, W = predictions.shape

        # ---- apply softmax to predictions so they're proper distributions --
        # flatten spatial dims, softmax, reshape back
        # pred_dist = F.softmax(predictions.view(B, T, -1), dim=-1).view(B, T, H, W)
        pred_dist = predictions  # already normalized by model.forward

        # ---- zero out padded steps in both pred and target -----------------
        # so padded positions don't contribute to the cost matrix
        mask_spatial = mask.unsqueeze(-1).unsqueeze(-1).float()  # (B, T, 1, 1)
        pred_dist = pred_dist * mask_spatial
        targets   = targets   * mask_spatial

        # ---- build pairwise KL cost matrix ---------------------------------
        cost = _cost_matrix(pred_dist, targets)   # (B, T, T)

        # ---- soft-DTW over the cost matrix ---------------------------------
        dtw_loss = soft_dtw(cost, gamma=self.gamma)   # (B,)

        # ---- normalise by the number of real fixations per sequence --------
        seq_lengths = mask.sum(dim=1).float().clamp(min=1)   # (B,)
        dtw_loss = dtw_loss / seq_lengths

        return dtw_loss.mean()   # scalar
class CoordConv(nn.Module):
   
    def __init__(self, in_channels, out_channels, kernel_size=1, padding=0):
        super().__init__()

        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        batch_size, _, h, w = x.size()
        
    
        y_grid = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).expand(batch_size, 1, h, w)
        x_grid = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).expand(batch_size, 1, h, w)
        
        
        x_coord = torch.cat([x, y_grid, x_grid], dim=1)
        return self.conv(x_coord)
class GazeLLNArch(pl.LightningModule):

    def __init__(self, image_h: int = 256, image_w: int = 384, S: int = 8, 
                 hidden_size: int = 512, backbone_units: int = 1024):
        super().__init__()
        
        self.save_hyperparameters()

        
        self.hmap_h = self.hparams.image_h // self.hparams.S
        self.hmap_w = self.hparams.image_w // self.hparams.S
        self.hmap_feature_size = self.hmap_h * self.hmap_w

        
        mobilenet = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        self.feature_extractor = mobilenet.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.image_feature_size = 960 

        
        # Input CoordConv: enriches prev_hmap with spatial coords before CfC
        self.coordconv_in = CoordConv(in_channels=1, out_channels=1, kernel_size=3, padding=1)

        # Output CoordConv: applied to projected heatmap before feedback
        self.coordconv_out = CoordConv(in_channels=1, out_channels=1, kernel_size=3, padding=1)

        
        cfc_input_size = self.image_feature_size + self.hmap_feature_size
        self.cfc_cell = CfCCell(
            input_size=cfc_input_size,
            hidden_size=self.hparams.hidden_size,
            backbone_activation="lecun_tanh",
            backbone_units=self.hparams.backbone_units,
            backbone_layers=1
        )
        
        
        self.project = nn.Linear(self.hparams.hidden_size, self.hmap_feature_size)

    def extract_features(self, x):
        """Extracts and caches visual features using the MobileNet backbone."""
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)  # Flattens to (B, 960)
        return x

    def forward(self, vis_features, prev_hmap, hx, ts):
        """
        Args:
            vis_features: (B, 960) — precomputed by extract_features
            prev_hmap: Previous fixation heatmap of shape (B, hmap_h, hmap_w) or (B, 1, hmap_h, hmap_w)
            hx: Hidden state tensor for the CfCCell
            ts: Elapsed time timespan tensor of shape (B,)
        """

        # Ensure prev_hmap has a channel dimension (B, 1, hmap_h, hmap_w)
        if prev_hmap.dim() == 3:
            prev_hmap = prev_hmap.unsqueeze(1)              
        hmap_coords = self.coordconv_in(prev_hmap).flatten(1)          # Shape: (B, 1536)
        
        # Ensure ts has shape (B, 1) instead of just (B,)
        if ts.dim() == 1:
            ts = ts.view(-1, 1)
        
        # hx cannot be set as None
        if hx is None:
            hx = torch.zeros(vis_features.size(0), self.hparams.hidden_size, device=vis_features.device)

        x = torch.cat([vis_features, hmap_coords], dim=1)
        x, hx = self.cfc_cell(x, hx, ts)                # Shape: (B, 1536)
        
        
        out_hmap = self.project(x).view(-1, 1, self.hmap_h, self.hmap_w)  # Shape: (B, 1, 32, 48)
        out_hmap = self.coordconv_out(out_hmap)
        
        # normalize to a proper spatial distribution before feedback
        B_ = out_hmap.shape[0]
        out_hmap_flat = out_hmap.view(B_, -1)                              # (B, 32*48)
        out_hmap_flat = F.softmax(out_hmap_flat, dim=-1)                   # sums to 1 per sample
        out_hmap = out_hmap_flat.view(B_, 1, self.hmap_h, self.hmap_w)    # (B, 1, 32, 48)

        return out_hmap, hx