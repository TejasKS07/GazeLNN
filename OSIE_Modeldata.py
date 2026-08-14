import numpy as np
import pandas as pd
import cv2 as cv

import torch
import torch.nn as nn
import torchvision.models as models
import pytorch_lightning as pl
import torch.optim as optim

from ncps.torch import CfCCell
from fractions import Fraction

import os
import random
from tqdm import tqdm
from typing import Tuple, List, Dict, Optional

import numpy as np
import scipy.io as sio
from PIL import Image
from scipy.ndimage import gaussian_filter

from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision import transforms    

def _gaussian_heatmap(
    x: float,
    y: float,
    img_w: int,
    img_h: int,
    hmap_w: int,
    hmap_h: int,
    sigma: float = 1.5,
) -> np.ndarray:
    """
    Place a Gaussian blob at pixel (x, y) on a downsampled heatmap grid.

    Args:
        x, y      : fixation location in *original* image pixel coordinates
        img_w/h   : original image size (before downsampling)
        hmap_w/h  : heatmap size (= img / downsample)
        sigma     : Gaussian std-dev in heatmap pixels

    Returns:
        hmap : (hmap_h, hmap_w) float32 array summing to 1
    """
    hmap = np.zeros((hmap_h, hmap_w), dtype=np.float32)

    # scale fixation coords to heatmap space
    xi = int(round(x * hmap_w / img_w))
    yi = int(round(y * hmap_h / img_h))

    xi = np.clip(xi, 0, hmap_w - 1)
    yi = np.clip(yi, 0, hmap_h - 1)

    hmap[yi, xi] = 1.0
    hmap = gaussian_filter(hmap, sigma=sigma)

    total = hmap.sum()
    if total > 0:
        hmap /= total   

    return hmap

def _pad_or_truncate(
    seq: List,
    max_len: int,
    pad_value,
) -> Tuple[List, List[bool]]:
    """Truncate to max_len and return (padded_seq, mask).
    mask[i] = True  →  real fixation
    mask[i] = False →  padding (loss should ignore these)
    """
    real_len = min(len(seq), max_len)
    mask = [True] * real_len + [False] * (max_len - real_len)
    seq = seq[:real_len] + [pad_value] * (max_len - real_len)
    return seq, mask

def extract_scanpaths(
    entry: Dict,
    min_len: int = 4,
    max_len: int = 8,
) -> List[Dict]:
    """
    Extract per-subject scanpaths from one OSIE entry.

    Returns a list of dicts, each with keys:
        img_name  : str
        fix_x     : list[float]  (length <= max_len)
        fix_y     : list[float]
        fix_dt    : list[float]  (durations in ms)
        mask      : list[bool]   (True = real, False = padding)
    """
    scanpaths = []
    img_name = entry['img']

    for subj in entry['subjects']:
        xs  = np.atleast_1d(np.array(subj['fix_x'],        dtype=np.float32)).tolist()
        ys  = np.atleast_1d(np.array(subj['fix_y'],        dtype=np.float32)).tolist()
        dts = np.atleast_1d(np.array(subj['fix_duration'], dtype=np.float32)).tolist()

        if len(xs) < min_len:
            continue  # discard short scanpaths per GazeLNN / tSPM-Net

        xs, mask = _pad_or_truncate(xs, max_len, pad_value=xs[-1])
        ys, _    = _pad_or_truncate(ys, max_len, pad_value=ys[-1])
        dts, _   = _pad_or_truncate(dts, max_len, pad_value=0.0)

        scanpaths.append({
            'img_name': img_name,
            'fix_x':    xs,
            'fix_y':    ys,
            'fix_dt':   dts,
            'mask':     mask,
        })

    return scanpaths

class OSIEDataset(Dataset):
    """
    PyTorch Dataset for OSIE scanpath prediction.

    Each item is one (image, scanpath) pair — one subject's viewing of one
    image.  With 700 images × ~15 subjects the dataset has ~10 500 items
    before filtering.

    Args:
        data_root   : path to the osie/ folder containing stimuli/ and eye/
        split       : 'train' | 'val' | 'test'
        img_size    : (H, W) to resize images to. GazeLNN uses (256, 384).
        downsample  : spatial downsampling factor for heatmaps.
                      heatmap size = (img_size[0]//ds, img_size[1]//ds)
        min_len     : discard scanpaths shorter than this
        max_len     : truncate/pad scanpaths to this length
        sigma       : Gaussian std-dev (in heatmap pixels) for fixation blobs
        seed        : random seed for the 80/10/10 image split
    """

    # 80 / 10 / 10  image-level split (consistent with the paper)
    SPLIT_RATIOS = {'train': 0.80, 'val': 0.10, 'test': 0.10}

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        img_size: Tuple[int, int] = (256, 384),
        downsample: int = 8,
        min_len: int = 4,
        max_len: int = 8,
        sigma: float = 1.5,
        seed: int = 42,
    ):
        assert split in ('train', 'val', 'test')
        self.stimuli_dir = os.path.join(data_root, 'stimuli')
        self.img_size    = img_size          # (H, W)
        self.hmap_size   = (img_size[0] // downsample, img_size[1] // downsample)
        self.max_len     = max_len
        self.sigma       = sigma

        # ---- load fixations ------------------------------------------------
        mat_path = os.path.join(data_root, 'eye', 'fixations.mat')
        raw = sio.loadmat(mat_path, simplify_cells=True)
        all_entries = raw['fixations']          # length 700

        # ---- deterministic image-level split --------------------------------
        n_total = len(all_entries)
        indices = list(range(n_total))
        rng = random.Random(seed)
        rng.shuffle(indices)

        n_train = int(n_total * 0.80)
        n_val   = int(n_total * 0.10)

        if split == 'train':
            chosen = indices[:n_train]
        elif split == 'val':
            chosen = indices[n_train : n_train + n_val]
        else:
            chosen = indices[n_train + n_val:]

        # ---- flatten to (image, subject) scanpath pairs --------------------
        self.samples: List[Dict] = []
        for idx in chosen:
            entry = all_entries[idx]
            scanpaths = extract_scanpaths(entry, min_len=min_len, max_len=max_len)
            self.samples.extend(scanpaths)

        # ---- image transforms ----------------------------------------------
        self.transform = transforms.Compose([
            transforms.Resize(img_size),          # PIL Resize takes (H, W)
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],       # ImageNet stats
                std =[0.229, 0.224, 0.225],
            ),
        ])

        print(
            f"[OSIEDataset] split={split:5s} | "
            f"images={len(chosen)} | scanpath pairs={len(self.samples)}"
        )

    
    def __len__(self) -> int:
        return len(self.samples)

    
    def __getitem__(self, idx: int):
        sample   = self.samples[idx]
        img_H, img_W = self.img_size
        hmap_H, hmap_W = self.hmap_size

        #  load & transform image 
        img_path = os.path.join(self.stimuli_dir, sample['img_name'])
        img = Image.open(img_path).convert('RGB')

        # original size needed to scale fixation coords
        orig_W, orig_H = img.size   # PIL gives (W, H)

        img_tensor = self.transform(img)   # (3, H, W)

        # ---- build heatmap sequence ----------------------------------------
        heatmap_seq = []
        for i in range(self.max_len):
            if sample['mask'][i]:
                hmap = _gaussian_heatmap(
                    x=sample['fix_x'][i],
                    y=sample['fix_y'][i],
                    img_w=orig_W,
                    img_h=orig_H,
                    hmap_w=hmap_W,
                    hmap_h=hmap_H,
                    sigma=self.sigma,
                )
            else:
                # padding step — zero heatmap
                hmap = np.zeros((hmap_H, hmap_W), dtype=np.float32)

            heatmap_seq.append(hmap)

        heatmap_seq = torch.from_numpy(
            np.stack(heatmap_seq, axis=0)
        )  # (T, hmap_H, hmap_W)

        # ---- fixation durations (∆t for CfC) --------------------------------
        dt_seq = torch.tensor(sample['fix_dt'], dtype=torch.float32)  # (T,)

        # first fixation ∆t = 0 per the paper
        dt_seq[0] = 0.0

        # ---- padding mask ---------------------------------------------------
        mask = torch.tensor(sample['mask'], dtype=torch.bool)  # (T,)

        return img_tensor, heatmap_seq, dt_seq, mask

def build_dataloaders(
    data_root: str,
    img_size: Tuple[int, int] = (256, 384),
    downsample: int = 8,
    min_len: int = 4,
    max_len: int = 8,
    sigma: float = 1.5,
    batch_size: int = 8,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).

    Each batch yields:
        imgs         : (B, 3, H, W)       — normalised image tensor
        heatmap_seq  : (B, T, Hd, Wd)     — per-fixation Gaussian heatmaps
        dt_seq       : (B, T)             — fixation durations in ms
        padding_mask : (B, T)             — True where fixation is real
    """
    kwargs = dict(
        data_root  = data_root,
        img_size   = img_size,
        downsample = downsample,
        min_len    = min_len,
        max_len    = max_len,
        sigma      = sigma,
        seed       = seed,
    )

    train_ds = OSIEDataset(split='train', **kwargs)
    val_ds   = OSIEDataset(split='val',   **kwargs)
    test_ds  = OSIEDataset(split='test',  **kwargs)

    loader_kwargs = dict(
        batch_size  = batch_size,
        num_workers = num_workers,
        pin_memory  = True,
    )

    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader

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


# ---------------------------------------------------------------------------
# Soft-DTW  (differentiable DTW via log-sum-exp smoothing)
# ---------------------------------------------------------------------------

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
        pred_dist = F.softmax(
            predictions.view(B, T, -1), dim=-1
        ).view(B, T, H, W)

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

"""
Training and evaluation loops for GazeLNN scanpath prediction.

Usage:
    from train import train

    train(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        device       = "cuda",
        epochs       = 100,
        lr           = 0.0001,
        patience     = 20,
        checkpoint   = "OSIE_Model.pt",
    )
"""






# ---------------------------------------------------------------------------
# single training epoch
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device):
    """
    Runs one full pass over the training dataloader.

    Args:
        model     : GazeLLNArch instance
        loader    : training DataLoader
        optimizer : Adam optimizer
        criterion : KLDTWLoss instance
        device    : "cuda" or "cpu"

    Returns:
        avg_loss : float — mean loss over all batches
    """
    model.train()
    total_loss = 0.0

    # Adds a progress bar to see the batches processing
    for imgs, hmap_seq, dt_seq, mask in tqdm(loader, desc="Training", leave=False):

        # ---- move everything to device -------------------------------------
        imgs    = imgs.to(device)       # (B, 3, H, W)
        hmap_seq = hmap_seq.to(device)  # (B, T, hmap_H, hmap_W)  ← targets
        mask    = mask.to(device)       # (B, T)
        # dt_seq not used since ts=1 constantly per paper

        B  = imgs.shape[0]
        T  = hmap_seq.shape[1]
        hmap_H = hmap_seq.shape[2]
        hmap_W = hmap_seq.shape[3]

        # ---- initialise recurrent state ------------------------------------
        prev_hmap = torch.zeros(B, 1, hmap_H, hmap_W, device=device)    # (B, 1, hmap_H, hmap_W)
        hx        = None                                                # CfC hidden state
        ts        = torch.ones(B, device=device)                        # fixed ts=1 per paper

        # ---- autoregressive forward pass -----------------------------------
        predictions = []

        for t in range(T):
            out_hmap, hx = model(imgs, prev_hmap, hx, ts)  # (B, hmap_H, hmap_W)
            predictions.append(out_hmap)

            # next step uses current prediction as history
            # detach to avoid backprop through the full sequence history
            prev_hmap = out_hmap.detach().unsqueeze(1)     # (B, 1, hmap_H, hmap_W)

        # stack predictions: (B, T, hmap_H, hmap_W)
        predictions = torch.stack(predictions, dim=1)

        # ---- compute loss --------------------------------------------------
        loss = criterion(predictions, hmap_seq, mask)

        # ---- backprop ------------------------------------------------------
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ---------------------------------------------------------------------------
# single evaluation epoch (val or test)
# ---------------------------------------------------------------------------


def eval_epoch(model, loader, criterion, device):
    """
    Runs one full pass over a validation or test dataloader.
    No gradient computation — inference only.

    Args:
        model     : GazeLLNArch instance
        loader    : val or test DataLoader
        criterion : KLDTWLoss instance
        device    : "cuda" or "cpu"

    Returns:
        avg_loss : float — mean loss over all batches
    """
    
    model.eval()
    with torch.no_grad():
        total_loss = 0.0

        for imgs, hmap_seq, dt_seq, mask in tqdm(loader, desc="Validating", leave=False):

            imgs     = imgs.to(device)
            hmap_seq = hmap_seq.to(device)
            mask     = mask.to(device)

            B      = imgs.shape[0]
            T      = hmap_seq.shape[1]
            hmap_H = hmap_seq.shape[2]
            hmap_W = hmap_seq.shape[3]

            prev_hmap = torch.zeros(B, 1, hmap_H, hmap_W, device=device)
            hx        = None
            ts        = torch.ones(B, device=device)

            predictions = []

            for t in range(T):
                out_hmap, hx = model(imgs, prev_hmap, hx, ts)
                predictions.append(out_hmap)
                prev_hmap = out_hmap.unsqueeze(1)   # no detach needed — no_grad context

            predictions = torch.stack(predictions, dim=1)
            loss = criterion(predictions, hmap_seq, mask)
            total_loss += loss.item()

        return total_loss / len(loader)


def train(
    model,
    train_loader,
    val_loader,
    device:      str   = "cuda",
    epochs:      int   = 100,
    lr:          float = 0.0001,
    patience:    int   = 20,
    gamma:       float = 0.1,
    checkpoint:  str   = "OSIE_Model.pt",
):
    """
    Full training loop with Adam optimiser, KL-DTW loss, and early stopping.

    Args:
        model        : GazeLLNArch instance
        train_loader : training DataLoader
        val_loader   : validation DataLoader
        device       : "cuda" or "cpu"
        epochs       : maximum number of epochs (default 100)
        lr           : learning rate for Adam (default 0.0001)
        patience     : early stopping patience in epochs (default 20)
        gamma        : soft-DTW smoothing temperature (default 0.1)
        checkpoint   : path to save the best model weights

    Returns:
        history : dict with keys "train_loss" and "val_loss" (lists)
    """
    model     = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = KLDTWLoss(gamma=gamma)

    best_val_loss    = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'val_loss': []}

    print(f"Training on {device} | epochs={epochs} | lr={lr} | patience={patience}\n")

    for epoch in range(1, epochs + 1):

        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = eval_epoch(model, val_loader,   criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        print(f"Epoch {epoch:03d}/{epochs} | train={train_loss:.4f} | val={val_loss:.4f}", end="")

        # ---- check for improvement ----------------------------------------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint)
            print("  ← saved best model", end="")
        else:
            epochs_no_improve += 1
            print(f"  (no improvement for {epochs_no_improve}/{patience} epochs)", end="")

        print()

        # ---- early stopping -----------------------------------------------
        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            print(f"Best val loss: {best_val_loss:.4f}")
            break

    print("\nTraining complete.")
    print(f"Best model saved to: {checkpoint}")

    return history


# ---------------------------------------------------------------------------
# convenience: load best weights and run on test set
# ---------------------------------------------------------------------------

def evaluate_test(model, test_loader, device="cuda", checkpoint="OSIE_Model.pt"):
    """
    Loads the best saved weights and evaluates on the test set.

    Returns:
        test_loss : float
    """
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model = model.to(device)
    criterion = KLDTWLoss()

    test_loss = eval_epoch(model, test_loader, criterion, device)
    print(f"Test loss: {test_loss:.4f}")
    return test_loss


 

    # model = GazeLLNArch().to(device)

    # history = train(
    #     model        = model,
    #     train_loader = train_loader,
    #     val_loader   = val_loader,
    #     device       = device,
    #     epochs       = 100,
    #     lr           = 0.0001,
    #     patience     = 20,
    #     checkpoint   = "OSIE_Model.pt",
    # )

    # evaluate_test(model, test_loader, device=device)

checkpoint = "GazeLNN_bestmodel.pt"

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

        
        self.coordconv = CoordConv(in_channels=1, out_channels=1, kernel_size=3, padding=1)

        
        cfc_input_size = self.image_feature_size + self.hmap_feature_size
        self.cfc_cell = CfCCell(
            input_size=cfc_input_size,
            hidden_size=self.hparams.hidden_size,
            backbone_activation="lecun_tanh",
            backbone_units=self.hparams.backbone_units,
            backbone_layers=1
        )
        
        
        self.project = nn.Linear(self.hparams.hidden_size, self.hmap_feature_size)

    def forward(self, image, prev_hmap, hx, ts):
        """
        Args:
            image: Visual stimulus tensor of shape (B, 3, H, W)
            prev_hmap: Previous fixation heatmap of shape (B, hmap_h, hmap_w) or (B, 1, hmap_h, hmap_w)
            hx: Hidden state tensor for the CfCCell
            ts: Elapsed time timespan tensor of shape (B,)
        """
        
        vis_features = self.feature_extractor(image)
        vis_features = self.pool(vis_features).flatten(1)  # Shape: (B, 960)

        # Ensure prev_hmap has a channel dimension (B, 1, hmap_h, hmap_w)
        if prev_hmap.dim() == 3:
            prev_hmap = prev_hmap.unsqueeze(1)
            
    
        # hmap_coords = self.coordconv(prev_hmap).flatten(1)
        hmap_coords = prev_hmap.flatten(1) # Shape: (B, hmap_h * hmap_w) (using coordconv before flattenning might be useless)
        
        # Ensure ts has shape (B, 1) instead of just (B,)
        if ts.dim() == 1:
            ts = ts.view(-1, 1)
        
        # hx cannot be set as None
        if hx is None:
            batch_size = image.size(0)
            hx = torch.zeros(batch_size, self.hparams.hidden_size, device=image.device)

        x = torch.cat([vis_features, hmap_coords], dim=1)
        x, hx = self.cfc_cell(x, hx, ts)
        
        
        out_hmap = self.project(x)

        
        out_hmap = out_hmap.view(-1, self.hmap_h, self.hmap_w)
        

        return out_hmap, hx

DATA_ROOT = 'OSIE'

if __name__ == "__main__":
    train_loader, val_loader, test_loader = build_dataloaders(
        data_root  = DATA_ROOT,
        img_size   = (256, 384),
        downsample = 8,
        batch_size = 8,                         # Decrease this if OOM error occurs
        num_workers=0
)