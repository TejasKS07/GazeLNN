"""
Fine-tuning loop for GazeLLNArch on continuous video (DIEM), building on
dataset.py's rolling-window Dataset -- rewritten against the ACTUAL model in
modeldata.py and the real training conventions in trainGazeLNN.ipynb (which
produced OSIE_model.pt), not an earlier draft. Concretely, this version
matches (and was verified against, see the chat writeup) the following:

  1. forward() takes precomputed `vis_features` (B, 960), not raw images --
     call model.extract_features(images) first.
  2. The model's output is ALREADY softmax-normalized internally (each
     sample sums to 1) and has a channel dim: (B, 1, Hg, Wg).
  3. hx=None and ts of shape (B,) are handled INSIDE forward() now.
  4. Training in trainGazeLNN.ipynb is FREE-RUNNING / autoregressive, not
     teacher-forced: prev_hmap for step t+1 is the model's own prediction
     from step t (detached during training). Kept as the default here, with
     an opt-in teacher_forcing=True mode available.
  5. The autoregressive prior (prev_hmap before the first real prediction)
     is a small Gaussian centered on the heatmap grid (sigma=1.5) -- see
     _center_gaussian() below, ported directly from trainGazeLNN.ipynb.
  6. train_epoch/eval_epoch in trainGazeLNN.ipynb wrap `extract_features` in
     torch.no_grad() UNCONDITIONALLY -- the backbone is never updated
     through their entire OSIE run. Matched here via train_backbone=False
     by default.

This script extends their per-batch single-image feature-caching pattern to
the continuous-video equivalent: one batched extract_features() call over
all T distinct frames in a window.

Loss: ContinuousKLLoss replaces KLDTWLoss (continuous frames are already 1:1
time-aligned to ground truth, so DTW's alignment search is unnecessary),
updated to match the model's already-softmaxed output (no double softmax).

Cross-window continuation: both the CfC hidden state and the free-running
prev_hmap are tracked in VideoStateStore, keyed by video_name (not batch
position, since VideoOrderedBatchSampler's batches can shrink mid-epoch).
"""

from __future__ import annotations

import time
from tqdm import tqdm
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader

import config as cfg
from dataset import build_datasets, VideoOrderedBatchSampler


def _gaussian_heatmap(x, y, img_w, img_h, hmap_w, hmap_h, sigma=1.5):
    hmap = np.zeros((hmap_h, hmap_w), dtype=np.float32)
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


def _center_gaussian(batch_size, hmap_h, hmap_w, device, sigma=1.5):
    """Gaussian blob at the heatmap center. Shape: (B, 1, hmap_h, hmap_w).
    Ported from trainGazeLNN.ipynb so the fine-tuning prior matches what
    OSIE.pt was actually trained with."""
    hmap = _gaussian_heatmap(
        x=hmap_w / 2, y=hmap_h / 2,
        img_w=hmap_w, img_h=hmap_h, hmap_w=hmap_w, hmap_h=hmap_h, sigma=sigma,
    )
    hmap_t = torch.from_numpy(hmap).to(device)
    return hmap_t.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1).clone()


class ContinuousKLLoss(nn.Module):
    """Per-timestep masked KL divergence. predictions are assumed ALREADY
    softmax-normalized by the model -- do not apply softmax again here."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        B, T, H, W = predictions.shape
        pred = predictions.reshape(B, T, -1).clamp(min=self.eps)
        target = targets.reshape(B, T, -1).clamp(min=self.eps)
        pred = pred / pred.sum(dim=-1, keepdim=True)
        target = target / target.sum(dim=-1, keepdim=True)

        kl = (target * (target.log() - pred.log())).sum(dim=-1)  # (B, T)
        mask_f = mask.float()
        kl = kl * mask_f

        denom = mask_f.sum(dim=1).clamp(min=1)
        per_sequence = kl.sum(dim=1) / denom
        return per_sequence.mean()


class VideoStateStore:
    def __init__(self):
        self._store: Dict[str, Dict[str, torch.Tensor]] = {}

    def clear(self):
        self._store.clear()

    def gather(self, video_names, is_video_start, hidden_size, hmap_h, hmap_w, device):
        B = len(video_names)
        hx = torch.zeros(B, hidden_size, device=device)
        prev_hmap = _center_gaussian(B, hmap_h, hmap_w, device=device)
        for i, name in enumerate(video_names):
            if bool(is_video_start[i]):
                continue
            state = self._store.get(name)
            if state is None:
                continue
            hx[i] = state["hx"].to(device)
            prev_hmap[i] = state["prev_hmap"].to(device)
        return hx, prev_hmap

    def scatter(self, video_names, hx, last_prev_hmap):
        for i, name in enumerate(video_names):
            self._store[name] = {
                "hx": hx[i].detach().clone(),
                "prev_hmap": last_prev_hmap[i].detach().clone(),
            }


def run_window(model, batch, state_store: VideoStateStore, device,
               ts_mode: str = "real", teacher_forcing: bool = False,
               train_backbone: bool = False):
    """Returns: predictions (B,T,Hg,Wg), targets (B,T,Hg,Wg), valid (B,T) bool"""
    frames = batch["frames"].to(device)
    heatmaps = batch["heatmaps"].to(device)
    valid = batch["valid"].to(device)
    delta_t = batch["delta_t"].to(device)
    is_start = batch["is_video_start"]
    video_names = batch["video_name"]

    B, T = frames.shape[:2]
    hidden_size = model.hparams.hidden_size
    Hg, Wg = model.hmap_h, model.hmap_w

    hx, prev_hmap = state_store.gather(video_names, is_start, hidden_size, Hg, Wg, device)

    flat_frames = frames.reshape(B * T, *frames.shape[2:])
    if train_backbone:
        vis_features_all = model.extract_features(flat_frames)
    else:
        with torch.no_grad():
            vis_features_all = model.extract_features(flat_frames)
    vis_features_all = vis_features_all.view(B, T, -1)

    predictions = []
    for t in range(T):
        ts_t = delta_t[:, t] if ts_mode == "real" else torch.ones(B, device=device)

        out_hmap, hx = model(vis_features_all[:, t], prev_hmap, hx, ts_t)
        predictions.append(out_hmap.squeeze(1))

        if teacher_forcing:
            prev_hmap = heatmaps[:, t].unsqueeze(1)
        else:
            prev_hmap = out_hmap.detach()

    predictions = torch.stack(predictions, dim=1)
    state_store.scatter(video_names, hx, prev_hmap)

    return predictions, heatmaps, valid


def train_epoch(model, loader, optimizer, criterion, device, scaler=None, ts_mode="real",
                 teacher_forcing=False, train_backbone=False, grad_clip=1.0, epoch_idx=0):
    model.train()
    state_store = VideoStateStore()
    total_loss, n_batches = 0.0, 0

    pbar = tqdm(loader, desc=f"Train Epoch {epoch_idx}", leave=False)

    for batch in pbar:
        with torch.amp.autocast('cuda', enabled=scaler is not None and scaler.is_enabled()):
            predictions, targets, valid = run_window(
                model, batch, state_store, device, ts_mode, teacher_forcing, train_backbone
            )
            if not valid.any():
                continue

            loss = criterion(predictions, targets, valid)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

    return total_loss / max(n_batches, 1)


def eval_epoch(model, loader, criterion, device, scaler=None, ts_mode="real", teacher_forcing=False, epoch_idx=0):
    model.eval()
    state_store = VideoStateStore()
    total_loss, n_batches = 0.0, 0

    with torch.no_grad():

        pbar = tqdm(loader, desc=f"Eval Epoch {epoch_idx}", leave=False)

        for batch in pbar:
            with torch.amp.autocast('cuda', enabled=scaler is not None and scaler.is_enabled()):
                predictions, targets, valid = run_window(
                    model, batch, state_store, device, ts_mode, teacher_forcing, train_backbone=False
                )
                if not valid.any():
                    continue
                loss = criterion(predictions, targets, valid)
                
            total_loss += loss.item()
            n_batches += 1

            pbar.set_postfix({"batch_loss": f"{loss.item():.4f}"})

    return total_loss / max(n_batches, 1)


def fine_tune(
    model,
    train_loader,
    val_loader,
    device: str = "cuda",
    epochs: int = 100,
    backbone_lr: float = 1e-6,
    head_lr: float = 1e-4,
    patience: int = 20,
    grad_clip: float = 1.0,
    ts_mode: str = "real",
    teacher_forcing: bool = False,
    train_backbone: bool = False,
    unfreeze_backbone_after_epoch: Optional[int] = None,
    pretrained_checkpoint: Optional[str] = None,
    checkpoint: str = "DIEM_model.pt",
    use_amp: bool = True,
):
    if pretrained_checkpoint is not None:
        state_dict = torch.load(pretrained_checkpoint, map_location=device)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained checkpoint {pretrained_checkpoint}")
        if missing:
            print(f"  [WARN] missing keys (left at current init): {missing}")
        if unexpected:
            print(f"  [WARN] unexpected keys (ignored): {unexpected}")

    model = model.to(device)
    criterion = ContinuousKLLoss()
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp and device == "cuda")

    backbone_params = list(model.feature_extractor.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith("feature_extractor.")]
    optimizer = optim.Adam([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": other_params, "lr": head_lr},
    ], fused=(device == "cuda"))

    effective_train_backbone = train_backbone if unfreeze_backbone_after_epoch is None else False
    for p in backbone_params:
        p.requires_grad = effective_train_backbone
    print(f"Backbone trainable: {effective_train_backbone}"
          + (f" (unfreezing after epoch {unfreeze_backbone_after_epoch})" if unfreeze_backbone_after_epoch else ""))

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    print(f"Fine-tuning on {device} | epochs={epochs} | backbone_lr={backbone_lr} "
          f"| head_lr={head_lr} | patience={patience} | ts_mode={ts_mode} "
          f"| teacher_forcing={teacher_forcing}\n")

    for epoch in range(1, epochs + 1):
        if unfreeze_backbone_after_epoch is not None and epoch == unfreeze_backbone_after_epoch + 1:
            effective_train_backbone = train_backbone
            for p in backbone_params:
                p.requires_grad = effective_train_backbone
            print(f"[epoch {epoch}] backbone trainable set to {effective_train_backbone}")

        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, scaler,
                                  ts_mode, teacher_forcing, effective_train_backbone, grad_clip, epoch)
        val_loss = eval_epoch(model, val_loader, criterion, device, scaler, ts_mode, teacher_forcing, epoch)
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch:03d}/{epochs} | train={train_loss:.4f} | val={val_loss:.4f} | {dt:.1f}s", end="")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint)
            print("  <- saved best model", end="")
        else:
            epochs_no_improve += 1
            print(f"  (no improvement for {epochs_no_improve}/{patience} epochs)", end="")
        print()

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping triggered at epoch {epoch}. Best val loss: {best_val_loss:.4f}")
            break

    print(f"\nFine-tuning complete. Best model saved to: {checkpoint}")
    return history


if __name__ == "__main__":
    from modeldata import GazeLLNArch  # your actual model file -- keep
    # modeldata.py in this same folder, nothing to copy/paste
 
    # -----------------------------------------------------------------------
    # Edit these directly to configure a run -- no command-line flags.
    # -----------------------------------------------------------------------
    RUN_CONFIG = dict(
        pretrained_checkpoint="OSIE.pt",   # your OSIE-trained checkpoint to fine-tune from; None to train from scratch
        checkpoint="DIEM_model.pt",  # where the best fine-tuned checkpoint gets saved
        epochs=100,
        backbone_lr=1e-6,
        head_lr=1e-4,
        patience=20,
 
        # False (default) matches how OSIE.pt was actually trained --
        # the backbone stayed frozen for its entire OSIE run. Set True to
        # let it adapt to video frames instead.
        train_backbone=False,
        # If set (e.g. 5), starts with the backbone frozen regardless of
        # train_backbone, then switches to train_backbone's value after this
        # many epochs. None = no schedule, just use train_backbone as-is
        # for the whole run.
        unfreeze_backbone_after_epoch=None,
 
        # False (default) matches trainGazeLNN.ipynb's free-running
        # autoregressive training (prev_hmap = model's own last prediction).
        # True uses ground-truth teacher forcing instead -- a legitimate
        # alternative, just not what OSIE.pt was trained with.
        teacher_forcing=False,
 
        # "real" uses actual per-frame delta_t (1/fps) -- matches
        # trainGazeLNN.ipynb's current dt_seq usage. "constant_one" instead
        # fixes ts=1.0 for every step.
        ts_mode="real",
 
        videos_per_batch=8,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_amp=True,
        num_workers=4,
    )
    # -----------------------------------------------------------------------
 
    if RUN_CONFIG["device"] == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    train_ds, val_ds, test_ds = build_datasets()
 
    train_loader = DataLoader(
        train_ds, 
        batch_sampler=VideoOrderedBatchSampler(train_ds, videos_per_batch=RUN_CONFIG["videos_per_batch"]),
        num_workers=RUN_CONFIG["num_workers"],
        pin_memory=(RUN_CONFIG["device"] == "cuda")
    )
    val_loader = DataLoader(
        val_ds, 
        batch_sampler=VideoOrderedBatchSampler(val_ds, videos_per_batch=RUN_CONFIG["videos_per_batch"], shuffle_videos=False),
        num_workers=RUN_CONFIG["num_workers"],
        pin_memory=(RUN_CONFIG["device"] == "cuda")
    )
 
    model = GazeLLNArch(image_h=cfg.MODEL_INPUT_HEIGHT, image_w=cfg.MODEL_INPUT_WIDTH, S=cfg.S)
 
    history = fine_tune(
        model, train_loader, val_loader,
        device=RUN_CONFIG["device"],
        epochs=RUN_CONFIG["epochs"],
        backbone_lr=RUN_CONFIG["backbone_lr"],
        head_lr=RUN_CONFIG["head_lr"],
        patience=RUN_CONFIG["patience"],
        ts_mode=RUN_CONFIG["ts_mode"],
        teacher_forcing=RUN_CONFIG["teacher_forcing"],
        train_backbone=RUN_CONFIG["train_backbone"],
        unfreeze_backbone_after_epoch=RUN_CONFIG["unfreeze_backbone_after_epoch"],
        pretrained_checkpoint=RUN_CONFIG["pretrained_checkpoint"],
        checkpoint=RUN_CONFIG["checkpoint"],
        use_amp=RUN_CONFIG["use_amp"],
    )
 