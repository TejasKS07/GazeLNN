"""
Step 3: PyTorch Dataset serving rolling windows of (frame, heatmap, valid,
delta_t) tuples for continuous (Option B) fine-tuning of GazeLNN on DIEM.

Design notes (see chat discussion this implements):
- Splitting is done at the VIDEO level, never at the frame level, to avoid
  leaking correlated content/gaze between train/val/test.
- Each __getitem__ returns one WINDOW_LEN_FRAMES-long window from one video.
  The CfC hidden state should be carried forward across windows from the
  same video within a training epoch and detached at window boundaries
  (truncated BPTT) -- that stitching happens in your training loop, not here;
  this Dataset just needs to hand you windows in-order per video, which is
  why the sampler groups windows by video and preserves order (see
  `VideoOrderedBatchSampler` below) rather than shuffling arbitrary windows
  across videos in a single batch.
- delta_t per step is 1/GAZE_FPS (real elapsed time between frames), NOT the
  fixation-duration delta_t the original GazeLNN paper uses for its
  within-image scanpath steps. This is a deliberate change in what delta_t
  means, discussed in the fine-tuning strategy.
- Frames flagged invalid (< MIN_SUBJECTS_PER_FRAME valid subjects) should be
  masked out of the loss -- the `valid` mask is returned per-frame for this.
- `heatmaps` here are (T, cfg.HMAP_HEIGHT, cfg.HMAP_WIDTH) = (T, 32, 48) by
  default, matching GazeLLNArch's own downsampled heatmap grid (image_size
  // S) exactly -- built in build_heatmaps.py using the same coordinate
  rescaling + sigma convention as the existing OSIE preprocessing, so this
  plugs directly into forward() without any resolution mismatch.
- GazeLLNArch.forward(image, prev_hmap, hx, ts) takes the PREVIOUS step's
  heatmap as an input (teacher forcing during training). This Dataset gives
  you the target heatmap sequence; derive prev_hmap in your training loop by
  shifting it one step and using a zero (or uniform) heatmap for the first
  step of each video -- see `get_prev_hmaps()` below.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

import config as cfg

try:
    import torch
    from torch.utils.data import Dataset, Sampler
except ImportError:  # allow inspecting/testing this module without torch installed
    torch = None
    Dataset = object
    Sampler = object


@dataclass
class VideoEntry:
    name: str
    frames_dir: Path          # directory of extracted+resized video frames (see note below)
    heatmap_npz: Path
    num_frames: int


def _discover_videos(processed_root: Path, frames_root: Path) -> List[VideoEntry]:
    """
    Assumes you've already extracted and resized video frames to
    frames_root/<video_name>/frame_{i:06d}.png (e.g. via ffmpeg), at
    cfg.MODEL_INPUT_WIDTH x cfg.MODEL_INPUT_HEIGHT, matching the heatmap
    resolution built in build_heatmaps.py. Frame extraction itself is a
    one-line ffmpeg call per video and is not included here since it depends
    on which video codec/container your DIEM download uses -- e.g.:

        ffmpeg -i video.mp4 -vf scale=384:256 frames/frame_%06d.png

    Make sure the ffmpeg frame indexing matches the 1-indexed frame numbers
    in the gaze files (ffmpeg's %06d starts at 1 by default, which lines up).
    """
    heatmaps_dir = processed_root / "heatmaps"
    entries = []
    for npz_path in sorted(heatmaps_dir.glob("*.npz")):
        video_name = npz_path.stem
        frames_dir = frames_root / video_name
        if not frames_dir.exists():
            print(f"[WARN] no extracted frames for {video_name} at {frames_dir}, skipping")
            continue
        with np.load(npz_path) as d:
            num_frames = d["valid"].shape[0]
        entries.append(VideoEntry(video_name, frames_dir, npz_path, num_frames))
    return entries


def split_videos(entries: List[VideoEntry], val_frac=0.15, test_frac=0.15, seed=0):
    """Video-level train/val/test split -- never split a single video's
    frames across sets, since adjacent/nearby frames of the same video are
    highly correlated in content and gaze behavior."""
    rng = np.random.RandomState(seed)
    idx = np.arange(len(entries))
    rng.shuffle(idx)
    n = len(entries)
    n_val = max(1, int(n * val_frac))
    n_test = max(1, int(n * test_frac))
    val_idx = set(idx[:n_val].tolist())
    test_idx = set(idx[n_val:n_val + n_test].tolist())
    train, val, test = [], [], []
    for i, e in enumerate(entries):
        if i in val_idx:
            val.append(e)
        elif i in test_idx:
            test.append(e)
        else:
            train.append(e)
    return train, val, test


class DIEMWindowDataset(Dataset):
    """One item = one rolling window from one video: frames, heatmaps, valid
    mask, and delta_t (constant = 1/fps here, kept as a tensor for API
    parity with the fixation-duration delta_t used in the original paper)."""

    def __init__(self, entries: List[VideoEntry],
                 window_len: int = cfg.WINDOW_LEN_FRAMES,
                 stride: int = cfg.WINDOW_STRIDE_FRAMES,
                 transform=None):
        if torch is None:
            raise ImportError("PyTorch is required to use DIEMWindowDataset")
        self.entries = entries
        self.window_len = window_len
        self.stride = stride
        self.transform = transform
        self.index: List[Tuple[int, int]] = []  # (entry_idx, start_frame)
        for ei, e in enumerate(entries):
            last_start = max(0, e.num_frames - window_len)
            starts = list(range(0, last_start + 1, stride)) or [0]
            for s in starts:
                self.index.append((ei, s))

    def __len__(self):
        return len(self.index)

    def _load_frame(self, frames_dir: Path, frame_idx: int) -> np.ndarray:
        # 1-indexed to match DIEM gaze-file frame numbering / ffmpeg output
        path = frames_dir / f"frame_{frame_idx:06d}.png"
        if not path.exists():
            # pad with a black frame at sequence edges rather than crashing --
            # the corresponding `valid` entry should already be False here in
            # most cases since dropped/short videos also lack gaze coverage
            return np.zeros((cfg.MODEL_INPUT_HEIGHT, cfg.MODEL_INPUT_WIDTH, 3), dtype=np.uint8)
        import imageio.v2 as imageio
        img = imageio.imread(path)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        return img[..., :3]

    def __getitem__(self, idx: int):
        ei, start = self.index[idx]
        entry = self.entries[ei]
        end = min(start + self.window_len, entry.num_frames)

        with np.load(entry.heatmap_npz) as d:
            heatmaps = d["heatmaps"][start:end]  # (T, H, W)
            valid = d["valid"][start:end]        # (T,)

        frames = np.stack(
            [self._load_frame(entry.frames_dir, f) for f in range(start, end)], axis=0
        )  # (T, H, W, 3) uint8

        T = frames.shape[0]
        if T < self.window_len:
            pad_t = self.window_len - T
            frames = np.concatenate(
                [frames, np.zeros((pad_t, *frames.shape[1:]), dtype=frames.dtype)], axis=0
            )
            heatmaps = np.concatenate(
                [heatmaps, np.zeros((pad_t, *heatmaps.shape[1:]), dtype=heatmaps.dtype)], axis=0
            )
            valid = np.concatenate([valid, np.zeros(pad_t, dtype=bool)], axis=0)

        frames_t = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0  # (T,3,H,W)
        heatmaps_t = torch.from_numpy(heatmaps).float()                          # (T,H,W)
        valid_t = torch.from_numpy(valid.copy())                                 # (T,)
        delta_t = torch.full((T,), 1.0 / cfg.GAZE_FPS, dtype=torch.float32)

        if self.transform is not None:
            frames_t = self.transform(frames_t)

        # is_window_start flags the first window of a video, so the training
        # loop knows when to reset (rather than carry forward) the CfC hidden
        # state -- reset only here, detach-and-carry at every other window.
        is_video_start = torch.tensor(start == 0)

        return {
            "video_name": entry.name,
            "start_frame": start,
            "frames": frames_t,
            "heatmaps": heatmaps_t,
            "valid": valid_t,
            "delta_t": delta_t,
            "is_video_start": is_video_start,
        }


def get_prev_hmaps(heatmaps: "torch.Tensor", is_video_start: "torch.Tensor") -> "torch.Tensor":
    """Derive the `prev_hmap` input GazeLLNArch.forward() expects, from a
    batch's target heatmap sequence, via teacher forcing (shift by one step).

    Args:
        heatmaps: (B, T, Hg, Wg) ground-truth heatmap sequence for the batch
                  (as returned by DIEMWindowDataset, stacked over the batch
                  dim by your DataLoader).
        is_video_start: (B,) bool -- True where this window is the first
                  window of its video (i.e. where the CfC hidden state gets
                  reset rather than carried over from the previous window).

    Returns:
        prev_hmap: (B, T, Hg, Wg) -- prev_hmap[:, t] = heatmaps[:, t-1] for
                  t>0. For t=0: uses a uniform distribution (maximum-
                  uncertainty prior) if is_video_start is True for that
                  batch element, since there's no real previous fixation to
                  condition on; otherwise (t=0 of a later window in the same
                  video) it's on you to carry the true previous heatmap over
                  from the last step of the prior window in your training
                  loop -- this function only fills in the video-start case,
                  since it doesn't have access to that cross-window state.
    """
    if torch is None:
        raise ImportError("PyTorch is required to use get_prev_hmaps")
    B, T, Hg, Wg = heatmaps.shape
    prev = torch.zeros_like(heatmaps)
    prev[:, 1:] = heatmaps[:, :-1]
    uniform = torch.full((Hg, Wg), 1.0 / (Hg * Wg), dtype=heatmaps.dtype, device=heatmaps.device)
    for b in range(B):
        if bool(is_video_start[b]):
            prev[b, 0] = uniform
    return prev


class VideoOrderedBatchSampler(Sampler):
    """Yields window indices grouped and ordered by video, so consecutive
    batches for a given "lane" in your training loop correspond to
    consecutive windows of the same video -- required for correctly carrying
    the CfC hidden state across windows via truncated BPTT."""

    def __init__(self, dataset: DIEMWindowDataset, videos_per_batch: int = 4, shuffle_videos: bool = True, seed: int = 0):
        self.dataset = dataset
        self.videos_per_batch = videos_per_batch
        self.shuffle_videos = shuffle_videos
        self.seed = seed

        by_video: dict[int, List[int]] = {}
        for i, (ei, start) in enumerate(dataset.index):
            by_video.setdefault(ei, []).append(i)
        for ei in by_video:
            by_video[ei].sort(key=lambda i: dataset.index[i][1])  # order by start_frame
        self.by_video = by_video

    def __iter__(self):
        video_ids = list(self.by_video.keys())
        if self.shuffle_videos:
            rng = np.random.RandomState(self.seed)
            rng.shuffle(video_ids)

        for batch_start in range(0, len(video_ids), self.videos_per_batch):
            group = video_ids[batch_start:batch_start + self.videos_per_batch]
            max_len = max(len(self.by_video[v]) for v in group)
            for step in range(max_len):
                batch = []
                for v in group:
                    seq = self.by_video[v]
                    if step < len(seq):
                        batch.append(seq[step])
                if batch:
                    yield batch

    def __len__(self):
        video_ids = list(self.by_video.keys())
        total = 0
        for batch_start in range(0, len(video_ids), self.videos_per_batch):
            group = video_ids[batch_start:batch_start + self.videos_per_batch]
            total += max(len(self.by_video[v]) for v in group)
        return total


def build_datasets(processed_root: Path = None, frames_root: Path = None):
    processed_root = processed_root or cfg.OUTPUT_ROOT
    frames_root = frames_root or (cfg.OUTPUT_ROOT / "frames")
    entries = _discover_videos(processed_root, frames_root)
    if not entries:
        raise RuntimeError(
            "No videos discovered -- check that build_heatmaps.py has been run "
            "and that video frames have been extracted to OUTPUT_ROOT/frames/<video_name>/"
        )
    train_e, val_e, test_e = split_videos(entries)
    print(f"videos: train={len(train_e)} val={len(val_e)} test={len(test_e)}")
    return (
        DIEMWindowDataset(train_e),
        DIEMWindowDataset(val_e),
        DIEMWindowDataset(test_e),
    )
