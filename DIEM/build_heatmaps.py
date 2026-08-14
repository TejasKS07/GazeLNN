"""
Step 2: turn parsed per-subject fixations into per-frame ground-truth
fixation heatmaps, matching GazeLLNArch's actual heatmap-grid convention
(HMAP_HEIGHT x HMAP_WIDTH = image_size // S, sigma in grid pixels -- see
config.py), aggregated across all subjects who watched that video.

Two-pass per video:
  1. Pool every subject's fixation (x, y) for the video and fit a
     screen-coords -> video-pixel-coords affine calibration (see
     config.CALIBRATION_MODE) -- needed because DIEM's raw gaze coordinates
     are NOT guaranteed to be in the same coordinate space as a given
     video's native resolution (confirmed empirically; see config.py).
  2. Re-walk each subject's fixations, apply the calibration, project
     directly into heatmap-grid coordinates, splat+blur+normalize per frame.

Output per video: a compressed .npz with
    heatmaps: (num_frames, HMAP_HEIGHT, HMAP_WIDTH) float32, each valid
              frame's heatmap sums to 1
    valid: (num_frames,) bool mask (True where >= MIN_SUBJECTS_PER_FRAME
           contributed)
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

import config as cfg


# ---------------------------------------------------------------------------
# Native video resolution detection
# ---------------------------------------------------------------------------

_RES_SUFFIX_RE = re.compile(r"(\d{3,5})x(\d{3,5})")


def get_native_resolution(video_name: str, diem_root: Path) -> Tuple[int, int]:
    """Return (width, height) for a video, via ffprobe on the actual file,
    falling back to parsing the folder name's trailing _WxH suffix (DIEM
    folder names consistently end this way in both samples checked so far,
    e.g. ..._1280x720, ..._720x576) if ffprobe is unavailable or fails."""
    video_dir = diem_root / video_name
    matches = sorted(video_dir.glob(cfg.VIDEO_FILE_GLOB))
    if matches:
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height",
                 "-of", "csv=p=0", str(matches[0])],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
            w, h = out.split(",")[:2]
            return int(w), int(h)
        except Exception as e:
            print(f"  [WARN] ffprobe failed for {video_name} ({e}), "
                  f"falling back to folder-name suffix")

    m = _RES_SUFFIX_RE.search(video_name)
    if m:
        return int(m.group(1)), int(m.group(2))

    raise ValueError(
        f"Could not determine native resolution for {video_name} -- neither "
        f"ffprobe nor folder-name suffix worked. Add an entry to "
        f"config.MANUAL_CALIBRATION or fix the video file."
    )


# ---------------------------------------------------------------------------
# Screen -> video-pixel calibration
# ---------------------------------------------------------------------------

def _pool_all_fixation_xy(subjects: Dict[str, List[dict]]) -> np.ndarray:
    xs, ys = [], []
    for fxs in subjects.values():
        for fx in fxs:
            xs.append(fx["x"])
            ys.append(fx["y"])
    return np.array(xs), np.array(ys)


def fit_calibration(video_name: str, subjects: Dict[str, List[dict]],
                     native_w: int, native_h: int) -> dict:
    """Return dict(scale_x, scale_y, offset_x, offset_y) mapping raw gaze
    (x, y) -> video-pixel (x, y): video_x = raw_x * scale_x + offset_x."""
    mode = cfg.CALIBRATION_MODE_OVERRIDES.get(video_name, cfg.CALIBRATION_MODE)

    if mode == "manual":
        if video_name not in cfg.MANUAL_CALIBRATION:
            raise ValueError(
                f"CALIBRATION_MODE='manual' but no entry for {video_name} in "
                f"config.MANUAL_CALIBRATION"
            )
        return cfg.MANUAL_CALIBRATION[video_name]

    if mode == "identity":
        return dict(scale_x=1.0, scale_y=1.0, offset_x=0.0, offset_y=0.0)

    if mode == "auto_recenter":
        # Translation only -- shifts the pooled fixation median to the frame
        # center, without touching scale/spread. See config.py
        # CALIBRATION_MODE_OVERRIDES for why this is preferred over
        # "auto_percentile" for videos with evidence of a pure offset bug.
        xs, ys = _pool_all_fixation_xy(subjects)
        if len(xs) < 50:
            print(f"  [WARN] only {len(xs)} pooled fixation samples for "
                  f"{video_name} -- recenter calibration may be unstable")
        med_x, med_y = np.median(xs), np.median(ys)
        offset_x = native_w / 2.0 - med_x
        offset_y = native_h / 2.0 - med_y
        return dict(scale_x=1.0, scale_y=1.0, offset_x=offset_x, offset_y=offset_y)

    if mode == "auto_percentile":
        xs, ys = _pool_all_fixation_xy(subjects)
        if len(xs) < 50:
            print(f"  [WARN] only {len(xs)} pooled fixation samples for "
                  f"{video_name} -- percentile calibration may be unstable, "
                  f"consider CALIBRATION_MODE='identity' for this video")
        lo_x, hi_x = np.percentile(xs, [cfg.CALIB_LOW_PCT, cfg.CALIB_HIGH_PCT])
        lo_y, hi_y = np.percentile(ys, [cfg.CALIB_LOW_PCT, cfg.CALIB_HIGH_PCT])
        span_x, span_y = max(hi_x - lo_x, 1e-6), max(hi_y - lo_y, 1e-6)
        scale_x = native_w / span_x
        scale_y = native_h / span_y
        offset_x = -lo_x * scale_x
        offset_y = -lo_y * scale_y
        return dict(scale_x=scale_x, scale_y=scale_y,
                     offset_x=offset_x, offset_y=offset_y)

    raise ValueError(f"Unknown calibration mode: {mode}")


def _load_subject_fixations(video_out_dir: Path) -> Dict[str, List[dict]]:
    subjects = {}
    for p in sorted(video_out_dir.glob("*.json")):
        with open(p) as f:
            subjects[p.stem] = json.load(f)
    return subjects


# ---------------------------------------------------------------------------
# Heatmap building (direct to HMAP_HEIGHT x HMAP_WIDTH grid, matching
# GazeLLNArch's own _gaussian_heatmap convention: coords rescaled straight
# into the downsampled grid, single-pixel spike, then gaussian_filter with
# sigma in GRID pixels)
# ---------------------------------------------------------------------------

def build_video_heatmaps(video_name: str, fixations_root: Path,
                          diem_root: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    video_out_dir = fixations_root / video_name
    subjects = _load_subject_fixations(video_out_dir)
    if not subjects:
        raise ValueError(f"No parsed fixation files found for {video_name} in {video_out_dir}")

    native_w, native_h = get_native_resolution(video_name, diem_root)
    calib = fit_calibration(video_name, subjects, native_w, native_h)

    max_frame = max(
        (fx["end_frame"] for fxs in subjects.values() for fx in fxs), default=0
    )
    num_frames = max_frame + 1  # frames are 1-indexed in DIEM; keep index 0 unused/empty

    Hg, Wg = cfg.HMAP_HEIGHT, cfg.HMAP_WIDTH
    heatmaps = np.zeros((num_frames, Hg, Wg), dtype=np.float32)
    subject_counts = np.zeros(num_frames, dtype=np.int32)

    n_dropped, n_total = 0, 0
    for subj, fxs in subjects.items():
        for fx in fxs:
            # raw gaze coords -> calibrated video-pixel coords
            vx = fx["x"] * calib["scale_x"] + calib["offset_x"]
            vy = fx["y"] * calib["scale_y"] + calib["offset_y"]
            n_total += 1
            if not (0 <= vx < native_w and 0 <= vy < native_h):
                n_dropped += 1
                continue  # off-frame even after calibration -- treat as noise
            # video-pixel coords -> heatmap-grid coords (matches notebook's
            # _gaussian_heatmap: xi = round(x * hmap_w / img_w))
            xi = int(round(vx * Wg / native_w))
            yi = int(round(vy * Hg / native_h))
            xi = min(max(xi, 0), Wg - 1)
            yi = min(max(yi, 0), Hg - 1)
            for frame_idx in range(fx["start_frame"], fx["end_frame"] + 1):
                if frame_idx >= num_frames:
                    continue
                heatmaps[frame_idx, yi, xi] += 1.0
                subject_counts[frame_idx] += 1

    for i in range(num_frames):
        if subject_counts[i] == 0:
            continue
        heatmaps[i] = gaussian_filter(heatmaps[i], sigma=cfg.HEATMAP_SIGMA)
        s = heatmaps[i].sum()
        if s > 0:
            heatmaps[i] /= s

    valid = subject_counts >= cfg.MIN_SUBJECTS_PER_FRAME
    drop_frac = n_dropped / max(n_total, 1)
    if drop_frac > 0.15:
        print(f"  [WARN] {drop_frac:.0%} of fixation samples fell outside the "
              f"video frame even after calibration -- inspect this video's "
              f"calibration manually (e.g. via the overlay debug helper) "
              f"before trusting its heatmaps")

    calib_info = dict(mode=cfg.CALIBRATION_MODE_OVERRIDES.get(video_name, cfg.CALIBRATION_MODE),
                       native_w=native_w, native_h=native_h,
                       drop_fraction=drop_frac, **calib)
    return heatmaps, valid, calib_info


def main():
    import argparse
    import csv

    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=None,
                     help="comma-separated video names to restrict to "
                          "(default: all videos with parsed fixations)")
    ap.add_argument("--skip-existing", action="store_true",
                     help="skip a video if its heatmap .npz already exists "
                          "(useful for resuming full-dataset runs; NOTE: if "
                          "you're re-running just to apply a new "
                          "CALIBRATION_MODE_OVERRIDES entry, pass --videos "
                          "for that specific video instead of --skip-existing, "
                          "or it'll be skipped without picking up the change)")
    args = ap.parse_args()

    fixations_root = cfg.OUTPUT_ROOT / "fixations"
    heatmaps_root = cfg.OUTPUT_ROOT / "heatmaps"
    heatmaps_root.mkdir(parents=True, exist_ok=True)

    video_dirs = sorted(d for d in fixations_root.iterdir() if d.is_dir())
    if args.videos:
        wanted = set(args.videos.split(","))
        video_dirs = [d for d in video_dirs if d.name in wanted]

    report_path = cfg.OUTPUT_ROOT / "calibration_report.csv"
    existing_rows = {}
    if report_path.exists():
        with open(report_path) as f:
            for row in csv.DictReader(f):
                existing_rows[row["video"]] = row

    for vd in video_dirs:
        out_path = heatmaps_root / f"{vd.name}.npz"
        if args.skip_existing and out_path.exists():
            print(f"Skipping {vd.name} (heatmap already exists, --skip-existing)")
            continue
        print(f"Building heatmaps for {vd.name} ...")
        heatmaps, valid, calib_info = build_video_heatmaps(vd.name, fixations_root, cfg.DIEM_ROOT)
        np.savez_compressed(out_path, heatmaps=heatmaps, valid=valid)
        print(f"  calibration: {calib_info}")
        print(f"  {valid.sum()}/{len(valid)} frames valid -> {out_path}")
        existing_rows[vd.name] = dict(
            video=vd.name,
            mode=calib_info["mode"],
            native_w=calib_info["native_w"],
            native_h=calib_info["native_h"],
            drop_fraction=f"{calib_info['drop_fraction']:.4f}",
            needs_review="yes" if calib_info["drop_fraction"] > 0.15 else "no",
        )

    # Persist a sortable summary so you can triage across many videos without
    # scrolling console output, e.g.:
    #   sort -t, -k5 -rn calibration_report.csv | head   # worst drop_fraction first
    #   awk -F, '$6=="yes"' calibration_report.csv        # only flagged videos
    fieldnames = ["video", "mode", "native_w", "native_h", "drop_fraction", "needs_review"]
    with open(report_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(existing_rows.values(), key=lambda r: r["video"]):
            writer.writerow(row)
    n_flagged = sum(1 for r in existing_rows.values() if r["needs_review"] == "yes")
    print(f"\nWrote {report_path} ({len(existing_rows)} videos, {n_flagged} flagged for review)")


if __name__ == "__main__":
    main()
