"""
Step 1: parse raw per-subject DIEM gaze files into discrete fixation events.

DIEM already provides an event flag per sample (fixation / saccade / blink /
dropped), so "fixation detection" here just means grouping contiguous runs of
event==FIXATION_FLAG samples and collapsing each run into a single fixation
with a centroid (x, y) and a duration.

Output: one JSON-serializable list of fixations per subject per video:
    [{"start_frame": int, "end_frame": int, "duration_s": float,
      "x": float, "y": float}, ...]
in SCREEN coordinate space (not yet resized to model input resolution --
that happens later, once we know the per-video screen resolution).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

import numpy as np

import config as cfg


@dataclass
class Fixation:
    start_frame: int
    end_frame: int
    duration_s: float
    x: float
    y: float


def _load_raw_gaze_file(path: Path) -> np.ndarray:
    """Load a single subject's raw gaze file into an (N, 9) array.

    Robust to whitespace- or comma-delimited files and the occasional
    header line some DIEM releases include.
    """
    rows = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 9:
                continue  # skip malformed/header lines
            try:
                rows.append([float(p) for p in parts[:9]])
            except ValueError:
                continue  # header line, skip
    if not rows:
        raise ValueError(f"No parsable rows found in {path}")
    return np.array(rows, dtype=np.float64)


def _eye_position(row_block: np.ndarray) -> np.ndarray:
    """Average left/right eye coordinates per sample, falling back to
    whichever eye is valid if one is dropped (flag == DROPPED_FLAG).

    row_block: (N, 9) raw rows.
    returns: (N, 2) array of (x, y) in screen coordinates, NaN where both
    eyes are invalid.
    """
    left_xy = row_block[:, [cfg.COL_LEFT_X, cfg.COL_LEFT_Y]]
    right_xy = row_block[:, [cfg.COL_RIGHT_X, cfg.COL_RIGHT_Y]]
    left_valid = row_block[:, cfg.COL_LEFT_EVENT] != cfg.DROPPED_FLAG
    right_valid = row_block[:, cfg.COL_RIGHT_EVENT] != cfg.DROPPED_FLAG

    xy = np.full((row_block.shape[0], 2), np.nan)

    both = left_valid & right_valid
    xy[both] = (left_xy[both] + right_xy[both]) / 2.0

    only_left = left_valid & ~right_valid
    xy[only_left] = left_xy[only_left]

    only_right = right_valid & ~left_valid
    xy[only_right] = right_xy[only_right]

    return xy


def extract_fixations(raw: np.ndarray) -> List[Fixation]:
    """Group contiguous fixation-flagged samples into Fixation events.

    We require BOTH eyes to agree the sample is a fixation is too strict in
    practice (tracking noise), so we treat a sample as "fixation" if either
    eye's event flag says so and that eye's coordinates are valid. Adjust
    this rule if your data quality differs.
    """
    frames = raw[:, cfg.COL_FRAME].astype(int)
    xy = _eye_position(raw)

    is_fixation = (
        (raw[:, cfg.COL_LEFT_EVENT] == cfg.FIXATION_FLAG)
        | (raw[:, cfg.COL_RIGHT_EVENT] == cfg.FIXATION_FLAG)
    ) & ~np.isnan(xy[:, 0])

    fixations: List[Fixation] = []
    n = len(raw)
    i = 0
    while i < n:
        if not is_fixation[i]:
            i += 1
            continue
        j = i
        while j < n and is_fixation[j] and frames[j] == frames[i] + (j - i):
            j += 1
        # contiguous run [i, j)
        run_xy = xy[i:j]
        centroid = np.nanmean(run_xy, axis=0)
        start_f, end_f = int(frames[i]), int(frames[j - 1])
        duration_s = (end_f - start_f + 1) / cfg.GAZE_FPS
        fixations.append(
            Fixation(
                start_frame=start_f,
                end_frame=end_f,
                duration_s=duration_s,
                x=float(centroid[0]),
                y=float(centroid[1]),
            )
        )
        i = j
    return fixations


def process_video_folder(video_dir: Path, output_dir: Path) -> None:
    """Parse all subject gaze files for one DIEM video folder and write
    one fixations JSON per subject to output_dir/<video_name>/<subject>.json
    """
    video_name = video_dir.name
    subject_files = sorted(video_dir.glob(cfg.GAZE_FILE_GLOB))
    if not subject_files:
        print(f"  [WARN] no gaze files found in {video_dir} "
              f"(pattern={cfg.GAZE_FILE_GLOB}) -- check config.GAZE_FILE_GLOB")
        return

    out_dir = output_dir / video_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for sf in subject_files:
        try:
            raw = _load_raw_gaze_file(sf)
        except ValueError as e:
            print(f"  [WARN] skipping {sf}: {e}")
            continue
        fixations = extract_fixations(raw)
        out_path = out_dir / f"{sf.stem}.json"
        with open(out_path, "w") as f:
            json.dump([asdict(fx) for fx in fixations], f)
        print(f"  {sf.name}: {len(fixations)} fixations -> {out_path}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=None,
                     help="comma-separated video folder names to restrict to "
                          "(default: all videos under DIEM_ROOT)")
    ap.add_argument("--skip-existing", action="store_true",
                     help="skip a video if its fixations output dir already "
                          "has JSON files (useful for resuming/incremental runs "
                          "over the full dataset without redoing finished videos)")
    args = ap.parse_args()

    if not cfg.DIEM_ROOT.exists():
        raise SystemExit(
            f"DIEM_ROOT {cfg.DIEM_ROOT} does not exist -- set it in config.py "
            f"to point at your extracted DIEM download."
        )
    video_dirs = sorted(d for d in cfg.DIEM_ROOT.iterdir() if d.is_dir())
    if args.videos:
        wanted = set(args.videos.split(","))
        video_dirs = [d for d in video_dirs if d.name in wanted]
    print(f"Found {len(video_dirs)} video folders to process under {cfg.DIEM_ROOT}")

    fixations_root = cfg.OUTPUT_ROOT / "fixations"
    for vd in video_dirs:
        if args.skip_existing and (fixations_root / vd.name).exists() \
                and any((fixations_root / vd.name).glob("*.json")):
            print(f"Skipping {vd.name} (already processed, --skip-existing)")
            continue
        print(f"Processing {vd.name} ...")
        process_video_folder(vd, fixations_root)


if __name__ == "__main__":
    main()
