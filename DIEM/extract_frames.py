"""
Extract and resize video frames to MODEL_INPUT_WIDTH x MODEL_INPUT_HEIGHT via
ffmpeg, indexed to match the DIEM gaze files' 1-indexed frame numbers.

Confirmed against the 50_people_brooklyn_no_voices_1280x720 sample:
- native video is exactly 1280x720 @ 30/1 fps (matches GAZE_FPS and
  DEFAULT_SCREEN_RESOLUTION), duration ~122.3s
- video frame count (ffprobe nb_frames) and gaze-file row count differ by
  only ~2 frames at the tail, which the dataset/heatmap code already
  tolerates (short sequences get zero-padded, see dataset.py)

Requires ffmpeg on PATH.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import config as cfg


def extract_frames_for_video(video_dir: Path, frames_root: Path) -> None:
    video_name = video_dir.name
    matches = sorted(video_dir.glob(cfg.VIDEO_FILE_GLOB))
    if not matches:
        print(f"  [WARN] no video file found in {video_dir} "
              f"(pattern={cfg.VIDEO_FILE_GLOB}) -- check config.VIDEO_FILE_GLOB")
        return
    video_path = matches[0]

    out_dir = frames_root / video_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # -start_number 1 matches DIEM's 1-indexed gaze-file frame numbering.
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"scale={cfg.MODEL_INPUT_WIDTH}:{cfg.MODEL_INPUT_HEIGHT}",
        "-start_number", "1",
        str(out_dir / "frame_%06d.png"),
    ]
    print(f"  running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] ffmpeg failed for {video_path}:\n{result.stderr[-2000:]}")
    else:
        n_frames = len(list(out_dir.glob("frame_*.png")))
        print(f"  extracted {n_frames} frames -> {out_dir}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default=None,
                     help="comma-separated video names to restrict to "
                          "(default: all videos under DIEM_ROOT)")
    ap.add_argument("--skip-existing", action="store_true",
                     help="skip a video if its frames dir already has PNGs -- "
                          "recommended for full-dataset runs, since this is "
                          "the slowest and most disk-heavy step (one PNG per "
                          "frame per video; a ~90s clip at 30fps is ~2700 "
                          "files). Re-running the whole dataset from scratch "
                          "every time you add a new video is wasteful.")
    args = ap.parse_args()

    frames_root = cfg.OUTPUT_ROOT / "frames"
    video_dirs = sorted(d for d in cfg.DIEM_ROOT.iterdir() if d.is_dir())
    if args.videos:
        wanted = set(args.videos.split(","))
        video_dirs = [d for d in video_dirs if d.name in wanted]

    for vd in video_dirs:
        out_dir = frames_root / vd.name
        if args.skip_existing and out_dir.exists() and any(out_dir.glob("frame_*.png")):
            print(f"Skipping {vd.name} (frames already extracted, --skip-existing)")
            continue
        print(f"Extracting frames for {vd.name} ...")
        extract_frames_for_video(vd, frames_root)


if __name__ == "__main__":
    main()
