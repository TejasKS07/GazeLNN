"""
Debug tool: overlay raw fixation points on actual extracted video frames, so
you can visually judge whether identity calibration (screen coords == video
pixel coords) looks right for a given video, BEFORE trusting any heatmaps
built from it.

Why this exists (read before using auto_percentile calibration):

Testing on the two real DIEM videos available so far showed the percentile-
based auto-calibration in build_heatmaps.py is NOT a safe default:

  - 50_people_brooklyn (1280x720): identity mapping already gives ~0.5% of
    fixation samples falling outside the frame -- essentially perfect.
    Auto-percentile calibration, applied anyway, INCREASES the drop rate to
    ~3.9%, because it forcibly stretches the (legitimately!) center-clustered
    gaze distribution to fill the frame edge-to-edge. People fixating on a
    face in the middle of frame is normal viewing behavior, not evidence of
    a coordinate offset -- percentile-stretching misreads this as one.

  - ami_ib4010_closeup (720x576): identity mapping gives ~27.6% of samples
    outside frame, and the median fixation sits almost at the frame's corner
    rather than near center. This COULD mean a genuine coordinate-space
    offset (e.g. video letterboxed within a larger recording display) -- or
    it could just mean the camera framed its subject off-center (plausible
    for AMI meeting-room corpus footage, which is often a fixed corner-
    mounted camera). Aggregate statistics alone can't distinguish these two
    explanations.

Conclusion: there is no single automatic rule that's safe for every DIEM
video. Use this script to actually look at a few frames with fixations
overlaid, per video, and decide:
  - Points landing sensibly on faces/salient content -> identity is fine,
    use CALIBRATION_MODE="identity".
  - Points landing on scene edges or blank margins in a way that doesn't
    match where a viewer would obviously look -> the video likely does need
    correction; consider CALIBRATION_MODE="manual" with a scale/offset you
    derive from this same visual check (nudge scale/offset until points
    align with plausible gaze targets), rather than trusting
    "auto_percentile" blindly.

Usage:
    python inspect_calibration.py <video_name> --frames 5
    python inspect_calibration.py --videos videoA,videoB,videoC --frames 5
(requires frames already extracted via extract_frames.py, and fixations
already parsed via parse_gaze.py, for each video given)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config as cfg
from build_heatmaps import get_native_resolution, fit_calibration, _load_subject_fixations


def active_fixations_at_frame(subjects: dict, frame_idx: int):
    pts = []
    for fxs in subjects.values():
        for fx in fxs:
            if fx["start_frame"] <= frame_idx <= fx["end_frame"]:
                pts.append((fx["x"], fx["y"]))
    return pts


def inspect_one_video(video_name: str, n_frames: int, mode_override: str = None):
    if mode_override:
        cfg.CALIBRATION_MODE = mode_override

    fixations_root = cfg.OUTPUT_ROOT / "fixations"
    frames_dir = cfg.OUTPUT_ROOT / "frames" / video_name
    subj_dir = fixations_root / video_name
    if not subj_dir.exists():
        print(f"[SKIP] {video_name}: no parsed fixations found at {subj_dir} "
              f"-- run parse_gaze.py --videos {video_name} first")
        return
    if not frames_dir.exists() or not any(frames_dir.glob("frame_*.png")):
        print(f"[SKIP] {video_name}: no extracted frames found at {frames_dir} "
              f"-- run extract_frames.py --videos {video_name} first")
        return

    subjects = _load_subject_fixations(subj_dir)
    native_w, native_h = get_native_resolution(video_name, cfg.DIEM_ROOT)
    calib = fit_calibration(video_name, subjects, native_w, native_h)
    resolved_mode = cfg.CALIBRATION_MODE_OVERRIDES.get(video_name, cfg.CALIBRATION_MODE)
    print(f"video={video_name} native={native_w}x{native_h} "
          f"calibration_mode={resolved_mode} calib={calib}")

    max_frame = max((fx["end_frame"] for fxs in subjects.values() for fx in fxs), default=0)
    sample_frames = np.linspace(1, max_frame, n_frames, dtype=int)

    out_dir = Path("/tmp/calibration_debug") / video_name
    out_dir.mkdir(parents=True, exist_ok=True)

    for f_idx in sample_frames:
        pts = active_fixations_at_frame(subjects, int(f_idx))
        frame_path = frames_dir / f"frame_{f_idx:06d}.png"

        fig, ax = plt.subplots(figsize=(8, 8 * native_h / native_w))
        if frame_path.exists():
            img = plt.imread(frame_path)
            img_h, img_w = img.shape[0], img.shape[1]
            ax.imshow(img, extent=[0, img_w, img_h, 0])
        else:
            img_w, img_h = native_w, native_h
            ax.set_xlim(0, img_w)
            ax.set_ylim(img_h, 0)
            ax.set_facecolor("black")

        for (x, y) in pts:
            vx = x * calib["scale_x"] + calib["offset_x"]
            vy = y * calib["scale_y"] + calib["offset_y"]
            in_frame = (0 <= vx < native_w and 0 <= vy < native_h)
            disp_x = vx * img_w / native_w
            disp_y = vy * img_h / native_h
            color = "lime" if in_frame else "red"
            ax.plot(disp_x, disp_y, "o", color=color, markersize=8, alpha=0.7)

        ax.set_title(f"{video_name} frame {f_idx} "
                     f"({len(pts)} active fixations, green=in-frame, red=dropped)")
        out_path = out_dir / f"frame_{f_idx:06d}_overlay.png"
        fig.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out_path}")

    print(f"  -> inspect the PNGs in {out_dir}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video_name", nargs="?", default=None,
                     help="single video name (positional). For multiple videos, use --videos instead.")
    ap.add_argument("--videos", default=None,
                     help="comma-separated video names to inspect in one run, "
                          "e.g. --videos vidA,vidB,vidC (alternative to the "
                          "positional single-video argument)")
    ap.add_argument("--frames", type=int, default=5, help="number of sample frames to visualize per video")
    ap.add_argument("--mode", choices=["identity", "auto_percentile", "auto_recenter", "manual"], default=None,
                     help="override config.CALIBRATION_MODE for this run (applies to all videos given)")
    args = ap.parse_args()

    if args.videos:
        video_names = [v.strip() for v in args.videos.split(",") if v.strip()]
    elif args.video_name:
        video_names = [args.video_name]
    else:
        raise SystemExit("Specify a video name (positional) or --videos v1,v2,...")

    for name in video_names:
        inspect_one_video(name, args.frames, args.mode)


if __name__ == "__main__":
    main()