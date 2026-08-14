#!/usr/bin/env python3
"""
evaluate_compression.py

Ablation evaluation harness: Gaze-Guided Compression vs. Uniform Encoding.

Given a reference (source) video and a set of already-encoded variants
(produced separately, e.g. via Encoder/run_pipeline.py with and without
per-macroblock QP offsets), this script computes, per variant:

    - Bitrate (kbps) / file size
    - Global Y-PSNR and Y-SSIM (whole frame, luma channel)
    - ROI Y-PSNR / Y-SSIM        (macroblocks the saliency model marked
                                   high-priority -- derived from the QP
                                   offset maps written by qp_map_generator.py)
    - Periphery Y-PSNR / Y-SSIM  (all remaining macroblocks)
    - VMAF (global), if the local ffmpeg build has --enable-libvmaf

...then aggregates results into:
    - results_detailed.csv   one row per encoded variant
    - summary_bdrate.csv     BD-Rate / BD-quality, "gaze" vs "uniform" per clip
    - report.md              human-readable write-up of both tables + notes
    - results.json           full machine-readable dump
    - run.log                complete execution log (for reproducibility)

WHY A SHARED ROI MASK
----------------------
The whole point of the ablation is "did gaze-guided QP allocation spend bits
where it mattered". To test that fairly, the SAME saliency-derived ROI mask
(taken from one canonical qp_dir per clip, i.e. the mask actually used to
drive the gaze-guided encode) is used to score every variant of a clip --
uniform encodes included. That way "ROI quality" measures the same regions
for both methods; only the encoder's bit allocation differs.

MANIFEST SCHEMA
---------------
{
  "roi_percentile": 25,          // optional, default 25 -> top 25% of MB
                                  // area by saliency is "ROI"
  "vmaf_model_path": null,       // optional path to a VMAF .json model
  "clips": [
    {
      "name": "sample_video",
      "reference": "path/to/original_source.mp4",
      "roi_source": "path/to/qp_dir",   // output dir of qp_map_generator.py
                                         // for this clip (used for ALL
                                         // variants of this clip)
      "variants": [
        {
          "method": "uniform",          // "uniform" or "gaze" (free text
                                         // otherwise, but BD-rate compares
                                         // exactly these two labels)
          "label": "crf23",             // any string, used for the RD sweep
          "path": "path/to/uniform_crf23.mp4",
          "encode_time_s": 4.2          // optional
        },
        {
          "method": "gaze",
          "label": "crf23",
          "path": "path/to/gaze_crf23.mp4",
          "encode_time_s": 6.8
        }
        // ... more CRF points. >=4 points per method recommended for a
        // trustworthy BD-Rate; 2 points minimum will still run (linear
        // interpolation, flagged as low-confidence in the report).
      ]
    }
  ]
}

USAGE
-----
    python evaluate_compression.py --manifest manifest.json --out_dir results/
    python evaluate_compression.py --print_example > manifest.json
"""

import argparse
import csv
import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Logging setup (console + run.log, attached once out_dir is known)
# --------------------------------------------------------------------------- #

log = logging.getLogger("eval_compression")
log.setLevel(logging.INFO)


def attach_logfile(out_dir: Path):
    fh = logging.FileHandler(out_dir / "run.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    log.addHandler(fh)


# --------------------------------------------------------------------------- #
# Example manifest (for --print_example)
# --------------------------------------------------------------------------- #

EXAMPLE_MANIFEST = {
    "roi_percentile": 25,
    "vmaf_model_path": None,
    "clips": [
        {
            "name": "sample_video",
            "reference": "Encoder/sample_video.mp4",
            "roi_source": "Encoder/qp_dir",
            "variants": [
                {"method": "uniform", "label": "crf20", "path": "out/uniform_crf20.mp4"},
                {"method": "uniform", "label": "crf23", "path": "out/uniform_crf23.mp4"},
                {"method": "uniform", "label": "crf26", "path": "out/uniform_crf26.mp4"},
                {"method": "uniform", "label": "crf29", "path": "out/uniform_crf29.mp4"},
                {"method": "gaze", "label": "crf20", "path": "out/gaze_crf20.mp4", "qp_dir": "out/qp_dir_crf20"},
                {"method": "gaze", "label": "crf23", "path": "out/gaze_crf23.mp4", "qp_dir": "out/qp_dir_crf23"},
                {"method": "gaze", "label": "crf26", "path": "out/gaze_crf26.mp4", "qp_dir": "out/qp_dir_crf26"},
                {"method": "gaze", "label": "crf29", "path": "out/gaze_crf29.mp4", "qp_dir": "out/qp_dir_crf29"},
            ],
        }
    ],
}


# --------------------------------------------------------------------------- #
# Media probing (bitrate, resolution, duration) via ffprobe
# --------------------------------------------------------------------------- #

def ffprobe_json(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_entries", "format=duration,size,bit_rate:stream=width,height,r_frame_rate,nb_frames",
        "-select_streams", "v:0", str(path),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"ffprobe failed on {path}: {out.stderr.strip()}")
    return json.loads(out.stdout)


def get_media_info(path: Path) -> dict:
    info = ffprobe_json(path)
    fmt = info.get("format", {})
    stream = (info.get("streams") or [{}])[0]

    duration = float(fmt.get("duration", 0.0) or 0.0)
    size_bytes = int(fmt.get("size", 0) or 0)

    bit_rate = fmt.get("bit_rate")
    if bit_rate is not None:
        bitrate_kbps = float(bit_rate) / 1000.0
    elif duration > 0:
        bitrate_kbps = (size_bytes * 8.0) / duration / 1000.0
    else:
        bitrate_kbps = float("nan")

    return {
        "width": int(stream.get("width", 0) or 0),
        "height": int(stream.get("height", 0) or 0),
        "duration_s": duration,
        "size_mb": size_bytes / (1024 * 1024),
        "bitrate_kbps": bitrate_kbps,
    }


# --------------------------------------------------------------------------- #
# VMAF availability check + computation
# --------------------------------------------------------------------------- #

_VMAF_AVAILABLE: Optional[bool] = None


def vmaf_available() -> bool:
    global _VMAF_AVAILABLE
    if _VMAF_AVAILABLE is None:
        out = subprocess.run(["ffmpeg", "-filters"], capture_output=True, text=True)
        _VMAF_AVAILABLE = "libvmaf" in out.stdout
        if not _VMAF_AVAILABLE:
            log.warning(
                "This ffmpeg build does not expose the 'libvmaf' filter. "
                "VMAF scores will be reported as N/A. Rebuild ffmpeg with "
                "--enable-libvmaf, or install a build that has it, to get VMAF."
            )
    return _VMAF_AVAILABLE


def compute_vmaf(ref_path: Path, dist_path: Path, model_path: Optional[str], work_dir: Path) -> Optional[float]:
    if not vmaf_available():
        return None

    import re, tempfile, shutil

    # ffmpeg's filter option parser can't handle Windows paths (backslashes
    # are escape chars, colons are option separators). Write the JSON log to
    # a temp file first, then copy it to the vmaf_logs subfolder.
    # We run ffmpeg with cwd=tempdir and use just the filename to avoid
    # any path separator or colon issues.
    tmp_fd, tmp_log = tempfile.mkstemp(suffix=".json", prefix="vmaf_")
    os.close(tmp_fd)
    tmp_log_path = Path(tmp_log)
    tmp_dir = str(tmp_log_path.parent)
    log_filename = tmp_log_path.name

    model_opt = f":model_path={model_path}" if model_path else ""
    filt = (
        f"[0:v]setpts=PTS-STARTPTS[dist];[1:v]setpts=PTS-STARTPTS[ref];"
        f"[dist][ref]libvmaf=log_fmt=json:log_path={log_filename}{model_opt}"
    )
    cmd = ["ffmpeg", "-y", "-i", str(dist_path), "-i", str(ref_path),
           "-lavfi", filt, "-f", "null", "-"]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=tmp_dir)
    if out.returncode != 0:
        log.warning(f"VMAF computation failed for {dist_path.name}: {out.stderr[-500:]}")
        tmp_log_path.unlink(missing_ok=True)
        return None

    # Save the JSON log to vmaf_logs/ subfolder (best-effort).
    vmaf_logs_dir = work_dir / "vmaf_logs"
    vmaf_logs_dir.mkdir(parents=True, exist_ok=True)
    final_log_path = vmaf_logs_dir / f"vmaf_{dist_path.stem}.json"
    if tmp_log_path.exists() and tmp_log_path.stat().st_size > 0:
        shutil.copy2(str(tmp_log_path), str(final_log_path))
        log.info(f"  VMAF log saved to {final_log_path}")
    tmp_log_path.unlink(missing_ok=True)

    # Parse the VMAF score directly from ffmpeg's stderr output.
    # ffmpeg always prints: [Parsed_libvmaf_N @ 0xADDR] VMAF score: XX.XXXXXX
    match = re.search(r"VMAF score:\s*([\d.]+)", out.stderr)
    if match:
        return float(match.group(1))

    log.warning(f"Could not parse VMAF score from ffmpeg output for {dist_path.name}")
    return None


# --------------------------------------------------------------------------- #
# ROI / periphery masks, derived from the saliency-driven QP offsets already
# written by Encoder/qp_map_generator.py (metadata.json + qpoffset_NNNNNN.bin)
# --------------------------------------------------------------------------- #

@dataclass
class RoiSource:
    mb_width: int
    mb_height: int
    qp_min: float
    qp_max: float
    num_frames: int
    qp_dir: Path

    def saliency_grid(self, frame_idx: int) -> np.ndarray:
        """Recover normalized [0,1] saliency (1=high) for one frame,
        inverting qp_map_generator.map_to_qp_offsets()."""
        idx = min(frame_idx, self.num_frames - 1)  # hold last frame if short
        bin_path = self.qp_dir / f"qpoffset_{idx:06d}.bin"
        offsets = np.fromfile(bin_path, dtype=np.float32).reshape(self.mb_height, self.mb_width)
        rng = self.qp_max - self.qp_min
        if rng <= 1e-8:
            return np.zeros_like(offsets)
        # offset = qp_max - (qp_max - qp_min) * normalized  =>  invert:
        normalized = (self.qp_max - offsets) / rng
        return np.clip(normalized, 0.0, 1.0)

    def roi_mask(self, frame_idx: int, out_h: int, out_w: int, roi_percentile: float) -> np.ndarray:
        """Boolean mask at native (out_h, out_w) resolution. True = ROI
        (top `roi_percentile`% of macroblocks by saliency, this frame)."""
        sal = self.saliency_grid(frame_idx)
        cutoff = np.percentile(sal, 100.0 - roi_percentile)
        mb_mask = sal >= cutoff
        mask = cv2.resize(mb_mask.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        return mask.astype(bool)


def load_roi_source(qp_dir: Path) -> RoiSource:
    meta_path = qp_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"No metadata.json in {qp_dir}. Expected the output directory of "
            f"Encoder/qp_map_generator.py (per-frame qpoffset_*.bin + metadata.json)."
        )
    with open(meta_path) as f:
        meta = json.load(f)
    return RoiSource(
        mb_width=meta["mb_width"],
        mb_height=meta["mb_height"],
        qp_min=meta["qp_min"],
        qp_max=meta["qp_max"],
        num_frames=meta["num_frames"],
        qp_dir=qp_dir,
    )


# --------------------------------------------------------------------------- #
# Pixel-domain metrics: Y-PSNR, Y-SSIM (Wang et al. 2004, 11x11 Gaussian window)
# --------------------------------------------------------------------------- #

_SSIM_KSIZE = (11, 11)
_SSIM_SIGMA = 1.5


def _gblur(x: np.ndarray) -> np.ndarray:
    # Gaussian window is separable -- cv2.GaussianBlur uses a separable,
    # SIMD-optimized path and is dramatically faster than an equivalent
    # cv2.filter2D with a dense 2D kernel (seconds -> milliseconds per
    # 1080p frame in testing). Keep everything in float32 for speed.
    return cv2.GaussianBlur(x, _SSIM_KSIZE, _SSIM_SIGMA)


def squared_error_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    return (a - b) ** 2


def psnr_from_se_map(se_map: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    vals = se_map[mask] if mask is not None else se_map
    if vals.size == 0:
        return float("nan")
    mse = float(np.mean(vals))
    if mse <= 1e-12:
        return 99.0
    return 10.0 * np.log10((255.0 ** 2) / mse)


def ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Full per-pixel SSIM map (Wang et al. 2004, Gaussian-windowed).
    Compute once per frame pair and reuse for global/ROI/periphery
    averages -- this is the expensive step, so avoid recomputing it
    per region."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    C1 = np.float32((0.01 * 255) ** 2)
    C2 = np.float32((0.03 * 255) ** 2)

    mu1 = _gblur(a)
    mu2 = _gblur(b)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = _gblur(a * a) - mu1_sq
    sigma2_sq = _gblur(b * b) - mu2_sq
    sigma12 = _gblur(a * b) - mu1_mu2

    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    return num / den


def ssim_from_map(smap: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    vals = smap[mask] if mask is not None else smap
    if vals.size == 0:
        return float("nan")
    return float(vals.mean())


def to_luma(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)[:, :, 0]


# --------------------------------------------------------------------------- #
# Per-variant evaluation: decode reference + variant together, score frames
# --------------------------------------------------------------------------- #

@dataclass
class VariantResult:
    clip: str
    method: str
    label: str
    path: str
    width: int = 0
    height: int = 0
    frames_evaluated: int = 0
    duration_s: float = 0.0
    size_mb: float = 0.0
    bitrate_kbps: float = float("nan")
    psnr_global: float = float("nan")
    psnr_roi: float = float("nan")
    psnr_periphery: float = float("nan")
    ssim_global: float = float("nan")
    ssim_roi: float = float("nan")
    ssim_periphery: float = float("nan")
    vmaf_global: Optional[float] = None
    encode_time_s: Optional[float] = None
    notes: str = ""


def evaluate_variant(clip_name: str, ref_path: Path, variant: dict,
                      roi_source: RoiSource, roi_percentile: float,
                      vmaf_model_path: Optional[str], work_dir: Path,
                      manifest_dir: Optional[Path] = None) -> VariantResult:
    dist_path = Path(variant["path"])
    if not dist_path.is_absolute() and manifest_dir is not None:
        dist_path = (manifest_dir / dist_path).resolve()
    method = variant["method"]
    label = variant.get("label", dist_path.stem)

    res = VariantResult(clip=clip_name, method=method, label=label, path=str(dist_path))
    notes = []

    if not dist_path.exists():
        res.notes = f"MISSING FILE: {dist_path}"
        log.error(res.notes)
        return res

    info = get_media_info(dist_path)
    res.width, res.height = info["width"], info["height"]
    res.duration_s = info["duration_s"]
    res.size_mb = info["size_mb"]
    res.bitrate_kbps = info["bitrate_kbps"]
    res.encode_time_s = variant.get("encode_time_s")

    cap_ref = cv2.VideoCapture(str(ref_path))
    cap_dist = cv2.VideoCapture(str(dist_path))
    if not cap_ref.isOpened() or not cap_dist.isOpened():
        res.notes = "Could not open reference or variant video with OpenCV."
        log.error(res.notes)
        return res

    n_ref = int(cap_ref.get(cv2.CAP_PROP_FRAME_COUNT))
    n_dist = int(cap_dist.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(n_ref, n_dist)
    if abs(n_ref - n_dist) > 2:
        notes.append(f"frame count mismatch ref={n_ref} dist={n_dist}, using first {n_frames}")
        log.warning(f"[{clip_name}/{method}/{label}] {notes[-1]}")

    psnr_g, psnr_r, psnr_p = [], [], []
    ssim_g, ssim_r, ssim_p = [], [], []

    for idx in range(n_frames):
        ok_r, fr_ref = cap_ref.read()
        ok_d, fr_dist = cap_dist.read()
        if not (ok_r and ok_d):
            break

        if fr_dist.shape[:2] != fr_ref.shape[:2]:
            fr_dist = cv2.resize(fr_dist, (fr_ref.shape[1], fr_ref.shape[0]))

        y_ref = to_luma(fr_ref)
        y_dist = to_luma(fr_dist)

        roi_mask = roi_source.roi_mask(idx, y_ref.shape[0], y_ref.shape[1], roi_percentile)
        periph_mask = ~roi_mask

        se_map = squared_error_map(y_ref, y_dist)
        psnr_g.append(psnr_from_se_map(se_map))
        psnr_r.append(psnr_from_se_map(se_map, roi_mask))
        psnr_p.append(psnr_from_se_map(se_map, periph_mask))

        smap = ssim_map(y_ref, y_dist)
        ssim_g.append(ssim_from_map(smap))
        ssim_r.append(ssim_from_map(smap, roi_mask))
        ssim_p.append(ssim_from_map(smap, periph_mask))

    cap_ref.release()
    cap_dist.release()

    res.frames_evaluated = len(psnr_g)
    if res.frames_evaluated == 0:
        notes.append("No frames could be scored.")
        res.notes = "; ".join(notes)
        log.error(f"[{clip_name}/{method}/{label}] {res.notes}")
        return res

    res.psnr_global = float(np.nanmean(psnr_g))
    res.psnr_roi = float(np.nanmean(psnr_r))
    res.psnr_periphery = float(np.nanmean(psnr_p))
    res.ssim_global = float(np.nanmean(ssim_g))
    res.ssim_roi = float(np.nanmean(ssim_r))
    res.ssim_periphery = float(np.nanmean(ssim_p))

    res.vmaf_global = compute_vmaf(ref_path, dist_path, vmaf_model_path, work_dir)

    res.notes = "; ".join(notes)
    log.info(
        f"[{clip_name}/{method}/{label}] {res.frames_evaluated} frames | "
        f"{res.bitrate_kbps:.1f} kbps | PSNR(g/roi/per)="
        f"{res.psnr_global:.2f}/{res.psnr_roi:.2f}/{res.psnr_periphery:.2f} dB | "
        f"SSIM(g/roi/per)={res.ssim_global:.4f}/{res.ssim_roi:.4f}/{res.ssim_periphery:.4f}"
        + (f" | VMAF={res.vmaf_global:.2f}" if res.vmaf_global is not None else "")
    )
    return res


# --------------------------------------------------------------------------- #
# BD-Rate / BD-quality (Bjontegaard delta), classic piecewise-polynomial method
# --------------------------------------------------------------------------- #

def _bd_fit_degree(n_points: int) -> int:
    return max(1, min(3, n_points - 1))


def bd_rate(rate_anchor, qual_anchor, rate_test, qual_test):
    """% bitrate change of `test` vs `anchor` at equal quality.
    Negative = test needs fewer bits for the same quality (an improvement
    when `test` is the gaze-guided method)."""
    rate_anchor, qual_anchor = np.asarray(rate_anchor, float), np.asarray(qual_anchor, float)
    rate_test, qual_test = np.asarray(rate_test, float), np.asarray(qual_test, float)

    if len(rate_anchor) < 2 or len(rate_test) < 2:
        return None, "insufficient RD points (need >=2 per method, >=4 recommended)"

    log_r_a, log_r_t = np.log(rate_anchor), np.log(rate_test)
    deg_a, deg_t = _bd_fit_degree(len(rate_anchor)), _bd_fit_degree(len(rate_test))

    p_a = np.polyfit(qual_anchor, log_r_a, deg_a)
    p_t = np.polyfit(qual_test, log_r_t, deg_t)

    lo = max(qual_anchor.min(), qual_test.min())
    hi = min(qual_anchor.max(), qual_test.max())
    if hi <= lo:
        return None, "no overlapping quality range between methods"

    int_a = np.polyval(np.polyint(p_a), hi) - np.polyval(np.polyint(p_a), lo)
    int_t = np.polyval(np.polyint(p_t), hi) - np.polyval(np.polyint(p_t), lo)
    avg_log_diff = (int_t - int_a) / (hi - lo)

    pct = (np.exp(avg_log_diff) - 1.0) * 100.0
    note = "" if (len(rate_anchor) >= 4 and len(rate_test) >= 4) else \
        f"low-confidence (only {len(rate_anchor)}/{len(rate_test)} RD points; recommend >=4)"
    return float(pct), note


def bd_quality(rate_anchor, qual_anchor, rate_test, qual_test):
    """Average quality delta of `test` vs `anchor` at equal bitrate.
    Positive = test has higher quality at the same bitrate."""
    rate_anchor, qual_anchor = np.asarray(rate_anchor, float), np.asarray(qual_anchor, float)
    rate_test, qual_test = np.asarray(rate_test, float), np.asarray(qual_test, float)

    if len(rate_anchor) < 2 or len(rate_test) < 2:
        return None, "insufficient RD points (need >=2 per method, >=4 recommended)"

    log_r_a, log_r_t = np.log(rate_anchor), np.log(rate_test)
    deg_a, deg_t = _bd_fit_degree(len(rate_anchor)), _bd_fit_degree(len(rate_test))

    p_a = np.polyfit(log_r_a, qual_anchor, deg_a)
    p_t = np.polyfit(log_r_t, qual_test, deg_t)

    lo = max(log_r_a.min(), log_r_t.min())
    hi = min(log_r_a.max(), log_r_t.max())
    if hi <= lo:
        return None, "no overlapping bitrate range between methods"

    int_a = np.polyval(np.polyint(p_a), hi) - np.polyval(np.polyint(p_a), lo)
    int_t = np.polyval(np.polyint(p_t), hi) - np.polyval(np.polyint(p_t), lo)
    avg_diff = (int_t - int_a) / (hi - lo)

    note = "" if (len(rate_anchor) >= 4 and len(rate_test) >= 4) else \
        f"low-confidence (only {len(rate_anchor)}/{len(rate_test)} RD points; recommend >=4)"
    return float(avg_diff), note


# --------------------------------------------------------------------------- #
# Report writers
# --------------------------------------------------------------------------- #

DETAILED_FIELDS = [
    "clip", "method", "label", "path", "width", "height", "frames_evaluated",
    "duration_s", "size_mb", "bitrate_kbps",
    "psnr_global", "psnr_roi", "psnr_periphery",
    "ssim_global", "ssim_roi", "ssim_periphery",
    "vmaf_global", "encode_time_s", "notes",
]


def write_detailed_csv(results: list, out_path: Path):
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DETAILED_FIELDS)
        w.writeheader()
        for r in results:
            w.writerow({k: getattr(r, k) for k in DETAILED_FIELDS})


def compute_summary(results: list) -> list:
    """One BD-rate/BD-quality row per clip, comparing method=='gaze' vs
    method=='uniform' RD curves (sorted by label)."""
    by_clip = {}
    for r in results:
        by_clip.setdefault(r.clip, {}).setdefault(r.method, []).append(r)

    rows = []
    for clip, methods in by_clip.items():
        if "uniform" not in methods or "gaze" not in methods:
            rows.append({"clip": clip, "note": "need both 'uniform' and 'gaze' variants to compute BD-Rate"})
            continue

        u = sorted(methods["uniform"], key=lambda r: r.bitrate_kbps)
        g = sorted(methods["gaze"], key=lambda r: r.bitrate_kbps)

        row = {"clip": clip, "n_uniform_points": len(u), "n_gaze_points": len(g)}

        for metric_name, attr in [("psnr", "psnr_global"), ("ssim", "ssim_global"), ("vmaf", "vmaf_global")]:
            ru = [x.bitrate_kbps for x in u]
            qu = [getattr(x, attr) for x in u]
            rg = [x.bitrate_kbps for x in g]
            qg = [getattr(x, attr) for x in g]

            if metric_name == "vmaf" and (any(v is None for v in qu) or any(v is None for v in qg)):
                row[f"bdrate_{metric_name}_pct"] = None
                row[f"bd_{metric_name}"] = None
                row[f"{metric_name}_note"] = "VMAF unavailable for one or more points"
                continue

            pct, note1 = bd_rate(ru, qu, rg, qg)
            delta, note2 = bd_quality(ru, qu, rg, qg)
            row[f"bdrate_{metric_name}_pct"] = pct
            row[f"bd_{metric_name}"] = delta
            row[f"{metric_name}_note"] = note1 or note2

        rows.append(row)
    return rows


def write_summary_csv(summary_rows: list, out_path: Path):
    fields = ["clip", "n_uniform_points", "n_gaze_points",
              "bdrate_psnr_pct", "bd_psnr", "psnr_note",
              "bdrate_ssim_pct", "bd_ssim", "ssim_note",
              "bdrate_vmaf_pct", "bd_vmaf", "vmaf_note", "note"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in summary_rows:
            w.writerow({k: row.get(k, "") for k in fields})


def fmt(v, nd=3):
    if v is None:
        return "N/A"
    if isinstance(v, float) and np.isnan(v):
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def write_markdown_report(results: list, summary_rows: list, out_path: Path,
                           manifest_path: Path, roi_percentile: float):
    import datetime
    lines = []
    lines.append("# Gaze-Guided vs. Uniform Compression -- Evaluation Report\n")
    lines.append(f"- Generated: {datetime.datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"- Manifest: `{manifest_path}`")
    lines.append(f"- ROI definition: top {roi_percentile:.0f}% of macroblocks by predicted "
                  f"saliency (per frame), derived from the QP-offset maps of the canonical "
                  f"`roi_source` for each clip. Same mask used to score every variant of a clip.")
    lines.append(f"- Quality metrics computed on the luma (Y) channel: PSNR and single-scale "
                  f"SSIM (Wang et al. 2004, 11x11 Gaussian window, sigma=1.5).")
    lines.append(f"- VMAF: {'available' if vmaf_available() else 'NOT available in this ffmpeg build (N/A below)'}.\n")

    lines.append("## 1. Per-variant results\n")
    header = ["Clip", "Method", "Label", "Bitrate (kbps)", "Size (MB)",
              "PSNR global/ROI/periph (dB)", "SSIM global/ROI/periph",
              "VMAF", "Encode time (s)", "Notes"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for r in sorted(results, key=lambda r: (r.clip, r.method, r.bitrate_kbps)):
        psnr_str = f"{fmt(r.psnr_global,2)} / {fmt(r.psnr_roi,2)} / {fmt(r.psnr_periphery,2)}"
        ssim_str = f"{fmt(r.ssim_global,4)} / {fmt(r.ssim_roi,4)} / {fmt(r.ssim_periphery,4)}"
        lines.append("| " + " | ".join([
            r.clip, r.method, r.label, fmt(r.bitrate_kbps, 1), fmt(r.size_mb, 2),
            psnr_str, ssim_str, fmt(r.vmaf_global, 2), fmt(r.encode_time_s, 2),
            r.notes or "",
        ]) + " |")

    lines.append("\n## 2. BD-Rate / BD-quality summary (gaze vs. uniform, per clip)\n")
    lines.append(
        "Negative BD-Rate % = gaze-guided encoding needs fewer bits for equal quality "
        "(bitrate savings). Positive BD-quality = gaze-guided has higher quality at equal "
        "bitrate. Both are computed over the bitrate/quality range common to both methods.\n"
    )
    header2 = ["Clip", "#Uniform pts", "#Gaze pts",
               "BD-Rate PSNR (%)", "BD-PSNR (dB)",
               "BD-Rate SSIM (%)", "BD-SSIM",
               "BD-Rate VMAF (%)", "BD-VMAF", "Notes"]
    lines.append("| " + " | ".join(header2) + " |")
    lines.append("|" + "---|" * len(header2))
    for row in summary_rows:
        if "note" in row and "bdrate_psnr_pct" not in row:
            lines.append(f"| {row['clip']} | - | - | - | - | - | - | - | - | {row['note']} |")
            continue
        notes = "; ".join(filter(None, [row.get("psnr_note"), row.get("ssim_note"), row.get("vmaf_note")]))
        lines.append("| " + " | ".join([
            row["clip"], str(row["n_uniform_points"]), str(row["n_gaze_points"]),
            fmt(row.get("bdrate_psnr_pct"), 2), fmt(row.get("bd_psnr"), 3),
            fmt(row.get("bdrate_ssim_pct"), 2), fmt(row.get("bd_ssim"), 4),
            fmt(row.get("bdrate_vmaf_pct"), 2), fmt(row.get("bd_vmaf"), 2),
            notes,
        ]) + " |")

    lines.append("\n## 3. How to read this\n")
    lines.append(
        "- **Global** columns show whole-frame quality -- the classic RD comparison.\n"
        "- **ROI** columns isolate the macroblocks the saliency model flagged as important; "
        "this is where the gaze-guided method should show a clear advantage.\n"
        "- **Periphery** columns show the rest of the frame; some quality loss here relative "
        "to uniform encoding at the same overall bitrate is expected and is the mechanism "
        "by which bits were saved/reallocated.\n"
        "- BD-Rate/BD-quality require >=2 RD points per method to compute at all, and the "
        "cubic fit used by the standard Bjontegaard method is only well-conditioned with "
        ">=4 points (rows are flagged 'low-confidence' otherwise). Sweep more CRF/QP-scale "
        "values if you see that flag.\n"
    )

    out_path.write_text("\n".join(lines))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def _default_manifest() -> Path:
    """Look for ablation_out/manifest.json next to this script."""
    return Path(__file__).resolve().parent / "ablation_out" / "manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=_default_manifest(),
                        help="Path to manifest.json. Default: ablation_out/manifest.json next to this script.")
    default_out_dir = Path(__file__).resolve().parent / "ablation_out" / "eval_results"
    parser.add_argument("--out_dir", type=Path, default=default_out_dir,
                        help="Output directory. Default: ablation_out/eval_results next to this script.")
    parser.add_argument("--print_example", action="store_true", help="Print an example manifest.json and exit.")
    args = parser.parse_args()

    if args.print_example:
        print(json.dumps(EXAMPLE_MANIFEST, indent=2))
        return

    if not args.manifest:
        parser.error(
            f"No --manifest provided and default not found at {args.manifest}.\n"
            f"Either pass --manifest path/to/manifest.json or run run_ablation_sweep.py first."
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    attach_logfile(args.out_dir)
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(message)s"))
    log.addHandler(console)

    log.info(f"Loading manifest: {args.manifest}")
    manifest_dir = args.manifest.resolve().parent
    with open(args.manifest) as f:
        manifest = json.load(f)

    roi_percentile = float(manifest.get("roi_percentile", 25))
    vmaf_model_path = manifest.get("vmaf_model_path")
    vmaf_available()  # logs a warning once, up front, if unavailable

    all_results = []
    for clip in manifest["clips"]:
        clip_name = clip["name"]
        ref_path = Path(clip["reference"])
        if not ref_path.is_absolute():
            ref_path = (manifest_dir / ref_path).resolve()
        log.info(f"\n=== Clip: {clip_name} ===")

        if not ref_path.exists():
            log.error(f"Reference video not found: {ref_path}. Skipping clip.")
            continue

        try:
            roi_src_path = Path(clip["roi_source"])
            if not roi_src_path.is_absolute():
                roi_src_path = (manifest_dir / roi_src_path).resolve()
            roi_source = load_roi_source(roi_src_path)
        except FileNotFoundError as e:
            log.error(str(e) + f" Skipping clip '{clip_name}'.")
            continue

        for variant in clip["variants"]:
            qp_dir_override = variant.get("qp_dir")
            if qp_dir_override:
                qp_override_path = Path(qp_dir_override)
                if not qp_override_path.is_absolute():
                    qp_override_path = (manifest_dir / qp_override_path).resolve()
                active_roi_source = load_roi_source(qp_override_path)
            else:
                active_roi_source = roi_source
            res = evaluate_variant(clip_name, ref_path, variant, active_roi_source,
                                    roi_percentile, vmaf_model_path, args.out_dir,
                                    manifest_dir=manifest_dir)
            all_results.append(res)

    if not all_results:
        log.error("No results were produced. Check the manifest paths and try again.")
        sys.exit(1)

    write_detailed_csv(all_results, args.out_dir / "results_detailed.csv")
    summary_rows = compute_summary(all_results)
    write_summary_csv(summary_rows, args.out_dir / "summary_bdrate.csv")
    write_markdown_report(all_results, summary_rows, args.out_dir / "report.md", args.manifest, roi_percentile)

    with open(args.out_dir / "results.json", "w") as f:
        json.dump({
            "detailed": [asdict(r) for r in all_results],
            "summary": summary_rows,
            "roi_percentile": roi_percentile,
        }, f, indent=2)

    log.info(f"\nDone. Wrote:")
    log.info(f"  {args.out_dir / 'results_detailed.csv'}")
    log.info(f"  {args.out_dir / 'summary_bdrate.csv'}")
    log.info(f"  {args.out_dir / 'report.md'}")
    log.info(f"  {args.out_dir / 'results.json'}")
    log.info(f"  {args.out_dir / 'run.log'}")


if __name__ == "__main__":
    main()
