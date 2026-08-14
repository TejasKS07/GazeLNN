
"""
run_ablation_sweep.py  (drop this into the Encoder/ folder, next to run_pipeline.py)

Ablation-sweep version of run_pipeline.py: instead of one gaze-guided encode
at CRF 26, this compiles the encoder once, then produces BOTH a uniform and
a gaze-guided encode at each CRF in CRF_SWEEP, and writes them out in the
layout evaluate_compression.py's manifest.json expects.

Uniform baseline = encoder.exe pointed at an EMPTY diasrectory. encoder.c
falls back to all-zero QP offsets when it can't find a frame's .bin file
(see the fopen()/fread() fallback around line ~330), so this is the exact
same binary/CRF/preset as the gaze-guided run, just without per-MB offsets.
No changes to encoder.c needed.

Output layout (under Encoder/):
    Encoder/
      qp_dir/                      # one saliency map, reused across CRFs
      empty_qp_dir/                # empty on purpose -> forces zero offsets
      ablation_out/
        uniform_crf20.mp4 ... crf29.mp4
        gaze_crf20.mp4 ... crf29.mp4
        manifest.json              # ready to feed straight into evaluate_compression.py

Edit CRF_SWEEP / INPUT_VIDEO / QP params below as needed, then run:
    python run_ablation_sweep.py
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# --------------------------------------------------------------------------- #
# Config -- edit these for your test clip / sweep
# --------------------------------------------------------------------------- #

WORK_DIR = Path(__file__).resolve().parent
INPUT_VIDEO = WORK_DIR / "Sample_Video.mp4"   # <- point this at your test clip
print(repr(WORK_DIR))
print(repr(INPUT_VIDEO))
print("exists?", INPUT_VIDEO.exists())
OUT_DIR = WORK_DIR / "ablation_out"
CRF_SWEEP = [20, 23, 26, 29]                          # >=4 points recommended for BD-Rate
PRESET = "medium"

# qp_map_generator.py settings (same for every CRF -- one saliency pass is
# enough since the model doesn't depend on CRF; only the encoder call does)
QP_MIN = -3.0
QP_MAX = 10.0
SPATIAL_SIGMA = 1.5
TEMPORAL_ALPHA = 1.0


def run_cmd(cmd, env=None):
    print(f"Running command: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, shell=False)
    if result.returncode != 0:
        print("--- STDOUT ---")
        print(result.stdout)
        print("--- STDERR ---")
        print(result.stderr)
        raise RuntimeError(f"Command failed with exit code {result.returncode}")
    print("Command executed successfully!")
    return result


def main():
    if not INPUT_VIDEO.exists():
        print(f"Error: {INPUT_VIDEO} not found.", file=sys.stderr)
        sys.exit(1)

    env = os.environ.copy()
    env["PATH"] = "C:\\msys64\\mingw64\\bin;" + env.get("PATH", "")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    qp_dir = WORK_DIR / "qp_dir"
    empty_qp_dir = WORK_DIR / "empty_qp_dir"
    empty_qp_dir.mkdir(parents=True, exist_ok=True)
    # keep it empty every run, in case a previous run left files in it
    for f in empty_qp_dir.iterdir():
        f.unlink()

    # 1. Compile the encoder once
    print("=== Compiling encoder.c ===")
    gcc_cmd = [
        "C:\\msys64\\mingw64\\bin\\gcc.exe", "-O3", "-Wall",
        str(WORK_DIR / "encoder.c"), "-o", str(WORK_DIR / "encoder.exe"),
        "-lavcodec", "-lavformat", "-lavutil", "-lswscale", "-lx264",
    ]
    run_cmd(gcc_cmd, env=env)
    encoder_exe = WORK_DIR / "encoder.exe"

    # 2. Generate the saliency-driven QP offset maps ONCE for this clip.
    #    Reused for every CRF in the sweep -- the model's output doesn't
    #    depend on the encoder's CRF.
    print("\n=== Generating QP offset maps (qp_map_generator.py) ===")
    if qp_dir.exists():
        shutil.rmtree(qp_dir)
    qp_dir.mkdir(parents=True)
    run_cmd([
        sys.executable, "qp_map_generator.py",
        "--video_path", str(INPUT_VIDEO),
        "--output_dir", str(qp_dir),
        "--qp_min", str(QP_MIN),
        "--qp_max", str(QP_MAX),
        "--spatial_sigma", str(SPATIAL_SIGMA),
        "--temporal_alpha", str(TEMPORAL_ALPHA),
    ], env=env)

    # 3. Encode uniform + gaze-guided at every CRF in the sweep
    manifest_variants = []
    for crf in CRF_SWEEP:
        label = f"crf{crf}"

        # --- uniform: same encoder/CRF/preset, empty qp dir -> zero offsets
        uniform_out = OUT_DIR / f"uniform_{label}.mp4"
        print(f"\n=== Encoding UNIFORM crf={crf} ===")
        t0 = time.time()
        run_cmd([
            str(encoder_exe), str(INPUT_VIDEO), str(uniform_out), str(empty_qp_dir),
            "--crf", str(crf), "--preset", PRESET,
        ], env=env)
        uniform_time = time.time() - t0
        manifest_variants.append({
            "method": "uniform", "label": label,
            "path": str(uniform_out.relative_to(OUT_DIR)),
            "encode_time_s": round(uniform_time, 2),
        })

        # --- gaze-guided: same encoder/CRF/preset, real qp dir
        gaze_out = OUT_DIR / f"gaze_{label}.mp4"
        print(f"\n=== Encoding GAZE-GUIDED crf={crf} ===")
        t0 = time.time()
        run_cmd([
            str(encoder_exe), str(INPUT_VIDEO), str(gaze_out), str(qp_dir),
            "--crf", str(crf), "--preset", PRESET,
        ], env=env)
        gaze_time = time.time() - t0
        manifest_variants.append({
            "method": "gaze", "label": label,
            "path": str(gaze_out.relative_to(OUT_DIR)),
            "qp_dir": os.path.relpath(qp_dir, OUT_DIR),
            "encode_time_s": round(gaze_time, 2),
        })

    # 4. Write a manifest.json ready for evaluate_compression.py
    manifest = {
        "roi_percentile": 25,
        "clips": [{
            "name": INPUT_VIDEO.stem,
            "reference": str(INPUT_VIDEO.relative_to(OUT_DIR)) if INPUT_VIDEO.is_relative_to(OUT_DIR)
                         else os.path.relpath(INPUT_VIDEO, OUT_DIR),
            "roi_source": os.path.relpath(qp_dir, OUT_DIR),
            "variants": manifest_variants,
        }],
    }
    manifest_path = OUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n=== Sweep complete ===")
    print(f"Outputs in: {OUT_DIR}")
    print(f"Manifest written to: {manifest_path}")
    print(
        f"\nNext step (from {OUT_DIR}):\n"
        f"  python evaluate_compression.py --manifest manifest.json --out_dir results/"
    )


if __name__ == "__main__":
    main()
