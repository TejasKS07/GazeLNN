#!/usr/bin/env python3
# The above line allows the file to run on mac

"""
This script:
1. Compiles the C encoder using GCC.
2. Extracts resolution/framecount from sample_video.mp4.
3. Converts heatmaps to flat binary QP offset files.
4. Encodes the video using encoder.exe.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_cmd(cmd, env = None):
    print(f"Running command: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, shell=False)
    if result.returncode != 0:
        print("--- STDOUT ---")
        print(result.stdout)
        print("--- STDERR ---")
        print(result.stderr)
        raise RuntimeError(f"Command failed with exit code {result.returncode}")
    else:
        print("Command executed successfully!")

def main():
    work_dir = Path(__file__).resolve().parent

    # Setting up MSYS2 DLL environment paths for all subprocesses
    env = os.environ.copy()
    env["PATH"] = "C:\\msys64\\mingw64\\bin;" + env.get("PATH", "")

    input_video = work_dir / "sample_video.mp4"             # Edit this to choose the video to run the script on
    output_video = work_dir / "output_video.mp4"

    if not input_video.exists():
        print(f"Error: sample_video.mp4 not found in {work_dir}", file=sys.stderr)
        sys.exit(1)

    # Compiling C code
    print("Compiling Encoder.C")

    gcc_path = "C:\\msys64\\mingw64\\bin\\gcc.exe"
    gcc_cmd = [
        gcc_path,
        "-O3",
        "-Wall",
        str(work_dir / "encoder.c"),
        "-o",
        str(work_dir / "encoder.exe"),
        "-lavcodec",
        "-lavformat",
        "-lavutil",
        "-lswscale",
        "-lx264"
    ]
    run_cmd(gcc_cmd, env=env)

    # Clean up previous runs
    for f in [output_video]:      
        if f.exists():
            f.unlink()

    # Re-create directories
    qp_dir = work_dir / "qp_dir"
    qp_dir.mkdir(parents=True, exist_ok=True)
    for child in qp_dir.iterdir():
        if child.is_file():
            child.unlink()

    # Generate QP offsets using qp_map_generator.py
    print("\nRunning qp_map_generator.py")

    qpoffset_cmd = [
        sys.executable,
        "qp_map_generator.py",
        "--video_path", str(input_video),
        "--output_dir", str(qp_dir),
        "--qp_min", "-3.0",
        "--qp_max", "10.0",
        "--spatial_sigma", "1.5",
        "--temporal_alpha", "1.0"
    ]
    run_cmd(qpoffset_cmd, env=env)

    # Run compiled C encoder
    print("\nRunning encoder.exe")

    encoder_cmd = [
        str(work_dir / "encoder.exe"),
        str(input_video),
        str(output_video),
        str(qp_dir),
        "--crf", "26",
        "--preset", "medium"
    ]
    run_cmd(encoder_cmd, env=env)

    # Pipeline Check
    print("\nPipeline Status Report")

    if output_video.exists() and output_video.stat().st_size > 0:
        print("SUCCESS: Encoded output video created.")
        print(f"  -> Path: {output_video}")
        print(f"  -> Size: {output_video.stat().st_size / (1024*1024):.2f} MB")
    else:
        print("ERROR: Output video was not created or is empty.")

if __name__ == "__main__":
    main()