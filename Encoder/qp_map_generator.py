import argparse
import json
import os
import sys

from scipy.ndimage import gaussian_filter
from pathlib import Path

import numpy as np
import torch
from torchvision import transforms
import cv2

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR / "DIEM"))

from DIEM_Modeldata import GazeLLNArch

def mb_grid(native_width = 1920, native_height = 1080):
    """
    Prepares the macroblock grid.

    Args:
        native_width: The width of the original frame
        native_height: The height of the original frame
    """

    # Can try to implement this using numpy functions
    num_mb_width = (native_width + 15) // 16            # number of horizontal macroblock in integer format
    num_mb_height = (native_height + 15) // 16          # number of vertical macroblock in integer format

    return num_mb_width, num_mb_height

# Global Model and Recurrent State Cache

_MODEL_CACHE = {
    "model" : None,
    "device" : None,
    "prev_hmap": None,
    "hx": None,
    "transform": transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

def _init_center_gaussian(batch_size=1, hmap_h=32, hmap_w=48, sigma=1.5):
    hmap = np.zeros((hmap_h, hmap_w), dtype=np.float32)
    hmap[hmap_h // 2, hmap_w // 2] = 1.0
    hmap = gaussian_filter(hmap, sigma=sigma)
    hmap /= hmap.sum()
    hmap_t = torch.from_numpy(hmap).unsqueeze(0).unsqueeze(0)
    return hmap_t.expand(batch_size, 1, -1, -1).clone()



def call_model(frame: np.ndarray, checkpoint_path: str = None, dt: float = 1) -> torch.Tensor:
    """
    Passes a video frame [H, W, C] (OpenCV BGR) through GazeLLNArch and returns the saliency heatmap.

    Args:
        frame: 3D numpy array [H, W, C] representing the raw video frame (BGR order from OpenCV).
        dt: The time delta between the current frame and the previous frame. 
        
    Returns:
        hmap: 3D torch.Tensor [1, mb_height: 32, mb_width: 48] representing the saliency map.
    """
    global _MODEL_CACHE

    if _MODEL_CACHE["model"] is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _MODEL_CACHE["device"] = device
        
        model = GazeLLNArch().to(device)
        
        # Determine path to DIEM_model.pt

        if checkpoint_path is None:
            ckpt_file = ROOT_DIR / "DIEM_model.pt"
        else:
            ckpt_file = Path(checkpoint_path)

        # Load state_dict weights
        if ckpt_file.exists():
            print(f"Loading model weights from {ckpt_file}")
            state_dict = torch.load(ckpt_file, map_location=device)
            model.load_state_dict(state_dict)
        else:
            print(f"Warning: {ckpt_file} not found. Running with un-trained weights.")
            
        model.eval()
        _MODEL_CACHE["model"] = model
        _MODEL_CACHE["prev_hmap"] = _init_center_gaussian().to(device)
        _MODEL_CACHE["hx"] = None

    model = _MODEL_CACHE["model"]
    device = _MODEL_CACHE["device"]

    # Frame Preprocessing

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = _MODEL_CACHE["transform"](rgb_frame).unsqueeze(0).to(device)

    # Model Inference

    with torch.no_grad():
        ts = torch.tensor([dt], device = device)

        # DIEM arch separates feature extraction from the recurrent forward pass.
        # extract_features() returns (B, 960) visual features from MobileNet.
        vis_features = model.extract_features(img_tensor)

        out_hmap, hx = model(vis_features, _MODEL_CACHE["prev_hmap"], _MODEL_CACHE["hx"], ts)
        # Update recurrent state cache for next frame
        _MODEL_CACHE["prev_hmap"] = out_hmap
        _MODEL_CACHE["hx"] = hx

        return out_hmap.squeeze(1).cpu()


def apply_spatial_smoothing(hmap: torch.Tensor, sigma = 0.0) -> torch.Tensor:
    # Apply Gaussian spatial smoothing. sigma=0 means no smoothing

    if sigma <= 0.0:
        print("No guassian smoothing applied")
        return hmap
    
    hmap_np = hmap.detach().cpu().numpy()
    # Apply smoothing only to the spatial dimensions (last two axes)
    sigmas = [0.0] * (hmap.ndim - 2) + [sigma, sigma]
    smoothed_np = gaussian_filter(hmap_np, sigma=sigmas).astype(np.float32)
    
    return torch.from_numpy(smoothed_np).to(hmap.device)

def normalize_heatmap(hmap, num_mb_width, num_mb_height) -> np.ndarray:
    """
    Prepares the QP map by min/max normalizing the qp values.

    Args:
        hmap: The input heatmap - Shape: (B, hmap_h, hmap_w)
        num_mb_width: The number of horizontal macroblocks
        num_mb_height: The number of vertical macroblocks
    """

    # Resizing the heatmap to match the macroblock grid
    resized_hmap = torch.nn.functional.interpolate(hmap.unsqueeze(1),
                                                    size=(int(num_mb_height), int(num_mb_width)),
                                                    mode='bilinear',
                                                    align_corners=False)
    # Shape: (B, 1, num_mb_height, num_mb_width) Ensure B = 1
    
    hmap_grid = resized_hmap.squeeze(1).squeeze(0)

    hmap_min = hmap_grid.min()
    hmap_max = hmap_grid.max()
    hmap_range = hmap_max - hmap_min
    
    if hmap_range > 1e-8:
        normalized_hmap = (hmap_grid - hmap_min) / hmap_range
    else:
        normalized_hmap = torch.zeros_like(hmap_grid)

    return normalized_hmap.cpu().numpy()                 # Removing the channel and batch dimension -> Shape: (num_mb_height, num_mb_width)

def map_to_qp_offsets(normalized: np.ndarray, qp_min: float, qp_max: float) -> np.ndarray:
    """
    Map normalized [0,1] saliency to QP offsets.

    - normalized=1.0 (high saliency) → qp_min (negative, preserve quality)
    - normalized=0.0 (low saliency)  → qp_max (positive, compress harder)

    Linear mapping: qp_offset = qp_max - (qp_max - qp_min) * normalized
    """
    qp_offsets = qp_max - (qp_max - qp_min) * normalized
    return np.clip(qp_offsets, qp_min, qp_max).astype(np.float32)



def main(sys_args=None):
    parser = argparse.ArgumentParser(
        description="Convert video frames to QP offset .bin files using a saliency model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--video_path",
        type=Path,
        default=None,
        help="Path to the video file. If specified, heatmaps will be generated frame-by-frame using the model.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory for output .bin QP offset files",
    )
    parser.add_argument(
        "--qp_min",
        type=float,
        default=-6.0,
        help="Min QP offset (applied to highest saliency). Default: -6.0",
    )
    parser.add_argument(
        "--qp_max",
        type=float,
        default=6.0,
        help="Max QP offset (applied to lowest saliency). Default: +6.0",
    )
    parser.add_argument(
        "--spatial_sigma",
        type=float,
        default=0.0,
        help="Gaussian blur sigma for spatial smoothing. 0=disabled. Default: 0.0",
    )
    parser.add_argument(
        "--temporal_alpha",
        type=float,
        default=1.0,
        help=(
            "EMA alpha for temporal smoothing (1.0=disabled, 0.0=full smoothing). "
            "new = alpha*current + (1-alpha)*previous. Default: 1.0"
        ),
    )

    args = parser.parse_args(sys_args)

    # Validate arguments
    if args.qp_min >= args.qp_max:
        parser.error(f"qp_min ({args.qp_min}) must be less than qp_max ({args.qp_max})")
    if not (0.0 <= args.temporal_alpha <= 1.0):
        parser.error(f"temporal_alpha must be in [0, 1], got {args.temporal_alpha}")
    if not args.video_path:
        parser.error("--video_path must be specified.")

    # Discover and load heatmaps
    heatmaps = []
    
        
    cap = cv2.VideoCapture(str(args.video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {args.video_path}")
    
    native_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    native_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        hm = call_model(frame)
        
        # Append torch.Tensor directly (already torch.float32)
        heatmaps.append(hm)
        frame_idx += 1
        
    cap.release()
    num_frames = len(heatmaps)
    print(f"Generated {num_frames} heatmaps using the model")

    mb_width, mb_height = mb_grid(native_width, native_height)
    print(f"Macroblock grid: {mb_width} x {mb_height} ({mb_width * mb_height} MBs/frame)")

    # Spatial smoothing 
    if args.spatial_sigma > 0:
        print(f"Applying spatial Gaussian smoothing (sigma={args.spatial_sigma})")
        heatmaps = [apply_spatial_smoothing(h, args.spatial_sigma) for h in heatmaps]

    # Temporal smoothing (EMA)
    if args.temporal_alpha < 1.0:
        alpha = args.temporal_alpha
        print(f"Applying temporal EMA smoothing (alpha={alpha})")
        smoothed = [heatmaps[0].clone()]

        for i in range(1, len(heatmaps)):
            s = alpha * heatmaps[i] + (1.0 - alpha) * smoothed[i - 1]
            smoothed.append(s.to(torch.float32))

        heatmaps = smoothed

    # Normalization
    print("Normalizing heatmaps")
    heatmaps = [normalize_heatmap(k, mb_width, mb_height) for k in heatmaps]

    # Map to QP offsets
    print(f"Mapping to QP offsets [{args.qp_min}, {args.qp_max}]")
    qp_offsets = [map_to_qp_offsets(h, args.qp_min, args.qp_max) for h in heatmaps]

    # Write output
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for i, qp in enumerate(qp_offsets):
        out_path = args.output_dir / f"qpoffset_{i:06d}.bin"
        # Flatten in row-major (C) order = raster scan
        qp.flatten(order="C").tofile(out_path)

    # Write metadata
    metadata = {
        "num_frames": num_frames,
        "mb_width": int(mb_width),
        "mb_height": int(mb_height),
        "num_mbs_per_frame": int(mb_width * mb_height),
        "bytes_per_frame": int(mb_width * mb_height * 4),  # float32
        "qp_min": args.qp_min,
        "qp_max": args.qp_max,
        "spatial_sigma": args.spatial_sigma,
        "temporal_alpha": args.temporal_alpha,
        "dtype": "float32",
        "order": "raster_scan_row_major",
    }
    meta_path = args.output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nWrote {num_frames} QP offset files to {args.output_dir}")
    print(f"Metadata: {meta_path}")

    # Print statistics
    all_offsets = np.concatenate([q.flatten() for q in qp_offsets])
    print(f"\nQP offset statistics:")
    print(f"  Min:    {all_offsets.min():.2f}")
    print(f"  Max:    {all_offsets.max():.2f}")
    print(f"  Mean:   {all_offsets.mean():.2f}")
    print(f"  Std:    {all_offsets.std():.2f}")
    print(f"  Median: {np.median(all_offsets):.2f}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        # Fallback default runner behavior 
        main([
            "--video_path", "sample_video.mp4",
            "--output_dir", "qp_dir",
            "--qp_min", "-6.0",
            "--qp_max", "6.0",
            "--spatial_sigma", "0.0",
            "--temporal_alpha", "1.0"
        ])
