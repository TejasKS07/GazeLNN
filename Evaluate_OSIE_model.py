import os
import sys
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

from OSIE_Modeldata import KLDTWLoss

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

def _pad_or_truncate(seq, max_len, pad_value):
    real_len = min(len(seq), max_len)
    mask = [True] * real_len + [False] * (max_len - real_len)
    seq = seq[:real_len] + [pad_value] * (max_len - real_len)
    return seq, mask

def _center_gaussian(batch_size, hmap_h, hmap_w, device, sigma=1.5):
    hmap = _gaussian_heatmap(
        x=hmap_w / 2, y=hmap_h / 2,
        img_w=hmap_w, img_h=hmap_h,
        hmap_w=hmap_w, hmap_h=hmap_h,
        sigma=sigma,
    )
    hmap_t = torch.from_numpy(hmap).to(device)
    return hmap_t.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1).clone()

from metrics import evaluate_metrics

class MITDataset(Dataset):
    """
    Dataset for loading MIT Low Resolution dataset from manifest.csv and .npy files.
    """
    def __init__(
        self,
        data_root: str,
        img_size=(256, 384),
        downsample: int = 8,
        min_len: int = 4,
        max_len: int = 8,
        sigma: float = 1.5,
        seed: int = 42,
    ):
        self.stimuli_dir = os.path.join(data_root, 'images')
        self.img_size = img_size
        self.hmap_size = (img_size[0] // downsample, img_size[1] // downsample)
        self.max_len = max_len
        self.sigma = sigma
        
        manifest_path = os.path.join(data_root, 'manifest.csv')
        df = pd.read_csv(manifest_path)
        
        self.samples = []
        
        for _, row in df.iterrows():
            img_name = os.path.basename(row['image_path'])
            npy_path = os.path.join(data_root, row['fixation_npy_path'])
            num_subjects = int(row['num_subjects'])
            try:
                fix_data = np.load(npy_path, allow_pickle=True)
                
                # fix_data is (N, 2) with ALL subjects' gaze data concatenated.
                # Split into per-subject blocks for proper scanpath evaluation.
                points_per_subj = len(fix_data) // num_subjects
                if points_per_subj == 0:
                    continue
                
                for s in range(num_subjects):
                    start_idx = s * points_per_subj
                    end_idx = start_idx + points_per_subj
                    subj_data = fix_data[start_idx:end_idx]
                    
                    # Extract fixation events (handles both raw gaze and fixation-level data)
                    fixations = MITDataset._process_subject_block(subj_data, max_len)
                    
                    if len(fixations) < min_len:
                        continue
                    
                    xs = fixations[:, 0].tolist()
                    ys = fixations[:, 1].tolist()
                    
                    # Δt: 0 for first fixation, 1.0 for rest (paper Section III-B:
                    # "during deployment, Δt is fixed to 1")
                    dts = [0.0] + [1.0] * (len(xs) - 1)
                    
                    xs, mask = _pad_or_truncate(xs, max_len, pad_value=xs[-1])
                    ys, _    = _pad_or_truncate(ys, max_len, pad_value=ys[-1])
                    dts, _   = _pad_or_truncate(dts, max_len, pad_value=1.0)

                    self.samples.append({
                        'img_name': img_name,
                        'fix_x':    xs,
                        'fix_y':    ys,
                        'fix_dt':   dts,
                        'mask':     mask,
                    })
            except Exception as e:
                print(f"Skipping {npy_path}: {e}")
                
        if len(self.samples) == 0:
            raise ValueError("No scanpaths were loaded! The structure of fix_data does not match expectations.")
                
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print(f"[MITDataset] Loaded {len(self.samples)} scanpath pairs.")

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _process_subject_block(points, max_len):
        """Process a subject's gaze data block into a fixation scanpath.
        
        Handles both raw gaze samples (consecutive near-duplicates that need
        fixation extraction) and fixation-level data (already sparse).
        
        Args:
            points: numpy array of shape (N, 2) with gaze/fixation coordinates
            max_len: maximum number of fixations to return
            
        Returns:
            numpy array of shape (M, 2) where M <= max_len
        """
        if len(points) <= max_len:
            return points
        
        # Check median inter-point distance to determine data type:
        # Raw gaze samples → very small distances (< 5px)
        # Fixation-level data → larger distances (saccade amplitudes)
        sample_size = min(50, len(points) - 1)
        diffs = np.linalg.norm(np.diff(points[:sample_size + 1], axis=0), axis=1)
        median_diff = np.median(diffs)
        
        if median_diff < 5.0:
            # Raw gaze samples — extract fixation events using velocity threshold
            fixations = [points[0].copy()]
            current_center = points[0].copy()
            n_in_fix = 1
            
            for pt in points[1:]:
                dist = np.linalg.norm(pt - current_center)
                if dist > 20.0:  # Saccade threshold in pixels
                    fixations.append(pt.copy())
                    current_center = pt.copy()
                    n_in_fix = 1
                    if len(fixations) >= max_len:
                        break
                else:
                    # Update running average of current fixation center
                    n_in_fix += 1
                    current_center += (pt - current_center) / n_in_fix
            
            return np.array(fixations[:max_len])
        else:
            # Already fixation-level data — take first max_len
            return points[:max_len]

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_H, img_W = self.img_size
        hmap_H, hmap_W = self.hmap_size

        img_path = os.path.join(self.stimuli_dir, sample['img_name'])
        img = Image.open(img_path).convert('RGB')
        orig_W, orig_H = img.size

        img_tensor = self.transform(img)

        heatmap_seq = []
        for i in range(self.max_len):
            if sample['mask'][i]:
                hmap = _gaussian_heatmap(
                    x=sample['fix_x'][i], y=sample['fix_y'][i],
                    img_w=orig_W, img_h=orig_H,
                    hmap_w=hmap_W, hmap_h=hmap_H,
                    sigma=self.sigma,
                )
            else:
                hmap = np.zeros((hmap_H, hmap_W), dtype=np.float32)
            heatmap_seq.append(hmap)

        heatmap_seq = torch.from_numpy(np.stack(heatmap_seq, axis=0))
        dt_seq = torch.tensor(sample['fix_dt'], dtype=torch.float32)
        dt_seq[0] = 0.0
        mask = torch.tensor(sample['mask'], dtype=torch.bool)
        fix_coords = torch.tensor(list(zip(sample['fix_x'], sample['fix_y'])), dtype=torch.float32)
        
        # Scale GT coordinates from original image space to model space (384×256)
        # so they are in the same coordinate system as model predictions
        scale_x = float(img_W) / float(orig_W)
        scale_y = float(img_H) / float(orig_H)
        fix_coords[:, 0] *= scale_x
        fix_coords[:, 1] *= scale_y

        return img_tensor, heatmap_seq, dt_seq, mask, fix_coords

def build_mit_dataloader(data_root, batch_size=8, num_workers=4):
    dataset = MITDataset(data_root=data_root)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)

def eval_epoch_with_metrics(model, loader, criterion, device):
    model.eval()
    with torch.no_grad():
        total_loss = 0.0
        all_metrics = {
            "Levenshtein": [], "ScanMatch": [], "Hausdorff": [],
            "Frechet": [], "FastDTW": [], "TDE": []
        }

        for imgs, hmap_seq, dt_seq, mask, fix_coords in tqdm(loader, desc="Validating", leave=False):
            imgs       = imgs.to(device)
            hmap_seq   = hmap_seq.to(device)
            mask       = mask.to(device)
            dt_seq     = dt_seq.to(device)
            fix_coords = fix_coords.to(device)

            B, T, hmap_H, hmap_W = hmap_seq.shape

            has_extract = hasattr(model, 'extract_features')
            if has_extract:
                vis_features = model.extract_features(imgs)
            
            prev_hmap = _center_gaussian(B, hmap_H, hmap_W, device=device, sigma=1.5)
            hx = None

            predictions = []
            pred_scanpaths = [[] for _ in range(B)]

            for t in range(T):
                ts = dt_seq[:, t].view(-1, 1)
                
                if has_extract:
                    out_hmap, hx = model(vis_features, prev_hmap, hx, ts)
                else:
                    out_hmap, hx = model(imgs, prev_hmap, hx, ts)
                
                predictions.append(out_hmap.squeeze(1))
                prev_hmap = out_hmap
                
                # Extract predicted coordinate
                for b in range(B):
                    if mask[b, t]:
                        heatmap = out_hmap[b, 0].cpu().numpy()
                        y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                        orig_x = float(x * (384 / hmap_W))
                        orig_y = float(y * (256 / hmap_H))
                        
                        # Avoid redundant consecutive predicted points
                        # Stop Removing deduplication
                        # if len(pred_scanpaths[b]) == 0 or pred_scanpaths[b][-1] != [orig_x, orig_y]:
                        #     pred_scanpaths[b].append([orig_x, orig_y])
                        pred_scanpaths[b].append([orig_x, orig_y])

            predictions = torch.stack(predictions, dim=1)
            loss = criterion(predictions, hmap_seq, mask)
            total_loss += loss.item()
            
            # Ground truth scanpath extraction using exact coordinates
            for b in range(B):
                gt_scanpath = []
                for t in range(T):
                    if mask[b, t]:
                        x_val = float(fix_coords[b, t, 0].cpu())
                        y_val = float(fix_coords[b, t, 1].cpu())
                        gt_scanpath.append([x_val, y_val])
                
                if len(pred_scanpaths[b]) > 0 and len(gt_scanpath) > 0:
                    metrics_res = evaluate_metrics(np.array(pred_scanpaths[b]), np.array(gt_scanpath))
                    for k, v in metrics_res.items():
                        all_metrics[k].append(v)

        print("\nEvaluation Metrics:")
        for k, v in all_metrics.items():
            avg_val = sum(v) / len(v) if len(v) > 0 else 0
            print(f"{k}: {avg_val:.4f}")
            
        return total_loss / len(loader)

if __name__ == "__main__":
    print("Starting MIT Dataset Evaluation...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    current_dir = Path(__file__).resolve().parent
    file_path = current_dir / "OSIE_model.pt"

    # To get the metric values from the Finetuned Model
    # file_path = current_dir / "DIEM" / "DIEM_model.pt"

    if not file_path.exists():
        print(f"Error: Could not find file at {file_path}")
        sys.exit(1)
    
    # Load state dict first to detect which architecture was used
    state_dict = torch.load(file_path, map_location=device)
    
    # Detect architecture from actual state dict keys, not filename.
    # DIEM arch has coordconv_in + coordconv_out; OSIE arch has a single coordconv.
    has_diem_keys = any("coordconv_in" in k for k in state_dict.keys())
    
    if has_diem_keys:
        print("Detected DIEM architecture weights (coordconv_in/coordconv_out).")
        from DIEM.DIEM_Modeldata import GazeLLNArch
    else:
        print("Detected OSIE architecture weights (single coordconv).")
        from OSIE_Modeldata import GazeLLNArch

    model = GazeLLNArch().to(device)
    model.load_state_dict(state_dict)
    print(f"Model loaded successfully from {file_path}")
    
    # Build dataloader
    # The dataset root is the mit_lowres_highres_only folder
    data_root = "mit_lowres_highres_only"
    loader = build_mit_dataloader(data_root, batch_size=8, num_workers=0) # using num_workers=0 to avoid windows multiprocessing issues
    
    # Define criterion
    criterion = KLDTWLoss()
    
    # Run evaluation
    eval_epoch_with_metrics(model, loader, criterion, device)
