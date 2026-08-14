# DIEM preprocessing pipeline for GazeLNN fine-tuning (Option B / continuous)

Verified end-to-end against **six** real DIEM videos spanning five distinct
native resolutions and content types (interviews, meeting-room closeup,
wildlife documentary, movie trailer, news montage):

| video | resolution | subjects | identity-mode drop rate |
|---|---|---|---|
| 50_people_brooklyn | 1280x720 | 44 | 0.5% |
| BBC_wildlife_serpent | 1280x704 | 51 | 0.5% |
| documentary_adrenaline_rush | 1280x720 | 123 | 4.1% |
| harry_potter_6_trailer | 1280x544 | 50 | 5.5% |
| ami_ib4010_closeup | 720x576 | 44 | **27.6%** |
| news_sherry_drinking_mice | 768x576 | 166 | **30.9%** |

Confirmed facts baked into `config.py`:

- Archive layout: `<video_name>/event_data/*.txt` (one file per subject),
  `<video_name>/video/*.mp4`, `<video_name>/audio/*.wav` -- consistent
  across all six videos checked, including a 166-subject video with
  multi-session filename prefixes (`diem1s..` through `diem4s..` in one
  folder) and disambiguated duplicate stems (e.g. `diem4s04b`) -- subjects
  are keyed by filename stem so this caused no collisions.
- Gaze file format, 30fps, and event-flag convention: identical across all
  six.
- Heatmap resolution/convention matches your `GazeLLNArch` exactly:
  `(image_h // S, image_w // S)` = `(32, 48)` by default, sigma=1.5 in that
  grid's own coordinate space, same as your existing OSIE `_gaussian_heatmap`.

## The 576-height pattern (generalizes cleanly, worth knowing about)

Both videos with **native height 576** (`ami_ib4010_closeup_720x576`,
`news_sherry_drinking_mice_768x576`) showed identity-mapping drop rates of
28-31%, far above every other video (0.5-5.5%). Two things distinguish this
from ordinary per-video noise:

1. In both cases the pooled fixation median sits near the frame's
   bottom-right *corner*, not just generically off-center.
2. A **pure translation** (recentering the pooled median to the frame
   center, no rescaling) brings both to ~99% in-bounds, with a strikingly
   consistent required y-offset (-221px vs -230px) across two otherwise
   unrelated videos (a corporate meeting recording and a news montage).

That consistency across unrelated content is the key evidence this is a
genuine shared calibration/recording-setup difference for whatever batch
produced the 576-tall videos, not coincidental content-driven gaze
clustering. A new `CALIBRATION_MODE = "auto_recenter"` (translation-only,
safer than the older `"auto_percentile"` because it doesn't distort the
fixation distribution's spread) is now applied to both via
`config.CALIBRATION_MODE_OVERRIDES`.

**Working rule of thumb for new videos**: if a video's printed
`drop_fraction` under `identity` is high (>15%) and its native height is
576, `auto_recenter` is a good first thing to try, then re-verify with
`inspect_calibration.py`. This is a pattern, not a certainty -- keep
checking new videos individually rather than assuming every 576-tall video
needs this exact fix.

## Why `"auto_percentile"` remains a bad default (confirmed again)

Re-tested against all 6 videos: `auto_percentile` continues to make already-
good videos worse (e.g. brooklyn's drop rate goes from 0.5% to ~3.9%) by
force-stretching legitimately center-clustered gaze to fill the frame
edge-to-edge. Kept in `config.py` as an available mode, but `"identity"`
remains the global default, with `"auto_recenter"` as a validated,
non-distorting override for the two videos it's currently confirmed to fix.

## Pipeline order

```bash
# 1. Unzip/7z-extract each DIEM video's archive under DIEM_ROOT, e.g.:
#    /data/diem_raw/50_people_brooklyn_no_voices_1280x720/{event_data,video,audio}/...
#    (repeat for every video you download)

# 2. Parse raw per-subject gaze files -> per-subject fixation events (JSON)
python parse_gaze.py

# 3. Aggregate fixations across subjects -> per-frame heatmaps at the
#    model's actual (Hg, Wg) grid resolution (.npz). Watch the printed
#    drop_fraction per video -- anything >15% gets flagged automatically.
python build_heatmaps.py

# 4. For any newly-flagged video (or periodically, to spot-check "clean"
#    ones too), visually confirm calibration before trusting its heatmaps:
python inspect_calibration.py <video_name> --frames 5
#    -> writes overlay PNGs to /tmp/calibration_debug/<video_name>/
#    Green = fixation landed in-frame, red = dropped. Do the green points
#    land on plausible gaze targets (faces, moving subjects)? If a video
#    needs correction, add it to config.CALIBRATION_MODE_OVERRIDES (try
#    "auto_recenter" first, "manual" if you later obtain true calibration
#    values, "auto_percentile" only with extra scrutiny) and re-run step 3.

# 5. Extract + resize video frames to match model input resolution (needs ffmpeg)
python extract_frames.py

# 6. Use dataset.py in your training script:
```

```python
from dataset import build_datasets, VideoOrderedBatchSampler, get_prev_hmaps
from torch.utils.data import DataLoader

train_ds, val_ds, test_ds = build_datasets()
train_loader = DataLoader(train_ds, batch_sampler=VideoOrderedBatchSampler(train_ds, videos_per_batch=4))

for batch in train_loader:
    prev_hmap = get_prev_hmaps(batch["heatmaps"], batch["is_video_start"])
    # prev_hmap[:, 0] is a uniform prior when batch["is_video_start"] is True;
    # for windows that continue a video, carry the true previous heatmap
    # over from the prior window's last step yourself in the training loop.
    # model.forward(image=batch["frames"][:, t], prev_hmap=prev_hmap[:, t], hx=hx, ts=batch["delta_t"][:, t])
    ...
```

Confirmed on the full 6-video set: `build_datasets()` auto-discovers all
videos, splits at the video level (4 train / 1 val / 1 test with the default
15%/15% split on 6 videos), and `DIEMWindowDataset` + `VideoOrderedBatchSampler`
produce correctly-shaped, correctly-ordered windows across the whole set.

Each batch item has: `frames (T,3,H,W)`, `heatmaps (T,Hg,Wg)` (matches
`GazeLLNArch`'s own grid resolution), `valid (T,)` bool mask (exclude
invalid frames from the loss), `delta_t (T,)` (= 1/30s per step), and
`is_video_start` (reset the CfC hidden state when True, otherwise carry it
forward from the previous window of the same video and detach before
backprop -- truncated-BPTT stitching, done in your training loop).

## Automated Pipeline Execution

You can run the full sequence of steps automatically using the provided `run_diem_pipeline.py` script. It will sequentially run:
1. `parse_gaze.py`
2. `build_heatmaps.py`
3. `extract_frames.py`
4. Automatically read `calibration_report.csv`, extract all videos that do **not** need review (`needs_review == "no"`), and run `inspect_calibration.py` on them.
5. Execute `fine_tune.py`.

```bash
python run_diem_pipeline.py
```

## Running on the full DIEM dataset (all ~85 videos)

**1. Directory setup.** Extract every video's archive so `DIEM_ROOT` looks
like:
```
DIEM_ROOT/
  50_people_brooklyn_no_voices_1280x720/{event_data,video,audio}/
  ami_ib4010_closeup_720x576/{event_data,video,audio}/
  ... one folder per video ...
```
No renaming needed -- folder names become the `video_name` used everywhere
downstream (heatmap files, frame dirs, dataset splits).

**2. Config changes.** In `config.py`, set:
```python
DIEM_ROOT = Path("/your/actual/path/diem_raw")
OUTPUT_ROOT = Path("/your/actual/path/diem_processed")
```
That's it for a first pass -- everything else (calibration, sigma, window
size) has working defaults. `CALIBRATION_MODE_OVERRIDES` will need entries
added incrementally as you discover more problem videos (see step 4).

**3. Run in order, once, over the whole directory:**
```bash
python parse_gaze.py          # all videos, ~seconds-minutes depending on subject count
python build_heatmaps.py      # all videos, writes calibration_report.csv
python extract_frames.py      # all videos -- SLOW, see note below
```
Each script auto-discovers every video under its input directory and loops
over all of them -- no per-video invocation needed for a first pass.

**4. Triage calibration across all videos** using the report instead of
scrolling console output:
```bash
sort -t, -k5 -rn --skip-line /data/diem_processed/calibration_report.csv | head -20
# or: awk -F, '$6=="yes"' /data/diem_processed/calibration_report.csv
```
For each flagged video: `python inspect_calibration.py <video_name>`, look
at the overlay PNGs, then either accept it as noisy-but-fine, or add it to
`CALIBRATION_MODE_OVERRIDES` in `config.py` (try `"auto_recenter"` first per
the 576-height pattern above) and re-run just that video:
```bash
python build_heatmaps.py --videos <video_name>
```
This updates only that video's `.npz` and merges its row into the existing
`calibration_report.csv` without touching the other videos.

**5. Re-running efficiently.** All three scripts accept `--videos
name1,name2` (restrict to specific videos) and `--skip-existing` (skip
videos already processed) -- use these once you're iterating rather than
re-processing all ~85 videos from scratch every time, especially for
`extract_frames.py`, which is the slow, disk-heavy step (~2,700 PNG files
per 90-second clip at 384x256 -- the full dataset will be many tens of GB).
A reasonable full-dataset workflow:
```bash
python parse_gaze.py --skip-existing
python build_heatmaps.py --skip-existing   # first pass; then re-run flagged videos individually via --videos
python extract_frames.py --skip-existing
```

## What you end up with

```
OUTPUT_ROOT/
  fixations/<video_name>/<subject>.json       # per-subject fixation events, screen coords
  heatmaps/<video_name>.npz                   # heatmaps (T,32,48) + valid (T,) mask, per video
  frames/<video_name>/frame_000001.png ...    # resized 384x256 RGB frames, 1-indexed
  calibration_report.csv                      # one row per video: mode, resolution, drop_fraction, needs_review
```

From there, `dataset.build_datasets()` scans `heatmaps/` and `frames/`,
keeps only videos that have both, and returns three `DIEMWindowDataset`s
split at the video level (default 70/15/15 train/val/test). Feed those into
a `DataLoader` with `VideoOrderedBatchSampler` as shown above, and each
batch is ready to go straight into `GazeLLNArch.forward()` (via
`get_prev_hmaps()` for the teacher-forcing input) for fine-tuning.



1. **`config.DIEM_ROOT` / `config.OUTPUT_ROOT`** -- currently placeholders,
   point them at your actual paths.
2. **Run `inspect_calibration.py` on new videos**, especially any with
   native height 576 or other resolutions not yet checked -- the pattern
   above is a useful prior, not a guarantee, for videos outside the six
   tested.
3. **`HEATMAP_SIGMA` / `S`** -- pinned to match your notebook's existing
   OSIE convention (sigma=1.5, S=8) for architecture compatibility. Don't
   change these independently of the model itself.
4. **`WINDOW_LEN_FRAMES` / `WINDOW_STRIDE_FRAMES`** -- current defaults
   (24-frame windows, 50% overlap = 0.8s of context at 30fps) are a
   starting point, not a tuned value.
5. **Subject-count filtering** (`MIN_SUBJECTS_PER_FRAME`) -- check the
   `valid.mean()` printed by `build_heatmaps.py` per video.
6. **fps mismatch with your existing video pipeline**: your notebook's
   `resample_by_pts`/`SinglePassVideoPipeline` code defaults to
   `target_fps=24`, but DIEM is recorded/delivered at 30fps across all six
   videos checked. Decide explicitly whether you train at DIEM's native
   30fps (simplest, what this pipeline assumes) and only resample at the
   final foveation-encoding stage, or resample training data down to 24fps
   for consistency.

## Known limitations / things I couldn't verify without more data

- Six DIEM videos were available to test against, covering five
  resolutions. File-naming, columns, and fps are confirmed consistent
  across all six; calibration behavior clearly correlates with native
  height (576 vs not) in the two data points available for that pattern,
  which is suggestive but not proof it holds for every 576-tall video in
  the full dataset -- keep spot-checking new videos, especially at other
  resolutions not yet seen (e.g. anything other than 720/704/544/576-tall).
- No authoritative published table of DIEM's recording-display parameters
  was found (checked the project site, an independent user's identical
  complaint about the harry_potter_6_trailer video specifically, and
  papers citing DIEM); the calibration handling here is validated
  empirically against real data, not against ground-truth documentation.
- Frame-count mismatches between the gaze file's last row and the video's
  actual frame count (off by ~1-2 frames at the tail) are handled by
  truncating/padding in the dataset, not frame-accurate alignment -- fine
  for training, worth knowing if debugging subtle misalignment later.


