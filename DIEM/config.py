"""
Config for DIEM preprocessing.

Adjust DIEM_ROOT and the glob patterns to match whatever directory structure
you get after unzipping the Mediafire archive -- DIEM ships as one folder per
video, each containing the stimulus video and a set of per-subject gaze files.
The exact filenames vary a bit by release, so GAZE_FILE_GLOB is a pattern you
may need to tweak once you've unzipped things and looked at what's actually
inside a video folder (e.g. it might be "*.txt", "event_data/*.txt", etc.).
"""

from pathlib import Path

# Root folder containing one subdirectory per DIEM video after extraction.
# Confirmed real layout (from 50_people_brooklyn_no_voices_1280x720.7z):
#   <DIEM_ROOT>/<video_name>/event_data/*.txt   (one file per subject)
#   <DIEM_ROOT>/<video_name>/video/*.mp4
#   <DIEM_ROOT>/<video_name>/audio/*.wav        (not used here)
DIEM_ROOT = Path("DIEM_ROOT")

# Where processed (parsed fixations, heatmaps, extracted frames) get written.
OUTPUT_ROOT = Path("DIEM_Processed")

# Glob (relative to each video's folder) matching one file per subject.
GAZE_FILE_GLOB = "event_data/*.txt"

# Where the stimulus video lives within each video's folder.
VIDEO_FILE_GLOB = "video/*.mp4"

# Recording frame rate DIEM eye-tracking data is sampled at (fixed, per DIEM docs
# and confirmed via ffprobe on the brooklyn sample: 30/1 fps, duration matches
# gaze-file row count closely).
GAZE_FPS = 30.0

# Column layout per DIEM docs:
# frame, left_x, left_y, left_dil, left_event, right_x, right_y, right_dil, right_event
COL_FRAME = 0
COL_LEFT_X = 1
COL_LEFT_Y = 2
COL_LEFT_DIL = 3
COL_LEFT_EVENT = 4
COL_RIGHT_X = 5
COL_RIGHT_Y = 6
COL_RIGHT_DIL = 7
COL_RIGHT_EVENT = 8

FIXATION_FLAG = 1
SACCADE_FLAG = 2
BLINK_FLAG = 0
DROPPED_FLAG = -1

# NOTE: screen-resolution handling has moved to the CALIBRATION_MODE block
# below -- native video resolution is now auto-detected per video (ffprobe,
# with folder-name-suffix as fallback) rather than assumed from a single
# global constant, since it turned out NOT to reliably equal the eye
# tracker's screen coordinate space for every video (see that section).

# Target resolution to resize video frames to for model input, matching the
# GazeLNN paper's convention (H x W).
MODEL_INPUT_HEIGHT = 256
MODEL_INPUT_WIDTH = 384

# Spatial downsample factor from image resolution to heatmap grid resolution,
# matching GazeLLNArch (hmap_h = image_h // S, hmap_w = image_w // S). This is
# an architecture constant, not a preprocessing free parameter -- must match
# whatever S the model was actually built with.
S = 8
HMAP_HEIGHT = MODEL_INPUT_HEIGHT // S   # 32
HMAP_WIDTH = MODEL_INPUT_WIDTH // S     # 48

# Gaussian sigma, in HEATMAP-GRID pixels (i.e. at HMAP_HEIGHT x HMAP_WIDTH
# resolution, not full image resolution) -- matches the existing OSIE
# preprocessing's _gaussian_heatmap(sigma=1.5) convention, so fine-tuning
# heatmap statistics stay comparable to what the model was originally trained
# on. Retune only with a clear reason (e.g. if DIEM's viewing behavior turns
# out to need a wider/narrower blob than OSIE's static-image fixations).
HEATMAP_SIGMA = 1.5

# --- Screen-space calibration --------------------------------------------
# IMPORTANT: gaze (x, y) values in DIEM's raw files are in the eye tracker's
# SCREEN coordinate space, which is NOT guaranteed to equal a given video's
# native resolution. Confirmed empirically: for 50_people_brooklyn (native
# 1280x720), ~100% of fixation-flagged samples fall within [0,1280)x[0,720).
# For ami_ib4010_closeup (native 720x576), only ~55-60% of samples fall
# within [0,720)x[0,576) even after excluding blinks/dropped samples, and the
# shortfall is spread smoothly across nearly all subjects (15%-90%+ per
# subject) rather than concentrated in a few bad-quality recordings -- i.e.
# it looks like a systematic screen/video coordinate offset (e.g. the 4:3
# PAL video letterboxed/pillarboxed within a larger, fixed recording display)
# rather than random tracking noise. This matches an independent report of
# the same symptom (fixations appearing shifted relative to the video frame)
# from another user on the DIEM project's own site -- so treat it as a real,
# recurring property of this dataset, not a one-off preprocessing bug.
#
# No authoritative per-video calibration/display-resolution table is
# published alongside the raw archives (checked the project site and several
# papers citing DIEM; none give exact screen dimensions for public reuse).
# In the absence of that, CALIBRATION_MODE="auto_percentile" below estimates
# a per-video affine (scale + offset) mapping screen coords -> video pixel
# coords, by aligning the pooled (all-subjects) fixation coordinate
# distribution's [CALIB_LOW_PCT, CALIB_HIGH_PCT] percentile range to the
# video's [0, native_w] x [0, native_h] extent. This assumes viewers'
# fixations roughly span the full visible video content at the population
# level, which is a reasonable approximation for free-viewing but IS a
# heuristic -- sanity-check the result (e.g. via the debug overlay in
# build_heatmaps.py) before trusting it for a video you haven't spot-checked.
#
# TESTED ON BOTH REAL VIDEOS AVAILABLE SO FAR, WITH A CLEAR RESULT:
#   - brooklyn (1280x720): identity mapping -> 0.5% of fixations fall outside
#     frame (excellent). auto_percentile calibration, applied anyway,
#     INCREASES this to 3.9%, because it force-stretches the (legitimately!)
#     center-clustered gaze distribution to fill the frame edge-to-edge --
#     people fixating on a face in the middle of frame is normal viewing
#     behavior, not evidence of a coordinate offset needing correction.
#   - ami_ib4010_closeup (720x576): identity mapping -> 27.6% fall outside
#     frame, and the pooled median sits near the frame's corner rather than
#     center. Visual inspection (inspect_calibration.py, overlaying identity-
#     mapped fixations on actual frames) shows the in-frame points DO land on
#     people's faces around the table, not on blank background -- so this
#     looks like a genuinely noisier recording (more blinks/tracking loss),
#     not a real coordinate-space bug requiring correction either.
#
# Conclusion: "auto_percentile" is NOT a safe default -- it actively hurts
# well-behaved videos by assuming every video's gaze spans edge-to-edge,
# which is false for typical face/subject-centric footage (i.e. most of
# DIEM). Default is therefore "identity". Only reach for "auto_percentile" or
# "manual" on a SPECIFIC video after using inspect_calibration.py yourself to
# visually confirm identity mapping looks wrong for it (fixations landing on
# blank margins / clearly missing the visible subject) -- don't apply either
# mode dataset-wide without that per-video check.
CALIBRATION_MODE = "identity"  # "identity" | "auto_percentile" | "auto_recenter" | "manual"

# --- Per-video overrides for CALIBRATION_MODE ------------------------------
# Tested on 6 real DIEM videos so far. Result: videos with native height 576
# (both cases checked: ami_ib4010_closeup_720x576, news_sherry_drinking_mice
# _768x576) show identity-mapping drop rates of 28-31%, MUCH higher than
# every non-576-tall video checked (0.5-5.5%). In both 576-tall cases the
# pooled fixation median sits near the frame's bottom-right corner rather
# than center -- and critically, a PURE TRANSLATION (recenter the pooled
# median to the frame center, no rescaling) brings both to ~99% in-bounds,
# with a remarkably consistent required y-offset (-221 vs -230 px) across
# two otherwise-unrelated videos (a corporate meeting closeup and a news
# montage). That consistency is the key evidence this is a genuine shared
# calibration/recording-setup difference for whatever batch produced the
# 576-tall videos -- not per-video content-driven gaze clustering (which
# "auto_percentile"'s aggressive edge-to-edge stretch would wrongly assume,
# and which coincidentally-identical corner clustering across two unrelated
# videos would not otherwise explain).
#
# "auto_recenter" (translation-only, via pooled median -> frame center) is
# therefore added as a THIRD calibration mode, safer than "auto_percentile"
# because it doesn't distort the fixation distribution's spread/shape, only
# its position -- appropriate when the evidence points to an offset bug
# specifically, as it does here, rather than an unknown scale mismatch.
#
# Working rule of thumb for NEW videos you add: if a video's printed
# drop_fraction under "identity" is high (>15%) AND its native height is
# 576 (or generally, whenever you see a corner-skewed pooled median rather
# than a plausibly content-driven off-center-but-not-cornered one), try
# "auto_recenter" for that video specifically and re-check with
# inspect_calibration.py before trusting it -- don't apply it dataset-wide.
CALIBRATION_MODE_OVERRIDES = {
    "ami_ib4010_closeup_720x576": "auto_recenter",
    "news_sherry_drinking_mice_768x576": "auto_recenter",
}
CALIB_LOW_PCT = 1.0
CALIB_HIGH_PCT = 99.0

# Manual per-video overrides: (scale_x, scale_y, offset_x, offset_y) mapping
# raw_x * scale_x + offset_x -> video pixel x (same for y). Only consulted
# when CALIBRATION_MODE == "manual". Fill these in if/when you obtain ground
# truth display parameters (e.g. from the original Mital et al. 2011 paper's
# Methods section, if it turns out to specify them).
MANUAL_CALIBRATION = {
    # "ami_ib4010_closeup_720x576": dict(scale_x=1.0, scale_y=1.0, offset_x=0.0, offset_y=0.0),
}

# Rolling window settings for continuous (Option B) training.
WINDOW_LEN_FRAMES = 24      # ~0.8s of context at 30fps
WINDOW_STRIDE_FRAMES = 12   # 50% overlap

# Minimum number of valid (non-dropped) subjects required for a frame to be
# included in the aggregated ground-truth heatmap.
MIN_SUBJECTS_PER_FRAME = 3
