#!/usr/bin/env python3
"""
plot_bdrate.py

Generate a BD-rate RD-curve plot (VMAF vs. bitrate) from results_detailed.csv.

The BD-rate computation reuses the exact same polynomial fitting approach as
evaluate_compression.py:
    log(rate) = f(vmaf)   via np.polyfit, degree = max(1, min(3, n_points - 1))
    BD-rate %  = (exp(avg_log_rate_diff over overlapping quality range) - 1) * 100

Usage:
    python plot_bdrate.py --csv results_detailed.csv
    python plot_bdrate.py --csv results_detailed.csv --clip sample_video
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# Data loading

def load_csv(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def filter_clip(rows: list[dict], clip_name: str | None) -> tuple[str, list[dict]]:
    """Return (clip_name, filtered_rows).  Auto-detect if only one clip exists."""
    clips = sorted(set(r["clip"] for r in rows))
    if clip_name is None:
        if len(clips) == 1:
            clip_name = clips[0]
        else:
            print(f"Error: CSV contains multiple clips {clips}.  "
                  f"Specify one with --clip.", file=sys.stderr)
            sys.exit(1)
    filtered = [r for r in rows if r["clip"] == clip_name]
    if not filtered:
        print(f"Error: no rows found for clip '{clip_name}'.", file=sys.stderr)
        sys.exit(1)
    return clip_name, filtered


def split_methods(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """Split into uniform (anchor) and gaze (test) rows, sorted by bitrate."""
    uniform = sorted(
        [r for r in rows if r["method"] == "uniform"],
        key=lambda r: float(r["bitrate_kbps"]),
    )
    gaze = sorted(
        [r for r in rows if r["method"] == "gaze"],
        key=lambda r: float(r["bitrate_kbps"]),
    )
    return uniform, gaze


def extract_rd(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Return (bitrate_kbps, vmaf) arrays from a list of row dicts."""
    rates = np.array([float(r["bitrate_kbps"]) for r in rows])
    quals = np.array([float(r["vmaf_global"]) for r in rows])
    return rates, quals


# BD-rate fitting (identical math to evaluate_compression.py)

def fit_degree(n_points: int) -> int:
    return max(1, min(3, n_points - 1))


def fit_lograte_vs_quality(rates: np.ndarray, quals: np.ndarray) -> np.poly1d:
    """Fit log(rate) = f(quality) polynomial."""
    log_r = np.log(rates)
    deg = fit_degree(len(rates))
    coeffs = np.polyfit(quals, log_r, deg)
    return np.poly1d(coeffs)


def compute_bdrate(rate_anchor, qual_anchor, rate_test, qual_test):
    """BD-rate %: negative means test saves bits at equal quality."""
    rate_anchor = np.asarray(rate_anchor, float)
    qual_anchor = np.asarray(qual_anchor, float)
    rate_test = np.asarray(rate_test, float)
    qual_test = np.asarray(qual_test, float)

    if len(rate_anchor) < 2 or len(rate_test) < 2:
        return None, "insufficient RD points (need >=2 per method)"

    log_r_a, log_r_t = np.log(rate_anchor), np.log(rate_test)
    deg_a = fit_degree(len(rate_anchor))
    deg_t = fit_degree(len(rate_test))

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
    note = ""
    if len(rate_anchor) < 4 or len(rate_test) < 4:
        note = (f"low-confidence ({len(rate_anchor)}/{len(rate_test)} "
                f"RD points; >=4 recommended)")
    return float(pct), note


# Plotting

def plot_bdrate(
    clip_name: str,
    rate_u: np.ndarray, qual_u: np.ndarray,
    rate_g: np.ndarray, qual_g: np.ndarray,
    out_dir: Path,
):
    # Fit both curves
    poly_u = fit_lograte_vs_quality(rate_u, qual_u)
    poly_g = fit_lograte_vs_quality(rate_g, qual_g)

    # Overlapping quality range
    lo = max(qual_u.min(), qual_g.min())
    hi = min(qual_u.max(), qual_g.max())
    if hi <= lo:
        print("Error: no overlapping VMAF quality range between uniform and "
              "gaze methods — cannot shade BD-rate region or compute BD-rate.",
              file=sys.stderr)
        return

    # BD-rate
    bdrate_pct, bdrate_note = compute_bdrate(rate_u, qual_u, rate_g, qual_g)

    # Dense quality arrays for smooth curves
    q_dense_u = np.linspace(qual_u.min(), qual_u.max(), 300)
    q_dense_g = np.linspace(qual_g.min(), qual_g.max(), 300)
    r_dense_u = np.exp(poly_u(q_dense_u))
    r_dense_g = np.exp(poly_g(q_dense_g))

    # Overlap region for shading
    q_shade = np.linspace(lo, hi, 300)
    r_shade_u = np.exp(poly_u(q_shade))
    r_shade_g = np.exp(poly_g(q_shade))

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Fitted curves (full range of each method)
    ax.plot(r_dense_u, q_dense_u, color="#3366CC", linewidth=2,
            label="Uniform (anchor) — fit", zorder=3)
    ax.plot(r_dense_g, q_dense_g, color="#DC3912", linewidth=2,
            label="Gaze-guided (test) — fit", zorder=3)

    # Scatter: raw data points
    ax.scatter(rate_u, qual_u, marker="o", s=60, edgecolors="#3366CC",
               facecolors="white", linewidths=1.8, zorder=4,
               label="Uniform data points")
    ax.scatter(rate_g, qual_g, marker="^", s=70, edgecolors="#DC3912",
               facecolors="white", linewidths=1.8, zorder=4,
               label="Gaze data points")

    # Shade overlap region
    ax.fill_betweenx(q_shade, r_shade_u, r_shade_g,
                     color="#FFD700", alpha=0.25, zorder=1,
                     label="BD-rate region")

    # Horizontal dashed lines at overlap boundaries
    ax.axhline(lo, color="grey", linestyle="--", linewidth=0.7, alpha=0.6)
    ax.axhline(hi, color="grey", linestyle="--", linewidth=0.7, alpha=0.6)

    # BD-rate annotation
    if bdrate_pct is not None:
        sign = "−" if bdrate_pct < 0 else "+"
        annotation = f"BD-rate (VMAF): {sign}{abs(bdrate_pct):.2f}%"
        if bdrate_note:
            annotation += f"\n({bdrate_note})"
        # Place annotation roughly in the middle of the shaded region
        mid_q = (lo + hi) / 2
        mid_r = np.exp((poly_u(mid_q) + poly_g(mid_q)) / 2)
        ax.annotate(
            annotation,
            xy=(mid_r, mid_q),
            fontsize=10, fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="grey",
                      alpha=0.9),
            zorder=5,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Bitrate (kbps)", fontsize=11)
    ax.set_ylabel("VMAF", fontsize=11)
    ax.set_title(f"BD-Rate RD Curve — {clip_name}", fontsize=13, fontweight="bold")

    # Readable log-scale tick labels (e.g. 200, 500, 1000, 2000 …)
    from matplotlib.ticker import ScalarFormatter
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)
    ax.xaxis.get_major_formatter().set_useOffset(False)
    ax.ticklabel_format(axis="x", style="plain")

    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    fig.tight_layout()

    # Save
    stem = f"bdrate_{clip_name}"
    png_path = out_dir / f"{stem}.png"
    fig.savefig(png_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {png_path}")
    
def default_csv() -> Path:
    return Path(__file__).resolve().parent / "ablation_out" / "eval_results" / "results_detailed.csv"
# CLI

def main():
    parser = argparse.ArgumentParser(
        description="Plot BD-rate RD curve (VMAF) from results_detailed.csv.",
    )
    parser.add_argument(
        "--csv", type=Path, default= default_csv(),
        help="Path to results_detailed.csv.",
    )
    parser.add_argument(
        "--clip", type=str, default=None,
        help="Clip name to plot.  Auto-detected if only one clip exists in the CSV.",
    )
    parser.add_argument(
        "--out_dir", type=Path, default=None,
        help="Output directory for plots.  Defaults to the same directory as the CSV.",
    )
    args = parser.parse_args()

    if not args.csv.exists():
        print(f"Error: CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.out_dir or args.csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_csv(args.csv)
    clip_name, clip_rows = filter_clip(rows, args.clip)
    uniform_rows, gaze_rows = split_methods(clip_rows)

    if len(uniform_rows) < 2:
        print(f"Error: need >=2 'uniform' rows, found {len(uniform_rows)}.",
              file=sys.stderr)
        sys.exit(1)
    if len(gaze_rows) < 2:
        print(f"Error: need >=2 'gaze' rows, found {len(gaze_rows)}.",
              file=sys.stderr)
        sys.exit(1)

    rate_u, qual_u = extract_rd(uniform_rows)
    rate_g, qual_g = extract_rd(gaze_rows)

    # Validate VMAF values are present
    if np.any(np.isnan(qual_u)) or np.any(np.isnan(qual_g)):
        print("Error: some vmaf_global values are NaN or missing.  "
              "Ensure VMAF was computed for all variants.", file=sys.stderr)
        sys.exit(1)

    print(f"Clip: {clip_name}")
    print(f"  Uniform: {len(rate_u)} points, "
          f"bitrate {rate_u.min():.0f}–{rate_u.max():.0f} kbps, "
          f"VMAF {qual_u.min():.2f}–{qual_u.max():.2f}")
    print(f"  Gaze:    {len(rate_g)} points, "
          f"bitrate {rate_g.min():.0f}–{rate_g.max():.0f} kbps, "
          f"VMAF {qual_g.min():.2f}–{qual_g.max():.2f}")

    bdrate_pct, note = compute_bdrate(rate_u, qual_u, rate_g, qual_g)
    if bdrate_pct is not None:
        print(f"  BD-rate (VMAF): {bdrate_pct:+.2f}%"
              + (f"  [{note}]" if note else ""))
    else:
        print(f"  BD-rate: could not compute — {note}")

    plot_bdrate(clip_name, rate_u, qual_u, rate_g, qual_g, out_dir)


if __name__ == "__main__":
    main()
