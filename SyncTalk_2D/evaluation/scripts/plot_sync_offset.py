"""Plot SyncNet offset-sweep curves.

Produces the "is the metric actually working?" figure: mean sync score against
audio/video offset. A working expert peaks at offset 0 and falls away either
side; a collapsed one is flat.

Usage:
    python evaluation/scripts/plot_sync_offset.py \
        --run  evaluation/runs/<matched_run>   "Matched audio/video" \
        --run  evaluation/runs/<control_run>   "Mismatched (control)" \
        --output evaluation/reports/sync_offset_curve.png
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_curve(run_dir: Path):
    rows = []
    with open(run_dir / "sync_offset.csv", newline="") as f:
        for r in csv.DictReader(f):
            if r["mean_sync_score"]:
                rows.append((int(r["offset"]), float(r["mean_sync_score"])))
    rows.sort()
    meta = {}
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists():
        meta = json.loads(metrics_path.read_text())
    return [r[0] for r in rows], [r[1] for r in rows], meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs=2, action="append", metavar=("RUN_DIR", "LABEL"),
                    required=True, help="Run directory and its legend label. Repeatable.")
    ap.add_argument("--output", default="evaluation/reports/sync_offset_curve.png")
    ap.add_argument("--title", default="SyncNet offset sweep")
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = ["#1b6ca8", "#c0392b", "#27ae60", "#8e44ad"]

    for i, (run_dir, label) in enumerate(args.run):
        offsets, scores, meta = load_curve(Path(run_dir))
        c = colors[i % len(colors)]
        ax.plot(offsets, scores, marker="o", ms=3.5, lw=1.8, color=c, label=label)

        # Only mark a peak on curves that actually have one. Starring the
        # argmax of a flat control curve would imply it found an alignment.
        best = meta.get("best_offset")
        passed = meta.get("acceptance", {}).get("best_offset_within_2_frames")
        if passed and best and best.get("offset") is not None:
            bo, bs = int(best["offset"]), float(best["mean_sync_score"])
            ax.plot([bo], [bs], marker="*", ms=15, color=c, zorder=5,
                    markeredgecolor="white", markeredgewidth=0.6)
            ax.annotate(f"peak @ offset {bo:+d}\nscore {bs:.3f}", xy=(bo, bs),
                        xytext=(14, -4), textcoords="offset points",
                        fontsize=8.5, color=c, va="top")

    ax.axvline(0, color="#888", lw=0.9, ls="--", zorder=0)
    ax.axhline(0, color="#ccc", lw=0.8, zorder=0)
    ax.set_xlabel("Audio/video offset (frames)")
    ax.set_ylabel("Mean sync score (cosine similarity)")
    ax.set_title(args.title)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.grid(alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"Wrote {out}")
    print(f"Wrote {out.with_suffix('.pdf')}  (vector, for the paper)")


if __name__ == "__main__":
    main()
