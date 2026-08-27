#!/usr/bin/env python3
"""
Speedup of the three RVV strategies (indexed, contiguous-run, adaptive) over the
scalar baseline, per matrix, ordered by mean intermediate products per row.

Usage:
    python3 speedup_overview.py [summary.csv] [out_dir]

Figures are written next to the summary unless out_dir is given.
"""

import csv
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

CSV = sys.argv[1] if len(sys.argv) > 1 else "bench/results/summary.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else (os.path.dirname(CSV) or ".")

os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 9,
    "xtick.labelsize": 7, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "figure.dpi": 150,
})

def fnum(x):
    try: return float(x)
    except (TypeError, ValueError): return None

rows = []
with open(CSV) as fh:
    for r in csv.DictReader(fh):
        rows.append(r)

op_mean = {}
for r in rows:
    v = fnum(r["op_mean"])
    if v is not None:
        op_mean[r["matrix"]] = v

def get(kern):
    d = {}
    for r in rows:
        if r["kernel"] == kern:
            s = fnum(r["speedup_vs_baseline"])
            if s is not None:
                d[r["matrix"]] = s
    return d

indexed = get("rvv_f32")
contig = get("contig_f32")
adaptive = get("adaptive_f32")

mats = sorted(op_mean.keys(), key=lambda m: op_mean[m])
x = np.arange(len(mats))
w = 0.26

def series(d): return [d.get(m, np.nan) for m in mats]

fig, ax = plt.subplots(figsize=(7.0, 3.0))
ax.axhline(1.0, color="#999999", linestyle="--", linewidth=0.8, zorder=1)
ax.bar(x - w, series(indexed), w, color="#1f4e79", label="indexed", zorder=2)
ax.bar(x,     series(contig),   w, color="#8c1d1d", label="contiguous-run", zorder=2)
ax.bar(x + w, series(adaptive), w, color="#c98a3a", label="adaptive", zorder=2)

ax.set_ylabel("Speedup over scalar")
ax.set_xticks(x)
ax.set_xticklabels(mats, rotation=55, ha="right")
ax.set_ylim(0.80, 1.05)
ax.set_xlim(-0.6, len(mats) - 0.4)
ax.grid(True, axis="y", linestyle=":", linewidth=0.4, alpha=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(frameon=False, ncol=3, loc="lower center",
          bbox_to_anchor=(0.5, 1.01), handlelength=1.3, columnspacing=1.4)

fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(OUT, "speedup_overview.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(OUT, "speedup_overview.png"), bbox_inches="tight", dpi=200)

for name, d in [("indexed", indexed), ("contiguous-run", contig), ("adaptive", adaptive)]:
    vals = list(d.values())
    print(f"{name}: mean={np.mean(vals):.3f} min={min(vals):.3f} max={max(vals):.3f} n={len(vals)}")
print(f"wrote {OUT}/speedup_overview.pdf and .png")