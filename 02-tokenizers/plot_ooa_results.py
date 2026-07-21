#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_ooa_results.py
===================
Render figures from one or more ooa_trajectory_eval.py JSON result files.

Per result file:
  <stem>_scatter.png        developmental tier (x) vs acquisition step/exposure (y).

Across ALL inputs (the comparison figures):
  combined_trajectory.png   tau-vs-exposure curves, one line per tokenizer.
  combined_acquisition.png  per-morpheme acquisition exposure, one line per
                            tokenizer -- the "who acquires what, when" view that
                            actually compares two tokenizers' learning curves.

Usage:
  python3 plot_ooa_results.py results/*/*.json --outdir <last_tokenizer>/figs
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def label_of(res):
    return res.get("variant") or res.get("lang", "?")


def strict_block(res):
    return res["analyses"].get("strict_stable") or res["analyses"].get("strict_first")


def is_eventlog(results):
    return any(r.get("source") == "event_log" for r in results)


def _exposure_axis(ax, eventlog):
    if eventlog:
        ax.set_xlabel("token exposure")
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"{v/1e6:.1f}M" if v >= 1e6
                                  else f"{v/1e3:.0f}k"))
    else:
        ax.set_xlabel("snapshot index (token exposure -> )")


def _declutter_fallback(ax, texts, point_xy, n_iter=600, pad=1.5):
    """Dependency-free label de-overlap, used only when adjustText is missing.

    Repels label boxes off one another and off the data points (in display
    coordinates), then draws a thin leader line from each label back to its
    anchor so displaced labels remain traceable.
    """
    fig = ax.figure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    pts = ax.transData.transform(np.asarray(point_xy, dtype=float))   # anchors (display px)
    pos = pts.copy()
    pos += np.random.default_rng(0).uniform(-6, 6, pos.shape)         # break exact ties
    for _ in range(n_iter):
        boxes = []
        for t, p in zip(texts, pos):
            t.set_position(ax.transData.inverted().transform(p))
            boxes.append(t.get_window_extent(rend))
        disp = np.zeros_like(pos)
        for i in range(len(texts)):
            bi = boxes[i]
            for j in range(len(texts)):
                if i == j:
                    continue
                bj = boxes[j]
                dx = min(bi.x1, bj.x1) - max(bi.x0, bj.x0)
                dy = min(bi.y1, bj.y1) - max(bi.y0, bj.y0)
                if dx > 0 and dy > 0:                                 # overlapping -> push apart
                    cx = (bi.x0 + bi.x1) - (bj.x0 + bj.x1)
                    cy = (bi.y0 + bi.y1) - (bj.y0 + bj.y1)
                    if dx < dy:
                        disp[i, 0] += np.sign(cx or 1) * (dx + pad)
                    else:
                        disp[i, 1] += np.sign(cy or 1) * (dy + pad)
            for px, py in pts:                                        # keep labels off the dots
                ox = min(bi.x1, px + 3) - max(bi.x0, px - 3)
                oy = min(bi.y1, py + 3) - max(bi.y0, py - 3)
                if ox > 0 and oy > 0:
                    disp[i, 1] += (oy + pad) * (1 if (bi.y0 + bi.y1) / 2 >= py else -1)
        if not disp.any():
            break
        pos += np.clip(disp, -25, 25) * 0.5
    for t, p, anchor in zip(texts, pos, pts):
        t.set_position(ax.transData.inverted().transform(p))
        t.set_horizontalalignment("center")
        ax.annotate("", xy=ax.transData.inverted().transform(anchor),
                    xytext=t.get_position(),
                    arrowprops=dict(arrowstyle="-", color="0.55", lw=0.5, alpha=0.7),
                    zorder=2)


def scatter(res, path):
    a = strict_block(res)
    if not a or not a.get("rows"):
        return False
    rows = a["rows"]
    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    xs = [r["gold"] for r in rows]
    ys = [r["ooa_step"] for r in rows]
    ax.scatter(xs, ys, s=46, zorder=3)

    labels = [r["label"] + (f"\n{r['example']}" if r.get("example") else "") for r in rows]
    texts = [ax.text(x, y, lab, fontsize=6.5, ha="center", va="center", zorder=4)
             for x, y, lab in zip(xs, ys, labels)]

    # Preferred: adjustText handles overlap with leader lines. Fall back to a
    # built-in repeller so the pipeline never hard-fails if it isn't installed.
    try:
        from adjustText import adjust_text
        adjust_text(texts, x=xs, y=ys, ax=ax,
                    expand=(1.25, 1.5), force_text=(0.4, 0.6),
                    arrowprops=dict(arrowstyle="-", color="0.55", lw=0.5, alpha=0.7))
    except Exception as exc:
        print(f"  (adjustText unavailable: {exc}; using built-in declutter)")
        _declutter_fallback(ax, texts, list(zip(xs, ys)))

    ax.set_xlabel("developmental tier (earlier -> later)")
    ax.set_ylabel("acquisition exposure" if res.get("source") == "event_log"
                  else "OOA acquisition step")
    tau = a.get("tau"); ptau = a.get("partial_tau")
    sub = f"tau-b={tau:+.3f}" if tau is not None else ""
    if ptau is not None:
        sub += f"   partial(|freq)={ptau:+.3f}"
    ax.set_title(f"{label_of(res)}  ({sub})", fontsize=10)
    ax.margins(0.12)                      # headroom so displaced labels aren't clipped
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return True


def combined_trajectory(results, path):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    plotted = False
    for res in results:
        traj = res.get("trajectory")
        if not traj:
            continue
        xs = [s["step"] for s in traj if s["tau"] is not None]
        ys = [s["tau"] for s in traj if s["tau"] is not None]
        if len(xs) >= 1:
            a = strict_block(res) or {}
            ft = a.get("tau")
            lab = label_of(res) + (f"  (final tau={ft:+.2f})" if ft is not None else "")
            ax.plot(xs, ys, marker="o", ms=5, label=lab)
            plotted = True
    if not plotted:
        plt.close(fig); return False
    ax.axhline(0, color="k", lw=0.7, alpha=0.5)
    _exposure_axis(ax, is_eventlog(results))
    ax.set_ylabel("Kendall tau-b (acquired-so-far vs tiers)")
    ax.set_title("Developmental alignment vs exposure", fontsize=11)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return True


def combined_acquisition(results, path):
    order = {}
    for res in results:
        a = strict_block(res)
        if not a:
            continue
        for r in a.get("rows", []):
            order.setdefault(r["label"], r["gold"])
    if not order:
        return False
    labels = sorted(order, key=lambda l: (order[l], l))
    xpos = {l: i for i, l in enumerate(labels)}

    fig, ax = plt.subplots(figsize=(max(7.5, 0.9 * len(labels)), 4.8))
    plotted = False
    for res in results:
        a = strict_block(res)
        if not a:
            continue
        pts = sorted(((xpos[r["label"]], r["ooa_step"])
                      for r in a.get("rows", []) if r["label"] in xpos),
                     key=lambda p: p[0])
        if not pts:
            continue
        ft = a.get("tau")
        lab = label_of(res) + (f"  (tau={ft:+.2f})" if ft is not None else "")
        ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=6, label=lab)
        plotted = True
    if not plotted:
        plt.close(fig); return False

    tiers = [order[l] for l in labels]
    for i in range(1, len(labels)):
        if tiers[i] != tiers[i - 1]:
            ax.axvline(i - 0.5, color="grey", ls=":", lw=0.8, alpha=0.6)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels([f"T{order[l]}  {l}" for l in labels],
                       rotation=40, ha="right", fontsize=7)
    # ax.set_yscale("log")
    ax.set_yscale("symlog", linthresh=1e4)
    ax.set_ylabel("acquisition exposure (tokens, log)")
    ax.set_title("Per-morpheme acquisition: tokenizer comparison\n"
                 "(left->right = earlier->later developmental tier)", fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsons", nargs="+")
    ap.add_argument("--outdir", default="figs")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    results = [load(p) for p in args.jsons]
    for p, res in zip(args.jsons, results):
        stem = os.path.splitext(os.path.basename(p))[0]
        sp = os.path.join(args.outdir, f"{stem}_scatter.png")
        if scatter(res, sp):
            print("wrote", sp)
    ct = os.path.join(args.outdir, "combined_trajectory.png")
    if combined_trajectory(results, ct):
        print("wrote", ct)
    ca = os.path.join(args.outdir, "combined_acquisition.png")
    if combined_acquisition(results, ca):
        print("wrote", ca)


if __name__ == "__main__":
    main()