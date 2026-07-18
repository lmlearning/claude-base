#!/usr/bin/env python3
"""Camera-ready figure set for the AAAI submission.

PDF output, colourblind-safe Okabe-Ito palette (validated: lightness
band, chroma, CVD separation, normal-vision floor), AAAI column sizing
(3.3in single column / 6.9in double).  Every figure reads only the
analysis JSONs and run summaries - regenerating after new runs updates
everything.

Usage: python scripts/camera_figures.py [--figdir figures/camera]
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, GREEN, PINK = "#0072B2", "#E69F00", "#009E73", "#CC79A7"
GRAY = "#5f5e5b"
COL_W, DBL_W = 3.3, 6.9

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "legend.fontsize": 6.8, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e8e7e4", "grid.linewidth": 0.5,
    "axes.axisbelow": True, "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 300,
})


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


S = load("results/analysis_summary.json")
E = load("results/envs_summary.json")
A = S.get("expA", {})
B = S.get("expB_live", {})


def save(fig, figdir, name):
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(figdir, name), bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def yerr(rows, key):
    p = np.array([r[key]["p"] for r in rows])
    lo = np.array([r[key]["lo"] for r in rows])
    hi = np.array([r[key]["hi"] for r in rows])
    return p, np.vstack([np.clip(p - lo, 0, 1), np.clip(hi - p, 0, 1)])


# ---------------------------------------------------------------------------
def fig_convergence(figdir):
    """Genesis consensus times: substrate distribution vs live runs."""
    fig, ax = plt.subplots(figsize=(COL_W, 2.1))
    gc = A.get("genesis_core", {})
    med = gc.get("consensus_time", {}).get("estimate")
    iqr = gc.get("consensus_time_iqr", [None, None])
    live_t = []
    for d in sorted(glob.glob("results/expB_live/b_genesis_*_r*/summary.json")):
        s = load(d)
        g = s.get("genesis", {})
        if g.get("converged"):
            live_t.append(g["consensus_time"])
    if iqr[0] is not None:
        ax.axvspan(iqr[0], iqr[1], color=BLUE, alpha=0.15, lw=0,
                   label="substrate IQR (n=1000)")
        ax.axvline(med, color=BLUE, lw=1.4)
    if live_t:
        ax.plot(live_t, np.arange(1, len(live_t) + 1), "o", color=ORANGE,
                ms=4, label=f"live LLM runs (n={len(live_t)})")
        ax.axvline(float(np.median(live_t)), color=ORANGE, lw=1.4, ls="--")
    ax.set_xlabel("interactions to consensus (N=24, 10 names)")
    ax.set_ylabel("live run (rank order)")
    ax.legend(loc="lower right")
    save(fig, figdir, "fig_convergence.pdf")


def fig_survival(figdir):
    rows = A.get("survival_vs_rate", [])
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    k = np.array([r["k"] for r in rows])
    for key, color, label in (("gen1", BLUE, "1 generation"),
                              ("gen2", ORANGE, "2 generations")):
        p, err = yerr(rows, key)
        ax.errorbar(k, p, yerr=err, fmt="o-", color=color, ms=3, lw=1.2,
                    capsize=1.5, elinewidth=0.7, label=label)
    for key, color in (("critical_k_gen1", BLUE), ("critical_k_gen2", ORANGE)):
        x = A.get(key, {}).get("x50")
        if x:
            ax.axvline(x, color=color, lw=0.8, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("turnover rate k (replacements per 24 interactions)")
    ax.set_ylabel("survival probability")
    ax.legend(loc="lower left")
    save(fig, figdir, "fig_survival.pdf")


def fig_socialisation(figdir):
    soc = B.get("socialisation", {})
    if not soc:
        return
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    arms = [("dialogue_on", "dialogue on", BLUE),
            ("dialogue_off", "dialogue off", ORANGE)]
    for xi, (key, label, color) in enumerate(arms):
        d = soc.get(key, {})
        for gi, (grp, mk) in enumerate((("newcomer_conformity", "o"),
                                        ("founder_conformity", "s"))):
            c = d.get(grp, {})
            if c.get("estimate") is None:
                continue
            x = xi + (gi - 0.5) * 0.18
            ax.errorbar([x], [c["estimate"]],
                        yerr=[[max(0, c["estimate"] - c["lo"])],
                              [max(0, c["hi"] - c["estimate"])]],
                        fmt=mk, color=color, ms=5, capsize=2)
    ax.plot([], [], "o", color=GRAY, label="newcomers")
    ax.plot([], [], "s", color=GRAY, label="founders")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["dialogue on", "dialogue off"])
    ax.set_ylabel("interactions to conformity\n(median, run-clustered CI)")
    ax.legend(loc="upper left")
    save(fig, figdir, "fig_socialisation.pdf")


def _live_sweep_points():
    """Live minority sweep results: f -> (flips, n) per condition."""
    out = {"founder": {}, "posttrans": {}}
    for d in glob.glob("results/expB_live/b_minority_*_b2k_r*/summary.json"):
        s = load(d)
        mi = s.get("minority", {})
        if not mi.get("usable"):
            continue
        cond = "posttrans" if "posttrans" in d else "founder"
        f = s["config"].get("minority_fraction", 0.25)
        k, n = out[cond].get(f, (0, 0))
        out[cond][f] = (k + bool(mi.get("flipped")), n + 1)
    return out


def fig_minority(figdir):
    fig, ax = plt.subplots(figsize=(COL_W, 2.3))
    for rows_key, color, label in (("minority_founder", BLUE,
                                    "substrate, founder"),
                                   ("minority_posttrans", GREEN,
                                    "substrate, post-transmission")):
        rows = A.get(rows_key, [])
        if not rows:
            continue
        f = np.array([r["f"] for r in rows])
        p, err = yerr(rows, "flip")
        ax.errorbar(f, p, yerr=err, fmt="o-", ms=2.5, lw=1.1, capsize=1.5,
                    elinewidth=0.6, color=color, label=label)
    sweep = _live_sweep_points()
    for cond, mk, color in (("founder", "^", ORANGE), ("posttrans", "v", PINK)):
        pts = sorted(sweep[cond].items())
        if pts:
            fs = [f for f, _ in pts]
            ps = [k / n for _, (k, n) in pts]
            ax.plot(fs, ps, mk, color=color, ms=6, mec="white", mew=0.5,
                    label=f"live LLM, {cond} (2k budget)")
    x = A.get("f50_founder", {}).get("x50")
    if x:
        ax.axvline(x, color=GRAY, lw=0.8, ls=":")
        ax.annotate(f"substrate $f_{{50}}$={x:.3f}", xy=(x, 0.55),
                    fontsize=6, rotation=90, va="center", ha="right",
                    color=GRAY)
    ax.set_xlabel("committed-minority fraction f")
    ax.set_ylabel("flip probability")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(loc="center right", fontsize=6)
    save(fig, figdir, "fig_minority.pdf")


def fig_esuite(figdir):
    """Concentration vs cross-population diversity, substrate vs live."""
    fig, axes = plt.subplots(1, 2, figsize=(DBL_W, 2.3))
    cells = [("e1_tight", "E1 12w"), ("e1_squeeze", "E1 6w"),
             ("e3_squeeze", "E3 3w")]
    for ax, mode, tier in ((axes[0], "mock", "substrate"),
                           (axes[1], "live", "live LLM")):
        d = E.get(mode, {})
        xs, cs, ds, labels = [], [], [], []
        for i, (cell, label) in enumerate(cells):
            c = d.get(cell)
            if not c:
                continue
            conc = (c.get("modal_share", {}).get("estimate")
                    if cell.startswith("e1")
                    else c.get("mean_item_share", {}).get("estimate"))
            n = c.get("n_pops", 1)
            distinct = (c.get("distinct_modal_variants")
                        if cell.startswith("e1")
                        else c.get("distinct_schemes")) or 0
            div = distinct / max(1, n)
            xs.append(i)
            cs.append(conc or 0)
            ds.append(div)
            labels.append(label)
        x = np.arange(len(xs))
        ax.bar(x - 0.19, cs, width=0.36, color=BLUE,
               label="within-pop concentration")
        ax.bar(x + 0.19, ds, width=0.36, color=ORANGE,
               label="cross-pop diversity\n(distinct schemes / n)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.05)
        ax.set_title(tier)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("share")
    handles, labels_ = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.08))
    save(fig, figdir, "fig_esuite.pdf")


def fig_inversion(figdir):
    """The priors-vs-history inversion at a glance: share of populations
    forming an ARBITRARY convention, substrate vs live, per setting."""
    entries = []  # (label, substrate p, live p)
    gm = S.get("expB_live", {}).get("genesis", {}).get("convergence_rate", {})
    ga = A.get("genesis_core", {}).get("convergence_rate", {})
    entries.append(("naming game\n(alignment paid directly)",
                    ga.get("p"), gm.get("p")))
    for cell, label in (("e1_tight", "E1 wire format, 12w"),
                        ("e1_squeeze", "E1 wire format, 6w\n(prior infeasible)"),
                        ("e2_think", "E2 grid partition"),
                        ("e3_squeeze", "E3 description scheme")):
        sm = E.get("mock", {}).get(cell, {})
        lv = E.get("live", {}).get(cell, {})

        def conv_of(c):
            if not c:
                return None
            if cell.startswith("e3"):
                # arbitrary-convention share = distinct schemes beyond bias:
                # conventionalized-and-diverse populations; with shared items
                # this is (distinct - 1 shared-bias scheme) / n, floored at 0
                n = c.get("n_pops", 1)
                k = max(0, (c.get("distinct_schemes") or 1) - 1)
                return k / max(1, n)
            return (c.get("conventionalized") or {}).get("p")
        entries.append((label, conv_of(sm), conv_of(lv)))
    e4s = E.get("mock", {}).get("e4", {}).get("class_share", {})
    e4l = E.get("live", {}).get("e4_think", {}).get("class_share", {})
    entries.append(("E4 bargaining classes\n(fairness prior wins)",
                    e4s.get("p"), e4l.get("p")))

    entries = [e for e in entries if e[1] is not None and e[2] is not None]
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    y = np.arange(len(entries))[::-1]
    for yi, (label, ps, pl) in zip(y, entries):
        ax.plot([ps, pl], [yi, yi], "-", color=GRAY, lw=1.0, zorder=1)
    ax.scatter([e[1] for e in entries], y, s=52, zorder=3,
               facecolors="none", edgecolors=BLUE, linewidths=1.5,
               label="substrate")
    ax.scatter([e[2] for e in entries], y, color=ORANGE, s=24, zorder=2,
               label="live LLM")
    ax.set_yticks(y)
    ax.set_yticklabels([e[0] for e in entries], fontsize=6.5)
    ax.set_xlabel("share forming an arbitrary convention")
    ax.set_xlim(-0.04, 1.04)
    ax.legend(loc="lower right")
    ax.grid(axis="y", visible=False)
    save(fig, figdir, "fig_inversion.pdf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figdir", default="figures/camera")
    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)
    fig_convergence(args.figdir)
    fig_survival(args.figdir)
    fig_socialisation(args.figdir)
    fig_minority(args.figdir)
    fig_esuite(args.figdir)
    fig_inversion(args.figdir)


if __name__ == "__main__":
    main()
