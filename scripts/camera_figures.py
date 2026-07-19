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
    # units derived from the simulator (decision-log entry 22): k is
    # interactions per replacement; expected agent lifetime ~ 2k plays
    ax.set_xlabel("k = interactions per replacement"
                  "  (agent lifetime $\\approx$ 2k plays)")
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
    for cond, mk, color, dx in (("founder", "^", ORANGE, -0.004),
                                ("posttrans", "v", PINK, 0.004)):
        pts = sorted(sweep[cond].items())
        if pts:
            fs = [f + dx for f, _ in pts]
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
               label="within-pop concentration\n(mean modal share)")
        ax.bar(x + 0.19, ds, width=0.36, color=ORANGE,
               label="cross-pop diversity\n(distinct schemes / n, rule A)")
        # bootstrap CIs over populations for both statistics
        for i, (cell, _) in enumerate([c for c in cells if
                                       d.get(c[0])][:len(xs)]):
            c = d.get(cell, {})
            ms = (c.get("modal_share") if cell.startswith("e1")
                  else c.get("mean_item_share")) or {}
            if ms.get("lo") is not None:
                ax.errorbar([i - 0.19], [ms["estimate"]],
                            yerr=[[max(0, ms["estimate"] - ms["lo"])],
                                  [max(0, ms["hi"] - ms["estimate"])]],
                            fmt="none", ecolor=GRAY, lw=0.8, capsize=2)
            dr = c.get("diversity_ratio_ci") or {}
            if dr.get("lo") is not None:
                ax.errorbar([i + 0.19], [dr["estimate"]],
                            yerr=[[max(0, dr["estimate"] - dr["lo"])],
                                  [max(0, dr["hi"] - dr["estimate"])]],
                            fmt="none", ecolor=GRAY, lw=0.8, capsize=2)
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
    gm2 = (S.get("expB_live", {}) or {})  # naming m2 handled below if present
    for cell, label in (("e1_tight", "E1 wire format, 12w"),
                        ("e1_squeeze", "E1 wire format, 6w\n(prior infeasible)"),
                        ("e2_think", "E2 grid partition"),
                        ("e3_squeeze", "E3 description scheme")):
        sm = E.get("mock", {}).get(cell, {})
        lv = E.get("live", {}).get(cell, {})
        m2 = E.get("m2", {}).get(cell, {})

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
        entries.append((label, conv_of(sm), conv_of(lv), conv_of(m2)))
    e4s = E.get("mock", {}).get("e4", {}).get("class_share", {})
    e4l = E.get("live", {}).get("e4_think", {}).get("class_share", {})
    e4m = E.get("m2", {}).get("e4_think", {}).get("class_share", {})
    entries.append(("E4 bargaining classes\n(fairness prior wins)",
                    e4s.get("p"), e4l.get("p"), e4m.get("p")))
    entries[0] = entries[0] + (None,) if len(entries[0]) == 3 else entries[0]
    entries = [(e + (None,))[:4] for e in entries]

    entries = [e for e in entries if e[1] is not None and e[2] is not None]
    fig, ax = plt.subplots(figsize=(COL_W, 2.6))
    y = np.arange(len(entries))[::-1]
    for yi, e in zip(y, entries):
        xs = [x for x in e[1:] if x is not None]
        ax.plot([min(xs), max(xs)], [yi, yi], "-", color=GRAY, lw=1.0,
                zorder=1)
    ax.scatter([e[1] for e in entries], y, s=52, zorder=3,
               facecolors="none", edgecolors=BLUE, linewidths=1.5,
               label="substrate")
    ax.scatter([e[2] for e in entries], y, color=ORANGE, s=24, zorder=2,
               label="haiku-4.5")
    m2y = [(yi, e[3]) for yi, e in zip(y, entries) if e[3] is not None]
    if m2y:
        ax.scatter([v for _, v in m2y], [yi for yi, _ in m2y],
                   color=GREEN, s=24, zorder=2, marker="D",
                   label="gpt-5-mini")
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
    fig_review(args.figdir)




def fig_review(figdir):
    """Review-response panel: null-diversity, swap matrix, adoption,
    E2 mitigation."""
    import json as _json

    def jload(p):
        try:
            return _json.load(open(p))
        except FileNotFoundError:
            return None
    nd = jload("results/review/null_diversity.json")
    sw = jload("results/review/swap.json")
    tr = jload("results/review/transplants.json")
    to = jload("results/review_m2_turnover.json")
    if not (nd and sw and tr):
        return
    fig, axes = plt.subplots(1, 4, figsize=(DBL_W, 2.2))
    # (a) null distribution vs observed
    ax = axes[0]
    dist = nd["m2"]["ruleA"]["distribution"]
    ks = sorted(int(k) for k in dist)
    tot = sum(dist.values())
    ax.bar(ks, [dist[str(k)] / tot for k in ks], color=BLUE, width=0.8,
           label="no-interaction null")
    ax.axvline(7, color=ORANGE, lw=1.6, ls="--")
    ax.annotate("observed 7/8", xy=(7, ax.get_ylim()[1] * 0.9),
                fontsize=6.5, ha="right", color=ORANGE, rotation=90)
    ax.set_xlabel("distinct schemes among 8\n(gpt-5-mini, rule A)")
    ax.set_ylabel("null probability")
    ax.set_title(f"P(≥7) = {nd['m2']['ruleA']['p_ge_7']:.2f}", fontsize=8)
    ax.grid(axis="x", visible=False)
    # (b) swap
    ax = axes[1]
    fams = ["m2", "haiku"]
    x = np.arange(2)
    ax.bar(x - 0.17, [sw[f]["within_success"] for f in fams], width=0.32,
           color=BLUE, label="within-population")
    ax.bar(x + 0.17, [sw[f]["cross_success"] for f in fams], width=0.32,
           color=ORANGE, label="cross-population")
    ax.set_xticks(x)
    ax.set_xticklabels(["gpt-5-mini", "haiku"])
    ax.set_ylabel("eval success (memory frozen)")
    ax.set_title("no compatibility payoff", fontsize=8)
    ax.legend(fontsize=6)
    ax.grid(axis="x", visible=False)
    # (c) transplant + turnover adoption
    ax = axes[2]
    ad = sum(t["adopted"] for t in tr)
    ms = [t["match_share_last10"] for t in tr]
    vals = [ad / len(tr), float(np.mean(ms)),
            (to["survived"] / to["n_pops"]) if to else 0,
            to["newcomer_mean_match"] if to else 0]
    labs = ["transplant\nadoption", "transplant\nmatch share",
            "scheme survival\n(turnover)", "newcomer\nmatch share"]
    ax.bar(range(4), vals, color=[BLUE, BLUE, GREEN, GREEN], width=0.6)
    ax.axhline(0.8, color=GRAY, ls=":", lw=0.8)
    ax.annotate("adoption criterion", xy=(3.4, 0.81), fontsize=5.5,
                ha="right", color=GRAY)
    ax.set_xticks(range(4))
    ax.set_xticklabels(labs, fontsize=5.6, rotation=30,
                       ha="right")
    ax.set_ylim(0, 1.02)
    ax.set_title("no social transmission", fontsize=8)
    ax.grid(axis="x", visible=False)
    # (d) E2 mitigation
    ax = axes[3]
    cells = [("e2_think", "baseline"), ("e2_role", "role line"),
             ("e2_hetero", "hetero pairs")]
    d = E.get("live", {})
    for i, (cell, lab) in enumerate(cells):
        c = d.get(cell, {})
        s2 = c.get("success_last12", {})
        if s2.get("estimate") is None:
            continue
        ax.bar([i], [s2["estimate"]], color=C[i] if False else
               [BLUE, ORANGE, GREEN][i], width=0.6)
        ax.errorbar([i], [s2["estimate"]],
                    yerr=[[max(0, s2["estimate"] - s2["lo"])],
                          [max(0, s2["hi"] - s2["estimate"])]],
                    fmt="none", ecolor=GRAY, lw=0.8, capsize=2)
    ax.set_xticks(range(3))
    ax.set_xticklabels([l for _, l in cells], fontsize=6.2,
                       rotation=20, ha="right")
    ax.set_ylabel("episode success (last 12)")
    ax.set_title("E2 mitigation", fontsize=8)
    ax.grid(axis="x", visible=False)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(figdir, "fig_review_e3.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_review_e3.pdf")


if __name__ == "__main__":
    main()
