"""Convention battery for E1-E4: concentration, cross-population
diversity, prior correction, stranger-pool baseline, turnover adoption,
minority threshold.  Consumes journals only.

Battery definitions (pre-registered per environment):
  * E1 convention variable: modal note permutation over the trailing 60
    well-formed notes; conventionalized iff modal share >= 0.5 (chance
    1/120; 'copy input' is impossible as input order is randomized).
  * E2 variable: modal (badge -> region) mapping over the trailing 12
    episodes with classifiable region pairs; conventionalized iff the
    modal mapping covers >= 0.5 of trailing episodes.
  * E3 variable: per item, modal mentioned-trait-set over the trailing 15
    references; population scheme = tuple of modal sets; per-item
    conventionalized iff modal share >= 0.5.
  * E4 variable: modal demand per (own-badge, partner-badge) direction
    over the trailing 100 rounds; equilibrium class: egalitarian (all
    modal demands 50), class (cross-badge 70/30 asymmetry), other.
  * Diversity: across populations of a cell, distribution of modal
    variants; shared-bias null rejected when multiple distinct variants
    occur with substantial frequency (report counts + Simpson diversity).
  * Turnover adoption: newcomer's first-k-actions agreement with the
    pre-turnover modal variant vs founders' early actions (same k).
  * Shuffle control: concentration statistic recomputed on random
    regroupings of transcripts across populations of the same cell.
"""

from __future__ import annotations

import json
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..analysis.stats import binomial_ci, bootstrap_ci
from .e2_grid import REGIONS

C = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834"]
GRAY = "#52514e"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e2", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False,
    "figure.facecolor": "white",
})


def load_pops(outdir: str, cell_prefix: str) -> list[dict]:
    pops = []
    if not os.path.isdir(outdir):
        return pops
    for d in sorted(os.listdir(outdir)):
        if not d.startswith(cell_prefix):
            continue
        jp = os.path.join(outdir, d, "journal.jsonl")
        if not os.path.exists(jp):
            continue
        recs = [json.loads(line) for line in open(jp)]
        pops.append({"name": d, "recs": recs,
                     "config": next((r for r in recs
                                     if r["type"] == "config"), {}),
                     "priors": next((r for r in recs
                                     if r["type"] == "priors"), None)})
    return pops


def simpson_diversity(counts: list[int]) -> float:
    n = sum(counts)
    if n <= 1:
        return 0.0
    return 1.0 - sum(c * (c - 1) for c in counts) / (n * (n - 1))


def _modal(seq):
    c = Counter(seq)
    if not c:
        return None, 0.0
    v, k = c.most_common(1)[0]
    return v, k / sum(c.values())


# ===========================================================================
# E1
# ===========================================================================

def _e1_perm_stream(recs, phases=("formation",)):
    return [(r["t"], tuple(r["perm"])) for r in recs
            if r["type"] == "interaction" and r["phase"] in phases
            and r.get("perm") and not r.get("committed")]


def analyze_e1(outdir: str, cell: str) -> dict:
    pops = load_pops(outdir, cell + "_p")
    per_pop = []
    for p in pops:
        stream = _e1_perm_stream(p["recs"])
        tail = [x[1] for x in stream[-60:]]
        modal, share = _modal(tail)
        # turnover adoption
        pre_modal = modal
        newcomer_first, founder_first = [], []
        by_agent = {}
        for r in p["recs"]:
            if r["type"] == "interaction" and r.get("perm"):
                by_agent.setdefault(r["sender_id"], []).append(
                    (r["t"], tuple(r["perm"]), r["phase"]))
        n_agents = p["config"].get("n_agents", 12)
        for aid, sends in by_agent.items():
            first3 = [s[1] for s in sends[:3]]
            match = (sum(1 for x in first3 if x == pre_modal) / len(first3)
                     if first3 else None)
            if match is None:
                continue
            if aid >= n_agents:      # newcomer
                newcomer_first.append(match)
            else:
                founder_first.append(match)
        # minority flip
        mino = [r for r in p["recs"] if r["type"] == "minority_start"]
        flip = None
        if mino:
            alt = tuple(mino[0]["alt_perm"])
            after = [(t, pm) for t, pm in
                     [(r["t"], tuple(r["perm"])) for r in p["recs"]
                      if r["type"] == "interaction"
                      and r["phase"] == "minority" and r.get("perm")
                      and not r.get("committed")]]
            tail_m = [pm for _, pm in after[-40:]]
            m2, s2 = _modal(tail_m)
            flip = bool(tail_m and m2 == alt and s2 >= 0.5)
        # inter-agent agreement: share of agent pairs whose personal modal
        # ordering (formation tail) coincides — the social signature that
        # the stranger-pool control lacks
        agent_modals = []
        for aid, sends in by_agent.items():
            fs = [pm for _t, pm, ph in sends if ph == "formation"][-15:]
            if len(fs) >= 5:
                agent_modals.append(_modal(fs)[0])
        pairs = [(a, b) for ai, a in enumerate(agent_modals)
                 for b in agent_modals[ai + 1:]]
        inter_agree = (float(np.mean([a == b for a, b in pairs]))
                       if pairs else None)
        wf = [r for r in p["recs"] if r["type"] == "interaction"]
        per_pop.append({
            "inter_agent_agreement": inter_agree,
            "pop": p["name"], "modal": modal, "share": share,
            "conventionalized": bool(modal and share >= 0.5),
            "success_tail": float(np.mean(
                [r["correct"] for r in wf[-60:]])) if wf else None,
            "wellformed_share": float(np.mean(
                [r.get("perm") is not None for r in wf])) if wf else None,
            "newcomer_adopt": (float(np.mean(newcomer_first))
                               if newcomer_first else None),
            "founder_early": (float(np.mean(founder_first))
                              if founder_first else None),
            "flip": flip,
            "prior_copy_input": (p["priors"] or {}).get("copy_input_share"),
        })
    modals = [p["modal"] for p in per_pop if p["conventionalized"]]
    counts = list(Counter(modals).values())
    out = {
        "n_pops": len(per_pop),
        "conventionalized": binomial_ci(
            sum(p["conventionalized"] for p in per_pop), len(per_pop)),
        "modal_share": bootstrap_ci([p["share"] for p in per_pop]),
        "distinct_modal_variants": len(set(modals)),
        "modal_variant_counts": Counter(
            [str(list(m)) for m in modals]).most_common(6),
        "simpson_diversity": simpson_diversity(counts),
        "newcomer_adopt": bootstrap_ci(
            [p["newcomer_adopt"] for p in per_pop]),
        "founder_early": bootstrap_ci(
            [p["founder_early"] for p in per_pop]),
        "minority_flip": binomial_ci(
            sum(bool(p["flip"]) for p in per_pop if p["flip"] is not None),
            sum(1 for p in per_pop if p["flip"] is not None)),
        "prior_copy_input_share": bootstrap_ci(
            [p["prior_copy_input"] for p in per_pop
             if p["prior_copy_input"] is not None]),
        "inter_agent_agreement": bootstrap_ci(
            [p["inter_agent_agreement"] for p in per_pop]),
        "_per_pop": per_pop,
    }
    # shuffle (pseudo-population) control on formation tails
    tails = [[x[1] for x in _e1_perm_stream(p["recs"])[-60:]]
             for p in pops]
    tails = [t for t in tails if len(t) >= 30]
    if len(tails) >= 4:
        rng = np.random.default_rng(0)
        all_msgs = [m for t in tails for m in t]
        sh = []
        for _ in range(200):
            rng.shuffle(all_msgs)
            k = len(tails[0])
            groups = [all_msgs[i * k:(i + 1) * k]
                      for i in range(len(tails))]
            sh.extend(_modal(g)[1] for g in groups if g)
        out["shuffle_control_modal_share"] = {
            "mean": float(np.mean(sh)),
            "p97.5": float(np.quantile(sh, 0.975))}
    return out


def analyze_e1_stranger(outdir: str) -> dict:
    pops = load_pops(outdir, "e1_stranger_p")
    shares = []
    for p in pops:
        stream = [(r["t"], tuple(r["perm"]), r["sender_id"])
                  for r in p["recs"] if r["type"] == "interaction"
                  and r.get("perm")]
        by_sender = {}
        for t, pm, sid in stream:
            by_sender.setdefault(sid, []).append(pm)
        pop_modals = []
        for sid, perms in by_sender.items():
            if len(perms) >= 15:
                shares.append(_modal(perms[-30:])[1])
                pop_modals.append(_modal(perms[-30:])[0])
        p["_modals"] = pop_modals
    # agreement BETWEEN tracked informers (they never meet each other):
    # the no-social baseline for inter-agent agreement
    all_modals = [m for p in pops for m in p.get("_modals", [])]
    pairs = [(a, b) for ai, a in enumerate(all_modals)
             for b in all_modals[ai + 1:]]
    agree = float(np.mean([a == b for a, b in pairs])) if pairs else None
    return {"n_tracked_informers": len(shares),
            "self_consistency_modal_share": bootstrap_ci(shares),
            "between_informer_agreement": agree}


# ===========================================================================
# E2
# ===========================================================================

COMPLEMENT = {"left": "right", "right": "left", "top": "bottom",
              "bottom": "top", "even": "odd", "odd": "even"}


def analyze_e2(outdir: str, cell: str = "e2") -> dict:
    pops = load_pops(outdir, cell + "_p")
    per_pop = []
    for p in pops:
        eps = [r for r in p["recs"] if r["type"] == "episode"]
        n_agents = p["config"].get("n_agents", 8)

        def mapping_of(e):
            a, b = e["class_a"], e["class_b"]
            if a in REGIONS and COMPLEMENT.get(a) == b:
                return (a, b)     # badgeA-region, badgeB-region
            return None

        tail = [mapping_of(e) for e in eps[-12:]]
        pair_rate = float(np.mean([m is not None for m in tail])) if tail else None
        maps = [m for m in tail if m]
        modal, share_among = _modal(maps)
        share = (len([m for m in maps if m == modal]) / len(tail)
                 if tail and modal else 0.0)
        succ_first = float(np.mean([e["success"] for e in eps[:12]]))
        succ_last = float(np.mean([e["success"] for e in eps[-12:]]))
        per_pop.append({
            "pop": p["name"], "modal_mapping": modal,
            "mapping_share": share,
            "conventionalized": bool(modal and share >= 0.5),
            "complementary_rate_tail": pair_rate,
            "success_first12": succ_first, "success_last12": succ_last,
            "collisions_last12": float(np.mean(
                [e["collisions"] for e in eps[-12:]])),
        })
    modals = [p["modal_mapping"] for p in per_pop if p["conventionalized"]]
    return {
        "n_pops": len(per_pop),
        "conventionalized": binomial_ci(
            sum(p["conventionalized"] for p in per_pop), len(per_pop)),
        "mapping_share": bootstrap_ci([p["mapping_share"] for p in per_pop]),
        "success_first12": bootstrap_ci([p["success_first12"]
                                         for p in per_pop]),
        "success_last12": bootstrap_ci([p["success_last12"]
                                        for p in per_pop]),
        "distinct_modal_mappings": len(set(modals)),
        "modal_mapping_counts": Counter(
            [str(m) for m in modals]).most_common(8),
        "_per_pop": per_pop,
    }


# ===========================================================================
# E3
# ===========================================================================

def analyze_e3(outdir: str, cell: str = "e3") -> dict:
    pops = load_pops(outdir, cell + "_p")
    per_pop = []
    for p in pops:
        inter = [r for r in p["recs"] if r["type"] == "interaction"]
        n_items = 6
        item_modal, item_share = [], []
        for it in range(n_items):
            refs = [tuple(r["mentioned"]) for r in inter
                    if r["target"] == it and r["phase"] == "formation"]
            m, s = _modal(refs[-15:])
            item_modal.append(m)
            item_share.append(s)
        # prior modal sets per item
        prior_sets = {}
        if p["priors"]:
            for probe in p["priors"]["probes"]:
                prior_sets.setdefault(probe["item"], []).append(
                    tuple(probe["set"]))
        matches_prior = [
            item_modal[it] in prior_sets.get(it, [])
            for it in range(n_items) if item_modal[it] is not None]
        coin = [c for r in inter for c in r.get("coinages", [])]
        lens = [r["note_words"] for r in inter if r["phase"] == "formation"]
        third = max(1, len(lens) // 3)
        per_pop.append({
            "pop": p["name"],
            "scheme": [list(m) if m else None for m in item_modal],
            "mean_item_share": float(np.mean(item_share)),
            "items_conventionalized": sum(1 for s in item_share if s >= 0.5),
            "success_tail": float(np.mean([r["correct"]
                                           for r in inter[-60:]])),
            "coinage_count": len(coin),
            "distinct_coinages": len(set(coin)),
            "len_first_third": float(np.mean(lens[:third])),
            "len_last_third": float(np.mean(lens[-third:])),
            "modal_matches_prior_share": (float(np.mean(matches_prior))
                                          if matches_prior else None),
        })
    schemes = [str(p["scheme"]) for p in per_pop]
    return {
        "n_pops": len(per_pop),
        "items_conventionalized_of6": bootstrap_ci(
            [p["items_conventionalized"] for p in per_pop]),
        "mean_item_share": bootstrap_ci([p["mean_item_share"]
                                         for p in per_pop]),
        "distinct_schemes": len(set(schemes)),
        "scheme_counts_top": Counter(schemes).most_common(3),
        "coinage_total": int(sum(p["coinage_count"] for p in per_pop)),
        "note_len_first_third": bootstrap_ci([p["len_first_third"]
                                              for p in per_pop]),
        "note_len_last_third": bootstrap_ci([p["len_last_third"]
                                             for p in per_pop]),
        "modal_matches_prior": bootstrap_ci(
            [p["modal_matches_prior_share"] for p in per_pop]),
        "_per_pop": per_pop,
    }


# ===========================================================================
# E4
# ===========================================================================

def analyze_e4(outdir: str, cell: str = "e4") -> dict:
    pops = load_pops(outdir, cell + "_p")
    per_pop = []
    for p in pops:
        inter = [r for r in p["recs"] if r["type"] == "interaction"]
        badges = p["config"].get("badges")
        tail = inter[-100:]
        cross = [r for r in tail if r["badge_a"] != r["badge_b"]]
        same = [r for r in tail if r["badge_a"] == r["badge_b"]]

        def demands_of(rows, badge):
            out = []
            for r in rows:
                if r["badge_a"] == badge:
                    out.append(r["demand_a"])
                if r["badge_b"] == badge:
                    out.append(r["demand_b"])
            return out

        d0 = demands_of(cross, badges[0])
        d1 = demands_of(cross, badges[1])
        m0, s0 = _modal(d0)
        m1, s1 = _modal(d1)
        all_d = [d for r in tail for d in (r["demand_a"], r["demand_b"])]
        egal_share = all_d.count(50) / len(all_d) if all_d else 0
        if egal_share >= 0.8:
            eq = "egalitarian"
        elif (m0, m1) in ((70, 30), (30, 70)) and min(s0, s1) >= 0.6:
            eq = "class"
        elif s0 >= 0.6 and s1 >= 0.6:
            eq = f"other_stable_{m0}_{m1}"
        else:
            eq = "fractious"
        per_pop.append({
            "pop": p["name"], "equilibrium": eq,
            "egal_share": egal_share,
            "cross_modal": (m0, m1), "cross_shares": (s0, s1),
            "compat_tail": float(np.mean([r["compatible"] for r in tail])),
            "same_badge_egal": (float(np.mean(
                [d == 50 for r in same
                 for d in (r["demand_a"], r["demand_b"])]))
                if same else None),
        })
    eqs = Counter(p["equilibrium"] for p in per_pop)
    return {
        "n_pops": len(per_pop),
        "equilibrium_counts": dict(eqs),
        "class_share": binomial_ci(
            sum(1 for p in per_pop if p["equilibrium"] == "class"),
            len(per_pop)),
        "egalitarian_share": binomial_ci(
            sum(1 for p in per_pop if p["equilibrium"] == "egalitarian"),
            len(per_pop)),
        "compat_tail": bootstrap_ci([p["compat_tail"] for p in per_pop]),
        "_per_pop": per_pop,
    }


# ===========================================================================
# figures + entry
# ===========================================================================

def _fig_diversity(summary, figdir, mode):
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.9))
    # E1 modal variant counts
    ax = axes[0]
    for ci_, (cell, key) in enumerate((("e1_tight", "E1 tight"),
                                       ("e1_loose", "E1 loose"))):
        d = summary.get(cell, {})
        counts = [c for _, c in d.get("modal_variant_counts", [])]
        if counts:
            ax.bar(np.arange(len(counts)) + ci_ * 0.4, counts, width=0.38,
                   color=C[ci_], label=key)
    ax.set_xlabel("modal note-ordering (rank)")
    ax.set_ylabel("populations")
    ax.set_title("E1: distinct format variants across populations",
                 fontsize=8.5)
    ax.legend(fontsize=7.5)
    ax.grid(axis="x", visible=False)
    # E2 mappings
    ax = axes[1]
    d = summary.get("e2", {})
    items = d.get("modal_mapping_counts", [])
    if items:
        ax.bar(range(len(items)), [c for _, c in items], color=C[2],
               width=0.6)
        ax.set_xticks(range(len(items)))
        ax.set_xticklabels([m for m, _ in items], rotation=45, fontsize=6,
                           ha="right")
    ax.set_title("E2: badge→region mappings", fontsize=8.5)
    ax.set_ylabel("populations")
    ax.grid(axis="x", visible=False)
    # E4 equilibria
    ax = axes[2]
    d = summary.get("e4", {})
    eqs = d.get("equilibrium_counts", {})
    if eqs:
        keys = sorted(eqs)
        ax.bar(range(len(keys)), [eqs[k] for k in keys], color=C[3],
               width=0.6)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=30, fontsize=6.5, ha="right")
    ax.set_title("E4: equilibrium types", fontsize=8.5)
    ax.set_ylabel("populations")
    ax.grid(axis="x", visible=False)
    fig.suptitle(f"Cross-population diversity of side-product conventions "
                 f"({mode} tier)", fontsize=9.5, y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, f"fig7_envs_diversity_{mode}.png"),
                bbox_inches="tight")
    plt.close(fig)


def _fig_e1_concentration(outdir, figdir, mode):
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    for ci_, cell in enumerate(("e1_tight", "e1_loose")):
        pops = load_pops(outdir, cell + "_p")
        curves = []
        for p in pops:
            stream = _e1_perm_stream(p["recs"])
            perms = [x[1] for x in stream]
            shares = []
            for end in range(40, len(perms) + 1, 20):
                shares.append(_modal(perms[max(0, end - 60):end])[1])
            if shares:
                curves.append(shares)
        if not curves:
            continue
        L = min(len(c) for c in curves)
        M = np.vstack([c[:L] for c in curves])
        x = np.arange(L) * 20 + 40
        ax.plot(x, np.median(M, axis=0), color=C[ci_], lw=1.8,
                label=f"{cell.split('_')[1]} budget (n={len(curves)})")
        ax.fill_between(x, np.percentile(M, 25, axis=0),
                        np.percentile(M, 75, axis=0), color=C[ci_],
                        alpha=0.15, lw=0)
    ax.axhline(1 / 120, color=GRAY, ls=":", lw=1)
    ax.annotate("chance (1/120)", xy=(ax.get_xlim()[1], 0.02), ha="right",
                fontsize=7, color=GRAY)
    ax.set_xlabel("well-formed notes so far (formation)")
    ax.set_ylabel("modal ordering share (trailing 60)")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(f"E1: format conventionalization ({mode} tier)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, f"fig8_e1_concentration_{mode}.png"),
                bbox_inches="tight")
    plt.close(fig)


def analyze_envs(outdir: str, figdir: str, mode: str) -> dict:
    summary = {}
    for cell in ("e1_tight", "e1_loose"):
        if any(d.startswith(cell + "_p") for d in
               (os.listdir(outdir) if os.path.isdir(outdir) else [])):
            summary[cell] = analyze_e1(outdir, cell)
    if any(d.startswith("e1_stranger") for d in
           (os.listdir(outdir) if os.path.isdir(outdir) else [])):
        summary["e1_stranger"] = analyze_e1_stranger(outdir)
    for cell, fn in (("e2", analyze_e2), ("e3", analyze_e3),
                     ("e4", analyze_e4)):
        if any(d.startswith(cell + "_p") for d in
               (os.listdir(outdir) if os.path.isdir(outdir) else [])):
            summary[cell] = fn(outdir, cell)
    os.makedirs(figdir, exist_ok=True)
    _fig_diversity(summary, figdir, mode)
    _fig_e1_concentration(outdir, figdir, mode)
    return summary


def main_envs_analysis(results_dir: str = "results",
                       figdir: str = "figures") -> None:
    out = {}
    for mode in ("mock", "live"):
        d = os.path.join(results_dir, f"envs_{mode}")
        if os.path.isdir(d):
            print(f"analysing envs ({mode}) ...")
            out[mode] = analyze_envs(d, figdir, mode)
    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if k != "_per_pop"}
        return o
    path = os.path.join(results_dir, "envs_summary.json")
    with open(path, "w") as f:
        json.dump(strip(out), f, indent=2, default=str)
    print(f"wrote {path}")


if __name__ == "__main__":
    main_envs_analysis()
