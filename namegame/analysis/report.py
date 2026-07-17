"""Full analysis: statistics + publication figures + JSON summary.

Consumes only recorded transcripts/JSONL.  Produces:
  figures/fig1_convergence.png     A vs B convergence curves
  figures/fig2_winners.png         winner distribution vs uniform
  figures/fig3_survival.png        survival vs turnover rate (phase boundary)
  figures/fig4_socialisation.png   founder vs newcomer conformity; B ablation
  figures/fig5_enforcement.png     normative share vs convention age + solitary
  figures/fig6_minority.png        flip probability vs f, founder vs post-trans
  results/analysis_summary.json    every number cited in RESULTS.md
"""

from __future__ import annotations

import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sps

from . import load
from .stats import (binomial_ci, bootstrap_ci, cluster_bootstrap_slope,
                    km_survival, logistic_threshold, uniformity_test,
                    _fit_logistic)

# Validated categorical palette (dataviz reference, light mode, fixed order)
C = ["#2a78d6", "#008300", "#e87ba4", "#eda100", "#1baf7a", "#eb6834"]
GRAY = "#52514e"

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 300, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e5e2", "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False,
    "figure.facecolor": "white",
})


def _save(fig, figdir, name):
    os.makedirs(figdir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, name), bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# Experiment A
# ===========================================================================

def analyze_a(adir: str, figdir: str) -> dict:
    out: dict = {}

    # --- Genesis ----------------------------------------------------------
    gen = load.load_cell(adir, "genesis_core")
    gen_rl = load.load_cell(adir, "genesis_rl")
    gen_par = load.load_cell(adir, "genesis_parallel_check")
    conv = [r for r in gen if r["converged"]]
    out["genesis_core"] = {
        "n_runs": len(gen),
        "convergence_rate": binomial_ci(len(conv), len(gen)),
        "consensus_time": bootstrap_ci([r["consensus_time"] for r in conv],
                                       np.median),
        "consensus_time_iqr": [
            float(np.percentile([r["consensus_time"] for r in conv], q))
            for q in (25, 75)] if conv else None,
    }
    winners = {}
    for r in conv:
        winners[r["winner"]] = winners.get(r["winner"], 0) + 1
    for w in range(10):
        winners.setdefault(w, 0)
    out["winner_uniformity"] = uniformity_test(winners)

    # RL robustness + parallelisation check
    conv_rl = [r for r in gen_rl if r["converged"]]
    out["genesis_rl"] = {
        "n_runs": len(gen_rl),
        "convergence_rate": binomial_ci(len(conv_rl), len(gen_rl)),
        "consensus_time": bootstrap_ci([r["consensus_time"] for r in conv_rl],
                                       np.median),
    }
    conv_par = [r for r in gen_par if r["converged"]]
    ks = sps.ks_2samp([r["consensus_time"] for r in conv],
                      [r["consensus_time"] for r in conv_par]) \
        if conv and conv_par else None
    out["parallelisation_check"] = {
        "note": ("round-based scheduler (parallel within round) vs the "
                 "published sequential dynamics; KS test on consensus-time "
                 "distributions"),
        "n_round_runs": len(gen_par),
        "convergence_rate_round": binomial_ci(len(conv_par), len(gen_par)),
        "consensus_time_round": bootstrap_ci(
            [r["consensus_time"] for r in conv_par], np.median),
        "ks_stat": float(ks.statistic) if ks else None,
        "ks_p": float(ks.pvalue) if ks else None,
    }

    # --- Transmission: survival vs rate -----------------------------------
    # unify the 1/k sweep and the super-fast r-per-interaction sweep on a
    # single axis: k_eff = interactions per replacement (k, or 1/r)
    trans_cells: dict[float, list[dict]] = {}
    for cell, runs in load.cells_matching(adir, "transmission_core_k").items():
        trans_cells[float(cell.split("_k")[1])] = runs
    for cell, runs in load.cells_matching(adir, "transmission_fast_r").items():
        trans_cells[1.0 / int(cell.split("_r")[1])] = runs
    surv_rows = []
    for k_eff, runs in sorted(trans_cells.items()):
        usable = [r for r in runs if r.get("converged_genesis")]
        g1 = sum(bool(r["survived_gen1"]) for r in usable)
        g2 = sum(bool(r["survived_gen2"]) for r in usable)
        surv_rows.append({
            "k": k_eff, "rate": 1.0 / k_eff, "n": len(usable),
            "gen1": binomial_ci(g1, len(usable)),
            "gen2": binomial_ci(g2, len(usable)),
        })
    out["survival_vs_rate"] = surv_rows

    # phase boundary: critical k_eff where survival crosses 50%
    for gen_key, out_key in (("survived_gen1", "critical_k_gen1"),
                             ("survived_gen2", "critical_k_gen2")):
        xs, ys = [], []
        for k_eff, runs in trans_cells.items():
            for r in runs:
                if r.get("converged_genesis"):
                    xs.append(k_eff)
                    ys.append(int(bool(r[gen_key])))
        if len(set(ys)) > 1:
            out[out_key] = logistic_threshold(xs, ys, log_x=True)
        else:
            out[out_key] = {"x50": None,
                            "note": "no variation: survival saturated"}

    # discrete-generation survival analysis (life table over generations,
    # per rate; probes exist at generation boundaries)
    km = {}
    for row in surv_rows:
        cell_runs = trans_cells[row["k"]]
        durations, events = [], []
        for r in cell_runs:
            if not r.get("converged_genesis"):
                continue
            if not r["survived_gen1"]:
                durations.append(1); events.append(1)
            elif not r["survived_gen2"]:
                durations.append(2); events.append(1)
            else:
                durations.append(2); events.append(0)
        km[row["k"]] = km_survival(durations, events)
    out["survival_km_by_k"] = km

    # newcomer vs founder conformity (core condition)
    newcomer_by_k, founder_all = {}, []
    newcomer_censored = {}
    for k, runs in trans_cells.items():
        ct, cens = [], 0
        for r in runs:
            if not r.get("converged_genesis"):
                continue
            founder_all.extend([c for c in (r.get("founder_conformity") or [])
                                if c is not None])
            for nc in r["newcomers"]:
                if nc["conformity_time"] is not None:
                    ct.append(nc["conformity_time"])
                else:
                    cens += 1
        newcomer_by_k[k] = ct
        newcomer_censored[k] = cens
    out["founder_conformity"] = bootstrap_ci(founder_all, np.median)
    out["newcomer_conformity_by_k"] = {
        str(k): {"median_ci": bootstrap_ci(v, np.median),
                 "censored": newcomer_censored[k]}
        for k, v in sorted(newcomer_by_k.items())}

    # --- Committed minority: founder vs post-transmission ------------------
    def minority_curve(prefix, n_agents=24):
        rows, xs, ys = [], [], []
        for cell, runs in load.cells_matching(adir, prefix).items():
            m = re.search(r"_c(\d+)$", cell)
            if not m:
                continue
            c = int(m.group(1))
            usable = [r for r in runs if r.get("usable")]
            flips = sum(bool(r["flipped"]) for r in usable)
            rows.append({"n_committed": c, "f": c / n_agents,
                         "flip": binomial_ci(flips, len(usable))})
            xs.extend([c / n_agents] * len(usable))
            ys.extend(int(bool(r["flipped"])) for r in usable)
        rows.sort(key=lambda r: r["n_committed"])
        thr = logistic_threshold(xs, ys) if xs else None
        return rows, thr

    out["minority_founder"], out["f50_founder"] = minority_curve(
        "minority_founder_c")
    out["minority_posttrans"], out["f50_posttrans"] = minority_curve(
        "minority_posttrans_c")
    rl_rows, rl_thr = minority_curve("minority_rl_founder_c")
    out["minority_rl_founder"], out["f50_rl_founder"] = rl_rows, rl_thr
    if out["f50_founder"] and out["f50_posttrans"]:
        out["f50_difference_posttrans_minus_founder"] = {
            "estimate": (None if (out["f50_posttrans"]["x50"] is None or
                                  out["f50_founder"]["x50"] is None)
                         else out["f50_posttrans"]["x50"] -
                         out["f50_founder"]["x50"]),
            "note": ("CIs on each threshold are bootstrap over runs; "
                     "difference significant if CIs disjoint"),
        }

    # --- Transplant --------------------------------------------------------
    tp = [r for r in load.load_cell(adir, "transplant_core") if r.get("usable")]
    out["transplant"] = {
        "n_usable": len(tp),
        "switched": binomial_ci(sum(r["switched"] for r in tp), len(tp)),
        "switch_time": bootstrap_ci(
            [r["switch_time"] for r in tp if r["switch_time"] is not None],
            np.median),
        "host_still_consensus": binomial_ci(
            sum(r["host_still_consensus"] for r in tp), len(tp)),
    }

    # --- Population-size sweep --------------------------------------------
    nsweep = {}
    for n in (12, 24, 48):
        cell = "genesis_core" if n == 24 else f"genesis_N{n}"
        runs = load.load_cell(adir, cell)
        convn = [r for r in runs if r["converged"]]
        entry = {"consensus_time": bootstrap_ci(
            [r["consensus_time"] for r in convn], np.median),
            "convergence_rate": binomial_ci(len(convn), len(runs))}
        rows, thr = (minority_curve(f"minority_N{n}_c", n_agents=n)
                     if n != 24 else (None, None))
        if n != 24:
            entry["minority_rows"] = rows
            entry["f50"] = thr
            tcells = load.cells_matching(adir, f"transmission_N{n}_k")
            entry["survival"] = []
            for cname, cruns in sorted(
                    tcells.items(), key=lambda kv: int(kv[0].split("_k")[1])):
                kk = int(cname.split("_k")[1])
                us = [r for r in cruns if r.get("converged_genesis")]
                entry["survival"].append(
                    {"k": kk, "gen2": binomial_ci(
                        sum(bool(r["survived_gen2"]) for r in us), len(us))})
        nsweep[n] = entry
    out["population_size_sweep"] = nsweep

    _figures_a(adir, figdir, gen, gen_rl, out, trans_cells,
               founder_all, newcomer_by_k)
    return out


def _figures_a(adir, figdir, gen, gen_rl, out, trans_cells,
               founder_all, newcomer_by_k):
    # ---- fig2: winner distribution ----------------------------------
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    obs = out["winner_uniformity"]["observed"]
    keys = sorted(obs, key=lambda k: int(k))
    vals = [obs[k] for k in keys]
    ax.bar(range(len(keys)), vals, color=C[0], width=0.62)
    exp = sum(vals) / len(keys)
    ax.axhline(exp, color=GRAY, lw=1.2, ls="--")
    ax.annotate(f"uniform ({exp:.0f})", xy=(len(keys) - 0.5, exp),
                ha="right", va="bottom", fontsize=8, color=GRAY)
    ax.set_xlabel("name index")
    ax.set_ylabel("runs won")
    p = out["winner_uniformity"]["p_value"]
    ax.set_title(f"Winning name across {sum(vals)} runs "
                 f"(χ² p = {p:.2f})", fontsize=9)
    ax.grid(axis="x", visible=False)
    _save(fig, figdir, "fig2_winners.png")

    # ---- fig3: survival vs turnover rate -----------------------------
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    rows = out["survival_vs_rate"]
    ks = [r["k"] for r in rows]
    for gi, (gen_key, color, label) in enumerate(
            [("gen1", C[0], "after 1 generation"),
             ("gen2", C[1], "after 2 generations")]):
        p = [r[gen_key]["p"] for r in rows]
        lo = [max(0.0, r[gen_key]["p"] - r[gen_key]["lo"]) for r in rows]
        hi = [max(0.0, r[gen_key]["hi"] - r[gen_key]["p"]) for r in rows]
        ax.errorbar(ks, p, yerr=[lo, hi], fmt="o", ms=4.5, lw=1.4,
                    capsize=2.5, color=color, label=label)
    # logistic fit curves (on log k)
    for gen_key, color, thr_key in (("gen1", C[0], "critical_k_gen1"),
                                    ("gen2", C[1], "critical_k_gen2")):
        xs, ys = [], []
        for k, runs in trans_cells.items():
            for r in runs:
                if r.get("converged_genesis"):
                    xs.append(np.log(k))
                    ys.append(int(bool(r[f"survived_{gen_key}"])))
        if len(set(ys)) < 2:
            continue
        fit = _fit_logistic(np.array(xs), np.array(ys))
        if fit:
            a, b = fit
            kk = np.geomspace(min(ks), max(ks), 200)
            ax.plot(kk, 1 / (1 + np.exp(-(a + b * np.log(kk)))),
                    color=color, lw=1.6, alpha=0.75)
        thr = out.get(thr_key) or {}
        if thr.get("x50"):
            ax.axvline(thr["x50"], color=color, lw=1.0, ls=":", alpha=0.8)
    ax.set_xscale("log")
    tick_ks = [0.125, 0.25, 0.5, 1, 2, 8, 32, 128]
    ax.set_xticks([t for t in tick_ks if min(ks) <= t <= max(ks)])
    ax.set_xticklabels([("1/%d" % round(1 / t)) if t < 1 else str(int(t))
                        for t in tick_ks if min(ks) <= t <= max(ks)])
    ax.minorticks_off()
    ax.axhline(0.1, color=GRAY, lw=0.9, ls=":")
    ax.annotate("chance re-convergence (1/W)", xy=(max(ks), 0.105),
                ha="right", va="bottom", fontsize=7.5, color=GRAY)
    ax.set_xlabel("interactions per replacement, k  (slower turnover →)")
    ax.set_ylabel("P(original name survives)")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="center right", fontsize=8)
    thr2 = out.get("critical_k_gen2") or {}
    if thr2.get("x50"):
        ax.set_title(f"Turnover phase boundary: k₅₀ = {thr2['x50']:.2f} "
                     f"[{thr2['lo']:.2f}, {thr2['hi']:.2f}] (2 gen)",
                     fontsize=9)
    _save(fig, figdir, "fig3_survival.png")

    # ---- fig4 (left half drawn here; right half in analyze_b) --------
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    xs = np.sort(np.asarray(founder_all, dtype=float))
    if len(xs):
        ax.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=GRAY, lw=1.8,
                label=f"founders (n={len(xs)})")
    for ki, k in enumerate([0.25, 2.0, 128.0]):
        v = np.sort(np.asarray(newcomer_by_k.get(k, []), dtype=float))
        if len(v):
            klab = f"1/{int(round(1/k))}" if k < 1 else f"{int(k)}"
            ax.plot(v, np.arange(1, len(v) + 1) / len(v), color=C[ki],
                    lw=1.6, label=f"newcomers, k={klab} (n={len(v)})")
    ax.set_xscale("log")
    ax.set_xlabel("own plays until conformity (10-play block, ≥9 matches)")
    ax.set_ylabel("cumulative fraction")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.set_title("Time-to-conformity: founders vs newcomers (minimal agents)",
                 fontsize=9)
    _save(fig, figdir, "fig4a_socialisation_A.png")

    # ---- fig6: committed minority ------------------------------------
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for rows, thr, color, label in (
            (out["minority_founder"], out["f50_founder"], C[0], "founder"),
            (out["minority_posttrans"], out["f50_posttrans"], C[1],
             "post-transmission"),
            (out["minority_rl_founder"], out["f50_rl_founder"], C[3],
             "founder, RL agents")):
        if not rows:
            continue
        f = [r["f"] for r in rows]
        p = [r["flip"]["p"] for r in rows]
        lo = [max(0.0, r["flip"]["p"] - r["flip"]["lo"]) for r in rows]
        hi = [max(0.0, r["flip"]["hi"] - r["flip"]["p"]) for r in rows]
        ax.errorbar(f, p, yerr=[lo, hi], fmt="o", ms=4, lw=1.2, capsize=2.2,
                    color=color, label=label,
                    alpha=0.95 if "RL" not in label else 0.7)
        if thr and thr["x50"]:
            xs_f, ys_f = [], []
            for r in rows:
                xs_f.extend([r["f"]] * r["flip"]["n"])
                ys_f.extend([1] * r["flip"]["k"] +
                            [0] * (r["flip"]["n"] - r["flip"]["k"]))
            fit = _fit_logistic(np.array(xs_f, dtype=float),
                                np.array(ys_f, dtype=float))
            if fit:
                a, b = fit
                ff = np.linspace(min(f), max(f), 200)
                ax.plot(ff, 1 / (1 + np.exp(-(a + b * ff))), color=color,
                        lw=1.5, alpha=0.7)
    ax.set_xlabel("committed fraction f")
    ax.set_ylabel("P(flip to minority name)")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(fontsize=8, loc="upper left")
    tf, tp_ = out["f50_founder"], out["f50_posttrans"]
    if tf and tp_ and tf["x50"] and tp_["x50"]:
        ax.set_title(f"Critical mass: f₅₀ founder = {tf['x50']:.3f} "
                     f"[{tf['lo']:.3f},{tf['hi']:.3f}]   "
                     f"post-trans = {tp_['x50']:.3f} "
                     f"[{tp_['lo']:.3f},{tp_['hi']:.3f}]", fontsize=8.5)
    _save(fig, figdir, "fig6_minority.png")


# ===========================================================================
# Experiment B
# ===========================================================================

def _b_conformity(run, target, block=10, need=9):
    """Per-agent conformity times from a replayed journal."""
    from ..core.engine import conformity_time
    founders, newcomers = [], []
    for aid, a in run["agents"].items():
        if aid < 0:
            continue
        plays = [nm for _t, nm in a["plays"]]
        ct = conformity_time(plays, target, block, need)
        if a["born_at"] == 0 and aid < run["config"]["n_agents"]:
            founders.append(ct)
        else:
            newcomers.append(ct)
    return founders, newcomers


def analyze_b(bdir: str, figdir: str, mode: str) -> dict:
    out: dict = {"mode": mode,
                 "note": ("MOCK MODE: cheap-tier policy behind the LLM "
                          "interface; validates the pipeline, not LLM "
                          "behaviour" if mode == "mock" else "live")}
    runs = load.load_b_all(bdir)
    if not runs:
        return {"mode": mode, "missing": True}
    labels = load.load_judge_labels(bdir)

    # --- integrity metrics -------------------------------------------------
    comp_pass = [r["summary"].get("comprehension_passed") for r in runs]
    mal = [r["summary"].get("final", {}).get("malformed_rate") for r in runs
           if r["summary"].get("final")]
    out["comprehension_pass_rate"] = binomial_ci(sum(bool(x) for x in comp_pass),
                                                 len(comp_pass))
    out["malformed_rate"] = bootstrap_ci(mal, np.mean)

    # priors: pooled uniformity + winner-prior arbitrariness correction
    pooled = {}
    winner_priors = []
    for r in runs:
        pr = r["summary"].get("priors", {})
        tot = sum(pr.values()) or 1
        for tok, c in pr.items():
            pooled[tok] = pooled.get(tok, 0) + c
        g = r["summary"].get("genesis")
        if g and g.get("winner"):
            winner_priors.append((pr.get(g["winner"], 0) + 0.5) /
                                 (tot + 5))  # smoothed
    # pooled uniformity across token *positions* is meaningless across runs
    # (pools differ); instead test each run's prior spread + winner-prior.
    out["winner_prior_mean"] = bootstrap_ci(winner_priors, np.mean)

    # token audit: real-word incidence in pools (the embedded blocklist was
    # imperfect for early live runs; fixed to a full dictionary after run
    # start — report the incidence and a sensitivity split honestly)
    try:
        from english_words import get_english_words_set
        D = get_english_words_set(["web2"], lower=True)
        n_tok, real_toks, flagged_runs = 0, [], []
        for r in runs:
            pool = (r["config"] or {}).get("pool", [])
            n_tok += len(pool)
            real_toks.extend(t for t in pool if t.lower() in D)
            g = r["summary"].get("genesis") or {}
            if g.get("winner") and g["winner"].lower() in D:
                flagged_runs.append(r["name"])
        wp_clean = [wp for r, wp in zip(
            [r for r in runs if r["summary"].get("genesis", {}).get("winner")],
            winner_priors)
            if r["name"] not in flagged_runs]
        out["token_audit"] = {
            "pool_tokens": n_tok,
            "real_word_tokens": len(real_toks),
            "real_word_share": len(real_toks) / max(1, n_tok),
            "runs_with_real_word_winner": flagged_runs,
            "winner_prior_mean_excluding_flagged":
                bootstrap_ci(wp_clean, np.mean),
        }
    except Exception as e:  # pragma: no cover
        out["token_audit"] = {"error": repr(e)}
    out["winner_prior_note"] = ("mean measured zero-shot prior of the "
                                "eventual winner; 0.1 = winner independent "
                                "of prior (W=10)")

    # framing invariance: consensus time by framing
    by_framing = {}
    for r in runs:
        g = r["summary"].get("genesis")
        if g and g.get("converged"):
            by_framing.setdefault(r["config"]["framing"], []).append(
                g["consensus_time"])
    out["framing_consensus_times"] = {
        k: bootstrap_ci(v, np.median) for k, v in by_framing.items()}
    if len(by_framing) >= 2 and all(len(v) >= 2 for v in by_framing.values()):
        kw = sps.kruskal(*by_framing.values())
        out["framing_invariance_kw"] = {"stat": float(kw.statistic),
                                        "p": float(kw.pvalue)}

    # --- genesis -----------------------------------------------------------
    gen_runs = [r for r in runs if r["config"]["phase"] == "genesis"]
    conv = [r for r in gen_runs if r["summary"].get("genesis", {}).get("converged")]
    out["genesis"] = {
        "n_runs": len(gen_runs),
        "convergence_rate": binomial_ci(len(conv), len(gen_runs)),
        "consensus_time": bootstrap_ci(
            [r["summary"]["genesis"]["consensus_time"] for r in conv],
            np.median),
    }

    # --- transmission ablation: socialisation speed ------------------------
    soc = {}
    for dlg, cellkey in ((True, "b_trans_dlg"), (False, "b_trans_nodlg")):
        nc_all, founders_all, censored = [], [], 0
        survived, usable = 0, 0
        for r in runs:
            if not r["name"].startswith(cellkey):
                continue
            g = r["summary"].get("genesis", {})
            tr = r["summary"].get("transmission")
            if not (g.get("converged") and tr):
                continue
            usable += 1
            survived += bool(tr["survived"])
            f, n = _b_conformity(r, g["winner"])
            founders_all.extend(c for c in f if c is not None)
            nc_all.extend(c for c in n if c is not None)
            censored += sum(1 for c in n if c is None)
        soc[f"dialogue_{'on' if dlg else 'off'}"] = {
            "n_runs": usable,
            "survived": binomial_ci(survived, usable),
            "newcomer_conformity": bootstrap_ci(nc_all, np.median),
            "newcomer_censored": censored,
            "founder_conformity": bootstrap_ci(founders_all, np.median),
            "_nc_raw": nc_all,
        }
    if soc["dialogue_on"]["_nc_raw"] and soc["dialogue_off"]["_nc_raw"]:
        mw = sps.mannwhitneyu(soc["dialogue_on"]["_nc_raw"],
                              soc["dialogue_off"]["_nc_raw"])
        out["socialisation_ablation_mannwhitney_p"] = float(mw.pvalue)
    out["socialisation"] = {k: {kk: vv for kk, vv in v.items()
                                if not kk.startswith("_")}
                            for k, v in soc.items()}

    # --- enforcement: normative share vs convention age --------------------
    enforcement = {"gated": True,
                   "gate_note": ("enforcement results are reportable only "
                                 "after judge validation (kappa >= 0.6 on "
                                 "the hand-labelled sample)")}
    if len(labels):
        lab = labels[~labels["solitary"]].copy()
        # attach consensus time + run phase
        consensus_by_run = {}
        for r in runs:
            g = r["summary"].get("genesis")
            if g and g.get("converged"):
                consensus_by_run[r["name"]] = g["consensus_time"]
        lab = lab[lab["run"].isin(consensus_by_run)]
        lab["age"] = lab.apply(lambda x: x["t"] - consensus_by_run[x["run"]],
                               axis=1)
        lab["normative"] = (lab["label"] == "normative").astype(int)
        bins = [-10**9, -1, 100, 250, 500, 10**9]
        names = ["pre-consensus", "0-100", "100-250", "250-500", "500+"]
        lab["age_bin"] = np.digitize(lab["age"], bins[1:-1])
        share_by_bin = {}
        for bi, bn in enumerate(names):
            sel = lab[lab["age_bin"] == bi]
            if len(sel):
                # run-level: mean of per-run shares (clustered)
                per_run = sel.groupby("run")["normative"].mean()
                share_by_bin[bn] = bootstrap_ci(per_run.values, np.mean)
        enforcement["normative_share_by_age"] = share_by_bin
        post = lab[lab["age"] >= 0]
        if len(post) > 10:
            enforcement["age_trend_cluster_bootstrap"] = \
                cluster_bootstrap_slope(post["run"].values,
                                        post["age"].values,
                                        post["normative"].values)
        # asymmetry: incumbent -> newcomer vs newcomer -> incumbent
        born = {}
        for r in runs:
            for aid, a in r["agents"].items():
                born[(r["name"], aid)] = a["born_at"]
        trans_lab = labels[labels["run"].str.startswith("b_trans_dlg")].copy()
        if len(trans_lab):
            def is_newcomer(row):
                return born.get((row["run"], row["sender"]), 0) > 0
            trans_lab["from_newcomer"] = trans_lab.apply(is_newcomer, axis=1)
            inc = trans_lab[~trans_lab["from_newcomer"]]
            new = trans_lab[trans_lab["from_newcomer"]]
            enforcement["asymmetry"] = {
                "incumbent_messages": int(len(inc)),
                "incumbent_normative_share": binomial_ci(
                    int((inc["label"] == "normative").sum()), len(inc)),
                "newcomer_messages": int(len(new)),
                "newcomer_normative_share": binomial_ci(
                    int((new["label"] == "normative").sum()), len(new)),
            }
        # solitary control — interpretively load-bearing
        sol = labels[labels["solitary"]]
        enforcement["solitary_control"] = {
            "n_messages": int(len(sol)),
            "normative_share": binomial_ci(
                int((sol["label"] == "normative").sum()), len(sol)),
        }
        out["judge_label_counts"] = labels["label"].value_counts().to_dict()
    out["enforcement"] = enforcement

    # judge validation status
    val_csv = os.path.join(bdir, "validation_sample_TO_HAND_LABEL.csv")
    from ..expb.judge import compute_agreement
    agree = compute_agreement(val_csv) if os.path.exists(val_csv) else \
        {"n": 0, "note": "validation sample not found"}
    out["judge_validation"] = agree
    if mode == "mock":
        out["judge_validation"]["mock_pipeline_check"] = \
            _mock_judge_check(labels)

    _figures_b(figdir, runs, out, soc, enforcement)
    return out


def _mock_judge_check(labels) -> dict:
    """In mock mode message texts come from templates with known intended
    classes; measure rule-judge agreement against that ground truth."""
    from ..expb.backend import MOCK_MESSAGES
    lookup = {}
    for cls, tmpls in MOCK_MESSAGES.items():
        for t in tmpls:
            pat = "^" + re.escape(t).replace(r"\{name\}", r"[a-z]+") + "$"
            lookup[pat] = cls
    n, agree = 0, 0
    for _, row in labels.iterrows():
        for pat, cls in lookup.items():
            if re.match(pat, row["text"]):
                n += 1
                agree += int(cls == row["label"])
                break
    return {"n_matched": n, "agreement": agree / n if n else None}


def _figures_b(figdir, runs, out, soc, enforcement):
    # ---- fig1: convergence curves A vs B ------------------------------
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    # B trajectories from journals (interaction success, rolling mean 50)
    curves = []
    for r in runs:
        if r["config"]["phase"] != "genesis":
            continue
        s = np.array([int(x[1]) for x in r["interactions"]], dtype=float)
        if len(s) >= 100:
            w = np.convolve(s, np.ones(50) / 50, mode="valid")
            curves.append(w)
    if curves:
        L = min(len(c) for c in curves)
        M = np.vstack([c[:L] for c in curves])
        med = np.median(M, axis=0)
        lo = np.percentile(M, 25, axis=0)
        hi = np.percentile(M, 75, axis=0)
        x = np.arange(L)
        ax.fill_between(x, lo, hi, color=C[0], alpha=0.18, lw=0)
        ax.plot(x, med, color=C[0], lw=1.8,
                label=f"LLM-tier pipeline ({'mock' if out['mode']=='mock' else 'live'}, n={len(curves)})")
    # A trajectories from genesis_core keep_trajectory runs
    a_traj = [r["success_traj_100"] for r in
              load.load_cell("results/expA", "genesis_core")
              if r.get("success_traj_100")]
    if a_traj:
        L = min(len(tt) for tt in a_traj if len(tt)) if any(a_traj) else 0
        M = np.vstack([tt[:L] for tt in a_traj if len(tt) >= L])
        x = np.arange(L) * 100 + 50   # bin centres of 100-interaction bins
        ax.plot(x, np.median(M, axis=0), color=GRAY, lw=1.8,
                label=f"minimal agents (n={len(M)})")
        ax.fill_between(x, np.percentile(M, 25, axis=0),
                        np.percentile(M, 75, axis=0), color=GRAY,
                        alpha=0.15, lw=0)
        ax.set_xlim(0, 800)
    ax.set_xlabel("interaction")
    ax.set_ylabel("success rate (rolling)")
    ax.set_ylim(0, 1.02)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("Convergence to a shared name (median, IQR band)",
                 fontsize=9)
    _save(fig, figdir, "fig1_convergence.png")

    # ---- fig4b: socialisation ablation (B) ----------------------------
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    data, ticklabels = [], []
    for key, label, color in (("dialogue_on", "channel on", C[0]),
                              ("dialogue_off", "channel off", C[3])):
        raw = soc[key]["_nc_raw"]
        if raw:
            data.append(raw)
            ticklabels.append(f"{label}\n(n={len(raw)})")
    if data:
        parts = ax.violinplot(data, showmedians=True, widths=0.7)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor([C[0], C[3]][i])
            pc.set_alpha(0.45)
        for k in ("cmedians", "cbars", "cmins", "cmaxes"):
            parts[k].set_color(GRAY)
            parts[k].set_linewidth(1.1)
        ax.set_xticks(range(1, len(data) + 1))
        ax.set_xticklabels(ticklabels, fontsize=8)
    p = out.get("socialisation_ablation_mannwhitney_p")
    ax.set_ylabel("newcomer plays until conformity")
    ax.set_title("Newcomer socialisation, dialogue ablation"
                 + (f"  (MW p = {p:.3f})" if p is not None else ""),
                 fontsize=9)
    ax.grid(axis="x", visible=False)
    _save(fig, figdir, "fig4b_socialisation_B.png")

    # ---- fig5: normative share vs convention age ----------------------
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    share = enforcement.get("normative_share_by_age", {})
    if share:
        names = list(share)
        est = [share[n]["estimate"] for n in names]
        lo = [max(0.0, share[n]["estimate"] - share[n]["lo"]) for n in names]
        hi = [max(0.0, share[n]["hi"] - share[n]["estimate"]) for n in names]
        ax.errorbar(range(len(names)), est, yerr=[lo, hi], fmt="o-",
                    color=C[0], lw=1.5, ms=5, capsize=3,
                    label="population runs")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8)
    solitary = enforcement.get("solitary_control", {})
    if solitary.get("normative_share", {}).get("p") is not None:
        sp = solitary["normative_share"]
        ax.axhspan(sp["lo"], sp["hi"], color=C[3], alpha=0.15, lw=0)
        ax.axhline(sp["p"], color=C[3], lw=1.5, ls="--",
                   label=f"solitary control (n={solitary['n_messages']})")
    ax.set_xlabel("convention age at message (interactions since consensus)")
    ax.set_ylabel("normative share of messages")
    trend = enforcement.get("age_trend_cluster_bootstrap") or {}
    sub = (f"slope={trend.get('slope'):.2e}, p={trend.get('p_two_sided'):.3f} "
           "(run-clustered)") if trend.get("slope") is not None else ""
    ax.set_title("Normative utterances vs convention age  " + sub, fontsize=9)
    ax.legend(fontsize=8)
    _save(fig, figdir, "fig5_enforcement.png")


# ===========================================================================

def main_analysis(results_dir: str, figdir: str) -> None:
    summary = {}
    adir = os.path.join(results_dir, "expA")
    if os.path.isdir(adir):
        print("analysing Experiment A ...")
        summary["expA"] = analyze_a(adir, figdir)
    for mode in ("mock", "live"):
        bdir = os.path.join(results_dir, f"expB_{mode}")
        if os.path.isdir(bdir):
            print(f"analysing Experiment B ({mode}) ...")
            summary[f"expB_{mode}"] = analyze_b(bdir, figdir, mode)
    outpath = os.path.join(results_dir, "analysis_summary.json")
    with open(outpath, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"wrote {outpath} and figures to {figdir}/")
