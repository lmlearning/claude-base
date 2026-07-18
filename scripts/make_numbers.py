#!/usr/bin/env python3
"""Generate numbers.tex: one LaTeX macro per headline statistic, sourced
from the analysis JSONs so the paper never hand-transcribes a number.

Usage: python scripts/make_numbers.py > numbers.tex
Macros render as \\Macro{...}; missing inputs render \\textbf{TBD} so a
stale build is visible in the PDF, never silently wrong.
"""

import json
import os
import sys


def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


S = load("results/analysis_summary.json")
E = load("results/envs_summary.json")
F = load("results/enforcement_analysis.json")
A = S.get("expA", {})
B = S.get("expB_live", {})

MACROS: list[tuple[str, str]] = []


def m(name, value):
    MACROS.append((name, value))


def get(d, *path):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d


def num(x, fmt="{:.3f}"):
    return "\\textbf{TBD}" if x is None else fmt.format(x)


def ci(d, fmt="{:.2f}"):
    if not d or d.get("estimate") is None:
        return "\\textbf{TBD}"
    return (fmt.format(d["estimate"])
            + f" [{fmt.format(d['lo'])}, {fmt.format(d['hi'])}]")


def pci(d, fmt="{:.2f}"):
    if not d or d.get("p") is None:
        return "\\textbf{TBD}"
    return (fmt.format(d["p"])
            + f" [{fmt.format(d['lo'])}, {fmt.format(d['hi'])}]")


def x50(d, fmt="{:.3f}"):
    if not d or d.get("x50") is None:
        return "\\textbf{TBD}"
    return (fmt.format(d["x50"])
            + f" [{fmt.format(d['lo'])}, {fmt.format(d['hi'])}]")


# ---- Experiment A (substrate) ----
m("nSubstrateGenesisRuns", num(get(A, "genesis_core", "n_runs"), "{:d}"))
m("pSubstrateGenesisConv", pci(get(A, "genesis_core", "convergence_rate")))
m("tSubstrateConsensusMedian",
  num(get(A, "genesis_core", "consensus_time", "estimate"), "{:.0f}"))
m("pWinnerUniformityChiSq", num(get(A, "winner_uniformity", "p_value")))
m("kFiftyGenOne", x50(get(A, "critical_k_gen1")))
m("kFiftyGenTwo", x50(get(A, "critical_k_gen2")))
m("fFiftySubstrateFounder", x50(get(A, "f50_founder")))
m("fFiftySubstratePosttrans", x50(get(A, "f50_posttrans")))
m("fFiftySubstrateRL", x50(get(A, "f50_rl_founder")))
m("tFounderConformity", num(get(A, "founder_conformity", "estimate"), "{:.0f}"))

# ---- Experiment B (naming, live haiku) ----
m("nNamingGenesisRuns", num(get(B, "genesis", "n_runs"), "{:d}"))
m("pNamingGenesisConv", pci(get(B, "genesis", "convergence_rate")))
m("tNamingConsensusMedian", ci(get(B, "genesis", "consensus_time"), "{:.0f}"))
m("pNamingSurvivedDlgOn",
  pci(get(B, "socialisation", "dialogue_on", "survived")))
m("pNamingSurvivedDlgOff",
  pci(get(B, "socialisation", "dialogue_off", "survived")))
m("pSocialisationMWU", num(get(B, "socialisation_ablation_mannwhitney_p")))
m("rNamingMalformed", ci(get(B, "malformed_rate"), "{:.4f}"))

# ---- enforcement (annotation-invariant rebuild) ----
m("kJudgeStrict", num(get(F, "kappa", "pairwise_kappa", "judge-strict")))
m("kJudgeAuthor",
  num(get(F, "kappa", "pairwise_kappa", "judge-author")))
m("kAuthorStrict",
  num(get(F, "kappa", "pairwise_kappa", "author-strict")))
for feat, short in (("deontic", "Deontic"), ("correctness", "Correctness"),
                    ("group_appeal", "GroupAppeal"),
                    ("sanction_blame", "Sanction")):
    d = get(F, "prevalence", "haiku_population", feat)
    m(f"prev{short}Pop", pci(d, "{:.4f}") if d else "\\textbf{TBD}")
    d = get(F, "prevalence", "haiku_solitary", feat)
    m(f"prev{short}Sol", pci(d, "{:.4f}") if d else "\\textbf{TBD}")
    d = get(F, "prevalence", "gptfivemini_population", feat)
    m(f"prev{short}MTwo", pci(d, "{:.4f}") if d else "\\textbf{TBD}")
m("nEnforcementCorpus",
  num(get(F, "prevalence", "haiku_population", "n_messages"), "{:d}"))
m("nEnforcementSolitary",
  num(get(F, "prevalence", "haiku_solitary", "n_messages"), "{:d}"))

# ---- E-suite: per-cell headline stats, both tiers ----
CELL_MACROS = {
    "e1_tight": "EOneTwelve", "e1_loose": "EOneTwentyfive",
    "e1_squeeze": "EOneSqueeze", "e1_squeeze_mem": "EOneSqueezeMem",
    "e1_noisy": "EOneNoisy", "e2": "ETwo", "e2_think": "ETwoThink",
    "e2_dialogue": "ETwoDialogue", "e2_sonnet": "ETwoSonnet",
    "e3": "EThree", "e3_redo": "EThreeRedo", "e3_squeeze": "EThreeSqueeze",
    "e4": "EFour", "e4_think": "EFourThink", "e4_sonnet": "EFourSonnet",
}
for mode, suffix in (("mock", "Sub"), ("live", "Live"), ("m2", "MTwo")):
    d = E.get(mode, {})
    for cell, base in CELL_MACROS.items():
        c = d.get(cell)
        if not c:
            continue
        if cell.startswith("e1"):
            m(f"conv{base}{suffix}", pci(c.get("conventionalized")))
            m(f"agree{base}{suffix}",
              ci(c.get("inter_agent_agreement", {})))
            m(f"nvar{base}{suffix}",
              num(c.get("distinct_modal_variants"), "{:d}"))
        elif cell.startswith("e2"):
            m(f"conv{base}{suffix}", pci(c.get("conventionalized")))
            m(f"succ{base}{suffix}", ci(c.get("success_last12", {})))
        elif cell.startswith("e3"):
            m(f"schemes{base}{suffix}",
              num(c.get("distinct_schemes"), "{:d}"))
            m(f"prior{base}{suffix}", ci(c.get("modal_matches_prior", {})))
        elif cell.startswith("e4"):
            m(f"class{base}{suffix}", pci(c.get("class_share")))
            m(f"egal{base}{suffix}", pci(c.get("egalitarian_share")))
        m(f"npops{base}{suffix}", num(c.get("n_pops"), "{:d}"))

# ---- render ----
out = ["% numbers.tex - generated by scripts/make_numbers.py; DO NOT EDIT.",
       "% Every macro is sourced from the analysis JSONs."]
seen = set()
for name, val in MACROS:
    assert name not in seen, f"duplicate macro {name}"
    seen.add(name)
    assert name.isalpha(), f"non-alphabetic macro name {name}"
    out.append(f"\\newcommand{{\\{name}}}{{{val}}}")
sys.stdout.write("\n".join(out) + "\n")
