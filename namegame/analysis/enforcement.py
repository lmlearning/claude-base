"""Annotation-invariant enforcement analysis (final submission rebuild).

The categorical instrument failed validation (judge vs strict
second-annotator kappa ~ 0.06 on the 151-message sample), so
judge-derived category shares are dropped from the results (kept only as
an unvalidated appendix).  This module replaces them with:

 1. the pairwise Cohen's kappa matrix over all available annotators
    (LLM judge / author hand-label / strict second annotator), computed
    from the CSVs as source of truth, plus per-annotator label
    distributions;
 2. a deterministic surface-feature detector for the
    annotation-invariant properties of enforcement talk.  The lexicons
    and matching rules in ``FROZEN_FEATURES`` were frozen BEFORE any
    corpus scan (the freezing commit is the timestamp) and validated
    against the 151-message hand-labelled sample first;
 3. a scan of EVERY live dialogue message across all supplied result
    directories - populations, the solitary control, and the second
    model family - reporting per-feature prevalence with exact
    Clopper-Pearson CIs and journaling every positive hit for audit.

Positives for validation: the four-way taxonomy's "normative" category
is precisely "appeals to correctness, obligation, or the group", so a
message any annotator labelled normative is the annotator-positive set
for the deontic / correctness / group features (sanction language has no
category of its own; its validation is FP-rate only).  With annotators
agreeing mainly on the ABSENCE of these features, the load-bearing
number is the detector's false-positive rate on messages that all
annotators labelled non-normative.
"""

from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# FROZEN detector definition - do not edit after the freezing commit.
# Matching rule: case-insensitive regex search over the raw message text;
# a message is positive for a feature iff any pattern matches.
# ---------------------------------------------------------------------------

FROZEN_FEATURES: dict[str, list[str]] = {
    "deontic": [
        r"\bshould\b", r"\bmust\b", r"\bought\b", r"\bsupposed to\b",
        r"\bhave to\b", r"\bhas to\b",
    ],
    "correctness": [
        r"\bright\b", r"\bcorrect(ly)?\b", r"\bwrong\b", r"\bincorrect\b",
        r"\bproper(ly)?\b",
    ],
    "group_appeal": [
        r"\beveryone\b", r"\beverybody\b", r"\bwe all\b", r"\ball of us\b",
        r"\bothers\b", r"\bthe group\b", r"\bthe rest\b",
        r"\bmost (people|players|agents|participants)\b",
    ],
    "sanction_blame": [
        r"\bblame\b", r"\bfault\b", r"\bpenalt", r"\bpunish", r"\bsanction",
        r"\byou (failed|ruined|broke|caused)\b", r"\bbecause of you\b",
        r"\byou should have\b", r"\byou shouldn['’]t have\b",
        r"\breport you\b",
    ],
}

_COMPILED = {f: [re.compile(p, re.IGNORECASE) for p in pats]
             for f, pats in FROZEN_FEATURES.items()}


def detect(text: str) -> dict[str, bool]:
    return {f: any(p.search(text) for p in pats)
            for f, pats in _COMPILED.items()}


# ---------------------------------------------------------------------------
# annotator data
# ---------------------------------------------------------------------------

CATEGORIES = ["descriptive", "directive", "normative", "other"]


def cohens_kappa(a: list[str], b: list[str]) -> float:
    assert len(a) == len(b) and a
    po = np.mean([x == y for x, y in zip(a, b)])
    ca, cb = Counter(a), Counter(b)
    n = len(a)
    pe = sum(ca[c] * cb[c] for c in set(ca) | set(cb)) / (n * n)
    if pe == 1.0:
        return 1.0
    return float((po - pe) / (1 - pe))


def load_annotations(live_dir: str, author_csv: str | None) -> dict:
    """Rows: id, text, and one label column per available annotator."""
    rows = {}
    strict_csv = os.path.join(live_dir, "validation_sample_second_annotator.csv")
    with open(strict_csv) as f:
        for r in csv.DictReader(f):
            rows[int(r["id"])] = {
                "text": r["text"],
                "judge": r["judge_label"].strip().lower(),
                "strict": r["second_annotator_label"].strip().lower(),
            }
    author_found = False
    if author_csv and os.path.exists(author_csv):
        with open(author_csv) as f:
            for r in csv.DictReader(f):
                i = int(r["id"])
                lab = (r.get("hand_label(FILL IN)") or r.get("hand_label")
                       or "").strip().lower()
                if i in rows and lab:
                    rows[i]["author"] = lab
                    author_found = True
    return {"rows": rows, "author_found": author_found}


def kappa_matrix(rows: dict) -> dict:
    annos = ["judge", "strict"] + (
        ["author"] if any("author" in r for r in rows.values()) else [])
    out = {"annotators": annos, "n": len(rows), "pairwise_kappa": {},
           "label_distributions": {}}
    for a in annos:
        labs = [r[a] for r in rows.values() if a in r]
        out["label_distributions"][a] = dict(Counter(labs))
    for i, a in enumerate(annos):
        for b in annos[i + 1:]:
            ids = [k for k, r in rows.items() if a in r and b in r]
            out["pairwise_kappa"][f"{a}-{b}"] = round(cohens_kappa(
                [rows[k][a] for k in ids], [rows[k][b] for k in ids]), 3)
    return out


# ---------------------------------------------------------------------------
# detector validation on the hand-labelled sample
# ---------------------------------------------------------------------------

def validate_detector(rows: dict) -> dict:
    """Per-feature precision/recall against any-annotator normative
    positives, plus the FP rate on all-annotator-non-normative messages."""
    out = {}
    annos = [a for a in ("judge", "strict", "author")
             if any(a in r for r in rows.values())]
    pos_ids = {k for k, r in rows.items()
               if any(r.get(a) == "normative" for a in annos)}
    neg_ids = {k for k, r in rows.items()
               if all(r.get(a, "x") != "normative" for a in annos if a in r)}
    for feat in FROZEN_FEATURES:
        hits = {k for k, r in rows.items() if detect(r["text"])[feat]}
        tp = len(hits & pos_ids)
        fp_all_neg = len(hits & neg_ids)
        out[feat] = {
            "hits": len(hits),
            "precision_vs_any_normative": (round(tp / len(hits), 3)
                                           if hits else None),
            "recall_vs_any_normative": (round(tp / len(pos_ids), 3)
                                        if pos_ids else None),
            "fp_rate_on_all_negative": round(
                fp_all_neg / max(1, len(neg_ids)), 4),
            "fp_examples": [rows[k]["text"][:90] for k in
                            sorted(hits & neg_ids)[:5]],
        }
    out["_n_any_normative"] = len(pos_ids)
    out["_n_all_non_normative"] = len(neg_ids)
    return out


# ---------------------------------------------------------------------------
# full-corpus scan
# ---------------------------------------------------------------------------

def exact_ci(k: int, n: int, alpha: float = 0.05) -> dict:
    """Clopper-Pearson exact binomial CI."""
    if n == 0:
        return {"k": 0, "n": 0, "p": None, "lo": None, "hi": None}
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1 - alpha / 2, k + 1, n - k))
    return {"k": int(k), "n": int(n), "p": round(k / n, 5),
            "lo": round(lo, 5), "hi": round(hi, 5)}


def iter_messages(live_dir: str):
    """Every dialogue message in an expB result dir, from the journals
    (source of truth), with its run and solitary flag."""
    for d in sorted(os.listdir(live_dir)):
        jp = os.path.join(live_dir, d, "journal.jsonl")
        if not os.path.isdir(os.path.join(live_dir, d)) or \
                not os.path.exists(jp):
            continue
        solitary = d.startswith("b_solitary")
        with open(jp) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("type") != "interaction":
                    continue
                for key in ("msg_i", "msg_j"):
                    if rec.get(key):
                        yield {"run": d, "t": rec["t"], "text": rec[key],
                               "solitary": solitary}


def scan_corpus(live_dirs: dict[str, str], hits_path: str) -> dict:
    """live_dirs: label -> path (e.g. {'haiku': 'results/expB_live',
    'gpt4omini': 'results/expB_live_m2'}). Writes every positive hit to
    hits_path for audit; returns prevalence with exact CIs."""
    out = {}
    with open(hits_path, "w") as hf:
        for label, path in live_dirs.items():
            if not os.path.isdir(path):
                continue
            for arm in ("population", "solitary"):
                msgs = [m for m in iter_messages(path)
                        if m["solitary"] == (arm == "solitary")]
                n = len(msgs)
                feat_counts = Counter()
                for m in msgs:
                    d = detect(m["text"])
                    for f, hit in d.items():
                        if hit:
                            feat_counts[f] += 1
                            hf.write(json.dumps({"family": label,
                                                 "arm": arm, **m,
                                                 "feature": f}) + "\n")
                out[f"{label}_{arm}"] = {
                    "n_messages": n,
                    **{f: exact_ci(feat_counts[f], n)
                       for f in FROZEN_FEATURES}}
    return out


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------

def main_enforcement(results_dir: str = "results",
                     author_csv: str = "validation_sample_HAND_LABELED.csv",
                     extra_dirs: dict[str, str] | None = None) -> dict:
    live = os.path.join(results_dir, "expB_live")
    ann = load_annotations(live, author_csv)
    out = {
        "author_sheet_found": ann["author_found"],
        "kappa": kappa_matrix(ann["rows"]),
        "detector_frozen_lexicons": FROZEN_FEATURES,
        "detector_validation": validate_detector(ann["rows"]),
    }
    dirs = {"haiku": live}
    dirs.update(extra_dirs or {})
    out["prevalence"] = scan_corpus(
        dirs, os.path.join(results_dir, "enforcement_feature_hits.jsonl"))
    path = os.path.join(results_dir, "enforcement_analysis.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}")
    if not ann["author_found"]:
        print("!! AUTHOR HAND-LABEL SHEET NOT FOUND "
              f"({author_csv}); kappa matrix computed over available "
              "annotators only - three-way matrix pending the sheet.")
    return out


if __name__ == "__main__":
    main_enforcement()
