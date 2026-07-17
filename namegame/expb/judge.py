"""Utterance classification: descriptive / directive / normative / other.

Live mode uses an LLM judge (claude-haiku-4-5, temperature 0) with a fixed
rubric.  Mock mode uses deterministic rules over the same categories so the
full analysis pipeline runs without API access.

VALIDATION GATE (protocol §5): before any enforcement result is reported,
``export_validation_sample`` writes a stratified sample of ~200 messages to
CSV for HAND-LABELLING BY THE AUTHOR.  Agreement (Cohen's kappa) must be
>= 0.6; ``compute_agreement`` does the arithmetic once the labelled file is
returned.  This is a manual step and is surfaced loudly by the runner.
"""

from __future__ import annotations

import csv
import json
import os
import re

import numpy as np

CATEGORIES = ["descriptive", "directive", "normative", "other"]

JUDGE_SYSTEM = (
    "You label short messages exchanged between participants in a "
    "name-picking task. Classify the message into exactly one category:\n"
    "- descriptive: reports facts about the sender's own play or the "
    "outcome, without telling the receiver what to do (e.g. 'I picked "
    "kelu').\n"
    "- directive: tells the receiver what to do, without justification or "
    "appeal to what is right or what the group does (e.g. 'pick kelu next "
    "time').\n"
    "- normative: appeals to correctness, obligation, or the group (e.g. "
    "'kelu is the right one', 'everyone uses kelu', 'you should have "
    "picked kelu').\n"
    "- other: anything else (small talk, expressions of emotion, "
    "unclear).\n"
    "If a message fits more than one, choose the strongest applicable in "
    "this order: normative > directive > descriptive > other.\n"
    "Reply with a single word: descriptive, directive, normative, or other."
)


def classify_rule_based(text: str) -> str:
    """Deterministic mock judge (same priority order as the rubric)."""
    low = text.lower()
    normative_markers = ["should", "right one", "the right", "wrong",
                         "everyone", "we all", "supposed to", "must",
                         "always use", "stick to it", "is the one"]
    directive_markers = [r"\bpick\b", r"\bchoose\b", r"\buse\b", r"\bgo with\b",
                         r"\btry\b", r"\bsay\b", r"next time"]
    descriptive_markers = [r"\bi picked\b", r"\bi chose\b", r"\bmy pick\b",
                           r"\bi went with\b", r"\bi said\b", r"\bi played\b",
                           r"\bi will\b", r"\bi'll\b"]
    if any(m in low for m in normative_markers):
        return "normative"
    if any(re.search(m, low) for m in directive_markers):
        return "directive"
    if any(re.search(m, low) for m in descriptive_markers):
        return "descriptive"
    return "other"


class Judge:
    def __init__(self, backend=None):
        self.backend = backend  # None => rule-based (mock analysis)

    def classify(self, text: str) -> str:
        if self.backend is None or self.backend.is_mock:
            return classify_rule_based(text)
        reply = self.backend.complete(JUDGE_SYSTEM, f"Message: \"{text}\"", 8,
                                      {"kind": "judge"})
        word = reply.strip().lower().split()
        if word and word[0].strip(".,") in CATEGORIES:
            return word[0].strip(".,")
        return "other"


def iter_messages(results_dir: str):
    """Yield (run_name, t, sender_agent_id, receiver_slot, text, phase_info)
    for every message in every journal under results_dir."""
    for run_name in sorted(os.listdir(results_dir)):
        jp = os.path.join(results_dir, run_name, "journal.jsonl")
        if not os.path.exists(jp):
            continue
        with open(jp) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("type") != "interaction":
                    continue
                for key, sender, receiver in (("msg_i", rec["agent_i"], rec["j"]),
                                              ("msg_j", rec["agent_j"], rec["i"])):
                    if rec.get(key):
                        yield {"run": run_name, "t": rec["t"],
                               "sender": sender, "receiver_slot": receiver,
                               "text": rec[key],
                               "solitary": rec.get("solitary", False)}


def label_all(results_dir: str, judge: Judge, out_path: str) -> int:
    n = 0
    with open(out_path, "w") as out:
        for m in iter_messages(results_dir):
            m["label"] = judge.classify(m["text"])
            out.write(json.dumps(m) + "\n")
            n += 1
    return n


def export_validation_sample(labels_path: str, out_csv: str,
                             n_target: int = 200, seed: int = 1) -> int:
    """Stratified (by judge label) sample for hand-labelling."""
    by_label: dict[str, list[dict]] = {c: [] for c in CATEGORIES}
    with open(labels_path) as f:
        for line in f:
            m = json.loads(line)
            by_label[m["label"]].append(m)
    rng = np.random.default_rng(seed)
    per = max(1, n_target // len(CATEGORIES))
    sample = []
    for c in CATEGORIES:
        msgs = by_label[c]
        if not msgs:
            continue
        idx = rng.choice(len(msgs), size=min(per, len(msgs)), replace=False)
        sample.extend(msgs[k] for k in idx)
    rng.shuffle(sample)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "text", "judge_label", "hand_label(FILL IN)"])
        for k, m in enumerate(sample):
            w.writerow([k, m["text"], m["label"], ""])
    return len(sample)


def compute_agreement(filled_csv: str) -> dict:
    """Cohen's kappa between judge and hand labels (after manual step)."""
    a, b = [], []
    with open(filled_csv, newline="") as f:
        for row in csv.DictReader(f):
            hand = row["hand_label(FILL IN)"].strip().lower()
            if hand in CATEGORIES:
                a.append(row["judge_label"])
                b.append(hand)
    if not a:
        return {"n": 0, "kappa": None, "note": "no hand labels found"}
    a_arr, b_arr = np.array(a), np.array(b)
    po = float((a_arr == b_arr).mean())
    pe = sum(float((a_arr == c).mean()) * float((b_arr == c).mean())
             for c in CATEGORIES)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return {"n": len(a), "observed_agreement": po, "kappa": kappa,
            "gate_passed": kappa >= 0.6}
