"""Shared plumbing for the E1-E4 'conventions as side products' environments.

Design contract (per environment module):
  * a PROMPTS dict of >= 2 paraphrase framings with identical mechanics,
    linted at import against the extended banned-vocabulary list (nothing
    may request consistency, style, order, or agreement — the task rewards
    accomplishment only);
  * ``run_population(config, backend)`` executing formation -> turnover
    (-> minority where designed), journaling every interaction before the
    next begins, resumable by journal replay;
  * ``mock_reply(context, rng)`` — the cheap-tier policy behind the same
    text interface (substrate control);
  * extractor functions mapping raw transcripts to convention variables
    (used by both the runs and the analysis battery).

The convention battery itself (concentration, cross-population diversity,
prior correction, stranger-pool baseline, turnover adoption, minority
threshold) is computed post-hoc in ``namegame.envs.analysis``.
"""

from __future__ import annotations

import json
import os
import re

import numpy as np

from ..expb import prompts as bprompts
from ..expb.backend import Backend

# Extended lint: the base banned stems plus anything that could prime
# consistency/order/style as a goal.
EXTRA_BANNED = ["consist", "standard", "protocol", "style", "habit",
                "usual", "typical", "tradition", "uniform", "align",
                "in the same way", "stick to", "convention", "pattern"]


def lint_env(texts: list[str]) -> None:
    problems = []
    for t in texts:
        hits = bprompts.lint(t) + [s for s in EXTRA_BANNED if s in t.lower()]
        if hits:
            problems.append((t[:60], sorted(set(hits))))
    if problems:
        raise AssertionError(f"banned vocabulary in env prompts: {problems}")


def rng_for(seed: int, *tags: int) -> np.random.Generator:
    return np.random.default_rng([seed, *tags])


class Journal:
    def __init__(self, run_dir: str):
        os.makedirs(run_dir, exist_ok=True)
        self.path = os.path.join(run_dir, "journal.jsonl")

    def append(self, rec: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def replay(self):
        if not os.path.exists(self.path):
            return
        with open(self.path) as f:
            for line in f:
                yield json.loads(line)


class EnvMockBackend(Backend):
    """Cheap-tier policy behind the LLM text interface.  The env supplies
    ``mock_reply(context, rng)``; prompts are still built and replies still
    parsed by the same code paths as live."""

    is_mock = True

    def __init__(self, mock_reply, seed: int):
        self.mock_reply = mock_reply
        self.rng = np.random.default_rng(seed)

    def complete(self, system: str, user: str, max_tokens: int,
                 context: dict | None = None) -> str:
        assert context is not None
        return self.mock_reply(context, self.rng)


def run_comprehension(backend: Backend, system: str,
                      questions: list[tuple[str, str]],
                      journal: Journal, reps: int = 2) -> bool:
    """questions: (text, expected). Graded on first number / first word."""
    results = []
    for rep in range(reps):
        for qi, (q, expected) in enumerate(questions):
            reply = backend.complete(system, q, 40,
                                     {"kind": "comprehension",
                                      "true_answer": expected})
            m = re.search(r"-?\d+", reply.replace(",", ""))
            got = m.group() if m else (reply.strip().split() or [""])[0]
            ok = got.lower().lstrip("-") == str(expected).lower().lstrip("-")
            results.append({"rep": rep, "q": qi, "answer": reply[:80],
                            "pass": bool(ok)})
    passed = all(r["pass"] for r in results)
    journal.append({"type": "comprehension", "results": results,
                    "passed": passed})
    return passed


def generation_schedule(n_agents: int, seed: int, tag: int) -> list[int]:
    """One generation = a random permutation of slots (each replaced once)."""
    return [int(s) for s in rng_for(seed, 41, tag).permutation(n_agents)]


def word_count_clip(text: str, max_words: int) -> tuple[str, bool]:
    words = text.strip().split()
    if len(words) <= max_words:
        return text.strip(), False
    return " ".join(words[:max_words]), True
