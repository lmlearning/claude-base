"""Interaction engine and trailing-window consensus tracking.

Strictly sequential pairwise interactions (the published dynamics): at each
step two distinct agents are drawn uniformly at random, each independently
produces a name, both observe the outcome.  A round-based parallel
scheduler is also provided for the LLM-throughput check (see §2 of the
protocol); its equivalence is tested on the cheap tier.
"""

from __future__ import annotations

from collections import Counter, deque

import numpy as np

from .agents import Agent
from .config import GameConfig


class WindowTracker:
    """Incrementally maintains trailing-window consensus statistics."""

    def __init__(self, cfg: GameConfig):
        self.cfg = cfg
        self.window = cfg.consensus_window
        self.records = deque()              # (name_a, name_b, success, a_committed, b_committed)
        self.successes = 0
        self.prod_counts = Counter()        # all productions
        self.nc_prod_counts = Counter()     # productions by non-committed agents

    def push(self, na: int, nb: int, success: bool,
             a_committed: bool = False, b_committed: bool = False) -> None:
        self.records.append((na, nb, success, a_committed, b_committed))
        self.successes += success
        self.prod_counts[na] += 1
        self.prod_counts[nb] += 1
        if not a_committed:
            self.nc_prod_counts[na] += 1
        if not b_committed:
            self.nc_prod_counts[nb] += 1
        if len(self.records) > self.window:
            oa, ob, osucc, oac, obc = self.records.popleft()
            self.successes -= osucc
            self.prod_counts[oa] -= 1
            self.prod_counts[ob] -= 1
            if not oac:
                self.nc_prod_counts[oa] -= 1
            if not obc:
                self.nc_prod_counts[ob] -= 1

    @property
    def full(self) -> bool:
        return len(self.records) == self.window

    def success_rate(self) -> float:
        return self.successes / max(1, len(self.records))

    def dominant(self) -> tuple[int, float]:
        total = sum(self.prod_counts.values())
        if total == 0:
            return -1, 0.0
        name, cnt = self.prod_counts.most_common(1)[0]
        return name, cnt / total

    def consensus(self) -> int | None:
        """Return the consensus name if the pre-registered criterion holds."""
        if not self.full:
            return None
        if self.success_rate() < self.cfg.consensus_success_threshold:
            return None
        name, share = self.dominant()
        if share >= self.cfg.consensus_dominance_threshold:
            return name
        return None

    def noncommitted_dominant(self) -> tuple[int, float]:
        total = sum(self.nc_prod_counts.values())
        if total == 0:
            return -1, 0.0
        name, cnt = self.nc_prod_counts.most_common(1)[0]
        return name, cnt / total


class Population:
    def __init__(self, agents: list[Agent], cfg: GameConfig,
                 rng: np.random.Generator):
        self.agents = agents            # index = slot
        self.cfg = cfg
        self.rng = rng
        self.tracker = WindowTracker(cfg)
        self.t = 0                      # interactions played
        self.success_log: list[bool] = []   # per-interaction success (for trajectories)

    def step(self) -> tuple[int, int, int, int, bool]:
        """One sequential interaction. Returns (i, j, name_i, name_j, success)."""
        n = len(self.agents)
        i = int(self.rng.integers(n))
        j = int(self.rng.integers(n - 1))
        if j >= i:
            j += 1
        return self._interact(i, j)

    def _interact(self, i: int, j: int) -> tuple[int, int, int, int, bool]:
        a, b = self.agents[i], self.agents[j]
        na, nb = a.choose(), b.choose()
        success = na == nb
        payoff = self.cfg.reward if success else self.cfg.penalty
        a.observe(na, nb, payoff)
        b.observe(nb, na, payoff)
        a.record_play(na)
        b.record_play(nb)
        self.tracker.push(na, nb, success,
                          a.kind == "committed", b.kind == "committed")
        self.t += 1
        self.success_log.append(success)
        return i, j, na, nb, success

    def step_round(self) -> list[tuple[int, int, int, int, bool]]:
        """Round-based scheduler: a random perfect matching of all agents,
        all pairs played 'simultaneously' (choices collected before any
        observation).  Used only for the parallelisation-equivalence check."""
        n = len(self.agents)
        perm = self.rng.permutation(n)
        results = []
        pairs = [(int(perm[2 * k]), int(perm[2 * k + 1])) for k in range(n // 2)]
        choices = {}
        for i, j in pairs:
            choices[i] = self.agents[i].choose()
            choices[j] = self.agents[j].choose()
        for i, j in pairs:
            na, nb = choices[i], choices[j]
            success = na == nb
            payoff = self.cfg.reward if success else self.cfg.penalty
            self.agents[i].observe(na, nb, payoff)
            self.agents[j].observe(nb, na, payoff)
            self.agents[i].record_play(na)
            self.agents[j].record_play(nb)
            self.tracker.push(na, nb, success,
                              self.agents[i].kind == "committed",
                              self.agents[j].kind == "committed")
            self.t += 1
            self.success_log.append(success)
            results.append((i, j, na, nb, success))
        return results

    def clone(self) -> "Population":
        new = object.__new__(Population)
        new.cfg = self.cfg
        new.rng = np.random.default_rng(self.rng.integers(2**63))
        new.agents = [a.clone() for a in self.agents]
        # Rebuild the tracker from the current one's records.
        new.tracker = WindowTracker(self.cfg)
        for rec in self.tracker.records:
            new.tracker.push(*rec)
        new.t = self.t
        new.success_log = []
        return new


def conformity_time(plays: list[int], target: int, block: int,
                    min_matches: int) -> int | None:
    """First 1-based own-play index k such that plays[k-1:k-1+block]
    contains >= min_matches plays of target.  None if censored."""
    n = len(plays)
    if n < block:
        return None
    matches = np.asarray(plays) == target
    csum = np.concatenate([[0], np.cumsum(matches)])
    for k in range(0, n - block + 1):
        if csum[k + block] - csum[k] >= min_matches:
            return k + 1
    return None
