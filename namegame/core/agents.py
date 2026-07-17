"""Agent policies for the cheap (non-LLM) tier.

Names are integers 0..W-1 at this layer; the LLM tier maps them to
per-run nonsense tokens.

MinimalAgent — the primary policy, the theoretical baseline of the SOTA
work: maximise expected reward under the empirical partner-play
distribution in the memory window (the name with the best empirical
coordination record).

Exact decision rule (documented choice; fixed across all conditions):
  * Memory holds the last H records (own_name, partner_name, payoff).
  * Expected payoff of playing name x is (R - P) * phat(x) + P, where
    phat(x) is the frequency of x among the PARTNER plays in memory —
    monotone in phat, so the rule is: play the name partners produced
    most often in the window; ties broken uniformly.
  * Empty memory: play uniformly at random from the full pool.
  This variant was selected empirically as the formalisation that
  reproduces the published convergent dynamics in the core condition
  (15/15 seeds, median ~270 interactions); pure own-play payoff scoring
  variants fail to converge (see RESULTS.md, decision log).

RLAgent — independent robustness rule: epsilon-greedy incremental
Q-learning over names (Q <- Q + alpha*(payoff - Q) for the played name),
Q initialised to 0 for all names (so untried names dominate failed ones,
and a success locks in), epsilon = 0.01, alpha = 0.3; no windowed
memory.  Deliberately a different learning family.

CommittedAgent — always plays its assigned name; never updates.
"""

from __future__ import annotations

from collections import deque

import numpy as np


class Agent:
    """Base agent. Subclasses implement choose(); observe() feeds memory."""

    kind = "base"

    def __init__(self, agent_id: int, n_names: int, memory_size: int,
                 rng: np.random.Generator):
        self.agent_id = agent_id
        self.n_names = n_names
        self.memory = deque(maxlen=memory_size)
        self.rng = rng
        self.plays: list[int] = []          # full own-play history (for conformity analysis)
        self.born_at: int = 0               # interaction index at insertion

    def choose(self) -> int:
        raise NotImplementedError

    def observe(self, own: int, partner: int, payoff: float) -> None:
        self.memory.append((own, partner, payoff))

    def record_play(self, name: int) -> None:
        self.plays.append(name)

    def clone(self) -> "Agent":
        """Deep-ish copy for survival probes. RNG is shared-state-copied."""
        new = object.__new__(type(self))
        new.__dict__.update(self.__dict__)
        new.memory = deque(self.memory, maxlen=self.memory.maxlen)
        new.plays = []  # probes don't contribute to conformity analysis
        new.rng = np.random.default_rng(self.rng.integers(2**63))
        self._clone_extra(new)
        return new

    def _clone_extra(self, new: "Agent") -> None:
        pass


class MinimalAgent(Agent):
    kind = "minimal"

    def choose(self) -> int:
        if not self.memory:
            return int(self.rng.integers(self.n_names))
        counts: dict[int, int] = {}
        for _own, partner, _payoff in self.memory:
            counts[partner] = counts.get(partner, 0) + 1
        best = max(counts.values())
        candidates = [n for n, c in counts.items() if c == best]
        if len(candidates) == 1:
            return candidates[0]
        return int(candidates[self.rng.integers(len(candidates))])


class RLAgent(Agent):
    kind = "rl"
    alpha = 0.3
    epsilon = 0.01

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.q = np.zeros(self.n_names)

    def choose(self) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_names))
        best = self.q.max()
        candidates = np.flatnonzero(self.q == best)
        return int(candidates[self.rng.integers(len(candidates))])

    def observe(self, own: int, partner: int, payoff: float) -> None:
        super().observe(own, partner, payoff)
        self.q[own] += self.alpha * (payoff - self.q[own])

    def _clone_extra(self, new: "Agent") -> None:
        new.q = self.q.copy()


class CommittedAgent(Agent):
    kind = "committed"

    def __init__(self, agent_id: int, n_names: int, memory_size: int,
                 rng: np.random.Generator, committed_name: int = 0):
        super().__init__(agent_id, n_names, memory_size, rng)
        self.committed_name = committed_name

    def choose(self) -> int:
        return self.committed_name

    def observe(self, own: int, partner: int, payoff: float) -> None:
        pass  # never updates


AGENT_CLASSES = {"minimal": MinimalAgent, "rl": RLAgent}
