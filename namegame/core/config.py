"""Configuration dataclasses and pre-registered criteria constants.

All quantities that could affect results are fixed here, recorded verbatim
into every run's output, and held constant across conditions.

PRE-REGISTERED CRITERIA (fixed before any data collection; see RESULTS.md):

Consensus criterion
    Over a trailing window of ``consensus_window = 4 * N`` interactions
    (window counts interactions; each contributes two name productions):
    (a) interaction success rate >= 0.95, AND
    (b) the single most-produced name accounts for >= 0.95 of all
        productions in the window.
    Consensus time = index of the first interaction at which both hold
    (the window must be full).

Individual conformity criterion (symmetric for founders and newcomers)
    An agent is *conformal from its k-th own play* if, among its own plays
    k .. k+9 (a block of 10 consecutive plays by that agent), at least 9
    equal the target name.  Time-to-conformity = smallest such k (1-based,
    counted from the agent's first play in the relevant phase: run start
    for founders, insertion for newcomers).  Target name = the run's
    eventual/established convention.  Censored if no such k exists before
    observation ends.

Survival probe (transmission checkpoints)
    At a checkpoint the population state is cloned, turnover is frozen,
    the clone runs ``settle_interactions`` further interactions, and the
    consensus criterion is applied to the clone's final trailing window.
    The convention *survived* iff consensus holds for the ORIGINAL name.

Flip criterion (committed minority)
    Among the trailing ``4 * N`` productions by NON-committed agents, the
    alternative (minority) name has share >= 0.95.  Flip time = first
    interaction at which this holds; no flip within the budget => censored.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class GameConfig:
    """Core game parameters. The published core condition is the default."""

    n_agents: int = 24
    n_names: int = 10           # W
    memory_size: int = 5        # H
    reward: float = 100.0
    penalty: float = -50.0

    # Pre-registered criteria parameters (do not vary across conditions).
    consensus_window_factor: int = 4      # window = factor * n_agents interactions
    consensus_success_threshold: float = 0.95
    consensus_dominance_threshold: float = 0.95
    conformity_block: int = 10            # own-play block length
    conformity_min_matches: int = 9       # >= matches within block

    @property
    def consensus_window(self) -> int:
        return self.consensus_window_factor * self.n_agents


@dataclass(frozen=True)
class GenesisConfig:
    max_interactions: int = 30_000
    # Interactions played after consensus detection, so that founder
    # conformity is observed on the same footing as newcomer conformity
    # (both are watched beyond the event that defines their target).
    post_consensus_interactions: int = 1_000


@dataclass(frozen=True)
class TransmissionConfig:
    """Turnover: one replacement every ``interactions_per_replacement``
    interactions.  A *generation* is a random permutation of the N slots,
    replaced one by one, so N replacements turn over 100% of the
    population exactly once.  Two generations = 2N replacements."""

    interactions_per_replacement: int = 24   # k; rate = 1/k
    generations: int = 2
    settle_interactions: int = 2_000         # survival-probe settle length
    post_interactions: int = 0               # extra main-line interactions after last replacement


@dataclass(frozen=True)
class MinorityConfig:
    n_committed: int = 2
    budget_interactions: int = 15_000


DEFAULT_GAME = GameConfig()


def asdict_config(*cfgs) -> dict:
    out = {}
    for c in cfgs:
        out[type(c).__name__] = dataclasses.asdict(c)
    return out


def dumps_config(*cfgs) -> str:
    return json.dumps(asdict_config(*cfgs), sort_keys=True)
