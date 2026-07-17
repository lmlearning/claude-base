import numpy as np

from namegame.core.config import (GameConfig, GenesisConfig, MinorityConfig,
                                  TransmissionConfig)
from namegame.core.engine import WindowTracker, conformity_time
from namegame.core import phases


def test_conformity_time():
    assert conformity_time([1] * 10, 1, 10, 9) == 1
    assert conformity_time([0] + [1] * 10, 1, 10, 9) == 1
    assert conformity_time([0, 0] + [1] * 10, 1, 10, 9) == 2  # 9/10 at k=2
    assert conformity_time([0] * 20, 1, 10, 9) is None
    assert conformity_time([1] * 5, 1, 10, 9) is None  # too short => censored


def test_window_tracker_consensus():
    cfg = GameConfig(n_agents=2)  # window = 8
    tr = WindowTracker(cfg)
    for _ in range(8):
        tr.push(3, 3, True)
    assert tr.consensus() == 3
    tr.push(1, 2, False)  # 7/8 success = 0.875 < 0.95
    assert tr.success_rate() < 0.95
    assert tr.consensus() is None


def test_genesis_deterministic_and_converges():
    game, gen = GameConfig(), GenesisConfig()
    r1 = phases.genesis(game, gen, seed=42)
    r2 = phases.genesis(game, gen, seed=42)
    r1.pop("_population"); r2.pop("_population")
    assert r1 == r2
    assert r1["converged"]
    assert r1["winner"] in range(game.n_names)
    assert all(c is not None for c in r1["founder_conformity"])


def test_transmission_slow_rate_survives():
    game, gen = GameConfig(), GenesisConfig()
    trans = TransmissionConfig(interactions_per_replacement=64, generations=2)
    r = phases.transmission(game, gen, trans, seed=7)
    assert r["converged_genesis"]
    assert r["survived_gen2"] is not None
    assert len(r["newcomers"]) == 2 * game.n_agents


def test_minority_extremes():
    game, gen = GameConfig(), GenesisConfig()
    r_small = phases.committed_minority(game, gen, MinorityConfig(n_committed=1),
                                        seed=3)
    assert r_small["usable"] and r_small["flipped"] is False
    r_big = phases.committed_minority(game, gen, MinorityConfig(n_committed=12),
                                      seed=3)
    assert r_big["usable"] and r_big["flipped"] is True


def test_transplant_runs():
    game, gen = GameConfig(), GenesisConfig()
    for seed in range(5):
        r = phases.transplant(game, gen, seed=seed)
        if r.get("usable"):
            assert r["host_still_consensus"] in (True, False)
            return
    raise AssertionError("no usable transplant in 5 seeds")
