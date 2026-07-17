"""Experiment A runner: cell definitions, multiprocessing, JSONL output.

Each *cell* is (phase, configs, n_runs).  Per-run seeds come from a
numpy SeedSequence spawned from the master seed, so every cell is fully
deterministic and any single run can be reproduced from its recorded seed.
Results are one JSONL file per cell (one line per run) plus a meta.json
with the resolved configuration and code version.
"""

from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from ..core.config import (GameConfig, GenesisConfig, MinorityConfig,
                           TransmissionConfig, asdict_config)
from ..core import phases

MASTER_SEED = 20260717  # recorded master seed for the whole of Experiment A


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            cwd=os.path.dirname(__file__)).strip()
    except Exception:
        return "unknown"


def _cell_seeds(cell_name: str, n_runs: int) -> list[int]:
    ss = np.random.SeedSequence([MASTER_SEED,
                                 zlib.crc32(cell_name.encode())])
    return [int(s.generate_state(1)[0]) for s in ss.spawn(n_runs)]


def define_cells() -> dict[str, dict]:
    """The full Experiment A design. Fixed; recorded into meta.json."""
    core = GameConfig()
    gen = GenesisConfig()
    cells: dict[str, dict] = {}

    # --- Genesis (primary + robustness + scheduler check) -----------------
    cells["genesis_core"] = dict(phase="genesis", game=core, gen=gen,
                                 agent_kind="minimal", n_runs=1000,
                                 keep_traj_first=100)
    cells["genesis_rl"] = dict(phase="genesis", game=core, gen=gen,
                               agent_kind="rl", n_runs=300, keep_traj_first=50)
    cells["genesis_parallel_check"] = dict(phase="genesis", game=core, gen=gen,
                                           agent_kind="minimal", n_runs=500,
                                           scheduler="round", keep_traj_first=50)

    # --- Transmission: turnover-rate sweep --------------------------------
    for k in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        cells[f"transmission_core_k{k}"] = dict(
            phase="transmission", game=core, gen=gen,
            trans=TransmissionConfig(interactions_per_replacement=k),
            agent_kind="minimal", n_runs=500)
    # super-fast turnover (r replacements per single interaction) to locate
    # the phase boundary, which lies above 1 replacement / 2 interactions
    for r in [2, 3, 4, 6, 8, 12]:
        cells[f"transmission_fast_r{r}"] = dict(
            phase="transmission", game=core, gen=gen,
            trans=TransmissionConfig(interactions_per_replacement=1,
                                     replacements_per_event=r),
            agent_kind="minimal", n_runs=500)
    for k in [4, 16, 64]:
        cells[f"transmission_rl_k{k}"] = dict(
            phase="transmission", game=core, gen=gen,
            trans=TransmissionConfig(interactions_per_replacement=k),
            agent_kind="rl", n_runs=300)

    # --- Committed minority: founder vs post-transmission -----------------
    committed_counts = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]
    post_trans = TransmissionConfig(interactions_per_replacement=64)
    for c in committed_counts:
        cells[f"minority_founder_c{c}"] = dict(
            phase="minority", game=core, gen=gen,
            mino=MinorityConfig(n_committed=c), agent_kind="minimal",
            n_runs=500)
        cells[f"minority_posttrans_c{c}"] = dict(
            phase="minority", game=core, gen=gen,
            mino=MinorityConfig(n_committed=c), agent_kind="minimal",
            n_runs=500, pre_transmission=post_trans)
    for c in [2, 4, 6, 8]:
        cells[f"minority_rl_founder_c{c}"] = dict(
            phase="minority", game=core, gen=gen,
            mino=MinorityConfig(n_committed=c), agent_kind="rl", n_runs=300)

    # --- Transplant -------------------------------------------------------
    cells["transplant_core"] = dict(phase="transplant", game=core, gen=gen,
                                    agent_kind="minimal", n_runs=500)

    # --- Population-size sweep (smaller statistics) -----------------------
    for n in [12, 48]:
        game_n = GameConfig(n_agents=n)
        cells[f"genesis_N{n}"] = dict(phase="genesis", game=game_n, gen=gen,
                                      agent_kind="minimal", n_runs=300)
        for k in [8, 32, 128]:
            cells[f"transmission_N{n}_k{k}"] = dict(
                phase="transmission", game=game_n, gen=gen,
                trans=TransmissionConfig(interactions_per_replacement=k),
                agent_kind="minimal", n_runs=200)
        for c in sorted({max(1, round(f * n)) for f in
                         [0.04, 0.08, 0.125, 0.17, 0.25, 0.33, 0.5]}):
            cells[f"minority_N{n}_c{c}"] = dict(
                phase="minority", game=game_n, gen=gen,
                mino=MinorityConfig(n_committed=c), agent_kind="minimal",
                n_runs=200)
    return cells


def _run_one(args: tuple) -> dict:
    cell, seed, run_index = args
    phase = cell["phase"]
    game, gen = cell["game"], cell["gen"]
    kind = cell["agent_kind"]
    if phase == "genesis":
        keep = run_index < cell.get("keep_traj_first", 0)
        r = phases.genesis(game, gen, seed, kind,
                           scheduler=cell.get("scheduler", "sequential"),
                           keep_trajectory=keep)
        r.pop("_population", None)
    elif phase == "transmission":
        r = phases.transmission(game, gen, cell["trans"], seed, kind)
    elif phase == "minority":
        r = phases.committed_minority(game, gen, cell["mino"], seed, kind,
                                      pre_transmission=cell.get("pre_transmission"))
    elif phase == "transplant":
        r = phases.transplant(game, gen, seed, kind)
    else:
        raise ValueError(phase)
    r["run_index"] = run_index
    return r


def run_cell(name: str, cell: dict, outdir: str, procs: int,
             force: bool = False) -> str:
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{name}.jsonl")
    done_path = path + ".done"
    if os.path.exists(done_path) and not force:
        return path
    seeds = _cell_seeds(name, cell["n_runs"])
    jobs = [(cell, seeds[i], i) for i in range(cell["n_runs"])]
    t0 = time.time()
    with open(path, "w") as f, ProcessPoolExecutor(max_workers=procs) as ex:
        for r in ex.map(_run_one, jobs, chunksize=max(1, len(jobs) // (procs * 8))):
            f.write(json.dumps(r) + "\n")
    meta = {
        "cell": name,
        "master_seed": MASTER_SEED,
        "git_rev": _git_rev(),
        "n_runs": cell["n_runs"],
        "elapsed_s": round(time.time() - t0, 1),
        "config": {k: (dataclasses.asdict(v) if dataclasses.is_dataclass(v) else v)
                   for k, v in cell.items() if k not in ("game", "gen")},
        **asdict_config(cell["game"], cell["gen"]),
    }
    with open(os.path.join(outdir, f"{name}.meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    with open(done_path, "w") as f:
        f.write("ok\n")
    return path


def run_all(outdir: str, procs: int = 4, only: list[str] | None = None,
            force: bool = False) -> None:
    cells = define_cells()
    names = only if only else list(cells)
    for i, name in enumerate(names):
        if name not in cells:
            raise KeyError(f"unknown cell {name}")
        t0 = time.time()
        run_cell(name, cells[name], outdir, procs, force)
        print(f"[{i+1}/{len(names)}] {name} done in {time.time()-t0:.1f}s",
              flush=True)
