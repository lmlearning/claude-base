"""Experiment B runner: cell design, cost projection, spend cap, execution.

Cells (P0 core; P1 perturbation cells are included but run only when
explicitly selected):

  b_genesis_dlg / b_genesis_nodlg      genesis with/without dialogue
  b_trans_dlg / b_trans_nodlg          genesis + 1 generation turnover,
                                       dialogue on/off (the causal ablation
                                       for socialisation speed)
  b_solitary                           solitary control (dialogue on)
  b_minority_founder / b_minority_posttrans   (P1)

Framings rotate deterministically across runs within each cell, so
framing-invariance is testable.  A comprehension gate and zero-shot prior
measurement run before any play; pass rates and priors are journaled.

Live mode requires ANTHROPIC_API_KEY; it prints a cost projection and the
hard cap before the first call.  Mock mode exercises the identical
pipeline with the cheap-tier policy behind the LLM interface.
"""

from __future__ import annotations

import json
import os
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import prompts
from .backend import AnthropicBackend, CostTracker, MockBackend, SpendCapExceeded
from .engine import BRun
from .judge import Judge, export_validation_sample, label_all

MODEL = "claude-haiku-4-5"
MASTER_SEED_B = 20260718
FRAMING_CYCLE = ["rounds", "study", "market"]

CORE = dict(n_agents=24, n_names=10, memory_size=5)
GENESIS_CAP = 1500          # max interactions before declaring no consensus
POST_CONSENSUS = 100
TRANS_K = 8                 # interactions per replacement (1 generation)
TRANS_SETTLE = 100
SOLITARY_ROUNDS = 300


def define_cells_b() -> dict[str, dict]:
    cells = {}
    cells["b_genesis_dlg"] = dict(phase="genesis", dialogue=True, n_runs=9)
    cells["b_genesis_nodlg"] = dict(phase="genesis", dialogue=False, n_runs=6)
    cells["b_trans_dlg"] = dict(phase="transmission", dialogue=True, n_runs=6)
    cells["b_trans_nodlg"] = dict(phase="transmission", dialogue=False,
                                  n_runs=6)
    cells["b_solitary"] = dict(phase="solitary", dialogue=True, n_runs=3)
    # P1 (run with --only):
    cells["b_minority_founder"] = dict(phase="minority", dialogue=True,
                                       n_runs=6, pre_transmission=False)
    cells["b_minority_posttrans"] = dict(phase="minority", dialogue=True,
                                         n_runs=6, pre_transmission=True)
    return cells


def _run_config(cell_name: str, cell: dict, idx: int) -> dict:
    seed = int(np.random.SeedSequence(
        [MASTER_SEED_B, zlib.crc32(cell_name.encode()),
         idx]).generate_state(1)[0])
    return {"cell": cell_name, "run_index": idx, "seed": seed,
            "phase": cell["phase"], "dialogue": cell["dialogue"],
            "framing": FRAMING_CYCLE[idx % len(FRAMING_CYCLE)],
            "model": MODEL, **CORE,
            "pre_transmission": cell.get("pre_transmission", False)}


def execute_run(config: dict, outdir: str, backend) -> dict:
    run_dir = os.path.join(outdir, f"{config['cell']}_r{config['run_index']:02d}")
    done_path = os.path.join(run_dir, "summary.json")
    if os.path.exists(done_path):
        with open(done_path) as f:
            return json.load(f)
    run = BRun(run_dir, config, backend)
    run.resume()

    comp = run.comprehension_check()
    priors = run.measure_priors()
    summary = {"config": config, "comprehension_passed": comp["passed"],
               "priors": priors["counts"],
               "priors_malformed": priors["malformed"]}
    if not comp["passed"]:
        summary["aborted"] = "comprehension_gate_failed"
        _write(done_path, summary)
        return summary

    phase = config["phase"]
    if phase == "genesis":
        summary["genesis"] = run.run_genesis(GENESIS_CAP, POST_CONSENSUS)
    elif phase == "transmission":
        summary["genesis"] = run.run_genesis(GENESIS_CAP, POST_CONSENSUS)
        if summary["genesis"]["converged"]:
            summary["transmission"] = run.run_transmission(
                TRANS_K, generations=1, settle=TRANS_SETTLE)
    elif phase == "solitary":
        summary["solitary"] = run.run_solitary(SOLITARY_ROUNDS)
    elif phase == "minority":
        summary["genesis"] = run.run_genesis(GENESIS_CAP, POST_CONSENSUS)
        if summary["genesis"]["converged"]:
            if config["pre_transmission"]:
                summary["pre_transmission"] = run.run_transmission(
                    TRANS_K, generations=1, settle=TRANS_SETTLE)
            summary["minority"] = _run_minority(run)
    summary["final"] = run.summary()
    _write(done_path, summary)
    return summary


def _run_minority(run: BRun, fraction: float = 0.25,
                  budget: int = 600) -> dict:
    """P1 committed-minority: replace fraction*N agents with scripted
    committed agents that always play an alternative token.  Committed
    agents are scripted (not LLM) — they are the perturbation, not the
    subject."""
    original = run.tracker.consensus()
    if original is None:
        return {"usable": False}
    rng = np.random.default_rng([run.seed, 13])
    alts = [p for p in run.pool if p != original]
    alt = alts[int(rng.integers(len(alts)))]
    n_committed = max(1, round(fraction * run.game.n_agents))
    committed_slots = set(int(s) for s in
                          rng.choice(run.game.n_agents, size=n_committed,
                                     replace=False))
    flipped = False
    for _ in range(budget):
        t = run.t + 1
        i, j = run._random_pair(t)
        a, b = run.agents[i], run.agents[j]
        na = alt if i in committed_slots else run._choose(a, t)[0]
        nb = alt if j in committed_slots else run._choose(b, t)[0]
        success = na == nb
        rec = {"type": "interaction", "t": t, "i": i, "j": j,
               "agent_i": a.agent_id, "agent_j": b.agent_id,
               "name_i": na, "name_j": nb, "success": success,
               "malformed_i": 0, "malformed_j": 0, "retries": 0,
               "msg_i": None, "msg_j": None,
               "committed_i": i in committed_slots,
               "committed_j": j in committed_slots}
        if not success and run.dialogue:
            if i not in committed_slots:
                rec["msg_i"] = run._maybe_message(a, na, nb, t)
            if j not in committed_slots:
                rec["msg_j"] = run._maybe_message(b, nb, na, t)
        run._apply_interaction(rec)
        run._append(rec)
        # flip check: dominance of alt among all productions in the window
        name, share = run.tracker.dominant()
        if name == alt and share >= 0.9 and run.tracker.full:
            flipped = True
            break
    return {"usable": True, "original": original, "alt": alt,
            "n_committed": n_committed, "flipped": flipped,
            "interactions_used": run.t}


def _write(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# --------------------------------------------------------------------------
# Cost projection
# --------------------------------------------------------------------------

def project_cost(cells: dict[str, dict]) -> dict:
    """Upper-bound projection printed before any live call."""
    # measured from a sample rendered prompt (chars/4 heuristic + margin)
    from .tokens import generate_pool
    pool = generate_pool(CORE["n_names"], np.random.default_rng(0))
    system = prompts.build_system_prompt("rounds", pool, CORE["n_names"],
                                         CORE["memory_size"], 100, -50)
    sys_tok = len(system) / 3.5
    choice_in = sys_tok + 220   # history + instruction
    choice_out = 12
    msg_in = sys_tok + 240
    msg_out = 30
    pin, pout = 1.0e-6, 5.0e-6

    def cost_calls(n_in_calls, in_tok, out_tok):
        return n_in_calls * (in_tok * pin + out_tok * pout)

    per_run = {}
    genesis_inter = GENESIS_CAP  # upper bound; typical much less
    fail_share = 0.5             # share of failed interactions (upper bound)
    for name, cell in cells.items():
        phase = cell["phase"]
        inter = 0
        if phase in ("genesis", "transmission", "minority"):
            inter += genesis_inter + POST_CONSENSUS
        if phase == "transmission" or (phase == "minority" and
                                       cell.get("pre_transmission")):
            inter += TRANS_K * CORE["n_agents"] + TRANS_SETTLE
        if phase == "minority":
            inter += 600
        if phase == "solitary":
            inter = SOLITARY_ROUNDS
        choice_calls = inter * (1 if phase == "solitary" else 2) * 1.05
        msg_calls = (inter * fail_share *
                     (1 if phase == "solitary" else 2)) if cell["dialogue"] else 0
        c = (cost_calls(choice_calls, choice_in, choice_out) +
             cost_calls(msg_calls, msg_in, msg_out) +
             cost_calls(32, 400, 15))    # comprehension + priors
        per_run[name] = c
    total = sum(per_run[n] * c["n_runs"] for n, c in cells.items())
    judge_cost = 5000 * (300 * pin + 8 * pout)   # generous message count
    return {"per_run_usd": {k: round(v, 2) for k, v in per_run.items()},
            "runs_total_usd": round(total, 2),
            "judge_usd": round(judge_cost, 2),
            "grand_total_upper_bound_usd": round(total + judge_cost, 2)}


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main_expb(args) -> None:
    prompts.lint_all()
    cells = define_cells_b()
    if args.list:
        for name, cell in cells.items():
            print(f"{name}\t{cell['phase']}\tdialogue={cell['dialogue']}\t"
                  f"n_runs={cell['n_runs']}")
        return

    default_p0 = ["b_genesis_dlg", "b_genesis_nodlg", "b_trans_dlg",
                  "b_trans_nodlg", "b_solitary"]
    selected = args.only if args.only else default_p0
    outdir = args.outdir or f"results/expB_{args.mode}"
    os.makedirs(outdir, exist_ok=True)

    proj = project_cost({k: cells[k] for k in selected})
    print("=== Cost projection (upper bound), live pricing for "
          f"{MODEL} ===")
    print(json.dumps(proj, indent=2))
    if args.project_cost:
        return

    if args.mode == "live":
        if proj["grand_total_upper_bound_usd"] > args.spend_cap_usd:
            raise SystemExit(
                f"projection ${proj['grand_total_upper_bound_usd']} exceeds "
                f"spend cap ${args.spend_cap_usd}; refusing to start "
                "(protocol: stop and flag, do not trim silently)")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY not set; cannot run live")
        cost = CostTracker(args.spend_cap_usd,
                           os.path.join(outdir, "cost_state.json"))
        make_backend = lambda seed: AnthropicBackend(MODEL, cost)
    else:
        make_backend = lambda seed: MockBackend(seed=seed)

    jobs = []
    for name in selected:
        cell = cells[name]
        for idx in range(cell["n_runs"]):
            jobs.append(_run_config(name, cell, idx))

    def do(config):
        backend = make_backend(config["seed"])
        try:
            s = execute_run(config, outdir, backend)
            print(f"done {config['cell']} r{config['run_index']}: "
                  f"{json.dumps(s.get('genesis') or s.get('solitary') or {})}",
                  flush=True)
            return s
        except SpendCapExceeded as e:
            print(f"SPEND CAP HIT during {config['cell']} "
                  f"r{config['run_index']}: {e}", flush=True)
            raise

    if args.mode == "live":
        with ThreadPoolExecutor(max_workers=4) as ex:
            list(ex.map(do, jobs))
    else:
        for j in jobs:
            do(j)

    # Post-hoc: label every message, then surface the manual validation step.
    labels_path = os.path.join(outdir, "judge_labels.jsonl")
    jbackend = None if args.mode == "mock" else AnthropicBackend(
        MODEL, CostTracker(args.spend_cap_usd,
                           os.path.join(outdir, "cost_state.json")))
    n = label_all(outdir, Judge(jbackend), labels_path)
    print(f"judge labelled {n} messages -> {labels_path}")
    if n:
        csv_path = os.path.join(outdir, "validation_sample_TO_HAND_LABEL.csv")
        k = export_validation_sample(labels_path, csv_path)
        print("\n" + "!" * 70)
        print(f"!! MANUAL STEP REQUIRED: {k} messages exported to")
        print(f"!!   {csv_path}")
        print("!! Hand-label the last column, then run analysis; enforcement")
        print("!! results are gated on Cohen's kappa >= 0.6.")
        print("!" * 70)
