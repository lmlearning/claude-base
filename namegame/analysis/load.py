"""Loaders: analysis consumes transcripts/JSONL only, never live state."""

from __future__ import annotations

import json
import os
from collections import deque

import pandas as pd


def load_cell(results_dir: str, cell: str) -> list[dict]:
    path = os.path.join(results_dir, f"{cell}.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f]


def cells_matching(results_dir: str, prefix: str) -> dict[str, list[dict]]:
    out = {}
    for fn in sorted(os.listdir(results_dir)):
        if fn.startswith(prefix) and fn.endswith(".jsonl"):
            cell = fn[:-6]
            out[cell] = load_cell(results_dir, cell)
    return out


# ---------------------------------------------------------------------------
# Experiment B journal replay (pure transcript consumption)
# ---------------------------------------------------------------------------

def load_b_run(run_dir: str) -> dict | None:
    """Parse one B run: config, summary, per-agent play histories with
    birth times, replacement list, message list, per-interaction success."""
    jp = os.path.join(run_dir, "journal.jsonl")
    sp = os.path.join(run_dir, "summary.json")
    if not (os.path.exists(jp) and os.path.exists(sp)):
        return None
    with open(sp) as f:
        summary = json.load(f)
    config = None
    agents: dict[int, dict] = {}
    slot_agent: dict[int, int] = {}
    replacements = []
    messages = []
    interactions = []

    def ensure(agent_id: int, born_at: int):
        if agent_id not in agents:
            agents[agent_id] = {"plays": [], "born_at": born_at}

    with open(jp) as f:
        for line in f:
            rec = json.loads(line)
            typ = rec["type"]
            if typ == "config":
                config = rec
                for s in range(rec["n_agents"]):
                    slot_agent[s] = s
                    ensure(s, 0)
            elif typ == "replacement":
                slot_agent[rec["slot"]] = rec["agent_id"]
                ensure(rec["agent_id"], rec["t"])
                replacements.append(rec)
            elif typ == "interaction":
                interactions.append((rec["t"], rec["success"]))
                ensure(rec["agent_i"], 0)
                agents[rec["agent_i"]]["plays"].append((rec["t"], rec["name_i"]))
                if rec["agent_j"] >= 0:
                    ensure(rec["agent_j"], 0)
                    agents[rec["agent_j"]]["plays"].append((rec["t"], rec["name_j"]))
                for key, sender in (("msg_i", rec["agent_i"]),
                                    ("msg_j", rec["agent_j"])):
                    if rec.get(key):
                        messages.append({"t": rec["t"], "sender": sender,
                                         "text": rec[key],
                                         "solitary": rec.get("solitary", False)})
    return {"config": config, "summary": summary, "agents": agents,
            "replacements": replacements, "messages": messages,
            "interactions": interactions,
            "name": os.path.basename(run_dir)}


def load_b_all(results_dir: str) -> list[dict]:
    runs = []
    if not os.path.isdir(results_dir):
        return runs
    for d in sorted(os.listdir(results_dir)):
        rd = os.path.join(results_dir, d)
        if os.path.isdir(rd):
            r = load_b_run(rd)
            if r is not None:
                runs.append(r)
    return runs


def load_judge_labels(results_dir: str) -> pd.DataFrame:
    path = os.path.join(results_dir, "judge_labels.jsonl")
    if not os.path.exists(path):
        return pd.DataFrame()
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)
