#!/usr/bin/env python3
"""Label all live messages with the LLM judge in parallel, then export the
stratified validation sample for hand labelling."""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from namegame.expb.judge import Judge, export_validation_sample, iter_messages
from namegame.expb.runner import _live_backend_factory

outdir = sys.argv[1] if len(sys.argv) > 1 else "results/expB_live"
cost, factory = _live_backend_factory(outdir, 250.0)
msgs = list(iter_messages(outdir))
print(f"labelling {len(msgs)} messages", flush=True)
judges = [Judge(factory()) for _ in range(16)]


def do(args):
    k, m = args
    m["label"] = judges[k % 16].classify(m["text"])
    return m


with ThreadPoolExecutor(max_workers=16) as ex:
    labelled = list(ex.map(do, enumerate(msgs)))
path = os.path.join(outdir, "judge_labels.jsonl")
with open(path, "w") as f:
    for m in labelled:
        f.write(json.dumps(m) + "\n")
print(f"judge labelled {len(labelled)} messages -> {path}", flush=True)
k = export_validation_sample(path,
                             os.path.join(outdir,
                                          "validation_sample_TO_HAND_LABEL.csv"))
print(f"MANUAL STEP: exported {k} messages for hand labelling", flush=True)
print(f"final cost: ${cost.cost_usd:.2f}", flush=True)
