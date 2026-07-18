#!/usr/bin/env python3
"""Build the supplementary reproducibility pack.

Outputs (under dist/):
  seed_manifest.json      master seeds + per-run/pop seeds and configs,
                          harvested from every summary.json
  repro_snapshot.zip      anonymised repo snapshot: tracked files only,
                          no .git, no operator paths; scrubs any API key
                          pattern defensively (none should exist)

The AAAI reproducibility checklist answers live in
docs/aaai_reproducibility_checklist.md (hand-written against what the
suite actually does; verify before submission).
"""

import glob
import json
import os
import re
import subprocess
import zipfile

KEY_PAT = re.compile(r"sk-or-v1-[0-9a-f]{16,}|sk-ant-[A-Za-z0-9-]{16,}")


def seed_manifest() -> dict:
    man = {"master_seeds": {"expA": "recorded per-cell in results/expA",
                            "expB": 20260718, "envs": 20260719},
           "runs": []}
    for p in sorted(glob.glob("results/**/summary.json", recursive=True)):
        try:
            s = json.load(open(p))
        except json.JSONDecodeError:
            continue
        cfg = s.get("config", {})
        if "seed" in cfg:
            man["runs"].append({
                "dir": os.path.dirname(p),
                "seed": cfg["seed"],
                "cell": cfg.get("cell"),
                "model": cfg.get("model"),
                "framing": cfg.get("framing"),
            })
    man["n_runs"] = len(man["runs"])
    return man


def build_snapshot(zip_path: str) -> int:
    tracked = subprocess.run(["git", "ls-files"], capture_output=True,
                             text=True, check=True).stdout.splitlines()
    n_scrubbed = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in tracked:
            if not os.path.exists(f):
                continue
            try:
                with open(f, "rb") as fh:
                    data = fh.read()
                text = data.decode("utf-8")
                if KEY_PAT.search(text):
                    text = KEY_PAT.sub("[REDACTED-KEY]", text)
                    n_scrubbed += 1
                    data = text.encode("utf-8")
            except UnicodeDecodeError:
                pass
            z.writestr(f"namegame/{f}", data)
    return n_scrubbed


def main():
    os.makedirs("dist", exist_ok=True)
    man = seed_manifest()
    with open("dist/seed_manifest.json", "w") as f:
        json.dump(man, f, indent=1)
    print(f"seed manifest: {man['n_runs']} runs")
    n = build_snapshot("dist/repro_snapshot.zip")
    print(f"snapshot written (scrubbed files: {n}; expected 0)")
    if n:
        raise SystemExit("KEY MATERIAL FOUND IN TRACKED FILES - investigate")


if __name__ == "__main__":
    main()
