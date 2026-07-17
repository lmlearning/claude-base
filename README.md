# Norm Persistence, Enforcement, and Adversarial Fragility in Agent Populations

Experimental suite for a naming-game study of (1) convention persistence
under population turnover, (2) spontaneous normative enforcement in LLM
populations, and (3) history-dependent committed-minority thresholds.

See **RESULTS.md** for the pre-registered criteria, the decision log, and
the full results summary. Figures are in `figures/`; every number in the
summary is also in `results/analysis_summary.json`.

## Layout

```
namegame/core       shared simulation engine (agents, criteria, phases)
namegame/expa       Experiment A: minimal-agent tier (high statistics)
namegame/expb       Experiment B: LLM tier (Anthropic API + free mock mode)
namegame/analysis   statistics, figures, summary generation
tests/              pytest suite
results/            run outputs (JSONL summaries; full JSONL journals for B)
figures/            publication figures
```

## Reproduce

```bash
pip install numpy scipy pandas matplotlib anthropic pytest
python -m pytest tests/ -q

# Experiment A (deterministic from recorded master seed; ~30-60 min, 4 cores)
python -m namegame expa --outdir results/expA --procs 4

# Experiment B, free mock mode (full pipeline, no API calls; ~10 min)
python -m namegame expb --mode mock

# Experiment B, live (requires ANTHROPIC_API_KEY; prints a cost projection
# and enforces a hard spend cap in code before any call)
python -m namegame expb --mode live --spend-cap-usd 250 --project-cost  # projection only
python -m namegame expb --mode live --spend-cap-usd 250                 # run

# Analysis + figures (consumes transcripts only)
python -m namegame analyze --results results --figdir figures
```

Every run records its seed, resolved configuration, and code version.
Experiment B runs journal every interaction before the next begins and are
resumable: rerunning the same command replays the journal and continues.

## Manual step (live enforcement results only)

The LLM judge's utterance labels must be validated by hand before any
enforcement claim is reported: label the exported
`results/expB_live/validation_sample_TO_HAND_LABEL.csv`, then rerun the
analysis; the gate requires Cohen's kappa >= 0.6.
