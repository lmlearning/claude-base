"""Command-line entry point.

Usage:
  python -m namegame expa [--outdir results/expA] [--procs 4] [--only CELL ...]
  python -m namegame expb --mode mock|live [--outdir results/expB] ...
  python -m namegame analyze [--results results] [--figdir figures]
"""

from __future__ import annotations

import argparse


def main() -> None:
    p = argparse.ArgumentParser(prog="namegame")
    sub = p.add_subparsers(dest="cmd", required=True)

    pa = sub.add_parser("expa", help="run Experiment A (minimal agents)")
    pa.add_argument("--outdir", default="results/expA")
    pa.add_argument("--procs", type=int, default=4)
    pa.add_argument("--only", nargs="*", default=None)
    pa.add_argument("--force", action="store_true")
    pa.add_argument("--list", action="store_true", help="list cells and exit")

    pb = sub.add_parser("expb", help="run Experiment B (LLM populations)")
    pb.add_argument("--mode", choices=["mock", "live"], default="mock")
    pb.add_argument("--outdir", default=None,
                    help="default results/expB_<mode>")
    pb.add_argument("--only", nargs="*", default=None)
    pb.add_argument("--list", action="store_true")
    pb.add_argument("--spend-cap-usd", type=float, default=250.0)
    pb.add_argument("--project-cost", action="store_true",
                    help="print cost projection and exit")

    pan = sub.add_parser("analyze", help="run analysis and figures")
    pan.add_argument("--results", default="results")
    pan.add_argument("--figdir", default="figures")

    args = p.parse_args()

    if args.cmd == "expa":
        from .expa.runner import define_cells, run_all
        if args.list:
            for name, cell in define_cells().items():
                print(f"{name}\t{cell['phase']}\tn_runs={cell['n_runs']}")
            return
        run_all(args.outdir, procs=args.procs, only=args.only, force=args.force)
    elif args.cmd == "expb":
        from .expb.runner import main_expb
        main_expb(args)
    elif args.cmd == "analyze":
        from .analysis.report import main_analysis
        main_analysis(args.results, args.figdir)


if __name__ == "__main__":
    main()
