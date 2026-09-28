"""Command line: ``d4r gameset | bench | contest | report``.

Examples::

    d4r gameset --scenario B --seeds 1-50
    d4r bench --gameset runs/gameset/B-s1-50.jsonl --engine jev --workers 4 --tag pilot
    d4r contest --scenario B --seeds 1-20 --engine rule-slack --tag pilot
    d4r report --tag pilot
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

__all__ = ["main"]

RUNS = Path("runs")


def _seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def cmd_gameset(a: argparse.Namespace) -> None:
    from d4r.contest.gameset import build_gameset
    from d4r.sim.scenario import PRESETS

    out = Path(a.out or RUNS / "gameset" / f"{a.scenario}-s{a.seeds.replace(',', '_')}.jsonl")
    recs = build_gameset(PRESETS[a.scenario], _seeds(a.seeds), out)
    print(
        json.dumps(
            {
                "out": str(out),
                "cards": len(recs),
                "consequential": sum(r["spread"] > 0 for r in recs),
            }
        )
    )


def cmd_bench(a: argparse.Namespace) -> None:
    from d4r.apikeys import load_secrets
    from d4r.contest.bench import run_bench
    from d4r.contest.gameset import load_gameset
    from d4r.engines.registry import make_engine

    load_secrets()
    cards = [lc for g in a.gameset for lc in load_gameset(Path(g))]
    if not a.all_cards:
        cards = [lc for lc in cards if lc["spread"] > 0]
    if a.limit:
        cards = cards[: a.limit]
    engine = make_engine(a.engine, RUNS / "cache", salt=a.salt)
    suffix = f".{a.salt}" if a.salt else ""
    out = RUNS / "bench" / a.tag / f"{a.engine}{suffix}.jsonl"
    recs = run_bench(cards, engine, out, workers=a.workers)
    from d4r.analysis.metrics import summarize

    print(json.dumps({"out": str(out), **summarize(recs)}, default=float))


def cmd_contest(a: argparse.Namespace) -> None:
    from d4r.apikeys import load_secrets
    from d4r.contest.closed_loop import run_contest
    from d4r.engines.registry import make_engine
    from d4r.sim.scenario import PRESETS

    load_secrets()
    engine = make_engine(a.engine, RUNS / "cache")
    out = RUNS / "contest" / a.tag / f"{a.scenario}-{a.engine}.jsonl"
    rows = run_contest(PRESETS[a.scenario], _seeds(a.seeds), engine, out)
    arr = [r["result"]["arrival_share"] for r in rows]
    print(
        json.dumps(
            {
                "out": str(out),
                "episodes": len(rows),
                "mean_arrival_share": sum(arr) / max(1, len(arr)),
            }
        )
    )


def cmd_report(a: argparse.Namespace) -> None:
    from d4r.analysis.report import write_report

    print(write_report(a.tag))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        prog="d4r", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gameset", help="build a labelled game set from DLA episodes")
    g.add_argument("--scenario", required=True)
    g.add_argument("--seeds", required=True, help="e.g. 1-50 or 1,2,5")
    g.add_argument("--out")
    g.set_defaults(fn=cmd_gameset)

    b = sub.add_parser("bench", help="answer a game set with one engine")
    b.add_argument("--gameset", nargs="+", required=True)
    b.add_argument("--engine", required=True)
    b.add_argument("--tag", required=True)
    b.add_argument("--workers", type=int, default=1)
    b.add_argument("--salt", default="", help="separate cache for repeated calls (test-retest)")
    b.add_argument("--all-cards", action="store_true", help="include non-consequential cards")
    b.add_argument("--limit", type=int, default=0)
    b.set_defaults(fn=cmd_bench)

    c = sub.add_parser("contest", help="closed-loop episodes with one engine")
    c.add_argument("--scenario", required=True)
    c.add_argument("--seeds", required=True)
    c.add_argument("--engine", required=True)
    c.add_argument("--tag", required=True)
    c.set_defaults(fn=cmd_contest)

    r = sub.add_parser("report", help="tables for a tag (bench + contest)")
    r.add_argument("--tag", required=True)
    r.set_defaults(fn=cmd_report)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":  # pragma: no cover
    main(sys.argv[1:])
