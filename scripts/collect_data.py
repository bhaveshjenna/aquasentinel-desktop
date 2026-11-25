"""Phase 1: the ONE-TIME online data collection step.

    python scripts/collect_data.py               # discover + fetch + harmonize + manifest
    python scripts/collect_data.py --refresh-index
    python scripts/collect_data.py --limit 4     # quick smoke run
    python scripts/collect_data.py --summary     # print manifest stats, no network

This is the only script in the project that touches the network. Everything after it -
training, the dashboard, evaluation - runs offline against data_cache/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.runtime import load_config, set_seed  # noqa: E402
from src.data import collect as C  # noqa: E402
from src.data import registry as R  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect real water-quality time series (one-time).")
    ap.add_argument("--refresh-index", action="store_true", help="re-run site discovery")
    ap.add_argument("--limit", type=int, default=None, help="stop after N sites (smoke test)")
    ap.add_argument("--summary", action="store_true", help="print manifest summary and exit")
    ap.add_argument("--source", action="append", default=None,
                    help="collect only this source (repeatable), e.g. --source kaggle_pond; "
                         "only that source's manifest is rewritten")
    args = ap.parse_args()

    set_seed()

    if args.summary:
        path = C.manifest_path_for(args.source[0]) if args.source else None
        for k, v in R.summary(path).items():
            print(f"{k:>14}: {v}")
        return 0

    cfg = load_config("data")
    print("AquaSentinel - online data collection")
    print(f"  window   : {cfg['window']['start']} -> {cfg['window']['end']} "
          f"@ {cfg['window']['cadence_minutes']} min")
    print(f"  target   : {cfg['targets']['total_sites']} sites, "
          f">= {cfg['targets']['min_rows']} rows each")
    print("  NOTE     : rivers/streams/lakes are a PROXY for ponds. See README section on the")
    print("             domain gap. No dataset quality labels are consumed.\n")

    manifest = C.collect(cfg, refresh_index=args.refresh_index, limit=args.limit,
                         only=args.source)
    if manifest.empty:
        print("\nNo sites collected. Check network access and configs/data.yaml.")
        return 1

    print("\nPer-source:")
    print(manifest.groupby("source").agg(sites=("client_id", "count"), rows=("rows", "sum")))
    print("\nChannel coverage across sites:")
    counts: dict = {}
    for chans in manifest["channels"]:
        for c in str(chans).split("|"):
            counts[c] = counts.get(c, 0) + 1
    for c, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {c:<14} {n} sites")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
