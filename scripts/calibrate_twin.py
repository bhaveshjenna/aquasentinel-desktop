"""Fit the twin's structural defaults to the collected real data (offline).

    python scripts/calibrate_twin.py            # apply and write configs/twin.yaml
    python scripts/calibrate_twin.py --dry-run  # show what would change
    python scripts/calibrate_twin.py --adopt-mean   # also take the real temperature mean

--adopt-mean and --adopt-diurnal are off by default on purpose: the monitoring sites are
temperate rivers and lakes, and their temperature mean and daily swing are set by depth, flow
and thermal mass. Copying either into a 1.5 m warm-water pond would make the simulation wrong
in exactly the way the rest of the calibration is designed to avoid. The fitted values are
still recorded in configs/twin.yaml for inspection. See src/twin/calibrate.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common.runtime import set_seed  # noqa: E402
from src.twin import calibrate as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Calibrate the pond twin to real-data statistics.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--adopt-mean", action="store_true")
    ap.add_argument("--adopt-diurnal", action="store_true",
                    help="also apply the fitted diurnal amplitude and phase (see module docstring)")
    ap.add_argument("--max-sites", type=int, default=None)
    args = ap.parse_args()

    set_seed()
    print("Collecting statistics from cached real sites ...")
    stats = C.collect_statistics(max_sites=args.max_sites)
    print(f"  {stats['n_sites']} sites")
    for ch, d in stats.get("diurnal", {}).items():
        print(f"  diurnal {ch:<9} amplitude {d['amplitude']:.3f}  peak {d['peak_hour']:.2f} h  "
              f"(n={d['n']} sites)")
    for ch, sig in (stats.get("ou_sigma") or {}).items():
        print(f"  OU sigma {ch:<8} {sig:.4f} per sqrt(h)  "
              f"({stats['sigma_site_counts'].get(ch, 0)} sites)")

    changes = C.calibrate(stats, adopt_mean=args.adopt_mean, adopt_diurnal=args.adopt_diurnal,
                          write=not args.dry_run)
    print("\nChanges" + (" (dry run, nothing written)" if args.dry_run else " written to configs/twin.yaml"))
    for path, c in changes.items():
        print(f"  {path:<28} {c['from']} -> {c['to']}    [{c['source']}]")
    if not changes:
        print("  none - not enough real data to fit anything transferable")
    print()
    print("NOT calibrated, deliberately: the diurnal cycle (amplitude / phase / mean), every")
    print("biological rate constant, and stocking density. The fitted diurnal numbers printed")
    print("above are recorded in configs/twin.yaml under calibration.fitted_diurnal_not_applied")
    print("so a reader can see them - a river's 0.5 C daily swing is not a 1.5 m pond's.")
    print("Pass --adopt-diurnal / --adopt-mean to apply them anyway.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
