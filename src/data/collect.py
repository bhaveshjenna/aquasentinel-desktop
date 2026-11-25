"""Orchestrate collection: discover -> fetch -> harmonize -> cache -> manifest.

Idempotent by construction. A site whose parquet already exists is never re-downloaded, and
every raw payload stays on disk, so after one online run the whole project is offline.

``data_cache/manifest.csv`` is the single source of truth for "how many sites / how many
rows" - the README and every table read it rather than transcribing numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..common.runtime import ROOT, load_config
from . import harmonize as HZ
from .sources import kaggle_pond, ukea, usgs

# kaggle_pond is optional and disabled in configs/data.yaml: it is the only source that needs
# a token, and the core pipeline stays keyless without it.
SOURCE_MODULES = {"usgs": usgs, "ukea": ukea, "kaggle_pond": kaggle_pond}


def _path(cfg_path: str) -> Path:
    p = Path(cfg_path)
    return p if p.is_absolute() else ROOT / p


def _selected(name: str, scfg: Dict[str, Any], only: Optional[List[str]]) -> bool:
    if name not in SOURCE_MODULES:
        return False
    if only:
        return name in only          # naming a source explicitly runs it even if disabled
    return bool(scfg.get("enabled"))


def manifest_path_for(name: str, cfg: Optional[Dict[str, Any]] = None) -> Path:
    """Where a source's sites are listed. A source with its own ``manifest`` key is kept out of
    the main manifest - that is how a different region stays out of the river study."""
    cfg = cfg or load_config("data")
    own = (cfg["sources"].get(name) or {}).get("manifest")
    return _path(own) if own else _path(cfg["cache"]["manifest"])


def discover(
    cfg: Optional[Dict[str, Any]] = None,
    refresh: bool = False,
    only: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """Run each enabled source's site discovery; cache the index per source."""
    cfg = cfg or load_config("data")
    index_dir = _path(cfg["cache"]["site_index"])
    index_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, pd.DataFrame] = {}

    for name, scfg in cfg["sources"].items():
        if not _selected(name, scfg, only):
            continue
        index_csv = index_dir / f"{name}.csv"
        if index_csv.exists() and not refresh:
            out[name] = pd.read_csv(index_csv, dtype={"site_no": str})
            print(f"[{name}] site index (cached): {len(out[name])} candidates")
            continue
        print(f"[{name}] discovering sites ...")
        module = SOURCE_MODULES[name]
        idx = module.discover_sites(scfg, cfg["http"], index_dir, cfg["window"])
        if idx.empty:
            print(f"[{name}] no candidate sites found")
            out[name] = idx
            continue
        idx = idx.head(int(scfg.get("max_sites", 25)))
        idx.to_csv(index_csv, index=False)
        out[name] = idx
        print(f"[{name}] {len(idx)} candidate sites -> {index_csv.name}")
    return out


def collect(
    cfg: Optional[Dict[str, Any]] = None,
    refresh_index: bool = False,
    limit: Optional[int] = None,
    only: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Fetch + harmonize every discovered site; write parquet per site and the manifest(s).

    ``only`` restricts the run to the named sources. Only the manifests those sources write to
    are rewritten, so collecting the pond dataset never touches the river manifest.
    """
    cfg = cfg or load_config("data")
    window, http_cfg, qc_cfg, targets = cfg["window"], cfg["http"], cfg["qc"], cfg["targets"]
    raw_dir = _path(cfg["cache"]["raw_dir"])
    site_dir = _path(cfg["cache"]["site_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    site_dir.mkdir(parents=True, exist_ok=True)

    indexes = discover(cfg, refresh=refresh_index, only=only)
    manifests: Dict[Path, List[Dict[str, Any]]] = {}
    collected = 0

    for name, scfg in cfg["sources"].items():
        if not _selected(name, scfg, only):
            continue
        manifest_rows = manifests.setdefault(manifest_path_for(name, cfg), [])
        # A source with its own manifest has its own site budget; it is not competing with the
        # river sites for `targets.total_sites`.
        if scfg.get("manifest"):
            collected = 0
        module = SOURCE_MODULES[name]
        index = indexes.get(name, pd.DataFrame())
        explicit = [str(s) for s in (scfg.get("sites") or [])]
        site_ids = explicit or (index["site_no"].astype(str).tolist() if not index.empty else [])

        for site_no in site_ids:
            if limit is not None and collected >= limit:
                break
            if collected >= int(targets["total_sites"]):
                break
            client_id = f"{name}_{site_no}"
            parquet = site_dir / f"{client_id}.parquet"

            if parquet.exists() and not cfg["cache"].get("overwrite", False):
                row = _manifest_row_from_cache(parquet, client_id, name, site_no, scfg, module, index)
                if row is not None:
                    manifest_rows.append(row)
                    collected += 1
                    print(f"[{name}] {site_no}: cached ({row['rows']} rows)")
                continue

            print(f"[{name}] {site_no}: fetching ...")
            try:
                if name == "ukea":
                    measures = _measures_for(index, site_no)
                    long_df = module.fetch_site(
                        site_no, scfg, window, http_cfg, raw_dir, measures=measures
                    )
                else:
                    long_df = module.fetch_site(site_no, scfg, window, http_cfg, raw_dir)
            except Exception as exc:  # one bad site must not abort a long collection run
                print(f"[{name}] {site_no}: FAILED ({exc})")
                continue

            values, mask, report = HZ.harmonize_site(long_df, window["cadence_minutes"], qc_cfg)
            present = HZ.channels_present(mask)
            keep, reason = _qualifies(values, mask, present, targets)
            if not keep:
                print(f"[{name}] {site_no}: dropped ({reason})")
                continue

            _write_site(parquet, values, mask)
            meta = module.site_metadata(site_no, index)
            manifest_rows.append(
                {
                    "client_id": client_id,
                    "source": name,
                    "site_no": site_no,
                    "name": meta["name"],
                    "lat": meta["lat"],
                    "lon": meta["lon"],
                    "region": meta["region"],
                    "channels": "|".join(present),
                    "n_channels": len(present),
                    "rows": report.get("rows", len(values)),
                    "start": report.get("start"),
                    "end": report.get("end"),
                    "missing_frac": round(HZ.missing_fraction(mask, present), 4),
                    "cadence_minutes": window["cadence_minutes"],
                    "license": scfg["license"],
                    "path": str(parquet.relative_to(ROOT)).replace("\\", "/"),
                }
            )
            collected += 1
            print(f"[{name}] {site_no}: kept {report.get('rows')} rows, channels={present}")

    frames = []
    for manifest_path, rows in manifests.items():
        manifest = pd.DataFrame(rows)
        if not manifest.empty:
            manifest = manifest.sort_values(["source", "client_id"]).reset_index(drop=True)
        manifest.to_csv(manifest_path, index=False)
        print(f"\nmanifest -> {manifest_path} ({len(manifest)} sites, "
              f"{int(manifest['rows'].sum()) if not manifest.empty else 0} rows)")
        frames.append(manifest)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _measures_for(index: pd.DataFrame, site_no: str) -> Optional[Dict[str, str]]:
    if index.empty or "measures" not in index.columns:
        return None
    hit = index[index["site_no"].astype(str) == str(site_no)]
    if hit.empty:
        return None
    raw = hit.iloc[0]["measures"]
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def _qualifies(values, mask, present, targets) -> tuple:
    required_any = set(targets["required_any"])
    must_have = set(targets["must_have"])
    if not must_have.issubset(set(present)):
        return False, f"missing required channel(s) {sorted(must_have - set(present))}"
    n_useful = len(set(present) & required_any)
    if n_useful < int(targets["min_channels"]):
        return False, f"only {n_useful} usable channels"
    if len(values) < int(targets["min_rows"]):
        return False, f"only {len(values)} rows"
    miss = HZ.missing_fraction(mask, [c for c in present if c in required_any])
    if miss > float(targets["max_missing_frac"]):
        return False, f"missing_frac {miss:.2f}"
    return True, ""


def _write_site(parquet: Path, values: pd.DataFrame, mask: pd.DataFrame) -> None:
    out = values.copy()
    out.index.name = "timestamp"
    for c in mask.columns:
        out[f"mask_{c}"] = mask[c].to_numpy()
    out.reset_index().to_parquet(parquet, index=False)


def _manifest_row_from_cache(parquet, client_id, source, site_no, scfg, module, index):
    try:
        df = pd.read_parquet(parquet)
    except Exception:
        return None
    mask_cols = [c for c in df.columns if c.startswith("mask_")]
    mask = df[mask_cols].rename(columns=lambda c: c[len("mask_"):])
    present = HZ.channels_present(mask)
    meta = module.site_metadata(site_no, index)
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    return {
        "client_id": client_id,
        "source": source,
        "site_no": site_no,
        "name": meta["name"],
        "lat": meta["lat"],
        "lon": meta["lon"],
        "region": meta["region"],
        "channels": "|".join(present),
        "n_channels": len(present),
        "rows": int(len(df)),
        "start": ts.min().isoformat() if len(df) else None,
        "end": ts.max().isoformat() if len(df) else None,
        "missing_frac": round(HZ.missing_fraction(mask, present), 4),
        "cadence_minutes": load_config("data")["window"]["cadence_minutes"],
        "license": scfg["license"],
        "path": str(parquet.relative_to(ROOT)).replace("\\", "/"),
    }
