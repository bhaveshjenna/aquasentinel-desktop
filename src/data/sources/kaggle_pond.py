"""Real aquaculture pond IoT data from Kaggle - the project's data source.

This is the project's only data source: real catfish aquaponics ponds on local farms in
Nsukka, Enugu State, Nigeria (Udanor et al., Data in Brief 2022) - one region. It needs a
Kaggle token (``~/.kaggle``) only the first time the dataset is downloaded; after that the CSVs
sit under ``data_cache/raw/kaggle_<slug>/`` and everything runs offline.

A source may still be given its own ``manifest`` key in ``data.yaml`` to keep its sites out of
the main manifest; this one currently writes the main manifest.

**One CSV = one pond = one site.** A dataset entry with ``split_files: true`` turns each file
into its own site (``<prefix><NN>`` from the number in the filename), which is what lets every
pond become an independent federated client.

Rules this module enforces regardless of which dataset is configured:

1. **Never consume a dataset's own quality labels.** Public water-quality datasets routinely
   ship a "potable" / "WQI" / "quality" column produced by applying a threshold rule to the
   very columns it sits beside. Training on it and then reporting accuracy against it measures
   nothing but the ability to rediscover an arithmetic rule. ``use_labels`` exists in the
   config only so it can be seen to be ``false``; setting it to true raises.
2. **Record the licence.** The licence string goes into the manifest next to every row it
   covers.
3. **Only map columns that are physically what they claim to be.** The raw files are noisy
   (``-127`` from disconnected temperature probes, pH below 0, ammonia at 1e11). Those values
   are left for ``harmonize.py``'s QC ranges to drop. A column whose values are not in the
   claimed unit at all - this dataset's "nitrate", which is a raw 0-4095 ADC count - is simply
   not mapped.

The output contract is identical to ``usgs.py`` and ``ukea.py``: a long frame with columns
``[timestamp, channel, value]`` using the common channel schema, so ``harmonize.py`` and
everything downstream cannot tell the sources apart.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


class KaggleUnavailable(RuntimeError):
    """Raised when the source is enabled but cannot be used, with the reason."""


def _require_client():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise KaggleUnavailable(
            "the `kaggle` package is not installed. `pip install kaggle`, then place an API "
            "token under ~/.kaggle. The core pipeline does not need any of this."
        ) from exc
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # pragma: no cover - depends on local credentials
        raise KaggleUnavailable(f"Kaggle authentication failed: {exc}") from exc
    return api


# ----------------------------------------------------------------------------- helpers
def _entries(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for entry in cfg.get("datasets") or []:
        out.append(dict(entry) if isinstance(entry, dict) else {"slug": str(entry)})
    return out


def _dataset_dir(raw_dir: Path, slug: str) -> Path:
    return Path(raw_dir) / f"kaggle_{slug.replace('/', '__')}"


def _ensure_downloaded(raw_dir: Path, slug: str) -> Path:
    target = _dataset_dir(raw_dir, slug)
    if not target.exists() or not any(target.rglob("*.csv")):
        api = _require_client()
        target.mkdir(parents=True, exist_ok=True)
        api.dataset_download_files(slug, path=str(target), unzip=True, quiet=False)
    return target


def _file_number(path: Path) -> Optional[int]:
    nums = re.findall(r"(\d+)", path.stem)
    return int(nums[-1]) if nums else None


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _map_columns(columns: List[str], cfg: Dict[str, Any]) -> Dict[str, str]:
    """source column -> channel.

    ``parameters`` maps exact column names. ``column_contains`` maps a normalised substring
    (lower case, alphanumerics only) - needed when one dataset spells the same column a
    different way in every file, as this one does ("Temperature (C)", "temperature(C)", ...).
    """
    exact: Dict[str, str] = cfg.get("parameters") or {}
    contains: Dict[str, str] = cfg.get("column_contains") or {}
    out: Dict[str, str] = {}
    for col in columns:
        if col in exact:
            out[col] = exact[col]
            continue
        n = _norm(col)
        for key, channel in contains.items():
            k = _norm(key)
            # A very short key ("ph") must match the whole name, or "phosphate" would hit it.
            if (n == k) if len(k) <= 3 else (k in n):
                out[col] = channel
                break
    # One column per channel: the first match wins, a duplicate is dropped rather than averaged.
    seen, dedup = set(), {}
    for col, ch in out.items():
        if ch not in seen:
            dedup[col] = ch
            seen.add(ch)
    return dedup


def _parse_time(series: pd.Series, assume_offset: str) -> pd.Series:
    """Timestamps -> UTC. Rows with only a date, or no time at all, become NaT and are dropped.

    This dataset mixes ``2021-06-18 13:02:10 CET`` with ``2021-09-25T04:27:42+01:00``; both are
    UTC+1, so a bare zone abbreviation is replaced by the configured offset before parsing.
    """
    s = series.astype(str).str.strip()
    s = s.str.replace(r"\s+[A-Z]{2,4}$", assume_offset, regex=True)
    has_time = s.str.contains(r"\d{1,2}:\d{2}", regex=True)
    out = pd.to_datetime(s.where(has_time), utc=True, errors="coerce", format="ISO8601")
    return out


# ----------------------------------------------------------------------------- source API
def discover_sites(
    cfg: Dict[str, Any],
    http_cfg: Dict[str, Any],
    cache_dir: Path,
    window: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """One row per site. Downloads a dataset on first use; offline after that."""
    raw_dir = Path(cfg.get("raw_dir") or (Path(cache_dir).parent / "raw"))
    rows: List[Dict[str, Any]] = []
    for entry in _entries(cfg):
        slug = entry["slug"]
        base = {
            "lat": entry.get("lat"),
            "lon": entry.get("lon"),
            "state": entry.get("region", "pond"),
            "params": "|".join(sorted(set((cfg.get("parameters") or {}).values())
                                      | set((cfg.get("column_contains") or {}).values()))),
            "days": float("nan"),
            "slug": slug,
        }
        if entry.get("split_files"):
            directory = _ensure_downloaded(raw_dir, slug)
            prefix = entry.get("prefix", "pond")
            for csv in sorted(directory.rglob("*.csv"), key=lambda p: (_file_number(p) or 0)):
                n = _file_number(csv)
                if n is None:
                    continue
                rows.append({
                    **base,
                    "site_no": f"{prefix}{n:02d}",
                    "station_nm": f"{entry.get('name', slug)} - {csv.stem}",
                    "file": csv.name,
                })
        else:
            rows.append({**base, "site_no": slug.replace("/", "__"),
                         "station_nm": entry.get("name", slug), "file": ""})
    return pd.DataFrame(rows)


def fetch_site(
    site_no: str,
    cfg: Dict[str, Any],
    window: Dict[str, Any],
    http_cfg: Dict[str, Any],
    raw_dir: Path,
) -> pd.DataFrame:
    """Reshape one site's CSV(s) into the common long format."""
    if cfg.get("use_labels", False):
        raise KaggleUnavailable(
            "sources.kaggle_pond.use_labels must stay false. These datasets' quality labels are "
            "typically scripted thresholds over the same columns; consuming them would make "
            "every downstream accuracy figure circular. See this module's docstring."
        )
    if not (cfg.get("parameters") or cfg.get("column_contains")):
        raise KaggleUnavailable(
            "sources.kaggle_pond has no column mapping: fill `parameters` or `column_contains` "
            "with the dataset's own column names -> the common channel schema."
        )

    csvs = _site_files(site_no, cfg, Path(raw_dir))
    if not csvs:
        raise KaggleUnavailable(f"no CSV found for site '{site_no}'")

    time_cols = cfg.get("time_column", "timestamp")
    time_cols = [time_cols] if isinstance(time_cols, str) else list(time_cols)
    offset = str(cfg.get("assume_utc_offset", "+00:00"))

    frames: List[pd.DataFrame] = []
    for path in csvs:
        df = pd.read_csv(path, low_memory=False)
        tcol = next((c for c in time_cols if c in df.columns), None)
        if tcol is None:
            continue
        ts = _parse_time(df[tcol], offset)
        for source_col, channel in _map_columns(list(df.columns), cfg).items():
            frames.append(pd.DataFrame({
                "timestamp": ts,
                "channel": channel,
                "value": pd.to_numeric(df[source_col], errors="coerce"),
            }))
    if not frames:
        raise KaggleUnavailable(f"site '{site_no}': no time column or no mappable channel")
    out = pd.concat(frames, ignore_index=True)
    # inf appears in the raw ammonia columns; it is not a reading.
    out = out[out["value"].abs() != float("inf")]
    return out.dropna(subset=["timestamp", "value"])


def _site_files(site_no: str, cfg: Dict[str, Any], raw_dir: Path) -> List[Path]:
    for entry in _entries(cfg):
        slug = entry["slug"]
        directory = _dataset_dir(raw_dir, slug)
        if entry.get("split_files"):
            prefix = entry.get("prefix", "pond")
            if not site_no.startswith(prefix):
                continue
            try:
                n = int(site_no[len(prefix):])
            except ValueError:
                continue
            directory = _ensure_downloaded(raw_dir, slug)
            hits = [p for p in directory.rglob("*.csv") if _file_number(p) == n]
            if hits:
                return hits
        elif site_no == slug.replace("/", "__"):
            return sorted(_ensure_downloaded(raw_dir, slug).rglob("*.csv"))
    return []


def site_metadata(site_no: str, index: Optional[pd.DataFrame]) -> Dict[str, Any]:
    if index is None or index.empty or "site_no" not in index.columns:
        return {"name": site_no, "lat": None, "lon": None, "region": "pond"}
    hit = index[index["site_no"].astype(str) == str(site_no)]
    if hit.empty:
        return {"name": site_no, "lat": None, "lon": None, "region": "pond"}
    r = hit.iloc[0]
    return {
        "name": r.get("station_nm", site_no),
        "lat": r.get("lat"),
        "lon": r.get("lon"),
        "region": r.get("state", "pond"),
    }
