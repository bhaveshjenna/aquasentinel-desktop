"""USGS NWIS Instantaneous Values client. Keyless, no registration.

Site discovery uses the RDB site service with ``outputDataTypeCd=iv``, which returns one row
per (site, parameter) with that parameter's period of record - so we can pick sites that
genuinely report DO plus a few more channels, rather than transcribing site numbers that go
stale. Discovery output is cached per state under ``data_cache/site_index``.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from . import http as H

SITE_URL = "{base}/site/"
IV_URL = "{base}/iv/"


def _rdb_to_frame(text: str) -> pd.DataFrame:
    """NWIS RDB is tab-separated with hash comments, a header row, then a column-types row."""
    lines = [ln for ln in text.splitlines() if not ln.startswith("#")]
    if len(lines) < 3:
        return pd.DataFrame()
    header, _types, *rows = lines
    buf = io.StringIO("\n".join([header, *rows]))
    return pd.read_csv(buf, sep="\t", dtype=str, low_memory=False)


def discover_sites(
    cfg: Dict[str, Any],
    http_cfg: Dict[str, Any],
    cache_dir: Path,
    window: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Candidate sites, one row each: site_no, station_nm, lat, lon, state, params, days.

    A parameter only counts if its period of record actually covers the collection window -
    NWIS lists plenty of sites whose DO probe stopped years ago.
    """
    disc = cfg["discover"]
    base = cfg["base_url"]
    param_codes = ",".join(cfg["parameters"].keys())
    frames: List[pd.DataFrame] = []

    for state in disc["state_codes"]:
        cache = cache_dir / f"usgs_sites_{state}.tsv"
        if cache.exists():
            text = cache.read_text(encoding="utf-8")
        else:
            try:
                resp = H.get(
                    SITE_URL.format(base=base),
                    params={
                        "format": "rdb",
                        "stateCd": state,
                        "parameterCd": param_codes,
                        "hasDataTypeCd": "iv",
                        "outputDataTypeCd": "iv",
                        "siteStatus": "active",
                        "siteType": "ST,LK,ES",
                    },
                    source="usgs",
                    accept="text/plain",
                    **H.http_kwargs(http_cfg),
                )
            except H.FetchError as exc:
                print(f"  [usgs] site discovery failed for {state}: {exc}")
                continue
            text = resp.text
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(text, encoding="utf-8")

        df = _rdb_to_frame(text)
        if df.empty or "site_no" not in df.columns or "parm_cd" not in df.columns:
            continue
        df["state"] = state
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    raw = pd.concat(frames, ignore_index=True)
    raw = raw[raw["parm_cd"].isin(cfg["parameters"].keys())].copy()
    raw["begin_date"] = pd.to_datetime(raw.get("begin_date"), errors="coerce")
    raw["end_date"] = pd.to_datetime(raw.get("end_date"), errors="coerce")
    raw["days"] = (raw["end_date"] - raw["begin_date"]).dt.days

    win_start = pd.Timestamp(str(window["start"])) if window else None
    win_end = pd.Timestamp(str(window["end"])) if window else None

    out: List[Dict[str, Any]] = []
    for (site, state), grp in raw.groupby(["site_no", "state"]):
        good = grp[grp["days"].fillna(0) >= disc["min_days_of_record"]]
        if win_start is not None:
            # Require the record to straddle the window, allowing a fortnight of slack.
            good = good[
                (good["begin_date"] <= win_start + pd.Timedelta(days=14))
                & (good["end_date"] >= win_end - pd.Timedelta(days=14))
            ]
        if good.empty or disc["require_param"] not in set(good["parm_cd"]):
            continue
        first = good.iloc[0]
        out.append(
            {
                "site_no": site,
                "station_nm": str(first.get("station_nm", "")).strip(),
                "lat": pd.to_numeric(first.get("dec_lat_va"), errors="coerce"),
                "lon": pd.to_numeric(first.get("dec_long_va"), errors="coerce"),
                "state": state,
                "params": "|".join(sorted(set(good["parm_cd"]))),
                "n_params": len(set(good["parm_cd"])),
                "days": float(good["days"].max()),
            }
        )

    cand = pd.DataFrame(out)
    if cand.empty:
        return cand
    # Prefer richer, longer-running sites, but spread the picks across states so the client
    # set is geographically heterogeneous - that heterogeneity is the point of the fed study.
    cand = cand.sort_values(["n_params", "days"], ascending=False)
    picked = cand.groupby("state", group_keys=False).head(disc["sites_per_state"])
    return picked.reset_index(drop=True)


def fetch_site(
    site_no: str,
    cfg: Dict[str, Any],
    window: Dict[str, Any],
    http_cfg: Dict[str, Any],
    raw_dir: Path,
) -> pd.DataFrame:
    """One site's IV series as a long frame [timestamp, channel, value].

    Raw JSON is written to ``raw_dir`` so harmonisation is auditable and replayable offline.
    """
    base = cfg["base_url"]
    params = {
        "format": "json",
        "sites": site_no,
        "parameterCd": ",".join(cfg["parameters"].keys()),
        "startDT": str(window["start"]),
        "endDT": str(window["end"]),
        "siteStatus": "all",
    }
    raw_path = raw_dir / f"usgs_{site_no}.json"
    if raw_path.exists():
        payload = raw_path.read_text(encoding="utf-8")
    else:
        resp = H.get(
            IV_URL.format(base=base), params=params, source="usgs", **H.http_kwargs(http_cfg)
        )
        payload = resp.text
        H.save_raw(raw_dir, raw_path.name, payload)

    data = json.loads(payload)
    rows: List[Dict[str, Any]] = []
    for series in data.get("value", {}).get("timeSeries", []):
        code = series["variable"]["variableCode"][0]["value"]
        channel = cfg["parameters"].get(code)
        if channel is None:
            continue
        no_data = series["variable"].get("noDataValue", -999999.0)
        for block in series.get("values", []):
            for pt in block.get("value", []):
                try:
                    val = float(pt["value"])
                except (TypeError, ValueError):
                    continue
                if val == no_data:
                    continue
                rows.append({"timestamp": pt["dateTime"], "channel": channel, "value": val})
    if not rows:
        return pd.DataFrame(columns=["timestamp", "channel", "value"])
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce", format="ISO8601")
    return df.dropna(subset=["timestamp"])


def site_metadata(site_no: str, index: Optional[pd.DataFrame]) -> Dict[str, Any]:
    if index is None or index.empty or "site_no" not in index.columns:
        return {"name": site_no, "lat": None, "lon": None, "region": ""}
    hit = index[index["site_no"].astype(str) == str(site_no)]
    if hit.empty:
        return {"name": site_no, "lat": None, "lon": None, "region": ""}
    r = hit.iloc[0]
    return {
        "name": r.get("station_nm", site_no),
        "lat": r.get("lat"),
        "lon": r.get("lon"),
        "region": r.get("state", ""),
    }
