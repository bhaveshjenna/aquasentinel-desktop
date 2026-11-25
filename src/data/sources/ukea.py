"""UK Environment Agency Hydrology API client. Keyless, Open Government Licence v3.0.

This is the source that supplies the nutrient channels USGS lacks: ammonium and nitrate,
alongside DO / temperature / pH / turbidity / conductivity / chlorophyll / blue-green algae,
at 15-minute cadence. Stations -> measures -> readings.

Fair use: we request only the 15-min instantaneous measures we actually model, in monthly
CSV chunks (far lighter than the JSON representation), and cache every chunk on disk.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from . import http as H

STATIONS_URL = "{base}/id/stations"
ACTIVE_STATUS = "http://environment.data.gov.uk/flood-monitoring/def/core/statusActive"

# Measure `parameter` label -> our common channel. Note ammonium (NH4-N) is mapped onto
# `tan`: total ammonia nitrogen is the channel the pond model reasons about, and the EA
# ammonium series is reported as mg/L N. That equivalence is an approximation, stated here.
PARAM_TO_CHANNEL: Dict[str, str] = {
    "DISSOLVED OXYGEN": "do",
    "TEMPERATURE": "temp_c",
    "PH": "ph",
    "TURBIDITY": "turbidity",
    "AMMONIUM": "tan",
    "NITRATE": "no3",
    "CONDUCTIVITY": "conductivity",
    "CHLOROPHYLL": "chlorophyll",
    "BLUE GREEN ALGAE": "bga",
    "SALINITY": "salinity",
}

# When a parameter is published in several units, take the first unit we find here.
# DO in mg/L (not % saturation) is the one the pond model needs.
UNIT_PREFERENCE: Dict[str, List[str]] = {
    "do": ["mgL"],
    "temp_c": ["C"],
    "turbidity": ["ntu"],
    "tan": ["mgL"],
    "no3": ["mgL"],
    "conductivity": ["uS", "mS"],
    "chlorophyll": ["ugpL"],
    "bga": ["rfu"],
    "salinity": ["psu"],
    "ph": ["subdaily"],  # pH measure ids carry no unit token
}


def _unit_of(measure_id: str) -> str:
    return measure_id.rsplit("-", 1)[-1]


def _pick_measures(station: Dict[str, Any], period_seconds: int) -> Dict[str, str]:
    """channel -> measure @id, keeping one measure per channel at the wanted cadence."""
    chosen: Dict[str, str] = {}
    for m in station.get("measures", []) or []:
        if int(m.get("period") or 0) != int(period_seconds):
            continue
        channel = PARAM_TO_CHANNEL.get(str(m.get("parameter", "")).upper())
        if channel is None:
            continue
        mid = m.get("@id", "")
        unit = _unit_of(mid)
        prefs = UNIT_PREFERENCE.get(channel, [])
        if prefs and unit not in prefs:
            continue
        if channel in chosen:
            # Lower index in the preference list wins.
            if prefs.index(unit) >= prefs.index(_unit_of(chosen[channel])):
                continue
        chosen[channel] = mid
    return chosen


def discover_sites(
    cfg: Dict[str, Any],
    http_cfg: Dict[str, Any],
    cache_dir: Path,
    window: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Active water-quality stations reporting DO, one row per station.

    Stations commissioned after the collection window are dropped - the EA adds new sondes
    constantly and a 2025 station has nothing to say about a 2024 window.
    """
    disc = cfg["discover"]
    cache = cache_dir / "ukea_stations.json"
    if cache.exists():
        payload = cache.read_text(encoding="utf-8")
    else:
        resp = H.get(
            STATIONS_URL.format(base=cfg["base_url"]),
            params={
                "observedProperty": disc["observed_property"],
                "status": ACTIVE_STATUS,
                "_limit": disc["station_limit"],
            },
            source="ukea",
            **H.http_kwargs(http_cfg),
        )
        payload = resp.text
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(payload, encoding="utf-8")

    import json

    items = json.loads(payload).get("items", [])
    period = int(cfg.get("period_seconds", 900))
    rows: List[Dict[str, Any]] = []
    for st in items:
        if st.get("dateClosed"):
            continue
        measures = _pick_measures(st, period)
        if "do" not in measures or len(measures) < int(disc.get("min_measures", 2)):
            continue
        rows.append(
            {
                "site_no": st.get("notation"),
                "station_nm": st.get("label", ""),
                "lat": st.get("lat"),
                "lon": st.get("long"),
                "state": "UK",
                "params": "|".join(sorted(measures)),
                "n_params": len(measures),
                "days": float("nan"),
                "measures": json.dumps(measures),
                "date_opened": st.get("dateOpened"),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date_opened"] = pd.to_datetime(df["date_opened"], errors="coerce")
    if window is not None:
        cutoff = pd.Timestamp(str(window["start"]))
        df = df[df["date_opened"].notna() & (df["date_opened"] <= cutoff)]
        if df.empty:
            return df
    # Nutrient-bearing stations first: they are the ones that give us TAN/NO3.
    df["has_nutrient"] = df["params"].str.contains("tan") | df["params"].str.contains("no3")
    return df.sort_values(["has_nutrient", "n_params"], ascending=False).reset_index(drop=True)


def _month_edges(start: str, end: str) -> List[tuple]:
    ts = pd.Timestamp(start).normalize()
    te = pd.Timestamp(end).normalize()
    edges = []
    cur = ts
    while cur < te:
        nxt = min(cur + pd.offsets.MonthBegin(1), te)
        if nxt <= cur:
            nxt = te
        edges.append((cur.date().isoformat(), (nxt - pd.Timedelta(days=1)).date().isoformat()))
        cur = nxt
    return edges


def fetch_site(
    site_no: str,
    cfg: Dict[str, Any],
    window: Dict[str, Any],
    http_cfg: Dict[str, Any],
    raw_dir: Path,
    measures: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """One station's series as a long frame [timestamp, channel, value].

    ``measures`` comes from discovery; without it the station document is re-fetched.
    """
    import json

    if not measures:
        resp = H.get(
            f"{cfg['base_url']}/id/stations/{site_no}",
            source="ukea",
            **H.http_kwargs(http_cfg),
        )
        items = resp.json().get("items", [])
        station = items[0] if isinstance(items, list) and items else items
        measures = _pick_measures(station, int(cfg.get("period_seconds", 900)))

    frames: List[pd.DataFrame] = []
    for channel, measure_id in measures.items():
        for m_start, m_end in _month_edges(str(window["start"]), str(window["end"])):
            raw_path = raw_dir / f"ukea_{site_no}_{channel}_{m_start}.csv"
            if raw_path.exists():
                text = raw_path.read_text(encoding="utf-8")
            else:
                try:
                    resp = H.get(
                        measure_id + "/readings.csv",
                        params={
                            "mineq-date": m_start,
                            "maxeq-date": m_end,
                            "_limit": 200000,
                        },
                        source="ukea",
                        accept="text/csv",
                        **H.http_kwargs(http_cfg),
                    )
                except H.FetchError as exc:
                    print(f"  [ukea] {site_no}/{channel} {m_start}: {exc}")
                    continue
                text = resp.text
                H.save_raw(raw_dir, raw_path.name, text)
            if not text.strip():
                continue
            try:
                chunk = pd.read_csv(io.StringIO(text))
            except Exception:  # malformed chunk - skip rather than kill the whole site
                continue
            if chunk.empty or "dateTime" not in chunk.columns:
                continue
            # `quality` of "Missing"/"Invalid" is the EA's own QC verdict - honour it.
            if "quality" in chunk.columns:
                chunk = chunk[~chunk["quality"].astype(str).isin(["Missing", "Invalid"])]
            frames.append(
                pd.DataFrame(
                    {
                        "timestamp": pd.to_datetime(
                            chunk["dateTime"], errors="coerce", format="ISO8601", utc=True
                        ),
                        "channel": channel,
                        "value": pd.to_numeric(chunk["value"], errors="coerce"),
                    }
                )
            )

    if not frames:
        return pd.DataFrame(columns=["timestamp", "channel", "value"])
    out = pd.concat(frames, ignore_index=True)
    return out.dropna(subset=["timestamp", "value"])


def site_metadata(site_no: str, index: Optional[pd.DataFrame]) -> Dict[str, Any]:
    if index is None or index.empty or "site_no" not in index.columns:
        return {"name": site_no, "lat": None, "lon": None, "region": "UK"}
    hit = index[index["site_no"].astype(str) == str(site_no)]
    if hit.empty:
        return {"name": site_no, "lat": None, "lon": None, "region": "UK"}
    r = hit.iloc[0]
    return {
        "name": r.get("station_nm", site_no),
        "lat": r.get("lat"),
        "lon": r.get("lon"),
        "region": "UK",
    }
