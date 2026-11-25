"""Read the cached manifest and yield per-site client datasets.

This is the only door between collected real data and the rest of the project. A federated
``PondClient`` gets exactly one ``SiteDataset`` and never sees another's raw arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd

from ..common.runtime import ROOT, load_config
from .harmonize import derive_nh3


@dataclass
class SiteDataset:
    """One monitoring site = one independent pond/client."""

    client_id: str
    source: str
    site_no: str
    name: str
    region: str
    channels: List[str]                 # channels this site genuinely reports
    frame: pd.DataFrame                 # timestamp-indexed values (may contain NaN)
    mask: pd.DataFrame                  # True where the value is a real observation
    cadence_minutes: int
    license: str

    def __len__(self) -> int:
        return len(self.frame)

    def matrix(self, channels: Sequence[str]) -> np.ndarray:
        """(T, C) array over the requested channels; missing channels become all-NaN columns."""
        cols = []
        for c in channels:
            cols.append(
                self.frame[c].to_numpy(dtype=float)
                if c in self.frame.columns
                else np.full(len(self.frame), np.nan)
            )
        return np.column_stack(cols) if cols else np.empty((len(self.frame), 0))

    def mask_matrix(self, channels: Sequence[str]) -> np.ndarray:
        cols = []
        for c in channels:
            cols.append(
                self.mask[c].to_numpy(dtype=bool)
                if c in self.mask.columns
                else np.zeros(len(self.frame), dtype=bool)
            )
        return np.column_stack(cols) if cols else np.empty((len(self.frame), 0), dtype=bool)

    def with_nh3(self) -> pd.DataFrame:
        """Add the derived un-ionised NH3 column (needs tan + ph + temp_c)."""
        f = self.frame.copy()
        if {"tan", "ph", "temp_c"}.issubset(f.columns):
            f["nh3"] = derive_nh3(
                f["tan"].to_numpy(), f["ph"].to_numpy(), f["temp_c"].to_numpy()
            )
        else:
            f["nh3"] = np.nan
        return f

    def coverage(self, channels: Sequence[str]) -> Dict[str, float]:
        return {c: float(self.mask[c].mean()) if c in self.mask.columns else 0.0 for c in channels}


def manifest(path: Optional[Path] = None) -> pd.DataFrame:
    cfg = load_config("data")
    p = path or (ROOT / cfg["cache"]["manifest"])
    if not Path(p).exists():
        raise FileNotFoundError(
            f"{p} not found. Run the one-time online step first:\n"
            f"    python scripts/collect_data.py"
        )
    return pd.read_csv(p, dtype={"site_no": str})


def load_site(row: pd.Series) -> SiteDataset:
    df = pd.read_parquet(ROOT / row["path"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.set_index("timestamp").sort_index()
    mask_cols = [c for c in df.columns if c.startswith("mask_")]
    mask = df[mask_cols].rename(columns=lambda c: c[len("mask_") :]).astype(bool)
    values = df[[c for c in df.columns if not c.startswith("mask_")]]
    return SiteDataset(
        client_id=row["client_id"],
        source=row["source"],
        site_no=str(row["site_no"]),
        name=str(row.get("name", "")),
        region=str(row.get("region", "")),
        channels=str(row["channels"]).split("|") if isinstance(row["channels"], str) else [],
        frame=values,
        mask=mask,
        cadence_minutes=int(row["cadence_minutes"]),
        license=str(row.get("license", "")),
    )


def pond_manifest_path() -> Path:
    """The manifest listing the pond sites. The ponds are now the only data source, so this is
    the main manifest unless ``sources.kaggle_pond.manifest`` names a separate file."""
    cfg = load_config("data")
    p = cfg["sources"].get("kaggle_pond", {}).get("manifest") or cfg["cache"]["manifest"]
    return ROOT / p


def iter_sites(
    min_channels: int = 0,
    require: Sequence[str] = (),
    sources: Optional[Sequence[str]] = None,
    manifest_path: Optional[Path] = None,
) -> Iterator[SiteDataset]:
    """Sites from the main manifest, or from ``manifest_path`` if given."""
    mf = manifest(manifest_path)
    for _, row in mf.iterrows():
        chans = set(str(row["channels"]).split("|"))
        if int(row["n_channels"]) < min_channels:
            continue
        if require and not set(require).issubset(chans):
            continue
        if sources and row["source"] not in sources:
            continue
        yield load_site(row)


def load_all(**kwargs) -> List[SiteDataset]:
    return list(iter_sites(**kwargs))


def summary(manifest_path: Optional[Path] = None) -> Dict[str, object]:
    """Numbers the README quotes - always read from the manifest, never transcribed."""
    mf = manifest(manifest_path)
    return {
        "n_sites": int(len(mf)),
        "n_rows": int(mf["rows"].sum()),
        "sources": mf["source"].value_counts().to_dict(),
        "regions": int(mf["region"].nunique()),
        "date_range": (str(mf["start"].min()), str(mf["end"].max())),
        "mean_channels": round(float(mf["n_channels"].mean()), 2),
    }
