"""ERA5 hourly single-level reanalysis loader (ARCO-ERA5 public Zarr)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

ARCO_ERA5_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

ERA5_VARIABLES = {
    "T":  "2m_temperature",
    "P":  "mean_sea_level_pressure",
    "Td": "2m_dewpoint_temperature",
    "u":  "10m_u_component_of_wind",
    "v":  "10m_v_component_of_wind",
    "r":  "total_precipitation",
}


@dataclass
class WindowSpec:
    station_id: int
    lat: float
    lon: float
    elevation: float
    koppen_zone: str
    t_start: np.datetime64
    T_w: int = 192


class ERA5Loader:
    def __init__(self, source: str = ARCO_ERA5_URL):
        self.source = source
        self._ds = None

    def _open(self):
        if self._ds is None:
            import xarray as xr
            self._ds = xr.open_zarr(self.source, consolidated=True, chunks={"time": 24})
        return self._ds

    def fetch(self, spec: WindowSpec) -> Dict[str, np.ndarray]:
        ds = self._open()
        t_end = spec.t_start + np.timedelta64(spec.T_w - 1, "h")
        sub = ds.sel(time=slice(spec.t_start, t_end))
        sub = sub.sel(latitude=_snap(spec.lat), longitude=_snap(spec.lon), method="nearest")

        out: Dict[str, np.ndarray] = {}
        for short, full in ERA5_VARIABLES.items():
            if full in sub:
                out[short] = sub[full].values

        if "T" in out and "Td" in out:
            out["q"] = _q_from_dewpoint(out["T"], out["Td"], P_pa=out.get("P"))
            out.pop("Td")

        if "P" in out:
            out["P"] = _adjust_pressure(out["P"], spec.elevation)

        # ARCO-ERA5 stores per-hour de-accumulated total precipitation, so a
        # *1000 conversion gives mm/h directly. For raw ECMWF chunks (which
        # are cumulative since 00:00), de-accumulate before this multiply.
        if "r" in out:
            out["r"] = out["r"] * 1000.0
        return out


def load_era5_window(
    station_id: int,
    lat: float,
    lon: float,
    elevation: float,
    koppen_zone: str,
    t_start: np.datetime64,
    T_w: int = 192,
    source: str = ARCO_ERA5_URL,
) -> Dict[str, np.ndarray]:
    return ERA5Loader(source).fetch(
        WindowSpec(station_id, lat, lon, elevation, koppen_zone, t_start, T_w)
    )


def _snap(value: float, resolution: float = 0.25) -> float:
    return round(value / resolution) * resolution


def _q_from_dewpoint(T_k: np.ndarray, Td_k: np.ndarray, P_pa: Optional[np.ndarray] = None) -> np.ndarray:
    Td_c = Td_k - 273.15
    e = 6.112 * np.exp(17.67 * Td_c / (Td_c + 243.5))
    P_hpa = (P_pa / 100.0) if P_pa is not None else 1013.0
    return 0.622 * e / (P_hpa - 0.378 * e)


def _adjust_pressure(mslp_pa: np.ndarray, elevation_m: float, scale_height_m: float = 8400.0) -> np.ndarray:
    """P = MSLP * exp(-h / H_s), in hPa."""
    return (mslp_pa / 100.0) * np.exp(-elevation_m / scale_height_m)
