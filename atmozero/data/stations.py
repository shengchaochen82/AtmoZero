"""Virtual station network stratified across the 30 Köppen-Geiger zones."""
from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class Station:
    station_id: int
    lat: float
    lon: float
    elevation: float
    koppen_zone: str


# Approximate latitude bands of each Köppen zone, used to sample a
# geographically realistic point set without requiring the full raster.
_ZONE_LAT_BAND = {
    "Af":  (-15.0,  15.0), "Am":  (-15.0,  15.0), "Aw":  (-25.0,  25.0),
    "BWh": (-30.0,  30.0), "BWk": ( 30.0,  55.0),
    "BSh": (-30.0,  30.0), "BSk": ( 30.0,  55.0),
    "Csa": ( 25.0,  45.0), "Csb": ( 30.0,  55.0), "Csc": ( 40.0,  60.0),
    "Cwa": ( 15.0,  35.0), "Cwb": ( 15.0,  35.0), "Cwc": ( 20.0,  40.0),
    "Cfa": ( 25.0,  45.0), "Cfb": ( 35.0,  60.0), "Cfc": ( 45.0,  65.0),
    "Dsa": ( 35.0,  55.0), "Dsb": ( 40.0,  60.0), "Dsc": ( 45.0,  65.0), "Dsd": ( 50.0,  70.0),
    "Dwa": ( 30.0,  50.0), "Dwb": ( 35.0,  55.0), "Dwc": ( 45.0,  65.0), "Dwd": ( 50.0,  70.0),
    "Dfa": ( 35.0,  55.0), "Dfb": ( 40.0,  60.0), "Dfc": ( 50.0,  70.0), "Dfd": ( 55.0,  72.0),
    "ET":  ( 60.0,  82.0), "EF":  ( 70.0,  90.0),
}
ZONES = tuple(_ZONE_LAT_BAND.keys())


class VirtualStationNetwork:
    def __init__(self, stations: List[Station]):
        self.stations: List[Station] = stations

    def __len__(self) -> int:
        return len(self.stations)

    @classmethod
    def from_json(cls, path: Path) -> "VirtualStationNetwork":
        data = json.loads(Path(path).read_text())
        return cls([Station(**row) for row in data])

    def to_json(self, path: Path) -> None:
        Path(path).write_text(json.dumps([asdict(s) for s in self.stations], indent=2))

    @classmethod
    def build(
        cls,
        n_stations: int = 8192,
        koppen_grid_path: Optional[Path] = None,
        elevation_grid_path: Optional[Path] = None,
        min_separation_deg: float = 1.0,
        seed: int = 42,
    ) -> "VirtualStationNetwork":
        rng = random.Random(seed)

        if koppen_grid_path is not None and elevation_grid_path is not None:
            zone_grid, elev_grid = _load_rasters(koppen_grid_path, elevation_grid_path)
        else:
            zone_grid, elev_grid = None, None

        per_zone = max(1, n_stations // len(ZONES))
        stations: List[Station] = []
        accepted: List[Station] = []
        sid = 0
        for z in ZONES:
            placed = 0
            tries = 0
            while placed < per_zone and tries < per_zone * 50:
                tries += 1
                if zone_grid is not None:
                    lat, lon, elev = _sample_from_raster(rng, z, zone_grid, elev_grid)
                    if lat is None:
                        continue
                else:
                    lat_lo, lat_hi = _ZONE_LAT_BAND[z]
                    if z in ("Af", "Am", "Aw", "BWh", "BSh"):
                        lat = rng.uniform(lat_lo, lat_hi) * (1 if rng.random() < 0.5 else -1)
                    else:
                        lat = rng.uniform(lat_lo, lat_hi)
                    lon = rng.uniform(-180.0, 180.0)
                    elev = rng.uniform(0.0, 3000.0) if not z.startswith("E") else rng.uniform(500.0, 4000.0)

                if any(_haversine(lat, lon, s.lat, s.lon) < min_separation_deg * 111
                       for s in accepted):
                    continue
                stations.append(Station(sid, lat, lon, elev, z))
                accepted.append(stations[-1])
                placed += 1
                sid += 1
        return cls(stations[:n_stations])


def sample_virtual_stations(n: int = 8192, seed: int = 42):
    """Tabular DataFrame of N stratified virtual stations."""
    import pandas as pd
    network = VirtualStationNetwork.build(n_stations=n, seed=seed)
    rows = [asdict(s) for s in network.stations]
    return pd.DataFrame(rows, columns=["station_id", "lat", "lon", "elevation", "koppen_zone"])


def _load_rasters(zone_path: Path, elev_path: Path):
    import xarray as xr
    return xr.open_dataarray(zone_path), xr.open_dataarray(elev_path)


def _sample_from_raster(rng, zone: str, zone_grid, elev_grid):
    mask = (zone_grid == zone)
    flat_idx = mask.values.ravel()
    if not flat_idx.any():
        return None, None, None
    candidates = flat_idx.nonzero()[0]
    pick = rng.choice(list(candidates))
    rows, cols = mask.shape
    r, c = divmod(int(pick), cols)
    lat = float(zone_grid.coords["latitude"].values[r])
    lon = float(zone_grid.coords["longitude"].values[c])
    elev = float(elev_grid.values[r, c])
    return lat, lon, elev


def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))
