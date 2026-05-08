"""Spatial neighbour graph for the spatial-coherence rule family."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List

from atmozero.data.stations import Station


@dataclass
class NeighborGraph:
    K: int = 8
    radius_km: float = 250.0
    edges: Dict[int, List[int]] = field(default_factory=dict)

    @classmethod
    def build(cls, stations: List[Station], K: int = 8, radius_km: float = 250.0) -> "NeighborGraph":
        edges: Dict[int, List[int]] = {}
        for s in stations:
            ranked = sorted(
                (n for n in stations if n.station_id != s.station_id),
                key=lambda n: _haversine(s.lat, s.lon, n.lat, n.lon),
            )
            within = [
                n.station_id for n in ranked
                if _haversine(s.lat, s.lon, n.lat, n.lon) <= radius_km
            ]
            edges[s.station_id] = within[:K]
        return cls(K=K, radius_km=radius_km, edges=edges)


def build_neighbor_graph(stations, K: int = 8, radius_km: float = 250.0):
    """DataFrame of (focal, K-neighbour) edges with columns
    ``station_id, neighbor_id, rank, distance_km``."""
    import pandas as pd

    if hasattr(stations, "iterrows"):
        rows: List[Station] = [
            Station(int(r.station_id), float(r.lat), float(r.lon),
                    float(r.elevation), str(r.koppen_zone))
            for _, r in stations.iterrows()
        ]
    else:
        rows = list(stations)

    out_rows = []
    for s in rows:
        ranked = sorted(
            ((n, _haversine(s.lat, s.lon, n.lat, n.lon)) for n in rows
             if n.station_id != s.station_id),
            key=lambda kv: kv[1],
        )
        within = [(n, d) for (n, d) in ranked if d <= radius_km][:K]
        for rank, (n, d) in enumerate(within):
            out_rows.append({
                "station_id": s.station_id,
                "neighbor_id": n.station_id,
                "rank": rank,
                "distance_km": float(d),
            })
    return pd.DataFrame(out_rows, columns=["station_id", "neighbor_id", "rank", "distance_km"])


def _haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return 2 * R * math.asin(math.sqrt(a))
