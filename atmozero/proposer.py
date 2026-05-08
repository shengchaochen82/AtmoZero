"""AtmoZero Proposer: window-difficulty sampler with the cold-start gate.

The Proposer is *not* a neural policy under verl; verl's actor optimization
already handles the LM. The Proposer here decides which (window, regime) row
the data loader emits next, gated by the running verifier precision per the
cold-start protocol.

State is persisted to a JSON file so the verl reward worker (which scores
rollouts) and the verl data worker (which samples next batches) can share
the same view. Both processes read/write atomically by replacing the file.
"""
from __future__ import annotations

import json
import os
import tempfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional

import numpy as np


COLD_START_THRESHOLD = 0.84
RUNNING_WINDOW = 50


@dataclass
class _State:
    step: int = 0
    engaged_at_step: Optional[int] = None
    rolling_precisions: List[float] = field(default_factory=list)
    per_window_difficulty: Dict[int, float] = field(default_factory=dict)


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp_", suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def _read_state(path: Path) -> _State:
    if not path.exists():
        return _State()
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return _State()
    return _State(
        step=int(data.get("step", 0)),
        engaged_at_step=data.get("engaged_at_step"),
        rolling_precisions=list(data.get("rolling_precisions", [])),
        per_window_difficulty={int(k): float(v) for k, v in data.get("per_window_difficulty", {}).items()},
    )


def _write_state(path: Path, st: _State) -> None:
    _atomic_write(path, {
        "step": st.step,
        "engaged_at_step": st.engaged_at_step,
        "rolling_precisions": st.rolling_precisions[-RUNNING_WINDOW:],
        "per_window_difficulty": {str(k): v for k, v in st.per_window_difficulty.items()},
    })


class ColdStartController:
    """Gates the Proposer's switch from uniform to adversarial sampling."""

    def __init__(
        self,
        state_path: str | Path,
        threshold: float = COLD_START_THRESHOLD,
        smoothing: int = RUNNING_WINDOW,
    ):
        self.state_path = Path(state_path)
        self.threshold = threshold
        self.smoothing = smoothing

    def update(self, precision: float) -> bool:
        """Append a precision sample. Returns True the first step the gate opens."""
        st = _read_state(self.state_path)
        st.step += 1
        st.rolling_precisions.append(float(precision))
        st.rolling_precisions = st.rolling_precisions[-self.smoothing:]
        just_engaged = False
        if st.engaged_at_step is None and len(st.rolling_precisions) >= self.smoothing // 2:
            rolling = sum(st.rolling_precisions) / len(st.rolling_precisions)
            if rolling >= self.threshold:
                st.engaged_at_step = st.step
                just_engaged = True
        _write_state(self.state_path, st)
        return just_engaged

    def is_engaged(self) -> bool:
        return _read_state(self.state_path).engaged_at_step is not None


class WindowSampler:
    """Per-window difficulty proxy with uniform-vs-weighted sampling.

    The verl reward function calls ``record(window_id, precision)`` after each
    rollout. The verl dataset calls ``sample(batch_size)`` to pick the next
    batch of window indices.
    """

    def __init__(
        self,
        n_windows: int,
        state_path: str | Path,
        cold_start_threshold: float = COLD_START_THRESHOLD,
        smoothing: int = RUNNING_WINDOW,
    ):
        self.n_windows = int(n_windows)
        self.state_path = Path(state_path)
        self.cold = ColdStartController(state_path, cold_start_threshold, smoothing)

    def record(self, window_id: int, precision: float) -> None:
        """Update the per-window difficulty (1 - precision) and the cold-start gate."""
        st = _read_state(self.state_path)
        st.step += 1
        st.rolling_precisions.append(float(precision))
        st.rolling_precisions = st.rolling_precisions[-RUNNING_WINDOW:]
        st.per_window_difficulty[int(window_id)] = float(max(0.0, 1.0 - precision))
        if st.engaged_at_step is None and len(st.rolling_precisions) >= RUNNING_WINDOW // 2:
            rolling = sum(st.rolling_precisions) / len(st.rolling_precisions)
            if rolling >= self.cold.threshold:
                st.engaged_at_step = st.step
        _write_state(self.state_path, st)

    def sample(self, batch_size: int, rng: Optional[np.random.Generator] = None) -> List[int]:
        rng = rng or np.random.default_rng()
        st = _read_state(self.state_path)
        if st.engaged_at_step is None:
            return rng.integers(0, self.n_windows, size=batch_size).tolist()
        weights = np.full(self.n_windows, 1e-3, dtype=np.float64)
        for k, v in st.per_window_difficulty.items():
            if 0 <= k < self.n_windows:
                weights[k] = v + 1e-3
        weights = weights / weights.sum()
        return rng.choice(self.n_windows, size=batch_size, p=weights).tolist()
