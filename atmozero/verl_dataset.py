"""AtmoZero data wrapper around verl's RLHFDataset.

The dataset reads a parquet file produced by ``scripts/prepare_verl_data.py``
where each row carries ``prompt`` and an ``extra_info`` dict shipping the
window context the verifier needs. Cold-start gating is implemented by the
:class:`atmozero.proposer.WindowSampler` shared with the reward worker.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from verl.utils.dataset.rl_dataset import RLHFDataset
except ImportError:
    RLHFDataset = object  # type: ignore[assignment, misc]


class AtmoZeroDataset(RLHFDataset):
    """RLHFDataset subclass that consults the AtmoZero Proposer for index selection.

    Falls back to the parent's behaviour when the cold-start gate has not yet
    engaged.
    """

    def __init__(self, *args, **kwargs):
        if RLHFDataset is object:
            raise ImportError(
                "verl is required to use AtmoZeroDataset. Install via `pip install verl`."
            )
        super().__init__(*args, **kwargs)
        from atmozero.proposer import WindowSampler

        state_path = os.environ.get("ATMOZERO_STATE_PATH")
        n_windows = int(os.environ.get("ATMOZERO_N_WINDOWS", len(self)))
        self._sampler = (
            WindowSampler(n_windows=n_windows, state_path=Path(state_path))
            if state_path is not None
            else None
        )
        self._rng = np.random.default_rng(int(os.environ.get("ATMOZERO_SAMPLER_SEED", 0)))

    def __getitem__(self, idx):
        if self._sampler is not None and self._sampler.cold.is_engaged():
            idx = self._sampler.sample(1, rng=self._rng)[0] % len(self)
        return super().__getitem__(idx)
