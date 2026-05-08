"""Cold-start protocol: hold the Proposer to a uniform prior until tau_cold."""
from __future__ import annotations

from collections import deque
from typing import Optional


COLD_START_THRESHOLD = 0.84
RUNNING_WINDOW = 50


class ColdStartController:
    def __init__(self, threshold: float = COLD_START_THRESHOLD, smoothing: int = RUNNING_WINDOW):
        self.threshold = threshold
        self.smoothing = smoothing
        self.precisions: deque[float] = deque(maxlen=smoothing)
        self.engaged_at_step: Optional[int] = None

    def update(self, step: int, precision: float) -> bool:
        """Returns True exactly on the step where the threshold is reached."""
        self.precisions.append(precision)
        if self.engaged_at_step is not None:
            return False
        if len(self.precisions) < self.smoothing // 2:
            return False
        rolling = sum(self.precisions) / len(self.precisions)
        if rolling >= self.threshold:
            self.engaged_at_step = step
            return True
        return False
