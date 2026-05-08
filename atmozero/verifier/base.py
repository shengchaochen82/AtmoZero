"""Abstract base class and registry for the seven station-observable rule families."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Type

import numpy as np


@dataclass
class WindowContext:
    x: Dict[str, np.ndarray]
    metadata: Dict[str, float]
    neighbors: Optional[List[Dict[str, np.ndarray]]] = None
    climatology: Optional[Dict[str, float]] = None


class RuleFamily(ABC):
    family_id: int
    family_name: str
    channels: tuple
    lexicon: tuple
    tau_minus: float = 0.30
    tau_plus: float = 0.70

    def __init__(self, tau_minus: Optional[float] = None, tau_plus: Optional[float] = None):
        if tau_minus is not None:
            self.tau_minus = tau_minus
        if tau_plus is not None:
            self.tau_plus = tau_plus

    @abstractmethod
    def trigger(self, ctx: WindowContext) -> int:
        ...

    @abstractmethod
    def grade(self, v_q: str, ctx: WindowContext) -> float:
        ...

    def has_required_channels(self, available: set) -> bool:
        return all(ch in available for ch in self.channels)

    def supports(self, v_q: str, ctx: WindowContext) -> bool:
        return self.grade(v_q, ctx) >= self.tau_plus

    def refutes(self, v_q: str, ctx: WindowContext) -> bool:
        return self.grade(v_q, ctx) <= self.tau_minus


RULE_REGISTRY: Dict[int, Type[RuleFamily]] = {}


def register_rule(cls: Type[RuleFamily]) -> Type[RuleFamily]:
    if not issubclass(cls, RuleFamily):
        raise TypeError(f"{cls.__name__} must subclass RuleFamily")
    if cls.family_id in RULE_REGISTRY:
        raise ValueError(f"Family {cls.family_id} already registered")
    RULE_REGISTRY[cls.family_id] = cls
    return cls
