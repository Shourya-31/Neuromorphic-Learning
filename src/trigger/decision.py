from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class TriggerDecision:
    """
    Standard trigger output used across the trigger subsystem.

    Produced by:
        TriggerPolicy

    Consumed by:
        BypassController
        Metrics
        Logging
        Visualization
    """

    score: float
    threshold: float

    triggered: bool

    spike_density: float
    temporal_variance: float

    metadata: Optional[Dict[str, Any]] = None

    @property
    def should_refine(self) -> bool:
        return self.triggered

    @property
    def should_bypass(self) -> bool:
        return not self.triggered