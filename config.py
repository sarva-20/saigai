from dataclasses import dataclass


@dataclass(frozen=True)
class GestureThresholds:
    pinch_distance: float = 0.05
    extended_finger_margin: float = 0.02
    active_region_margin: float = 0.20
    one_euro_min_cutoff: float = 1.0
    one_euro_beta: float = 0.007


GESTURE_THRESHOLDS = GestureThresholds()
