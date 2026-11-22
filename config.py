from dataclasses import dataclass


@dataclass(frozen=True)
class GestureThresholds:
    pinch_distance: float = 0.05
    extended_finger_margin: float = 0.02


GESTURE_THRESHOLDS = GestureThresholds()
