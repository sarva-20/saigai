"""Pure hand-gesture logic. This module must not access the OS or camera."""

from math import pi
from math import dist
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


def landmark_distance(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    return dist(first[:3] if len(first) >= 3 else first, second[:3] if len(second) >= 3 else second)


def is_pinching(
    landmarks: Sequence[Sequence[float]],
    threshold: float,
) -> bool:
    if len(landmarks) <= 8:
        return False

    return landmark_distance(landmarks[4], landmarks[8]) < threshold


def normalized_pinch_distance(landmarks: Sequence[Sequence[float]]) -> float:
    """Return thumb-index distance normalized by wrist-to-middle-MCP size."""
    if len(landmarks) <= 9:
        raise ValueError("hand landmarks must include landmarks 0, 4, 8, and 9")

    hand_size = landmark_distance(landmarks[0], landmarks[9])
    if hand_size <= 0:
        raise ValueError("hand size must be positive")
    return landmark_distance(landmarks[4], landmarks[8]) / hand_size


@dataclass(frozen=True)
class PinchEvent:
    pressed: bool = False
    released: bool = False
    clicked: bool = False
    dragging: bool = False


class PinchStateMachine:
    """Hysteretic pinch state machine with click and drag classification."""

    def __init__(
        self,
        press_threshold: float,
        release_threshold: float,
        click_max_duration: float = 0.30,
    ) -> None:
        if press_threshold <= 0 or release_threshold <= press_threshold:
            raise ValueError("release threshold must exceed a positive press threshold")
        if click_max_duration <= 0:
            raise ValueError("click_max_duration must be positive")
        self.press_threshold = press_threshold
        self.release_threshold = release_threshold
        self.click_max_duration = click_max_duration
        self.pressed = False
        self._press_time: Optional[float] = None

    def update(self, normalized_distance: float, timestamp: float) -> PinchEvent:
        if not self.pressed:
            if normalized_distance <= self.press_threshold:
                self.pressed = True
                self._press_time = timestamp
                return PinchEvent(pressed=True)
            return PinchEvent()

        if normalized_distance < self.release_threshold:
            return PinchEvent(dragging=True)

        press_time = self._press_time
        self.pressed = False
        self._press_time = None
        clicked = press_time is not None and timestamp - press_time < self.click_max_duration
        return PinchEvent(released=True, clicked=clicked)


def map_index_finger_to_screen(
    landmark: Sequence[float],
    frame_width: int,
    frame_height: int,
    screen_width: float,
    screen_height: float,
    active_region_margin: float = 0.20,
) -> Tuple[float, float]:
    """Map normalized landmark 8 coordinates into the active screen region."""
    if not 0.0 <= active_region_margin < 0.5:
        raise ValueError("active_region_margin must be between 0 and 0.5")
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")
    if screen_width <= 0 or screen_height <= 0:
        raise ValueError("screen dimensions must be positive")

    x = float(landmark[0])
    y = float(landmark[1])
    left = active_region_margin
    right = 1.0 - active_region_margin
    top = active_region_margin
    bottom = 1.0 - active_region_margin

    normalized_x = (max(left, min(right, x)) - left) / (right - left)
    normalized_y = (max(top, min(bottom, y)) - top) / (bottom - top)

    # Mirroring the horizontal axis makes fingertip movement follow the
    # mirrored preview naturally.
    return (1.0 - normalized_x) * screen_width, normalized_y * screen_height


def map_to_screen(*args, **kwargs) -> Tuple[float, float]:
    """Backward-friendly short name for active-region screen mapping."""
    return map_index_finger_to_screen(*args, **kwargs)


class OneEuroFilter:
    def __init__(
        self,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        derivate_cutoff: float = 1.0,
    ) -> None:
        if min_cutoff <= 0 or derivate_cutoff <= 0:
            raise ValueError("cutoff frequencies must be positive")
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivate_cutoff = derivate_cutoff
        self._x_previous: Optional[float] = None
        self._dx_previous = 0.0
        self._timestamp_previous: Optional[float] = None

    @staticmethod
    def _smoothing_factor(cutoff: float, delta_time: float) -> float:
        tau = 1.0 / (2.0 * pi * cutoff)
        return 1.0 / (1.0 + tau / delta_time)

    def __call__(self, value: float, timestamp: float) -> float:
        value = float(value)
        if self._x_previous is None or self._timestamp_previous is None:
            self._x_previous = value
            self._timestamp_previous = timestamp
            return value

        delta_time = timestamp - self._timestamp_previous
        if delta_time <= 0:
            return self._x_previous

        derivative = (value - self._x_previous) / delta_time
        alpha_derivative = self._smoothing_factor(
            self.derivate_cutoff,
            delta_time,
        )
        self._dx_previous = (
            alpha_derivative * derivative
            + (1.0 - alpha_derivative) * self._dx_previous
        )

        cutoff = self.min_cutoff + self.beta * abs(self._dx_previous)
        alpha = self._smoothing_factor(cutoff, delta_time)
        filtered = alpha * value + (1.0 - alpha) * self._x_previous

        self._x_previous = filtered
        self._timestamp_previous = timestamp
        return filtered
