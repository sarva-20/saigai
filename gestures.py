"""Pure hand-gesture logic. This module must not access the OS or camera."""

from math import pi
from math import dist
from dataclasses import dataclass
from collections import deque
from time import monotonic
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


def is_scroll_pose(
    landmarks: Sequence[Sequence[float]],
    pinch_active: bool = False,
) -> bool:
    """Return whether index/middle are extended and ring/pinky are curled."""
    if len(landmarks) <= 20 or pinch_active:
        return False

    index_extended = landmarks[8][1] < landmarks[6][1]
    middle_extended = landmarks[12][1] < landmarks[10][1]
    ring_curled = landmarks[16][1] > landmarks[14][1]
    pinky_curled = landmarks[20][1] > landmarks[18][1]
    return index_extended and middle_extended and ring_curled and pinky_curled


def get_pointer_anchor(
    landmarks: Sequence[Sequence[float]],
    anchor: str = "palm",
    palm_mcp_weight: float = 0.70,
) -> Tuple[float, float, float]:
    """Return a stable pointer anchor without using the pinching fingertips."""
    if len(landmarks) <= 8:
        raise ValueError("hand landmarks must include landmark 8")
    if not 0.0 <= palm_mcp_weight <= 1.0:
        raise ValueError("palm_mcp_weight must be between 0 and 1")

    if anchor == "fingertip":
        return (
            float(landmarks[8][0]),
            float(landmarks[8][1]),
            float(landmarks[8][2]),
        )
    if anchor == "mcp":
        return (
            float(landmarks[5][0]),
            float(landmarks[5][1]),
            float(landmarks[5][2]),
        )
    if anchor == "palm":
        wrist = landmarks[0]
        mcp = landmarks[5]
        return tuple(
            (1.0 - palm_mcp_weight) * float(wrist[index])
            + palm_mcp_weight * float(mcp[index])
            for index in range(3)
        )
    raise ValueError("anchor must be 'palm', 'mcp', or 'fingertip'")


def relative_pointer_delta(
    previous: Sequence[float],
    current: Sequence[float],
    delta_time: float,
    screen_width: float,
    screen_height: float,
    gain: float = 1.0,
    acceleration: float = 0.35,
    speed_reference: float = 0.8,
    invert_x: bool = False,
    invert_y: bool = False,
) -> Tuple[float, float]:
    """Convert normalized hand motion into mirrored, accelerated screen motion."""
    if delta_time <= 0 or screen_width <= 0 or screen_height <= 0:
        raise ValueError("time and screen dimensions must be positive")
    if gain <= 0 or acceleration < 0 or speed_reference <= 0:
        raise ValueError("gain and speed_reference must be positive")

    delta_x = float(current[0] - previous[0])
    delta_y = float(current[1] - previous[1])
    speed = (delta_x**2 + delta_y**2) ** 0.5 / delta_time
    speed_ratio = min(speed / speed_reference, 1.0)
    motion_gain = gain * (1.0 + acceleration * speed_ratio)
    if invert_x:
        delta_x = -delta_x
    if invert_y:
        delta_y = -delta_y
    return (
        delta_x * screen_width * motion_gain,
        delta_y * screen_height * motion_gain,
    )


class StillnessLock:
    def __init__(self, threshold: float) -> None:
        if threshold < 0:
            raise ValueError("stillness threshold must be non-negative")
        self.threshold = threshold
        self._previous: Optional[Sequence[float]] = None
        self.locked = False

    def update(self, position: Sequence[float]) -> bool:
        if self._previous is None:
            self._previous = position
            return False
        movement = landmark_distance(self._previous, position)
        self._previous = position
        self.locked = movement <= self.threshold
        return self.locked


class PointerHistory:
    def __init__(self, max_length: int = 32) -> None:
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self._positions: deque[Tuple[float, Tuple[float, float]]] = deque(
            maxlen=max_length
        )

    def add(
        self,
        position: Tuple[float, float],
        timestamp: Optional[float] = None,
    ) -> None:
        self._positions.append((monotonic() if timestamp is None else timestamp, position))

    def latch(
        self,
        timestamp: float,
        milliseconds_earlier: float,
    ) -> Optional[Tuple[float, float]]:
        target = timestamp - milliseconds_earlier / 1000.0
        candidates = [position for time, position in self._positions if time <= target]
        if candidates:
            return candidates[-1]
        return self._positions[0][1] if self._positions else None


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


@dataclass(frozen=True)
class ScrollEvent:
    active: bool = False
    started: bool = False
    ended: bool = False
    delta: float = 0.0


class ScrollStateMachine:
    def __init__(
        self,
        sensitivity: float,
        dead_zone: float,
        invert: bool = False,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
    ) -> None:
        if sensitivity <= 0 or dead_zone < 0:
            raise ValueError("sensitivity must be positive and dead_zone non-negative")
        self.sensitivity = sensitivity
        self.dead_zone = dead_zone
        self.invert = invert
        self._filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self.active = False
        self._previous_y: Optional[float] = None

    def update(
        self,
        landmarks: Optional[Sequence[Sequence[float]]],
        timestamp: float,
        pinch_active: bool = False,
    ) -> ScrollEvent:
        pose_active = (
            landmarks is not None
            and is_scroll_pose(landmarks, pinch_active=pinch_active)
        )
        if not pose_active:
            ended = self.active
            self.active = False
            self._previous_y = None
            return ScrollEvent(ended=ended)

        midpoint_y = (landmarks[8][1] + landmarks[12][1]) / 2.0
        filtered_y = self._filter(midpoint_y, timestamp)
        started = not self.active
        self.active = True
        if started:
            self._previous_y = filtered_y
            return ScrollEvent(active=True, started=True)

        movement = filtered_y - self._previous_y
        self._previous_y = filtered_y
        if abs(movement) <= self.dead_zone:
            return ScrollEvent(active=True)

        delta = -movement * self.sensitivity
        if self.invert:
            delta = -delta
        return ScrollEvent(active=True, delta=delta)


def map_pointer_to_screen(
    landmark: Sequence[float],
    frame_width: int,
    frame_height: int,
    screen_width: float,
    screen_height: float,
    active_region_margin: float = 0.20,
    active_region: Optional[Tuple[float, float, float, float]] = None,
    invert_x: bool = False,
    invert_y: bool = False,
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
    if active_region is None:
        left = active_region_margin
        right = 1.0 - active_region_margin
        top = active_region_margin
        bottom = 1.0 - active_region_margin
    else:
        left, top, right, bottom = active_region
        if not 0 <= left < right <= 1 or not 0 <= top < bottom <= 1:
            raise ValueError("active_region must contain normalized bounds")

    normalized_x = (max(left, min(right, x)) - left) / (right - left)
    normalized_y = (max(top, min(bottom, y)) - top) / (bottom - top)

    if invert_x:
        normalized_x = 1.0 - normalized_x
    if invert_y:
        normalized_y = 1.0 - normalized_y
    return normalized_x * screen_width, normalized_y * screen_height


def map_index_finger_to_screen(*args, **kwargs) -> Tuple[float, float]:
    """Compatibility alias for the original absolute pointer mapping."""
    return map_pointer_to_screen(*args, **kwargs)


def map_to_screen(*args, **kwargs) -> Tuple[float, float]:
    """Backward-friendly short name for active-region screen mapping."""
    return map_pointer_to_screen(*args, **kwargs)


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
