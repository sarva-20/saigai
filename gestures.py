"""Pure hand-gesture logic. This module must not access the OS or camera."""

from math import pi
from math import dist
from statistics import median
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
    """Normalize thumb-index distance by the larger stable palm dimension."""
    if len(landmarks) <= 17:
        raise ValueError("hand landmarks must include palm landmarks")

    palm_length = landmark_distance(landmarks[0], landmarks[9])
    palm_width = landmark_distance(landmarks[5], landmarks[17])
    hand_size = max(palm_length, palm_width)
    if hand_size <= 0:
        raise ValueError("hand size must be positive")
    return landmark_distance(landmarks[4], landmarks[8]) / hand_size


def normalized_thumb_middle_distance(
    landmarks: Sequence[Sequence[float]],
) -> float:
    if len(landmarks) <= 17:
        raise ValueError("hand landmarks must include palm landmarks")
    palm_length = landmark_distance(landmarks[0], landmarks[9])
    palm_width = landmark_distance(landmarks[5], landmarks[17])
    hand_size = max(palm_length, palm_width)
    if hand_size <= 0:
        raise ValueError("hand size must be positive")
    return landmark_distance(landmarks[4], landmarks[12]) / hand_size


class RatioMedianFilter:
    def __init__(self, size: int = 3) -> None:
        if size <= 0:
            raise ValueError("median filter size must be positive")
        self._values: deque[float] = deque(maxlen=size)

    def update(self, value: float) -> float:
        self._values.append(float(value))
        return float(median(self._values))


def calibrated_pinch_thresholds(
    open_ratio: float,
    pinched_ratio: float,
) -> tuple[float, float]:
    """Map a pinch between open/pinch samples at 35% and 55% of the range."""
    span = open_ratio - pinched_ratio
    if span <= 0:
        raise ValueError("open pinch ratio must exceed pinched ratio")
    press = pinched_ratio + 0.35 * span
    release = pinched_ratio + 0.55 * span
    return press, release


def is_scroll_pose(
    landmarks: Sequence[Sequence[float]],
    pinch_active: bool = False,
    extension_threshold: float = 1.15,
) -> bool:
    """Detect two-finger scroll independent of hand orientation."""
    if len(landmarks) <= 20 or pinch_active:
        return False
    extensions = finger_extension_states(landmarks, extension_threshold)
    index_extended = extensions["index"]
    middle_extended = extensions["middle"]
    ring_extended = extensions["ring"]
    pinky_extended = extensions["pinky"]
    ring_curled = not ring_extended
    pinky_curled = not pinky_extended
    return index_extended and middle_extended and ring_curled and pinky_curled


def finger_extension_states(
    landmarks: Sequence[Sequence[float]],
    threshold: float = 1.15,
) -> dict[str, bool]:
    if len(landmarks) <= 20:
        return {name: False for name in ("index", "middle", "ring", "pinky")}
    wrist = landmarks[0]
    values = {}
    for name, tip, pip in (
        ("index", 8, 6),
        ("middle", 12, 10),
        ("ring", 16, 14),
        ("pinky", 20, 18),
    ):
        tip_distance = landmark_distance(wrist, landmarks[tip])
        pip_distance = landmark_distance(wrist, landmarks[pip])
        values[name] = (
            pip_distance > 0
            and tip_distance / pip_distance >= threshold
        )
    return values


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
class InteractionEvent:
    state: str = "READY"
    left_pressed: bool = False
    right_pressed: bool = False
    left_released: bool = False
    left_clicked: bool = False
    drag_started: bool = False
    drag_ended: bool = False
    right_clicked: bool = False


class HandInteractionStateMachine:
    def __init__(
        self,
        left_press: float,
        left_release: float,
        right_press: float,
        right_release: float,
        drag_hold_ms: float = 250.0,
        click_max_duration: float = 0.30,
        click_flash_ms: float = 400.0,
        min_pinch_ms: float = 40.0,
    ) -> None:
        self.left_press = left_press
        self.left_release = left_release
        self.right_press = right_press
        self.right_release = right_release
        self.drag_hold_ms = drag_hold_ms
        self.click_max_duration = click_max_duration
        self.click_flash_ms = click_flash_ms
        self.min_pinch_ms = min_pinch_ms
        self.mode: Optional[str] = None
        self.state = "READY"
        self._press_time: Optional[float] = None
        self._click_flash_end: Optional[float] = None
        self.press_releases: deque[tuple[float, float]] = deque(maxlen=16)

    def reset(self) -> InteractionEvent:
        was_dragging = self.mode == "left" and self.state == "DRAGGING"
        was_left = self.mode == "left"
        self.mode = None
        self._press_time = None
        self._click_flash_end = None
        self.state = "NO HAND"
        return InteractionEvent(
            state="NO HAND",
            left_released=was_left,
            drag_ended=was_dragging,
        )

    def update(
        self,
        left_ratio: float,
        right_ratio: float,
        timestamp: float,
    ) -> InteractionEvent:
        if self.state == "CLICK" and self._click_flash_end is not None:
            if timestamp >= self._click_flash_end:
                self.state = "READY"
                self._click_flash_end = None
            else:
                return InteractionEvent(state="CLICK")

        if self.mode is None:
            if left_ratio <= self.left_press and right_ratio <= self.right_press:
                if left_ratio <= right_ratio:
                    kind = "left"
                else:
                    kind = "right"
            elif left_ratio <= self.left_press:
                kind = "left"
            elif right_ratio <= self.right_press:
                kind = "right"
            else:
                self.state = "READY"
                return InteractionEvent(state="READY")

            self.mode = kind
            self._press_time = timestamp
            self.state = "PINCHING"
            event = InteractionEvent(state="PINCHING")
            if kind == "left":
                event = InteractionEvent(state="PINCHING", left_pressed=True)
            else:
                event = InteractionEvent(state="PINCHING", right_pressed=True)
            return event

        elapsed_ms = (
            (timestamp - self._press_time) * 1000.0
            if self._press_time is not None
            else 0.0
        )
        if self.mode == "left":
            if left_ratio >= self.left_release:
                held_ms = elapsed_ms
                if held_ms < self.min_pinch_ms:
                    self.mode = None
                    self._press_time = None
                    self.state = "READY"
                    self.press_releases.append((held_ms, 0.0))
                    return InteractionEvent(state="READY")
                quick_click = held_ms < self.click_max_duration * 1000.0
                was_dragging = held_ms >= self.drag_hold_ms
                self.mode = None
                self._press_time = None
                self.state = "CLICK" if quick_click else "READY"
                self.press_releases.append((held_ms, 0.0))
                if quick_click:
                    self._click_flash_end = timestamp + self.click_flash_ms / 1000.0
                return InteractionEvent(
                    state=self.state,
                    left_released=True,
                    left_clicked=quick_click,
                    drag_ended=was_dragging,
                )
            if elapsed_ms >= self.drag_hold_ms:
                if self.state != "DRAGGING":
                    self.state = "DRAGGING"
                    return InteractionEvent(state="DRAGGING", drag_started=True)
                return InteractionEvent(state="DRAGGING")
            self.state = "PINCHING"
            return InteractionEvent(state="PINCHING")

        if right_ratio >= self.right_release:
            held_ms = elapsed_ms
            if held_ms < self.min_pinch_ms:
                self.mode = None
                self._press_time = None
                self.state = "READY"
                self.press_releases.append((held_ms, 0.0))
                return InteractionEvent(state="READY")
            quick_click = held_ms <= self.click_max_duration * 1000.0
            self.mode = None
            self._press_time = None
            self.state = "CLICK" if quick_click else "READY"
            if quick_click:
                self._click_flash_end = timestamp + self.click_flash_ms / 1000.0
            return InteractionEvent(
                state=self.state,
                left_released=True,
                right_clicked=quick_click,
            )
        self.state = "PINCHING"
        return InteractionEvent(state="PINCHING")


@dataclass(frozen=True)
class ScrollEvent:
    active: bool = False
    started: bool = False
    ended: bool = False
    delta: float = 0.0
    raw_midpoint_delta: float = 0.0
    deadzone_delta: float = 0.0
    scaled_delta: float = 0.0
    sent_amount: int = 0
    finger_states: dict[str, bool] | None = None


class ScrollStateMachine:
    def __init__(
        self,
        sensitivity: float,
        dead_zone: float,
        invert: bool = False,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        extension_threshold: float = 1.15,
    ) -> None:
        if sensitivity <= 0 or dead_zone < 0:
            raise ValueError("sensitivity must be positive and dead_zone non-negative")
        self.sensitivity = sensitivity
        self.dead_zone = dead_zone
        self.invert = invert
        self._filter = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        self.active = False
        self._previous_y: Optional[float] = None
        self.extension_threshold = extension_threshold
        self.remainder = 0.0
        self.last_diagnostics = {
            "raw_midpoint_delta": 0.0,
            "deadzone_delta": 0.0,
            "scaled_delta": 0.0,
            "sent_amount": 0,
        }

    def update(
        self,
        landmarks: Optional[Sequence[Sequence[float]]],
        timestamp: float,
        pinch_active: bool = False,
    ) -> ScrollEvent:
        pose_active = (
            landmarks is not None
            and is_scroll_pose(
                landmarks,
                pinch_active=pinch_active,
                extension_threshold=self.extension_threshold,
            )
        )
        fingers = (
            finger_extension_states(landmarks, self.extension_threshold)
            if landmarks is not None
            else {name: False for name in ("index", "middle", "ring", "pinky")}
        )
        if not pose_active:
            ended = self.active
            self.active = False
            self._previous_y = None
            self.remainder = 0.0
            self.last_diagnostics = {
                "raw_midpoint_delta": 0.0,
                "deadzone_delta": 0.0,
                "scaled_delta": 0.0,
                "sent_amount": 0,
            }
            return ScrollEvent(ended=ended, finger_states=fingers)

        midpoint_y = (landmarks[8][1] + landmarks[12][1]) / 2.0
        filtered_y = self._filter(midpoint_y, timestamp)
        started = not self.active
        self.active = True
        if started:
            self._previous_y = filtered_y
            return ScrollEvent(active=True, started=True, finger_states=fingers)

        movement = filtered_y - self._previous_y
        self._previous_y = filtered_y
        if abs(movement) <= self.dead_zone:
            return ScrollEvent(
                active=True,
                raw_midpoint_delta=movement,
                finger_states=fingers,
            )

        delta = -movement * self.sensitivity
        if self.invert:
            delta = -delta
        combined = delta + self.remainder
        sent_amount = int(combined)
        self.remainder = combined - sent_amount
        self.last_diagnostics = {
            "raw_midpoint_delta": movement,
            "deadzone_delta": movement,
            "scaled_delta": delta,
            "sent_amount": sent_amount,
        }
        return ScrollEvent(
            active=True,
            delta=float(sent_amount),
            raw_midpoint_delta=movement,
            deadzone_delta=movement,
            scaled_delta=delta,
            sent_amount=sent_amount,
            finger_states=fingers,
        )


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
