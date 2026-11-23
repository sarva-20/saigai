"""Quartz mouse and scroll control with a live-action safety gate."""

from dataclasses import dataclass
from math import exp
from threading import Event, Lock, Thread
from time import monotonic, sleep
from typing import Optional


@dataclass(frozen=True)
class Controller:
    live: bool = False
    screen_width: Optional[float] = None
    screen_height: Optional[float] = None

    def require_live(self) -> None:
        if not self.live:
            raise RuntimeError("OS control is disabled; pass --live to enable it")

    def screen_size(self) -> tuple[float, float]:
        if self.screen_width is not None and self.screen_height is not None:
            return self.screen_width, self.screen_height

        try:
            import Quartz

            bounds = Quartz.CGDisplayBounds(Quartz.CGMainDisplayID())
            return bounds.size.width, bounds.size.height
        except ImportError as exc:
            raise RuntimeError("Quartz mouse control requires macOS") from exc

    def move_mouse(self, x: float, y: float) -> bool:
        if not self.live:
            return False

        try:
            import Quartz

            point = Quartz.CGPoint(x=x, y=y)
            event = Quartz.CGEventCreateMouseEvent(
                None,
                Quartz.kCGEventMouseMoved,
                point,
                Quartz.kCGMouseButtonLeft,
            )
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
            return True
        except ImportError as exc:
            raise RuntimeError("Quartz mouse control requires macOS") from exc

    def _mouse_event(self, event_type: int, x: float, y: float) -> bool:
        if not self.live:
            return False

        try:
            import Quartz

            point = Quartz.CGPoint(x=x, y=y)
            event = Quartz.CGEventCreateMouseEvent(
                None,
                event_type,
                point,
                Quartz.kCGMouseButtonLeft,
            )
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
            return True
        except ImportError as exc:
            raise RuntimeError("Quartz mouse control requires macOS") from exc

    def mouse_down(self, x: float, y: float) -> bool:
        try:
            import Quartz
        except ImportError as exc:
            raise RuntimeError("Quartz mouse control requires macOS") from exc
        return self._mouse_event(Quartz.kCGEventLeftMouseDown, x, y)

    def mouse_dragged(self, x: float, y: float) -> bool:
        try:
            import Quartz
        except ImportError as exc:
            raise RuntimeError("Quartz mouse control requires macOS") from exc
        return self._mouse_event(Quartz.kCGEventLeftMouseDragged, x, y)

    def mouse_up(self, x: float, y: float) -> bool:
        try:
            import Quartz
        except ImportError as exc:
            raise RuntimeError("Quartz mouse control requires macOS") from exc
        return self._mouse_event(Quartz.kCGEventLeftMouseUp, x, y)

    def right_click(self, x: float, y: float) -> bool:
        try:
            import Quartz
        except ImportError as exc:
            raise RuntimeError("Quartz mouse control requires macOS") from exc
        if not self.live:
            return False
        for event_type in (
            Quartz.kCGEventRightMouseDown,
            Quartz.kCGEventRightMouseUp,
        ):
            point = Quartz.CGPoint(x=x, y=y)
            event = Quartz.CGEventCreateMouseEvent(
                None,
                event_type,
                point,
                Quartz.kCGMouseButtonRight,
            )
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        return True

    def left_click(self, x: float, y: float, click_count: int = 1) -> bool:
        if not self.live:
            return False
        try:
            import Quartz
        except ImportError as exc:
            raise RuntimeError("Quartz mouse control requires macOS") from exc
        point = Quartz.CGPoint(x=x, y=y)
        for event_type in (
            Quartz.kCGEventLeftMouseDown,
            Quartz.kCGEventLeftMouseUp,
        ):
            event = Quartz.CGEventCreateMouseEvent(
                None,
                event_type,
                point,
                Quartz.kCGMouseButtonLeft,
            )
            Quartz.CGEventSetIntegerValueField(
                event,
                Quartz.kCGMouseEventClickState,
                click_count,
            )
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
        return True

    def scroll(self, delta: float, x: Optional[float] = None, y: Optional[float] = None) -> bool:
        if not self.live:
            return False

        try:
            import Quartz

            event = Quartz.CGEventCreateScrollWheelEvent(
                None,
                Quartz.kCGScrollEventUnitPixel,
                1,
                int(round(delta)),
            )
            if hasattr(Quartz, "CGEventSetLocation"):
                Quartz.CGEventSetLocation(
                    event,
                    Quartz.CGPoint(
                        x=float(x if x is not None else 0),
                        y=float(y if y is not None else 0),
                    ),
                )
            Quartz.CGEventPost(Quartz.kCGHIDEventTap, event)
            return True
        except ImportError as exc:
            raise RuntimeError("Quartz scroll control requires macOS") from exc


class CursorOutput:
    """Live-only output loop that smoothly follows the latest target."""

    def __init__(
        self,
        controller: Controller,
        initial_position: tuple[float, float],
        rate_hz: float = 120.0,
        smoothing_time_constant: float = 0.08,
    ) -> None:
        if rate_hz <= 0 or smoothing_time_constant <= 0:
            raise ValueError("output rate and smoothing time constant must be positive")
        self.controller = controller
        self.rate_hz = rate_hz
        self.smoothing_time_constant = smoothing_time_constant
        self._target = initial_position
        self._actual = initial_position
        self._enabled = True
        self._dragging = False
        self._lock = Lock()
        self._stop = Event()
        self._thread: Optional[Thread] = None

    @property
    def actual_position(self) -> tuple[float, float]:
        with self._lock:
            return self._actual

    def set_target(self, position: tuple[float, float]) -> None:
        with self._lock:
            self._target = position

    def set_enabled(self, enabled: bool) -> None:
        with self._lock:
            self._enabled = enabled

    def set_dragging(self, dragging: bool) -> None:
        with self._lock:
            self._dragging = dragging

    def start(self) -> None:
        if not self.controller.live or self._thread is not None:
            return
        self._thread = Thread(target=self._run, name="saigai-cursor-output", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run(self) -> None:
        interval = 1.0 / self.rate_hz
        previous = monotonic()
        while not self._stop.is_set():
            now = monotonic()
            delta_time = max(now - previous, 0.0)
            previous = now
            with self._lock:
                alpha = 1.0 - exp(-delta_time / self.smoothing_time_constant)
                self._actual = (
                    self._actual[0] + (self._target[0] - self._actual[0]) * alpha,
                    self._actual[1] + (self._target[1] - self._actual[1]) * alpha,
                )
                position = self._actual
                enabled = self._enabled
                dragging = self._dragging
            if enabled:
                if dragging:
                    self.controller.mouse_dragged(*position)
                else:
                    self.controller.move_mouse(*position)
            self._stop.wait(interval)
