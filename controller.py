"""Quartz mouse and scroll control with a live-action safety gate."""

from dataclasses import dataclass
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
