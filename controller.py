"""Quartz mouse and scroll control with a live-action safety gate."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Controller:
    live: bool = False

    def require_live(self) -> None:
        if not self.live:
            raise RuntimeError("OS control is disabled; pass --live to enable it")
