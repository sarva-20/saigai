"""Pure hand-gesture logic. This module must not access the OS or camera."""

from math import dist
from typing import Sequence


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
