from pathlib import Path
from typing import Any, Iterable, Optional
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np


MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


def ensure_model() -> Path:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


class HandTracker:
    def __init__(self) -> None:
        self._tasks_detector: Optional[Any] = None
        self._legacy_detector: Optional[Any] = None
        self._using_tasks = hasattr(mp, "tasks")

        if self._using_tasks:
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_options = python.BaseOptions(model_asset_path=str(ensure_model()))
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=2,
                running_mode=vision.RunningMode.VIDEO,
            )
            self._tasks_detector = vision.HandLandmarker.create_from_options(options)
        else:
            self._legacy_detector = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def process(self, frame: np.ndarray, timestamp_ms: int) -> list[list[tuple[float, float, float]]]:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self._tasks_detector is not None:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._tasks_detector.detect_for_video(mp_image, timestamp_ms)
            return [
                [(lm.x, lm.y, lm.z) for lm in hand]
                for hand in result.hand_landmarks
            ]

        if self._legacy_detector is None:
            return []

        result = self._legacy_detector.process(rgb)
        if not result.multi_hand_landmarks:
            return []
        return [
            [(lm.x, lm.y, lm.z) for lm in hand.landmark]
            for hand in result.multi_hand_landmarks
        ]

    def close(self) -> None:
        if self._legacy_detector is not None:
            self._legacy_detector.close()
        if self._tasks_detector is not None:
            self._tasks_detector.close()


def draw_landmarks(
    frame: np.ndarray,
    hands: Iterable[list[tuple[float, float, float]]],
) -> None:
    height, width = frame.shape[:2]
    connections = HAND_CONNECTIONS

    for hand in hands:
        points = [(int(lm[0] * width), int(lm[1] * height)) for lm in hand]
        for start, end in connections:
            cv2.line(frame, points[start], points[end], (0, 255, 0), 2)
        for point in points:
            cv2.circle(frame, point, 4, (0, 0, 255), -1)
