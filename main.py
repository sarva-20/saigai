import argparse
import json
import time
from collections import deque
from pathlib import Path
from statistics import median
from typing import Optional, Tuple

import cv2
import numpy as np

from controller import Controller, CursorOutput
from config import GESTURE_THRESHOLDS
from gestures import (
    OneEuroFilter,
    HandInteractionStateMachine,
    InteractionEvent,
    ModeStateMachine,
    PoseDebouncer,
    classify_hand_pose,
    PointerHistory,
    RatioMedianFilter,
    calibrated_pinch_thresholds,
    ScrollStateMachine,
    StillnessLock,
    finger_extension_states,
    get_pointer_anchor,
    map_pointer_to_screen,
    normalized_pinch_distance,
    normalized_thumb_middle_distance,
    relative_pointer_delta,
)
from tracker import HandTracker, draw_landmarks


def read_latest_frame(camera: cv2.VideoCapture) -> Tuple[bool, Optional[np.ndarray]]:
    """Use a one-frame capture buffer so inference never processes stale frames."""
    if not camera.grab():
        return False, None
    return camera.retrieve()


def clamp_pointer(
    position: Tuple[float, float],
    screen_width: float,
    screen_height: float,
) -> Tuple[float, float]:
    return (
        max(0.0, min(screen_width - 1.0, position[0])),
        max(0.0, min(screen_height - 1.0, position[1])),
    )


class EventLog:
    def __init__(self, limit: int = 5) -> None:
        self.limit = limit
        self.entries: deque[tuple[float, str]] = deque(maxlen=limit)

    def add(self, timestamp: float, label: str) -> None:
        self.entries.append((timestamp, label))

    def draw(self, frame: np.ndarray) -> None:
        base_y = frame.shape[0] - 20
        for index, (timestamp, label) in enumerate(reversed(self.entries)):
            cv2.putText(
                frame,
                f"{timestamp:.1f}s  {label}",
                (10, base_y - index * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (220, 220, 220),
                1,
            )


def load_calibration(path: Path) -> Optional[Tuple[float, float, float, float]]:
    if not path.exists():
        return None
    with path.open() as calibration_file:
        values = json.load(calibration_file)
    return tuple(values[key] for key in ("left", "top", "right", "bottom"))


def load_pinch_calibration(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    with path.open() as calibration_file:
        values = json.load(calibration_file)
    return {
        key: float(value)
        for key, value in values.items()
        if key in (
            "left_press",
            "left_release",
            "right_press",
            "right_release",
        )
    }


def calibrate(
    camera: cv2.VideoCapture,
    tracker: HandTracker,
    path: Path,
) -> None:
    labels = ("TOP-LEFT", "TOP-RIGHT", "BOTTOM-RIGHT", "BOTTOM-LEFT")
    samples: list[Tuple[float, float]] = []
    for label in labels:
        start = time.perf_counter()
        while time.perf_counter() - start < GESTURE_THRESHOLDS.calibration_countdown_seconds:
            success, frame = read_latest_frame(camera)
            if not success or frame is None:
                continue
            # The camera image is mirrored exactly once here; mapping does not mirror it again.
            frame = cv2.flip(frame, 1)
            remaining = GESTURE_THRESHOLDS.calibration_countdown_seconds - (
                time.perf_counter() - start
            )
            cv2.putText(
                frame,
                f"Point {label} - {max(1, int(remaining) + 1)}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )
            cv2.imshow("saigai calibration", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

        collected: list[Tuple[float, float]] = []
        start = time.perf_counter()
        while time.perf_counter() - start < GESTURE_THRESHOLDS.calibration_sample_seconds:
            success, frame = read_latest_frame(camera)
            if not success or frame is None:
                continue
            frame = cv2.flip(frame, 1)
            hands = tracker.process(frame, int(time.perf_counter() * 1000))
            draw_landmarks(frame, hands)
            if hands:
                anchor = get_pointer_anchor(
                    hands[0],
                    GESTURE_THRESHOLDS.pointer_anchor,
                    GESTURE_THRESHOLDS.palm_anchor_mcp_weight,
                )
                collected.append((anchor[0], anchor[1]))
            cv2.putText(
                frame,
                f"Hold {label}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("saigai calibration", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt
        if not collected:
            raise RuntimeError(f"No hand landmarks collected for {label}")
        samples.append(
            (
                sum(point[0] for point in collected) / len(collected),
                sum(point[1] for point in collected) / len(collected),
            )
        )

    left = (samples[0][0] + samples[3][0]) / 2
    top = (samples[0][1] + samples[1][1]) / 2
    right = (samples[1][0] + samples[2][0]) / 2
    bottom = (samples[2][1] + samples[3][1]) / 2
    with path.open("w") as calibration_file:
        json.dump({"left": left, "top": top, "right": right, "bottom": bottom}, calibration_file)


def calibrate_pinch(camera: cv2.VideoCapture, tracker: HandTracker, path: Path) -> None:
    poses = (
        "Hold hand open (relaxed)",
        "Pinch and hold",
        "Hold hand open (relaxed)",
        "Pinch and hold",
    )
    left_samples: dict[str, list[float]] = {"open": [], "pinched": []}
    right_samples: dict[str, list[float]] = {"open": [], "pinched": []}

    for index, prompt in enumerate(poses):
        pose = "pinched" if "Pinch" in prompt else "open"
        start = time.perf_counter()
        while time.perf_counter() - start < 2.0:
            success, frame = read_latest_frame(camera)
            if not success or frame is None:
                continue
            frame = cv2.flip(frame, 1)
            hands = tracker.process(frame, int(time.perf_counter() * 1000))
            draw_landmarks(frame, hands)
            if hands:
                left_samples[pose].append(normalized_pinch_distance(hands[0]))
                right_samples[pose].append(normalized_thumb_middle_distance(hands[0]))
            cv2.putText(
                frame,
                prompt,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 255),
                2,
            )
            cv2.imshow("saigai pinch calibration", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

    left_open = median(left_samples["open"])
    left_pinched = median(left_samples["pinched"])
    right_open = median(right_samples["open"])
    right_pinched = median(right_samples["pinched"])
    left_press, left_release = calibrated_pinch_thresholds(left_open, left_pinched)
    right_press, right_release = calibrated_pinch_thresholds(right_open, right_pinched)

    values = {}
    if path.exists():
        with path.open() as calibration_file:
            values = json.load(calibration_file)
    values.update(
        {
            "left_press": left_press,
            "left_release": left_release,
            "right_press": right_press,
            "right_release": right_release,
        }
    )
    with path.open("w") as calibration_file:
        json.dump(values, calibration_file)


def draw_debug_overlay(
    frame: np.ndarray,
    raw_position: Optional[Tuple[float, float]],
    smoothed_position: Tuple[float, float],
    screen_width: float,
    screen_height: float,
    pinch_ratio: Optional[float],
    press_threshold: float,
    release_threshold: float,
    fps: float,
    latency_ms: float,
    output_rate_hz: float,
    scroll_event=None,
    pinch_history=None,
    pinch_events=None,
) -> None:
    height, width = frame.shape[:2]

    def preview_point(position: Tuple[float, float]) -> Tuple[int, int]:
        return (
            int(position[0] / screen_width * width),
            int(position[1] / screen_height * height),
        )

    if raw_position is not None:
        cv2.circle(frame, preview_point(raw_position), 7, (0, 255, 255), 2)
    cv2.circle(frame, preview_point(smoothed_position), 7, (255, 0, 255), 2)
    cv2.putText(
        frame,
        "raw target (yellow) / actual cursor (magenta)",
        (10, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    if pinch_ratio is not None:
        bar_x, bar_y, bar_width, bar_height = 10, 95, 250, 18
        scale = max(release_threshold * 1.5, pinch_ratio, 1.0)
        cv2.rectangle(
            frame,
            (bar_x, bar_y),
            (bar_x + bar_width, bar_y + bar_height),
            (180, 180, 180),
            1,
        )
        for threshold, color in (
            (press_threshold, (0, 0, 255)),
            (release_threshold, (0, 255, 255)),
        ):
            x = bar_x + int(min(threshold / scale, 1.0) * bar_width)
            cv2.line(frame, (x, bar_y), (x, bar_y + bar_height), color, 2)
        marker_x = bar_x + int(min(pinch_ratio / scale, 1.0) * bar_width)
        cv2.circle(
            frame,
            (marker_x, bar_y + bar_height // 2),
            5,
            (255, 255, 255),
            -1,
        )
        cv2.putText(
            frame,
            f"pinch ratio {pinch_ratio:.3f}",
            (10, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

    cv2.putText(
        frame,
        f"FPS {fps:.1f}  latency {latency_ms:.1f} ms  output {output_rate_hz:.0f} Hz",
        (10, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )

    if pinch_history:
        graph_x, graph_y, graph_width, graph_height = 260, 95, 300, 90
        cv2.rectangle(
            frame,
            (graph_x, graph_y),
            (graph_x + graph_width, graph_y + graph_height),
            (180, 180, 180),
            1,
        )
        scale = max(release_threshold * 1.5, 1.0)
        points = []
        values = list(pinch_history)
        for index, (_, ratio) in enumerate(values):
            x = graph_x + int(index * graph_width / max(len(values) - 1, 1))
            y = graph_y + graph_height - int(min(ratio / scale, 1.0) * graph_height)
            points.append((x, y))
        if len(points) > 1:
            cv2.polylines(frame, [points], False, (255, 255, 255), 2)
        for threshold, color in (
            (press_threshold, (0, 0, 255)),
            (release_threshold, (0, 255, 255)),
        ):
            y = graph_y + graph_height - int(min(threshold / scale, 1.0) * graph_height)
            cv2.line(frame, (graph_x, y), (graph_x + graph_width, y), color, 1)
        if pinch_events:
            for event_index, (timestamp, label, duration) in enumerate(pinch_events):
                cv2.putText(
                    frame,
                    f"{label} {duration:.0f} ms",
                    (
                        graph_x,
                        graph_y + graph_height + 14 + event_index * 14,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                )

    if scroll_event is not None:
        fingers = scroll_event.finger_states or {}
        cv2.putText(
            frame,
            "finger " + " ".join(
                f"{name[:1].upper()}={'T' if fingers.get(name, False) else 'C'}"
                for name in ("index", "middle", "ring", "pinky")
            ),
            (10, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            (
                f"scroll pose={scroll_event.active} "
                f"raw={scroll_event.raw_midpoint_delta:+.4f} "
                f"dead={scroll_event.deadzone_delta:+.4f}"
            ),
            (10, 238),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            f"scaled={scroll_event.scaled_delta:+.2f} sent={scroll_event.sent_amount}",
            (10, 261),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )


def run_scroll_selftest() -> None:
    controller = Controller(live=True)
    screen_width, screen_height = controller.screen_size()
    print("Posting scroll self-test events: down/up every 0.5 seconds")
    started = time.monotonic()
    direction = 1
    while time.monotonic() - started < 5.0:
        controller.scroll(direction * 120, screen_width / 2, screen_height / 2)
        print("scroll", "down" if direction > 0 else "up")
        time.sleep(0.5)
        direction *= -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="saigai hand gesture webcam viewer")
    parser.add_argument(
        "--live",
        action="store_true",
        help="enable OS cursor and scroll control",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="calibrate the active screen region and exit",
    )
    parser.add_argument(
        "--selftest-scroll",
        action="store_true",
        help="post alternating down/up scroll events for 5 seconds without a camera",
    )
    parser.add_argument(
        "--calibrate-pinch",
        action="store_true",
        help="calibrate left and right pinch thresholds",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.selftest_scroll:
        run_scroll_selftest()
        return
    controller = Controller(live=args.live)
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, GESTURE_THRESHOLDS.camera_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, GESTURE_THRESHOLDS.camera_height)
    camera.set(cv2.CAP_PROP_FPS, GESTURE_THRESHOLDS.camera_fps)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not camera.isOpened():
        camera.release()
        raise SystemExit(
            "Could not open the default webcam.\n"
            "On macOS, grant camera access in System Settings -> Privacy & Security -> Camera "
            "to Terminal or your IDE, then reopen it and run python main.py again."
        )

    tracker = HandTracker()
    screen_width, screen_height = controller.screen_size()
    calibration_path = Path(__file__).resolve().parent / GESTURE_THRESHOLDS.calibration_file
    if args.calibrate:
        try:
            calibrate(camera, tracker, calibration_path)
        finally:
            camera.release()
            tracker.close()
            cv2.destroyAllWindows()
        return
    if args.calibrate_pinch:
        try:
            calibrate_pinch(camera, tracker, calibration_path)
        finally:
            camera.release()
            tracker.close()
            cv2.destroyAllWindows()
        return
    active_region = load_calibration(calibration_path)
    pinch_calibration = load_pinch_calibration(calibration_path)
    x_filter = OneEuroFilter(
        min_cutoff=GESTURE_THRESHOLDS.one_euro_min_cutoff,
        beta=GESTURE_THRESHOLDS.one_euro_beta,
    )
    y_filter = OneEuroFilter(
        min_cutoff=GESTURE_THRESHOLDS.one_euro_min_cutoff,
        beta=GESTURE_THRESHOLDS.one_euro_beta,
    )
    pinch_filter = OneEuroFilter(
        min_cutoff=GESTURE_THRESHOLDS.pinch_min_cutoff,
        beta=GESTURE_THRESHOLDS.pinch_beta,
    )
    right_filter = OneEuroFilter(
        min_cutoff=GESTURE_THRESHOLDS.pinch_min_cutoff,
        beta=GESTURE_THRESHOLDS.pinch_beta,
    )
    left_ratio_median = RatioMedianFilter(3)
    right_ratio_median = RatioMedianFilter(3)
    interaction = HandInteractionStateMachine(
        left_press=pinch_calibration.get(
            "left_press", GESTURE_THRESHOLDS.pinch_press_threshold
        ),
        left_release=pinch_calibration.get(
            "left_release", GESTURE_THRESHOLDS.pinch_release_threshold
        ),
        right_press=pinch_calibration.get(
            "right_press", GESTURE_THRESHOLDS.right_press_threshold
        ),
        right_release=pinch_calibration.get(
            "right_release", GESTURE_THRESHOLDS.right_release_threshold
        ),
        drag_hold_ms=GESTURE_THRESHOLDS.drag_hold_ms,
        click_max_duration=GESTURE_THRESHOLDS.click_max_duration,
        click_flash_ms=GESTURE_THRESHOLDS.click_flash_ms,
        min_pinch_ms=GESTURE_THRESHOLDS.min_pinch_ms,
    )
    pose_debounce = PoseDebouncer(
        GESTURE_THRESHOLDS.pose_stable_frames,
        GESTURE_THRESHOLDS.pose_exit_frames,
    )
    mode_state = ModeStateMachine(
        GESTURE_THRESHOLDS.clutch_hold_ms,
        GESTURE_THRESHOLDS.double_click_ms,
    )
    scroll = ScrollStateMachine(
        sensitivity=GESTURE_THRESHOLDS.scroll_sensitivity,
        dead_zone=GESTURE_THRESHOLDS.scroll_dead_zone,
        invert=GESTURE_THRESHOLDS.scroll_invert,
        min_cutoff=GESTURE_THRESHOLDS.scroll_min_cutoff,
        beta=GESTURE_THRESHOLDS.scroll_beta,
        extension_threshold=GESTURE_THRESHOLDS.finger_extension_threshold,
    )
    stillness = StillnessLock(GESTURE_THRESHOLDS.stillness_threshold)
    history = PointerHistory(GESTURE_THRESHOLDS.pointer_history_size)
    event_log = EventLog()
    freeze_frames = 0
    cursor_position = (screen_width / 2, screen_height / 2)
    press_position = cursor_position
    previous_anchor: Optional[Tuple[float, float, float]] = None
    previous_pointer_time: Optional[float] = None
    output = CursorOutput(
        controller,
        cursor_position,
        GESTURE_THRESHOLDS.output_rate_hz,
        GESTURE_THRESHOLDS.output_smoothing_time_constant,
    )
    output.start()
    debug_overlay = False
    previous_time = time.perf_counter()
    fps = 0.0
    scroll_event = None
    pinch_ratio = None
    raw_position = None
    pinch_history: deque[tuple[float, float]] = deque(maxlen=90)
    pinch_events: deque[tuple[float, str, float]] = deque(maxlen=8)
    pinch_press_timestamp: Optional[float] = None

    try:
        while True:
            frame_started = time.perf_counter()
            success, frame = read_latest_frame(camera)
            if not success or frame is None:
                continue

            frame = cv2.flip(frame, 1)
            timestamp_ms = int(time.perf_counter() * 1000)
            hands = tracker.process(frame, timestamp_ms)
            draw_landmarks(frame, hands)

            if hands:
                anchor = get_pointer_anchor(
                    hands[0],
                    GESTURE_THRESHOLDS.pointer_anchor,
                    GESTURE_THRESHOLDS.palm_anchor_mcp_weight,
                )
                timestamp_seconds = timestamp_ms / 1000.0
                current_pose = classify_hand_pose(
                    hands[0],
                    GESTURE_THRESHOLDS.finger_extension_threshold,
                )
                active_pose = pose_debounce.update(current_pose)
                mode_event = mode_state.update_pose(active_pose, timestamp_seconds)
                if mode_event.drag_started:
                    output.set_dragging(True)
                    controller.mouse_down(*cursor_position)
                    event_log.add(timestamp_seconds, "DRAG START")
                if mode_event.drag_ended:
                    output.set_dragging(False)
                    controller.mouse_up(*cursor_position)
                    event_log.add(timestamp_seconds, "DRAG END")
                pinch_ratio = pinch_filter(
                    left_ratio_median.update(normalized_pinch_distance(hands[0])),
                    timestamp_seconds,
                )
                right_ratio = right_filter(
                    right_ratio_median.update(normalized_thumb_middle_distance(hands[0])),
                    timestamp_seconds,
                )
                pinch_history.append((timestamp_seconds, pinch_ratio))
                interaction_event = (
                    interaction.update(pinch_ratio, right_ratio, timestamp_seconds)
                    if mode_event.mode == "POINT"
                    else interaction.reset()
                )
                if interaction_event.left_pressed or interaction_event.right_pressed:
                    pinch_press_timestamp = timestamp_seconds
                    pinch_events.append(
                        (timestamp_seconds, "PRESS", 0.0)
                    )
                if interaction_event.left_released or interaction_event.right_clicked:
                    duration = (
                        (timestamp_seconds - pinch_press_timestamp) * 1000.0
                        if pinch_press_timestamp is not None
                        else 0.0
                    )
                    pinch_events.append(
                        (timestamp_seconds, "RELEASE", duration)
                    )
                    pinch_press_timestamp = None
                pinch_event = interaction_event
                scroll_event = scroll.update(
                    hands[0],
                    timestamp_seconds,
                    pinch_active=(mode_event.mode != "SCROLL"),
                ) if mode_event.mode == "SCROLL" else scroll.update(
                    None,
                    timestamp_seconds,
                    pinch_active=True,
                )

                if GESTURE_THRESHOLDS.pointer_mode == "relative":
                    if previous_anchor is None or previous_pointer_time is None:
                        raw_position = cursor_position
                    else:
                        delta = relative_pointer_delta(
                            previous_anchor,
                            anchor,
                            timestamp_seconds - previous_pointer_time,
                            screen_width,
                            screen_height,
                            GESTURE_THRESHOLDS.pointer_gain,
                            GESTURE_THRESHOLDS.pointer_acceleration,
                            GESTURE_THRESHOLDS.pointer_speed_reference,
                            GESTURE_THRESHOLDS.invert_x,
                            GESTURE_THRESHOLDS.invert_y,
                        )
                        raw_position = clamp_pointer(
                            (cursor_position[0] + delta[0], cursor_position[1] + delta[1]),
                            screen_width,
                            screen_height,
                        )
                elif GESTURE_THRESHOLDS.pointer_mode == "absolute":
                    raw_position = map_pointer_to_screen(
                        anchor,
                        frame.shape[1],
                        frame.shape[0],
                        screen_width,
                        screen_height,
                        GESTURE_THRESHOLDS.active_region_margin,
                        active_region,
                        GESTURE_THRESHOLDS.invert_x,
                        GESTURE_THRESHOLDS.invert_y,
                    )
                else:
                    raise ValueError("pointer_mode must be 'absolute' or 'relative'")
                if mode_event.mode == "THREE":
                    centroid = (
                        sum(hands[0][index][0] for index in (8, 12, 16)) / 3.0,
                        sum(hands[0][index][1] for index in (8, 12, 16)) / 3.0,
                        sum(hands[0][index][2] for index in (8, 12, 16)) / 3.0,
                    )
                    raw_position = map_pointer_to_screen(
                        centroid,
                        frame.shape[1],
                        frame.shape[0],
                        screen_width,
                        screen_height,
                        GESTURE_THRESHOLDS.active_region_margin,
                        active_region,
                        GESTURE_THRESHOLDS.invert_x,
                        GESTURE_THRESHOLDS.invert_y,
                    )
                previous_anchor = anchor
                previous_pointer_time = timestamp_seconds
                history.add(raw_position, timestamp_seconds)
                is_still = stillness.update(anchor)

                if interaction_event.left_pressed:
                    freeze_frames = GESTURE_THRESHOLDS.pinch_freeze_frames
                    press_position = history.latch(
                        timestamp_seconds,
                        GESTURE_THRESHOLDS.latch_ms,
                    ) or cursor_position
                    output.set_enabled(True)
                    output.set_dragging(False)
                    output.set_target(press_position)
                elif interaction_event.right_pressed:
                    press_position = history.latch(
                        timestamp_seconds,
                        GESTURE_THRESHOLDS.latch_ms,
                    ) or cursor_position
                elif interaction_event.left_released:
                    output.set_dragging(False)
                    if interaction_event.left_clicked:
                        event_log.add(timestamp_seconds, "CLICK")
                        if args.live:
                            controller.move_mouse(*press_position)
                            click_count = mode_state.left_click(timestamp_seconds)
                            controller.left_click(
                                *press_position,
                                click_count=click_count,
                            )
                            if click_count == 2:
                                event_log.add(timestamp_seconds, "DOUBLE CLICK")
                elif interaction_event.right_clicked:
                    event_log.add(timestamp_seconds, "RIGHT CLICK")
                    controller.right_click(*press_position)

                if mode_event.paused:
                    output.set_enabled(False)
                    output.set_dragging(False)
                    if not args.live:
                        cv2.putText(frame, "PAUSED", (10, 65),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif mode_event.mode == "THREE":
                    cursor_x = x_filter(raw_position[0], timestamp_seconds)
                    cursor_y = y_filter(raw_position[1], timestamp_seconds)
                    cursor_position = (cursor_x, cursor_y)
                    output.set_enabled(True)
                    output.set_dragging(True)
                    output.set_target(cursor_position)
                elif mode_event.mode == "SCROLL" and scroll_event.active:
                    output.set_enabled(False)
                    if scroll_event.sent_amount:
                        controller.scroll(
                            scroll_event.sent_amount,
                            cursor_position[0],
                            cursor_position[1],
                        )
                        event_log.add(timestamp_seconds, "SCROLL")
                    if not args.live:
                        cv2.putText(
                            frame,
                            f"SCROLL {scroll_event.delta:+.1f}",
                            (10, 65),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 0, 0),
                            2,
                        )
                elif mode_event.mode != "POINT":
                    pass
                elif freeze_frames > 0:
                    freeze_frames -= 1
                elif is_still:
                    output.set_enabled(True)
                else:
                    cursor_x = x_filter(raw_position[0], timestamp_seconds)
                    cursor_y = y_filter(raw_position[1], timestamp_seconds)
                    cursor_position = (cursor_x, cursor_y)
                    output.set_enabled(True)
                    output.set_target(cursor_position)

                if not args.live:
                    state = interaction_event.state
                    cv2.putText(
                        frame,
                        f"{active_pose or 'UNKNOWN'} / {mode_event.mode}",
                        (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.1,
                        (0, 255, 255),
                        2,
                    )
            else:
                previous_anchor = None
                previous_pointer_time = None
                interaction_event = interaction.reset()
                mode_event = mode_state.hand_lost()
                if mode_event.drag_ended:
                    output.set_dragging(False)
                    controller.mouse_up(*cursor_position)
                    event_log.add(timestamp_ms / 1000.0, "DRAG END")
                scroll_event = scroll.update(None, timestamp_ms / 1000.0)
                if interaction_event.drag_ended:
                    output.set_dragging(False)
                    controller.mouse_up(*press_position)
                    event_log.add(timestamp_ms / 1000.0, "DRAG END")
                if not args.live:
                    cv2.putText(
                        frame,
                        "NO HAND",
                        (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 165, 255),
                        2,
                    )

            current_time = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(current_time - previous_time, 1e-6))
            previous_time = current_time
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            if debug_overlay:
                actual_position = (
                    output.actual_position if args.live else cursor_position
                )
                draw_debug_overlay(
                    frame,
                    raw_position if hands else None,
                    actual_position,
                    screen_width,
                    screen_height,
                    pinch_ratio if hands else None,
                    GESTURE_THRESHOLDS.pinch_press_threshold,
                    GESTURE_THRESHOLDS.pinch_release_threshold,
                    fps,
                    (time.perf_counter() - frame_started) * 1000.0,
                    GESTURE_THRESHOLDS.output_rate_hz if args.live else 0.0,
                    scroll_event,
                    pinch_history,
                    pinch_events,
                )

            event_log.draw(frame)
            cv2.imshow("saigai", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                debug_overlay = not debug_overlay
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        tracker.close()
        output.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
