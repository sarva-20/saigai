import argparse
import json
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from controller import Controller, CursorOutput
from config import GESTURE_THRESHOLDS
from gestures import (
    OneEuroFilter,
    PointerHistory,
    PinchStateMachine,
    ScrollStateMachine,
    StillnessLock,
    get_pointer_anchor,
    map_pointer_to_screen,
    normalized_pinch_distance,
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


def load_calibration(path: Path) -> Optional[Tuple[float, float, float, float]]:
    if not path.exists():
        return None
    with path.open() as calibration_file:
        values = json.load(calibration_file)
    return tuple(values[key] for key in ("left", "top", "right", "bottom"))


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
    active_region = load_calibration(calibration_path)
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
    pinch = PinchStateMachine(
        press_threshold=GESTURE_THRESHOLDS.pinch_press_threshold,
        release_threshold=GESTURE_THRESHOLDS.pinch_release_threshold,
        click_max_duration=GESTURE_THRESHOLDS.click_max_duration,
    )
    scroll = ScrollStateMachine(
        sensitivity=GESTURE_THRESHOLDS.scroll_sensitivity,
        dead_zone=GESTURE_THRESHOLDS.scroll_dead_zone,
        invert=GESTURE_THRESHOLDS.scroll_invert,
        min_cutoff=GESTURE_THRESHOLDS.scroll_min_cutoff,
        beta=GESTURE_THRESHOLDS.scroll_beta,
    )
    stillness = StillnessLock(GESTURE_THRESHOLDS.stillness_threshold)
    history = PointerHistory(GESTURE_THRESHOLDS.pointer_history_size)
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
                pinch_ratio = pinch_filter(
                    normalized_pinch_distance(hands[0]),
                    timestamp_seconds,
                )
                pinch_event = pinch.update(
                    pinch_ratio,
                    timestamp_seconds,
                )
                scroll_event = scroll.update(
                    hands[0],
                    timestamp_seconds,
                    pinch_active=(
                        pinch.pressed
                        or pinch_event.pressed
                        or pinch_event.released
                    ),
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
                previous_anchor = anchor
                previous_pointer_time = timestamp_seconds
                history.add(raw_position, timestamp_seconds)
                is_still = stillness.update(anchor)

                if pinch_event.pressed:
                    freeze_frames = GESTURE_THRESHOLDS.pinch_freeze_frames
                    press_position = history.latch(
                        timestamp_seconds,
                        GESTURE_THRESHOLDS.latch_ms,
                    ) or cursor_position
                    output.set_enabled(True)
                    output.set_dragging(False)
                    output.set_target(press_position)
                    controller.mouse_down(*press_position)
                elif pinch_event.released:
                    output.set_dragging(False)
                    controller.mouse_up(*press_position)

                if pinch.pressed:
                    if freeze_frames > 0:
                        freeze_frames -= 1
                    elif pinch_event.dragging:
                        cursor_x = x_filter(raw_position[0], timestamp_seconds)
                        cursor_y = y_filter(raw_position[1], timestamp_seconds)
                        cursor_position = (cursor_x, cursor_y)
                        output.set_target(cursor_position)
                        output.set_dragging(True)
                elif scroll_event.active:
                    output.set_enabled(False)
                    if scroll_event.delta:
                        controller.scroll(scroll_event.delta)
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

                if not args.live and pinch.pressed:
                    state = "PINCH" if pinch.pressed else "READY"
                    cv2.putText(
                        frame,
                        state,
                        (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255) if pinch.pressed else (0, 255, 0),
                        2,
                    )
            elif pinch.pressed:
                previous_anchor = None
                previous_pointer_time = None
                pinch_event = pinch.update(float("inf"), timestamp_ms / 1000.0)
                scroll.update(None, timestamp_ms / 1000.0, pinch_active=True)
                if pinch_event.released:
                    output.set_dragging(False)
                    controller.mouse_up(*press_position)
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
            else:
                previous_anchor = None
                previous_pointer_time = None
                scroll_event = scroll.update(None, timestamp_ms / 1000.0)
                if not args.live and scroll_event.ended:
                    cv2.putText(
                        frame,
                        "READY",
                        (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
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
                )

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
