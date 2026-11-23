import argparse
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from controller import Controller
from config import GESTURE_THRESHOLDS
from gestures import (
    OneEuroFilter,
    PinchStateMachine,
    ScrollStateMachine,
    get_pointer_anchor,
    map_index_finger_to_screen,
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
        f"FPS {fps:.1f}  latency {latency_ms:.1f} ms",
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
    freeze_frames = 0
    cursor_position = (screen_width / 2, screen_height / 2)
    previous_anchor: Optional[Tuple[float, float, float]] = None
    previous_pointer_time: Optional[float] = None
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
                        )
                        raw_position = clamp_pointer(
                            (cursor_position[0] + delta[0], cursor_position[1] + delta[1]),
                            screen_width,
                            screen_height,
                        )
                elif GESTURE_THRESHOLDS.pointer_mode == "absolute":
                    raw_position = map_index_finger_to_screen(
                        anchor,
                        frame.shape[1],
                        frame.shape[0],
                        screen_width,
                        screen_height,
                        GESTURE_THRESHOLDS.active_region_margin,
                    )
                else:
                    raise ValueError("pointer_mode must be 'absolute' or 'relative'")
                previous_anchor = anchor
                previous_pointer_time = timestamp_seconds

                if pinch_event.pressed:
                    freeze_frames = GESTURE_THRESHOLDS.pinch_freeze_frames
                    controller.mouse_down(*cursor_position)
                elif pinch_event.released:
                    controller.mouse_up(*cursor_position)

                if pinch.pressed:
                    if freeze_frames > 0:
                        freeze_frames -= 1
                    elif pinch_event.dragging:
                        cursor_x = x_filter(raw_position[0], timestamp_seconds)
                        cursor_y = y_filter(raw_position[1], timestamp_seconds)
                        cursor_position = (cursor_x, cursor_y)
                        controller.mouse_dragged(cursor_x, cursor_y)
                elif scroll_event.active:
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
                else:
                    cursor_x = x_filter(raw_position[0], timestamp_seconds)
                    cursor_y = y_filter(raw_position[1], timestamp_seconds)
                    cursor_position = (cursor_x, cursor_y)
                    controller.move_mouse(cursor_x, cursor_y)

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
                    controller.mouse_up(*cursor_position)
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
                draw_debug_overlay(
                    frame,
                    raw_position if hands else None,
                    cursor_position,
                    screen_width,
                    screen_height,
                    pinch_ratio if hands else None,
                    GESTURE_THRESHOLDS.pinch_press_threshold,
                    GESTURE_THRESHOLDS.pinch_release_threshold,
                    fps,
                    (time.perf_counter() - frame_started) * 1000.0,
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
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
