import argparse
import time

import cv2

from controller import Controller
from config import GESTURE_THRESHOLDS
from gestures import (
    OneEuroFilter,
    PinchStateMachine,
    map_index_finger_to_screen,
    normalized_pinch_distance,
)
from tracker import HandTracker, draw_landmarks


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
    pinch = PinchStateMachine(
        press_threshold=GESTURE_THRESHOLDS.pinch_press_threshold,
        release_threshold=GESTURE_THRESHOLDS.pinch_release_threshold,
        click_max_duration=GESTURE_THRESHOLDS.click_max_duration,
    )
    freeze_frames = 0
    cursor_position = (screen_width / 2, screen_height / 2)
    previous_time = time.perf_counter()
    fps = 0.0

    try:
        while True:
            success, frame = camera.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            timestamp_ms = int(time.perf_counter() * 1000)
            hands = tracker.process(frame, timestamp_ms)
            draw_landmarks(frame, hands)

            if hands:
                index_finger = hands[0][8]
                target_x, target_y = map_index_finger_to_screen(
                    index_finger,
                    frame.shape[1],
                    frame.shape[0],
                    screen_width,
                    screen_height,
                    GESTURE_THRESHOLDS.active_region_margin,
                )
                timestamp_seconds = timestamp_ms / 1000.0
                pinch_event = pinch.update(
                    normalized_pinch_distance(hands[0]),
                    timestamp_seconds,
                )

                if pinch_event.pressed:
                    freeze_frames = GESTURE_THRESHOLDS.pinch_freeze_frames
                    controller.mouse_down(*cursor_position)
                elif pinch_event.released:
                    controller.mouse_up(*cursor_position)

                if freeze_frames > 0:
                    freeze_frames -= 1
                elif not pinch.pressed:
                    cursor_x = x_filter(target_x, timestamp_seconds)
                    cursor_y = y_filter(target_y, timestamp_seconds)
                    cursor_position = (cursor_x, cursor_y)
                    controller.move_mouse(cursor_x, cursor_y)
                elif pinch_event.dragging:
                    cursor_x = x_filter(target_x, timestamp_seconds)
                    cursor_y = y_filter(target_y, timestamp_seconds)
                    cursor_position = (cursor_x, cursor_y)
                    controller.mouse_dragged(cursor_x, cursor_y)

                if not args.live:
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
                pinch_event = pinch.update(float("inf"), timestamp_ms / 1000.0)
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

            cv2.imshow("saigai", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
