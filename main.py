import argparse
import time

import cv2

from controller import Controller
from config import GESTURE_THRESHOLDS
from gestures import OneEuroFilter, map_index_finger_to_screen
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
                cursor_x, cursor_y = map_index_finger_to_screen(
                    index_finger,
                    frame.shape[1],
                    frame.shape[0],
                    screen_width,
                    screen_height,
                    GESTURE_THRESHOLDS.active_region_margin,
                )
                timestamp_seconds = timestamp_ms / 1000.0
                cursor_x = x_filter(cursor_x, timestamp_seconds)
                cursor_y = y_filter(cursor_y, timestamp_seconds)

                if args.live:
                    controller.move_mouse(cursor_x, cursor_y)
                else:
                    preview_x = int(cursor_x / screen_width * frame.shape[1])
                    preview_y = int(cursor_y / screen_height * frame.shape[0])
                    cv2.circle(
                        frame,
                        (preview_x, preview_y),
                        8,
                        (255, 0, 255),
                        -1,
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
