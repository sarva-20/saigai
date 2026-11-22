# Copilot instructions

- Target macOS on Apple Silicon.
- Use Python 3.11 or 3.12 in a virtual environment with a pinned `requirements.txt`.
- Use `opencv-python`, `mediapipe`, and `pyobjc-framework-Quartz` for mouse and scroll events.
- Keep the project modular:
  - `tracker.py` for camera capture and hand landmarks.
  - `gestures.py` for pure gesture logic with no OS calls.
  - `controller.py` for Quartz mouse and scroll events.
  - `main.py` for the application entry point.
  - `config.py` for gesture thresholds.
- Keep gesture logic pure and unit-testable without a camera.
- Never move the real cursor or scroll unless `--live` is passed.
