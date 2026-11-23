# saigai

A macOS hand-gesture controller for Apple Silicon.

## Setup

Requires Python 3.11 or 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py
```

If macOS reports that the camera is not authorized, open **System Settings >
Privacy & Security > Camera** and enable access for the app running the terminal
(for example, Terminal or Visual Studio Code). Fully quit and reopen that app,
then run `python main.py` again. Close apps that may already be using the camera.

The viewer does not control the cursor by default. Reserved live control can be
enabled later with `python main.py --live`.

Without `--live`, saigai draws a preview dot where the cursor would move. Pass
`--live` only after verifying the preview, since it enables real Quartz cursor
movement. Pinch and release performs a left click; holding the pinch while
moving performs a left-button drag. Clicking and dragging are only enabled with
`--live`.

With two fingers extended and the ring and pinky curled, vertical movement
scrolls instead of moving the cursor. Scroll events are also live-only.
