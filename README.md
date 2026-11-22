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

The viewer does not control the cursor by default. Reserved live control can be
enabled later with `python main.py --live`.
