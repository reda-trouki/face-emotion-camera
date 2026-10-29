# Facial Expression Recognition — Webcam Demo

A small Python project that opens your webcam, estimates facial-expression categories, and overlays the top category and model scores on the live video.

## What it does

- Reads frames from your webcam using OpenCV.
- Uses DeepFace's pretrained emotion model to estimate categories such as happy, sad, angry, fear, surprise, disgust, and neutral.
- Draws a box around detected face regions and displays estimated scores.
- Processes every few frames to reduce CPU usage.
- Runs locally after dependencies and model weights have been installed.

> **Important:** These are uncertain model estimates of visible facial expressions, not reliable measurements of a person's true emotions, intent, personality, or mental state. Lighting, pose, camera quality, and individual differences can affect results. Use this as a learning/demo project, not for hiring, diagnosis, surveillance, or other consequential decisions. The script does not save or upload camera frames itself, but DeepFace may download model weights during its first run.

## Requirements

- Python 3.10 or 3.11 recommended for smoother compatibility with the ML dependencies.
- A webcam.
- Internet access on the first run may be needed to download pretrained model files.
- Windows, macOS, or Linux.

## Setup

### 1. Create and activate a virtual environment

**Windows (PowerShell):**
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

**macOS / Linux:**
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

DeepFace relies on machine-learning dependencies, so installation can take a few minutes.

### 3. Start the webcam app

```bash
python main.py
```

Press **Q** or **Esc** while the video window is focused to exit.

## Options

Use a different camera index if your computer has multiple cameras:

```bash
python main.py --camera 1
```

Analyze less often for better performance:

```bash
python main.py --analyze-every 8
```

Analyze a smaller frame for better performance (boxes are scaled back to the displayed image):

```bash
python main.py --scale 0.6
```

Combine options:

```bash
python main.py --camera 0 --analyze-every 6 --scale 0.7
```

## Troubleshooting

- **Camera will not open:** Close other apps using it, check OS camera permissions, and try `--camera 1`.
- **Slow first startup:** The model may be downloading or initializing. Let the first analysis finish.
- **TensorFlow / dependency installation errors:** Try Python 3.10 or 3.11 in a fresh virtual environment and reinstall the requirements.
- **No face detected:** Face the camera, improve lighting, and move into view.
- **Video window does not respond:** Click the video window and press Q or Esc.

## Project structure

```text
face_emotion_camera/
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

<!-- ## Possible next improvements

- Add a simple desktop UI with Start/Stop controls.
- Show a rolling chart of model scores.
- Add configurable camera selection.
- Add unit tests for the overlay and result-processing functions. -->
