# Virtual Mouse & Screen Controller

**CMSC 191 — Computer Vision Final Project**

A contactless hand-tracking system that controls your computer's mouse cursor
using a webcam.  Built with OpenCV, MediaPipe Tasks API, and CustomTkinter.

---

## Quick Start

```powershell
pip install -r requirements.txt
python download_model.py
python main.py
```

---

## Project Structure

```
virtual-mouse/
├── main.py                  # Entry point — run this to launch the app
├── requirements.txt         # Python package dependencies
├── download_model.py        # Downloads the MediaPipe .task model
├── models/
│   └── hand_landmarker.task # Pre-trained hand landmark detection model
└── src/
    ├── __init__.py          # Package exports
    ├── app.py               # GUI dashboard (CustomTkinter)
    ├── hand_tracker.py      # CV Tech 1 — Deep learning hand detection
    ├── coordinate_mapper.py # CV Tech 2 — Coordinate interpolation
    ├── gesture_engine.py    # CV Tech 3 — Gesture thresholding
    └── mouse_controller.py  # OS mouse control (pyautogui)
```

---

## How to Use

1. **Start the app**
   ```
   python main.py
   ```

2. **Click "Start Camera"** — the webcam turns on.  Your hand skeleton
   appears on screen.  The status reads `CALIBRATING`.

3. **Flip the "Mouse Control" switch** — status changes to `ACTIVE_TRACKING`.
   Move your hand to control the cursor.

4. **Pinch** (thumb + index finger) to click.  Hold the pinch and move to
   drag.  Release to drop.

5. **Adjust the sliders** to tune responsiveness:

   | Slider | Range | Default | What it does |
   |--------|-------|---------|--------------|
   | Mouse Sensitivity | 0.5× – 3.0× | 1.0× | Cursor speed relative to hand movement |
   | Smoothing Factor | 1 – 15 | 5 | Number of frames averaged together (higher = smoother but more lag) |
   | Click Threshold | 0.01 – 0.12 | 0.035 | How close thumb and index must be to register a pinch (lower = tighter pinch needed) |

6. Click **"Pause"** to freeze the display.  Click **"Resume"** to continue.

7. Click **"Stop Camera"** to shut down.

---

## Three Computer Vision Techniques

This project integrates three distinct CV techniques from the course syllabus:

### 1. Deep Learning-Based Keypoint Detection (`hand_tracker.py`)

Uses the modern MediaPipe Tasks API (`HandLandmarker`) with a local `.task`
model file to detect 21 3D hand landmarks per frame.  This replaces the
deprecated `mp.solutions` API.

**Course connection:** Lectures 5 (Object Detection), 8 (ML in CV), 11 (CNN)

### 2. Spatial Coordinate Interpolation & Frame Mapping (`coordinate_mapper.py`)

Defines a virtual interaction zone (a bounding box inset 15% from each edge
of the camera frame).  Hand positions within this zone are linearly mapped
to the full screen resolution using `numpy.interp`.  Hand positions outside
the zone are clamped to the nearest edge.

**Course connection:** Lectures 3 (Image Manipulation), 4 (Image Segmentation)

### 3. Mathematical Feature Extraction & Gesture Thresholding (`gesture_engine.py`)

Computes the Euclidean distance between landmark 4 (thumb tip) and
landmark 8 (index fingertip) every frame.  If this distance falls below an
adjustable threshold, a pinch is detected.  A state machine tracks the
pinch across consecutive frames to produce events: `MOVING`, `CLICK`,
`DRAGGING`, `RELEASE`, or `NONE`.

**Course connection:** Lecture 9 (Motion Analysis & Object Tracking)

---

## Architecture

### Thread-Safe Two-Loop Design

Tkinter is not thread-safe, but CV processing is too slow for the main
thread.  The app uses two concurrent loops communicating through a
thread-safe queue:

```
┌─────────────────────────────────────────────────────────────────────┐
│  BACKGROUND THREAD (daemon)                                         │
│                                                                     │
│  while running:                                                     │
│    1. Read frame from camera                                        │
│    2. Run MediaPipe detection  ←── Technique 1                      │
│    3. Map coordinates          ←── Technique 2                      │
│    4. Detect gesture           ←── Technique 3                      │
│    5. Control OS mouse                                              │
│    6. Draw overlays (bounding box, text, FPS)                       │
│    7. Push annotated frame to queue                                 │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │ Queue (maxsize=2)
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│  MAIN THREAD (tkinter)                                              │
│                                                                     │
│  every 25ms (via root.after):                                       │
│    1. Pop latest frame from queue                                   │
│    2. Resize to fit canvas                                          │
│    3. Display on tkinter Canvas                                     │
│    4. Update status labels                                          │
└─────────────────────────────────────────────────────────────────────┘
```

### Application States

```
IDLE ──► CALIBRATING ──► ACTIVE_TRACKING
  ▲          │  ▲               │
  └──────────┘  └───────────────┘
            (Pause/Resume)
          │              │
          ▼              ▼
         PAUSED  ◄──  PAUSED
```

- **IDLE:** Camera off, no tracking.
- **CALIBRATING:** Camera on, landmarks visible, mouse control OFF.
- **ACTIVE_TRACKING:** Camera on, landmarks visible, mouse control ON.
- **PAUSED:** Camera on but display frozen, no mouse commands.

### Edge Cases

| Situation | Behaviour |
|-----------|-----------|
| Hand leaves the frame | Cursor freezes at last known position (no corner-snapping) |
| Pinch releases | Mouse button releases immediately |
| Window closes | Camera released, MediaPipe model closed, thread joined |
| Model file missing | Error message in status label (no crash) |
| Camera fails to open | Error message in status label (no crash) |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `customtkinter` | ≥ 5.2.0 | Modern themed GUI framework |
| `opencv-python` | ≥ 4.8.0 | Camera I/O, image processing, drawing |
| `mediapipe` | ≥ 0.10.9 | Hand landmark detection (Tasks API) |
| `numpy` | ≥ 1.24.0 | Array operations, interpolation |
| `pyautogui` | ≥ 0.9.54 | OS-level mouse control |
| `Pillow` | ≥ 10.0.0 | Image conversion for tkinter display |
| `requests` | ≥ 2.28.0 | Model download helper |

---

## Model File

The `hand_landmarker.task` file is a pre-trained deep learning model from the
[MediaPipe Model Zoo](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker).
It contains a palm detector CNN followed by a landmark regression CNN.
Run `download_model.py` to download it automatically.
