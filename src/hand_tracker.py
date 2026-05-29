"""
CV TECHNIQUE 1 — Deep Learning-Based Hand Keypoint Detection
=============================================================

PURPOSE
-------
This module handles all interaction with the webcam and the MediaPipe Tasks
API to detect a hand in the video feed and locate 21 specific landmarks
(keypoints) on it.  These landmarks are the foundation that the other two
CV techniques (coordinate mapping and gesture recognition) build upon.

WHY MEDIAPIPE TASKS API (NOT THE LEGACY mp.solutions)
------------------------------------------------------
MediaPipe's older API (``mp.solutions.hands``) was deprecated.  The modern
``mediapipe.tasks.python.vision.HandLandmarker`` is faster, more accurate,
and uses a separate ``.task`` model file (``hand_landmarker.task``) that is
loaded at runtime.  This is the officially recommended API as of MediaPipe
0.10.x and later.

THE 21 HAND LANDMARKS
---------------------
MediaPipe detects these points on each hand:

    WRIST:  0
    THUMB:  1 (CMC), 2 (MCP), 3 (IP),  4 (TIP)
    INDEX:  5 (MCP), 6 (PIP), 7 (DIP), 8 (TIP)
    MIDDLE: 9 (MCP), 10 (PIP), 11 (DIP), 12 (TIP)
    RING:  13 (MCP), 14 (PIP), 15 (DIP), 16 (TIP)
    PINKY: 17 (MCP), 18 (PIP), 19 (DIP), 20 (TIP)

    TIP = fingertip,  PIP/DIP = middle/bottom knuckles,  MCP = knuckle at palm

    We particularly care about:
      - Landmark 9  (middle finger MCP)  →  stable cursor control point
      - Landmark 4  (thumb tip)           →  gesture distance calculation
      - Landmark 8  (index tip)           →  gesture distance calculation

STEP-BY-STEP PIPELINE
---------------------
  1. Open the webcam via cv2.VideoCapture.
  2. Read a frame.
  3. Horizontally flip it (so the display acts like a mirror — intuitive).
  4. Convert from BGR (OpenCV's native format) to RGB (MediaPipe's format).
  5. Wrap the RGB data in a MediaPipe Image object.
  6. Call detector.detect() — this runs the deep learning model.
  7. If a hand is found, draw the skeleton overlay on the frame.
  8. Return the annotated frame + the list of 21 landmarks.

RELATED LECTURES
----------------
  Lecture 5  — Object Detection (generic object detection with classifiers)
  Lecture 8  — Machine Learning in Computer Vision (pre-trained models)
  Lecture 11 — Convolutional Neural Networks (CNN-based landmark regression)

This is one of the three required CV techniques for the project.
"""

from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ------------------------------------------------------------------
# Hand skeleton topology.
#
# Each pair (a, b) is a connection between two landmarks that defines
# the hand's kinematic structure.  We draw a line between each pair
# to produce the "skeleton" overlay on the video feed.
#
# The pairs are grouped by finger:
#   Thumb:   0→1→2→3→4
#   Index:   0→5→6→7→8
#   Middle:  0→9→10→11→12
#   Ring:    0→13→14→15→16
#   Pinky:   0→17→18→19→20
# All fingers connect at the wrist (landmark 0) and at their MCP joints
# (landmarks 5, 9, 13, 17) which roughly form the palm.
# ------------------------------------------------------------------
_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
]

# Colors for the overlay (BGR format — OpenCV uses BGR by default).
_LANDMARK_COLOR = (0, 255, 0)          # bright green for dots
_CONNECTION_COLOR = (255, 255, 255)    # white for the skeleton lines


def _draw_landmarks(frame: np.ndarray, landmarks: list) -> np.ndarray:
    """Draw the hand skeleton overlay on a frame (in-place).

    For each of the 21 landmarks, draw a small filled circle at its pixel
    position.  Then for each connection in the topology above, draw a line
    between the two landmarks.

    Parameters
    ----------
    frame : np.ndarray
        The BGR video frame to draw on (modified in-place).
    landmarks : list
        List of 21 NormalizedLandmark objects from MediaPipe.

    Returns
    -------
    np.ndarray
        The annotated frame (same object, modified in-place).
    """
    h, w, _ = frame.shape

    # Draw a green circle at each landmark's pixel position
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (cx, cy), 4, _LANDMARK_COLOR, -1)

    # Draw white lines connecting landmarks per the skeleton topology
    for a, b in _HAND_CONNECTIONS:
        p1 = (int(landmarks[a].x * w), int(landmarks[a].y * h))
        p2 = (int(landmarks[b].x * w), int(landmarks[b].y * h))
        cv2.line(frame, p1, p2, _CONNECTION_COLOR, 2)

    return frame


class HandTracker:
    """Manages the webcam (or video file) and the MediaPipe HandLandmarker detector.

    This class is the bridge between the physical camera and our CV
    pipeline.  It handles:
      - Opening and configuring the camera or a video file
      - Loading the pre-trained .task model via the Tasks API
      - Reading and flipping frames (live camera only)
      - Running the deep learning model on each frame
      - Drawing the detection results (landmarks + skeleton)
      - Cleanly releasing resources when done
    """

    def __init__(
        self,
        model_path: str,
        camera_id: int = 0,
        video_path: str | None = None,
    ):
        """
        Parameters
        ----------
        model_path : str
            File path to the ``hand_landmarker.task`` model asset.
        camera_id : int, optional
            The device index of the webcam (0 = default camera).
            Ignored when ``video_path`` is provided.
        video_path : str | None, optional
            Path to a video file to use as input instead of the webcam.
            Supported formats: anything OpenCV can decode (mp4, avi, mov, …).
        """
        self._model_path = model_path
        self._camera_id = camera_id
        self._video_path = video_path
        self._is_video = video_path is not None

        # These are created in start() — None before that.
        self._cap: cv2.VideoCapture | None = None
        self._detector: vision.HandLandmarker | None = None

        # Default frame dimensions (overwritten with actual camera values).
        self.frame_width = 640
        self.frame_height = 480

    def start(self) -> bool:
        """Open the camera and initialize the MediaPipe HandLandmarker.

        Step-by-step
        ------------
        1.  Open the camera device with cv2.VideoCapture.
        2.  If the camera fails to open, return False (the GUI will show
            an error message instead of crashing).
        3.  Request a 640×480 resolution.  The camera may not support this
            exactly, so we read back the actual resolution and store it.
        4.  Create a HandLandmarker from the Tasks API by:
            a) Pointing BaseOptions to the local .task model file.
            b) Configuring options (IMAGE mode, 1 hand, confidence thresholds).
            c) Calling create_from_options() to load the model into memory.
        5.  Return True to signal success.

        Returns
        -------
        bool
            True if the camera and model loaded successfully.
        """
        # STEP 1 & 2: open the camera or video file.
        # cv2.VideoCapture accepts both integer device IDs and file path strings.
        source = self._video_path if self._is_video else self._camera_id
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            self._cap = None
            return False

        # STEP 3: configure resolution.
        # For a live camera we request 640×480 and read back the actual value.
        # For a video file the resolution is fixed by the file itself.
        if not self._is_video:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        actual_w = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if actual_w > 0 and actual_h > 0:
            self.frame_width = int(actual_w)
            self.frame_height = int(actual_h)

        # STEP 4: load the deep learning model via the Tasks API
        # (This is the modern replacement for mp.solutions.hands)
        base_options = python.BaseOptions(model_asset_path=self._model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._detector = vision.HandLandmarker.create_from_options(options)

        return True

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        """Capture one frame from the webcam.

        The frame is horizontally flipped so the video acts like a mirror
        (more intuitive when controlling a cursor with your hand).

        Returns
        -------
        tuple[bool, np.ndarray | None]
            (True, frame) on success, (False, None) if the camera failed.
        """
        if self._cap is None:
            return False, None

        ret, frame = self._cap.read()
        if not ret:
            if self._is_video:
                # End of file — seek back to the first frame and loop.
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self._cap.read()
                if not ret:
                    return False, None
            else:
                return False, None

        # Mirror only live camera frames so the display is intuitive.
        # Video files are already oriented correctly (no flip needed).
        if not self._is_video:
            frame = cv2.flip(frame, 1)
        return True, frame

    def detect(self, frame: np.ndarray) -> tuple[np.ndarray, list | None]:
        """Run the MediaPipe deep learning model on a frame.

        Step-by-step
        ------------
        1.  Convert the frame from BGR to RGB (MediaPipe expects RGB).
        2.  Wrap the RGB data in an ``mp.Image`` object required by the API.
        3.  Call ``self._detector.detect(mp_image)`` — this runs the CNN-
            based palm detector + landmark regressor on the image.
        4.  Check the result for hand landmarks.
        5.  If landmarks are found, draw the skeleton overlay on the frame
            using our ``_draw_landmarks()`` helper.
        6.  Return the annotated frame and the landmark list.

        Parameters
        ----------
        frame : np.ndarray
            A BGR video frame from read_frame().

        Returns
        -------
        tuple[np.ndarray, list | None]
            (annotated_frame, landmarks_list) where landmarks_list is
            None if no hand was detected.
        """
        # STEP 1 & 2: BGR → RGB → MediaPipe Image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # STEP 3: run inference
        result = self._detector.detect(mp_image)

        # STEP 4 & 5: extract landmarks and draw
        landmarks = None
        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            landmarks = result.hand_landmarks[0]
            _draw_landmarks(frame, landmarks)

        return frame, landmarks

    def stop(self):
        """Release the camera and close the MediaPipe detector.

        This MUST be called when shutting down to avoid leaving the
        camera hardware locked and to free GPU / model memory.
        """
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._detector is not None:
            self._detector.close()
            self._detector = None

    @property
    def is_running(self) -> bool:
        """Check if the camera is currently open."""
        return self._cap is not None and self._cap.isOpened()
