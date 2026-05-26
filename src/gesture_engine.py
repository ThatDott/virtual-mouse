"""
CV TECHNIQUE 3 — Mathematical Feature Extraction & Gesture Event Thresholding
==============================================================================

PURPOSE
-------
The HandTracker gives us 21 landmarks, but raw coordinates alone don't tell us
what the user is DOING with their hand.  This class extracts a meaningful
feature — the distance between the thumb tip (landmark 4) and the index
fingertip (landmark 8) — and converts that distance into action commands.

HOW PINCH DETECTION WORKS
-------------------------
When you bring your thumb and index finger close together, the distance
between landmarks 4 and 8 drops below a threshold.  This is detected as a
"pinch".  The engine tracks this across consecutive frames so it can
distinguish four states:

    State        Meaning                         Mouse Controller Action
    -----------  ------------------------------  ------------------------
    MOVING       Fingers apart, hand detected    Move cursor
    CLICK        Fingers just pinched            Mouse press (drag start)
    DRAGGING     Fingers still pinched           Move while holding
    RELEASE      Fingers just separated          Mouse release (drag end)
    NONE         No hand in frame                Freeze cursor + release

This is a classic temporal state machine — we remember what happened in the
previous frame to decide what to do in the current frame.

RELATED LECTURES
----------------
  Lecture 9 — Motion Analysis & Object Tracking
    (frame-by-frame feature extraction, state tracking across time)

This is one of the three required CV techniques for the project.
"""

import math


class GestureEngine:
    """Frame-by-frame gesture recognizer based on pinch distance.

    Extracts the Euclidean distance between thumb tip (landmark 4) and index
    fingertip (landmark 8), normalizes it by the frame width, and compares it
    against an adjustable threshold.  A state machine tracks whether the pinch
    started, is ongoing, or ended in this frame.
    """

    PINCH = "PINCH"
    NO_PINCH = "NO_PINCH"

    def __init__(self, threshold: float = 0.035):
        """
        Parameters
        ----------
        threshold : float, optional
            Normalized distance threshold for pinch detection.
            Default 0.035 (about 3.5% of the frame width).
            A smaller value means you must bring your fingers closer together.
            A larger value makes clicking easier but may cause false positives.
        """

        # ------------------------------------------------------------------
        # Configurable threshold — updated live when the user drags the
        # "Click Distance Threshold" slider in the GUI.
        # ------------------------------------------------------------------
        self.threshold = threshold

        # ------------------------------------------------------------------
        # REMEMBER: was the user pinching in the previous frame?
        # This is the key to the state machine.
        # If they WEREN'T pinching and NOW they ARE  →  CLICK event
        # If they WERE   pinching and STILL are      →  DRAGGING
        # If they WERE   pinching and NOW aren't     →  RELEASE event
        # If they WEREN'T pinching and STILL aren't  →  MOVING
        # ------------------------------------------------------------------
        self.prev_pinching = False

        # The most recent action determined by update().
        self.current_action = "MOVING"

    def set_threshold(self, value: float):
        """Update the pinch threshold (called when the GUI slider moves)."""
        self.threshold = value

    def update(self, landmarks: list, frame_w: int, frame_h: int) -> tuple[str, float]:
        """Process one frame of landmark data and return the detected action.

        Step-by-step
        ------------
        1.  If no hand is visible (landmarks is None), handle the edge case:
            - If we were dragging, force a RELEASE so the mouse doesn't
              get stuck in the pressed state.
            - Otherwise return NONE (cursor freezes in place).
        2.  Convert normalized landmark coordinates to pixel coordinates
            for landmarks 4 (thumb tip) and 8 (index fingertip).
        3.  Compute the Euclidean distance between them:
                distance = sqrt( (x₁ - x₂)² + (y₁ - y₂)² )
        4.  Normalize by the frame width so the threshold remains consistent
            regardless of camera resolution.
        5.  Compare against self.threshold → is the user pinching?
        6.  Consult the state machine (prev_pinching vs current pinch) to
            determine the action (CLICK / DRAGGING / RELEASE / MOVING).
        7.  Remember this frame's pinch state for the next call.

        Parameters
        ----------
        landmarks : list or None
            The 21 MediaPipe landmarks for the detected hand, or None if
            no hand was found.
        frame_w, frame_h : int
            Width and height of the camera frame (for pixel conversion).

        Returns
        -------
        tuple[str, float]
            (action, norm_distance) where:
            - action is one of "MOVING", "CLICK", "DRAGGING", "RELEASE", "NONE"
            - norm_distance is the thumb-index distance normalized by frame_w
        """

        # ------------------------------------------------------------------
        # Edge case: no hand visible in this frame.
        # ------------------------------------------------------------------
        if landmarks is None or len(landmarks) < 9:
            # If we were in the middle of a drag, release the mouse button.
            if self.prev_pinching:
                self.prev_pinching = False
                self.current_action = "RELEASE"
                return self.current_action, 0.0
            # Otherwise just report NONE — the mouse controller will freeze
            # the cursor at its last position.
            self.current_action = "NONE"
            return self.current_action, 0.0

        # ------------------------------------------------------------------
        # STEP 2: Convert landmarks 4 and 8 to pixel coordinates.
        #
        #   MediaPipe landmarks are normalized to [0, 1] where (0,0) is the
        #   top-left of the frame and (1,1) is the bottom-right.
        #   We multiply by the frame dimensions to get pixel positions.
        #
        #   Landmark 4  =  thumb tip  ("TIP")
        #   Landmark 8  =  index fingertip  ("IP")
        # ------------------------------------------------------------------
        x1, y1 = landmarks[4].x * frame_w, landmarks[4].y * frame_h  # thumb tip
        x2, y2 = landmarks[8].x * frame_w, landmarks[8].y * frame_h  # index tip

        # ------------------------------------------------------------------
        # STEP 3 & 4: Euclidean distance, then normalize by frame width.
        #
        #   Why normalize?  A pinch distance of 20 pixels means something
        #   different on a 640-wide frame vs a 1280-wide frame.  Dividing
        #   by frame_w makes the threshold work the same at any resolution.
        #
        #   Formula:  norm_distance = sqrt((x₁-x₂)² + (y₁-y₂)²) / frame_w
        # ------------------------------------------------------------------
        distance = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        norm_distance = distance / frame_w

        # ------------------------------------------------------------------
        # STEP 5: Is this distance small enough to count as a pinch?
        # ------------------------------------------------------------------
        is_pinching = norm_distance < self.threshold

        # ------------------------------------------------------------------
        # STEP 6: State machine — compare previous frame to this frame.
        #
        #   prev_pinching  |  is_pinching  |  Result
        #   ---------------|---------------|------------------
        #   False          |  False        |  "MOVING"   (no change)
        #   False          |  True         |  "CLICK"    (pinch JUST started)
        #   True           |  True         |  "DRAGGING" (pinch HELD)
        #   True           |  False        |  "RELEASE"  (pinch JUST ended)
        # ------------------------------------------------------------------
        if is_pinching:
            if not self.prev_pinching:
                # TRANSITION: not pinching → pinching
                self.prev_pinching = True
                self.current_action = "CLICK"
            else:
                # STILL pinching from last frame
                self.current_action = "DRAGGING"
        else:
            if self.prev_pinching:
                # TRANSITION: pinching → not pinching
                self.prev_pinching = False
                self.current_action = "RELEASE"
            else:
                # STILL not pinching
                self.current_action = "MOVING"

        # ------------------------------------------------------------------
        # STEP 7: Return the action and raw distance (for debugging / GUI).
        # ------------------------------------------------------------------
        return self.current_action, norm_distance
