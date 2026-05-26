"""
CV TECHNIQUE 2 — Spatial Coordinate Interpolation & Frame Mapping
=================================================================

PURPOSE
-------
The webcam sees your hand at certain pixel coordinates (e.g., x=320, y=240).
Your monitor has a different resolution (e.g., 1920 × 1080).
This class bridges the two: it maps a hand position inside the camera frame
to the corresponding cursor position on your screen.

THE PROBLEM IT SOLVES
---------------------
Without mapping, you would have to move your hand across the entire camera
frame to reach every corner of the screen — exhausting and impractical.
Instead, we define a virtual "interaction zone" (a bounding box inset from
the frame edges).  Moving your hand inside this zone controls the full
screen.  You never need to reach the extreme edges of the camera view.

STEP-BY-STEP LOGIC
------------------
  1  Receive a normalized landmark coordinate (0..1) from MediaPipe.
  2  Convert it to pixel coordinates inside the camera frame.
  3  Clamp the pixel position to stay inside the interaction zone.
  4  Use linear interpolation (np.interp) to map the clamped camera-
     space position to the full screen resolution.
  5  Return the final (screen_x, screen_y) for the mouse controller.

RELATED LECTURES
----------------
  Lecture 3  — Image Manipulation (geometric transforms, affine mapping)
  Lecture 4  — Image Segmentation (defining a region of interest / ROI)

This is one of the three required CV techniques for the project.
"""

import numpy as np


class CoordinateMapper:
    """Maps normalized hand landmark coordinates to screen pixel coordinates.

    A virtual bounding box (inset from the frame edges by ``margin``) creates
    a comfortable "dead zone" so the user does not have to reach the extreme
    edges of the camera frame to reach the screen edges.
    """

    def __init__(
        self,
        screen_width: int,
        screen_height: int,
        frame_width: int,
        frame_height: int,
        margin: float = 0.15,
    ):
        """
        Parameters
        ----------
        screen_width, screen_height : int
            The monitor resolution (in pixels).
        frame_width, frame_height : int
            The camera frame resolution (in pixels).
        margin : float, optional
            Fraction of the frame to leave as a border (default 0.15 = 15%).
            A 15% margin on each side means the interaction zone is the
            central 70% of the frame width and height.
        """

        # ------------------------------------------------------------------
        # Store both coordinate systems — we need them for every mapping call.
        # ------------------------------------------------------------------
        self.screen_w = screen_width
        self.screen_h = screen_height
        self.frame_w = frame_width
        self.frame_h = frame_height
        self.margin = margin

        # ------------------------------------------------------------------
        # Precompute the interaction zone boundaries (in camera pixels).
        #   x_min, y_min  = top-left corner of the active zone
        #   x_max, y_max  = bottom-right corner of the active zone
        # e.g., for a 640×480 frame with 15% margin:
        #   x_min = 640 * 0.15 = 96
        #   y_min = 480 * 0.15 = 72
        #   x_max = 640 * 0.85 = 544
        #   y_max = 480 * 0.85 = 408
        # ------------------------------------------------------------------
        self.x_min = self.frame_w * self.margin
        self.x_max = self.frame_w * (1.0 - self.margin)
        self.y_min = self.frame_h * self.margin
        self.y_max = self.frame_h * (1.0 - self.margin)

    def map(self, norm_x: float, norm_y: float) -> tuple[int, int]:
        """Convert a MediaPipe landmark position to a screen cursor position.

        Step-by-step
        ------------
        1.  Convert normalized (0..1) → camera pixel coordinates.
        2.  Clamp the pixel coordinates to the interaction zone boundaries
            using np.clip.  This prevents the cursor from snapping to the
            screen corner if the hand briefly leaves the zone.
        3.  Use np.interp (linear interpolation / remapping) to transform
            the clamped camera-space position to the full screen resolution.
            For example, if the zone left-edge is x=96 and the right-edge is
            x=544, then x=96 maps to screen 0 and x=544 maps to screen 1920.
        4.  Return integer screen coordinates.

        Parameters
        ----------
        norm_x, norm_y : float
            Normalized landmark coordinates from MediaPipe in the range [0, 1].

        Returns
        -------
        tuple[int, int]
            (screen_x, screen_y) — the cursor position on the monitor.
        """

        # STEP 1: normalized → camera-frame pixel coordinates
        pixel_x = norm_x * self.frame_w
        pixel_y = norm_y * self.frame_h

        # STEP 2: clamp to the interaction zone
        clamped_x = float(np.clip(pixel_x, self.x_min, self.x_max))
        clamped_y = float(np.clip(pixel_y, self.y_min, self.y_max))

        # STEP 3: camera-space → screen-space via linear interpolation
        screen_x = int(
            np.interp(clamped_x, [self.x_min, self.x_max], [0, self.screen_w])
        )
        screen_y = int(
            np.interp(clamped_y, [self.y_min, self.y_max], [0, self.screen_h])
        )

        # STEP 4: return the final cursor position
        return screen_x, screen_y

    def get_bounding_box(self) -> tuple[int, int, int, int]:
        """Return the interaction zone rectangle for drawing on the frame.

        Returns
        -------
        tuple[int, int, int, int]
            (x0, y0, x1, y1) — the top-left and bottom-right corners
            of the yellow bounding box drawn on the video overlay.
        """
        x0 = int(self.x_min)
        y0 = int(self.y_min)
        x1 = int(self.x_max)
        y1 = int(self.y_max)
        return x0, y0, x1, y1
