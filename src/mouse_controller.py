"""
OS-Level Mouse Controller
=========================

PURPOSE
-------
Once the CoordinateMapper tells us WHERE on screen the cursor should go,
and the GestureEngine tells us WHAT to do (move, click, drag, release),
this class actually executes those actions on the operating system using
the ``pyautogui`` library.

TWO KEY FEATURES
----------------
  1. SMOOTHING (moving average)
     Raw hand tracking is jittery.  We keep a history of the last N cursor
     positions (controlled by the "Smoothing Factor" slider) and return
     their average.  This smooths out small tremors.

  2. SENSITIVITY SCALING
     The "Mouse Sensitivity" slider controls how far the cursor moves
     relative to hand movement.  Sensitivity > 1.0 amplifies movement;
     sensitivity < 1.0 reduces it.  The scaling is centered on the screen
     so the cursor doesn't drift toward edges.
"""

from collections import deque

import pyautogui

# ------------------------------------------------------------------
# Disable pyautogui's built-in "fail-safe" that throws an exception
# when the mouse is at screen position (0,0).  We manage the cursor
# ourselves and don't want accidental corner touches to crash the app.
# ------------------------------------------------------------------
pyautogui.FAILSAFE = False


class MouseController:
    """Executes mouse actions (move, click, drag, release) on the OS.

    Handles smoothing (moving average), sensitivity scaling, and gesture-
    driven state transitions through the ``handle_action()`` method.
    """

    def __init__(self, sensitivity: float = 1.0, smooth_factor: int = 5):
        """
        Parameters
        ----------
        sensitivity : float
            Cursor speed multiplier (0.5 = half speed, 2.0 = double speed).
        smooth_factor : int
            Number of recent positions to average for smoothing.
            Higher values = smoother but more lag.
        """

        # ------------------------------------------------------------------
        # Sensitivity: 1.0 means "no scaling" (hand movement = cursor movement)
        # ------------------------------------------------------------------
        self.sensitivity = sensitivity

        # ------------------------------------------------------------------
        # Smoothing: a fixed-size deque (ring buffer) of recent positions.
        # When a new position arrives, we append it and the deque
        # automatically discards the oldest one if at maxlen.
        # The average is computed over the current contents.
        # ------------------------------------------------------------------
        self.smooth_factor = smooth_factor
        self._history: deque[tuple[int, int]] = deque(maxlen=smooth_factor)

        # State flag: is the mouse button currently held down?
        self.is_dragging = False

    # ------------------------------------------------------------------
    # Configuration updaters (called when the user moves GUI sliders).
    # ------------------------------------------------------------------

    def set_sensitivity(self, value: float):
        self.sensitivity = value

    def set_smooth_factor(self, value: int):
        """Reset the smoothing buffer when the user changes its size."""
        self.smooth_factor = value
        self._history = deque(maxlen=value)

    # ------------------------------------------------------------------
    # Smoothing logic
    # ------------------------------------------------------------------

    def _apply_smoothing(self, x: int, y: int) -> tuple[int, int]:
        """Append the new position and return the moving average.

        Step-by-step
        ------------
        1.  Add (x, y) to the history deque.
        2.  Compute the average of all stored positions:
            avg_x = sum of all x's / number of stored positions
            avg_y = sum of all y's / number of stored positions
        3.  Return the averaged coordinates.

        This is a simple moving average (SMA).  The ``smooth_factor``
        slider controls how many positions are averaged — a larger window
        gives smoother but more laggy cursor motion.
        """
        self._history.append((x, y))
        avg_x = int(sum(p[0] for p in self._history) / len(self._history))
        avg_y = int(sum(p[1] for p in self._history) / len(self._history))
        return avg_x, avg_y

    # ------------------------------------------------------------------
    # Core mouse operations
    # ------------------------------------------------------------------

    def move(self, x: int, y: int):
        """Move the cursor to (x, y) with sensitivity scaling and smoothing.

        Step-by-step
        ------------
        1.  If sensitivity != 1.0, scale the displacement from screen center.
            For example, at 2.0x sensitivity, a hand movement from center
            that would produce x=600 instead becomes:
                dx = (600 - 960) * 2.0 = -720
                x  = 960 + (-720) = 240
            The cursor moves TWICE as far as the hand movement.
        2.  Apply the moving-average smoother.
        3.  Send the final smoothed position to the OS via pyautogui.moveTo.
        """
        if self.sensitivity != 1.0:
            # Get screen dimensions
            center_w, center_h = pyautogui.size()
            cx, cy = center_w / 2, center_h / 2
            # Scale the offset from center
            dx = (x - cx) * self.sensitivity
            dy = (y - cy) * self.sensitivity
            x = int(cx + dx)
            y = int(cy + dy)
            # Clamp to valid screen bounds
            x = max(0, min(x, center_w - 1))
            y = max(0, min(y, center_h - 1))

        # Apply smoothing and move the OS cursor
        sx, sy = self._apply_smoothing(x, y)
        pyautogui.moveTo(sx, sy)

    def press(self):
        """Press and hold the left mouse button (drag start)."""
        pyautogui.mouseDown()
        self.is_dragging = True

    def release(self):
        """Release the left mouse button (drag end)."""
        pyautogui.mouseUp()
        self.is_dragging = False

    def click(self):
        """Perform a single click (press + release immediately)."""
        pyautogui.click()
        self.is_dragging = False

    # ------------------------------------------------------------------
    # Gesture-driven action dispatcher
    # ------------------------------------------------------------------

    def handle_action(self, action: str, x: int, y: int):
        """Route a gesture action to the appropriate OS command.

        This is the bridge between the GestureEngine and pyautogui:

            "MOVING"   → move cursor (release any held drag first)
            "CLICK"    → move cursor + press mouse button
            "DRAGGING" → move cursor (button is already held)
            "RELEASE"  → release mouse button
            "NONE"     → freeze cursor + release any held drag

        Parameters
        ----------
        action : str
            One of the five actions above.
        x, y : int
            Target screen coordinates for the cursor.
        """
        if action == "MOVING":
            # Safety: release if somehow stuck in a drag state
            if self.is_dragging:
                self.release()
            self.move(x, y)

        elif action == "CLICK":
            # Move to the pinch location first, then press
            self.move(x, y)
            self.press()

        elif action == "DRAGGING":
            # Keep moving while the button is held
            self.move(x, y)

        elif action == "RELEASE":
            # Release the mouse button if we're dragging
            if self.is_dragging:
                self.release()

        elif action == "NONE":
            # Hand left the frame — freeze cursor, release drag
            if self.is_dragging:
                self.release()

    def reset(self):
        """Emergency reset: clear position history and release any drag."""
        self._history.clear()
        if self.is_dragging:
            self.release()
