"""
Application GUI — CustomTkinter Dashboard
==========================================

PURPOSE
-------
This is the main window of the Virtual Mouse & Screen Controller.  It ties
together all three CV techniques into one interactive application:

    Technique 1  (HandTracker)    — deep learning keypoint detection
    Technique 2  (CoordinateMapper) — spatial interpolation & frame mapping
    Technique 3  (GestureEngine)  — feature extraction & gesture thresholding
    + MouseController             — OS-level action execution

ARCHITECTURE: THREAD-SAFE TWO-LOOP DESIGN
------------------------------------------
Tkinter (and by extension CustomTkinter) is NOT thread-safe — all GUI
updates must happen on the main thread.  However, running CV processing
(reading frames, running the MediaPipe model, drawing overlays) on the
main thread would freeze the GUI.  Our solution uses TWO loops:

    1.  BACKGROUND THREAD (``_processing_loop``):
        - Reads frames from the camera.
        - Runs MediaPipe detection.
        - Maps coordinates (Technique 2).
        - Detects gestures (Technique 3).
        - Controls the OS mouse.
        - Puts the annotated frame into a thread-safe queue.

    2.  MAIN THREAD (``_poll_frame_queue`` via ``root.after``):
        - Periodically checks the queue for new frames.
        - Updates the tkinter Canvas with the latest frame.
        - Updates status labels.

The queue acts as a thread-safe buffer.  Its maxsize=2 ensures we never
store stale frames — old frames are dropped if the GUI falls behind.

APPLICATION STATE MACHINE
--------------------------
The app transitions between four states:

    IDLE ──(Start Camera)──▸ CALIBRATING ──(Mouse ON)──▸ ACTIVE_TRACKING
     ▲                        │   ▲                        │
     │                        │   │                        │
     └──(Stop Camera)─────────┘   └──(Mouse OFF)───────────┘
                               (Pause)                   (Pause)
                                  │                        │
                                  ▼                        ▼
                               PAUSED ◄──────────────── PAUSED
                                  │                        │
                               (Resume)                 (Resume)
                                  ▼                        ▼
                              CALIBRATING              ACTIVE_TRACKING

    IDLE             Camera off, no tracking.
    CALIBRATING      Camera on, landmarks visible, mouse control OFF.
                     Use this to verify lighting and hand visibility.
    ACTIVE_TRACKING  Camera on, landmarks visible, mouse control ON.
    PAUSED           Camera on, display frozen, no mouse commands.
"""

from __future__ import annotations

import os
import queue
import threading
import time
import tkinter as tk

import customtkinter as ctk
import cv2
import numpy as np
from PIL import Image, ImageTk

from src.coordinate_mapper import CoordinateMapper
from src.gesture_engine import GestureEngine
from src.hand_tracker import HandTracker
from src.mouse_controller import MouseController


class AppGui:
    """Main application window built with CustomTkinter.

    This class is responsible for:
      - Building the GUI layout (video canvas + control panel).
      - Managing the application state (IDLE / CALIBRATING / etc.).
      - Starting / stopping the camera and background processing thread.
      - Displaying the annotated video feed on the canvas.
      - Updating status labels with current action, FPS, and cursor position.
      - Cleaning up resources on exit.
    """

    # ------------------------------------------------------------------
    # Application state constants
    # Four distinct states per the assignment requirements.
    # ------------------------------------------------------------------
    IDLE = "IDLE"
    CALIBRATING = "CALIBRATING"
    ACTIVE_TRACKING = "ACTIVE_TRACKING"
    PAUSED = "PAUSED"

    # ------------------------------------------------------------------
    #  CONSTRUCTOR
    # ------------------------------------------------------------------

    def __init__(self):
        """Set up the window, state variables, and launch the GUI.

        Step-by-step
        ------------
        1.  Configure CustomTkinter appearance (dark theme, blue accent).
        2.  Create the root window with a minimum size and title.
        3.  Register the close handler (WM_DELETE_WINDOW protocol).
        4.  Initialize state variables and component references (all None).
        5.  Get the screen resolution (needed for coordinate mapping).
        6.  Build all the widgets via _build_ui().
        7.  Start the periodic queue polling loop.
        8.  Enter the main event loop — this blocks until the window closes.
        """
        # --- STEP 1: Appearance ---
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # --- STEP 2: Window ---
        self.root = ctk.CTk()
        self.root.title("Virtual Mouse & Screen Controller - CMSC 191")
        self.root.minsize(1100, 650)

        # --- STEP 3: Close handler ---
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # --- STEP 4: State variables ---
        self.state = self.IDLE
        self.mouse_enabled = False
        self._running = False                     # signals thread to stop
        self._thread: threading.Thread | None = None
        self._frame_queue: queue.Queue[dict] = queue.Queue(maxsize=2)
        self._photo: ImageTk.PhotoImage | None = None  # keep a reference!

        # --- STEP 5: Screen dimensions ---
        self._screen_w = self.root.winfo_screenwidth()
        self._screen_h = self.root.winfo_screenheight()

        # Component references (created when camera starts, cleared on stop)
        self.tracker: HandTracker | None = None
        self.mapper: CoordinateMapper | None = None
        self.engine: GestureEngine | None = None
        self.mouse: MouseController | None = None

        # --- STEP 6 & 7: Build GUI + start polling ---
        self._build_ui()
        self._poll_frame_queue()

        # --- STEP 8: Enter main loop ---
        self.root.mainloop()

    # ================================================================== #
    #  GUI CONSTRUCTION
    # ================================================================== #

    def _build_ui(self):
        """Create all widgets for the main window.

        Layout
        ------
        +---------------------------------------------------------------+
        |  Left Panel (expands)          |  Right Panel (320px fixed)   |
        |  +---------------------------+ |  +-------------------------+ |
        |  |  Video Canvas (tk.Canvas) | |  |  INPUT SOURCE           | |
        |  |  Displays annotated feed  | |  |  [Live Webcam | Test]   | |
        |  |  with landmarks, box,     | |  |  Camera ID: [0]         | |
        |  |  action text, FPS         | |  |  ---                    | |
        |  |                           | |  |  CONTROL SETTINGS       | |
        |  |                           | |  |  Sensitivity [====]     | |
        |  |                           | |  |  Smoothing   [====]     | |
        |  |                           | |  |  Threshold   [====]     | |
        |  |                           | |  |  ---                    | |
        |  |                           | |  |  CONTROL                | |
        |  |                           | |  |  [Start Camera]         | |
        |  |                           | |  |  [Pause]                | |
        |  |                           | |  |  Mouse Control [OFF]   | |
        |  |                           | |  |  ---                    | |
        |  |                           | |  |  STATUS                 | |
        |  |                           | |  |  State: IDLE            | |
        |  |                           | |  |  Action: --             | |
        |  |                           | |  |  FPS: --                | |
        |  |                           | |  |  Cursor: --             | |
        |  +---------------------------+ |  +-------------------------+ |
        +---------------------------------------------------------------+
        """
        # Main horizontal container
        main = ctk.CTkFrame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ============================================================== #
        #  LEFT PANEL — Video Canvas
        # ============================================================== #
        left = ctk.CTkFrame(main, corner_radius=10)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # We use a plain tk.Canvas (not CTkCanvas) because it's faster
        # for real-time video rendering with ImageTk.PhotoImage.
        self.canvas = tk.Canvas(left, bg="#1a1a2e", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # ============================================================== #
        #  RIGHT PANEL — Controls
        # ============================================================== #
        right = ctk.CTkScrollableFrame(main, width=320, corner_radius=10)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right.grid_columnconfigure(0, weight=1)

        row = 0  # grid row counter — incremented after each widget

        # --------------------------------------------------------------- #
        #  INPUT SOURCE SECTION
        # --------------------------------------------------------------- #
        ctk.CTkLabel(
            right, text="INPUT SOURCE",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).grid(row=row, column=0, sticky="w", pady=(0, 5))
        row += 1

        self.source_var = ctk.StringVar(value="Live Webcam")
        ctk.CTkOptionMenu(
            right,
            values=["Live Webcam", "Test Video"],
            variable=self.source_var,
            command=self._on_source_change,
        ).grid(row=row, column=0, sticky="ew", pady=(0, 8))
        row += 1

        ctk.CTkLabel(right, text="Camera ID:").grid(
            row=row, column=0, sticky="w"
        )
        row += 1
        self.cam_id_var = ctk.StringVar(value="0")
        ctk.CTkEntry(right, textvariable=self.cam_id_var, width=60).grid(
            row=row, column=0, sticky="w", pady=(0, 10)
        )
        row += 1

        self._add_separator(right, row)
        row += 1

        # --------------------------------------------------------------- #
        #  CONTROL SETTINGS SECTION
        #
        #  Three sliders (Sensitivity, Smoothing, Click Threshold) that
        #  update the corresponding properties on the MouseController
        #  and GestureEngine in real time.  The label next to each slider
        #  shows the current value.
        # --------------------------------------------------------------- #
        ctk.CTkLabel(
            right, text="CONTROL SETTINGS",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).grid(row=row, column=0, sticky="w", pady=(0, 5))
        row += 1

        # --- Mouse Sensitivity ---
        ctk.CTkLabel(right, text="Mouse Sensitivity").grid(
            row=row, column=0, sticky="w"
        )
        row += 1
        self.sensitivity_var = ctk.DoubleVar(value=1.0)
        ctk.CTkSlider(
            right, from_=0.5, to=3.0, variable=self.sensitivity_var,
            command=self._on_sensitivity_change, number_of_steps=25,
        ).grid(row=row, column=0, sticky="ew", pady=(0, 2))
        self.sensitivity_label = ctk.CTkLabel(right, text="1.00x")
        self.sensitivity_label.grid(row=row, column=0, sticky="e", padx=(0, 5))
        row += 1

        # --- Smoothing Factor ---
        ctk.CTkLabel(right, text="Smoothing Factor").grid(
            row=row, column=0, sticky="w"
        )
        row += 1
        self.smooth_var = ctk.DoubleVar(value=5.0)
        ctk.CTkSlider(
            right, from_=1.0, to=15.0, variable=self.smooth_var,
            command=self._on_smooth_change, number_of_steps=14,
        ).grid(row=row, column=0, sticky="ew", pady=(0, 2))
        self.smooth_label = ctk.CTkLabel(right, text="5")
        self.smooth_label.grid(row=row, column=0, sticky="e", padx=(0, 5))
        row += 1

        # --- Click Distance Threshold ---
        ctk.CTkLabel(right, text="Click Distance Threshold").grid(
            row=row, column=0, sticky="w"
        )
        row += 1
        self.threshold_var = ctk.DoubleVar(value=0.035)
        ctk.CTkSlider(
            right, from_=0.01, to=0.12, variable=self.threshold_var,
            command=self._on_threshold_change, number_of_steps=22,
        ).grid(row=row, column=0, sticky="ew", pady=(0, 2))
        self.threshold_label = ctk.CTkLabel(right, text="0.035")
        self.threshold_label.grid(row=row, column=0, sticky="e", padx=(0, 5))
        row += 1

        self._add_separator(right, row)
        row += 1

        # --------------------------------------------------------------- #
        #  CONTROL SECTION
        #
        #  Start/Stop camera, Pause/Resume, and Mouse Control toggle.
        # --------------------------------------------------------------- #
        ctk.CTkLabel(
            right, text="CONTROL",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).grid(row=row, column=0, sticky="w", pady=(0, 5))
        row += 1

        self.start_btn = ctk.CTkButton(
            right, text="Start Camera", command=self._toggle_camera,
        )
        self.start_btn.grid(row=row, column=0, sticky="ew", pady=(0, 5))
        row += 1

        self.pause_btn = ctk.CTkButton(
            right, text="Pause", command=self._toggle_pause,
            state=tk.DISABLED,  # disabled until camera starts
        )
        self.pause_btn.grid(row=row, column=0, sticky="ew", pady=(0, 5))
        row += 1

        self.mouse_switch = ctk.CTkSwitch(
            right, text="Mouse Control",
            command=self._toggle_mouse, onvalue=True, offvalue=False,
        )
        self.mouse_switch.grid(row=row, column=0, sticky="w", pady=(0, 10))
        row += 1

        self._add_separator(right, row)
        row += 1

        # --------------------------------------------------------------- #
        #  STATUS SECTION
        #
        #  Read-only labels that display the current state, action, FPS,
        #  and cursor position.  Updated every frame from the queue.
        # --------------------------------------------------------------- #
        ctk.CTkLabel(
            right, text="STATUS",
            font=ctk.CTkFont(size=13, weight="bold"),
        ).grid(row=row, column=0, sticky="w", pady=(0, 5))
        row += 1

        self.state_label = ctk.CTkLabel(
            right, text=f"State: {self.IDLE}"
        )
        self.state_label.grid(row=row, column=0, sticky="w", pady=2)
        row += 1

        self.action_label = ctk.CTkLabel(right, text="Action: --")
        self.action_label.grid(row=row, column=0, sticky="w", pady=2)
        row += 1

        self.fps_label = ctk.CTkLabel(right, text="FPS: --")
        self.fps_label.grid(row=row, column=0, sticky="w", pady=2)
        row += 1

        self.coord_label = ctk.CTkLabel(right, text="Cursor: --")
        self.coord_label.grid(row=row, column=0, sticky="w", pady=2)
        row += 1

    @staticmethod
    def _add_separator(parent, row):
        """Draw a thin horizontal line to visually separate sections."""
        sep = ctk.CTkFrame(parent, height=2, fg_color="#333333")
        sep.grid(row=row, column=0, sticky="ew", pady=6)

    # ================================================================== #
    #  WIDGET CALLBACKS
    #
    #  These are called when the user interacts with GUI controls.
    #  They update the component settings and/or the application state.
    # ================================================================== #

    def _on_sensitivity_change(self, value: float):
        """Slider callback: update the cursor speed multiplier."""
        val = round(value, 2)
        self.sensitivity_label.configure(text=f"{val:.2f}x")
        if self.mouse is not None:
            self.mouse.set_sensitivity(val)

    def _on_smooth_change(self, value: float):
        """Slider callback: update the moving-average window size."""
        val = int(round(value))
        self.smooth_label.configure(text=str(val))
        if self.mouse is not None:
            self.mouse.set_smooth_factor(val)

    def _on_threshold_change(self, value: float):
        """Slider callback: update the pinch detection sensitivity."""
        val = round(value, 3)
        self.threshold_label.configure(text=f"{val:.3f}")
        if self.engine is not None:
            self.engine.set_threshold(val)

    def _on_source_change(self, _choice: str):
        """Input source selector (placeholder for future video file support)."""
        pass

    # --------------------------------------------------------------- #
    #  State transitions
    # --------------------------------------------------------------- #

    def _toggle_camera(self):
        """Start or stop the camera based on current state."""
        if self.state == self.IDLE:
            self._start_camera()
        else:
            self._stop_camera()

    def _toggle_pause(self):
        """Toggle between PAUSED and the previous active state.

        When pausing: release any held mouse drag so the user doesn't
        accidentally keep dragging after unpausing.
        """
        if self.state == self.PAUSED:
            # Restore the appropriate active state based on mouse toggle
            if self.mouse_enabled and self.mouse_switch.get():
                self.state = self.ACTIVE_TRACKING
            else:
                self.state = self.CALIBRATING
            self.pause_btn.configure(text="Pause")
        elif self.state in (self.CALIBRATING, self.ACTIVE_TRACKING):
            self.state = self.PAUSED
            self.pause_btn.configure(text="Resume")
            if self.mouse is not None:
                self.mouse.reset()  # release any held drag
        self._update_status()

    def _toggle_mouse(self):
        """Enable or disable OS mouse control.

        Transitions between CALIBRATING and ACTIVE_TRACKING.
        When disabling, reset the mouse state to release any held drag.
        """
        self.mouse_enabled = self.mouse_switch.get()
        if self.mouse_enabled and self.state == self.CALIBRATING:
            self.state = self.ACTIVE_TRACKING
        elif not self.mouse_enabled and self.state == self.ACTIVE_TRACKING:
            self.state = self.CALIBRATING
        if not self.mouse_enabled and self.mouse is not None:
            self.mouse.reset()
        self._update_status()

    # ================================================================== #
    #  CAMERA LIFECYCLE
    #
    #  _start_camera() — opens the camera, creates all CV components,
    #                    and launches the background processing thread.
    #  _stop_camera()  — signals the thread to stop, releases resources.
    # ================================================================== #

    def _start_camera(self):
        """Open the camera, initialise all CV components, start processing.

        Step-by-step
        ------------
        1.  Locate the .task model file (relative to the project root).
        2.  Read the camera ID from the GUI entry field.
        3.  Create and start the HandTracker (opens camera, loads DL model).
        4.  Create the CoordinateMapper with the actual camera resolution.
        5.  Create the GestureEngine with the current threshold slider value.
        6.  Create the MouseController with current sensitivity/smoothing.
        7.  Set state to CALIBRATING (hand visible, mouse OFF by default).
        8.  Launch the background processing thread.
        9.  Update the GUI buttons to reflect the running state.
        """
        # STEP 1: Find the model file
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "models",
            "hand_landmarker.task",
        )
        if not os.path.isfile(model_path):
            self.state_label.configure(
                text="State: ERROR - model file not found"
            )
            return

        # STEP 2: Parse camera ID
        try:
            cam_id = int(self.cam_id_var.get().strip())
        except ValueError:
            cam_id = 0

        # STEP 3: Start the hand tracker (Technique 1)
        self.tracker = HandTracker(model_path=model_path, camera_id=cam_id)
        if not self.tracker.start():
            self.state_label.configure(
                text="State: ERROR - cannot open camera"
            )
            self.tracker = None
            return

        # STEP 4: Create the coordinate mapper (Technique 2)
        self.mapper = CoordinateMapper(
            screen_width=self._screen_w,
            screen_height=self._screen_h,
            frame_width=self.tracker.frame_width,
            frame_height=self.tracker.frame_height,
            margin=0.15,
        )

        # STEP 5: Create the gesture engine (Technique 3)
        self.engine = GestureEngine(threshold=self.threshold_var.get())

        # STEP 6: Create the mouse controller
        self.mouse = MouseController(
            sensitivity=self.sensitivity_var.get(),
            smooth_factor=int(round(self.smooth_var.get())),
        )

        # STEP 7: Set initial state
        self._running = True
        self.state = self.CALIBRATING
        self.mouse_enabled = False

        # STEP 8: Launch the background processing thread
        # Daemon=True means the thread will automatically exit when the
        # main program exits, even if we forget to join it.
        self._thread = threading.Thread(
            target=self._processing_loop, daemon=True
        )
        self._thread.start()

        # STEP 9: Update GUI state
        self.start_btn.configure(text="Stop Camera")
        self.pause_btn.configure(state=tk.NORMAL)
        self.mouse_switch.deselect()  # start with mouse control OFF
        self._update_status()

    def _stop_camera(self):
        """Stop the camera, terminate the processing thread, reset GUI.

        Step-by-step
        ------------
        1.  Set _running = False (the thread checks this in its loop).
        2.  Join the thread with a 2-second timeout (daemon kills it
            if it doesn't finish).
        3.  Stop the HandTracker (releases camera + closes DL model).
        4.  Reset the mouse (release any held drag).
        5.  Clear component references.
        6.  Reset state to IDLE.
        7.  Update GUI buttons and clear the canvas.
        """
        # STEP 1 & 2: Signal and wait for the thread to stop
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

        # STEP 3: Release camera and MediaPipe model
        if self.tracker is not None:
            self.tracker.stop()
            self.tracker = None

        # STEP 4: Emergency mouse reset
        if self.mouse is not None:
            self.mouse.reset()
            self.mouse = None

        # STEP 5: Clear remaining component references
        self.mapper = None
        self.engine = None

        # STEP 6 & 7: Reset state and GUI
        self.state = self.IDLE
        self.mouse_enabled = False

        self.start_btn.configure(text="Start Camera")
        self.pause_btn.configure(text="Pause", state=tk.DISABLED)
        self.mouse_switch.deselect()

        self.canvas.delete("all")
        self._photo = None
        self._update_status()

    # ================================================================== #
    #  BACKGROUND PROCESSING LOOP
    #
    #  THIS RUNS ON A SEPARATE THREAD — do NOT touch tkinter widgets here!
    #
    #  The loop:
    #    1.  Reads a frame from the camera.
    #    2.  Runs the MediaPipe hand detector (Technique 1).
    #    3.  Maps landmarks to screen coordinates (Technique 2).
    #    4.  Detects gesture actions (Technique 3).
    #    5.  Controls the OS mouse if active.
    #    6.  Draws overlays (bounding box, action text, FPS).
    #    7.  Pushes the annotated frame + metadata to the queue.
    # ================================================================== #

    def _processing_loop(self):
        """Continuously capture, process, and queue frames.

        This is the heart of the CV pipeline.  It runs until ``_running``
        is set to False by the main thread.

        QUEUE PROTOCOL
        --------------
        Each item pushed to the queue is a dict with keys:
            frame  : np.ndarray  — the annotated BGR frame
            action : str         — current gesture action
            fps    : float       - frames per second (updated once per sec)
            cursor : tuple       - (screen_x, screen_y) or (-1, -1)

        The queue has maxsize=2.  If the display loop is slow and the queue
        is full, ``put_nowait`` silently drops the old frame.  This ensures
        we never block the camera and always show the latest frame.
        """
        fps_count = 0
        fps_timer = time.perf_counter()
        fps_display = 0.0

        while self._running:
            # ----------------------------------------------------------- #
            # STEP 1: Capture a frame from the camera.
            # ----------------------------------------------------------- #
            ret, frame = self.tracker.read_frame()
            if not ret:
                time.sleep(0.01)
                continue

            # ----------------------------------------------------------- #
            # STEP 2: Run deep learning detection (Technique 1).
            #
            # If the state is PAUSED, skip detection to save CPU.
            # The display loop also skips updates, so the last
            # pre-pause frame stays visible.
            # ----------------------------------------------------------- #
            paused = self.state == self.PAUSED

            if not paused:
                frame, landmarks = self.tracker.detect(frame)
            else:
                landmarks = None

            # ----------------------------------------------------------- #
            # STEPS 3 & 4: Map coordinates + detect gesture.
            #
            # If no hand is visible (landmarks is None), we pass None to
            # the GestureEngine, which returns "NONE".  The mouse
            # controller will freeze the cursor and release any drag.
            # ----------------------------------------------------------- #
            action = "NONE"
            cursor_x = cursor_y = -1

            if landmarks and not paused:
                # Use landmark 9 (middle finger MCP — base knuckle) as the
                # cursor control point.  This point is more stable than the
                # fingertip because it moves less with finger articulation.
                cx, cy = landmarks[9].x, landmarks[9].y

                # Technique 2: map camera coordinates → screen coordinates
                cursor_x, cursor_y = self.mapper.map(cx, cy)

                # Technique 3: compute gesture state from thumb-index distance
                action, _ = self.engine.update(
                    landmarks,
                    self.tracker.frame_width,
                    self.tracker.frame_height,
                )

                # ------------------------------------------------------- #
                # STEP 5: Control the OS mouse (if active).
                #
                # Mouse control is only enabled when:
                #   - state == ACTIVE_TRACKING
                #   - the user has toggled mouse_switch ON
                # ------------------------------------------------------- #
                if (
                    self.state == self.ACTIVE_TRACKING
                    and self.mouse_enabled
                    and self.mouse is not None
                ):
                    self.mouse.handle_action(action, cursor_x, cursor_y)

            elif not paused:
                # Hand left the frame — notify the gesture engine and
                # freeze the cursor by sending "NONE" to the mouse.
                action, _ = self.engine.update(
                    None,
                    self.tracker.frame_width,
                    self.tracker.frame_height,
                )
                if self.mouse is not None:
                    self.mouse.handle_action("NONE", 0, 0)

            # ----------------------------------------------------------- #
            # STEP 6: Draw overlays on the frame.
            #
            # These are drawn with OpenCV directly on the frame array
            # so the visual feedback appears in the canvas display.
            # ----------------------------------------------------------- #

            # Draw the yellow interaction bounding box (Technique 2 visualisation)
            if not paused and self.mapper is not None:
                x0, y0, x1, y1 = self.mapper.get_bounding_box()
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 2)

            # Draw the action text overlay
            if not paused:
                cv2.putText(
                    frame, f"Action: {action}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2,
                )
            else:
                cv2.putText(
                    frame, "PAUSED", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2,
                )

            # FPS counter (updated once per second)
            fps_count += 1
            now = time.perf_counter()
            elapsed = now - fps_timer
            if elapsed >= 1.0:
                fps_display = fps_count / elapsed
                fps_count = 0
                fps_timer = now

            cv2.putText(
                frame, f"FPS: {fps_display:.1f}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2,
            )

            # ----------------------------------------------------------- #
            # STEP 7: Push the result to the thread-safe queue.
            # ----------------------------------------------------------- #
            info = {
                "frame": frame,           # annotated BGR frame (np.ndarray)
                "action": action,         # gesture action string
                "fps": fps_display,       # current FPS
                "cursor": (cursor_x, cursor_y),  # screen coordinates
            }

            try:
                self._frame_queue.put_nowait(info)
            except queue.Full:
                # Queue is full — the display loop is lagging behind.
                # Drop this frame; the next frame will overwrite it.
                pass

    # ================================================================== #
    #  GUI DISPLAY LOOP
    #
    #  THIS RUNS ON THE MAIN THREAD via root.after().
    #
    #  Periodically polls the frame queue and updates the tkinter Canvas
    #  with the latest annotated frame.  Also updates the status labels.
    #
    #  This is the ONLY code that touches tkinter widgets.
    # ================================================================== #

    def _poll_frame_queue(self):
        """Check the queue for new frames and display them.

        Called every ~25ms via ``root.after(25, ...)``.  This is NOT a
        while loop — it's a recurring timer callback that tkinter runs
        whenever it has idle time on the main thread.

        We drain the queue completely (while True + get_nowait) so that
        if multiple frames arrived since the last poll, we only display
        the most recent one (the last one popped).  This keeps the display
        responsive and low-latency.
        """
        try:
            while True:
                info = self._frame_queue.get_nowait()
                # Don't update the display while paused — show the
                # last pre-pause frame plus the "PAUSED" overlay.
                if self.state != self.PAUSED:
                    self._display_frame(info)
        except queue.Empty:
            pass
        self.root.after(25, self._poll_frame_queue)

    def _display_frame(self, info: dict):
        """Render one annotated frame onto the Canvas.

        Step-by-step
        ------------
        1.  Extract the frame, action, FPS, and cursor from the info dict.
        2.  Compute the canvas dimensions (minimum 100px to avoid div-by-0).
        3.  Compute a scale factor to fit the frame inside the canvas.
        4.  Resize the frame to the computed display size.
        5.  Convert BGR → RGB (PIL expects RGB).
        6.  Create a PIL Image → ImageTk.PhotoImage for tkinter.
        7.  Clear the canvas and draw the new image centered.
        8.  Update the action, FPS, and cursor status labels.
        """
        # --- STEP 1: Unpack ---
        frame: np.ndarray = info["frame"]
        action: str = info["action"]
        fps: float = info["fps"]
        cursor: tuple[int, int] = info["cursor"]

        # --- STEP 2: Canvas size ---
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)

        # --- STEPS 3 & 4: Scale frame to fit canvas ---
        h, w = frame.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = int(w * scale), int(h * scale)
        disp = cv2.resize(frame, (nw, nh))

        # --- STEPS 5 & 6: Convert to PIL PhotoImage ---
        rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        self._photo = ImageTk.PhotoImage(img)

        # --- STEP 7: Draw centered on canvas ---
        self.canvas.delete("all")
        self.canvas.create_image(
            cw // 2, ch // 2, image=self._photo, anchor=tk.CENTER
        )

        # --- STEP 8: Update status labels ---
        self.action_label.configure(text=f"Action: {action}")
        self.fps_label.configure(text=f"FPS: {fps:.1f}")
        if cursor[0] >= 0:
            self.coord_label.configure(
                text=f"Cursor: ({cursor[0]}, {cursor[1]})"
            )

    def _update_status(self):
        """Refresh the state label (called after every state change)."""
        self.state_label.configure(text=f"State: {self.state}")

    # ================================================================== #
    #  CLEANUP
    #
    #  Called when the user closes the window (WM_DELETE_WINDOW).
    #  Ensures the camera, MediaPipe model, and background thread are
    #  all properly released before Python exits.
    # ================================================================== #

    def _on_close(self):
        """Clean shutdown: stop thread, release camera, destroy window.

        Step-by-step
        ------------
        1.  Signal the processing thread to stop (_running = False).
        2.  Join the thread (wait up to 2 seconds for it to finish).
        3.  Stop the HandTracker (releases camera + closes DL model).
        4.  Quit and destroy the tkinter window.
        """
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self.tracker is not None:
            self.tracker.stop()
        self.root.quit()
        self.root.destroy()
