"""Exports from the `src` package so other modules can do:

    from src import HandTracker, CoordinateMapper, GestureEngine, MouseController

Each class lives in its own module for clean separation of concerns.
"""

from src.coordinate_mapper import CoordinateMapper
from src.gesture_engine import GestureEngine
from src.mouse_controller import MouseController
from src.hand_tracker import HandTracker

__all__ = [
    "CoordinateMapper",
    "GestureEngine",
    "MouseController",
    "HandTracker",
]
