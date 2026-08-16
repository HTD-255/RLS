"""
controllers — Steering-calibration controllers for AutoCar III.

Available controllers
---------------------
LRController        : Offline Linear Regression baseline  (notebook 8.1/8.2)
PIDController       : Classical PID (online, model-free)
RLSController       : Recursive Least Squares with forgetting factor ★
MPCController       : Lightweight Model Predictive Control
OpenLoopController  : Fixed steering command, no feedback (mechanical-bias baseline)

All controllers inherit from ``BaseController`` and expose the same
``update() → float`` interface so that experiment scripts can swap them
transparently.
"""

from src.controllers.base_controller import BaseController
from src.controllers.lr_controller import LRController
from src.controllers.pid_controller import PIDController
from src.controllers.rls_controller import RLSController
from src.controllers.mpc_controller import MPCController
from src.controllers.open_loop_controller import OpenLoopController

__all__ = [
    "BaseController",
    "LRController",
    "PIDController",
    "RLSController",
    "MPCController",
    "OpenLoopController",
]

