"""
Defines the global state for the CBAS application.

This module serves as a centralized, shared memory space for all other backend
modules. It holds references to the current project, labeling session details,
background thread handles, and other application-wide variables. This avoids
the need for scattered global variables and makes state management more predictable.
"""

import queue
import threading
from typing import Union, List, Dict, Any, TYPE_CHECKING
import pandas as pd
import cv2
import subprocess

# Use TYPE_CHECKING block to import types only for static analysis (e.g., by linters
# or IDEs), preventing runtime circular import errors with the workthreads module.
if TYPE_CHECKING:
    import cbas
    from workthreads import TrainingThread, EncodeThread, ClassificationThread
    from watchdog.observers import Observer
    from cmap import Colormap


# =================================================================
# GLOBAL PROJECT STATE
# =================================================================

# By enclosing the type names in quotes, we create "forward references".
# This tells Python's typing system what the type *will be* at runtime,
# without needing to import the module directly, thus breaking the circular dependency.
proj: Union['cbas.Project', None] = None
"""The currently loaded cbas.Project instance. None if no project is open."""


# =================================================================
# LABELING INTERFACE STATE
# =================================================================

label_capture: Union[cv2.VideoCapture, None] = None
label_dataset: Union['cbas.Dataset', None] = None # <-- Fix: Add quotes
label_videos: List[str] = []
label_vid_index: int = -1
label_index: int = -1
label_start: int = -1
label_type: int = -1
label_session_buffer: List[Dict[str, Any]] = []
label_probability_df: Union[pd.DataFrame, None] = None
label_history: List[Dict[str, Any]] = []
selected_instance_index: int = -1
label_behavior_colors: List[str] = []
label_confirmation_mode: bool = False
label_confidence_threshold: int = 100
label_unfiltered_instances: List[Dict[str, Any]] = []
label_col_map: Union['Colormap', None] = None # <-- Fix: Add quotes


# =================================================================
# BACKGROUND THREADS & TASK QUEUES
# =================================================================

# --- Encoding Thread ---
encode_thread: Union['EncodeThread', None] = None # <-- Fix: Add quotes
encode_lock: Union[threading.Lock, None] = None
encode_tasks: List[str] = []

# --- Classification (Inference) Thread ---
classify_thread: Union['ClassificationThread', None] = None # <-- Fix: Add quotes
classify_lock: Union[threading.Lock, None] = None
classify_tasks: List[str] = []

# --- Training Thread ---
training_thread: Union['TrainingThread', None] = None # <-- Fix: Add quotes

# --- File System Watcher ---
recording_observer: Union['Observer', None] = None # <-- Fix: Add quotes


# =================================================================
# VISUALIZATION PAGE STATE
# =================================================================

cur_actogram: Union['cbas.Actogram', None] = None # <-- Fix: Add quotes
viz_task_lock: Union[threading.Lock, None] = threading.Lock()
latest_viz_task_id: int = 0

# =================================================================
# GLOBAL LOGGING QUEUE
# =================================================================
log_queue: queue.Queue = queue.Queue()

# =================================================================
# RECORDING PAGE STATE
# =================================================================

live_preview_process: Union[subprocess.Popen, None] = None

def get_all_active_processes() -> list:
    """Returns a list of all known, active Popen objects."""
    procs = []
    if proj and proj.active_recordings:
        procs.extend(proj.active_recordings.values())
    if live_preview_process:
        procs.append(live_preview_process)
    return procs