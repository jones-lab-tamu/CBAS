"""
Defines the global state for the CBAS application.

This module serves as a centralized, shared memory space for all other backend
modules. It holds references to the current project, labeling session details,
background thread handles, and other application-wide variables. This avoids
the need for scattered global variables and makes state management more predictable.
"""

import queue
import threading
from typing import TYPE_CHECKING, Any, List, Dict, Union
import pandas as pd

# Import project base classes for type hinting. This is safe as long as these
# modules do not import gui_state at the top level, avoiding circular dependencies.
import cbas
import cv2
import subprocess

# Use TYPE_CHECKING block to import types only for static analysis (e.g., by linters
# or IDEs), preventing runtime circular import errors with the workthreads module.
if TYPE_CHECKING:
    from workthreads import TrainingThread, EncodeThread, ClassificationThread
    from watchdog.observers import Observer
    from cmap import Colormap


# =================================================================
# GLOBAL PROJECT STATE
# =================================================================

proj: Union[cbas.Project, None] = None
"""The currently loaded cbas.Project instance. None if no project is open."""


# =================================================================
# LABELING INTERFACE STATE
# =================================================================

label_capture: Union[cv2.VideoCapture, None] = None
"""OpenCV video capture object for the video currently being labeled."""

label_dataset: Union[cbas.Dataset, None] = None
"""The cbas.Dataset object that is the target of the current labeling session."""

label_videos: List[str] = []
"""A list of video file paths for the current labeling session. Often just one video."""

label_vid_index: int = -1
"""The index of the current video being shown from the `label_videos` list."""

label_index: int = -1
"""The current frame number (playhead position) in the `label_capture`."""

label_start: int = -1
"""The start frame of a new label instance being defined by the user."""

label_type: int = -1
"""The behavior index of a new label instance being defined by the user."""

label_session_buffer: List[Dict[str, Any]] = []
"""In-memory buffer holding all labels (both pre-loaded and user-created) for the current video session."""

label_probability_df: Union[pd.DataFrame, None] = None
"""A pandas DataFrame holding the per-frame probabilities for the currently loaded video."""

label_history: List[Dict[str, Any]] = []
"""A stack of newly created instances, used for the 'Undo' (pop) functionality."""

selected_instance_index: int = -1
"""The index within the `label_session_buffer` of the currently highlighted instance. -1 means none."""

label_behavior_colors: List[str] = []
"""The pre-vetted, high-contrast list of hex color strings for the current set of behaviors."""

label_confirmation_mode: bool = False
"""A flag to indicate if the user is in the final 'commit' confirmation view."""

label_confidence_threshold: int = 100
"""The current confidence threshold (0-100) set by the user."""

label_unfiltered_instances: List[Dict[str, Any]] = []
"""Stores the complete, original list of pre-labeled instances before filtering."""

# Use a type hint for the colormap object if possible, otherwise fallback to Any.
if TYPE_CHECKING:
    label_col_map: Union[Colormap, None] = None
else:
    label_col_map: Any = None


# =================================================================
# BACKGROUND THREADS & TASK QUEUES
# =================================================================

# --- Encoding Thread ---
encode_thread: Union['EncodeThread', None] = None
"""Handle to the background thread responsible for video encoding."""
encode_lock: Union[threading.Lock, None] = None
"""A lock to ensure thread-safe access to the `encode_tasks` list."""
encode_tasks: List[str] = []
"""A queue of video file paths waiting to be encoded into HDF5 embeddings."""

# --- Classification (Inference) Thread ---
classify_thread: Union['ClassificationThread', None] = None
"""Handle to the background thread responsible for running model inference."""
classify_lock: Union[threading.Lock, None] = None
"""A lock to ensure thread-safe access to the `classify_tasks` list."""
classify_tasks: List[str] = []
"""A queue of HDF5 file paths waiting for classification."""

# --- Training Thread ---
training_thread: Union['TrainingThread', None] = None
"""Handle to the background thread responsible for training models."""
# Note: The training thread manages its own internal queue.

# --- File System Watcher ---
recording_observer: Union['Observer', None] = None
"""Handle to the watchdog observer that monitors the recordings directory for new files."""


# =================================================================
# VISUALIZATION PAGE STATE
# =================================================================

cur_actogram: Union[cbas.Actogram, None] = None
"""The currently generated cbas.Actogram object being displayed on the Visualize page."""

# --- Task Management for Visualization ---
viz_task_lock: Union[threading.Lock, None] = threading.Lock()
"""A lock to ensure thread-safe access to the visualization task ID."""
latest_viz_task_id: int = 0
"""The ID of the most recently requested visualization task. Used to discard obsolete results."""

# =================================================================
# GLOBAL LOGGING QUEUE
# =================================================================
log_queue: queue.Queue = queue.Queue()

# =================================================================
# RECORDING PAGE STATE
# =================================================================

live_preview_process: Union[subprocess.Popen, None] = None
"""Handle to the FFMPEG process for the currently active live preview."""