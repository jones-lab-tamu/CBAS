"""
Manages all backend logic for the 'Label/Train' page of the CBAS application.

This includes:
- Handling the labeling interface state (loading videos, drawing frames).
- Managing the session buffer for in-memory label editing.
- Saving corrected labels back to file.
- Providing data for the UI (dataset configs, video lists, etc.).
- Launching and managing training and inference tasks via the workthreads module.
"""

import os
import base64
import traceback
import yaml
import cv2
import numpy as np
import torch
import pandas as pd
import shutil
from datetime import datetime

# Project-specific imports
import cbas
import classifier_head
import gui_state
import workthreads
from cmap import Colormap

import eel
import sys
import subprocess
import threading


# =================================================================
# HELPER FUNCTIONS
# =================================================================

def _video_import_worker(session_name: str, subject_name: str, video_paths: list[str]):
    """
    (WORKER) Runs in a separate thread to handle slow file copy operations.
    """
    try:
        workthreads.log_message(f"Starting import of {len(video_paths)} videos to session '{session_name}' for subject '{subject_name}'.", "INFO")

        # The session_name is the top-level folder.
        session_dir = os.path.join(gui_state.proj.recordings_dir, session_name)

        # The final destination is now a subfolder named with the user-provided subject_name
        final_dest_dir = os.path.join(session_dir, subject_name)
        
        if os.path.exists(final_dest_dir) and os.listdir(final_dest_dir):
            raise FileExistsError(f"A folder for subject '{subject_name}' already exists in this session and is not empty.")
            
        os.makedirs(final_dest_dir, exist_ok=True)
        
        newly_copied_files = []
        for video_path in video_paths:
            try:
                basename = os.path.basename(video_path)
                dest_path = os.path.join(final_dest_dir, basename)
                shutil.copy(video_path, dest_path)
                newly_copied_files.append(dest_path)
            except Exception as copy_error:
                workthreads.log_message(f"Could not copy '{os.path.basename(video_path)}'. Skipping. Error: {copy_error}", "WARN")

        if newly_copied_files:
            with gui_state.encode_lock:
                new_files_to_queue = [f for f in newly_copied_files if f not in gui_state.encode_tasks]
                gui_state.encode_tasks.extend(new_files_to_queue)
            workthreads.log_message(f"Queued {len(new_files_to_queue)} imported files for encoding.", "INFO")
        
        gui_state.proj.reload_recordings()

        workthreads.log_message(f"Successfully imported {len(video_paths)} video(s).", "INFO")
        eel.notify_import_complete(True, f"Successfully imported {len(video_paths)} video(s) to session '{session_name}' under subject '{subject_name}'.")

    except Exception as e:
        print(f"ERROR in video import worker: {e}")
        traceback.print_exc()
        eel.notify_import_complete(False, f"Import failed: {e}")


def color_distance(rgb1, rgb2):
    """Calculates the perceived distance between two RGB colors for contrast checking."""
    rmean = (rgb1[0] + rgb2[0]) // 2
    r = rgb1[0] - rgb2[0]
    g = rgb1[1] - rgb2[1]
    b = rgb1[2] - rgb2[2]
    return (((512 + rmean) * r * r) >> 8) + (4 * g * g) + (((767 - rmean) * b * b) >> 8)


def hex_to_rgb(hex_color):
    """Converts a hex color string (e.g., '#RRGGBB') to an (R, G, B) tuple."""
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def tab20_map(val: int) -> int:
    """
    Maps an integer to an index for the 'seaborn:tab20' colormap.
    This version includes manual overrides for known problematic (grey/brown) colors
    to ensure good UI contrast against the timeline background.
    """
    remap = {7: 6, 14: 2, 15: 4}  # Remap grey/brown colors to vibrant ones
    if val in remap:
        return remap[val]
    # Default distinct color mapping logic
    return (val * 2) if val < 10 else ((val - 10) * 2 + 1)


# =================================================================
# EEL-EXPOSED FUNCTIONS: DATA & CONFIGURATION
# =================================================================

@eel.expose
def model_exists(model_name: str) -> bool:
    """Checks if a model with the given name exists in the current project."""
    return gui_state.proj and model_name in gui_state.proj.models


@eel.expose
def load_dataset_configs() -> dict:
    """Loads configurations for all available datasets."""
    if not gui_state.proj: return {}
    return {name: dataset.config for name, dataset in gui_state.proj.datasets.items()}


@eel.expose
def get_available_models() -> list[str]:
    """Returns a sorted list of all model names available in the project."""
    if not gui_state.proj: return []
    return sorted(list(gui_state.proj.models.keys()))


@eel.expose
def get_record_tree() -> dict:
    """Fetches the recording directory tree structure for modal dialogs."""
    if not gui_state.proj or not os.path.exists(gui_state.proj.recordings_dir):
        return {}
    tree = {}
    try:
        # The top-level directories inside 'recordings' are the session names.
        for session_dir in os.scandir(gui_state.proj.recordings_dir):
            if session_dir.is_dir():
                # The value is a list of the subject/camera directories inside the session.
                subjects = [subject.name for subject in os.scandir(session_dir.path) if subject.is_dir()]
                if subjects:
                    tree[session_dir.name] = subjects
    except Exception as e:
        print(f"Error building record tree: {e}")
    return tree


@eel.expose
def get_videos_for_dataset(dataset_name: str) -> list[tuple[str, str]]:
    """Finds all .mp4 files within a dataset's whitelist for 'Label from Scratch' mode."""
    if not gui_state.proj: return []
    dataset = gui_state.proj.datasets.get(dataset_name)
    if not dataset: return []
    
    whitelist = dataset.config.get("whitelist", [])
    if not whitelist: return []

    video_list = []
    if gui_state.proj.recordings_dir and os.path.exists(gui_state.proj.recordings_dir):
        for root, _, files in os.walk(gui_state.proj.recordings_dir):
            for file in files:
                if file.endswith(".mp4"):
                    video_path = os.path.join(root, file)
                    normalized_path = os.path.normpath(video_path)
                    if any(os.path.normpath(p) in normalized_path for p in whitelist):
                        display_name = os.path.relpath(video_path, gui_state.proj.recordings_dir)
                        video_list.append((video_path, display_name))
    
    return sorted(video_list, key=lambda x: x[1])


@eel.expose
def get_inferred_session_dirs(dataset_name: str, model_name: str) -> list[str]:
    """Finds unique sub-directories that contain videos inferred by a specific model."""
    if not gui_state.proj: return []
    dataset = gui_state.proj.datasets.get(dataset_name)
    if not dataset: return []
    
    whitelist_paths = [os.path.abspath(os.path.join(gui_state.proj.recordings_dir, p)) for p in dataset.config.get("whitelist", [])]
    if not whitelist_paths: return []

    inferred_dirs = set()
    for root, _, files in os.walk(gui_state.proj.recordings_dir):
        for file in files:
            if file.endswith(f"_{model_name}_outputs.csv"):
                csv_abs_path = os.path.abspath(os.path.join(root, file))
                if any(csv_abs_path.startswith(wl_path) for wl_path in whitelist_paths):
                    inferred_dirs.add(os.path.relpath(root, gui_state.proj.recordings_dir))
    
    return sorted(list(inferred_dirs))


@eel.expose
def get_inferred_videos_for_session(session_dir_rel: str, model_name: str) -> list[tuple[str, str]]:
    """Gets a list of inferred videos from a single specified session directory."""
    if not gui_state.proj: return []
    session_abs_path = os.path.join(gui_state.proj.recordings_dir, session_dir_rel)
    if not os.path.isdir(session_abs_path): return []
    
    ready_videos = []
    for file in os.listdir(session_abs_path):
        if file.endswith(f"_{model_name}_outputs.csv"):
            csv_path = os.path.join(session_abs_path, file)
            mp4_path = csv_path.replace(f"_{model_name}_outputs.csv", ".mp4")
            if os.path.exists(mp4_path):
                ready_videos.append((mp4_path, os.path.basename(mp4_path)))
                
    return sorted(ready_videos, key=lambda x: x[1])


@eel.expose
def get_existing_session_names() -> list[str]:
    """
    Scans the project's recordings directory and returns a sorted list of all
    top-level session folders.
    """
    if not gui_state.proj or not os.path.isdir(gui_state.proj.recordings_dir):
        return []
    
    # The session names are the names of the directories
    # immediately inside the main 'recordings' folder.
    try:
        session_names = [d.name for d in os.scandir(gui_state.proj.recordings_dir) if d.is_dir()]
        return sorted(session_names)
    except Exception as e:
        print(f"Error scanning for session names: {e}")
        return []


@eel.expose
def import_videos(session_name: str, subject_name: str, video_paths: list[str]) -> bool:
    """
    (LAUNCHER) Starts the video import process in a background thread.
    Returns immediately to keep the UI responsive.
    """
    if not gui_state.proj or not session_name or not subject_name or not video_paths:
        return False
    
    print(f"Spawning background thread to import {len(video_paths)} video(s)...")
    
    import_thread = threading.Thread(
        target=_video_import_worker,
        args=(session_name, subject_name, video_paths)
    )
    
    import_thread.daemon = True
    import_thread.start()
    
    return True

# =================================================================
# EEL-EXPOSED FUNCTIONS: LABELING WORKFLOW & ACTIONS
# =================================================================

@eel.expose
def get_model_configs() -> dict:
    """Loads configurations for all available models."""
    if not gui_state.proj:
        return {}
    return {name: model.config for name, model in gui_state.proj.models.items()}

def _start_labeling_worker(name: str, video_to_open: str = None, preloaded_instances: list = None, probability_df: pd.DataFrame = None):
    """
    (WORKER) This is the background task that prepares the labeling session. It handles
    all state setup and then calls back to the JavaScript UI when ready.
    """
    try:
        print("Labeling worker started.")
        if gui_state.proj is None: raise ValueError("Project is not loaded.")
        if name not in gui_state.proj.datasets: raise ValueError(f"Dataset '{name}' not found.")
        if not video_to_open or not os.path.exists(video_to_open):
            raise FileNotFoundError(f"Video to label does not exist: {video_to_open}")

        print("Resetting labeling state for new session.")
        if gui_state.label_capture and gui_state.label_capture.isOpened():
            gui_state.label_capture.release()
        
        gui_state.label_capture, gui_state.label_index = None, -1
        gui_state.label_videos, gui_state.label_vid_index = [], -1
        gui_state.label_type, gui_state.label_start = -1, -1
        gui_state.label_history, gui_state.label_behavior_colors = [], []
        gui_state.label_session_buffer, gui_state.selected_instance_index = [], -1
        gui_state.label_probability_df = probability_df
        gui_state.label_confirmation_mode = False
        gui_state.label_confidence_threshold = 100
        
        eel.setConfirmationModeUI(False)()
        
        dataset: cbas.Dataset = gui_state.proj.datasets[name]
        gui_state.label_dataset = dataset
        gui_state.label_col_map = Colormap("seaborn:tab20")

        print("Loading session buffer...")
        gui_state.label_videos = [video_to_open]
        current_video_path = gui_state.label_videos[0]

        for b_name, b_insts in gui_state.label_dataset.labels["labels"].items():
            for inst in b_insts:
                if inst.get("video") == current_video_path:
                    gui_state.label_session_buffer.append(inst)
        print(f"Loaded {len(gui_state.label_session_buffer)} existing human labels into buffer.")

        labeling_mode = 'scratch'
        model_name_for_ui = ''
        if preloaded_instances:
            labeling_mode = 'review'
            model_name_for_ui = name

            gui_state.label_unfiltered_instances = preloaded_instances.copy()
            
            initial_threshold = gui_state.label_confidence_threshold / 100.0
            filtered_instances = [
                p for p in preloaded_instances if p.get('confidence', 1.0) < initial_threshold
            ]
            
            print(f"Applying {len(filtered_instances)} initially filtered instances.")
            for pred_inst in filtered_instances:
                is_overlapping = any(max(pred_inst['start'], h['start']) <= min(pred_inst['end'], h['end']) for h in gui_state.label_session_buffer)
                if not is_overlapping:
                    gui_state.label_session_buffer.append(pred_inst)
        
        dataset_behaviors = gui_state.label_dataset.labels.get("behaviors", [])
        behavior_colors = [str(gui_state.label_col_map(tab20_map(i))) for i in range(len(dataset_behaviors))]
        gui_state.label_behavior_colors = behavior_colors

        print("Setup complete. Calling frontend to build UI.")
        eel.buildLabelingUI(dataset_behaviors, behavior_colors)()
        eel.setLabelingModeUI(labeling_mode, model_name_for_ui)()
        eel.setConfirmationModeUI(False)()

        if not gui_state.label_videos: raise ValueError("Video list is empty after setup.")
        print("Loading video and pushing initial frame...")
        next_video(0)

    except Exception as e:
        print(f"FATAL ERROR in labeling worker: {e}")
        traceback.print_exc()
        eel.showErrorOnLabelTrainPage(f"Failed to start labeling session: {e}")()


@eel.expose
def start_labeling(name: str, video_to_open: str = None, preloaded_instances: list = None) -> bool:
    """
    (LAUNCHER) Lightweight function to spawn the labeling worker in the background.
    """
    try:
        eel.spawn(_start_labeling_worker, name, video_to_open, preloaded_instances, None)
        print(f"Spawned labeling worker for dataset '{name}'.")
        return True
    except Exception as e:
        print(f"Failed to spawn labeling worker: {e}")
        return False


@eel.expose
def start_labeling_with_preload(dataset_name: str, model_name: str, video_path_to_label: str) -> bool:
    """
    Runs a quick inference step and then spawns the labeling worker.
    """
    try:
        print(f"Request to pre-label '{os.path.basename(video_path_to_label)}' with model '{model_name}'...")
        if gui_state.proj is None: raise ValueError("Project not loaded")
        
        dataset = gui_state.proj.datasets.get(dataset_name)
        model_obj = gui_state.proj.models.get(model_name)
        if not dataset or not model_obj: raise ValueError("Dataset or Model not found.")

        target_behaviors = set(dataset.config.get("behaviors", []))
        model_behaviors = set(model_obj.config.get("behaviors", []))

        if not target_behaviors.intersection(model_behaviors):
            error_message = (
                f"Model '{model_name}' and Dataset '{dataset_name}' have no behaviors in common. "
                "Pre-labeling cannot proceed."
            )
            print(f"ERROR: {error_message}")
            eel.showErrorOnLabelTrainPage(error_message)()
            return False

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_model = classifier_head.classifier(
            in_features=768,
            out_features=len(model_obj.config["behaviors"]),
            seq_len=model_obj.config["seq_len"],
        ).to(device)
        torch_model.load_state_dict(torch.load(model_obj.weights_path, map_location=device, weights_only=True))
        torch_model.eval()

        h5_path = video_path_to_label.replace(".mp4", "_cls.h5")
        if not os.path.exists(h5_path): raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
        
        csv_path = cbas.infer_file(
            file_path=h5_path, model=torch_model, dataset_name=model_obj.name,
            behaviors=model_obj.config["behaviors"], seq_len=model_obj.config["seq_len"], device=device
        )
        if not csv_path or not os.path.exists(csv_path):
            raise RuntimeError("Inference failed to produce a CSV output file.")

        preloaded_instances, probability_df = dataset.predictions_to_instances_with_confidence(csv_path, model_obj.name)
        
        eel.spawn(_start_labeling_worker, dataset_name, video_path_to_label, preloaded_instances, probability_df)
        
        print(f"Spawned pre-labeling worker for video '{os.path.basename(video_path_to_label)}'.")
        return True

    except Exception as e:
        print(f"ERROR in start_labeling_with_preload: {e}")
        eel.showErrorOnLabelTrainPage(f"Failed to start pre-labeling: {e}")()
        return False


@eel.expose
def save_session_labels():
    """
    Filters for only human-verified/confirmed instances and saves them.
    """
    if not gui_state.label_dataset or not gui_state.label_videos: return

    final_labels_to_save = []
    for inst in gui_state.label_session_buffer:
        if 'confidence' not in inst or inst.get('_confirmed', False):
            final_inst = inst.copy()
            for key in ['confidence', 'confidences', '_original_start', '_original_end', '_confirmed']:
                final_inst.pop(key, None)
            final_labels_to_save.append(final_inst)

    current_video_path = gui_state.label_videos[0]
    all_labels = gui_state.label_dataset.labels["labels"]

    for behavior_name in all_labels:
        all_labels[behavior_name][:] = [
            inst for inst in all_labels[behavior_name] if inst.get("video") != current_video_path
        ]
    
    for corrected_inst in final_labels_to_save:
        all_labels.setdefault(corrected_inst['label'], []).append(corrected_inst)
    
    with open(gui_state.label_dataset.labels_path, "w") as file:
        yaml.dump(gui_state.label_dataset.labels, file, allow_unicode=True)
    
    print(f"Saved {len(final_labels_to_save)} human-verified/corrected labels for video {os.path.basename(current_video_path)}.")
    
    gui_state.label_confirmation_mode = False
    eel.setConfirmationModeUI(False)()
    render_image()


@eel.expose
def refilter_instances(new_threshold: int):
    """
    Re-filters the session buffer based on a new confidence threshold and re-renders.
    """
    print(f"Refiltering instances with threshold < {new_threshold}%")
    gui_state.label_confidence_threshold = new_threshold
    
    unfiltered = gui_state.label_unfiltered_instances
    human_labels = [inst for inst in gui_state.label_session_buffer if 'confidence' not in inst]
    
    threshold_float = new_threshold / 100.0
    newly_filtered = [
        p for p in unfiltered if p.get('confidence', 1.0) < threshold_float
    ]
    
    gui_state.label_session_buffer = human_labels + newly_filtered
    
    gui_state.selected_instance_index = -1
    eel.highlightBehaviorRow(None)()
    eel.updateConfidenceBadge(None, None)()
    render_image()
    update_counts()

# =================================================================
# EEL-EXPOSED FUNCTIONS: IN-SESSION LABELING ACTIONS
# =================================================================

@eel.expose
def jump_to_frame(frame_number: int):
    """Jumps the video playhead to a specific frame number."""
    if not (gui_state.label_capture and gui_state.label_capture.isOpened()):
        return
    
    try:
        frame_num = int(frame_number)
        total_frames = gui_state.label_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        safe_frame_num = max(0, min(frame_num, int(total_frames) - 1))
        gui_state.label_index = safe_frame_num
        render_image()
    except (ValueError, TypeError):
        print(f"Invalid frame number received: {frame_number}")


@eel.expose
def confirm_selected_instance():
    """
    Toggles the 'confirmed' state of the currently selected instance.
    """
    if gui_state.selected_instance_index != -1 and gui_state.selected_instance_index < len(gui_state.label_session_buffer):
        instance = gui_state.label_session_buffer[gui_state.selected_instance_index]
        is_currently_confirmed = instance.get('_confirmed', False)

        if is_currently_confirmed:
            instance['_confirmed'] = False
            if '_original_start' in instance:
                instance['start'] = instance['_original_start']
                instance['end'] = instance['_original_end']
            print(f"Unlocked instance {gui_state.selected_instance_index} and reverted changes.")
        
        else:
            instance['_confirmed'] = True
            print(f"Confirmed instance {gui_state.selected_instance_index}.")
            
        render_image()


@eel.expose
def handle_click_on_label_image(x: int, y: int):
    """Handles a click on the timeline to scrub to a specific frame."""
    if gui_state.label_capture and gui_state.label_capture.isOpened():
        total_frames = gui_state.label_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if total_frames > 0:
            gui_state.label_index = int(x * total_frames / 500)
            render_image()


@eel.expose
def next_video(shift: int):
    """Loads the next or previous video in the labeling session's video list."""
    if not gui_state.label_videos:
        eel.updateLabelImageSrc(None, None, None); eel.updateFileInfo("No videos available."); return

    gui_state.label_start, gui_state.label_type = -1, -1
    gui_state.label_vid_index = (gui_state.label_vid_index + shift) % len(gui_state.label_videos)
    current_video_path = gui_state.label_videos[gui_state.label_vid_index]

    if gui_state.label_capture and gui_state.label_capture.isOpened():
        gui_state.label_capture.release()
    
    print(f"Loading video for labeling: {current_video_path}")
    capture = cv2.VideoCapture(current_video_path)

    if not capture.isOpened():
        eel.updateLabelImageSrc(None, None, None); eel.updateFileInfo(f"Error loading video."); gui_state.label_capture = None; return

    gui_state.label_capture = capture
    gui_state.label_index = 0
    render_image()
    update_counts()


@eel.expose
def next_frame(shift: int):
    """
    Moves the playhead forward or backward by a number of frames.
    """
    if not (gui_state.label_capture and gui_state.label_capture.isOpened()):
        return

    new_index = gui_state.label_index + shift
    total_frames = int(gui_state.label_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames > 0:
        gui_state.label_index = max(0, min(new_index, total_frames - 1))
    
    render_image()


@eel.expose
def jump_to_instance(direction: int):
    """Finds the next/previous instance and jumps the playhead."""
    if not gui_state.label_session_buffer:
        eel.highlightBehaviorRow(None)(); eel.updateConfidenceBadge(None, None)(); return

    for i, inst in enumerate(gui_state.label_session_buffer):
        if inst.get("start", -1) <= gui_state.label_index <= inst.get("end", -1):
            gui_state.selected_instance_index = i
            break
    else:
        gui_state.selected_instance_index = -1
        
    sorted_instances = sorted(enumerate(gui_state.label_session_buffer), key=lambda item: item[1]['start'])
    if not sorted_instances:
        eel.highlightBehaviorRow(None)(); eel.updateConfidenceBadge(None, None)(); return

    current_frame = gui_state.label_index
    target_item = None

    if direction > 0:
        found = next((item for item in sorted_instances if item[1]['start'] > current_frame), None)
        target_item = found or sorted_instances[0]
    else:
        found = next((item for item in reversed(sorted_instances) if item[1]['start'] < current_frame), None)
        target_item = found or sorted_instances[-1]
            
    if target_item:
        original_index, instance_data = target_item
        gui_state.label_index = instance_data['start']
        gui_state.selected_instance_index = gui_state.label_session_buffer.index(instance_data)
        confidence = instance_data.get('confidence')
        eel.updateConfidenceBadge(instance_data['label'], confidence)()
        eel.highlightBehaviorRow(instance_data['label'])()
        render_image()
    else:
        eel.highlightBehaviorRow(None)(); eel.updateConfidenceBadge(None, None)()


@eel.expose
def update_instance_boundary(boundary_type: str):
    """
    Directly updates the start or end frame of the currently selected instance.
    """
    if gui_state.selected_instance_index == -1 or gui_state.selected_instance_index >= len(gui_state.label_session_buffer):
        return

    active_instance = gui_state.label_session_buffer[gui_state.selected_instance_index]
    new_boundary_frame = gui_state.label_index

    if 'confidence' in active_instance and '_original_start' not in active_instance:
        active_instance['_original_start'] = active_instance['start']
        active_instance['_original_end'] = active_instance['end']
        active_instance['_confirmed'] = False

    if boundary_type == 'start':
        if new_boundary_frame >= active_instance['end']: return
        new_start, new_end = new_boundary_frame, active_instance['end']
    elif boundary_type == 'end':
        if new_boundary_frame <= active_instance['start']: return
        new_start, new_end = active_instance['start'], new_boundary_frame
    else: return

    indices_to_pop = []
    for i, neighbor in enumerate(gui_state.label_session_buffer):
        if i == gui_state.selected_instance_index: continue
        if max(new_start, neighbor['start']) <= min(new_end, neighbor['end']):
            if boundary_type == 'start' and new_start <= neighbor['end']: neighbor['end'] = new_start - 1
            elif boundary_type == 'end' and new_end >= neighbor['start']: neighbor['start'] = new_end + 1
            if neighbor['start'] >= neighbor['end']: indices_to_pop.append(i)

    if indices_to_pop:
        for i in sorted(indices_to_pop, reverse=True):
            if i < gui_state.selected_instance_index: gui_state.selected_instance_index -= 1
            gui_state.label_session_buffer.pop(i)

    active_instance_idx = gui_state.selected_instance_index
    if active_instance_idx < len(gui_state.label_session_buffer):
        active_instance = gui_state.label_session_buffer[active_instance_idx]
        if boundary_type == 'start': active_instance['start'] = new_boundary_frame
        elif boundary_type == 'end': active_instance['end'] = new_boundary_frame
    
    render_image()


@eel.expose
def get_zoom_range_for_click(x_pos: int) -> int:
    """Calculates a new frame index based on a click on the zoom bar."""
    if gui_state.selected_instance_index != -1 and gui_state.selected_instance_index < len(gui_state.label_session_buffer):
        instance = gui_state.label_session_buffer[gui_state.selected_instance_index]
        total_frames = gui_state.label_capture.get(cv2.CAP_PROP_FRAME_COUNT)

        original_start = instance.get('_original_start', instance['start'])
        original_end = instance.get('_original_end', instance['end'])
        
        inst_len = original_end - original_start
        context = inst_len * 2
        zoom_start = max(0, original_start - context)
        zoom_end = min(total_frames, original_end + context)
        
        if zoom_end > zoom_start:
            new_frame = int(zoom_start + (x_pos / 500.0) * (zoom_end - zoom_start))
            gui_state.label_index = new_frame
            render_image()


def add_instance_to_buffer():
    """Helper function to create a new label instance and add it to the session buffer."""
    if gui_state.label_type == -1 or gui_state.label_start == -1: return

    start_idx, end_idx = min(gui_state.label_start, gui_state.label_index), max(gui_state.label_start, gui_state.label_index)
    if start_idx == end_idx: return

    for inst in gui_state.label_session_buffer:
        if max(start_idx, inst['start']) <= min(end_idx, inst['end']):
            eel.showErrorOnLabelTrainPage("Overlapping behavior region! Behavior not recorded.")
            return

    behavior_name = gui_state.label_dataset.labels["behaviors"][gui_state.label_type]
    new_instance = { "video": gui_state.label_videos[gui_state.label_vid_index], "start": start_idx, "end": end_idx, "label": behavior_name }
    gui_state.label_session_buffer.append(new_instance)
    gui_state.label_history.append(new_instance)
    update_counts()


@eel.expose
def label_frame(value: int):
    """Handles user keypresses to start, end, or change labels."""
    if gui_state.label_dataset is None or not gui_state.label_videos: return
    behaviors = gui_state.label_dataset.labels.get("behaviors", [])
    if not 0 <= value < len(behaviors): return

    clicked_instance_index = -1
    for i, inst in enumerate(gui_state.label_session_buffer):
        if inst.get("start", -1) <= gui_state.label_index <= inst.get("end", -1):
            clicked_instance_index = i
            break
            
    if clicked_instance_index != -1 and gui_state.label_type == -1:
        new_behavior_name = behaviors[value]
        gui_state.label_session_buffer[clicked_instance_index]['label'] = new_behavior_name
        print(f"Changed instance {clicked_instance_index} to '{new_behavior_name}'.")
    else:
        if value == gui_state.label_type:
            add_instance_to_buffer()
            gui_state.label_type = -1; gui_state.label_start = -1
        elif gui_state.label_type == -1:
            gui_state.label_type, gui_state.label_start = value, gui_state.label_index
            gui_state.selected_instance_index = -1
            eel.updateConfidenceBadge(None, None)()
        else:
            gui_state.label_type, gui_state.label_start = value, gui_state.label_index
            eel.updateConfidenceBadge(None, None)()
            
    render_image()


@eel.expose
def delete_instance_from_buffer():
    """Finds and removes an instance from the session buffer at the current frame."""
    if not gui_state.label_session_buffer: return
    current_frame = gui_state.label_index
    idx_to_remove = -1
    for i, inst in enumerate(gui_state.label_session_buffer):
        if inst.get("start", -1) <= current_frame <= inst.get("end", -1):
            idx_to_remove = i
            break
    if idx_to_remove != -1:
        removed_inst = gui_state.label_session_buffer.pop(idx_to_remove)
        if removed_inst in gui_state.label_history:
            gui_state.label_history.remove(removed_inst)
        gui_state.selected_instance_index = -1
        eel.updateConfidenceBadge(None, None)()
        render_image(); update_counts()


@eel.expose
def pop_instance_from_buffer():
    """Undoes the last-added instance from the session buffer."""
    if not gui_state.label_history: return
    last_added = gui_state.label_history.pop()
    try:
        gui_state.label_session_buffer.remove(last_added)
        gui_state.selected_instance_index = -1
        render_image(); update_counts()
    except ValueError:
        print(f"Could not pop {last_added}, not found in buffer.")

@eel.expose
def stage_for_commit():
    """Enters confirmation mode and triggers a re-render showing only staged labels."""
    gui_state.label_confirmation_mode = True
    eel.setConfirmationModeUI(True)
    render_image()

@eel.expose
def cancel_commit_stage():
    """Exits confirmation mode and triggers a re-render of the normal view."""
    gui_state.label_confirmation_mode = False
    eel.setConfirmationModeUI(False)
    render_image()

# =================================================================
# EEL-EXPOSED FUNCTIONS: DATASET & MODEL MANAGEMENT
# =================================================================

@eel.expose
def reload_project_data():
    """
    Triggers a full reload of the current project's data from the disk.
    """
    if gui_state.proj:
        try:
            gui_state.proj.reload()
            return True
        except Exception as e:
            print(f"Error during project data reload: {e}")
            return False
    return False

@eel.expose
def reveal_dataset_files(dataset_name: str):
    """
    Opens the specified dataset's directory in the user's native file explorer.
    """
    if not gui_state.proj or dataset_name not in gui_state.proj.datasets:
        print(f"Error: Could not find dataset '{dataset_name}' to reveal.")
        return

    dataset_path = gui_state.proj.datasets[dataset_name].path
    print(f"Revealing path for '{dataset_name}': {dataset_path}")

    try:
        if sys.platform == "win32":
            os.startfile(dataset_path)
        elif sys.platform == "darwin":
            subprocess.run(["open", dataset_path])
        else:
            subprocess.run(["xdg-open", dataset_path])
    except Exception as e:
        print(f"Failed to open file explorer for path '{dataset_path}': {e}")
        eel.showErrorOnLabelTrainPage(f"Could not open the folder. Please navigate there manually:\n{dataset_path}")()

@eel.expose
def create_dataset(name: str, behaviors: list[str], recordings_whitelist: list[str]) -> bool:
    """Creates a new dataset via the project interface."""
    if not gui_state.proj: return False
    dataset = gui_state.proj.create_dataset(name, behaviors, recordings_whitelist)
    if dataset:
        gui_state.label_dataset = dataset
        return True
    return False


@eel.expose
def train_model(name: str, batch_size: str, learning_rate: str, epochs: str, sequence_length: str, training_method: str, patience: str):
    """Queues a training task for the specified dataset."""
    if not gui_state.proj or name not in gui_state.proj.datasets: return
    if not gui_state.training_thread: return

    try:
        task = workthreads.TrainingTask(
            name=name, dataset=gui_state.proj.datasets[name],
            behaviors=gui_state.proj.datasets[name].config.get('behaviors', []),
            batch_size=int(batch_size), learning_rate=float(learning_rate),
            epochs=int(epochs), sequence_length=int(sequence_length),
            training_method=training_method,
            patience=int(patience)  # Add patience to the task
        )
        gui_state.training_thread.queue_task(task)
        eel.updateTrainingStatusOnUI(name, "Training task queued...")()
    except ValueError:
        eel.showErrorOnLabelTrainPage("Invalid training parameters provided.")


@eel.expose
def start_classification(dataset_name_for_model: str, recordings_whitelist_paths: list[str]):
    """Queues HDF5 files for classification using a specified model."""
    if not gui_state.proj or not gui_state.classify_thread: return
    model_to_use = gui_state.proj.models.get(dataset_name_for_model)
    if not model_to_use: return

    gui_state.classify_thread.start_inferring(model_to_use, recordings_whitelist_paths)

    h5_files_to_classify = []
    for rel_path in recordings_whitelist_paths:
        search_root = os.path.join(gui_state.proj.recordings_dir, rel_path)
        if os.path.isdir(search_root):
            for dirpath, _, filenames in os.walk(search_root):
                for filename in filenames:
                    if filename.endswith("_cls.h5"):
                        h5_files_to_classify.append(os.path.join(dirpath, filename))
    
    if h5_files_to_classify:
        with gui_state.classify_lock:
            for h5_file in h5_files_to_classify:
                if h5_file not in gui_state.classify_tasks:
                    gui_state.classify_tasks.append(h5_file)
        print(f"Queued {len(h5_files_to_classify)} files for classification with '{model_to_use.name}'.")


# =================================================================
# RENDERING & INTERNAL LOGIC (Not Exposed)
# =================================================================

def render_image():
    """
    Renders the current video frame and timelines using the robust "crop and zoom" 
    approach, with a visible external bracket for selection.
    """
    if not gui_state.label_capture or not gui_state.label_capture.isOpened():
        eel.updateLabelImageSrc(None, None, None)(); return

    total_frames = int(gui_state.label_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        eel.updateLabelImageSrc(None, None, None)(); return

    current_frame_idx = max(0, min(int(gui_state.label_index), total_frames - 1))
    gui_state.label_capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
    ret, frame = gui_state.label_capture.read()

    if not ret or frame is None:
        eel.updateLabelImageSrc(None, None, None)(); return

    main_frame_blob, timeline_blob, zoom_blob = None, None, None

    frame_resized = cv2.resize(frame, (500, 500))
    _, encoded_main_frame = cv2.imencode(".jpg", frame_resized)
    main_frame_blob = base64.b64encode(encoded_main_frame.tobytes()).decode("utf-8")

    base_timeline_canvas = np.full((50, 500, 3), 100, dtype=np.uint8)
    fill_colors(base_timeline_canvas, total_frames)

    full_timeline_with_highlight = base_timeline_canvas.copy()
    if gui_state.selected_instance_index != -1 and gui_state.selected_instance_index < len(gui_state.label_session_buffer):
        instance = gui_state.label_session_buffer[gui_state.selected_instance_index]
        start_px = int(500 * instance['start'] / total_frames)
        end_px = int(500 * (instance['end'] + 1) / total_frames)
        if start_px >= end_px: end_px = start_px + 1
        cv2.rectangle(full_timeline_with_highlight, (start_px, 0), (end_px, 49), (255, 255, 255), 2)

    zoom_width_frames = total_frames * 0.10
    zoom_start_frame = max(0, current_frame_idx - zoom_width_frames / 2)
    zoom_end_frame = min(total_frames, current_frame_idx + zoom_width_frames / 2)
    start_px_crop = int(500 * zoom_start_frame / total_frames)
    end_px_crop = int(500 * zoom_end_frame / total_frames)
    
    if start_px_crop < end_px_crop:
        zoom_crop = base_timeline_canvas[:, start_px_crop:end_px_crop]
        zoom_canvas = cv2.resize(zoom_crop, (500, 50), interpolation=cv2.INTER_NEAREST)
    else:
        zoom_canvas = np.full((50, 500, 3), 100, dtype=np.uint8)
    
    if gui_state.selected_instance_index != -1 and gui_state.selected_instance_index < len(gui_state.label_session_buffer):
        instance = gui_state.label_session_buffer[gui_state.selected_instance_index]
        zoom_window_duration = zoom_end_frame - zoom_start_frame
        if zoom_window_duration > 0:
            bracket_start_x = int(500 * (instance['start'] - zoom_start_frame) / zoom_window_duration)
            bracket_end_x = int(500 * (instance['end'] + 1 - zoom_start_frame) / zoom_window_duration)
            clamped_start_x = max(0, bracket_start_x)
            clamped_end_x = min(499, bracket_end_x)
            if clamped_start_x <= clamped_end_x:
                y_pos, tick_height, thickness, color = 47, 5, 2, (255, 255, 255)
                cv2.line(zoom_canvas, (clamped_start_x, y_pos), (clamped_end_x, y_pos), color, thickness)
                if bracket_start_x >= 0 and bracket_start_x < 500: cv2.line(zoom_canvas, (bracket_start_x, y_pos), (bracket_start_x, y_pos - tick_height), color, thickness)
                if bracket_end_x > 0 and bracket_end_x < 500: cv2.line(zoom_canvas, (bracket_end_x, y_pos), (bracket_end_x, y_pos - tick_height), color, thickness)

    marker_color = (0, 0, 0)
    cv2.line(zoom_canvas, (249, 0), (249, 50), marker_color, 2)
    cv2.line(full_timeline_with_highlight, (int(500 * current_frame_idx / total_frames), 0), (int(500 * current_frame_idx / total_frames), 50), marker_color, 2)
    
    _, encoded_timeline = cv2.imencode(".jpg", full_timeline_with_highlight)
    timeline_blob = base64.b64encode(encoded_timeline.tobytes()).decode("utf-8")
    
    _, encoded_zoom = cv2.imencode(".jpg", zoom_canvas)
    zoom_blob = base64.b64encode(encoded_zoom.tobytes()).decode("utf-8")

    eel.updateLabelImageSrc(main_frame_blob, timeline_blob, zoom_blob)()


def fill_colors(canvas_img: np.ndarray, total_frames: int):
    """
    Draws colored bars on a timeline canvas.
    """
    if not all([gui_state.label_dataset, gui_state.label_videos, gui_state.label_capture, gui_state.label_capture.isOpened()]):
        return canvas_img

    behaviors = gui_state.label_dataset.labels.get("behaviors", [])
    if total_frames == 0: return canvas_img

    timeline_y, timeline_h, timeline_w = 0, canvas_img.shape[0], canvas_img.shape[1]

    for inst in gui_state.label_session_buffer:
        is_human_added = 'confidence' not in inst
        is_confirmed = inst.get('_confirmed', False)
        
        if gui_state.label_confirmation_mode and not (is_human_added or is_confirmed):
            continue

        try:
            b_idx = behaviors.index(inst['label'])
            color_hex = gui_state.label_behavior_colors[b_idx].lstrip("#")
            bgr_color = tuple(int(color_hex[i:i+2], 16) for i in (4, 2, 0))
            
            start_px = int(timeline_w * inst.get("start", 0) / total_frames)
            end_px = int(timeline_w * (inst.get("end", 0) + 1) / total_frames)

            if inst.get("start", 0) <= inst.get("end", 0) and start_px == end_px:
                end_px = start_px + 1

            if start_px < end_px:
                confidence = inst.get('confidence', 1.0)
                overlay = canvas_img.copy()
                cv2.rectangle(overlay, (start_px, timeline_y), (end_px, timeline_y + timeline_h - 1), bgr_color, -1)
                
                alpha = 1.0 if gui_state.label_confirmation_mode else confidence
                cv2.addWeighted(overlay, alpha, canvas_img, 1 - alpha, 0, canvas_img)
                
                if is_confirmed:
                   cv2.rectangle(canvas_img, (start_px, timeline_y), (end_px, timeline_y + timeline_h - 1), (255, 255, 255), 1)

        except (ValueError, IndexError):
            continue

    if gui_state.label_type != -1 and gui_state.label_start != -1:
        color_hex = gui_state.label_behavior_colors[gui_state.label_type].lstrip("#")
        bgr_color = tuple(int(color_hex[i:i+2], 16) for i in (4, 2, 0))
        start_f, end_f = min(gui_state.label_start, gui_state.label_index), max(gui_state.label_start, gui_state.label_index)
        start_px, end_px = int(timeline_w * start_f / total_frames), int(timeline_w * (end_f + 1) / total_frames)
        if start_px < end_px:
            cv2.rectangle(canvas_img, (start_px, timeline_y), (end_px, timeline_y + timeline_h - 1), bgr_color, -1)
            cv2.rectangle(canvas_img, (start_px, timeline_y), (end_px, timeline_y + timeline_h - 1), (255, 255, 255), 1)

    return canvas_img

def update_counts():
    """Updates instance/frame counts in the UI based on the current session buffer."""
    if not gui_state.label_dataset: return

    if gui_state.label_videos and gui_state.label_vid_index >= 0:
        rel_path = os.path.relpath(gui_state.label_videos[gui_state.label_vid_index], start=gui_state.proj.path)
        eel.updateFileInfo(rel_path)()
    else:
        eel.updateFileInfo("No video loaded.")()

    behaviors = gui_state.label_dataset.labels.get("behaviors", [])
    counts = {b: {'instances': 0, 'frames': 0} for b in behaviors}
    
    for inst in gui_state.label_session_buffer:
        b_name = inst.get('label')
        if b_name in counts:
            counts[b_name]['instances'] += 1
            counts[b_name]['frames'] += (inst.get("end", 0) - inst.get("start", 0) + 1)
    
    for b_name, data in counts.items():
        eel.updateLabelingStats(b_name, data['instances'], data['frames'])()