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
from collections import defaultdict

# Project-specific imports
import cbas
import classifier_head
import gui_state
import workthreads
from workthreads import log_message
from cmap import Colormap

import eel
import sys
import subprocess
import threading
import time

# =================================================================
# HELPER FUNCTIONS
# =================================================================

def _video_import_worker(session_name: str, subject_name: str, video_paths: list[str], standardize: bool, crop_data: dict):
    """
    (WORKER) Runs in a separate thread. Copies or standardizes/crops imported videos
    before queuing them for encoding.
    """
    try:
        workthreads.log_message(f"Starting import: Processing {len(video_paths)} videos for session '{session_name}'...", "INFO")

        session_dir = os.path.join(gui_state.proj.recordings_dir, session_name)
        final_dest_dir = os.path.join(session_dir, subject_name)
        
        os.makedirs(final_dest_dir, exist_ok=True)
        
        files_for_queue = []
        for video_path in video_paths:
            try:
                basename = os.path.basename(video_path)
                dest_path = os.path.join(final_dest_dir, basename)

                command = ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y', '-i', video_path]
                
                video_filters = []
                
                if crop_data and crop_data.get('apply', False):
                    crop_w = crop_data.get('w', 1.0)
                    crop_h = crop_data.get('h', 1.0)
                    crop_x = crop_data.get('x', 0.0)
                    crop_y = crop_data.get('y', 0.0)
                    if not (crop_w == 1.0 and crop_h == 1.0 and crop_x == 0.0 and crop_y == 0.0):
                        video_filters.append(f"crop=iw*{crop_w}:ih*{crop_h}:iw*{crop_x}:ih*{crop_y}")

                if standardize:
                    video_filters.append('fps=10')
                    
                    should_stretch = crop_data.get('stretch', False)
                    if should_stretch:
                        video_filters.append('scale=256:256')
                    else:
                        video_filters.append('scale=256:256:force_original_aspect_ratio=decrease,pad=256:256:(ow-iw)/2:(oh-ih)/2')

                if video_filters:
                    command.extend(['-vf', ",".join(video_filters)])

                command.extend(['-an', dest_path])
                
                is_cropping = any('crop' in f for f in video_filters)
                
                if not video_filters and not standardize:
                    workthreads.log_message(f"Copying '{basename}' without changes...", "INFO")
                    shutil.copy(video_path, dest_path)
                else:
                    action_str = "Cropping and standardizing" if is_cropping and standardize else "Cropping" if is_cropping else "Standardizing"
                    workthreads.log_message(f"{action_str} '{basename}'...", "INFO")
                    
                    creation_flags = 0
                    if sys.platform == "win32":
                        creation_flags = subprocess.CREATE_NO_WINDOW
                    subprocess.run(command, check=True, capture_output=True, text=True, creationflags=creation_flags)
                
                files_for_queue.append(dest_path)

            except subprocess.CalledProcessError as cpe:
                workthreads.log_message(f"Could not process '{os.path.basename(video_path)}'. FFMPEG Error: {cpe.stderr}", "ERROR")
            except Exception as e:
                workthreads.log_message(f"An unexpected error occurred while processing '{os.path.basename(video_path)}'. Skipping. Error: {e}", "ERROR")

        if files_for_queue:
            with gui_state.encode_lock:
                new_files_to_queue = [f for f in files_for_queue if f not in gui_state.encode_tasks]
                gui_state.encode_tasks.extend(new_files_to_queue)
            workthreads.log_message(f"Queued {len(new_files_to_queue)} files for encoding.", "INFO")
        
        if workthreads.file_watcher_handler:
            with workthreads.file_watcher_handler.pending_files_lock:
                workthreads.file_watcher_handler.pending_files.clear()
            workthreads.log_message("Cleared automatic file watcher queue to prevent duplicate encoding.", "INFO")

        success_msg_action = "imported"
        if crop_data and crop_data.get('apply', False) and not (crop_data.get('w') == 1.0 and crop_data.get('h') == 1.0 and crop_data.get('x') == 0.0 and crop_data.get('y') == 0.0):
            success_msg_action += " and cropped"
        if standardize:
            success_msg_action += " and standardized"
            
        workthreads.log_message(f"Successfully {success_msg_action} {len(files_for_queue)} video(s).", "INFO")
        eel.notify_import_complete(True, f"Successfully {success_msg_action} {len(files_for_queue)} video(s) to session '{session_name}' under subject '{subject_name}'.")

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
    remap = {7: 6, 14: 2, 15: 4}
    if val in remap:
        return remap[val]
    return (val * 2) if val < 10 else ((val - 10) * 2 + 1)


# =================================================================
# EEL-EXPOSED FUNCTIONS: DATA & CONFIGURATION
# =================================================================

def check_dataset_files_ready(dataset_name: str) -> tuple[bool, str]:
    """
    Performs a pre-flight check to ensure all H5 files for a dataset exist.
    This prevents training from starting before the EncodeThread is finished.
    """
    if not gui_state.proj or dataset_name not in gui_state.proj.datasets:
        return False, f"Dataset '{dataset_name}' not found."

    try:
        dataset = gui_state.proj.datasets[dataset_name]
        with open(dataset.labels_path, 'r') as f:
            labels_data = yaml.safe_load(f)

        all_video_paths = set(
            inst['video']
            for behavior_labels in labels_data.get('labels', {}).values()
            for inst in behavior_labels if 'video' in inst
        )

        if not all_video_paths:
            return False, "The dataset contains no labeled video instances."

        missing_files = []
        for rel_video_path in all_video_paths:
            abs_video_path = os.path.join(gui_state.proj.path, rel_video_path)
            h5_path = os.path.splitext(abs_video_path)[0] + "_cls.h5"
            if not os.path.exists(h5_path):
                missing_files.append(os.path.basename(rel_video_path))
        
        if not missing_files:
            return True, "All necessary files are ready for training."
        else:
            error_msg = (
                f"{len(missing_files)} of {len(all_video_paths)} required "
                f"feature files (.h5) are still missing."
            )
            if len(missing_files) <= 3:
                error_msg += f"\nMissing: {', '.join(missing_files)}"
            else:
                error_msg += f"\nMissing: {', '.join(missing_files[:3])}, and others..."
            
            return False, error_msg

    except Exception as e:
        print(f"Error checking dataset files for '{dataset_name}': {e}")
        return False, f"An unexpected error occurred: {e}"

def model_exists(model_name: str) -> bool:
    """Checks if a model with the given name exists in the current project."""
    return gui_state.proj and model_name in gui_state.proj.models



def load_dataset_configs() -> dict:
    """
    Loads configurations for all available datasets and determines their current state
    ('new', 'labeled', 'trained').
    """
    if not gui_state.proj:
        return {}
        
    dataset_configs = {}
    for name, dataset in gui_state.proj.datasets.items():
        config = dataset.config.copy()
        
        # The state is determined by finding the associated model.
        # First, check the config file for a direct link to the model.
        model_name = config.get("model") # e.g., "cbas_aug_model"
        
        # If a model name is linked, check if that model actually exists.
        if model_name and model_name in gui_state.proj.models:
            model_path = gui_state.proj.models[model_name].weights_path
            if os.path.exists(model_path):
                config['state'] = 'trained'
            else:
                # The link is broken, treat as labeled
                config['state'] = 'labeled'
        else:
            # No model is linked, check if it has labels.
            has_labels = False
            if dataset.labels and "labels" in dataset.labels:
                for behavior_name in dataset.labels["labels"]:
                    if dataset.labels["labels"][behavior_name]:
                        has_labels = True
                        break
            
            if has_labels:
                config['state'] = 'labeled'
            else:
                config['state'] = 'new'

        disagreement_report_path = os.path.join(dataset.path, "disagreement_report.yaml")
        config['has_disagreements'] = os.path.exists(disagreement_report_path)

        dataset_configs[name] = config
        
    return dataset_configs



def get_available_models() -> list[str]:
    """Returns a sorted list of all model names available in the project."""
    if not gui_state.proj: return []
    return sorted(list(gui_state.proj.models.keys()))



def get_record_tree() -> dict:
    """Fetches the recording directory tree structure for modal dialogs."""
    if not gui_state.proj or not os.path.exists(gui_state.proj.recordings_dir):
        return {}
    tree = {}
    try:
        for session_dir in os.scandir(gui_state.proj.recordings_dir):
            if session_dir.is_dir():
                subjects = [subject.name for subject in os.scandir(session_dir.path) if subject.is_dir()]
                if subjects:
                    tree[session_dir.name] = subjects
    except Exception as e:
        print(f"Error building record tree: {e}")
    return tree

def get_hierarchical_video_list(dataset_name: str) -> dict:
    """
    Scans the filesystem based on a dataset's whitelist and returns a nested
    dictionary representing the Session -> Subject -> [Videos] hierarchy.
    """
    if not gui_state.proj: return {}
    dataset = gui_state.proj.datasets.get(dataset_name)
    if not dataset: return {}
    
    whitelist = dataset.config.get("whitelist", [])
    if not whitelist: return {}

    recordings_root = gui_state.proj.recordings_dir
    absolute_whitelist_paths = [os.path.normpath(os.path.join(recordings_root, p)) for p in whitelist]
    
    video_hierarchy = defaultdict(lambda: defaultdict(list))

    if recordings_root and os.path.exists(recordings_root):
        # Use a dictionary to group videos by their directory to avoid re-walking
        videos_by_dir = defaultdict(list)
        for root, _, files in os.walk(recordings_root):
            for file in files:
                if file.endswith('.mp4'):
                    videos_by_dir[root].append(file)
        
        for root, files in videos_by_dir.items():
            norm_root = os.path.normpath(root)
            
            # Check if this directory is within a whitelisted path
            if not any(norm_root.startswith(wl_path) for wl_path in absolute_whitelist_paths):
                continue

            # Identify session and subject based on directory structure
            try:
                # Assumes structure is .../recordings_root/session/subject
                rel_path = os.path.relpath(norm_root, recordings_root)
                parts = rel_path.split(os.sep)
                if len(parts) >= 2:
                    session_name, subject_name = parts[0], parts[1]
                else:
                    continue # Skip directories that don't match the expected structure
            except ValueError:
                continue

            video_files_in_dir = []
            file_set = set(files)
            for file in sorted(files):
                if file.endswith(".mp4"):
                    is_genuinely_augmented = False
                    if file.endswith("_aug.mp4"):
                        source_filename = file[:-8] + ".mp4"
                        if source_filename in file_set:
                            is_genuinely_augmented = True
                    
                    if not is_genuinely_augmented:
                        video_path = os.path.join(root, file)

                        video_files_in_dir.append((video_path, file))
            
            if video_files_in_dir:
                video_hierarchy[session_name][subject_name].extend(video_files_in_dir)

    # Convert defaultdicts to regular dicts for clean JSON transfer
    final_structure = {sess: {subj: vids for subj, vids in subjects.items()} for sess, subjects in video_hierarchy.items()}
    return final_structure

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
                # Only add the video if it is NOT an augmented file.
                if not mp4_path.endswith("_aug.mp4"):
                    ready_videos.append((mp4_path, os.path.basename(mp4_path)))
                
    return sorted(ready_videos, key=lambda x: x[1])



def get_existing_session_names() -> list[str]:
    """
    Scans the project's recordings directory and returns a sorted list of all
    top-level session folders.
    """
    if not gui_state.proj or not os.path.isdir(gui_state.proj.recordings_dir):
        return []
    
    try:
        session_names = [d.name for d in os.scandir(gui_state.proj.recordings_dir) if d.is_dir()]
        return sorted(session_names)
    except Exception as e:
        print(f"Error scanning for session names: {e}")
        return []



def import_videos(session_name: str, subject_name: str, video_paths: list[str], standardize: bool, crop_data: dict):
    """
    (LAUNCHER) Spawns the video import process in a background thread,
    passing the user's choice for standardization.
    """
    if not gui_state.proj or not session_name or not subject_name or not video_paths:
        return False
    
    print(f"Spawning background thread to import {len(video_paths)} video(s)...")
    
    import_thread = threading.Thread(
        target=_video_import_worker,
        args=(session_name, subject_name, video_paths, standardize, crop_data)
    )
    
    import_thread.daemon = True
    import_thread.start()
    
    return True

# =================================================================
# EEL-EXPOSED FUNCTIONS: LABELING WORKFLOW & ACTIONS
# =================================================================

def get_label_coverage_report(dataset_name: str) -> dict:
    """
    Analyzes a dataset's labels to see which subjects are missing labels
    for which behaviors.
    """
    if not gui_state.proj or dataset_name not in gui_state.proj.datasets:
        return {"error": "Dataset not found."}

    dataset = gui_state.proj.datasets[dataset_name]
    
    try:
        with open(dataset.labels_path, 'r') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        return {"error": f"Could not read or parse labels.yaml: {e}"}

    master_behaviors = set(data.get("behaviors", []))
    if not master_behaviors:
        return {"error": "No behaviors defined in the dataset."}

    subject_behaviors = defaultdict(set)
    all_instances = [
        inst for behavior_list in data.get("labels", {}).values() for inst in behavior_list
    ]

    for inst in all_instances:
        video_path = inst.get("video")
        label = inst.get("label")
        if video_path and label:
            try:

                # The unique identifier for a subject is its full relative path,
                # not just its folder name.
                subject_path = os.path.dirname(video_path)
                
                # Normalize slashes for consistent display
                subject_path = subject_path.replace('\\', '/')

                subject_behaviors[subject_path].add(label)

            except Exception:
                continue

    if not subject_behaviors:
        return {"error": "No labeled instances found in the dataset."}

    report = {
        "master_behavior_list": sorted(list(master_behaviors)),
        "complete_subjects": [],
        "incomplete_subjects": []
    }

    for subject_path, behaviors in sorted(subject_behaviors.items()):
        missing_behaviors = master_behaviors - behaviors
        if not missing_behaviors:
            report["complete_subjects"].append({
                "name": subject_path, # Use the full path as the name
                "count": len(behaviors)
            })
        else:
            report["incomplete_subjects"].append({
                "name": subject_path, # Use the full path as the name
                "count": len(behaviors),
                "missing": sorted(list(missing_behaviors))
            })
            
    return report

def video_has_labels(dataset_name: str, video_path: str) -> bool:
    """Checks if a specific video has any existing labels in a given dataset."""
    if not gui_state.proj: return False
    dataset = gui_state.proj.datasets.get(dataset_name)
    if not dataset or not dataset.labels: return False
    
    # We need the relative path for comparison with what's in labels.yaml
    relative_video_path = os.path.relpath(video_path, start=gui_state.proj.path).replace('\\', '/')

    for behavior, instances in dataset.labels.get("labels", {}).items():
        for instance in instances:
            if instance.get("video") == relative_video_path:
                return True # Found at least one label, we can stop early
    return False

def get_model_configs() -> dict:
    """Loads configurations for all available models."""
    if not gui_state.proj:
        return {}
    return {name: model.config for name, model in gui_state.proj.models.items()}

def _start_labeling_worker(name: str, video_to_open: str = None, preloaded_instances: list = None, probability_df: pd.DataFrame = None, filter_for_behavior: str = None):
    """
    (WORKER) This is the background task that prepares the labeling session. It handles
    all state setup and then calls back to the JavaScript UI when ready.
    """
    try:
        gui_state.label_filter_for_behavior = filter_for_behavior
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
        gui_state.label_dirty_instances.clear()
        gui_state.label_session_buffer, gui_state.selected_instance_index = [], -1
        gui_state.label_probability_df = probability_df
        gui_state.label_confirmation_mode = False
        gui_state.label_confidence_threshold = 100
        
        eel.setConfirmationModeUI(False)()
        
        # 1. Force a reload of the specific dataset object from disk.
        gui_state.proj.datasets[name] = cbas.Dataset(gui_state.proj.datasets[name].path)
        dataset: cbas.Dataset = gui_state.proj.datasets[name]
        gui_state.label_dataset = dataset
               
        gui_state.label_col_map = Colormap("seaborn:tab20")
        print("Loading session buffer...")
        gui_state.label_videos = [video_to_open]
        
        relative_video_path = os.path.relpath(video_to_open, start=gui_state.proj.path).replace('\\', '/')

        # =========================================================================
        # ROBUST LABEL MERGING LOGIC
        # =========================================================================

        # Step 1: Load all existing human-verified labels for this video.
        # These are automatically kept and form the "base layer" of truth.
        human_labels = []
        for b_name, b_insts in gui_state.label_dataset.labels["labels"].items():
            for inst in b_insts:
                if inst.get("video") == relative_video_path:
                    human_labels.append(inst)
        
        # Add the trusted human labels to the session buffer immediately.
        gui_state.label_session_buffer.extend(human_labels)
        print(f"Loaded {len(human_labels)} existing human labels for '{relative_video_path}' into buffer.")

        labeling_mode = 'scratch'
        model_name_for_ui = ''

        # Step 2: If this is a guided session, process the model's predictions.
        if preloaded_instances:
            labeling_mode = 'review'
            model_name_for_ui = name 

            gui_state.label_unfiltered_instances = preloaded_instances.copy()
            
            initial_threshold = gui_state.label_confidence_threshold / 100.0
            filtered_predictions = [
                p for p in preloaded_instances if p.get('confidence', 1.0) < initial_threshold
            ]
            
            print(f"Processing {len(filtered_predictions)} initially filtered model predictions.")

            # Step 2a: Build the "Time Map" of occupied zones from human labels.
            human_intervals = sorted([(h['start'], h['end']) for h in human_labels])

            # Step 2b: Process each prediction against the time map.
            for pred_inst in filtered_predictions:
                # A prediction starts as a single continuous interval.
                pred_intervals_to_add = [(pred_inst['start'], pred_inst['end'])]

                # Check this prediction against every known human-occupied zone.
                for h_start, h_end in human_intervals:
                    # This list will hold the pieces of the prediction that survive the check.
                    surviving_pieces = []
                    
                    # For each piece of our current prediction, see if it overlaps.
                    while pred_intervals_to_add:
                        p_start, p_end = pred_intervals_to_add.pop(0)
                        
                        # No overlap if prediction ends before human label starts,
                        # or prediction starts after human label ends.
                        is_safe = p_end < h_start or p_start > h_end
                        
                        if is_safe:
                            surviving_pieces.append((p_start, p_end))
                            continue
                        
                        # If there is an overlap, create the non-overlapping pieces.
                        # Piece 1: The part of the prediction before the human label.
                        if p_start < h_start:
                            surviving_pieces.append((p_start, h_start - 1))
                        
                        # Piece 2: The part of the prediction after the human label.
                        if p_end > h_end:
                            surviving_pieces.append((h_end + 1, p_end))

                    # The surviving pieces become the new intervals to check against the *next* human label.
                    pred_intervals_to_add = surviving_pieces
                
                # After checking against all human labels, add any surviving pieces to the session buffer.
                for start, end in pred_intervals_to_add:
                    if start <= end: # Ensure the piece is valid
                        new_inst = pred_inst.copy()
                        new_inst['start'] = start
                        new_inst['end'] = end
                        gui_state.label_session_buffer.append(new_inst)
        
        dataset_behaviors = gui_state.label_dataset.labels.get("behaviors", [])
        behavior_colors = [str(gui_state.label_col_map(tab20_map(i))) for i in range(len(dataset_behaviors))]
        
        # Populate the new, safe state variables
        gui_state.label_session_behaviors = dataset_behaviors
        gui_state.label_session_colors = behavior_colors

        print("Setup complete. Calling frontend to build UI.")
        eel.buildLabelingUI(dataset_behaviors, behavior_colors, filter_for_behavior)()
        eel.setLabelingModeUI(labeling_mode, model_name_for_ui)()
        eel.setConfirmationModeUI(False)()

        if not gui_state.label_videos: raise ValueError("Video list is empty after setup.")
        print("Loading video and pushing initial frame...")
        next_video(0)

    except Exception as e:
        print(f"FATAL ERROR in labeling worker: {e}")
        traceback.print_exc()
        eel.showErrorOnLabelTrainPage(f"Failed to start labeling session: {e}")()

def get_instances_for_behavior(dataset_name: str, behavior_name: str) -> dict:
    """
    Finds all instances of a specific behavior in a dataset and groups them by video.
    """
    if not gui_state.proj or dataset_name not in gui_state.proj.datasets:
        return {}

    dataset = gui_state.proj.datasets[dataset_name]
    # Ensure we have the latest labels from disk
    with open(dataset.labels_path, 'r') as f:
        labels_data = yaml.safe_load(f)

    instances_by_video = {}
    
    # Find the list of instances for the requested behavior
    all_instances_for_behavior = labels_data.get("labels", {}).get(behavior_name, [])

    for instance in all_instances_for_behavior:
        video_path = instance.get("video")
        if not video_path:
            continue

        # Group by the video path
        if video_path not in instances_by_video:
            instances_by_video[video_path] = {
                "instance_count": 0,
                "display_name": video_path.replace('\\', '/') # Standardize slashes for display
            }
        
        instances_by_video[video_path]["instance_count"] += 1
    
    # Sort the results by display name for a consistent UI
    sorted_videos = sorted(instances_by_video.items(), key=lambda item: item[1]['display_name'])
    
    return dict(sorted_videos)


def update_dataset_whitelist(dataset_name: str, new_whitelist: list[str]) -> bool:
    """
    Updates the 'whitelist' key in a dataset's config.yaml file.
    """
    if not gui_state.proj or dataset_name not in gui_state.proj.datasets:
        print(f"Error: Could not find dataset '{dataset_name}' to update.")
        return False
    
    try:
        dataset = gui_state.proj.datasets[dataset_name]
        # Update the whitelist in the in-memory config
        dataset.config['whitelist'] = new_whitelist
        
        # Write the entire updated config back to the file
        with open(dataset.config_path, 'w') as f:
            yaml.dump(dataset.config, f, allow_unicode=True)
        
        workthreads.log_message(f"Updated labeled directories for dataset '{dataset_name}'.", "INFO")
        return True
    except Exception as e:
        workthreads.log_message(f"Failed to update whitelist for '{dataset_name}': {e}", "ERROR")
        traceback.print_exc()
        return False

def start_labeling(name: str, video_to_open: str = None, preloaded_instances: list = None, filter_for_behavior: str = None) -> bool:
    """
    Directly prepares the labeling session and returns a success flag.
    This is now a synchronous function from the frontend's perspective.
    """
    try:
        # We now call the worker's logic directly instead of spawning it.
        # The eel.sleep(0) calls allow the UI to remain responsive during setup.
        eel.sleep(0.01)
        _start_labeling_worker(name, video_to_open, preloaded_instances, None, filter_for_behavior)
        eel.sleep(0.01)
        return True
    except Exception as e:
        print(f"Failed to start labeling session: {e}")
        traceback.print_exc()
        # Ensure an error is reported back to the user if setup fails.
        eel.showErrorOnLabelTrainPage(f"Failed to start labeling session: {e}")()
        return False

def analyze_label_conflicts(dataset_name: str) -> dict:
    """
    Performs a 'dry run' analysis of a labels.yaml file to find duplicates
    and overlaps without modifying the file.
    """
    if not gui_state.proj or dataset_name not in gui_state.proj.datasets:
        return {"error": "Dataset not found."}

    dataset = gui_state.proj.datasets[dataset_name]
    labels_file_path = dataset.labels_path

    if not os.path.exists(labels_file_path):
        return {"error": "Labels file not found."}

    try:
        with open(labels_file_path, 'r') as f:
            data = yaml.safe_load(f)
    except Exception as e:
        return {"error": f"Could not parse YAML file: {e}"}

    # --- 1. Find Exact Duplicates ---
    total_duplicates = 0
    seen_instances = set()
    all_instances = []
    for behavior, instances in data.get("labels", {}).items():
        if not isinstance(instances, list): continue
        for instance in instances:
            instance_tuple = (
                instance.get("start"), instance.get("end"),
                instance.get("label"), instance.get("video")
            )
            if instance_tuple in seen_instances:
                total_duplicates += 1
            else:
                seen_instances.add(instance_tuple)
                # Ensure start/end are floats for consistency in the next step
                instance['start'] = float(instance['start'])
                instance['end'] = float(instance['end'])
                all_instances.append(instance)

    # --- 2. Find Overlaps ---
    total_overlaps = 0
    instances_by_video = defaultdict(list)
    for inst in all_instances:
        instances_by_video[inst.get('video')].append(inst)

    for video_path, instances in instances_by_video.items():
        if len(instances) < 2: continue
        instances.sort(key=lambda x: x['start'])
        
        for i in range(len(instances) - 1):
            # An overlap occurs if the start of the next instance is before the end of the current one
            if instances[i+1]['start'] <= instances[i]['end']:
                total_overlaps += 1
    
    return {
        "total_duplicates": total_duplicates,
        "total_overlaps": total_overlaps
    }


def clean_and_sort_labels(dataset_name: str) -> bool:
    """
    Performs a full cleanup, normalization, and sort of a labels.yaml file.
    This is a non-destructive operation that resolves conflicts by trimming.
    """
    if not gui_state.proj or dataset_name not in gui_state.proj.datasets:
        return False

    dataset = gui_state.proj.datasets[dataset_name]
    labels_file_path = dataset.labels_path

    try:
        with open(labels_file_path, 'r') as f:
            data = yaml.safe_load(f)
    except Exception:
        return False
    
    # --- 1. De-duplication ---
    seen_instances = set()
    all_instances = []
    for behavior, instances in data.get("labels", {}).items():
        if not isinstance(instances, list): continue
        for instance in instances:
            instance_tuple = (
                instance.get("start"), instance.get("end"),
                instance.get("label"), instance.get("video")
            )
            if instance_tuple not in seen_instances:
                seen_instances.add(instance_tuple)
                instance['start'] = float(instance['start'])
                instance['end'] = float(instance['end'])
                all_instances.append(instance)

    # --- 2. De-confliction (Merge/Trim) ---
    instances_by_video = defaultdict(list)
    for inst in all_instances:
        instances_by_video[inst.get('video')].append(inst)

    final_clean_instances = []
    for video_path, instances in instances_by_video.items():
        if len(instances) < 2:
            final_clean_instances.extend(instances)
            continue

        instances.sort(key=lambda x: (x['start'], -(x['end'] - x['start'])))
        
        deconflicted = []
        for new_inst in instances:
            pieces_to_add = [new_inst]
            for existing_inst in deconflicted:
                next_pieces = []
                while pieces_to_add:
                    piece = pieces_to_add.pop(0)
                    p_start, p_end = piece['start'], piece['end']
                    e_start, e_end = existing_inst['start'], existing_inst['end']

                    if max(p_start, e_start) <= min(p_end, e_end):
                        if piece['label'] == existing_inst['label']:
                            continue
                        
                        if p_start < e_start:
                            next_pieces.append({**piece, 'end': e_start - 1})
                        
                        if p_end > e_end:
                            next_pieces.append({**piece, 'start': e_end + 1})
                    else:
                        next_pieces.append(piece)
                
                pieces_to_add = next_pieces

            for piece in pieces_to_add:
                if piece['start'] <= piece['end']:
                    deconflicted.append(piece)
        
        deconflicted.sort(key=lambda x: (x['label'], x['start']))
        
        if not deconflicted: continue
        
        merged_instances = [deconflicted[0]]
        for i in range(1, len(deconflicted)):
            current_inst = deconflicted[i]
            last_merged = merged_instances[-1]
            
            if current_inst['label'] == last_merged['label'] and current_inst['start'] <= last_merged['end'] + 1:
                last_merged['end'] = max(last_merged['end'], current_inst['end'])
            else:
                merged_instances.append(current_inst)

        final_clean_instances.extend(merged_instances)

    # --- 3. Normalize all path separators to forward slashes ---
    for inst in final_clean_instances:
        if 'video' in inst and isinstance(inst['video'], str):
            inst['video'] = inst['video'].replace('\\', '/')

    # --- 4. Sort the final, clean list for readability ---
    final_clean_instances.sort(key=lambda x: (
        x.get('label', ''), 
        x.get('video', ''), 
        x.get('start', 0)
    ))

    # --- 5. Rebuild the final YAML structure ---
    cleaned_data = data.copy()
    cleaned_data["labels"] = defaultdict(list)
    for inst in final_clean_instances:
        inst.pop('_confirmed', None)
        cleaned_data["labels"][inst['label']].append(inst)
    
    final_labels_dict = {k: v for k, v in sorted(cleaned_data["labels"].items())}
    cleaned_data["labels"] = final_labels_dict

    # --- 6. Save the cleaned and sorted data back to the file ---
    try:
        with open(labels_file_path, 'w') as f:
            yaml.dump(cleaned_data, f, allow_unicode=True, sort_keys=False)
        log_message(f"Successfully cleaned and sorted labels for '{dataset_name}'.", "INFO")
        return True
    except Exception as e:
        log_message(f"ERROR: Could not write cleaned labels file for '{dataset_name}': {e}", "ERROR")
        return False

def start_labeling_with_preload(dataset_name: str, model_name: str, video_path_to_label: str, smoothing_window: int) -> bool:
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
        
        # Smart Loader Logic for Guided Labeling
        arch_type = model_obj.config.get('architecture', 'lstm_legacy')
        log_message(f"Guided Labeling using model with architecture: '{arch_type}'", "INFO")

        if arch_type == 'lstm_with_deltas':
            torch_model = classifier_head.ClassifierLSTMDeltas(
                in_features=768,
                out_features=len(model_obj.config["behaviors"]),
                seq_len=model_obj.config["seq_len"],
            ).to(device)
        else: # Handles legacy models
            torch_model = classifier_head.ClassifierLegacyLSTM(
                in_features=768,
                out_features=len(model_obj.config["behaviors"]),
                seq_len=model_obj.config["seq_len"],
            ).to(device)
           
        torch_model.load_state_dict(torch.load(model_obj.weights_path, map_location=device, weights_only=True))
        torch_model.eval()

        h5_path = os.path.splitext(video_path_to_label)[0] + "_cls.h5"
        if not os.path.exists(h5_path): raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
        
        csv_path = cbas.infer_file(
            file_path=h5_path, model=torch_model, dataset_name=model_obj.name,
            behaviors=model_obj.config["behaviors"], seq_len=model_obj.config["seq_len"], device=device
        )
        if not csv_path or not os.path.exists(csv_path):
            raise RuntimeError("Inference failed to produce a CSV output file.")

        preloaded_instances, probability_df = dataset.predictions_to_instances_with_confidence(
            csv_path,
            model_obj.name,
            smoothing_window=smoothing_window # Pass it to the modified cbas.py function
        )
        
        eel.spawn(_start_labeling_worker, dataset_name, video_path_to_label, preloaded_instances, probability_df)
        
        print(f"Spawned pre-labeling worker for video '{os.path.basename(video_path_to_label)}'.")
        return True

    except Exception as e:
        print(f"ERROR in start_labeling_with_preload: {e}")
        eel.showErrorOnLabelTrainPage(f"Failed to start pre-labeling: {e}")()
        return False

def save_session_labels():
    """
    Saves labels from the session buffer. The buffer is now always the complete
    source of truth for the video. This function filters for confirmed/human
    labels and overwrites the file for the specific video being edited.
    """
    if not gui_state.label_dataset or not gui_state.label_videos:
        return {'status': 'error', 'message': 'Labeling session not active.'}

    current_video_path_abs = gui_state.label_videos[0]
    current_video_path_rel = os.path.relpath(current_video_path_abs, start=gui_state.proj.path).replace('\\', '/')
    
    # Capture the dataset name before doing anything else.
    saved_dataset_name = gui_state.label_dataset.name

    # --- 1. Filter the buffer for the final ground truth to be committed ---
    final_commit_list = [
        inst for inst in gui_state.label_session_buffer
        if 'confidence' not in inst or inst.get('_confirmed', False)
    ]

    if not gui_state.label_dirty_instances and not any(inst.get('_confirmed') for inst in gui_state.label_session_buffer):
         workthreads.log_message("No changes detected in labeling session. Nothing to save.", "INFO")
         return {'status': 'no_changes'}

    workthreads.log_message(f"Saving {len(final_commit_list)} labels for {current_video_path_rel}.", "INFO")

    # --- 2. Load the master label file ---
    with open(gui_state.label_dataset.labels_path, "r") as f:
        master_labels = yaml.safe_load(f)

    # --- 3. "Scorched Earth" removal for the specific video ---
    for behavior in master_labels["labels"]:
        master_labels["labels"][behavior] = [
            inst for inst in master_labels["labels"].get(behavior, [])
            if inst.get("video") != current_video_path_rel
        ]

    # --- 4. Add back the new, complete set of labels from the commit list ---
    for final_inst in final_commit_list:
        inst_to_save = final_inst.copy()
        for key in ['confidence', 'confidences', '_original_start', '_original_end', '_confirmed']:
            inst_to_save.pop(key, None)
        master_labels["labels"].setdefault(inst_to_save['label'], []).append(inst_to_save)

    # --- 5. Write the updated data back to the file ---
    with open(gui_state.label_dataset.labels_path, "w") as file:
        yaml.dump(master_labels, file, allow_unicode=True)

    # --- 6. Finalize and Update UI ---
    try:
        gui_state.label_dataset.update_instance_counts_in_config(gui_state.proj)
    except Exception as e:
        workthreads.log_message(f"Could not update instance counts after saving: {e}", "ERROR")

    workthreads.log_message(f"Successfully saved labels for {os.path.basename(current_video_path_abs)}.", "INFO")

    gui_state.label_confirmation_mode = False
    eel.setConfirmationModeUI(False)()
    render_image()

    # Return the status, video path, AND the name of the dataset that was saved.
    return {'status': 'success', 'video_path': current_video_path_rel, 'dataset_name': saved_dataset_name}
    
def refilter_instances(new_threshold: int, mode: str = 'below'):
    """
    Re-filters the session buffer based on a confidence threshold and mode ('above' or 'below'),
    ensuring no overlaps with existing human labels.
    """
    print(f"Refiltering instances with mode '{mode}' and threshold {new_threshold}%")
    gui_state.label_confidence_threshold = new_threshold
    
    # =========================================================================
    # ROBUST RE-FILTERING LOGIC
    # =========================================================================

    # Step 1: Get the two master lists: all raw predictions and all human labels.
    unfiltered_predictions = gui_state.label_unfiltered_instances
    
    # It's crucial to get the human labels from the original dataset object,
    # not the potentially modified session buffer, to ensure a clean slate.
    human_labels = []
    if gui_state.label_dataset and gui_state.label_videos:
        relative_video_path = os.path.relpath(gui_state.label_videos[0], start=gui_state.proj.path).replace('\\', '/')
        for b_name, b_insts in gui_state.label_dataset.labels["labels"].items():
            for inst in b_insts:
                if inst.get("video") == relative_video_path:
                    human_labels.append(inst)

    # Step 2: Apply the new confidence filter to the raw predictions.
    threshold_float = new_threshold / 100.0
    newly_filtered_candidates = []
    if mode == 'above':
        newly_filtered_candidates = [
            p for p in unfiltered_predictions if p.get('confidence', 0.0) >= threshold_float
        ]
    else: # Default to 'below'
        newly_filtered_candidates = [
            p for p in unfiltered_predictions if p.get('confidence', 1.0) < threshold_float
        ]
    
    # Step 3: Run these new candidates through the same robust "Time Map" logic.
    
    # Step 3a: Build the "Time Map" of occupied zones from human labels.
    human_intervals = sorted([(h['start'], h['end']) for h in human_labels])
    
    final_guided_labels = []
    # Step 3b: Process each candidate prediction against the time map.
    for pred_inst in newly_filtered_candidates:
        pred_intervals_to_add = [(pred_inst['start'], pred_inst['end'])]
        for h_start, h_end in human_intervals:
            surviving_pieces = []
            while pred_intervals_to_add:
                p_start, p_end = pred_intervals_to_add.pop(0)
                is_safe = p_end < h_start or p_start > h_end
                if is_safe:
                    surviving_pieces.append((p_start, p_end))
                    continue
                if p_start < h_start:
                    surviving_pieces.append((p_start, h_start - 1))
                if p_end > h_end:
                    surviving_pieces.append((h_end + 1, p_end))
            pred_intervals_to_add = surviving_pieces
        
        for start, end in pred_intervals_to_add:
            if start <= end:
                new_inst = pred_inst.copy()
                new_inst['start'] = start
                new_inst['end'] = end
                final_guided_labels.append(new_inst)

    # Step 4: Rebuild the session buffer from the clean lists.
    gui_state.label_session_buffer = human_labels + final_guided_labels
    
    # Reset selection and re-render the UI with the new, clean data.
    gui_state.selected_instance_index = -1
    eel.highlightBehaviorRow(None)()
    eel.updateConfidenceBadge(None, None)()
    render_image()
    update_counts()

# =================================================================
# EEL-EXPOSED FUNCTIONS: IN-SESSION LABELING ACTIONS
# =================================================================

def get_current_labeling_video_path() -> str | None:
    """
    Returns the relative path of the video currently loaded in the labeling interface.
    """
    if not gui_state.proj or not gui_state.label_videos or gui_state.label_vid_index == -1:
        return None
    
    try:
        absolute_path = gui_state.label_videos[gui_state.label_vid_index]
        relative_path = os.path.relpath(absolute_path, start=gui_state.proj.path).replace('\\', '/')
        return relative_path
    except Exception:
        return None

def get_frame_from_video(video_path: str) -> str | None:
    """Extracts the first frame of a video and returns it as a base64 string."""
    if not os.path.exists(video_path):
        print(f"Video path not found for frame extraction: {video_path}")
        return None
    try:
        command = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error',
            '-i', video_path,
            '-vframes', '1',
            '-f', 'image2pipe',
            '-c:v', 'mjpeg',
            '-'
        ]
        
        creation_flags = 0
        if sys.platform == "win32":
            creation_flags = subprocess.CREATE_NO_WINDOW

        result = subprocess.run(command, capture_output=True, check=True, timeout=15, creationflags=creation_flags)
        
        return base64.b64encode(result.stdout).decode('utf-8')
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError) as e:
        print(f"Error extracting frame from {video_path}: {e}")
        return None

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



def handle_click_on_label_image(x: int, y: int):
    """Handles a click on the timeline to scrub to a specific frame."""
    if gui_state.label_capture and gui_state.label_capture.isOpened():
        total_frames = gui_state.label_capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if total_frames > 0:
            gui_state.label_index = int(x * total_frames / 500)
            render_image()



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



def jump_to_instance(direction: int):
    """Finds the next/previous instance and jumps the playhead."""
    if not gui_state.label_session_buffer:
        eel.highlightBehaviorRow(None)(); eel.updateConfidenceBadge(None, None)(); return

    # Sort instances by start time to ensure a predictable order
    sorted_instances = sorted(gui_state.label_session_buffer, key=lambda x: x.get('start', 0))
    if not sorted_instances:
        eel.highlightBehaviorRow(None)(); eel.updateConfidenceBadge(None, None)(); return

    current_instance_index_in_sorted_list = -1
    # Find which instance the playhead is currently inside
    for i, inst in enumerate(sorted_instances):
        if inst.get("start", -1) <= gui_state.label_index <= inst.get("end", -1):
            current_instance_index_in_sorted_list = i
            break
    
    target_instance = None
    if current_instance_index_in_sorted_list != -1:
        # If we are inside an instance, find the next/prev one from there
        next_idx = (current_instance_index_in_sorted_list + direction) % len(sorted_instances)
        target_instance = sorted_instances[next_idx]
    else:
        # If we are in a "gap", find the closest instance in the desired direction
        if direction > 0: # Find the next instance after the playhead
            found = next((inst for inst in sorted_instances if inst.get('start', -1) > gui_state.label_index), None)
            target_instance = found or sorted_instances[0] # Wrap around to the first
        else: # Find the previous instance before the playhead
            found = next((inst for inst in reversed(sorted_instances) if inst.get('start', -1) < gui_state.label_index), None)
            target_instance = found or sorted_instances[-1] # Wrap around to the last

    if target_instance:
        # Set the playhead to the start of the target instance
        gui_state.label_index = target_instance.get('start', 0)
        # Find the original index of this instance in the main (unsorted) buffer to update the selection state
        try:
            gui_state.selected_instance_index = gui_state.label_session_buffer.index(target_instance)
        except ValueError:
             gui_state.selected_instance_index = -1 # Should not happen, but a safeguard

        # Update the UI
        confidence = target_instance.get('confidence')
        eel.updateConfidenceBadge(target_instance.get('label'), confidence)()
        eel.highlightBehaviorRow(target_instance.get('label'))()
        render_image()
    else:
        # No instances found, clear the UI selection
        eel.highlightBehaviorRow(None)(); eel.updateConfidenceBadge(None, None)()



def update_instance_boundary(boundary_type: str):
    """
    Directly updates the start or end frame of the currently selected instance.
    """
    if gui_state.selected_instance_index == -1 or gui_state.selected_instance_index >= len(gui_state.label_session_buffer):
        return

    active_instance = gui_state.label_session_buffer[gui_state.selected_instance_index]
    gui_state.label_dirty_instances.add(id(active_instance))
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
    
    # Get the absolute path of the video currently being labeled
    absolute_video_path = gui_state.label_videos[gui_state.label_vid_index]
    # Create the path relative to the project's root directory
    relative_video_path = os.path.relpath(absolute_video_path, start=gui_state.proj.path).replace('\\', '/')
    
    new_instance = { 
        "video": relative_video_path, # Save the new relative path
        "start": start_idx, 
        "end": end_idx, 
        "label": behavior_name 
    }

    gui_state.label_session_buffer.append(new_instance)
    gui_state.label_dirty_instances.add(id(new_instance))
    gui_state.label_history.append(new_instance)
    update_counts()



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
        # Mark the instance as dirty BEFORE changing it
        instance_to_change = gui_state.label_session_buffer[clicked_instance_index]
        gui_state.label_dirty_instances.add(id(instance_to_change))
        instance_to_change['label'] = new_behavior_name
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
        gui_state.label_dirty_instances.add(f"deleted_{removed_inst['label']}")
        if removed_inst in gui_state.label_history:
            gui_state.label_history.remove(removed_inst)
        gui_state.selected_instance_index = -1
        eel.updateConfidenceBadge(None, None)()
        render_image(); update_counts()



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


def stage_for_commit():
    """Enters confirmation mode and triggers a re-render showing only staged labels."""
    gui_state.label_confirmation_mode = True
    eel.setConfirmationModeUI(True)
    render_image()


def cancel_commit_stage():
    """Exits confirmation mode and triggers a re-render of the normal view."""
    gui_state.label_confirmation_mode = False
    eel.setConfirmationModeUI(False)
    render_image()

# =================================================================
# EEL-EXPOSED FUNCTIONS: DATASET & MODEL MANAGEMENT
# =================================================================

def get_disagreement_playlist(dataset_name: str) -> list:
    """
    Reads the disagreement report for a given dataset, de-duplicates entries that
    point to the same source video, normalizes filenames for display, and returns
    a clean, prioritized list.
    """
    if not gui_state.proj or dataset_name not in gui_state.proj.datasets:
        return []
    
    dataset = gui_state.proj.datasets[dataset_name]
    report_path = os.path.join(dataset.path, "disagreement_report.yaml")
    
    if not os.path.exists(report_path):
        return []
        
    try:
        with open(report_path, 'r') as f:
            disagreements = yaml.safe_load(f)
        
        if not disagreements:
            return []

        is_augmented = dataset_name.endswith('_aug')
        source_dataset_name = dataset_name.replace('_aug', '')
        
        worst_disagreements = {}

        for item in disagreements:
            canonical_video_path = item['video_path']
            if is_augmented and canonical_video_path.endswith('_aug.mp4'):
                canonical_video_path = canonical_video_path.replace('_aug.mp4', '.mp4')
            
            if canonical_video_path not in worst_disagreements:
                worst_disagreements[canonical_video_path] = item
            else:
                existing_confidence = worst_disagreements[canonical_video_path]['model_confidence']
                current_confidence = item['model_confidence']
                
                if current_confidence > existing_confidence:
                    worst_disagreements[canonical_video_path] = item
        
        final_playlist = list(worst_disagreements.values())
        final_playlist.sort(key=lambda x: x['model_confidence'], reverse=True)

        # After de-duplicating, loop through the final list one more time to
        # ensure the user only ever sees the canonical source filename.
        for item in final_playlist:
            item['correction_dataset'] = source_dataset_name if is_augmented else dataset_name
            
            original_video_path = item['video_path']
            
            # This is the video that will be opened for editing
            video_to_open = original_video_path
            if is_augmented and original_video_path.endswith('_aug.mp4'):
                video_to_open = original_video_path.replace('_aug.mp4', '.mp4')
            item['video_to_open'] = video_to_open
            
            # This is the video path that will be DISPLAYED in the UI
            item['video_path'] = video_to_open
        
        return final_playlist[:50]
        
    except Exception as e:
        log_message(f"Error reading disagreement report for {dataset_name}: {e}", "ERROR")
        return []

def create_augmented_dataset(source_dataset_name: str, new_dataset_name: str):
    """
    (LAUNCHER) Spawns a background worker to create a new dataset with
    augmented (horizontally flipped) videos and labels.
    """
    if not all([gui_state.proj, source_dataset_name, new_dataset_name]):
        eel.showErrorOnLabelTrainPage("Project not loaded or invalid names provided.")()
        return

    print(f"Spawning worker to augment '{source_dataset_name}' into '{new_dataset_name}'")
    eel.spawn(workthreads.augment_dataset_worker, source_dataset_name, new_dataset_name)



def sync_augmented_dataset(source_dataset_name: str, target_dataset_name: str):
    """
    (LAUNCHER) Triggers the label synchronization worker. The actual spawning
    to a background thread is handled inside the worker module to prevent
    circular import issues.
    """
    if not all([gui_state.proj, source_dataset_name, target_dataset_name]):
        eel.showErrorOnLabelTrainPage("Project not loaded or invalid names provided.")()
        return

    # This calls the new launcher function we will create in workthreads.py
    workthreads.start_label_sync(source_dataset_name, target_dataset_name)



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

def recalculate_dataset_stats(dataset_name: str) -> bool:
    """
    Recalculates a dataset's stats and returns a simple success flag.
    """
    if not gui_state.proj or dataset_name not in gui_state.proj.datasets:
        return False

    try:
        workthreads.log_message(f"Recalculating instance counts for '{dataset_name}'...", "INFO")
        dataset_obj = gui_state.proj.datasets[dataset_name]
        
        # This function updates the config.yaml on disk.
        dataset_obj.update_instance_counts_in_config(gui_state.proj)
        
        workthreads.log_message(f"Stats recalculated successfully for '{dataset_name}'.", "INFO")
        
        # The project state is reloaded from disk in the refreshAllDatasets JS function.
        # We just need to signal that this step was successful.
        return True

    except Exception as e:
        msg = f"Failed to recalculate stats for '{dataset_name}': {e}"
        workthreads.log_message(msg, "ERROR")
        traceback.print_exc()
        eel.showErrorOnLabelTrainPage(msg)()
        return False

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


def create_dataset(name: str, behaviors: list[str], recordings_whitelist: list[str]) -> bool:
    """Creates a new dataset via the project interface."""
    if not gui_state.proj: return False
    dataset = gui_state.proj.create_dataset(name, behaviors, recordings_whitelist)
    if dataset:
        gui_state.label_dataset = dataset
        return True
    return False

def train_model(name: str, batch_size: str, learning_rate: str, epochs: str, sequence_length: str, training_method: str, patience: str, num_runs: str, num_trials: str, optimization_target: str, custom_weights: dict = None):
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
            patience=int(patience),
            num_runs=int(num_runs),
            num_trials=int(num_trials),
            optimization_target=optimization_target,
            custom_weights=custom_weights
        )
        gui_state.training_thread.queue_task(task)
        eel.updateTrainingStatusOnUI(name, "Training task queued...")()
    except ValueError:
        eel.showErrorOnLabelTrainPage("Invalid training parameters provided.")

def update_classification_progress(dataset_name, percent):
    """A helper function to call the JavaScript progress bar update function."""
    eel.updateDatasetLoadProgress(dataset_name, percent)()

def delete_dataset(name: str) -> bool:
    """Handles the request from the frontend to delete a dataset and its model."""
    if not gui_state.proj:
        return False
    try:
        # We will delegate the actual file system logic to the Project class
        return gui_state.proj.delete_dataset(name)
    except Exception as e:
        workthreads.log_message(f"Failed to delete dataset '{name}': {e}", "ERROR")
        traceback.print_exc()
        return False

def start_classification(model_name: str, recordings_whitelist_paths: list[str]):
    """
    Finds HDF5 files to classify and adds them to the persistent classification queue.
    """
    if not gui_state.proj or not gui_state.classify_thread:
        log_message("Project or classification thread not initialized.", "ERROR")
        return

    model_to_use = gui_state.proj.models.get(model_name)
    if not model_to_use:
        log_message(f"Model '{model_name}' not found for inference.", "ERROR")
        eel.showErrorOnLabelTrainPage(f"Model '{model_name}' not found.")()
        return

    h5_files_to_classify = []
    for rel_path in recordings_whitelist_paths:
        search_root = os.path.join(gui_state.proj.recordings_dir, rel_path)
        if os.path.isdir(search_root):
            for dirpath, _, filenames in os.walk(search_root):
                for filename in filenames:
                    if filename.endswith("_cls.h5"):
                        full_path = os.path.join(dirpath, filename)
                        output_csv = full_path.replace("_cls.h5", f"_{model_name}_outputs.csv")
                        if not os.path.exists(output_csv):
                            h5_files_to_classify.append(full_path)

    if not h5_files_to_classify:
        log_message(f"No new files found to classify with model '{model_name}'.", "WARN")
        eel.updateInferenceProgress(model_name, 100, "No new files to process.")()
        return


    # This function is now responsible for telling the UI that a new batch is starting.
    total_files = len(h5_files_to_classify)
    eel.updateInferenceProgress(model_name, 0, f"Processing {total_files} files...")()

    # Set the model for the thread and add all tasks to the queue.
    gui_state.live_inference_model_name = model_name
    with gui_state.classify_lock:
        new_files = [f for f in h5_files_to_classify if f not in gui_state.classify_tasks]
        gui_state.classify_tasks.extend(new_files)
    
    log_message(f"Queued {len(new_files)} files for manual inference with model '{model_name}'.", "INFO")


# =================================================================
# RENDERING & INTERNAL LOGIC (Not Exposed)
# =================================================================

def render_image():
    """
    Renders the current video frame and timelines.
    Implements a FIXED-WIDTH zoom for consistent visual representation.
    """
    if not gui_state.label_capture or not gui_state.label_capture.isOpened():
        eel.updateLabelImageSrc(None, None, None)(); return

    total_frames = int(gui_state.label_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0: return

    current_frame_idx = max(0, min(int(gui_state.label_index), total_frames - 1))
    gui_state.label_capture.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
    ret, frame = gui_state.label_capture.read()
    if not ret or frame is None: return

    main_frame_blob = base64.b64encode(cv2.imencode(".jpg", cv2.resize(frame, (500, 500)))[1].tobytes()).decode("utf-8")

    # --- 1. Draw Full Timeline ---
    full_timeline_canvas = np.full((50, 500, 3), 100, dtype=np.uint8)
    fill_colors(full_timeline_canvas, total_frames)
    
    # --- 2. Define Zoom Window (NEW FIXED-WIDTH LOGIC) ---
    zoom_center_frame = float(current_frame_idx)
    
    # If an instance is selected, center the view on it. Otherwise, center on the playhead.
    if gui_state.selected_instance_index != -1 and gui_state.selected_instance_index < len(gui_state.label_session_buffer):
        instance = gui_state.label_session_buffer[gui_state.selected_instance_index]
        zoom_center_frame = instance.get('start', 0) + (instance.get('end', 0) - instance.get('start', 0)) / 2.0
    
    # The zoom width is now a FIXED percentage of the total video length.
    zoom_width_frames = total_frames * 0.10 
    
    zoom_start_frame = max(0, zoom_center_frame - zoom_width_frames / 2.0)
    zoom_end_frame = min(total_frames, zoom_center_frame + zoom_width_frames / 2.0)
    zoom_duration = zoom_end_frame - zoom_start_frame

    # --- 3. Draw Zoom Timeline ---
    zoom_canvas = np.full((50, 500, 3), 100, dtype=np.uint8)
    if zoom_duration > 0:
        fill_colors(zoom_canvas, total_frames, zoom_start_frame, zoom_end_frame)

    # --- 4. Draw Highlights and Markers ---
    if gui_state.selected_instance_index != -1 and gui_state.selected_instance_index < len(gui_state.label_session_buffer):
        instance = gui_state.label_session_buffer[gui_state.selected_instance_index]
        # Highlight on Full Timeline
        start_px_full = int(500 * instance.get('start', 0) / total_frames)
        end_px_full = int(500 * (instance.get('end', 0) + 1) / total_frames)
        if start_px_full < end_px_full:
            cv2.rectangle(full_timeline_canvas, (start_px_full, 0), (end_px_full, 49), (255, 255, 255), 2)
        
        # Highlight on Zoom Timeline
        if zoom_duration > 0:
            start_x_zoom = int(500 * (instance.get('start', 0) - zoom_start_frame) / zoom_duration)
            end_x_zoom = int(500 * (instance.get('end', 0) + 1 - zoom_start_frame) / zoom_duration)
            cv2.rectangle(zoom_canvas, (start_x_zoom, 0), (end_x_zoom, 49), (255, 255, 255), 2)
            
    # Marker on Full Timeline
    marker_pos_full = int(500 * current_frame_idx / total_frames)
    cv2.line(full_timeline_canvas, (marker_pos_full, 0), (marker_pos_full, 49), (0, 0, 0), 2)
    
    # Marker on Zoom Timeline
    if zoom_duration > 0:
        marker_pos_zoom = int(500 * (current_frame_idx - zoom_start_frame) / zoom_duration)
        if 0 <= marker_pos_zoom < 500:
            cv2.line(zoom_canvas, (marker_pos_zoom, 0), (marker_pos_zoom, 49), (0, 0, 0), 2)

    # --- 5. Encode and Send ---
    _, encoded_timeline = cv2.imencode(".jpg", full_timeline_canvas)
    timeline_blob = base64.b64encode(encoded_timeline.tobytes()).decode("utf-8")
    
    _, encoded_zoom = cv2.imencode(".jpg", zoom_canvas)
    zoom_blob = base64.b64encode(encoded_zoom.tobytes()).decode("utf-8")

    eel.updateLabelImageSrc(main_frame_blob, timeline_blob, zoom_blob)()


def fill_colors(canvas_img: np.ndarray, total_frames: int, view_start_frame: int = 0, view_end_frame: int = -1):
    """
    Draws colored bars on a timeline. It now handles three states:
    1. Solid: Human-verified labels or confirmed model predictions.
    2. Semi-Transparent: Unconfirmed model predictions in Guided Mode.
    3. Ghosted/Inactive: Labels not being reviewed in Review by Behavior mode.
    """
    if view_end_frame == -1:
        view_end_frame = total_frames
    
    view_duration = view_end_frame - view_start_frame
    if view_duration <= 0: return

    behaviors = gui_state.label_session_behaviors
    colors = gui_state.label_session_colors
    timeline_w = canvas_img.shape[1]
    
    is_review_mode = gui_state.label_filter_for_behavior is not None

    for inst in gui_state.label_session_buffer:
        if gui_state.label_confirmation_mode and not ('confidence' not in inst or inst.get('_confirmed', False)):
            continue

        try:
            b_idx = behaviors.index(inst['label'])
            color_hex = colors[b_idx].lstrip("#")
            bgr_color = tuple(int(color_hex[i:i+2], 16) for i in (4, 2, 0))
            
            start_px = int(timeline_w * (inst.get("start", 0) - view_start_frame) / view_duration)
            end_px = int(timeline_w * (inst.get("end", 0) + 1 - view_start_frame) / view_duration)

            if start_px < end_px:
                is_active_behavior = not is_review_mode or inst['label'] == gui_state.label_filter_for_behavior
                is_model_prediction = 'confidence' in inst
                is_confirmed = inst.get('_confirmed', False)

                if is_active_behavior:
                    if is_model_prediction and not is_confirmed:
                        # State 1: Unconfirmed Model Prediction (draw semi-transparent)
                        overlay = canvas_img.copy()
                        cv2.rectangle(overlay, (start_px, 0), (end_px, 49), bgr_color, -1)
                        alpha = 0.4  # More transparent
                        cv2.addWeighted(overlay, alpha, canvas_img, 1 - alpha, 0, canvas_img)
                    else:
                        # State 2: Human Label or Confirmed Prediction (draw solid)
                        cv2.rectangle(canvas_img, (start_px, 0), (end_px, 49), bgr_color, -1)
                        if is_confirmed:
                           cv2.rectangle(canvas_img, (start_px, 0), (end_px, 49), (255, 255, 255), 1)
                else:
                    # State 3: Inactive/Ghosted Behavior in Review Mode (draw very transparent)
                    overlay = canvas_img.copy()
                    cv2.rectangle(overlay, (start_px, 0), (end_px, 49), bgr_color, -1)
                    alpha = 0.2 # <-- Even more transparent
                    cv2.addWeighted(overlay, alpha, canvas_img, 1 - alpha, 0, canvas_img)
                    cv2.rectangle(canvas_img, (start_px, 0), (end_px, 49), tuple(c*0.7 for c in bgr_color), 1)

        except (ValueError, IndexError):
            # If the behavior label is not found, draw it in a bright, obvious "error" color.
            print(f"Warning: Could not find a valid color for behavior '{inst.get('label', 'N/A')}'. Drawing in magenta.")
            bgr_color = (255, 0, 255) # Bright Magenta (BGR)
            start_px = int(timeline_w * (inst.get("start", 0) - view_start_frame) / view_duration)
            end_px = int(timeline_w * (inst.get("end", 0) + 1 - view_start_frame) / view_duration)
            if start_px < end_px:
                cv2.rectangle(canvas_img, (start_px, 0), (end_px, 49), bgr_color, -1)
                cv2.rectangle(canvas_img, (start_px, 0), (end_px, 49), (255, 255, 255), 2)
            continue

    # Logic for drawing a new, in-progress label remains the same.
    if gui_state.label_type != -1 and gui_state.label_start != -1:
        color_hex = gui_state.label_session_colors[gui_state.label_type].lstrip("#")
        bgr_color = tuple(int(color_hex[i:i+2], 16) for i in (4, 2, 0))
        start_f, end_f = min(gui_state.label_start, gui_state.label_index), max(gui_state.label_start, gui_state.label_index)
        
        start_px = int(timeline_w * (start_f - view_start_frame) / view_duration)
        end_px = int(timeline_w * (end_f + 1 - view_start_frame) / view_duration)
        
        if start_px < end_px:
            cv2.rectangle(canvas_img, (start_px, 0), (end_px, 49), bgr_color, -1)
            cv2.rectangle(canvas_img, (start_px, 0), (end_px, 49), (255, 255, 255), 1)

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