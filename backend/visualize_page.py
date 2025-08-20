"""
Manages all backend logic for the 'Visualize' page of the CBAS application.

This includes functions for:
- Building hierarchical data trees for both actogram and ethogram modes.
- Generating actogram plots for long-term data aggregation.
- Generating ethogram plots for single-video analysis.
- Exporting binned actogram data to a CSV file.
"""

import os
import io
import base64
import traceback
from datetime import datetime, timedelta
import re

import eel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import cbas
import gui_state
import workthreads

def get_predictions_for_video(video_path: str) -> dict | None:
    """
    Finds the latest classification CSV for a video and returns its data
    along with the model name and behavior list.
    """
    if not os.path.exists(video_path):
        return None
    try:
        recording_dir = os.path.dirname(video_path)
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        
        possible_csvs = [
            os.path.join(recording_dir, f) 
            for f in os.listdir(recording_dir) 
            if f.startswith(video_basename) and f.endswith("_outputs.csv")
        ]
        
        if not possible_csvs:
            return {"error": "No classification files found for this video."}

        latest_csv = max(possible_csvs, key=os.path.getmtime)
        model_name = os.path.basename(latest_csv).replace(f"{video_basename}_", "").replace("_outputs.csv", "")
        
        model_obj = gui_state.proj.models.get(model_name)
        if not model_obj:
            return {"error": f"Could not find the model '{model_name}' associated with the classification."}

        df = pd.read_csv(latest_csv)
        
        return {
            "model_name": model_name,
            "behaviors": model_obj.config.get("behaviors", []),
            "predictions": df.to_dict(orient='split') # Efficiently serialize DataFrame
        }
    except Exception as e:
        workthreads.log_message(f"Error getting predictions for {video_path}: {e}", "ERROR")
        traceback.print_exc()
        return {"error": str(e)}

# =================================================================
# ACTOGRAM: EXISTING FUNCTIONS
# =================================================================

def get_recording_tree() -> list:
    """
    Builds and returns a nested list representing the hierarchy for Actogram mode.
    Structure: [ (Session, [ (Subject, [ (Model, [Behaviors]) ]) ]) ]
    """
    if not gui_state.proj:
        return []

    tree = []
    for session_name, subjects in sorted(gui_state.proj.recordings.items()):
        subject_list = []
        for subject_name, recording in sorted(subjects.items()):
            model_list = []
            for model_name, classifications in recording.classifications.items():
                if model_name in gui_state.proj.models:
                    model_config = gui_state.proj.models[model_name].config
                    behaviors = model_config.get("behaviors", [])
                    if behaviors:
                        model_list.append((model_name, behaviors))
            if model_list:
                subject_list.append((subject_name, model_list))
        if subject_list:
            tree.append((session_name, subject_list))
    return tree

def _generate_actograms_task(root: str, sub_dir: str, model: str, behaviors: list, framerate_val: int,
                            binsize_minutes_val: int, start_val: float, threshold_val: float, lightcycle: str,
                            plot_acrophase: bool, task_id: int):
    """
    (WORKER) This function is now OPTIMIZED. It loads all CSV data into memory once,
    then processes each behavior from the in-memory DataFrame.
    """
    workthreads.log_message(f"Starting optimized actogram task {task_id} for: {behaviors}", "INFO")
    
    CBAS_COLOR_PALETTE = [
    # Strong, vibrant colors first
    '#1f78b4', '#33a02c',  # Strong Blue, Strong Green
    '#e31a1c', '#ff7f00',  # Strong Red, Strong Orange
    '#6a3d9a', '#006400',  # Deep Purple, Forest Green
    '#1b9e77', '#b35806',  # Dark Teal, Burnt Orange
    '#762a83', '#e7298a',  # Rich Purple, Vivid Magenta
    '#a6cee3', '#b2df8a',
    '#fb9a99', '#fdbf6f',
    '#cab2d6', '#ffffb3',
    '#8dd3c7', '#fdae61',
    '#c2a5cf', '#baff00'
    ]
    results = []
    
    try:
        recording = gui_state.proj.recordings.get(root, {}).get(sub_dir)
        if not recording or not os.path.isdir(recording.path):
            raise FileNotFoundError(f"Recording path does not exist for {root}/{sub_dir}")

        # Load all CSVs into one DataFrame
        workthreads.log_message(f"Task {task_id}: Loading and concatenating all CSVs...", "INFO")
        all_csv_paths = [os.path.join(recording.path, f) for f in os.listdir(recording.path) if f.endswith(f"_{model}_outputs.csv")]
        if not all_csv_paths:
             raise FileNotFoundError("No classification CSVs found for this model/subject.")
        
        # Sort files by their modification time to ensure correct chronological order,
        # which is robust to any user-provided file naming convention.
        all_csv_paths.sort(key=os.path.getmtime)
            
        master_df = pd.concat((pd.read_csv(f) for f in all_csv_paths), ignore_index=True)
        workthreads.log_message(f"Task {task_id}: Master DataFrame created with {len(master_df)} frames.", "INFO")

        all_model_behaviors = gui_state.proj.models[model].config.get("behaviors", [])

        for behavior_name in behaviors:
            with gui_state.viz_task_lock:
                if task_id != gui_state.latest_viz_task_id:
                    workthreads.log_message(f"Cancelling sub-task for '{behavior_name}' in obsolete task {task_id}.", "WARN")
                    return
            
            color_for_plot = None
            if len(behaviors) > 1:
                try:
                    behavior_index = all_model_behaviors.index(behavior_name)
                    color_for_plot = CBAS_COLOR_PALETTE[behavior_index % len(CBAS_COLOR_PALETTE)]
                except (ValueError, IndexError):
                    color_for_plot = '#FFFFFF'
            
            # Pass the preloaded DataFrame
            actogram = cbas.Actogram(
                behavior=behavior_name,
                framerate=framerate_val, start=start_val, binsize_minutes=binsize_minutes_val,
                threshold=threshold_val, lightcycle=lightcycle,
                plot_acrophase=plot_acrophase, base_color=color_for_plot,
                preloaded_df=master_df, model=model # Pass the DF and model name for the title
            )

            if actogram.blob:
                results.append({'behavior': behavior_name, 'blob': actogram.blob})
        
        with gui_state.viz_task_lock:
            is_latest = (task_id == gui_state.latest_viz_task_id)

        if is_latest:
            workthreads.log_message(f"Task {task_id} is the latest. Sending actogram results to UI.", "INFO")
            eel.updateActogramDisplay(results, task_id)()
        else:
            workthreads.log_message(f"Discarding results for obsolete actogram task {task_id}.", "WARN")

    except Exception as e:
        workthreads.log_message(f"Error in generate_actograms task {task_id}: {e}", "ERROR")
        traceback.print_exc()
        with gui_state.viz_task_lock:
            if task_id == gui_state.latest_viz_task_id:
                eel.updateActogramDisplay([], task_id)()

def generate_actograms(root: str, sub_dir: str, model: str, behaviors: list, framerate: str,
                       binsize_from_gui: str, start: str, threshold: str, lightcycle: str,
                       plot_acrophase: bool, task_id: int):
    """
    (LAUNCHER) Spawns the actogram generation worker in the background.
    """
    with gui_state.viz_task_lock:
        gui_state.latest_viz_task_id = task_id
    
    framerate_val = int(framerate)
    binsize_minutes_val = int(binsize_from_gui)
    start_val = float(start)
    threshold_val = float(threshold) / 100.0

    eel.spawn(
        _generate_actograms_task,
        root, sub_dir, model, behaviors, framerate_val,
        binsize_minutes_val, start_val, threshold_val, lightcycle,
        plot_acrophase, task_id
    )

def generate_and_save_data(output_directory: str, root: str, sub_dir: str, model: str, behaviors: list, framerate: str,
                           binsize_from_gui: str, start: str, threshold: str):
    """
    (WORKER) Generates binned actogram data for multiple behaviors and saves it to a single CSV file.
    """
    try:
        workthreads.log_message(f"Starting data export for {len(behaviors)} behavior(s)...", "INFO")
        
        # --- 1. Parameter Conversion ---
        framerate_val = int(framerate)
        binsize_minutes_val = int(binsize_from_gui)
        threshold_val = float(threshold) / 100.0

        # --- 2. Load and Concatenate All Relevant Data ---
        recording = gui_state.proj.recordings.get(root, {}).get(sub_dir)
        if not recording or not os.path.isdir(recording.path):
            raise FileNotFoundError(f"Recording path does not exist for {root}/{sub_dir}")

        all_csv_paths = [os.path.join(recording.path, f) for f in os.listdir(recording.path) if f.endswith(f"_{model}_outputs.csv")]
        if not all_csv_paths:
            raise FileNotFoundError("No classification CSVs found for this model/subject.")
        
        # Use the robust chronological sort by modification time
        all_csv_paths.sort(key=os.path.getmtime)
            
        master_df = pd.concat((pd.read_csv(f) for f in all_csv_paths), ignore_index=True)
        workthreads.log_message(f"Loaded master DataFrame with {len(master_df)} frames for export.", "INFO")

        # --- 3. Generate Binned Data for Each Behavior ---
        export_data = {}
        max_len = 0
        for behavior_name in behaviors:
            # Use the cbas.Actogram object to perform the binning calculation
            actogram = cbas.Actogram(
                behavior=behavior_name,
                framerate=framerate_val, start=float(start), binsize_minutes=binsize_minutes_val,
                threshold=threshold_val, lightcycle="LD", # lightcycle doesn't affect data
                preloaded_df=master_df, model=model
            )
            binned_activity = actogram.binned_activity
            export_data[behavior_name] = binned_activity
            if len(binned_activity) > max_len:
                max_len = len(binned_activity)

        # --- 4. Assemble and Save the Final CSV ---
        # Pad shorter behaviors with NaN to ensure all columns have the same length
        for behavior_name in behaviors:
            current_len = len(export_data[behavior_name])
            if current_len < max_len:
                export_data[behavior_name].extend([np.nan] * (max_len - current_len))

        export_df = pd.DataFrame(export_data)
        
        # Create a meaningful filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"BinnedData_{sub_dir}_{model}_{timestamp}.csv"
        output_path = os.path.join(output_directory, filename)
        
        export_df.to_csv(output_path, index_label="Bin")
        workthreads.log_message(f"Successfully exported binned data to: {output_path}", "INFO")
        
        # Notify the user of success
        eel.showErrorOnVisualizePage(f"Successfully exported data to:\n{output_path}")()

    except Exception as e:
        workthreads.log_message(f"Error during data export: {e}", "ERROR")
        traceback.print_exc()
        eel.showErrorOnVisualizePage(f"Failed to export data: {e}")()


# =================================================================
# ETHOGRAM: NEW FUNCTIONS
# =================================================================

def _create_matplotlib_ethogram(df: pd.DataFrame, title: str):
    """
    (WORKER-HELPER) Creates an ethogram plot from a classification DataFrame.
    """
    if df.empty:
        return None

    behaviors = df.columns.tolist()
    if 'background' in behaviors:
        behaviors.remove('background')
        behaviors.append('background')

    events = []
    for behavior in behaviors:
        df['block'] = (df[behavior] != df[behavior].shift()).cumsum()
        active_blocks = df[df[behavior] == 1]
        
        for block_num in active_blocks['block'].unique():
            block = active_blocks[active_blocks['block'] == block_num]
            if not block.empty:
                events.append({
                    'behavior': behavior,
                    'start': block.index[0],
                    'duration': len(block)
                })
    
    if not events:
        return None

    CBAS_COLOR_PALETTE = [
    # Strong, vibrant colors first
    '#1f78b4', '#33a02c',  # Strong Blue, Strong Green
    '#e31a1c', '#ff7f00',  # Strong Red, Strong Orange
    '#6a3d9a', '#006400',  # Deep Purple, Forest Green
    '#1b9e77', '#b35806',  # Dark Teal, Burnt Orange
    '#762a83', '#e7298a',  # Rich Purple, Vivid Magenta
    '#a6cee3', '#b2df8a',
    '#fb9a99', '#fdbf6f',
    '#cab2d6', '#ffffb3',
    '#8dd3c7', '#fdae61',
    '#c2a5cf', '#baff00'
    ]
    behavior_colors = {b: CBAS_COLOR_PALETTE[i % len(CBAS_COLOR_PALETTE)] for i, b in enumerate(behaviors)}
    
    fig, ax = plt.subplots(figsize=(12, max(4, len(behaviors) * 0.5)), dpi=120)
    fig.patch.set_facecolor('#343a40')
    ax.set_facecolor('#6c757d')

    for event in events:
        b = event['behavior']
        y_pos = behaviors.index(b)
        bar_color = behavior_colors.get(b, 'gray')
        ax.barh(y=y_pos, width=event['duration'], left=event['start'], height=0.7, 
                color=bar_color, edgecolor=bar_color)

    ax.set_yticks(range(len(behaviors)))
    ax.set_yticklabels(behaviors)
    
    for i, tick_label in enumerate(ax.get_yticklabels()):
        behavior_name = tick_label.get_text()
        tick_label.set_color(behavior_colors.get(behavior_name, 'white'))

    ax.tick_params(axis='x', colors='white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('white')
    ax.spines['left'].set_color('white')
    
    ax.set_xlabel('Frame Number', color='white')
    ax.set_ylabel('Behavior', color='white')
    ax.set_title(title, color='white', pad=15)
    
    ax.invert_yaxis()
    ax.grid(axis='x', linestyle='--', color='#343a40', alpha=0.7)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=fig.get_facecolor())
    buf.seek(0)
    blob = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    return blob

def get_classified_video_tree():
    """
    Builds and returns a nested list of all individual videos that have at least one classification file.
    """
    if not gui_state.proj or not os.path.isdir(gui_state.proj.recordings_dir):
        return []

    tree = []
    for session_dir in sorted(os.scandir(gui_state.proj.recordings_dir), key=lambda e: e.name):
        if not session_dir.is_dir():
            continue

        session_name = session_dir.name
        subjects_in_session = []

        for subject_dir in sorted(os.scandir(session_dir.path), key=lambda e: e.name):
            if not subject_dir.is_dir():
                continue
            
            subject_name = subject_dir.name
            classified_videos_in_subject = []
            all_files_in_dir = os.listdir(subject_dir.path)
            mp4_files = [f for f in all_files_in_dir if f.endswith(".mp4")]

            for mp4_file in sorted(mp4_files):
                video_base_name = os.path.splitext(mp4_file)[0]
                if any(f.startswith(video_base_name) and f.endswith("_outputs.csv") for f in all_files_in_dir):
                    full_video_path = os.path.join(subject_dir.path, mp4_file)
                    classified_videos_in_subject.append({
                        "name": mp4_file,
                        "path": full_video_path.replace('\\', '/')
                    })
            
            if classified_videos_in_subject:
                subjects_in_session.append((subject_name, classified_videos_in_subject))

        if subjects_in_session:
            tree.append((session_name, subjects_in_session))
            
    return tree

def generate_ethogram(video_path: str):
    """
    Generates an ethogram plot for a single video file.
    """
    if not os.path.exists(video_path):
        return None
    try:
        recording_dir = os.path.dirname(video_path)
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        
        possible_csvs = [
            os.path.join(recording_dir, f) 
            for f in os.listdir(recording_dir) 
            if f.startswith(video_basename) and f.endswith("_outputs.csv")
        ]
        
        if not possible_csvs:
            raise FileNotFoundError("No classification CSV files found for this video.")

        latest_csv = max(possible_csvs, key=os.path.getmtime)
        workthreads.log_message(f"Generating ethogram from: {os.path.basename(latest_csv)}", "INFO")
        
        df = pd.read_csv(latest_csv)
        df_binary = (df.T == df.max(axis=1)).T.astype(int)

        plot_title = f"Ethogram for: {os.path.basename(video_path)}"
        blob = _create_matplotlib_ethogram(df_binary, plot_title)

        if blob:
            return {"name": os.path.basename(video_path), "blob": blob}
        else:
            return None
    except Exception as e:
        workthreads.log_message(f"Error generating ethogram for {video_path}: {e}", "ERROR")
        traceback.print_exc()
        eel.showErrorOnVisualizePage(f"Failed to generate ethogram: {e}")()
        return None