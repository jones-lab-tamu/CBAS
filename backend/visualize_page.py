"""
Manages all backend logic for the 'Visualize' page of the CBAS application.

This includes functions for:
- Building a hierarchical tree of available classified recordings for selection in the UI.
- Generating new actograms based on user-selected parameters.
- Exporting binned actogram data to a CSV file.
"""

import os
import eel
import cbas
import gui_state
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import io
import base64
from datetime import datetime, timedelta
import re
import traceback
import workthreads


def get_recording_tree() -> list:
    """
    Builds and returns a nested list representing the hierarchy of classified recordings.
    """
    if not gui_state.proj:
        return []

    dates_list = []
    # --- The outer key is the session name (which often includes the date) ---
    for session_name, subjects in gui_state.proj.recordings.items():
        subject_list = [] # <-- This list will hold the subjects for this session
        
        # --- The inner key is the subject/camera name ---
        for subject_name, recording in subjects.items():
            model_list = []
            
            # This part of the logic was correct
            for model_name, classifications in recording.classifications.items():
                if model_name in gui_state.proj.models:
                    model_config = gui_state.proj.models[model_name].config
                    behaviors = model_config.get("behaviors", [])
                    if behaviors:
                        model_list.append((model_name, behaviors))
            
            if model_list:
                # --- Append the subject_name and its models ---
                subject_list.append((subject_name, model_list))
        
        if subject_list:
            # --- Append the session_name and its list of subjects ---
            dates_list.append((session_name, subject_list))
    
    return dates_list

# =================================================================
# ETHOGRAM
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
                start_frame = block.index[0]
                end_frame = block.index[-1]
                events.append({
                    'behavior': behavior,
                    'start': start_frame,
                    'duration': end_frame - start_frame + 1
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
        
    # Use the custom palette instead of the colormap
    behavior_colors = {b: CBAS_COLOR_PALETTE[i % len(CBAS_COLOR_PALETTE)] for i, b in enumerate(behaviors)}   
    
    fig, ax = plt.subplots(figsize=(12, max(4, len(behaviors) * 0.5)), dpi=120)
    fig.patch.set_facecolor('#343a40')
    ax.set_facecolor('#b7bcc1') # Use a lighter grey background

    # Plot each event as a horizontal bar
    for event in events:
        b = event['behavior']
        y_pos = behaviors.index(b)
        bar_color = behavior_colors.get(b, 'gray')
        
        # Set the edgecolor to be the same as the face color of the bar.
        ax.barh(y=y_pos, width=event['duration'], left=event['start'], height=0.7, 
                color=bar_color, edgecolor=bar_color, linewidth=0.5)

    # --- Formatting ---
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
    This version directly scans the file system for maximum reliability.
    """
    if not gui_state.proj or not os.path.isdir(gui_state.proj.recordings_dir):
        return []

    tree = []
    # 1. Walk through the top-level session directories
    for session_dir in sorted(os.scandir(gui_state.proj.recordings_dir), key=lambda e: e.name):
        if not session_dir.is_dir():
            continue

        session_name = session_dir.name
        subjects_in_session = []

        # 2. Walk through the subject directories within each session
        for subject_dir in sorted(os.scandir(session_dir.path), key=lambda e: e.name):
            if not subject_dir.is_dir():
                continue
            
            subject_name = subject_dir.name
            classified_videos_in_subject = []

            # 3. Find all .mp4 files and check if they have a corresponding .csv
            all_files_in_dir = os.listdir(subject_dir.path)
            mp4_files = [f for f in all_files_in_dir if f.endswith(".mp4")]

            for mp4_file in sorted(mp4_files):
                # The video's name without the .mp4 extension
                video_base_name = os.path.splitext(mp4_file)[0]
                
                # Check if there is ANY file in that directory that starts with the video's base name
                # and ends with _outputs.csv. This confirms it's been classified by at least one model.
                if any(f.startswith(video_base_name) and f.endswith("_outputs.csv") for f in all_files_in_dir):
                    full_video_path = os.path.join(subject_dir.path, mp4_file)
                    classified_videos_in_subject.append({
                        "name": mp4_file,
                        "path": full_video_path.replace('\\', '/') # Ensure forward slashes for JS
                    })
            
            if classified_videos_in_subject:
                subjects_in_session.append((subject_name, classified_videos_in_subject))

        if subjects_in_session:
            tree.append((session_name, subjects_in_session))
            
    return tree


def generate_ethogram(video_path: str):
    """
    Generates an ethogram plot for a single video file.
    It finds the most recently modified classification CSV for that video.
    """
    if not os.path.exists(video_path):
        return None

    try:
        recording_dir = os.path.dirname(video_path)
        video_basename = os.path.splitext(os.path.basename(video_path))[0]
        
        # Find all possible classification files for this video
        possible_csvs = [
            os.path.join(recording_dir, f) 
            for f in os.listdir(recording_dir) 
            if f.startswith(video_basename) and f.endswith("_outputs.csv")
        ]
        
        if not possible_csvs:
            raise FileNotFoundError("No classification CSV files found for this video.")

        # Use the most recently modified CSV file as the data source
        latest_csv = max(possible_csvs, key=os.path.getmtime)
        workthreads.log_message(f"Generating ethogram from: {os.path.basename(latest_csv)}", "INFO")
        
        df = pd.read_csv(latest_csv)
        
        # Binarize the data: 1 if it's the max probability for that frame, else 0
        df_binary = (df.T == df.max(axis=1)).T.astype(int)

        plot_title = f"Ethogram for: {os.path.basename(video_path)}"
        blob = _create_matplotlib_ethogram(df_binary, plot_title)

        if blob:
            return {
                "name": os.path.basename(video_path),
                "blob": blob
            }
        else:
            return None

    except Exception as e:
        print(f"Error generating ethogram for {video_path}: {e}")
        import traceback
        traceback.print_exc()
        # Optionally, send an error message back to the UI
        eel.showErrorOnVisualizePage(f"Failed to generate ethogram: {e}")()
        return None

# =================================================================
# WORKER FUNCTIONS (Run in Background Threads)
# =================================================================

def _generate_actograms_task(root: str, sub_dir: str, model: str, behaviors: list, framerate_val: int,
                            binsize_minutes_val: int, start_val: float, threshold_val: float, lightcycle: str,
                            plot_acrophase: bool, task_id: int):
    """
    (WORKER) This function runs in a background thread and does the actual heavy lifting for plotting.
    """
    print(f"Starting actogram task {task_id} for: {behaviors}")
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

        all_model_behaviors = gui_state.proj.models[model].config.get("behaviors", [])

        for behavior_name in behaviors:
            with gui_state.viz_task_lock:
                if task_id != gui_state.latest_viz_task_id:
                    print(f"Cancelling sub-task for '{behavior_name}' in obsolete task {task_id}.")
                    return
            color_for_plot = None
            if len(behaviors) > 1:
                try:
                    behavior_index = all_model_behaviors.index(behavior_name)
                    color_for_plot = CBAS_COLOR_PALETTE[behavior_index % len(CBAS_COLOR_PALETTE)]
                except (ValueError, IndexError):
                    color_for_plot = '#FFFFFF' # Fallback color
            actogram = cbas.Actogram(
                directory=recording.path, model=model, behavior=behavior_name,
                framerate=framerate_val, start=start_val, binsize_minutes=binsize_minutes_val,
                threshold=threshold_val, lightcycle=lightcycle,
                plot_acrophase=plot_acrophase, base_color=color_for_plot
            )
            if actogram.blob:
                results.append({'behavior': behavior_name, 'blob': actogram.blob})
        
        with gui_state.viz_task_lock:
            is_latest = (task_id == gui_state.latest_viz_task_id)

        if is_latest:
            print(f"Task {task_id} is the latest. Sending results to UI.")
            eel.updateActogramDisplay(results, task_id)()
        else:
            print(f"Discarding results for obsolete actogram task {task_id}.")
    except Exception as e:
        print(f"Error in generate_actograms task {task_id}: {e}")
        with gui_state.viz_task_lock:
            if task_id == gui_state.latest_viz_task_id:
                eel.updateActogramDisplay([], task_id)()

# =================================================================
# EEL-EXPOSED FUNCTIONS (Launchers and Direct Actions)
# =================================================================


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
    Generates binned data and saves each behavior to a CSV file inside the
    user-selected output directory.
    """
    if not output_directory:
        print("Export cancelled by user.")
        return
        
    print(f"Exporting data to directory: {output_directory}")
    
    try:
        # --- Parameter Parsing ---
        framerate_val = int(framerate)
        binsize_minutes_val = int(binsize_from_gui)
        start_val = float(start)
        threshold_val = float(threshold) / 100.0
        
        recording = gui_state.proj.recordings.get(root, {}).get(sub_dir)
        if not recording: raise FileNotFoundError(f"Recording not found for {root}/{sub_dir}")

        base_filename = f"{sub_dir}_{model}" # Base name for all files
        exported_files_count = 0

        for behavior_name in behaviors:
            actogram = cbas.Actogram(
                directory=recording.path, model=model, behavior=behavior_name,
                framerate=framerate_val, start=start_val, binsize_minutes=binsize_minutes_val,
                threshold=threshold_val, lightcycle="LD", plot_acrophase=False
            )
            if not actogram.binned_activity: continue
            
            day_one_start = datetime(2000, 1, 1) 
            start_time_obj = day_one_start + timedelta(hours=start_val)
            timestamps = [start_time_obj + timedelta(minutes=i * binsize_minutes_val) for i in range(len(actogram.binned_activity))]

            behavior_df = pd.DataFrame({
                'timestamp': timestamps,
                'behavior': behavior_name,
                'event_count': actogram.binned_activity
            })

            # Create a unique filename for each behavior using the base from the dialog
            # e.g., my_export_eating.csv, my_export_drinking.csv
            behavior_filename = f"{base_filename}_{behavior_name}.csv"
            final_save_path = os.path.join(output_directory, behavior_filename)
            
            behavior_df.to_csv(final_save_path, index=False)
            print(f"Data for '{behavior_name}' exported to {final_save_path}")
            exported_files_count += 1

        if exported_files_count == 0:
            eel.showErrorOnVisualizePage("No data was available to export for the selected behaviors.")()
        else:
            print(f"Successfully exported data for {exported_files_count} behavior(s).")
            # We can still add a success popup here if you like.
            # eel.showExportSuccess(output_directory, exported_files_count)()

    except Exception as e:
        print(f"Error during data export: {e}")
        traceback.print_exc()
        eel.showErrorOnVisualizePage(f"Failed to export data: {e}")()