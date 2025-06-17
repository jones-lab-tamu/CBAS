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
from datetime import datetime, timedelta
import re
import traceback


def get_recording_tree() -> list:
    """
    Builds and returns a nested list representing the hierarchy of classified recordings.
    """
    if not gui_state.proj:
        return []

    dates_list = []
    for date_str, sessions in gui_state.proj.recordings.items():
        session_list = []
        for session_name, recording in sessions.items():
            model_list = []
            for model_name, classifications in recording.classifications.items():
                if model_name in gui_state.proj.models:
                    model_config = gui_state.proj.models[model_name].config
                    behaviors = model_config.get("behaviors", [])
                    if behaviors:
                        model_list.append((model_name, behaviors))
            
            if model_list:
                session_list.append((recording.name, model_list))
        
        if session_list:
            dates_list.append((date_str, session_list))
    
    return dates_list

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
    color_map = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
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
                    color_for_plot = color_map[behavior_index % len(color_map)]
                except (ValueError, IndexError):
                    color_for_plot = '#FFFFFF'
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