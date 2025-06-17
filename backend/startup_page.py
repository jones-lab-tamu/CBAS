"""
Manages backend logic for the startup page of the CBAS application.

This includes functions for creating a new project and loading an existing one,
which are exposed to the JavaScript frontend via Eel.
"""

import os

# Local application imports
import cbas
import gui_state
import workthreads

import eel



def create_project(parent_directory: str, project_name: str) -> tuple[bool, dict | None]:
    """
    Creates a new CBAS project directory structure.

    Args:
        parent_directory (str): The directory where the new project folder will be created.
        project_name (str): The name for the new project folder.

    Returns:
        A tuple containing a success flag (bool) and a dictionary with project
        paths, or (False, None) on failure.
    """
    print(f"Attempting to create project '{project_name}' in '{parent_directory}'")
    
    # Call the static method on the Project class to handle directory creation
    gui_state.proj = cbas.Project.create_project(parent_directory, project_name)

    if gui_state.proj is None:
        print(f"Failed to create project '{project_name}'.")
        return False, None

    print(f"Project '{project_name}' created successfully.")
    
    # Return project paths to be stored in the frontend's local storage
    project_info = {
        "project_path": gui_state.proj.path,
        "cameras_dir": gui_state.proj.cameras_dir,
        "recordings_dir": gui_state.proj.recordings_dir,
        "models_dir": gui_state.proj.models_dir,
        "data_sets_dir": gui_state.proj.datasets_dir,
    }
    return True, project_info



def load_project(path: str) -> tuple[bool, dict | None]:
    """
    Loads an existing CBAS project from a given path.

    On successful load, this function also:
    1.  Queues any unencoded video files for background processing.
    2.  Starts the file system watcher to monitor for new recordings.

    Args:
        path (str): The path to the root directory of the CBAS project.

    Returns:
        A tuple containing a success flag (bool) and a dictionary with project
        paths, or (False, None) on failure.
    """
    print(f"Attempting to load project from: {path}")
    try:
        # Instantiate the Project class, which loads all sub-components
        gui_state.proj = cbas.Project(path)
        print(f"Project loaded successfully: {gui_state.proj.path}")
    except cbas.InvalidProject as e:
        print(f"Error: {e}. Path is not a valid project.")
        return False, None
    except Exception as e:
        print(f"An unexpected error occurred while loading project {path}: {e}")
        return False, None
    
    # --- Post-Load Tasks ---

    # 1. Queue up existing unencoded files
    if gui_state.proj and gui_state.encode_lock:
        files_to_queue = [
            f for day in gui_state.proj.recordings.values()
            for rec in day.values()
            for f in rec.unencoded_files
        ]
        
        if files_to_queue:
            with gui_state.encode_lock:
                # Add only files not already in the queue
                new_files = [f for f in files_to_queue if f not in gui_state.encode_tasks]
                gui_state.encode_tasks.extend(new_files)
            print(f"Queued {len(new_files)} unencoded files for processing.")
        else:
            print("No unencoded files found to queue.")

    # 2. Start the recording watcher
    if gui_state.proj:
        try:
            if not gui_state.recording_observer or not gui_state.recording_observer.is_alive():
                 print("Starting recording watcher...")
                 workthreads.start_recording_watcher()
            else:
                 print("Recording watcher is already active.")
        except Exception as e:
            print(f"Error trying to start recording watcher: {e}")

    # Return project paths to the frontend
    project_info = {
        "project_path": gui_state.proj.path,
        "cameras_dir": gui_state.proj.cameras_dir,
        "recordings_dir": gui_state.proj.recordings_dir,
        "models_dir": gui_state.proj.models_dir,
        "data_sets_dir": gui_state.proj.datasets_dir,
    }
    return True, project_info