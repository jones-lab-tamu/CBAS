"""
Manages backend logic for the startup page of the CBAS application.

This includes functions for creating a new project and loading an existing one,
which are exposed to the JavaScript frontend via Eel.
"""

import os
import torch

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
    """
    print(f"Attempting to load project from: {path}")
    try:
        gui_state.proj = cbas.Project(path)
        print(f"Project loaded successfully: {gui_state.proj.path}")
        
        # Now that the project is loaded, initialize the global DINO encoder
        # using the project-specific model identifier.
        device = "cuda" if torch.cuda.is_available() else "cpu"
        gui_state.dino_encoder = cbas.DinoEncoder(
            model_identifier=gui_state.proj.encoder_model_identifier,
            device=device
        )

    except cbas.InvalidProject as e:
        print(f"Error: {e}. Path is not a valid project.")
        return False, None
    except Exception as e:
        print(f"An unexpected error occurred while loading project {path}: {e}")
        # This will catch the model loading errors from the DinoEncoder
        eel.showErrorOnStartup(f"Could not load project. A common cause is an error loading the encoder model. Please check the terminal for details.\n\nOriginal Error: {e}")()
        return False, None
    
    # --- Post-Load Tasks ---
    if gui_state.proj and gui_state.encode_lock:
        files_to_queue = [
            f for day in gui_state.proj.recordings.values()
            for rec in day.values()
            for f in rec.unencoded_files
        ]
        
        if files_to_queue:
            with gui_state.encode_lock:
                new_files = [f for f in files_to_queue if f not in gui_state.encode_tasks]
                gui_state.encode_tasks.extend(new_files)
            print(f"Queued {len(new_files)} unencoded files for processing.")
        else:
            print("No unencoded files found to queue.")

    if gui_state.proj:
        try:
            if not gui_state.recording_observer or not gui_state.recording_observer.is_alive():
                 print("Starting recording watcher...")
                 workthreads.start_recording_watcher()
            else:
                 print("Recording watcher is already active.")
        except Exception as e:
            print(f"Error trying to start recording watcher: {e}")

    project_info = {
        "project_path": gui_state.proj.path,
        "cameras_dir": gui_state.proj.cameras_dir,
        "recordings_dir": gui_state.proj.recordings_dir,
        "models_dir": gui_state.proj.models_dir,
        "data_sets_dir": gui_state.proj.datasets_dir,
    }
    return True, project_info