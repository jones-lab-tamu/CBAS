"""
Manages backend logic for the startup page of the CBAS application.

This includes functions for creating a new project and loading an existing one,
which are exposed to the JavaScript frontend via Eel.
"""

import os
import torch
import h5py

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
        eel.showErrorOnStartup(f"Could not load project. A common cause is an error loading the encoder model. Please check the terminal for details.\n\nOriginal Error: {e}")()
        return False, None
    
    # --- Post-Load Tasks ---
    if gui_state.proj and gui_state.encode_lock:

        files_to_queue = []
        project_encoder = gui_state.proj.encoder_model_identifier
        
        # We need to get a flat list of all video files first.
        all_video_files = [
            f for day in gui_state.proj.recordings.values()
            for rec in day.values()
            for f in rec.video_files
        ]

        for video_path in all_video_files:
            h5_path = os.path.splitext(video_path)[0] + "_cls.h5"
            
            if not os.path.exists(h5_path):
                # Case 1: The .h5 file doesn't exist at all. Queue for encoding.
                files_to_queue.append(video_path)
                continue

            # Case 2: The .h5 file exists. We must check its metadata stamp.
            try:
                with h5py.File(h5_path, 'r') as h5f:
                    if 'encoder_model_identifier' not in h5f.attrs:
                        # File is old and has no stamp. It must be re-encoded.
                        workthreads.log_message(f"Found unstamped .h5 file: '{os.path.basename(h5_path)}'. Queuing for re-encoding.", "WARN")
                        files_to_queue.append(video_path)
                        continue
                    
                    file_encoder = h5f.attrs['encoder_model_identifier']
                    if file_encoder != project_encoder:
                        # The stamp does not match the current project's encoder. Re-encode.
                        workthreads.log_message(f"Mismatched encoder for '{os.path.basename(h5_path)}'. Expected '{project_encoder}', found '{file_encoder}'. Queuing for re-encoding.", "WARN")
                        files_to_queue.append(video_path)
            except Exception as e:
                # If we can't even read the file, it's likely corrupt. Re-encode.
                workthreads.log_message(f"Could not read metadata for '{os.path.basename(h5_path)}'. File may be corrupt. Queuing for re-encoding. Error: {e}", "WARN")
                files_to_queue.append(video_path)
        
        
        if files_to_queue:
            with gui_state.encode_lock:
                new_files = [f for f in files_to_queue if f not in gui_state.encode_tasks]
                gui_state.encode_tasks.extend(new_files)
            print(f"Queued {len(new_files)} unencoded or mismatched files for processing.")
        else:
            print("No unencoded or mismatched files found to queue.")

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