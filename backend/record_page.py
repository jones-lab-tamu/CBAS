import os
import base64
import subprocess
import shutil
import sys
import yaml
import cv2
import eel
import gui_state
import cbas
import threading
import time
import workthreads

# =================================================================
# LIVE PREVIEW MANAGEMENT (NEW)
# =================================================================

def _live_preview_worker(camera_name: str, rtsp_url: str, timeout_seconds: int = 30):
    """
    (WORKER) This function runs in a background thread to manage a single
    live preview ffmpeg process and pipe its frames to the UI.
    """
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-rtsp_transport", "tcp", "-max_delay", "5000000",
        "-i", rtsp_url,
        "-vf", "fps=10",
        "-f", "image2pipe",
        "-c:v", "mjpeg",
        "-"
    ]
    
    process = None # Use a local variable for the process
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
        gui_state.live_preview_process = process # Assign to global state
        print(f"Started live preview process for {camera_name} (PID: {process.pid})")

        start_time = time.time()
        
        while True:
            if time.time() - start_time > timeout_seconds:
                print(f"Live preview for {camera_name} timed out after {timeout_seconds}s.")
                break

            # Check if the process was terminated from outside
            if process.poll() is not None:
                print(f"Live preview process for {camera_name} terminated externally.")
                break

            # Reading logic (can remain the same)
            buffer = bytearray()
            while True:
                start_marker_pos = buffer.find(b'\xff\xd8')
                if start_marker_pos != -1:
                    buffer = buffer[start_marker_pos:]
                    break
                chunk = process.stdout.read(1024)
                if not chunk: break
                buffer.extend(chunk)
            if not buffer: break
            while True:
                end_marker_pos = buffer.find(b'\xff\xd9')
                if end_marker_pos != -1:
                    jpg_data = buffer[:end_marker_pos + 2]
                    buffer = buffer[end_marker_pos + 2:]
                    blob = base64.b64encode(jpg_data).decode('utf-8')
                    eel.update_live_frame(camera_name, blob)()
                    break
                chunk = process.stdout.read(1024)
                if not chunk: break
                buffer.extend(chunk)
            if not chunk: break
        
    except Exception as e:
        print(f"Error in live preview worker for {camera_name}: {e}")
    finally:
        # --- THREAD-SAFE CLEANUP ---
        # The worker thread is responsible for its own cleanup.
        if process and process.poll() is None:
            print(f"Terminating live preview process for {camera_name}.")
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
        
        # Only clear the global handle if it's still pointing to *this* thread's process
        if gui_state.live_preview_process is process:
            gui_state.live_preview_process = None

        eel.end_live_preview(camera_name)()
        print(f"Live preview for {camera_name} has ended.")


def start_live_preview(camera_name: str):
    """
    (LAUNCHER) Stops any existing live preview and starts a new one for the
    specified camera in a background thread.
    """
    stop_live_preview() # Ensure any old process is killed first
    
    if not gui_state.proj or camera_name not in gui_state.proj.cameras:
        return False
        
    cam = gui_state.proj.cameras[camera_name]
    
    preview_thread = threading.Thread(
        target=_live_preview_worker,
        args=(camera_name, cam.rtsp_url),
        daemon=True
    )
    preview_thread.start()
    return True


def stop_live_preview():
    """
    Public function to signal the current live preview process to terminate.
    It no longer clears the global handle directly.
    """
    if gui_state.live_preview_process:
        print("Signaling existing live preview process to terminate...")
        gui_state.live_preview_process.terminate()
        # The worker thread's finally block will handle the cleanup.
    return True


# =================================================================
# PUBLIC FUNCTIONS (Called by app.py)
# =================================================================

def reveal_recording_folder(session_name: str, camera_name: str):
    """Opens the specific folder where a camera is currently recording its segments."""
    if not all([gui_state.proj, session_name, camera_name]):
        return
    
    folder_path = os.path.join(gui_state.proj.recordings_dir, session_name, camera_name)
    
    if not os.path.isdir(folder_path):
        print(f"Cannot open folder, path does not exist: {folder_path}")
        eel.showErrorOnRecordPage(f"Recording folder not found for {camera_name} in session {session_name}.")()
        return

    try:
        if sys.platform == "win32":
            os.startfile(folder_path)
        elif sys.platform == "darwin": # macOS
            subprocess.run(["open", folder_path])
        else: # linux
            subprocess.run(["xdg-open", folder_path])
    except Exception as e:
        print(f"Failed to open file explorer for path '{folder_path}': {e}")

def _get_thumbnail_and_update_list(camera_dict, output_list, semaphore):
    """(HELPER FOR THREADING) Gets a single thumbnail and appends to a list."""
    try:
        rtsp_url = camera_dict.get('rtsp_url')
        status, thumbnail_blob = _get_thumbnail_blob_for_camera(rtsp_url)
        
        updated_cam_data = camera_dict.copy()
        updated_cam_data['thumbnail_blob'] = thumbnail_blob
        updated_cam_data['status'] = status
        
        active_streams = get_active_streams() or []
        updated_cam_data['is_recording'] = camera_dict.get('name') in active_streams
        
        output_list.append(updated_cam_data)
    finally:
        semaphore.release()

def get_cameras_with_thumbnails(cameras_to_process=None):
    """
    Gathers camera data sequentially and sends progress updates to the frontend.
    """
    if not gui_state.proj:
        return []

    camera_data_list = []
    active_streams = get_active_streams() or []
    
    if cameras_to_process is None:
        all_camera_configs = sorted([cam.settings_to_dict() for cam in gui_state.proj.cameras.values()], key=lambda x: x.get('name', ''))
    else:
        all_camera_configs = cameras_to_process

    total_cameras = len(all_camera_configs)
    for i, cam_dict in enumerate(all_camera_configs):
        cam_name = cam_dict.get('name')
        
        # --- NEW: Send progress update BEFORE fetching ---
        progress = int(((i + 1) / total_cameras) * 100)
        eel.update_thumbnail_progress(progress, f"Fetching {cam_name}... ({i+1}/{total_cameras})")()
        
        rtsp_url = cam_dict.get('rtsp_url')
        status, thumbnail_blob = _get_thumbnail_blob_for_camera(rtsp_url)
        
        updated_cam_data = cam_dict.copy()
        updated_cam_data['thumbnail_blob'] = thumbnail_blob
        updated_cam_data['status'] = status
        updated_cam_data['is_recording'] = cam_name in active_streams
        camera_data_list.append(updated_cam_data)
        
        time.sleep(0.25) # A smaller delay is fine now that the user sees progress.

    eel.update_thumbnail_progress(100, "Finished.")() # Signal completion
    return camera_data_list

def get_single_camera_thumbnail(camera_name: str) -> str | None:
    """
    Fetches a fresh thumbnail for a single specified camera and returns its blob.
    """
    if not gui_state.proj or camera_name not in gui_state.proj.cameras:
        return None
    
    rtsp_url = gui_state.proj.cameras[camera_name].rtsp_url
    return _get_thumbnail_blob_for_camera(rtsp_url)

def get_camera_settings(camera_name: str) -> dict | None:
    if not gui_state.proj: return None
    cam = gui_state.proj.cameras.get(camera_name)
    return cam.settings_to_dict() if cam else None


def save_camera_settings(camera_name: str, camera_settings: dict) -> bool:
    """
    Saves settings. If the name has changed, it will call the robust
    rename function first.
    """
    if not gui_state.proj: return False
    
    new_name = camera_settings.get("name")
    if not new_name or not new_name.strip(): return False

    # If the name is different, we must perform a rename operation first.
    if new_name != camera_name:
        if not rename_camera(camera_name, new_name):
            # The rename function will handle showing an error.
            return False
    
    # Now, save the settings to the (potentially new) camera object.
    if new_name in gui_state.proj.cameras:
        try:
            gui_state.proj.cameras[new_name].update_settings(camera_settings)
            return True
        except Exception as e:
            print(f"Error saving settings for {new_name}: {e}")
            return False
    
    return False


def rename_camera(old_name: str, new_name: str) -> bool:
    """
    Robustly renames a camera, its folder, its config file, and the
    in-memory project state.
    """
    if not all([gui_state.proj, old_name, new_name.strip()]): return False
    if old_name not in gui_state.proj.cameras: return False
    if new_name in gui_state.proj.cameras:
        eel.showErrorOnRecordPage(f"A camera named '{new_name}' already exists.")()
        return False

    print(f"Renaming camera '{old_name}' to '{new_name}'")
    # 1. Get the old camera object and its path
    old_cam_obj = gui_state.proj.cameras[old_name]
    old_cam_path = old_cam_obj.path
    new_cam_path = os.path.join(gui_state.proj.cameras_dir, new_name)

    try:
        # 2. Stop any processes associated with the old camera
        stop_camera_stream(old_name)
        stop_live_preview()

        # 3. Remove the old camera object from the project's in-memory dictionary
        del gui_state.proj.cameras[old_name]
        
        # 4. Rename the folder on disk
        shutil.move(old_cam_path, new_cam_path)

        # 5. Read the config, update the name, and create a NEW camera object
        config_path = os.path.join(new_cam_path, "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            config['name'] = new_name
            # Write the updated config back to the file
            with open(config_path, 'w') as f:
                yaml.dump(config, f, allow_unicode=True)
            
            # 6. Create a new Camera instance and add it to the project state
            new_cam_obj = cbas.Camera(config, gui_state.proj)
            gui_state.proj.cameras[new_name] = new_cam_obj
            
            print(f"Successfully renamed and updated state for '{new_name}'")
            return True
        else:
            # This is an error case, the config file should exist
            raise FileNotFoundError("config.yaml not found in renamed folder.")

    except Exception as e:
        print(f"Error renaming camera: {e}")
        # Attempt to revert state if something went wrong
        if old_name not in gui_state.proj.cameras:
            gui_state.proj.cameras[old_name] = old_cam_obj
        if not os.path.exists(old_cam_path) and os.path.exists(new_cam_path):
             shutil.move(new_cam_path, old_cam_path)
        eel.showErrorOnRecordPage(f"Could not rename camera: {e}")()
        return False


def create_camera(camera_name: str, rtsp_url: str) -> bool:
    if not gui_state.proj or not camera_name.strip() or not rtsp_url.strip(): return False
    settings = { "rtsp_url": rtsp_url, "framerate": 10, "resolution": 256, "crop_left_x": 0.0, "crop_top_y": 0.0, "crop_width": 1.0, "crop_height": 1.0, }
    new_cam = gui_state.proj.create_camera(camera_name, settings)
    return new_cam is not None


def get_cbas_status() -> dict:
    if not gui_state.proj or not gui_state.encode_lock: return {"streams": False, "encode_file_count": 0}
    streams = list(gui_state.proj.active_recordings.keys())
    with gui_state.encode_lock:
        encode_count = len(gui_state.encode_tasks)
    return {"streams": streams or False, "encode_file_count": encode_count}


def start_camera_stream(camera_name: str, session_name: str) -> bool:
    if not gui_state.proj or camera_name not in gui_state.proj.cameras: return False
    camera_obj = gui_state.proj.cameras[camera_name]
    # Now calls start_recording without the time argument
    return camera_obj.start_recording(session_name)


def stop_camera_stream(camera_name: str) -> bool:
    if not gui_state.proj or camera_name not in gui_state.proj.cameras: return False
    
    # Call the existing stop method and capture its return value
    success = gui_state.proj.cameras[camera_name].stop_recording()
    
    # If the method returns True, we log the success.
    if success:
        workthreads.log_message(f"Recording for camera '{camera_name}' stopped by user.", "INFO")
        
    return success


def get_active_streams() -> list[str] | bool:
    if not gui_state.proj: return False
    return list(gui_state.proj.active_recordings.keys()) or False


# =================================================================
# INTERNAL HELPER FUNCTIONS
# =================================================================

def _get_thumbnail_blob_for_camera(rtsp_url: str) -> tuple[str, str | None]:
    """
    A resilient helper to capture a frame. It will automatically retry once
    if the initial connection fails, which can resolve issues with cameras
    that are slow to initialize their streams.
    """
    if not rtsp_url:
        return 'offline', None

    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-rtsp_transport", "tcp",
        "-i", rtsp_url,
        "-vframes", "1",
        "-f", "image2pipe", "-c:v", "mjpeg", "-"
    ]
    
    # Use the correct creation flags for Windows to prevent console windows
    creation_flags = 0
    if sys.platform == "win32":
        creation_flags = subprocess.CREATE_NO_WINDOW

    # --- NEW: Retry Logic ---
    max_retries = 2
    for attempt in range(max_retries):
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                creationflags=creation_flags
            )
            stdout, stderr = process.communicate(timeout=15) # 15-second timeout per attempt

            if process.returncode == 0 and stdout:
                # Success on the first (or second) try!
                print(f"  - CAPTURE SUCCESS for {rtsp_url} on attempt {attempt + 1}.")
                return 'online', base64.b64encode(stdout).decode("utf-8")
            else:
                # The process ran but failed.
                error_message = stderr.decode('utf-8', errors='ignore').strip()
                print(f"  - FFMPEG failed on attempt {attempt + 1} for {rtsp_url}. Error: {error_message}")
                if attempt < max_retries - 1:
                    print("    ...retrying in 1 second.")
                    time.sleep(1) # Wait a second before retrying
                continue # Go to the next iteration of the loop

        except subprocess.TimeoutExpired:
            print(f"  - FFMPEG process timed out on attempt {attempt + 1} for stream: {rtsp_url}")
            if process: process.kill()
            if attempt < max_retries - 1:
                print("    ...retrying in 1 second.")
                time.sleep(1)
            continue
        
        except Exception as e:
            print(f"  - An exception occurred on attempt {attempt + 1} for {rtsp_url}: {e}")
            # Don't retry on unexpected exceptions
            return 'error', None

    # If the loop finishes without a successful return, it means all retries failed.
    return 'offline', None