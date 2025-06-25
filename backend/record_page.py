import os
import sys
import base64
import subprocess
import time
import shutil
from datetime import datetime
import threading

import eel
import gevent

import gui_state
import cbas
import workthreads

from multiprocessing import Process

# =================================================================
# Thumbnail Fetching Logic
# =================================================================

def _fetch_and_update_single_thumbnail(camera_name: str, rtsp_url: str):
    """
    (GEVENT-WORKER) This function is for single thumbnail refreshes.
    It fetches the frame and then directly calls the Eel function to update the UI.
    """
    workthreads.log_message(f"Fetching thumbnail for '{camera_name}'...", "INFO")
    frame_location = os.path.join(gui_state.proj.cameras_dir, camera_name, "frame.jpg")
    
    if os.path.exists(frame_location):
        try:
            os.remove(frame_location)
        except OSError as e:
            workthreads.log_message(f"Error removing existing thumbnail {frame_location}: {e}", "WARN")

    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-rtsp_transport", "tcp", "-timeout", "5000000", 
        "-i", rtsp_url, "-vframes", "1", "-y", frame_location
    ]
    
    creation_flags = 0
    if sys.platform == "win32":
        creation_flags = subprocess.CREATE_NO_WINDOW

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, creationflags=creation_flags)
        process.communicate(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        workthreads.log_message(f"Thumbnail fetch timed out for '{camera_name}'.", "WARN")
    except Exception as e:
        workthreads.log_message(f"Exception during single thumbnail fetch for {camera_name}: {e}", "ERROR")

    encoded_string = None
    if os.path.exists(frame_location) and os.path.getsize(frame_location) > 1000:
        try:
            with open(frame_location, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            workthreads.log_message(f"Successfully loaded thumbnail for '{camera_name}'.", "INFO")
        except Exception as e:
            workthreads.log_message(f"Could not read/encode thumbnail for {camera_name}: {e}", "ERROR")
    else:
        workthreads.log_message(f"Could not retrieve a valid thumbnail for '{camera_name}'.", "WARN")
    
    eel.updateImageSrc(camera_name, encoded_string)()


def _fetch_frame_worker_process(camera_name: str, rtsp_url: str, frame_location: str):
    """
    (MULTIPROCESSING-WORKER) Fetches a frame and saves it to a file.
    """
    if os.path.exists(frame_location):
        try: os.remove(frame_location)
        except OSError: pass

    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-rtsp_transport", "tcp",  "-timeout", "5000000",
        "-i", rtsp_url, "-vframes", "1", "-y", frame_location
    ]
    
    creation_flags = 0
    if sys.platform == "win32":
        creation_flags = subprocess.CREATE_NO_WINDOW

    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL, creationflags=creation_flags)
        process.communicate(timeout=15)
    except Exception:
        pass 


def get_camera_list():
    """Returns a simple list of camera configurations."""
    if not gui_state.proj:
        return []
    return sorted([cam.settings_to_dict() for cam in gui_state.proj.cameras.values()], key=lambda x: x.get('name', ''))

def get_single_camera_thumbnail(camera_name: str):
    """Fetches a thumbnail for a single specified camera using gevent."""
    if not gui_state.proj: return
    cam = gui_state.proj.cameras.get(camera_name)
    if not cam:
        eel.updateImageSrc(camera_name, None)()
        return
    gevent.spawn(_fetch_and_update_single_thumbnail, cam.name, cam.rtsp_url)


def fetch_specific_thumbnails(camera_names: list[str]):
    """Spawns background multiprocessing.Process tasks to fetch thumbnails."""
    if not gui_state.proj: return
    workthreads.log_message(f"Starting bulk thumbnail fetch for {len(camera_names)} camera(s).", "INFO")
    active_processes = {}
    for name in camera_names:
        cam = gui_state.proj.cameras.get(name)
        if cam:
            frame_location = os.path.join(gui_state.proj.cameras_dir, cam.name, "frame.jpg")
            p = Process(target=_fetch_frame_worker_process, args=(cam.name, cam.rtsp_url, frame_location))
            p.start()
            active_processes[cam.name] = [frame_location, False]
    
    time_limit = 15
    start_time = time.time()
    all_done = False
    
    while time.time() - start_time < time_limit and not all_done:
        all_done = True
        for cam_name, (frame_loc, loaded) in active_processes.items():
            if not loaded:
                if os.path.exists(frame_loc) and os.path.getsize(frame_loc) > 1000:
                    try:
                        with open(frame_loc, "rb") as image_file:
                            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                        eel.updateImageSrc(cam_name, encoded_string)()
                        active_processes[cam_name][1] = True
                        workthreads.log_message(f"Loaded thumbnail for '{cam_name}' via poll.", "INFO")
                    except Exception as e:
                        workthreads.log_message(f"Could not read/encode polled thumbnail for {cam_name}: {e}", "ERROR")
                else:
                    all_done = False
        
        if not all_done:
            time.sleep(0.5)

    for cam_name, (frame_loc, loaded) in active_processes.items():
        if not loaded:
            workthreads.log_message(f"Thumbnail fetch for {cam_name} timed out.", "WARN")
            eel.updateImageSrc(cam_name, None)()

# =================================================================
# Live Preview & Other Functions
# =================================================================

def _live_preview_worker(
    camera_name: str, rtsp_url: str,
    crop_w: float, crop_h: float, crop_x: float, crop_y: float,
    timeout_seconds: int = 30
):
    """Worker function that generates a live, CROPPED preview stream."""
    filter_str = (
        f"crop=iw*{crop_w}:ih*{crop_h}:iw*{crop_x}:ih*{crop_y},"
        f"fps=10"
    )

    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", 
        "-rtsp_transport", "tcp", "-max_delay", "5000000", 
        "-i", rtsp_url, 
        "-vf", filter_str,
        "-f", "image2pipe", "-c:v", "mjpeg", "-"
    ]
    process = None
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
        gui_state.live_preview_process = process
        start_time = time.time()
        while True:
            if time.time() - start_time > timeout_seconds: 
                workthreads.log_message(f"Live preview for '{camera_name}' timed out after {timeout_seconds}s.", "WARN")
                break
            if process.poll() is not None: break
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
        workthreads.log_message(f"Error in live preview worker for {camera_name}: {e}", "ERROR")
    finally:
        if process and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=2)
            except subprocess.TimeoutExpired: process.kill()
        if gui_state.live_preview_process is process: gui_state.live_preview_process = None
        eel.end_live_preview(camera_name)()
        workthreads.log_message(f"Live preview for '{camera_name}' has ended.", "INFO")

def start_live_preview(camera_name: str):
    """Starts the live preview by getting the camera's crop settings and passing them to the worker thread."""
    stop_live_preview()
    if not gui_state.proj or camera_name not in gui_state.proj.cameras: return False
    
    workthreads.log_message(f"Starting live preview for '{camera_name}'...", "INFO")
    cam = gui_state.proj.cameras[camera_name]
    
    preview_thread = threading.Thread(
        target=_live_preview_worker, 
        args=(
            cam.name, cam.rtsp_url, 
            cam.crop_width, cam.crop_height, 
            cam.crop_left_x, cam.crop_top_y
        ),
        daemon=True
    )
    preview_thread.start()
    return True

def stop_live_preview():
    if gui_state.live_preview_process and gui_state.live_preview_process.poll() is None:
        workthreads.log_message("Stopping active live preview.", "INFO")
        gui_state.live_preview_process.terminate()
    return True

def reveal_recording_folder(session_name: str, camera_name: str):
    if not all([gui_state.proj, session_name, camera_name]): return
    folder_path = os.path.join(gui_state.proj.recordings_dir, session_name, camera_name)
    workthreads.log_message(f"Revealing folder in file explorer: {folder_path}", "INFO")
    if not os.path.isdir(folder_path):
        eel.showErrorOnRecordPage(f"Recording folder not found for {camera_name} in session {session_name}.")()
        return
    try:
        if sys.platform == "win32": os.startfile(folder_path)
        elif sys.platform == "darwin": subprocess.run(["open", folder_path])
        else: subprocess.run(["xdg-open", folder_path])
    except Exception as e:
        workthreads.log_message(f"Failed to open file explorer for path '{folder_path}': {e}", "ERROR")

def get_camera_settings(camera_name: str) -> dict | None:
    if not gui_state.proj: return None
    cam = gui_state.proj.cameras.get(camera_name)
    return cam.settings_to_dict() if cam else None

def save_camera_settings(camera_name: str, camera_settings: dict) -> bool:
    if not gui_state.proj: return False
    new_name = camera_settings.get("name")
    if not new_name or not new_name.strip(): return False
    
    workthreads.log_message(f"Saving settings for camera '{new_name}'...", "INFO")
    if new_name != camera_name:
        if not rename_camera_internal(camera_name, new_name):
            return False
    if new_name in gui_state.proj.cameras:
        try:
            gui_state.proj.cameras[new_name].update_settings(camera_settings)
            workthreads.log_message(f"Settings for '{new_name}' saved successfully.", "INFO")
            return True
        except Exception as e:
            workthreads.log_message(f"Error saving settings for {new_name}: {e}", "ERROR")
            return False
    return False

def rename_camera_internal(old_name: str, new_name: str) -> bool:
    import yaml
    if not all([gui_state.proj, old_name, new_name.strip()]): return False
    if old_name not in gui_state.proj.cameras: return False
    if new_name in gui_state.proj.cameras:
        eel.showErrorOnRecordPage(f"A camera named '{new_name}' already exists.")()
        return False

    workthreads.log_message(f"Renaming camera '{old_name}' to '{new_name}'...", "INFO")
    old_cam_obj = gui_state.proj.cameras[old_name]
    old_cam_path = old_cam_obj.path
    new_cam_path = os.path.join(gui_state.proj.cameras_dir, new_name)
    try:
        stop_camera_stream(old_name)
        stop_live_preview()
        del gui_state.proj.cameras[old_name]
        shutil.move(old_cam_path, new_cam_path)
        config_path = os.path.join(new_cam_path, "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f: config = yaml.safe_load(f)
            config['name'] = new_name
            with open(config_path, 'w') as f: yaml.dump(config, f, allow_unicode=True)
            new_cam_obj = cbas.Camera(config, gui_state.proj)
            gui_state.proj.cameras[new_name] = new_cam_obj
            return True
        else:
            raise FileNotFoundError("config.yaml not found in renamed folder.")
    except Exception as e:
        workthreads.log_message(f"Error renaming camera: {e}", "ERROR")
        if old_name not in gui_state.proj.cameras:
            gui_state.proj.cameras[old_name] = old_cam_obj
        if not os.path.exists(old_cam_path) and os.path.exists(new_cam_path):
             shutil.move(new_cam_path, old_cam_path)
        eel.showErrorOnRecordPage(f"Could not rename camera: {e}")()
        return False

def stop_all_camera_streams() -> bool:
    if not gui_state.proj or not gui_state.proj.active_recordings: return False
    camera_names_to_stop = list(gui_state.proj.active_recordings.keys())
    workthreads.log_message(f"Received request to stop all streams. Targeting: {camera_names_to_stop}", "INFO")
    for name in camera_names_to_stop:
        stop_camera_stream(name)
        time.sleep(0.1)
    return True

def delete_camera(camera_name: str) -> bool:
    """
    Deletes a camera's configuration folder and removes it from the project state.
    """
    if not gui_state.proj or camera_name not in gui_state.proj.cameras:
        workthreads.log_message(f"Attempted to delete non-existent camera '{camera_name}'.", "WARN")
        return False

    # Prevent deleting a camera that is currently recording
    if camera_name in gui_state.proj.active_recordings:
        workthreads.log_message(f"Cannot delete camera '{camera_name}' while it is recording.", "ERROR")
        return False
        
    workthreads.log_message(f"Deleting camera '{camera_name}'...", "INFO")
    
    try:
        # Get the camera object to find its path
        camera_to_delete = gui_state.proj.cameras[camera_name]
        camera_path = camera_to_delete.path
        
        # Remove the camera from the in-memory project state
        del gui_state.proj.cameras[camera_name]
        
        # Delete the entire camera folder from the disk
        if os.path.isdir(camera_path):
            shutil.rmtree(camera_path)
        
        workthreads.log_message(f"Camera '{camera_name}' and its directory have been deleted.", "INFO")
        return True
    except Exception as e:
        workthreads.log_message(f"An error occurred while deleting camera '{camera_name}': {e}", "ERROR")
        # Attempt to reload the project to restore a consistent state
        gui_state.proj.reload()
        return False

def create_camera(camera_name: str, rtsp_url: str) -> bool:
    if not gui_state.proj or not camera_name.strip() or not rtsp_url.strip(): return False
    workthreads.log_message(f"Creating new camera: '{camera_name}'", "INFO")
    settings = { "rtsp_url": rtsp_url, "framerate": 10, "resolution": 256, "crop_left_x": 0.0, "crop_top_y": 0.0, "crop_width": 1.0, "crop_height": 1.0, }
    new_cam = gui_state.proj.create_camera(camera_name, settings)
    if new_cam:
        workthreads.log_message(f"Camera '{camera_name}' created successfully.", "INFO")
    else:
        workthreads.log_message(f"Failed to create camera '{camera_name}'. It may already exist.", "WARN")
    return new_cam is not None

def get_cbas_status() -> dict:
    if not gui_state.proj or not gui_state.encode_lock: return {"streams": False, "encode_file_count": 0}
    streams = list(gui_state.proj.active_recordings.keys())
    with gui_state.encode_lock:
        encode_count = len(gui_state.encode_tasks)
    return {"streams": streams or False, "encode_file_count": encode_count}

def start_camera_stream(camera_name: str, session_name: str) -> bool:
    if not gui_state.proj or camera_name not in gui_state.proj.cameras: return False
    workthreads.log_message(f"Starting recording for '{camera_name}' in session '{session_name}'.", "INFO")
    camera_obj = gui_state.proj.cameras[camera_name]
    return camera_obj.start_recording(session_name)

def stop_camera_stream(camera_name: str) -> bool:
    if not gui_state.proj or camera_name not in gui_state.proj.cameras: return False
    success = gui_state.proj.cameras[camera_name].stop_recording()
    if success:
        workthreads.log_message(f"Recording for camera '{camera_name}' stopped by user.", "INFO")
    return success

def get_active_streams() -> list[str] | bool:
    if not gui_state.proj: return False
    return list(gui_state.proj.active_recordings.keys()) or False