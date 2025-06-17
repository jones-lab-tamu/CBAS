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
        "-vf", "fps=10",       # Limit framerate to 10fps to reduce load
        "-f", "image2pipe",
        "-c:v", "mjpeg",
        "-"
    ]
    
    try:
        # Start the ffmpeg process and assign it to the global state
        gui_state.live_preview_process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
        print(f"Started live preview process for {camera_name} (PID: {gui_state.live_preview_process.pid})")

        start_time = time.time()
        
        # Continuously read from the process's stdout
        while True:
            # Check for timeout
            if time.time() - start_time > timeout_seconds:
                print(f"Live preview for {camera_name} timed out after {timeout_seconds}s.")
                break

            # Check if the process has been terminated from outside (e.g., by stop_live_preview)
            if gui_state.live_preview_process is None or gui_state.live_preview_process.poll() is not None:
                print(f"Live preview process for {camera_name} terminated.")
                break

            # Read raw JPEG data from ffmpeg's stdout. This is a complex but
            # necessary way to handle piped image data without blocking.
            buffer = bytearray()
            while True:
                # Find the start of the JPEG image
                start_marker_pos = buffer.find(b'\xff\xd8')
                if start_marker_pos != -1:
                    buffer = buffer[start_marker_pos:]
                    break
                chunk = gui_state.live_preview_process.stdout.read(1024)
                if not chunk: break
                buffer.extend(chunk)
            
            if not buffer: break

            while True:
                # Find the end of the JPEG image
                end_marker_pos = buffer.find(b'\xff\xd9')
                if end_marker_pos != -1:
                    jpg_data = buffer[:end_marker_pos + 2]
                    buffer = buffer[end_marker_pos + 2:]
                    
                    # We have a valid frame, send it to the UI
                    blob = base64.b64encode(jpg_data).decode('utf-8')
                    eel.update_live_frame(camera_name, blob)()
                    break # Break inner loop to wait for next frame
                
                chunk = gui_state.live_preview_process.stdout.read(1024)
                if not chunk: break
                buffer.extend(chunk)

            if not chunk: break
        
    except Exception as e:
        print(f"Error in live preview worker for {camera_name}: {e}")
    finally:
        # --- Cleanup ---
        if gui_state.live_preview_process and gui_state.live_preview_process.poll() is None:
            print(f"Terminating live preview process for {camera_name}.")
            gui_state.live_preview_process.terminate()
            try:
                gui_state.live_preview_process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                gui_state.live_preview_process.kill()
        
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
    Public function to forcefully stop the current live preview process, if any.
    """
    if gui_state.live_preview_process:
        print("Stopping existing live preview...")
        gui_state.live_preview_process.terminate()
        gui_state.live_preview_process = None # Clear the global handle
    return True


# =================================================================
# PUBLIC FUNCTIONS (Called by app.py)
# =================================================================

def get_cameras_with_thumbnails():
    """
    The primary function for the Record page.
    It gathers all camera data and generates a fresh thumbnail for each,
    returning a complete payload to the frontend.
    """
    if not gui_state.proj:
        return []

    camera_data_list = []
    for camera in gui_state.proj.cameras.values():
        thumbnail_blob = _get_thumbnail_blob_for_camera(camera.rtsp_url)
        
        camera_data = camera.settings_to_dict()
        camera_data['thumbnail_blob'] = thumbnail_blob
        camera_data_list.append(camera_data)

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
    if not gui_state.proj or camera_name not in gui_state.proj.cameras: return False
    try:
        gui_state.proj.cameras[camera_name].update_settings(camera_settings)
        return True
    except Exception as e:
        print(f"Error saving settings for {camera_name}: {e}"); return False


def rename_camera(old_name: str, new_name: str) -> bool:
    if not gui_state.proj or not old_name or not new_name.strip(): return False
    if old_name not in gui_state.proj.cameras or new_name in gui_state.proj.cameras: return False
    
    old_cam_path = gui_state.proj.cameras[old_name].path
    new_cam_path = os.path.join(gui_state.proj.cameras_dir, new_name)
    
    try:
        # Stop any recording before renaming
        stop_camera_stream(old_name)
        # Move the folder
        shutil.move(old_cam_path, new_cam_path)
        # Update the config file inside the folder
        config_path = os.path.join(new_cam_path, "config.yaml")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f: config = yaml.safe_load(f)
            config['name'] = new_name
            with open(config_path, 'w') as f: yaml.dump(config, f, allow_unicode=True)
        
        gui_state.proj._load_cameras() # Reload all cameras from disk
        return True
    except Exception as e:
        print(f"Error renaming camera: {e}")
        if not os.path.exists(old_cam_path) and os.path.exists(new_cam_path):
             shutil.move(new_cam_path, old_cam_path) # Attempt to revert
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


def start_camera_stream(camera_name: str, session_name: str, segment_time: int) -> bool:
    if not gui_state.proj or camera_name not in gui_state.proj.cameras: return False
    return gui_state.proj.cameras[camera_name].start_recording(session_name, segment_time)


def stop_camera_stream(camera_name: str) -> bool:
    if not gui_state.proj or camera_name not in gui_state.proj.cameras: return False
    return gui_state.proj.cameras[camera_name].stop_recording()


def get_active_streams() -> list[str] | bool:
    if not gui_state.proj: return False
    return list(gui_state.proj.active_recordings.keys()) or False


# =================================================================
# INTERNAL HELPER FUNCTIONS
# =================================================================

def _get_thumbnail_blob_for_camera(rtsp_url: str) -> str | None:
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-rtsp_transport", "tcp", "-max_delay", "5000000",
        "-i", rtsp_url,
        "-vframes", "1",
        "-f", "image2pipe",
        "-c:v", "mjpeg",
        "-"
    ]
    try:
        process = subprocess.run(command, capture_output=True, timeout=20)
        if process.returncode == 0 and process.stdout:
            return base64.b64encode(process.stdout).decode("utf-8")
        else:
            print(f"FFMPEG failed for {rtsp_url}. Error: {process.stderr.decode('utf-8', errors='ignore').strip()}")
            return None
    except subprocess.TimeoutExpired:
        print(f"FFMPEG process timed out for stream: {rtsp_url}")
        return None
    except Exception as e:
        print(f"An exception occurred getting thumbnail for {rtsp_url}: {e}")
        return None