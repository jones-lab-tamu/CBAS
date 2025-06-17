"""
Manages all backend logic for the 'Record' page of the CBAS application.

This includes functions for:
- Camera configuration (CRUD operations: Create, Read, Update, Delete/Rename).
- Generating and updating camera thumbnail previews using FFMPEG.
- Starting and stopping recording streams for individual or all cameras.
- Providing application status updates to the UI.
- Opening live camera views in an external player (VLC).
"""

# Standard library imports
import os
import base64
import subprocess
import threading
import shutil
import sys

# Third-party imports
import cv2
import eel

# Local application imports
import cbas
import gui_state


# =================================================================
# CAMERA THUMBNAIL GENERATION (Internal Helpers)
# =================================================================

def _fetch_single_frame_ffmpeg(rtsp_url: str, output_path: str) -> tuple[bool, str]:
    """
    (Helper) Uses FFMPEG to fetch a single frame from an RTSP stream.
    This runs synchronously and is intended to be called within a thread.

    Args:
        rtsp_url (str): The RTSP URL of the camera stream.
        output_path (str): The file path to save the output JPG frame.

    Returns:
        A tuple (success_boolean, message_string).
    """
    try:
        if os.path.exists(output_path):
            os.remove(output_path)

        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-rtsp_transport", "tcp",
            "-i", rtsp_url,
            "-vf", "select=eq(n\\,34)",  # Select an early frame (e.g., 35th)
            "-vframes", "1",
            output_path
        ]
        
        process = subprocess.run(command, capture_output=True, text=True, timeout=15)

        if process.returncode == 0 and os.path.exists(output_path):
            return True, "Success"
        else:
            err_msg = f"FFMPEG Error (Code {process.returncode}): {process.stderr.strip()}"
            print(err_msg)
            return False, err_msg
    except subprocess.TimeoutExpired:
        return False, f"FFMPEG timeout fetching frame from {rtsp_url}."
    except Exception as e:
        return False, f"Exception fetching frame: {e}"


def _process_and_send_thumbnail(camera_name: str, frame_path: str):
    """(Helper) Reads a frame from disk, encodes it, and sends it to the UI via Eel."""
    try:
        frame_image = cv2.imread(frame_path)
        if frame_image is None:
            print(f"Could not read thumbnail image: {frame_path}")
            return

        _, encoded_jpeg = cv2.imencode(".jpg", frame_image)
        blob = base64.b64encode(encoded_jpeg.tobytes()).decode("utf-8")
        eel.updateImageSrc(camera_name, blob)()
    except Exception as e:
        print(f"Error processing/sending thumbnail for {camera_name}: {e}")


def _download_thumbnails_task():
    """(WORKER) Target function for the thumbnail downloading thread."""
    if not gui_state.proj: return
    
    # Create a list of fetch tasks
    tasks = [
        {"name": cam.name, "rtsp_url": cam.rtsp_url, "output_path": os.path.join(cam.path, "frame.jpg")}
        for cam in gui_state.proj.cameras.values()
    ]

    for task in tasks:
        print(f"Attempting to download thumbnail for {task['name']}...")
        success, _ = _fetch_single_frame_ffmpeg(task["rtsp_url"], task["output_path"])
        if success:
            # Send to UI immediately for faster feedback
            _process_and_send_thumbnail(task["name"], task["output_path"])
    
    print("Thumbnail download process completed.")


def _download_thumbnail_for_single_camera(camera_obj: cbas.Camera):
    """(WORKER) Fetches a thumbnail for only one newly created camera."""
    output_path = os.path.join(camera_obj.path, "frame.jpg")
    success, _ = _fetch_single_frame_ffmpeg(camera_obj.rtsp_url, output_path)
    if success:
        _process_and_send_thumbnail(camera_obj.name, output_path)


# =================================================================
# EEL-EXPOSED FUNCTIONS: CAMERA MANAGEMENT
# =================================================================


def download_camera_thumbnails():
    """(LAUNCHER) Initiates the download of thumbnails for all cameras in a background thread."""
    print("Download camera thumbnails requested...")
    download_thread = threading.Thread(target=_download_thumbnails_task, daemon=True)
    download_thread.start()



def get_camera_list() -> list[tuple[str, dict]]:
    """Returns a list of (camera_name, camera_settings_dict) for all cameras."""
    if not gui_state.proj: return []
    return [(name, cam.settings_to_dict()) for name, cam in gui_state.proj.cameras.items()]



def get_camera_settings(camera_name: str) -> dict | None:
    """Returns the settings dictionary for a specific camera."""
    if not gui_state.proj: return None
    return gui_state.proj.cameras.get(camera_name, {}).settings_to_dict()



def save_camera_settings(camera_name: str, camera_settings: dict) -> bool:
    """Saves updated settings for a specific camera."""
    if not gui_state.proj or camera_name not in gui_state.proj.cameras:
        return False
    try:
        gui_state.proj.cameras[camera_name].update_settings(camera_settings)
        return True
    except Exception as e:
        print(f"Error saving settings for {camera_name}: {e}"); return False



def rename_camera(old_name: str, new_name: str) -> bool:
    """Renames a camera by creating a new one and removing the old one."""
    if not gui_state.proj: return False
    if old_name not in gui_state.proj.cameras or new_name in gui_state.proj.cameras:
        return False
    if not new_name.strip(): return False

    settings = gui_state.proj.cameras[old_name].settings_to_dict()
    new_cam = gui_state.proj.create_camera(new_name, settings)
    if new_cam:
        # Attempt to copy thumbnail to the new location
        old_thumb = os.path.join(gui_state.proj.cameras[old_name].path, "frame.jpg")
        if os.path.exists(old_thumb):
            try: shutil.copy(old_thumb, os.path.join(new_cam.path, "frame.jpg"))
            except Exception as e: print(f"Could not copy thumbnail during rename: {e}")
        
        gui_state.proj.remove_camera(old_name)
        return True
    return False



def create_camera(camera_name: str, rtsp_url: str) -> bool:
    """Creates a new camera with default settings."""
    if not gui_state.proj: return False
    if not camera_name.strip() or not rtsp_url.strip(): return False

    settings = {
        "rtsp_url": rtsp_url, "framerate": 10, "resolution": 256,
        "crop_left_x": 0.0, "crop_top_y": 0.0, "crop_width": 1.0, "crop_height": 1.0,
    }
    new_cam = gui_state.proj.create_camera(camera_name, settings)
    if new_cam:
        # Fetch thumbnail for the new camera in the background
        threading.Thread(target=_download_thumbnail_for_single_camera, args=(new_cam,), daemon=True).start()
        return True
    return False


# =================================================================
# EEL-EXPOSED FUNCTIONS: RECORDING & STATUS
# =================================================================


def get_cbas_status() -> dict:
    """Returns current status of streams and encoding tasks for the UI."""
    streams = list(gui_state.proj.active_recordings.keys()) if gui_state.proj else []
    with gui_state.encode_lock:
        encode_count = len(gui_state.encode_tasks) if gui_state.encode_lock else 0
    return {"streams": streams or False, "encode_file_count": encode_count}



def start_camera_stream(camera_name: str, session_name: str, segment_time: int) -> bool:
    """Starts recording for a specific camera into a structured path."""
    if not gui_state.proj or camera_name not in gui_state.proj.cameras: return False
    
    camera_obj = gui_state.proj.cameras[camera_name]
    return camera_obj.start_recording(session_name, segment_time)



def stop_camera_stream(camera_name: str) -> bool:
    """Stops recording for a specific camera."""
    if not gui_state.proj or camera_name not in gui_state.proj.cameras: return False
    return gui_state.proj.cameras[camera_name].stop_recording()



def get_active_streams() -> list[str] | bool:
    """Returns a list of names of actively recording cameras."""
    if not gui_state.proj: return False
    streams = list(gui_state.proj.active_recordings.keys())
    return streams or False



def open_camera_live_view(camera_name: str):
    """Opens the camera's RTSP stream in an external player (VLC)."""
    if not gui_state.proj or camera_name not in gui_state.proj.cameras: return
        
    rtsp_url = gui_state.proj.cameras[camera_name].rtsp_url
    vlc_paths = {
        "win32": r"C:\Program Files\VideoLAN\VLC\vlc.exe",
        "darwin": "/Applications/VLC.app/Contents/MacOS/VLC",
        "linux": "vlc"
    }
    vlc_exe = vlc_paths.get(sys.platform)

    # Check if the executable exists or is in the system's PATH
    if vlc_exe and (os.path.exists(vlc_exe) or shutil.which(vlc_exe)):
        try:
            subprocess.Popen([vlc_exe, rtsp_url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            eel.showErrorOnRecordPage(f"Could not open VLC: {e}")
    else:
        eel.showErrorOnRecordPage(f"VLC not found. Please open manually: {rtsp_url}")