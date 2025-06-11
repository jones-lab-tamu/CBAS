import os
import cv2
import base64

import datetime
import subprocess
import gevent
from gevent.subprocess import Popen, PIPE

import eel

import cbas

import gui_state

import shutil
from datetime import datetime

@eel.expose
def import_videos(experiment_name, subject_name, file_paths):
    """
    Imports external video files into the CBAS project structure.
    """
    try:
        # Create the 'experiment' directory (e.g., '20240523-CircadianStudy')
        date_str = datetime.now().strftime("%Y%m%d")
        experiment_dir_name = f"{date_str}-{experiment_name}"
        experiment_path = os.path.join(gui_state.proj.recordings_dir, experiment_dir_name)
        os.makedirs(experiment_path, exist_ok=True)

        # Create the 'subject' directory (e.g., 'Mouse1')
        subject_path = os.path.join(experiment_path, subject_name)
        os.makedirs(subject_path, exist_ok=True)

        # Copy each selected video file
        for i, src_path in enumerate(file_paths):
            # To avoid name collisions and maintain order, we can rename files.
            # E.g., imported_00000.mp4, imported_00001.mp4
            # This mimics the ffmpeg segmented output.
            file_name = f"imported_{str(i).zfill(5)}.mp4"
            dest_path = os.path.join(subject_path, file_name)
            
            print(f"Copying '{src_path}' to '{dest_path}'")
            shutil.copy(src_path, dest_path)
            
        return True
    except Exception as e:
        print(f"Error during video import: {e}")
        return False

@eel.expose
def get_camera_settings(camera_name):
    camera = gui_state.proj.cameras[camera_name]
    return camera.settings_to_dict()

@eel.expose
def save_camera_settings(camera_name, camera_settings):
    gui_state.proj.cameras[camera_name].update_settings(camera_settings)

@eel.expose
def rename_camera(old_camera_name, new_camera_name):
    settings = gui_state.proj.cameras[old_camera_name].settings_to_dict()

    gui_state.proj.create_camera(new_camera_name, settings)
    gui_state.proj.remove_camera(old_camera_name)


@eel.expose
def get_camera_list():
    camera_list = []

    for camera_name in gui_state.proj.cameras:
        camera_settings = get_camera_settings(camera_name)
        camera_list.append((camera_name, camera_settings))

    return camera_list


def fetch_frame(rtsp_url, frame_location):
    try:
        if os.path.exists(frame_location):
            os.remove(frame_location)

        command = f'ffmpeg -loglevel panic -rtsp_transport tcp -i {rtsp_url} -vf "select=eq(n\,34)" -vframes 1 -y "{frame_location}"'

        process = Popen(command, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        if process.returncode == 0 and os.path.exists(frame_location):
            return True, frame_location
        else:
            return False, stderr.decode("utf-8")
    except Exception as e:
        return False, str(e)


@eel.expose
def set_camera_thumbnails():
    for camera in gui_state.proj.cameras.values():
        frame_location = os.path.join(
            gui_state.proj.cameras_dir, camera.name, "frame.jpg"
        )

        frame = cv2.imread(frame_location)

        _, frame = cv2.imencode(".jpg", frame)

        frame = frame.tobytes()

        blob = base64.b64encode(frame)
        blob = blob.decode("utf-8")

        eel.updateImageSrc(camera.name, blob)()


@eel.expose
def download_camera_thumbnails():
    # Store all the camera names and rtsp urls.
    camera_urls = []
    for camera in gui_state.proj.cameras.values():
        camera_urls.append((camera.name, camera.rtsp_url))

    def fetch_frames(camera_urls):
        jobs = []
        for name, rtsp_url in camera_urls:
            frame_location = os.path.join(gui_state.proj.cameras_dir, name, "frame.jpg")

            # Launch each fetch operation as a greenlet
            job = eel.spawn(fetch_frame, rtsp_url, frame_location)
            jobs.append((name, job))

        # Wait for all greenlets to complete
        gevent.joinall([job[1] for job in jobs])

        set_camera_thumbnails()

    eel.spawn(fetch_frames, camera_urls)


@eel.expose
def create_camera(camera_name, rtsp_url):
    settings = {
        "rtsp_url": rtsp_url,
        "framerate": 10,
        "resolution": 256,
        "crop_left_x": 0,
        "crop_top_y": 0,
        "crop_width": 1,
        "crop_height": 1,
    }

    gui_state.proj.create_camera(camera_name, settings)


@eel.expose
def create_recording_dir(camera_name):
    return gui_state.proj.cameras[camera_name].create_recording_dir()

@eel.expose
def get_cbas_status():
    streams = get_active_streams()

    data = {}
    data["streams"] = streams

    data["encode_file_count"] = len(gui_state.encode_tasks)

    return data

@eel.expose
def start_camera_stream(camera_name, destination, segment_time):
    camera = gui_state.proj.cameras[camera_name]

    print(f"Recording on camera {camera_name}")
    return camera.start_recording(destination, segment_time)

@eel.expose
def stop_camera_stream(camera_name):
    camera = gui_state.proj.cameras[camera_name]

    return camera.stop_recording()

@eel.expose
def get_active_streams():
    streams = list(gui_state.proj.active_recordings.keys())
    if len(streams) == 0:
        return False

    return streams

@eel.expose
def open_camera_live_view(camera_name):
    rtsp_url = gui_state.proj.cameras[camera_name].rtsp_url
    vlc_path = r"C:\Program Files\VideoLAN\VLC\vlc.exe"  # Full path to VLC executable

    subprocess.Popen(
        [vlc_path, rtsp_url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )