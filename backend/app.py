"""
Main entry point for the CBAS application backend.

This script is responsible for:
1.  Initializing and starting all background worker threads.
2.  Starting a web server using Bottle and Gevent-WebSocket.
3.  Exposing all Python functions to be called from the frontend JavaScript.
"""
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

import os
import sys
import socket
import torch
import threading

import eel
import workthreads
import startup_page
import record_page
import label_train_page
import visualize_page
import gui_state

# =================================================================
# EEL EXPOSED FUNCTIONS
# =================================================================

@eel.expose
def get_project_root():
    if gui_state.proj:
        return gui_state.proj.path
    return None

@eel.expose
def get_encoding_queue_status():
    return workthreads.get_encoding_queue_status()

# --- Startup Page Functions ---

@eel.expose
def create_project(parent_dir, name):
    return startup_page.create_project(parent_dir, name)

@eel.expose
def load_project(path):
    return startup_page.load_project(path)

# --- Record Page Functions ---

@eel.expose
def get_live_inference_status():
    """Returns the name of the currently active live inference model, or None."""
    return gui_state.live_inference_model_name

@eel.expose
def save_all_camera_settings(settings):
    return record_page.save_all_camera_settings(settings)

@eel.expose
def reveal_recording_folder(session_name, camera_name):
    return record_page.reveal_recording_folder(session_name, camera_name)

@eel.expose
def delete_camera(name):
    return record_page.delete_camera(name)

@eel.expose
def get_camera_list():
    return record_page.get_camera_list()

@eel.expose
def get_single_camera_thumbnail(camera_name):
    return record_page.get_single_camera_thumbnail(camera_name)

@eel.expose
def fetch_specific_thumbnails(camera_names):
    return record_page.fetch_specific_thumbnails(camera_names)

@eel.expose
def get_camera_settings(name):
    return record_page.get_camera_settings(name)

@eel.expose
def save_camera_settings(name, settings):
    return record_page.save_camera_settings(name, settings)

@eel.expose
def create_camera(name, url):
    return record_page.create_camera(name, url)

@eel.expose
def get_cbas_status():
    return record_page.get_cbas_status()

@eel.expose
def start_camera_stream(name, session): 
    return record_page.start_camera_stream(name, session)

@eel.expose
def stop_camera_stream(name):
    return record_page.stop_camera_stream(name)

@eel.expose
def stop_all_camera_streams():
    return record_page.stop_all_camera_streams()
    
@eel.expose
def get_active_streams():
    return record_page.get_active_streams()

@eel.expose
def start_live_preview(camera_name):
    return record_page.start_live_preview(camera_name)

@eel.expose
def stop_live_preview():
    return record_page.stop_live_preview()

# --- Label/Train Page Functions ---

@eel.expose
def run_preflight_check(dataset_name: str, test_split: float):
    return label_train_page.run_preflight_check(dataset_name, test_split)

@eel.expose
def start_playback_session(video_path: str, behaviors: list, colors: list, predictions: dict):
    return label_train_page.start_playback_session(video_path, behaviors, colors, predictions)

@eel.expose
def get_label_coverage_report(name):
    return label_train_page.get_label_coverage_report(name)

@eel.expose
def analyze_label_conflicts(name):
    return label_train_page.analyze_label_conflicts(name)

@eel.expose
def clean_and_sort_labels(name):
    return label_train_page.clean_and_sort_labels(name)

@eel.expose
def get_disagreement_playlist(name):
    return label_train_page.get_disagreement_playlist(name)

@eel.expose
def get_instances_for_behavior(dataset_name, behavior_name):
    return label_train_page.get_instances_for_behavior(dataset_name, behavior_name)

@eel.expose
def get_frame_from_video(video_path):
    return label_train_page.get_frame_from_video(video_path)

@eel.expose
def update_dataset_whitelist(name, whitelist):
    return label_train_page.update_dataset_whitelist(name, whitelist)

@eel.expose
def video_has_labels(dataset_name: str, video_path: str) -> bool:
    return label_train_page.video_has_labels(dataset_name, video_path)

@eel.expose
def check_dataset_files_ready(name):
    return label_train_page.check_dataset_files_ready(name)

@eel.expose
def model_exists(name):
    return label_train_page.model_exists(name)

@eel.expose
def load_dataset_configs():
    return label_train_page.load_dataset_configs()

@eel.expose
def get_available_models():
    return label_train_page.get_available_models()

@eel.expose
def set_live_inference_model(model_name: str | None):
    """Sets the model to be used for live inference, or None to disable it."""
    gui_state.live_inference_model_name = model_name
    if model_name:
        workthreads.log_message(f"Live inference enabled with model: {model_name}", "INFO")
    else:
        workthreads.log_message("Live inference disabled.", "INFO")
    return True

@eel.expose
def get_record_tree():
    return label_train_page.get_record_tree()

@eel.expose
def get_hierarchical_video_list(name):
    return label_train_page.get_hierarchical_video_list(name)

@eel.expose
def get_inferred_session_dirs(d_name, m_name):
    return label_train_page.get_inferred_session_dirs(d_name, m_name)

@eel.expose
def get_inferred_videos_for_session(s_dir, m_name):
    return label_train_page.get_inferred_videos_for_session(s_dir, m_name)

@eel.expose
def get_existing_session_names():
    return label_train_page.get_existing_session_names()
    
@eel.expose
def import_videos(s_name, sub_name, paths, standardize, crop_data):
    return label_train_page.import_videos(s_name, sub_name, paths, standardize, crop_data)

@eel.expose
def get_model_configs():
    return label_train_page.get_model_configs()

@eel.expose
def start_labeling(name, video, instances, filter_for_behavior=None):
    return label_train_page.start_labeling(name, video, instances, filter_for_behavior)

@eel.expose
def start_labeling_with_preload(d_name, m_name, path, smoothing_window):
    return label_train_page.start_labeling_with_preload(d_name, m_name, path, smoothing_window)

@eel.expose
def save_session_labels():
    return label_train_page.save_session_labels()

@eel.expose
def refilter_instances(threshold, mode):
    return label_train_page.refilter_instances(threshold, mode)

@eel.expose
def jump_to_frame(frame_num):
    return label_train_page.jump_to_frame(frame_num)

@eel.expose
def confirm_selected_instance():
    return label_train_page.confirm_selected_instance()

@eel.expose
def handle_click_on_label_image(x, y):
    return label_train_page.handle_click_on_label_image(x, y)

@eel.expose
def next_video(shift):
    return label_train_page.next_video(shift)

@eel.expose
def next_frame(shift):
    return label_train_page.next_frame(shift)

@eel.expose
def jump_to_instance(direction):
    return label_train_page.jump_to_instance(direction)

@eel.expose
def update_instance_boundary(b_type):
    return label_train_page.update_instance_boundary(b_type)

@eel.expose
def get_zoom_range_for_click(x_pos):
    return label_train_page.get_zoom_range_for_click(x_pos)

@eel.expose
def label_frame(value):
    return label_train_page.label_frame(value)

@eel.expose
def delete_instance_from_buffer():
    return label_train_page.delete_instance_from_buffer()

@eel.expose
def pop_instance_from_buffer():
    return label_train_page.pop_instance_from_buffer()

@eel.expose
def get_current_labeling_video_path():
    return label_train_page.get_current_labeling_video_path()

@eel.expose
def stage_for_commit():
    return label_train_page.stage_for_commit()

@eel.expose
def cancel_commit_stage():
    return label_train_page.cancel_commit_stage()

@eel.expose
def create_augmented_dataset(source, new):
    return label_train_page.create_augmented_dataset(source, new)

@eel.expose
def sync_augmented_dataset(source, target):
    return label_train_page.sync_augmented_dataset(source, target)

@eel.expose
def reload_project_data():
    return label_train_page.reload_project_data()

@eel.expose
def reveal_dataset_files(name):
    return label_train_page.reveal_dataset_files(name)

@eel.expose
def create_dataset(name, behaviors, whitelist):
    return label_train_page.create_dataset(name, behaviors, whitelist)

@eel.expose
def train_model(name, b_size, lr, epochs, seq_len, method, patience, num_runs, num_trials, optimization_target, use_test, test_split, custom_weights,
                weight_decay, label_smoothing, lstm_hidden_size, lstm_layers):
    return label_train_page.train_model(name, b_size, lr, epochs, seq_len, method, patience, num_runs, num_trials, optimization_target, use_test, test_split, custom_weights,
                                        weight_decay, label_smoothing, lstm_hidden_size, lstm_layers)    
@eel.expose
def start_classification(model_name, whitelist):
    return label_train_page.start_classification(model_name, whitelist)
    
@eel.expose
def cancel_training_task(name):
    return workthreads.cancel_training_task(name)

@eel.expose
def recalculate_dataset_stats(name):
    return label_train_page.recalculate_dataset_stats(name)

@eel.expose
def delete_dataset(name: str) -> bool:
    return label_train_page.delete_dataset(name)

# --- Visualize Page Functions ---

@eel.expose
def get_predictions_for_video(video_path: str):
    return visualize_page.get_predictions_for_video(video_path)

@eel.expose
def get_recording_tree():
    return visualize_page.get_recording_tree()

@eel.expose
def generate_actograms(root, sub, model, behaviors, fr, bs, st, th, lc, pa, task_id):
    return visualize_page.generate_actograms(root, sub, model, behaviors, fr, bs, st, th, lc, pa, task_id)

@eel.expose
def generate_and_save_data(out_dir, root, sub, model, behaviors, fr, bs, st, th):
    return visualize_page.generate_and_save_data(out_dir, root, sub, model, behaviors, fr, bs, st, th)

@eel.expose
def get_classified_video_tree():
    return visualize_page.get_classified_video_tree()

@eel.expose
def generate_ethogram(video_path: str):
    return visualize_page.generate_ethogram(video_path)

@eel.expose
def kill_all_processes():
    """Forcefully terminates all known child processes (recording and preview)."""
    print("Shutdown signal received from main process. Terminating all child processes...")
    # Get all active processes from the gui_state
    active_procs = gui_state.get_all_active_processes()
    for proc in active_procs:
        try:
            if proc and proc.poll() is None: # Check if process is still running
                proc.terminate()
                proc.wait(timeout=2) # Wait briefly for it to close
        except Exception as e:
            # If wait fails, be more aggressive
            try:
                proc.kill()
            except Exception as kill_e:
                print(f"Failed to kill process {proc.pid}: {kill_e}")
    print(f"Terminated {len(active_procs)} processes.")

# =================================================================
# MAIN APPLICATION LOGIC
# =================================================================

def find_available_port(start_port=8000, max_tries=100) -> int:
    for i in range(max_tries):
        port_to_try = start_port + i
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port_to_try))
            return port_to_try
        except OSError:
            continue
    raise IOError("No free ports found for Eel application.")

def _log_forwarder_task():
    while True:
        try:
            message = gui_state.log_queue.get(block=True)
            if message is None:
                print("Log forwarder thread received stop signal.")
                break
            eel.spawn(eel.update_log_panel(message))
        except Exception as e:
            print(f"Error in log forwarder thread: {e}")

def main():
    print("--- PyTorch GPU Diagnostics ---")
    try:
        is_available = torch.cuda.is_available()
        print(f"CUDA available: {is_available}")
        if is_available:
            print(f"Device count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        else:
            print("PyTorch cannot find a CUDA-enabled GPU.")
    except Exception as e:
        print(f"An error occurred during GPU diagnostics: {e}")
    print("-----------------------------")

    eel.init('frontend')
    workthreads.start_threads()
    
    # Start the new recording monitor daemon thread
    monitor_thread = threading.Thread(target=workthreads._recording_monitor_worker, daemon=True)
    monitor_thread.start()
    print("Recording monitor thread started.")
    
    log_forwarder_thread = threading.Thread(target=_log_forwarder_task, daemon=True)
    log_forwarder_thread.start()
    port = find_available_port()
    print(f"Eel server starting on http://localhost:{port}")
    print("This server will wait for the Electron GUI to connect.")
    try:
        eel.start(
            'index.html',
            mode=None,
            host='localhost',
            port=port,
            block=True
        )
    except (SystemExit, MemoryError, KeyboardInterrupt):
        print("Shutdown signal received, Python process is terminating.")
    finally:
        if gui_state.log_queue:
            gui_state.log_queue.put(None)
        workthreads.stop_threads()

if __name__ == "__main__":
    main()