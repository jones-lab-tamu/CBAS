"""
Manages all background processing for the CBAS application.

This module defines the worker threads that handle long-running, resource-intensive
tasks, ensuring the main application and GUI remain responsive. This includes:
- EncodeThread: For converting video files into feature embeddings.
- LogHelper: A utility to centralize logging from threads to the UI.
- ClassificationThread: For running model inference on encoded data.
- TrainingThread: For handling the entire model training and evaluation process.
- VideoFileWatcher: For automatically detecting new recordings.
"""

# Standard library imports
import threading
import time
import ctypes
from datetime import datetime
import os
import yaml
import re
import traceback

# Third-party imports
import torch
import matplotlib
matplotlib.use("Agg")  # Set backend for non-GUI thread compatibility
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import numpy as np
import pandas as pd

# Local application imports
import eel
import cbas
import gui_state
import classifier_head
import subprocess
import shutil

_last_restart_times = {}
file_watcher_handler = None

# =================================================================
# LOGGING HELPER
# =================================================================

def log_message(message: str, level: str = "INFO"):
    """
    Puts a formatted message into the global log queue for the UI and
    also prints it to the console for developer debugging.
    """
    timestamp = datetime.now().strftime('%H:%M:%S')
    formatted_message = f"[{timestamp}] [{level}] {message}"
    
    # Use a lock to make the print statement atomic and thread-safe
    if gui_state.print_lock:
        with gui_state.print_lock:
            print(formatted_message)
    else:
        print(formatted_message) # Fallback if lock not initialized
    
    if gui_state.log_queue:
        gui_state.log_queue.put(formatted_message)


# =================================================================
# WORKER THREAD CLASSES
# =================================================================

def _recording_monitor_worker():
    """
    (WORKER) A daemon thread that periodically checks if ffmpeg processes
    are running. If a process has terminated, it automatically restarts it.
    """
    global _last_restart_times
    RESTART_COOLDOWN = 60 # Cooldown in seconds before restarting the same camera

    while True:
        time.sleep(5) # Check every 5 seconds
        
        if gui_state.proj and gui_state.proj.active_recordings:
            # Iterate over a copy to allow safe modification during the loop
            for name, (proc, start_time, session_name) in list(gui_state.proj.active_recordings.items()):
                if proc.poll() is not None:
                    log_message(f"Recording process for '{name}' terminated unexpectedly (exit code: {proc.returncode}).", "WARN")
                    
                    # Remove the dead process from the active list
                    del gui_state.proj.active_recordings[name]
                    
                    # --- SELF-HEALING LOGIC ---
                    current_time = time.time()
                    last_restart = _last_restart_times.get(name, 0)
                    
                    if (current_time - last_restart) > RESTART_COOLDOWN:
                        log_message(f"Attempting to automatically restart recording for '{name}'...", "INFO")
                        try:
                            # Get the camera object and tell it to start again with the correct session name
                            camera_obj = gui_state.proj.cameras.get(name)
                            if camera_obj:
                                success = camera_obj.start_recording(session_name)
                                if success:
                                    log_message(f"Successfully restarted recording for '{name}'.", "INFO")
                                    _last_restart_times[name] = current_time
                                else:
                                    log_message(f"Failed to restart recording for '{name}'.", "ERROR")
                            else:
                                log_message(f"Cannot restart: Camera object for '{name}' not found.", "ERROR")
                        except Exception as e:
                            log_message(f"An exception occurred while trying to restart '{name}': {e}", "ERROR")
                    else:
                        log_message(f"Skipping restart for '{name}'; it failed within the {RESTART_COOLDOWN}s cooldown period.", "WARN")

def sync_labels_worker(source_dataset_name: str, target_dataset_name: str):
    """
    (WORKER) Rebuilds the labels.yaml for an augmented dataset from its source.
    This is a fast, file-only operation.
    """
    try:
        log_message(f"Starting label sync for '{target_dataset_name}'...", "INFO")
        source_dataset = gui_state.proj.datasets.get(source_dataset_name)
        target_dataset = gui_state.proj.datasets.get(target_dataset_name)

        if not source_dataset or not target_dataset:
            raise ValueError("Could not find source or target dataset.")

        # 1. Re-read the source labels to ensure they are the absolute latest
        with open(source_dataset.labels_path, 'r') as f:
            source_labels_data = yaml.safe_load(f)

        # 2. Start with a fresh copy of the source labels
        new_target_labels = source_labels_data.copy()
        
        # 3. Create the list of augmented labels
        augmented_labels_to_add = []
        all_source_instances = [inst for behavior_list in source_labels_data["labels"].values() for inst in behavior_list]

        for instance in all_source_instances:
            new_instance = instance.copy()
            # Construct the expected augmented video path
            relative_video_path = new_instance["video"]
            base_name, ext = os.path.splitext(relative_video_path)
            augmented_video_rel_path = f"{base_name}_aug{ext}"
            
            # Check if the augmented video actually exists before adding the label
            augmented_video_abs_path = os.path.join(gui_state.proj.path, augmented_video_rel_path)
            if os.path.exists(augmented_video_abs_path):
                new_instance["video"] = augmented_video_rel_path
                augmented_labels_to_add.append(new_instance)
            else:
                log_message(f"Skipping sync for missing augmented video: {augmented_video_rel_path}", "WARN")

        # 4. Add the new augmented labels to the target dictionary
        for inst in augmented_labels_to_add:
            behavior_name = inst.get("label")
            if behavior_name in new_target_labels["labels"]:
                new_target_labels["labels"][behavior_name].append(inst)

        # 5. Overwrite the target labels.yaml with the new, complete data
        with open(target_dataset.labels_path, "w") as f:
            yaml.dump(new_target_labels, f, allow_unicode=True)

        # Now that labels.yaml is updated, reload the dataset's internal state
        # and recalculate the counts in its config.yaml
        target_dataset.labels = new_target_labels # Update in-memory labels
        target_dataset.update_instance_counts_in_config(gui_state.proj)

        log_message(f"Successfully synced labels from '{source_dataset_name}' to '{target_dataset_name}'.", "INFO")
        
        # Refresh the UI, which will now read the updated config.yaml
        eel.refreshAllDatasets()()

    except Exception as e:
        log_message(f"Label sync failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        eel.showErrorOnLabelTrainPage(f"Label sync failed: {e}")()

def augment_dataset_worker(source_dataset_name: str, new_dataset_name: str):
    """
    (WORKER) Creates a new dataset by creating augmented video files.
    This process is resumable and idempotent.
    """
    try:
        # 1. Get the source dataset
        source_dataset = gui_state.proj.datasets.get(source_dataset_name)
        if not source_dataset:
            raise ValueError(f"Source dataset '{source_dataset_name}' not found.")

        # =========================================================================
        # Get or Create the new dataset
        # =========================================================================
        if new_dataset_name in gui_state.proj.datasets:
            log_message(f"Found existing dataset '{new_dataset_name}'. Resuming augmentation...", "INFO")
            new_dataset = gui_state.proj.datasets[new_dataset_name]
        else:
            log_message(f"Creating new dataset '{new_dataset_name}'. Beginning video processing...", "INFO")
            new_dataset = gui_state.proj.create_dataset(
                name=new_dataset_name,
                behaviors=source_dataset.config.get("behaviors", []),
                recordings_whitelist=source_dataset.config.get("whitelist", [])
            )
            if not new_dataset:
                raise RuntimeError(f"Failed to create new dataset folder for '{new_dataset_name}'.")
        # =========================================================================

        ffmpeg_filter_chain = "hflip,eq=brightness=0.03:contrast=1.1,gblur=sigma=0.2"
        
        all_instances = [inst for behavior_list in source_dataset.labels["labels"].values() for inst in behavior_list]
        unique_video_paths = set(os.path.join(gui_state.proj.path, inst["video"]) for inst in all_instances if "video" in inst)
        total_videos_to_process = len(unique_video_paths)
        if total_videos_to_process == 0:
            log_message("No videos found in source dataset to augment.", "WARN")
            eel.refreshAllDatasets()()
            return

        videos_processed_count = 0
        processed_videos = {} 

        for source_video_path in unique_video_paths:
            videos_processed_count += 1
            progress_label = f"Processing video {videos_processed_count} of {total_videos_to_process}"
            percent_complete = (videos_processed_count / total_videos_to_process) * 100
            eel.update_augmentation_progress(percent_complete, progress_label)()
            
            video_dir = os.path.dirname(source_video_path)
            augmented_video_name = f"{os.path.splitext(os.path.basename(source_video_path))[0]}_aug.mp4"
            output_video_path = os.path.join(video_dir, augmented_video_name)

            # =========================================================================
            # Check Before Processing
            # =========================================================================
            if os.path.exists(output_video_path):
                log_message(f"Skipping already augmented video: {os.path.basename(output_video_path)}", "INFO")
            else:
                log_message(f"Augmenting: {os.path.basename(source_video_path)} -> {os.path.basename(output_video_path)}", "INFO")
                command = [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", source_video_path,
                    "-vf", ffmpeg_filter_chain,
                    "-c:a", "copy",
                    output_video_path
                ]
                subprocess.run(command, check=True)
            # =========================================================================
            
            processed_videos[source_video_path] = output_video_path

        # This final part now runs safely after all videos are confirmed to exist
        log_message("Finalizing label file...", "INFO")
        new_labels = []
        for instance in all_instances:
            source_path = os.path.join(gui_state.proj.path, instance["video"])
            if source_path in processed_videos:
                new_instance = instance.copy()
                augmented_video_path = processed_videos[source_path]
                new_instance["video"] = os.path.relpath(augmented_video_path, start=gui_state.proj.path)
                new_labels.append(new_instance)
        
        new_dataset.labels = source_dataset.labels.copy()
        for inst in new_labels:
            behavior_name = inst.get("label")
            if behavior_name in new_dataset.labels["labels"]:
                new_dataset.labels["labels"][behavior_name].append(inst)
        with open(new_dataset.labels_path, "w") as f:
            yaml.dump(new_dataset.labels, f, allow_unicode=True)

        log_message(f"Augmentation complete! The '{new_dataset_name}' dataset is ready.", "INFO")
        eel.refreshAllDatasets()()

    except Exception as e:
        log_message(f"Augmentation failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        eel.showErrorOnLabelTrainPage(f"Augmentation failed: {e}")()
    finally:
        eel.spawn(eel.update_augmentation_progress(-1))

class EncodeThread(threading.Thread):
    """A background thread that continuously processes video encoding tasks serially."""
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cuda_stream = torch.cuda.Stream(device=self.device) if self.device.type == 'cuda' else None
        self.total_initial_tasks = 0
        self.tasks_processed_in_batch = 0

    def run(self):
        """The main loop for the thread. Pulls tasks from the global queue."""
        while True:
            # Wait until a project and encoder are loaded.
            if not gui_state.proj or not gui_state.dino_encoder:
                time.sleep(2) # Wait for a project to be loaded
                continue

            file_to_encode = None
            with gui_state.encode_lock:
                if gui_state.encode_tasks:
                    if self.total_initial_tasks == 0:
                        self.total_initial_tasks = len(gui_state.encode_tasks)
                        self.tasks_processed_in_batch = 0
                    
                    file_to_encode = gui_state.encode_tasks.pop(0)

            if file_to_encode:
                try:
                    video_basename = os.path.basename(file_to_encode)
                    log_message(f"Starting encoding for: {video_basename}", "INFO")

                    last_reported_percent = -1
                    def progress_updater(current_file_percent):
                        nonlocal last_reported_percent
                        status = {
                            "overall_processed": self.tasks_processed_in_batch,
                            "overall_total": self.total_initial_tasks,
                            "current_percent": current_file_percent,
                            "current_file": video_basename
                        }
                        eel.update_global_encoding_progress(status)()

                        current_increment = int(current_file_percent // 10)
                        last_increment = int(last_reported_percent // 10)
                        if current_increment > last_increment:
                            log_message(f"Encoding '{video_basename}': {int(current_file_percent)}% complete...", "INFO")
                            last_reported_percent = current_file_percent

                    progress_updater(0)

                    # Use the globally stored dino_encoder
                    encoder_to_use = gui_state.dino_encoder
                    if self.cuda_stream:
                        with torch.cuda.stream(self.cuda_stream):
                            out_file = cbas.encode_file(encoder_to_use, file_to_encode, progress_callback=progress_updater)
                    else:
                        out_file = cbas.encode_file(encoder_to_use, file_to_encode, progress_callback=progress_updater)

                    if out_file:
                        log_message(f"Finished encoding: {video_basename}", "INFO")
                        if gui_state.live_inference_model_name:
                            with gui_state.classify_lock:
                                gui_state.classify_tasks.append(out_file)
                            log_message(f"Live inference: Queued '{os.path.basename(out_file)}' for classification.", "INFO")
                    else:
                        raise RuntimeError("encode_file returned None, indicating a processing failure.")

                    progress_updater(100)

                except Exception as e:
                    log_message(f"Failed to encode '{os.path.basename(file_to_encode)}'. The file may be corrupted or unreadable. Skipping. Error: {e}", "ERROR")
                    traceback.print_exc()
                
                finally:
                    self.tasks_processed_in_batch += 1

            else:
                if self.total_initial_tasks > 0:
                    status = {"overall_total": 0}
                    eel.update_global_encoding_progress(status)()
                
                self.total_initial_tasks = 0
                self.tasks_processed_in_batch = 0
                time.sleep(1)

    def get_id(self):
        """Returns the thread's unique identifier."""
        if hasattr(self, "_thread_id"): return self._thread_id
        for id, thread in threading._active.items():
            if thread is self: return id
        return None

    def raise_exception(self):
        """A method to forcefully terminate the thread from the outside."""
        thread_id = self.get_id()
        if thread_id is not None:
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, ctypes.py_object(SystemExit))
            if res > 1:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
                print("Exception raise failure in EncodeThread")


class ClassificationThread(threading.Thread):
    """A background thread that runs model inference on encoded files."""
    def __init__(self, device_str: str):
        super().__init__()
        self.device = torch.device(device_str)
        self.cuda_stream = torch.cuda.Stream(device=self.device) if self.device.type == 'cuda' else None

    def _load_model(self, model_name):
        """Helper to load a model and store it in the global state."""
        log_message(f"Loading model '{model_name}' for classification...", "INFO")
        try:
            model_obj = gui_state.proj.models.get(model_name)
            if not model_obj:
                raise ValueError(f"Model '{model_name}' not found in project.")
            
            weights = torch.load(model_obj.weights_path, map_location=self.device, weights_only=True)
            torch_model = classifier_head.classifier(
                in_features=768,
                out_features=len(model_obj.config["behaviors"]),
                seq_len=model_obj.config["seq_len"],
            )
            torch_model.load_state_dict(weights)
            torch_model.to(self.device).eval()
            
            gui_state.live_inference_model_object = torch_model
            log_message(f"Model '{model_name}' loaded successfully.", "INFO")
            return model_obj
        except Exception as e:
            log_message(f"Error loading model '{model_name}': {e}", "ERROR")
            gui_state.live_inference_model_object = None
            return None

    def run(self):
        """The main loop for the thread. Continuously processes the queue."""
        last_model_name = None
        model_meta = None
        
        # Variables for manual inference progress tracking
        manual_inference_total = 0
        manual_inference_processed = 0

        while True:
            current_model_name = gui_state.live_inference_model_name
            if current_model_name != last_model_name:
                if current_model_name:
                    model_meta = self._load_model(current_model_name)
                    # When a new model is loaded for a manual batch, reset progress
                    with gui_state.classify_lock:
                        manual_inference_total = len(gui_state.classify_tasks)
                    manual_inference_processed = 0
                else:
                    gui_state.live_inference_model_object = None
                    model_meta = None
                last_model_name = current_model_name

            file_to_classify = None
            if gui_state.live_inference_model_object and model_meta:
                with gui_state.classify_lock:
                    if gui_state.classify_tasks:
                        file_to_classify = gui_state.classify_tasks.pop(0)

            if file_to_classify:
                log_message(f"Classifying: {os.path.basename(file_to_classify)} with model '{model_meta.name}'", "INFO")
                try:
                    if self.cuda_stream:
                        with torch.cuda.stream(self.cuda_stream):
                            cbas.infer_file(file_to_classify, gui_state.live_inference_model_object, model_meta.name, model_meta.config["behaviors"], model_meta.config["seq_len"], device=self.device)
                    else:
                        cbas.infer_file(file_to_classify, gui_state.live_inference_model_object, model_meta.name, model_meta.config["behaviors"], model_meta.config["seq_len"], device=self.device)
                    

                    manual_inference_processed += 1
                    if manual_inference_total > 0:
                        percent_done = (manual_inference_processed / manual_inference_total) * 100
                        message = f"Processing {manual_inference_processed} / {manual_inference_total}: {os.path.basename(file_to_classify)}"
                        eel.updateInferenceProgress(model_meta.name, percent_done, message)()

                        if manual_inference_processed >= manual_inference_total:
                            log_message(f"Inference queue for model '{model_meta.name}' is empty. Classification complete.", "INFO")
                            eel.updateInferenceProgress(model_meta.name, 100, "Inference complete.")()
                            if gui_state.proj:
                                log_message("Reloading project data to discover new classification files.", "INFO")
                                gui_state.proj.reload()
                            # Reset for the next batch
                            manual_inference_total = 0
                            manual_inference_processed = 0
                            gui_state.live_inference_model_name = None # Clear the model after a manual batch

                except Exception as e:
                    log_message(f"Failed to classify '{os.path.basename(file_to_classify)}'. Error: {e}", "ERROR")
            else:
                time.sleep(1)

    def get_id(self):
        """Returns the thread's unique identifier."""
        if hasattr(self, "_thread_id"): return self._thread_id
        for id, thread in threading._active.items():
            if thread is self: return id
        return None

    def raise_exception(self):
        """A method to forcefully terminate the thread from the outside."""
        thread_id = self.get_id()
        if thread_id is not None:
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, ctypes.py_object(SystemExit))
            if res > 1:
                ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
                print("Exception raise failure in ClassificationThread")


class TrainingThread(threading.Thread):
    """A background thread that handles the entire model training workflow."""
    def __init__(self, device_str: str):
        super().__init__()
        self.device = torch.device(device_str)
        self.cuda_stream = torch.cuda.Stream(device=self.device) if self.device.type == 'cuda' else None
        self.training_queue_lock = threading.Lock()
        self.training_queue: list[TrainingTask] = []
        self.cancel_event = threading.Event()

    def queue_task(self, task: "TrainingTask"):
        """Adds a new training task to this thread's queue."""
        self.cancel_event.clear()
        with self.training_queue_lock:
            self.training_queue.append(task)
        log_message(f"Queued training task for dataset: {task.name}", "INFO")

    def run(self):
        """The main loop for the thread. Pulls tasks from its internal queue."""
        while True:
            if self.cancel_event.is_set():
                time.sleep(1)
                continue

            task_to_run = None
            with self.training_queue_lock:
                if self.training_queue:
                    task_to_run = self.training_queue.pop(0)

            if task_to_run:
                log_message(f"--- Starting Training for Dataset: {task_to_run.name} ---", "INFO")
                if self.cuda_stream:
                    with torch.cuda.stream(self.cuda_stream):
                        self._execute_training_task(task_to_run)
                else:
                    self._execute_training_task(task_to_run)
            else:
                time.sleep(1)

    def _execute_training_task(self, task: "TrainingTask"):
        """Orchestrates the training process across multiple random data splits."""
        overall_best_model = None
        overall_best_f1 = -1.0
        all_run_reports = []
        overall_best_reports_history = None
        overall_best_cm = None
        NUM_INNER_TRIALS = task.num_trials
        best_run_train_insts = None

        try:
            for run_num in range(task.num_runs):
                if self.cancel_event.is_set():
                    log_message(f"Skipping run {run_num + 1} for '{task.name}' due to cancellation.", "WARN")
                    break

                log_message(f"--- Starting Run {run_num + 1}/{task.num_runs} for Dataset: {task.name} ---", "INFO")
                eel.spawn(eel.updateTrainingStatusOnUI(task.name, f"Run {run_num + 1}/{task.num_runs}: Loading new data split..."))
                
                def update_progress(p): eel.spawn(eel.updateDatasetLoadProgress(task.name, p))
                
                weights = None
                if task.training_method == "weighted_loss":
                    train_ds, test_ds, weights, train_insts, _ = gui_state.proj.load_dataset_for_weighted_loss(
                        task.name, seq_len=task.sequence_length, progress_callback=update_progress, seed=run_num
                    )
                elif task.training_method == 'custom_weights' and task.custom_weights:
                    log_message(f"Using custom performance-based class weights: {task.custom_weights}", "INFO")
                    # Convert the dictionary of weights into an ordered list based on the dataset's behaviors
                    weights = [task.custom_weights.get(b, 1.0) for b in task.behaviors]
                    # Use the standard balanced loader, but we'll pass the weights to the loss function
                    train_ds, test_ds, train_insts, _ = gui_state.proj.load_dataset(
                        task.name, seq_len=task.sequence_length, progress_callback=update_progress, seed=run_num
                    )
                else: # Default to Balanced Sampling (oversampling)
                    train_ds, test_ds, train_insts, _ = gui_state.proj.load_dataset(
                        task.name, seq_len=task.sequence_length, progress_callback=update_progress, seed=run_num
                    )
                    weights = None # Ensure weights is None for balanced sampling
                
                if train_ds is None or len(train_ds) == 0:
                    log_message(f"Dataset loading for run {run_num + 1} failed or produced an empty training set. Aborting task.", "ERROR")
                    eel.spawn(eel.updateTrainingStatusOnUI(task.name, "Training failed: Empty training set in data split."))
                    return
                
                eel.spawn(eel.updateDatasetLoadProgress(task.name, 100))
                log_message(f"Dataset for run {run_num + 1} loaded successfully.", "INFO")

                run_best_model, run_best_f1 = None, -1.0
                run_best_reports, run_best_epoch = None, -1

                for i in range(NUM_INNER_TRIALS):
                    if self.cancel_event.is_set(): break
                    log_message(f"Run {run_num + 1}, Trial {i + 1}/{NUM_INNER_TRIALS} for '{task.name}'.", "INFO")
                    
                    def training_progress_updater(message: str):
                        f1_match = re.search(r"Val F1: ([\d\.]+)", message)
                        current_best = run_best_f1
                        if f1_match:
                            current_best = max(run_best_f1, float(f1_match.group(1)))
                        f1_text = f"{current_best:.4f}" if current_best >= 0 else "N/A"
                        display_message = f"Run {run_num + 1}/{task.num_runs}, Trial {i + 1}/{NUM_INNER_TRIALS}... Best F1: {f1_text}"
                        eel.spawn(eel.updateTrainingStatusOnUI(task.name, display_message, message))

                    trial_model, trial_reports, trial_best_epoch = cbas.train_lstm_model(
                        train_ds, test_ds, task.sequence_length, task.behaviors, self.cancel_event,
                        lr=task.learning_rate, batch_size=task.batch_size,
                        epochs=task.epochs, device=self.device, class_weights=weights,
                        patience=task.patience, progress_callback=training_progress_updater,
                        optimization_target=task.optimization_target
                    )

                    if trial_model and trial_reports and trial_best_epoch != -1:
                        optimization_key = task.optimization_target
                        f1 = trial_reports[trial_best_epoch].val_report.get(optimization_key, {}).get("f1-score", -1.0)
                        if f1 > run_best_f1:
                            run_best_f1, run_best_model, run_best_reports, run_best_epoch = f1, trial_model, trial_reports, trial_best_epoch
                
                if self.cancel_event.is_set(): break

                if run_best_model:
                    all_run_reports.append(run_best_reports[run_best_epoch])
                    if run_best_f1 > overall_best_f1:
                        log_message(f"New overall best model found in Run {run_num + 1} with F1: {run_best_f1:.4f}", "INFO")
                        overall_best_f1 = run_best_f1
                        overall_best_model = run_best_model
                        overall_best_reports_history = run_best_reports
                        overall_best_cm = run_best_reports[run_best_epoch].val_cm
                        best_run_train_insts = train_insts

            if self.cancel_event.is_set():
                log_message(f"Training for '{task.name}' was cancelled by user.", "WARN")
                eel.spawn(eel.updateTrainingStatusOnUI(task.name, "Training cancelled."))
                return

            if overall_best_model and all_run_reports:
                if best_run_train_insts:
                    self._generate_disagreement_report(task, overall_best_model, best_run_train_insts)

                self._save_averaged_training_results(
                    task, overall_best_model, all_run_reports, 
                    overall_best_reports_history, overall_best_cm
                )
            else:
                log_message(f"Training failed for '{task.name}'. No valid model could be trained.", "ERROR")
                eel.spawn(eel.updateTrainingStatusOnUI(task.name, "Training failed."))

        except Exception as e:
            log_message(f"Critical error during training task for {task.name}: {e}", "ERROR")
            traceback.print_exc()
            eel.spawn(eel.updateTrainingStatusOnUI(task.name, f"Training Error: {e}"))
            
    def _generate_disagreement_report(self, task, model, train_insts):
        """
        Runs inference on the training set to find where the model disagrees
        with the human labels. Saves a report for user review.
        """
        log_message(f"Generating disagreement report for '{task.name}'...", "INFO")
        eel.spawn(eel.updateTrainingStatusOnUI(task.name, "Analyzing model errors..."))

        disagreements = []
        
        instances_by_video = {}
        for inst in train_insts:
            video_path = inst.get("video")
            if video_path:
                instances_by_video.setdefault(video_path, []).append(inst)

        for rel_video_path, instances in instances_by_video.items():
            abs_video_path = os.path.join(gui_state.proj.path, rel_video_path)
            h5_path = os.path.splitext(abs_video_path)[0] + "_cls.h5"
            
            if not os.path.exists(h5_path):
                continue

            csv_path = h5_path.replace("_cls.h5", f"_{task.name}_outputs.csv")
            if not os.path.exists(csv_path):
                csv_path = cbas.infer_file(
                    file_path=h5_path, model=model, dataset_name=task.name,
                    behaviors=task.behaviors, seq_len=task.sequence_length, device=self.device
                )
            if not csv_path:
                continue
            
            try:
                pred_df = pd.read_csv(csv_path)
                pred_df['model_label'] = pred_df[task.behaviors].idxmax(axis=1)
                pred_df['model_confidence'] = pred_df[task.behaviors].max(axis=1)
            except Exception as e:
                log_message(f"Could not read or process CSV {csv_path}: {e}", "WARN")
                continue

            for inst in instances:
                try:
                    start = int(inst['start'])
                    end = int(inst['end'])
                    true_label = inst['label']
                except (ValueError, KeyError) as e:
                    log_message(f"Skipping malformed instance in disagreement report: {inst}. Error: {e}", "WARN")
                    continue
                
                instance_preds = pred_df.iloc[start:end+1].copy()
                if instance_preds.empty:
                    continue
                
                error_frames = instance_preds[instance_preds['model_label'] != true_label].copy()
                
                if not error_frames.empty:
                    error_frames['block'] = (error_frames.index.to_series().diff() != 1).cumsum()
                    
                    for block_num, block_df in error_frames.groupby('block'):
                        if block_df.empty:
                            continue
                        
                        block_start = block_df.index.min()
                        block_end = block_df.index.max()
                        
                        avg_confidence_in_error = block_df['model_confidence'].mean()
                        most_common_error_prediction = block_df['model_label'].mode()[0]

                        disagreements.append({
                            'video_path': rel_video_path,
                            'start_frame': int(block_start),
                            'end_frame': int(block_end),
                            'human_label': true_label,
                            'model_prediction': most_common_error_prediction,
                            'model_confidence': float(avg_confidence_in_error)
                        })
        
        disagreements.sort(key=lambda x: x['model_confidence'], reverse=True)
        
        report_path = os.path.join(task.dataset.path, "disagreement_report.yaml")
        with open(report_path, 'w') as f:
            yaml.dump(disagreements, f, allow_unicode=True)
            
        log_message(f"Disagreement report with {len(disagreements)} items saved.", "INFO")             

    def _save_averaged_training_results(self, task, best_model, all_reports, best_run_history, best_run_cm):
        """Averages reports from multiple runs and saves the single best model."""
        log_message(f"Averaging results from {len(all_reports)} runs...", "INFO")


        # The model name is now the dataset name with a "_model" suffix.
        model_name = f"{task.name}_model"
        model_dir = os.path.join(gui_state.proj.models_dir, model_name)

        avg_report = {}
        for b in task.behaviors:
            avg_report[b] = {
                'precision': float(np.mean([r.val_report.get(b, {}).get('precision', 0) for r in all_reports])),
                'recall': float(np.mean([r.val_report.get(b, {}).get('recall', 0) for r in all_reports])),
                'f1-score': float(np.mean([r.val_report.get(b, {}).get('f1-score', 0) for r in all_reports])),
            }
        
        avg_f1 = np.mean([r.val_report.get('weighted avg', {}).get('f1-score', 0) for r in all_reports])
        log_message(f"Final Averaged F1 Score: {avg_f1:.4f}", "INFO")
        eel.updateTrainingStatusOnUI(task.name, f"Training complete. Averaged F1: {avg_f1:.4f}")()

        os.makedirs(model_dir, exist_ok=True)
        torch.save(best_model.state_dict(), os.path.join(model_dir, "model.pth"))
        with open(os.path.join(model_dir, "config.yaml"), "w") as f:
            yaml.dump({"seq_len": task.sequence_length, "behaviors": task.behaviors}, f)
        
        with open(os.path.join(task.dataset.path, "performance_report.yaml"), 'w') as f:
            yaml.dump(avg_report, f)
        
        gui_state.proj.models[model_name] = cbas.Model(model_dir)
        task.dataset.config['model'] = model_name 
        with open(task.dataset.config_path, 'w') as f:
            yaml.dump(task.dataset.config, f, allow_unicode=True)       
        
        for b in task.behaviors:
            b_metrics = avg_report.get(b, {})
            task.dataset.update_metric(b, "F1 Score", round(b_metrics.get('f1-score', 0), 2))
            task.dataset.update_metric(b, "Recall", round(b_metrics.get('recall', 0), 2))
            task.dataset.update_metric(b, "Precision", round(b_metrics.get('precision', 0), 2))
            
        plot_dir = task.dataset.path
        
        run_reports_dir = os.path.join(plot_dir, "run_reports")
        os.makedirs(run_reports_dir, exist_ok=True)
        log_message(f"Saving detailed run reports to '{run_reports_dir}'.", "INFO")

        all_cms = []
        for i, report in enumerate(all_reports):
            run_cm = report.val_cm
            if run_cm is not None and run_cm.size > 0:
                all_cms.append(run_cm)
                run_cm_path = os.path.join(run_reports_dir, f"confusion_matrix_run_{i+1}.png")
                save_confusion_matrix_plot(run_cm, run_cm_path, labels=task.behaviors)

        if all_cms:
            avg_cm = np.mean(np.stack(all_cms), axis=0)
            avg_cm_path = os.path.join(plot_dir, "confusion_matrix_AVERAGE.png")
            
            avg_title = f"Average Confusion Matrix Across {len(all_cms)} Runs"
            save_confusion_matrix_plot(avg_cm, avg_cm_path, labels=task.behaviors, title=avg_title, values_format='.1f')
            log_message(f"Averaged confusion matrix saved to '{avg_cm_path}'.", "INFO")

        old_best_cm_path = os.path.join(plot_dir, "confusion_matrix_BEST.png")
        if os.path.exists(old_best_cm_path):
            try:
                os.remove(old_best_cm_path)
            except OSError as e:
                log_message(f"Could not remove old confusion matrix file: {e}", "WARN")

        val_reports_list = [r.val_report for r in all_reports]
        plot_averaged_run_metrics(
            reports=val_reports_list,
            behaviors=task.behaviors,
            out_dir=plot_dir
        )
        log_message(f"Averaged performance plots saved to '{plot_dir}'.", "INFO")

        if best_run_history:
            for metric in ['f1-score', 'precision', 'recall']:
                plot_report_list_metric(
                    reports=best_run_history,
                    metric=metric,
                    behaviors=task.behaviors,
                    out_dir=plot_dir
                )
            log_message(f"Epoch plots for the best run saved to '{plot_dir}'.", "INFO")

        log_message(f"Training for '{task.name}' complete. Model '{model_name}' and reports saved.", "INFO")
        eel.refreshAllDatasets()()
    
    def get_id(self):
        if hasattr(self, "_thread_id"): return self._thread_id
        for id, thread in threading._active.items():
            if thread is self: return id
    
    def raise_exception(self):
        thread_id = self.get_id()
        if thread_id:
            res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, ctypes.py_object(SystemExit))
            if res > 1: ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)



# =================================================================
# DATA & UTILITY CLASSES
# =================================================================

class TrainingTask():
    """A simple data class to hold all parameters for a single training job."""
    def __init__(self, name, dataset, behaviors, batch_size, learning_rate, epochs, sequence_length, training_method, patience, num_runs, num_trials, optimization_target, custom_weights=None):
        self.name, self.dataset, self.behaviors = name, dataset, behaviors
        self.batch_size, self.learning_rate = batch_size, learning_rate
        self.epochs, self.sequence_length = epochs, sequence_length
        self.training_method = training_method
        self.patience = patience
        self.num_runs = num_runs
        self.num_trials = num_trials
        self.optimization_target = optimization_target
        self.custom_weights = custom_weights

def cancel_training_task(dataset_name: str):
    """Finds the running training task and signals it to cancel."""
    if not gui_state.training_thread:
        return

    # Set the main cancel event for the thread
    gui_state.training_thread.cancel_event.set()
    log_message(f"Cancellation signal sent for training task on dataset '{dataset_name}'.", "WARN")
    
    # Also clear the queue to prevent any pending tasks from running
    with gui_state.training_thread.training_queue_lock:
        gui_state.training_thread.training_queue.clear()

    eel.updateTrainingStatusOnUI(dataset_name, f"Training cancelled by user.")()

def save_confusion_matrix_plot(cm_data: np.ndarray, path: str, labels: list = None, title: str = "Confusion Matrix", values_format: str = 'd'):
    """Saves a confusion matrix plot to a file."""
    if cm_data.size == 0: return
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_data, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation='vertical', values_format=values_format)
    ax.set_title(title)
    plt.tight_layout(); plt.savefig(path); plt.close(fig)

def plot_report_list_metric(reports: list, metric: str, behaviors: list, out_dir: str):
    """Plots a given metric over epochs for all behaviors, for both training and validation sets."""
    if not reports: return
    plt.figure(figsize=(10, 7))
    epochs = range(1, len(reports) + 1)
    
    colors = plt.cm.get_cmap('tab10', len(behaviors))

    for i, b_name in enumerate(behaviors):
        train_values = [r.train_report.get(b_name, {}).get(metric, np.nan) for r in reports]
        val_values = [r.val_report.get(b_name, {}).get(metric, np.nan) for r in reports]
        
        if not all(np.isnan(v) for v in train_values):
            plt.plot(epochs, train_values, marker='o', linestyle='-', label=f'{b_name} (Train)', color=colors(i))
        if not all(np.isnan(v) for v in val_values):
            plt.plot(epochs, val_values, marker='x', linestyle='--', label=f'{b_name} (Val)', color=colors(i))

    w_avg_train = [r.train_report.get('weighted avg', {}).get(metric, np.nan) for r in reports]
    w_avg_val = [r.val_report.get('weighted avg', {}).get(metric, np.nan) for r in reports]
    
    if not all(np.isnan(v) for v in w_avg_train):
        plt.plot(epochs, w_avg_train, marker='o', linestyle='-', color='black', linewidth=2, label='Weighted Avg (Train)')
    if not all(np.isnan(v) for v in w_avg_val):
        plt.plot(epochs, w_avg_val, marker='x', linestyle='--', color='grey', linewidth=2, label='Weighted Avg (Val)')

    plt.xlabel("Epochs")
    plt.ylabel(metric.replace('-', ' ').title())
    plt.title(f"{metric.replace('-', ' ').title()} Over Epochs")
    plt.legend(title="Behaviors", bbox_to_anchor=(1.04, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 0.8, 1])
    plt.savefig(os.path.join(out_dir, f"{metric.replace(' ', '_')}_epochs_plot.png"))
    plt.close()

def plot_averaged_run_metrics(reports: list, behaviors: list, out_dir: str):
    """
    Creates bar charts for precision, recall, and f1-score, showing the mean,
    standard deviation, and all individual data points across all training runs.
    """
    if not reports: return

    metrics = ['precision', 'recall', 'f1-score']
    for metric in metrics:
        plt.figure(figsize=(max(8, len(behaviors) * 0.6), 6))
        
        means = []
        stds = []

        # =========================================================================
        # This list will hold the individual data points for each behavior
        all_run_values = []
        # =========================================================================
        
        for b_name in behaviors:
            # Collect the metric for this behavior from each run's report
            run_values = [r.get(b_name, {}).get(metric, 0) for r in reports]
            means.append(np.mean(run_values))
            stds.append(np.std(run_values))
            # =========================================================================
            # Store the individual points for later plotting
            all_run_values.append(run_values)
            # =========================================================================

        x_pos = np.arange(len(behaviors))
        
        # Create the main bar plot with error bars
        plt.bar(x_pos, means, yerr=stds, align='center', alpha=0.7, ecolor='black', capsize=10, label=f'Mean (n={len(reports)})')
        
        # =========================================================================
        # Add the scatter plot of individual data points
        # =========================================================================
        for i, run_vals in enumerate(all_run_values):
            # Add a small amount of horizontal "jitter" to prevent points from overlapping
            # The amount of jitter is scaled by the number of runs to look good
            jitter = np.random.normal(0, 0.04, size=len(run_vals))
            plt.scatter([i + j for j in jitter], run_vals, color='black', alpha=0.6, zorder=3, label='Individual Run' if i == 0 else "")
        # =========================================================================
        
        plt.ylabel(metric.replace('-', ' ').title())
        plt.xticks(x_pos, behaviors, rotation='vertical')
        plt.title(f"Average {metric.replace('-', ' ').title()} Across {len(reports)} Runs")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Create a single, clean legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(out_dir, f"{metric.replace(' ', '_')}_runs_plot.png"))
        plt.close()

# =================================================================
# FILE SYSTEM WATCHER
# =================================================================

class VideoFileWatcher(FileSystemEventHandler):
    """
    A watchdog handler that queues newly created video files for encoding
    after a short delay to prevent I/O contention and ensure the file is complete.
    """
    def __init__(self):
        super().__init__()
        self.pending_files_lock = threading.Lock()
        self.pending_files = {} # {filepath: time_detected}
        self.timer_thread = threading.Thread(target=self._process_pending_files, daemon=True)
        self.timer_thread.start()
        # A 10-second "cool-down" period after a file is created.
        self.DELAY_SECONDS = 10

    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith('.mp4'):
            return

        video_path = event.src_path
        
        # Immediately stage the newly created file for delayed processing.
        with self.pending_files_lock:
            if video_path not in self.pending_files:
                self.pending_files[video_path] = time.time()
                log_message(f"Watcher staged '{os.path.basename(video_path)}' for delayed encoding.", "INFO")

    def _process_pending_files(self):
        """
        This method runs in a separate thread. It periodically checks the
        list of pending files and queues them for encoding if enough time has passed.
        """
        while True:
            time.sleep(5) # Check every 5 seconds
            
            files_to_queue_now = []
            with self.pending_files_lock:
                current_time = time.time()
                for file_path, detected_time in list(self.pending_files.items()):
                    if (current_time - detected_time) > self.DELAY_SECONDS:
                        files_to_queue_now.append(file_path)
                        del self.pending_files[file_path]

            if files_to_queue_now:
                with gui_state.encode_lock:
                    for f in files_to_queue_now:
                        # This is the original, simple logic. It just checks if the file
                        # is already in the encode queue. This is sufficient when the
                        # manual import worker clears the pending list.
                        if f not in gui_state.encode_tasks:
                            gui_state.encode_tasks.append(f)
                            log_message(f"Queued for encoding (auto-detected): '{os.path.basename(f)}'", "INFO")


# =================================================================
# GLOBAL THREAD MANAGEMENT FUNCTIONS
# =================================================================


def sync_augmented_dataset(source_dataset_name: str, target_dataset_name: str):
    """
    (LAUNCHER) Spawns a background worker to re-sync the labels of an augmented
    dataset from its original source dataset.
    """
    if not all([gui_state.proj, source_dataset_name, target_dataset_name]):
        eel.showErrorOnLabelTrainPage("Project not loaded or invalid names provided.")()
        return

    print(f"Spawning worker to sync labels from '{source_dataset_name}' to '{target_dataset_name}'")
    # We will create this worker function next
    eel.spawn(workthreads.sync_labels_worker, source_dataset_name, target_dataset_name)


def start_threads():
    """Initializes and starts all background worker threads."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing worker threads on device: {device}")

    # The DinoEncoder will be initialized only after a project is loaded,
    # using the project-specific settings.
    gui_state.dino_encoder = None

    gui_state.encode_lock = threading.Lock()
    # The EncodeThread is  initialized without the encoder model.
    gui_state.encode_thread = EncodeThread()
    gui_state.encode_thread.daemon = True
    gui_state.encode_thread.start()

    gui_state.training_thread = TrainingThread(device)
    gui_state.training_thread.daemon = True
    gui_state.training_thread.start()

    gui_state.classify_lock = threading.Lock()
    gui_state.classify_thread = ClassificationThread(device)
    gui_state.classify_thread.daemon = True
    gui_state.classify_thread.start()
    
    print("All background threads started.")

def start_recording_watcher():
    """Initializes and starts the file system watcher."""
    global file_watcher_handler # Declare we are using the global variable
    if not gui_state.proj or not os.path.exists(gui_state.proj.recordings_dir): return
    if gui_state.recording_observer and gui_state.recording_observer.is_alive(): return

    file_watcher_handler = VideoFileWatcher() # Assign the instance to our global var
    gui_state.recording_observer = Observer()
    gui_state.recording_observer.schedule(file_watcher_handler, gui_state.proj.recordings_dir, recursive=True)
    gui_state.recording_observer.start()
    log_message(f"Recording watcher started on: {gui_state.proj.recordings_dir}", "INFO")


def stop_threads():
    """Attempts to gracefully stop all running background threads."""
    log_message("Attempting to stop all worker threads...", "INFO")
    if gui_state.recording_observer and gui_state.recording_observer.is_alive():
        gui_state.recording_observer.stop()
        gui_state.recording_observer.join(timeout=2)

    for name, thread in [("Encode", gui_state.encode_thread), 
                         ("Classification", gui_state.classify_thread), 
                         ("Training", gui_state.training_thread)]:
        if thread and thread.is_alive():
            try:
                thread.raise_exception()
                thread.join(timeout=2)
                log_message(f"{name}Thread stopped.", "INFO")
            except Exception as e:
                log_message(f"Error stopping {name}Thread: {e}", "ERROR")
                
# =================================================================
# LABEL SYNCING FUNCTIONS
# =================================================================

def start_label_sync(source_dataset_name: str, target_dataset_name: str):
    """
    (LAUNCHER) Spawns the label synchronization worker in a background thread.
    This function is called from other modules to safely start the process.
    """
    log_message(f"Spawning worker to sync labels from '{source_dataset_name}' to '{target_dataset_name}'", "INFO")
    eel.spawn(sync_labels_worker, source_dataset_name, target_dataset_name)


def sync_labels_worker(source_dataset_name: str, target_dataset_name: str):
    """
    (WORKER) Rebuilds the labels.yaml for an augmented dataset from its source.
    This is a fast, file-only operation.
    """
    try:
        log_message(f"Starting label sync for '{target_dataset_name}'...", "INFO")
        source_dataset = gui_state.proj.datasets.get(source_dataset_name)
        target_dataset = gui_state.proj.datasets.get(target_dataset_name)

        if not source_dataset or not target_dataset:
            raise ValueError("Could not find source or target dataset for sync.")

        # 1. Re-read the source labels to ensure they are the absolute latest
        with open(source_dataset.labels_path, 'r') as f:
            source_labels_data = yaml.safe_load(f)

        # 2. Start with a fresh copy of the source labels for the new target
        new_target_labels = source_labels_data.copy()
        
        # 3. Create the list of augmented labels by re-mapping video paths
        augmented_labels_to_add = []
        all_source_instances = [inst for behavior_list in source_labels_data.get("labels", {}).values() for inst in behavior_list]

        for instance in all_source_instances:
            new_instance = instance.copy()
            relative_video_path = new_instance.get("video", "")
            if not relative_video_path:
                continue

            base_name, ext = os.path.splitext(relative_video_path)
            augmented_video_rel_path = f"{base_name}_aug{ext}"
            
            # Check if the augmented video file actually exists before adding its label
            augmented_video_abs_path = os.path.join(gui_state.proj.path, augmented_video_rel_path)
            if os.path.exists(augmented_video_abs_path):
                new_instance["video"] = augmented_video_rel_path
                augmented_labels_to_add.append(new_instance)
            else:
                log_message(f"Skipping sync for missing augmented video: {augmented_video_rel_path}", "WARN")

        # 4. Add the new augmented labels to the target dictionary
        for inst in augmented_labels_to_add:
            behavior_name = inst.get("label")
            if behavior_name in new_target_labels.get("labels", {}):
                new_target_labels["labels"][behavior_name].append(inst)

        # 5. Overwrite the target labels.yaml with the new, complete data
        with open(target_dataset.labels_path, "w") as f:
            yaml.dump(new_target_labels, f, allow_unicode=True)

        log_message(f"Successfully synced labels from '{source_dataset_name}' to '{target_dataset_name}'.", "INFO")
        
        # Refresh the UI to show any changes in instance counts
        eel.refreshAllDatasets()()

    except Exception as e:
        log_message(f"Label sync failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        eel.showErrorOnLabelTrainPage(f"Label sync failed: {e}")()
        
def get_encoding_queue_status():
    """
    Returns the current status of the encoding queue for UI synchronization.
    """
    # We need to access the tracking variables from the EncodeThread instance
    if gui_state.encode_thread:
        thread = gui_state.encode_thread
        # It's safer to access these attributes directly, as they are simple integers
        # and don't require a lock for a quick read.
        processed = thread.tasks_processed_in_batch
        total = thread.total_initial_tasks
        
        # This check is important. If total is 0, it means no batch is running.
        if total > 0:
            return {'processed': processed, 'total': total}
            
    return {'processed': 0, 'total': 0}