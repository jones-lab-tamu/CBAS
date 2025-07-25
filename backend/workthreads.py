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

# Third-party imports
import torch
import matplotlib
matplotlib.use("Agg")  # Set backend for non-GUI thread compatibility
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import numpy as np

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
    # Accept an encoder object, not a string
    def __init__(self, encoder_model: cbas.DinoEncoder):
        super().__init__()
        # Store the received model and its device
        self.encoder = encoder_model
        self.device = self.encoder.device
        self.cuda_stream = torch.cuda.Stream(device=self.device) if self.device.type == 'cuda' else None
        self.total_initial_tasks = 0
        self.tasks_processed_in_batch = 0

    def run(self):
        """The main loop for the thread. Pulls tasks from the global queue."""
        while True:
            file_to_encode = None
            with gui_state.encode_lock:
                if gui_state.encode_tasks:
                    if self.total_initial_tasks == 0:
                        self.total_initial_tasks = len(gui_state.encode_tasks)
                        self.tasks_processed_in_batch = 0
                    
                    file_to_encode = gui_state.encode_tasks.pop(0)

            if file_to_encode:
                video_basename = os.path.basename(file_to_encode)
                log_message(f"Starting encoding for: {video_basename}", "INFO")

                # --- Nested callback for dual progress reporting ---
                def progress_updater(current_file_percent):
                    status = {
                        "overall_processed": self.tasks_processed_in_batch,
                        "overall_total": self.total_initial_tasks,
                        "current_percent": current_file_percent,
                        "current_file": video_basename
                    }
                    eel.update_global_encoding_progress(status)()

                    # Also log to console/log panel, but less frequently
                    nonlocal last_reported_percent
                    current_increment = int(current_file_percent // 10)
                    last_increment = int(last_reported_percent // 10)
                    if current_increment > last_increment:
                        log_message(f"Encoding '{video_basename}': {int(current_file_percent)}% complete...", "INFO")
                        last_reported_percent = current_file_percent

                last_reported_percent = -1
                progress_updater(0) # Send an initial 0% update for the current file

                # Run the encoding process
                if self.cuda_stream:
                    with torch.cuda.stream(self.cuda_stream):
                        out_file = cbas.encode_file(self.encoder, file_to_encode, progress_callback=progress_updater)
                else:
                    out_file = cbas.encode_file(self.encoder, file_to_encode, progress_callback=progress_updater)

                if out_file:
                    log_message(f"Finished encoding: {video_basename}", "INFO")
                    self.tasks_processed_in_batch += 1
                    with gui_state.classify_lock:
                        if out_file not in gui_state.classify_tasks:
                            gui_state.classify_tasks.append(out_file)
                else:
                    log_message(f"Failed to encode: {video_basename}", "WARN")
                    self.tasks_processed_in_batch += 1

                # Send a final 100% update for the file that just finished
                progress_updater(100)

            else:
                # This block runs when the queue is empty
                if self.total_initial_tasks > 0:
                    status = {"overall_total": 0} # Signal to hide the bar
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
        
        # --- Use threading.Event for clearer state management ---
        self.new_task_event = threading.Event()
        self.torch_model = None
        self.model_meta = None

    def start_inferring(self, model_obj: cbas.Model, files_to_process: list[str]):
        """
        Loads a model and prepares the thread to start processing classification tasks.
        This is now the single entry point to give a new job to the thread.
        """
        # It's safer to not hold the lock while loading the model.
        log_message(f"Loading model '{model_obj.name}' for classification...", "INFO")
        try:
            weights = torch.load(model_obj.weights_path, map_location=self.device, weights_only=True)
            self.torch_model = classifier_head.classifier(
                in_features=768,
                out_features=len(model_obj.config["behaviors"]),
                seq_len=model_obj.config["seq_len"],
            )
            self.torch_model.load_state_dict(weights)
            self.torch_model.to(self.device).eval()
            self.model_meta = model_obj
            
            # --- Add files to the queue here and set the event ---
            with gui_state.classify_lock:
                gui_state.classify_tasks.clear() # Clear any old tasks
                gui_state.classify_tasks.extend(files_to_process)
            
            self.new_task_event.set() # Signal the run() loop to start processing
            log_message(f"Model '{model_obj.name}' loaded. {len(files_to_process)} tasks queued.", "INFO")

        except Exception as e:
            log_message(f"Error loading model '{model_obj.name}': {e}", "ERROR")
            # Ensure we don't proceed if loading fails
            self.torch_model = None
            self.model_meta = None
            self.new_task_event.clear()


    def run(self):
        """The main loop for the thread. It now waits for a signal to start work."""
        while True:
            # Wait for the new_task_event to be set. This is a non-blocking wait.
            # The thread will sleep efficiently until start_inferring() is called.
            self.new_task_event.wait()

            # Once woken up, check if a model is actually loaded
            if not self.torch_model or not self.model_meta:
                log_message("Classification thread woken up but no model is loaded. Resetting.", "WARN")
                self.new_task_event.clear() # Clear the event and go back to waiting
                continue

            total_files = 0
            with gui_state.classify_lock:
                total_files = len(gui_state.classify_tasks)

            if total_files > 0:
                eel.updateTrainingStatusOnUI(self.model_meta.name, f"Processing {total_files} files...")()

            # --- Main processing loop for the current batch of tasks ---
            while True:
                file_to_classify = None
                with gui_state.classify_lock:
                    if gui_state.classify_tasks:
                        file_to_classify = gui_state.classify_tasks.pop(0)
                        # --- Update progress here ---
                        remaining_count = len(gui_state.classify_tasks)
                        percent_done = ((total_files - remaining_count) / total_files) * 100
                        eel.updateDatasetLoadProgress(self.model_meta.name, percent_done)()
                    else:
                        # No more tasks in the queue, exit the processing loop
                        break

                if file_to_classify:
                    log_message(f"Classifying: {os.path.basename(file_to_classify)} with model '{self.model_meta.name}'", "INFO")

                    if self.cuda_stream:
                        with torch.cuda.stream(self.cuda_stream):
                            cbas.infer_file(file_to_classify, self.torch_model, self.model_meta.name, self.model_meta.config["behaviors"], self.model_meta.config["seq_len"], device=self.device)
                    else:
                        cbas.infer_file(file_to_classify, self.torch_model, self.model_meta.name, self.model_meta.config["behaviors"], self.model_meta.config["seq_len"], device=self.device)

            # --- This block runs after the queue is empty ---
            if self.model_meta: # Check if a model was actually run
                log_message(f"Inference queue for model '{self.model_meta.name}' is empty. Classification complete.", "INFO")
                eel.updateTrainingStatusOnUI(self.model_meta.name, "Inference complete.")()

                # After finishing, tell the project to re-scan the directories to find the new files.
                if gui_state.proj:
                    log_message("Reloading project data to discover new classification files.", "INFO")
                    gui_state.proj.reload()

            # Reset state for the next job
            self.torch_model = None
            self.model_meta = None
            self.new_task_event.clear()

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
        # Reset the cancel event every time a new task is queued
        self.cancel_event.clear()
        with self.training_queue_lock:
            self.training_queue.append(task)
        log_message(f"Queued training task for dataset: {task.name}", "INFO")

    def run(self):
        """The main loop for the thread. Pulls tasks from its internal queue."""
        while True:
            if self.cancel_event.is_set():
                time.sleep(1) # Wait if cancelled
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
        """Orchestrates the training process: data loading, trials, and result saving."""
        try:
            if self.cancel_event.is_set():
                log_message(f"Skipping task for '{task.name}' due to cancellation.", "WARN")
                return

            eel.spawn(eel.updateTrainingStatusOnUI(task.name, "Loading dataset..."))
            log_message(f"Loading and processing dataset '{task.name}' for training...", "INFO")
            def update_progress(p): eel.spawn(eel.updateDatasetLoadProgress(task.name, p))
            
            train_insts, test_insts = None, None # Initialize
            if task.training_method == "weighted_loss":
                train_ds, test_ds, weights, train_insts, test_insts = gui_state.proj.load_dataset_for_weighted_loss(task.name, seq_len=task.sequence_length, progress_callback=update_progress)
            else:
                train_ds, test_ds, train_insts, test_insts = gui_state.proj.load_dataset(task.name, seq_len=task.sequence_length, progress_callback=update_progress)
                weights = None
            
            if train_ds is None or test_ds is None or train_insts is None or test_insts is None:
                raise ValueError("Dataset loading returned None. Check for empty labels or other data issues.")
            
            eel.spawn(eel.updateDatasetLoadProgress(task.name, 100))
            log_message(f"Dataset '{task.name}' loaded successfully.", "INFO")
 
            if not train_ds or len(train_ds) == 0:
                if test_ds and len(test_ds) > 0:
                    error_message = (
                        f"Training failed for '{task.name}' because the training set is empty. "
                        "This typically happens when only one video has been labeled, forcing the entire video "
                        "into the test set. Please label a second video or use the 'Augment' feature."
                    )
                else:
                    error_message = f"Training failed for '{task.name}'. The dataset contains no valid labeled instances to train on."

                log_message(error_message, "ERROR")
                eel.spawn(eel.updateTrainingStatusOnUI(task.name, error_message))
                eel.spawn(eel.updateDatasetLoadProgress(task.name, -1))
                return
        except Exception as e:
            log_message(f"Critical error loading dataset {task.name}: {e}", "ERROR")
            eel.spawn(eel.updateTrainingStatusOnUI(task.name, f"Error loading dataset: {e}"))
            eel.spawn(eel.updateDatasetLoadProgress(task.name, -1))
            return

        best_model, best_reports, best_epoch = None, None, -1
        best_f1 = -1.0
        NUM_TRIALS = 10

        # PROGRESS UPDATER
        def training_progress_updater(message: str):
            # Check if the message contains an F1 score from the cbas module
            f1_match = re.search(r"Val F1: ([\d\.]+)", message)
            
            # The text we will actually display on the UI
            display_message = ""

            if f1_match:
                # This is an F1 score update from within an epoch
                epoch_f1 = float(f1_match.group(1))
                # We show the BEST F1 score seen so far across all trials and epochs
                display_message = f"Trial {i + 1}/{NUM_TRIALS}... Best F1: {max(best_f1, epoch_f1):.4f}"
            else:
                # This is a message like "Training Epoch X/Y..."
                # We still want to show the trial count and the best F1 so far.
                # If best_f1 is -1.0, we display N/A.
                f1_text = f"{best_f1:.4f}" if best_f1 >= 0 else "N/A"
                display_message = f"Trial {i + 1}/{NUM_TRIALS}... Best F1: {f1_text}"
            
            # We also pass the original message, which contains the epoch info for the progress bar
            eel.spawn(eel.updateTrainingStatusOnUI(task.name, display_message, message))

        for i in range(NUM_TRIALS):
            if self.cancel_event.is_set():
                log_message(f"Cancelling training for '{task.name}' before trial {i+1}.", "WARN")
                eel.spawn(eel.updateTrainingStatusOnUI(task.name, "Training cancelled."))
                return

            log_message(f"Starting training trial {i + 1}/{NUM_TRIALS} for '{task.name}'.", "INFO")
            
            # Initial message for the trial before epochs start
            f1_text = f"{best_f1:.4f}" if best_f1 >= 0 else "N/A"
            eel.spawn(eel.updateTrainingStatusOnUI(task.name, f"Trial {i + 1}/{NUM_TRIALS}... Best F1: {f1_text}", f"Trial {i + 1}/{NUM_TRIALS}..."))

            trial_model, trial_reports, trial_best_epoch = cbas.train_lstm_model(
                train_ds, test_ds, task.sequence_length, task.behaviors, self.cancel_event,
                lr=task.learning_rate, batch_size=task.batch_size,
                epochs=task.epochs, device=self.device, class_weights=weights,
                patience=task.patience,
                progress_callback=training_progress_updater
            )

            if trial_model and trial_reports and trial_best_epoch != -1:
                best_epoch_report = trial_reports[trial_best_epoch]
                f1 = best_epoch_report.val_report.get("weighted avg", {}).get("f1-score", -1.0)
                                              
                if f1 > best_f1:
                    log_message(f"New best model in Trial {i + 1} with F1: {f1:.4f}", "INFO")
                    best_f1, best_model, best_reports, best_epoch = f1, trial_model, trial_reports, trial_best_epoch

            if best_model is None and trial_model is not None:
                log_message("No validation data available to select best model. Saving model from final trial.", "WARN")
                best_model = trial_model
                best_reports = trial_reports
                best_epoch = trial_best_epoch

        if best_model:
            self._save_training_results(task, best_model, best_reports, best_epoch, train_insts, test_insts)
        else:
            log_message(f"Training failed for '{task.name}' after {NUM_TRIALS} trials. No valid model could be trained.", "ERROR")
            eel.spawn(eel.updateTrainingStatusOnUI(task.name, f"Training failed after {NUM_TRIALS} trials."))

    def _save_training_results(self, task, model, reports, best_epoch_idx, train_insts, test_insts):
        """Saves the best model, performance reports, and plots."""
        # Get the specific report for the best epoch
        best_report_obj = reports[best_epoch_idx]
        
        # The report to save to disk and display should be from the VALIDATION set
        best_validation_report = best_report_obj.val_report
        best_validation_cm = best_report_obj.val_cm
        
        # Check if the validation report is populated before trying to access it.
        if best_validation_report and 'weighted avg' in best_validation_report:
            val_f1_score = best_validation_report['weighted avg'].get('f1-score', 0.0)
            log_message(f"Saving results for best model (Validation F1: {val_f1_score:.4f})...", "INFO")
            # The UI update is now safely inside this block
            eel.updateTrainingStatusOnUI(task.name, f"Training complete. Best Validation F1: {val_f1_score:.4f}")()
        else:
            # This branch executes if there was no validation data.
            log_message("Saving results for best model (no validation data was available).", "INFO")
            # The UI update is also safely inside this block
            eel.updateTrainingStatusOnUI(task.name, "Training complete (no validation data).")()

        # Update the UI metrics table. This function is now robust to an empty report.
                              
        model_dir = os.path.join(gui_state.proj.models_dir, task.name)
        os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(model_dir, "model.pth"))
        
        with open(os.path.join(model_dir, "config.yaml"), "w") as f:
            yaml.dump({"seq_len": task.sequence_length, "behaviors": task.behaviors}, f)
        
        # Save the validation report to the dataset folder
        with open(os.path.join(task.dataset.path, "performance_report.yaml"), 'w') as f:
            yaml.dump(best_validation_report, f)
        
        # Save the validation confusion matrix
        save_confusion_matrix_plot(best_validation_cm, os.path.join(task.dataset.path, "confusion_matrix_BEST.png"))
        
        cm_dir = os.path.join(task.dataset.path, "epoch_confusion_matrices")
        os.makedirs(cm_dir, exist_ok=True)
        for i, r in enumerate(reports):
            # Save the validation CM for each epoch for potential review
            if r.val_cm.size > 0:
                save_confusion_matrix_plot(r.val_cm, os.path.join(cm_dir, f"epoch_{i+1}_cm.png"))

        # Now, call the plotting function with the full list of epoch reports
        for metric in ["f1-score", "recall", "precision"]:
            plot_report_list_metric(reports, metric, task.behaviors, task.dataset.path)
            
        gui_state.proj.models[task.name] = cbas.Model(model_dir)
        
        # Update the dataset's config file to link it to the newly created model.
        task.dataset.config['model'] = task.name 
        with open(task.dataset.config_path, 'w') as f:
            yaml.dump(task.dataset.config, f, allow_unicode=True)       
                
        # The UI metrics should reflect the model's performance on unseen (validation) data
        self._update_metrics_in_ui(task.name, best_validation_report, task.behaviors, task.dataset, train_insts, test_insts)
        
        # This log message is now the very last thing that happens.
        log_message(f"Training for '{task.name}' complete. Model and reports saved.", "INFO")

    def _update_metrics_in_ui(self, dataset_name, report_dict, behaviors, dataset_obj, train_insts, test_insts):
        """
        Helper to calculate final instance/frame counts, check for imbalance (both low and high),
        push all metrics to the UI table, and save them to the dataset's config file.
        """
        from collections import Counter
        import numpy as np

        # --- 1. Count INSTANCES and FRAMES per behavior ---
        train_instance_counts = Counter(inst['label'] for inst in train_insts)
        test_instance_counts = Counter(inst['label'] for inst in test_insts)
        train_frame_counts = Counter()
        for inst in train_insts:
            train_frame_counts[inst['label']] += (inst['end'] - inst['start'] + 1)
        test_frame_counts = Counter()
        for inst in test_insts:
            test_frame_counts[inst['label']] += (inst['end'] - inst['start'] + 1)

        # --- 2. Check for Class Imbalance (Low and High) ---
        total_behaviors = len(behaviors)
        if total_behaviors > 1:
            avg_train_inst = sum(train_instance_counts.values()) / total_behaviors
            avg_test_inst = sum(test_instance_counts.values()) / total_behaviors
            avg_train_frame = sum(train_frame_counts.values()) / total_behaviors
            avg_test_frame = sum(test_frame_counts.values()) / total_behaviors
            
            LOW_THRESHOLD = 0.25
            HIGH_THRESHOLD = 3.0 
        else:
            avg_train_inst, avg_test_inst, avg_train_frame, avg_test_frame = 0, 0, 0, 0

        # --- 3. Update all metrics and counts ---
        for b in behaviors:
            # Safely get metrics from the report_dict, which might be empty
            metrics = report_dict.get(b, {})
            
            f1 = round(metrics.get("f1-score", 0), 2)
            recall = round(metrics.get("recall", 0), 2)
            precision = round(metrics.get("precision", 0), 2)
            
            train_n_inst = train_instance_counts.get(b, 0)
            test_n_inst = test_instance_counts.get(b, 0)
            train_n_frame = train_frame_counts.get(b, 0)
            test_n_frame = test_frame_counts.get(b, 0)

            # --- Build structured dictionaries for the UI ---
            train_data = {
                'inst': train_n_inst,
                'frame': int(train_n_frame),
                'inst_status': 'none',
                'frame_status': 'none'
            }
            if total_behaviors > 1:
                # Check for low representation (critical warning)
                if train_n_inst < avg_train_inst * LOW_THRESHOLD: train_data['inst_status'] = 'low'
                # Check for high representation (informational notice)
                elif train_n_inst > avg_train_inst * HIGH_THRESHOLD: train_data['inst_status'] = 'high'
                
                if train_n_frame < avg_train_frame * LOW_THRESHOLD: train_data['frame_status'] = 'low'
            
            test_data = {
                'inst': test_n_inst,
                'frame': int(test_n_frame),
                'inst_status': 'none',
                'frame_status': 'none'
            }
            if total_behaviors > 1:
                 if test_n_inst < avg_test_inst * LOW_THRESHOLD: test_data['inst_status'] = 'low'
                 if test_n_frame < avg_test_frame * LOW_THRESHOLD: test_data['frame_status'] = 'low'

            # Save raw values to the config file
            # Get the display strings we already created
            train_display = f"{train_n_inst} ({int(train_n_frame)})"
            test_display = f"{test_n_inst} ({int(test_n_frame)})"

            # Save the final, formatted DISPLAY STRING to the config file
            dataset_obj.update_metric(b, "Train #", train_display)
            dataset_obj.update_metric(b, "Test #", test_display)
            
            # Keep saving raw numbers for other metrics
            dataset_obj.update_metric(b, "F1 Score", f1)
            dataset_obj.update_metric(b, "Recall", recall)
            dataset_obj.update_metric(b, "Precision", precision)
            # ... (we can optionally remove the now-redundant raw count saving)

            # Send all new values to the live UI via Eel
            eel.updateMetricsOnPage(dataset_name, b, "F1 Score", f1, "none")()
            eel.updateMetricsOnPage(dataset_name, b, "Recall", recall, "none")()
            eel.updateMetricsOnPage(dataset_name, b, "Precision", precision, "none")()
            eel.updateMetricsOnPage(dataset_name, b, "Train #", train_data, "none")()
            eel.updateMetricsOnPage(dataset_name, b, "Test #", test_data, "none")()
    
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
    def __init__(self, name, dataset, behaviors, batch_size, learning_rate, epochs, sequence_length, training_method, patience): # Add patience
        self.name, self.dataset, self.behaviors = name, dataset, behaviors
        self.batch_size, self.learning_rate = batch_size, learning_rate
        self.epochs, self.sequence_length = epochs, sequence_length
        self.training_method = training_method
        self.patience = patience # Store patience

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

def save_confusion_matrix_plot(cm_data: np.ndarray, path: str):
    """Saves a confusion matrix plot to a file."""
    if cm_data.size == 0: return
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_data)
    fig, ax = plt.subplots(figsize=(8, 7))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, xticks_rotation='vertical')
    ax.set_title("Confusion Matrix")
    plt.tight_layout(); plt.savefig(path); plt.close(fig)


def plot_report_list_metric(reports: list, metric: str, behaviors: list, out_dir: str):
    """Plots a given metric over epochs for all behaviors, for both training and validation sets."""
    if not reports: return
    plt.figure(figsize=(10, 7))
    epochs = range(1, len(reports) + 1)
    
    # Use seaborn's color palette for distinct colors
    colors = plt.cm.get_cmap('tab10', len(behaviors))

    # Plot Train and Validation metrics for each behavior
    for i, b_name in enumerate(behaviors):
        train_values = [r.train_report.get(b_name, {}).get(metric, np.nan) for r in reports]
        val_values = [r.val_report.get(b_name, {}).get(metric, np.nan) for r in reports]
        
        if not all(np.isnan(v) for v in train_values):
            plt.plot(epochs, train_values, marker='o', linestyle='-', label=f'{b_name} (Train)', color=colors(i))
        if not all(np.isnan(v) for v in val_values):
            plt.plot(epochs, val_values, marker='x', linestyle='--', label=f'{b_name} (Val)', color=colors(i))

    # Plot Weighted Averages
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
    plt.tight_layout(rect=[0, 0, 0.8, 1]) # Adjust rect to make space for the legend
    plt.savefig(os.path.join(out_dir, f"{metric.replace(' ', '_')}_epochs_plot.png"))
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

    # Initialize the DINO Encoder here
    print("Loading DINOv2 model for encoding thread...")
    dino_encoder = cbas.DinoEncoder(device=device)
    print("DINOv2 model loaded.")

    gui_state.encode_lock = threading.Lock()
    # Pass the initialized model object to the thread
    gui_state.encode_thread = EncodeThread(encoder_model=dino_encoder)
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