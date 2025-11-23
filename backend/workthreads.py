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
from __future__ import annotations # For forward reference in type hints
import threading
import time
import ctypes
from datetime import datetime
import os
import yaml
import re
import traceback
import json
from collections import defaultdict
import subprocess
import json

try:
    import importlib.metadata as pkg_resources
except ImportError:
    import pkg_resources # Fallback for Python < 3.8
from backend.splits import RandomSplitProvider # Import the new class

# Third-party imports
import torch
import matplotlib
matplotlib.use("Agg")
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

_last_restart_times = {}
file_watcher_handler = None

# Helper: get current commit hash safely (works even when git is absent or not in a repo)
def _safe_git_hash():
    try:
        # The command will be run from the current working directory.
        # stderr is redirected to DEVNULL to suppress "fatal: not a git repository" messages.
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            # Ensure the command runs from the script's directory context if needed
            cwd=os.path.dirname(os.path.abspath(__file__)) 
        ).decode('utf-8').strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        # This will catch errors if git is not installed or if it's not a repo.
        return "unknown"

# =================================================================
# LOGGING HELPER
# =================================================================

def log_message(message: str, level: str = "INFO"):
    """
    Puts a formatted message into the global log queue for the UI and
    also prints it to the console for developer debugging.
    Now safe to run in headless mode.
    """
    timestamp = datetime.now().strftime('%H:%M:%S')
    formatted_message = f"[{timestamp}] [{level}] {message}"
    
    # Always print to the console.
    if gui_state.print_lock:
        with gui_state.print_lock:
            print(formatted_message)
    else:
        print(formatted_message)
    
    # Only attempt to queue a message for the GUI if we are NOT in headless mode.
    if not gui_state.HEADLESS_MODE and gui_state.log_queue:
        try:
            # Use a non-blocking put to prevent hangs if the GUI is busy.
            gui_state.log_queue.put_nowait(formatted_message)
        except queue.Full:
            pass # It's okay to drop UI log messages if the queue is full.


# =================================================================
# WORKER THREAD CLASSES
# =================================================================

def fit_temperature(model, val_loader, device):
    """Finds the optimal temperature for calibrating model confidence."""
    
    # Ensure the model is on the correct device before any operations.
    model.to(device)

    model.eval()
    T = torch.nn.Parameter(torch.ones(1, device=device))
    optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=50)
    criterion = torch.nn.CrossEntropyLoss()
    
    all_logits, all_labels = [], []
    with torch.no_grad():
        for d, l in val_loader:
            # The in-RAM loader provides clean data, so no need to mask for l != -1
            logits, _ = model(d.to(device))
            all_logits.append(logits)
            all_labels.append(l)
    
    if not all_logits: return 1.0
    
    all_logits = torch.cat(all_logits).detach()
    all_labels = torch.cat(all_labels).to(device)

    def eval_loss():
        optimizer.zero_grad()
        # Use softplus for stability
        temp = torch.clamp(torch.nn.functional.softplus(T) + 1e-3, max=10.0)
        loss = criterion(all_logits / temp, all_labels)
        loss.backward()
        return loss
    
    optimizer.step(eval_loss)
    final_temp = torch.clamp(torch.nn.functional.softplus(T) + 1e-3, max=10.0)
    return float(final_temp.detach().cpu().item())

def _recording_monitor_worker():
    """
    (WORKER) A daemon thread that periodically checks if ffmpeg processes
    are running. If a process has terminated, it automatically restarts it.
    """
    global _last_restart_times
    RESTART_COOLDOWN = 60

    while True:
        time.sleep(5)
        
        if gui_state.proj and gui_state.proj.active_recordings:
            for name, (proc, start_time, session_name) in list(gui_state.proj.active_recordings.items()):
                if proc.poll() is not None:
                    log_message(f"Recording process for '{name}' terminated unexpectedly (exit code: {proc.returncode}).", "WARN")
                    
                    del gui_state.proj.active_recordings[name]
                    
                    current_time = time.time()
                    last_restart = _last_restart_times.get(name, 0)
                    
                    if (current_time - last_restart) > RESTART_COOLDOWN:
                        log_message(f"Attempting to automatically restart recording for '{name}'...", "INFO")
                        try:
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

def augment_dataset_worker(source_dataset_name: str, new_dataset_name: str):
    """
    (WORKER) Creates a new dataset by creating augmented video files.
    This process is resumable and idempotent.
    """
    try:
        source_dataset = gui_state.proj.datasets.get(source_dataset_name)
        if not source_dataset:
            raise ValueError(f"Source dataset '{source_dataset_name}' not found.")

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
            
            processed_videos[source_video_path] = output_video_path

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
        traceback.print_exc()
        eel.showErrorOnLabelTrainPage(f"Augmentation failed: {e}")()
    finally:
        eel.update_augmentation_progress(-1)()

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
            if not gui_state.proj or not gui_state.dino_encoder:
                time.sleep(2)
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
        if hasattr(self, "_thread_id"): return self._thread_id
        for id, thread in threading._active.items():
            if thread is self: return id
        return None

    def raise_exception(self):
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
        log_message(f"Loading model bundle '{model_name}'...", "INFO")
        try:
            model_obj = gui_state.proj.models.get(model_name)
            if not model_obj: raise ValueError(f"Model '{model_name}' not found.")

            meta_path = os.path.join(model_obj.path, "model_meta.json")
            if not os.path.exists(meta_path):
                log_message("Legacy model detected (no meta.json). Loading with default settings.", "WARN")
                meta = {
                    "head_architecture_version": "ClassifierLegacyLSTM",
                    "hyperparameters": model_obj.config,
                    "encoder_model_identifier": gui_state.proj.encoder_model_identifier
                }
            else:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)

            project_encoder = gui_state.proj.encoder_model_identifier
            model_encoder = meta.get("encoder_model_identifier")
            if model_encoder and model_encoder != project_encoder:
                err_msg = f"Encoder mismatch! Project is for '{project_encoder}', but model was trained with '{model_encoder}'. Please re-encode videos."
                log_message(err_msg, "ERROR")
                try:
                    eel.showErrorOnLabelTrainPage(err_msg)()
                except Exception as e:
                    print(f"Eel call failed: {e}")
                return None, None

            arch_version = meta.get("head_architecture_version", "ClassifierLegacyLSTM")
            hparams = meta.get("hyperparameters", {})
            
            # Ensure essential hparams exist, falling back to legacy config if needed
            if "behaviors" not in hparams: hparams["behaviors"] = model_obj.config.get("behaviors", [])
            if "seq_len" not in hparams: hparams["seq_len"] = model_obj.config.get("seq_len", 31)
            
            meta["hyperparameters"] = hparams # Ensure meta is fully populated for later use

            weights_path = os.path.join(model_obj.path, "model.pth")
            try:
                weights = torch.load(weights_path, map_location="cpu", weights_only=True)
            except TypeError:
                weights = torch.load(weights_path, map_location="cpu")

            if arch_version.startswith("ClassifierLSTMDeltas"):
                # Infer missing hyperparameters from the weights file itself for robustness
                if "lstm_hidden_size" not in hparams:
                    # Infer from the size of the attention or linear layer weights
                    inferred_h_size = weights.get("attention_head.weight", weights.get("lin2.weight")).shape[1] // 2
                    hparams["lstm_hidden_size"] = inferred_h_size if inferred_h_size else 64
                if "lstm_layers" not in hparams:
                    layer_keys = [int(k.split('_l')[1].split('.')[0]) for k in weights if 'lstm.weight_ih_l' in k]
                    hparams["lstm_layers"] = max(layer_keys) + 1 if layer_keys else 1

                torch_model = classifier_head.ClassifierLSTMDeltas(
                    in_features=768,
                    out_features=len(hparams["behaviors"]),
                    seq_len=hparams["seq_len"],
                    lstm_hidden_size=hparams["lstm_hidden_size"],
                    lstm_layers=hparams["lstm_layers"],
                )
            else:
                torch_model = classifier_head.ClassifierLegacyLSTM(
                    in_features=768,
                    out_features=len(hparams["behaviors"]),
                    seq_len=hparams["seq_len"],
                )

            torch_model.load_state_dict(weights, strict=False)
            torch_model.to(self.device).eval()

            gui_state.live_inference_model_object = torch_model
            log_message(f"Model '{model_name}' loaded successfully.", "INFO")
            return torch_model, meta
        except Exception as e:
            log_message(f"Error loading model bundle '{model_name}': {e}", "ERROR")
            traceback.print_exc()
            gui_state.live_inference_model_object = None
            return None, None

    def run(self):
        """The main loop for the thread. Continuously processes the queue."""
        last_model_name = None
        torch_model, model_meta = None, None
        
        manual_inference_total = 0
        manual_inference_processed = 0

        while True:
            current_model_name = gui_state.live_inference_model_name
            if current_model_name != last_model_name:
                if current_model_name:
                    torch_model, model_meta = self._load_model(current_model_name)
                    with gui_state.classify_lock:
                        manual_inference_total = len(gui_state.classify_tasks)
                    manual_inference_processed = 0
                else:
                    gui_state.live_inference_model_object, torch_model, model_meta = None, None, None
                last_model_name = current_model_name

            file_to_classify = None
            if torch_model and model_meta:
                with gui_state.classify_lock:
                    if gui_state.classify_tasks:
                        file_to_classify = gui_state.classify_tasks.pop(0)

            if file_to_classify:
                model_name_for_ui = last_model_name
                hparams = model_meta.get("hyperparameters", {})
                behaviors_for_infer = hparams.get("behaviors", [])
                seq_len_for_infer   = hparams.get("seq_len", 31)
                temperature_for_infer = float(model_meta.get("calibration", {}).get("temperature", 1.0))
                
                log_message(f"Classifying: {os.path.basename(file_to_classify)} with model '{model_name_for_ui}'", "INFO")
                try:
                    infer_args = {
                        "file_path": file_to_classify, "model": torch_model,
                        "dataset_name": model_name_for_ui, "behaviors": behaviors_for_infer,
                        "seq_len": seq_len_for_infer, "device": self.device,
                        "temperature": temperature_for_infer,
                    }
                    
                    if self.cuda_stream:
                        with torch.cuda.stream(self.cuda_stream): cbas.infer_file(**infer_args)
                    else:
                        cbas.infer_file(**infer_args)
                    
                    eel.spawn(eel.notify_new_data_available())
                    
                    manual_inference_processed += 1
                    if manual_inference_total > 0:
                        percent_done = (manual_inference_processed / manual_inference_total) * 100
                        message = f"Processing {manual_inference_processed}/{manual_inference_total}: {os.path.basename(file_to_classify)}"
                        eel.updateInferenceProgress(model_name_for_ui, percent_done, message)()

                        if manual_inference_processed >= manual_inference_total:
                            log_message(f"Inference queue for model '{model_name_for_ui}' is empty.", "INFO")
                            eel.updateInferenceProgress(model_name_for_ui, 100, "Inference complete.")()
                            if gui_state.proj: gui_state.proj.reload()
                            manual_inference_total, manual_inference_processed = 0, 0
                            gui_state.live_inference_model_name = None

                except Exception as e:
                    log_message(f"Failed to classify '{os.path.basename(file_to_classify)}'. Error: {e}", "ERROR")
                    traceback.print_exc()
            else:
                time.sleep(1)

    def get_id(self):
        if hasattr(self, "_thread_id"): return self._thread_id
        for id, thread in threading._active.items():
            if thread is self: return id
        return None

    def raise_exception(self):
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

    def _execute_training_task(self, task: TrainingTask, split_provider: cbas.SplitProvider = None, output_dir=None, plot_suffix='runs'):
        """Orchestrates the training process using a provided SplitProvider."""
        try:
            log_message(f"--- Starting Training Job for Dataset: {task.name} ---", "INFO")
            if not gui_state.HEADLESS_MODE:
                eel.updateTrainingStatusOnUI(task.name, "Preparing data splits...")()

            if split_provider is None:
                split_ratios = (1.0 - task.test_split - 0.15, 0.15, task.test_split if task.use_test else 0.0)
                split_provider = RandomSplitProvider(split_ratios=split_ratios)

            all_instances = [inst for b in task.behaviors for inst in task.dataset.labels.get("labels", {}).get(b, [])]
            all_subjects = list(set(os.path.dirname(inst['video']) for inst in all_instances))
            
            overall_best_model = None
            overall_best_f1 = -1.0
            all_run_reports = []
            overall_best_run_history = None
            final_split_assignments = {}

            for run_num in range(task.num_runs):
                if self.cancel_event.is_set(): break
                log_message(f"--- Starting Run {run_num + 1}/{task.num_runs} ---", "INFO")

                train_subjects, val_subjects, test_subjects = split_provider.get_split(
                    run_num, all_subjects, all_instances, task.behaviors, allow_relaxed_fallback=True
                )
                
                train_ds, val_ds, test_ds, train_insts, val_insts, test_insts, behaviors = cbas.create_datasets_from_splits(
                    gui_state.proj, task.name, train_subjects, val_subjects, test_subjects, task.sequence_length
                )

                # Relax the validation set requirement.
                # The training function itself can handle a None val_ds.
                if train_ds is None or len(train_ds) == 0:
                    log_message(f"Data splitting for run {run_num + 1} failed because the training set was empty. Skipping.", "WARN")
                    continue
                
                run_best_model_for_trials, run_best_f1_for_trials, run_best_reports_history, run_best_epoch = None, -1.0, None, -1

                for trial_num in range(task.num_trials):
                    if self.cancel_event.is_set(): break
                    log_message(f"Run {run_num + 1}, Trial {trial_num + 1}/{task.num_trials} for '{task.name}'.", "INFO")
                    
                    def training_progress_updater(message: str):
                        if not gui_state.HEADLESS_MODE:
                            f1_match = re.search(r"Val F1: ([\d\.]+)", message)
                            current_best = run_best_f1_for_trials
                            if f1_match:
                                current_best = max(run_best_f1_for_trials, float(f1_match.group(1)))
                            f1_text = f"{current_best:.4f}" if current_best >= 0 else "N/A"
                            display_message = f"Run {run_num + 1}/{task.num_runs}, Trial {trial_num + 1}/{task.num_trials}... Best Val F1: {f1_text}"
                            eel.updateTrainingStatusOnUI(task.name, display_message, message)()

                    weights = None
                    if task.training_method == "weighted_loss":
                        weights = cbas.compute_class_weights_from_instances(train_insts, behaviors)
                    elif task.training_method == 'custom_weights' and task.custom_weights:
                        weights = [task.custom_weights.get(b, 1.0) for b in behaviors]

                    trial_model, trial_reports, trial_best_epoch = cbas.train_lstm_model(
                        train_ds, val_ds, task.sequence_length, behaviors, self.cancel_event,
                        lr=task.learning_rate, batch_size=task.batch_size,
                        epochs=task.epochs, device=self.device, class_weights=weights,
                        patience=task.patience, progress_callback=training_progress_updater,
                        optimization_target=task.optimization_target,
                        weight_decay=task.weight_decay,
                        label_smoothing=task.label_smoothing,
                        lstm_hidden_size=task.lstm_hidden_size,
                        lstm_layers=task.lstm_layers
                    )

                    if trial_model and trial_reports and trial_best_epoch != -1:
                        # Handle the case where there is no validation report
                        f1 = -1.0
                        if trial_reports[trial_best_epoch].val_report:
                            f1 = trial_reports[trial_best_epoch].val_report.get(task.optimization_target, {}).get("f1-score", -1.0)
                        
                        # If there's no validation set, we still need to save the model from the first trial.
                        if f1 > run_best_f1_for_trials or run_best_model_for_trials is None:
                            run_best_f1_for_trials = f1
                            run_best_model_for_trials = trial_model
                            run_best_reports_history = trial_reports
                            run_best_epoch = trial_best_epoch
                
                if self.cancel_event.is_set(): break

                if run_best_model_for_trials:
                    run_winner_report = {
                        "best_epoch": run_best_epoch,
                        "validation_report": run_best_reports_history[run_best_epoch].val_report if run_best_reports_history else {},
                        "validation_cm": run_best_reports_history[run_best_epoch].val_cm if run_best_reports_history else np.array([]),
                        "test_report": {},
                        "test_cm": np.array([])
                    }

                    if task.use_test and test_ds and len(test_ds) > 0:
                        log_message(f"Run {run_num + 1}: Evaluating best model on held-out test set...", "INFO")
                        if not gui_state.HEADLESS_MODE:
                            eel.updateTrainingStatusOnUI(task.name, f"Run {run_num + 1}/{task.num_runs}: Evaluating on test set...")()
                        test_results = cbas.evaluate_on_split(run_best_model_for_trials, test_ds, behaviors, device=self.device)
                        run_winner_report["test_report"] = test_results["report"]
                        run_winner_report["test_cm"] = test_results["cm"]
                    
                    all_run_reports.append(run_winner_report)

                    if run_best_f1_for_trials > overall_best_f1 or overall_best_model is None:
                        log_message(f"New overall best model found in Run {run_num + 1} with Validation F1: {run_best_f1_for_trials:.4f}", "INFO")
                        overall_best_f1 = run_best_f1_for_trials
                        overall_best_model = run_best_model_for_trials
                        overall_best_run_history = run_best_reports_history
                        final_split_assignments = {
                            "master_seed": split_provider.initial_seed if isinstance(split_provider, RandomSplitProvider) else "N/A",
                            "train_groups": sorted(list(train_subjects)),
                            "val_groups": sorted(list(val_subjects)),
                            "test_groups": sorted(list(test_subjects))
                        }

            if self.cancel_event.is_set():
                log_message(f"Training for '{task.name}' was cancelled by user.", "WARN")
                if not gui_state.HEADLESS_MODE:
                    eel.updateTrainingStatusOnUI(task.name, "Training cancelled.")()
                return

            if overall_best_model and all_run_reports:
                self._save_averaged_training_results(
                    task=task,
                    best_model=overall_best_model,
                    all_run_reports=all_run_reports,
                    best_run_history=overall_best_run_history,
                    best_run_cm=None,
                    split_assignments=final_split_assignments,
                    train_insts=train_insts,
                    val_insts=val_insts,
                    test_insts=test_insts,
                    output_dir=output_dir,
                    plot_suffix=plot_suffix
                )
            else:
                log_message(f"Training failed for '{task.name}'. No valid model could be trained.", "ERROR")
                if not gui_state.HEADLESS_MODE:
                    eel.updateTrainingStatusOnUI(task.name, "Training failed.")()

        except Exception as e:
            log_message(f"Critical error during training task for {task.name}: {e}", "ERROR")
            traceback.print_exc()
            if not gui_state.HEADLESS_MODE:
                eel.updateTrainingStatusOnUI(task.name, f"Training Error: {e}")()
            
    def _generate_disagreement_report(self, task, model, train_insts):
        """
        Runs inference on the training set to find where the model disagrees
        with the human labels. Saves a report for user review.
        """
        log_message(f"Generating disagreement report for '{task.name}'...", "INFO")
        eel.updateTrainingStatusOnUI(task.name, "Analyzing model errors...")()

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

    def _save_averaged_training_results(self, task: TrainingTask, best_model, all_run_reports: list, best_run_history, best_run_cm, split_assignments: dict, train_insts: list, val_insts: list, test_insts: list, output_dir=None, plot_suffix='runs'):
        """
        Saves the final model bundle and enriched, auditable reports to a specified directory.
        """
        log_message(f"Finalizing and saving model bundle for '{task.name}'...", "INFO")
        
        if output_dir is None:
            output_dir = task.dataset.path
        
        os.makedirs(output_dir, exist_ok=True)
        
        model_name = f"{task.name}_model"
        model_dir = os.path.join(gui_state.proj.models_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        git_commit = _safe_git_hash()

        try:
            lib_versions = {
                "torch": pkg_resources.version("torch"),
                "transformers": pkg_resources.version("transformers")
            }
        except Exception:
            lib_versions = {"error": "Could not determine library versions."}

        val_manifest = gui_state.proj.convert_instances(gui_state.proj.path, val_insts, task.sequence_length, task.behaviors)
        val_ds = cbas.LazyStandardDataset(val_manifest, task.sequence_length) if val_manifest else None

        temperature = 1.0
        if val_ds and len(val_ds) > 0:
            val_loader = torch.utils.data.DataLoader(
                val_ds, 
                batch_size=task.batch_size, 
                num_workers=0, 
                pin_memory=(self.device.type == "cuda"),
                worker_init_fn=cbas.worker_init_fn
            )
            log_message("Calibrating model temperature on validation set...", "INFO")
            temperature = fit_temperature(best_model, val_loader, self.device)
            log_message(f"Optimal temperature found: {temperature:.4f}", "INFO")
        else:
            log_message("No validation set to calibrate on. Using default temperature of 1.0.", "WARN")

        torch.save(best_model.state_dict(), os.path.join(model_dir, "model.pth"))

        model_config = {
            "name": model_name,
            "behaviors": task.behaviors,
            "seq_len": task.sequence_length,
            "architecture": type(best_model).__name__
        }
        with open(os.path.join(model_dir, "config.yaml"), "w") as f:
            yaml.dump(model_config, f, allow_unicode=True)

        model_meta = {
            "model_bundle_schema": "1.0",
            "cbas_commit_hash": git_commit,
            "encoder_model_identifier": gui_state.proj.encoder_model_identifier,
            "head_architecture_version": type(best_model).__name__,
            "hyperparameters": {
                "behaviors": task.behaviors,
                "seq_len": task.sequence_length,
                "use_acceleration": bool(getattr(best_model, "use_acceleration", True)),
                "lstm_hidden_size": int(getattr(best_model.lstm, "hidden_size", 64)),
                "lstm_layers": int(getattr(best_model.lstm, "num_layers", 1)),
            },
            "training_run_info": {
                "num_runs": task.num_runs,
                "optimization_target": task.optimization_target,
            },
            "calibration": {"temperature": float(temperature)},
        }
        with open(os.path.join(model_dir, "model_meta.json"), "w") as f:
            json.dump(model_meta, f, indent=4)
        log_message(f"Wrote model metadata to '{os.path.join(model_dir, 'model_meta.json')}'.", "INFO")

        full_report = {
            "dataset_name": task.name,
            "model_name": model_name,
            "training_parameters": {
                "num_runs": task.num_runs,
                "num_trials": task.num_trials,
                "epochs": task.epochs,
                "learning_rate": task.learning_rate,
                "sequence_length": task.sequence_length,
                "optimization_target": task.optimization_target,
                "temperature": temperature,
                "weight_decay": task.weight_decay,
                "label_smoothing": task.label_smoothing,
                "lstm_hidden_size": task.lstm_hidden_size,
                "lstm_layers": task.lstm_layers
            },
            "reproducibility_info": {
                "cbas_git_commit": git_commit,
                "library_versions": lib_versions,
                "master_seed": split_assignments.get("master_seed")
            },
            "split_information": {
                "train_subjects": split_assignments.get("train_groups", []),
                "validation_subjects": split_assignments.get("val_groups", []),
                "test_subjects": split_assignments.get("test_groups", [])
            },
            "run_results": all_run_reports 
        }
        report_path = os.path.join(output_dir, "performance_report.yaml")
        with open(report_path, "w") as f:
            def numpy_dumper(data):
                if isinstance(data, np.integer): return int(data)
                if isinstance(data, np.floating): return float(data)
                if isinstance(data, np.ndarray): return data.tolist()
                return data
            yaml.dump(json.loads(json.dumps(full_report, default=numpy_dumper)), f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        log_message(f"Wrote comprehensive performance report to '{report_path}'.", "INFO")

        if all_run_reports:
            best_run_idx = int(np.argmax([r.get('validation_report', {}).get(task.optimization_target, {}).get('f1-score', -1.0) for r in all_run_reports]))
            best_run_report = all_run_reports[best_run_idx]
            
            best_val_cm = best_run_report.get("validation_cm")
            if best_val_cm is not None and np.array(best_val_cm).size > 0:
                save_confusion_matrix_plot(np.array(best_val_cm), os.path.join(output_dir, "confusion_matrix_validation_BEST.png"), labels=task.behaviors, title="Best Run: Validation Confusion Matrix")

            best_test_cm = best_run_report.get("test_cm")
            if best_test_cm is not None and np.array(best_test_cm).size > 0:
                save_confusion_matrix_plot(np.array(best_test_cm), os.path.join(output_dir, "confusion_matrix_test_FINAL.png"), labels=task.behaviors, title="Final Model: Held-Out Test Confusion Matrix")

            if best_run_history:
                for metric in ['f1-score', 'precision', 'recall']:
                    plot_report_list_metric(
                        reports=best_run_history,
                        metric=metric,
                        behaviors=task.behaviors,
                        out_dir=output_dir
                    )
                log_message(f"Epoch plots for the best run saved to '{output_dir}'.", "INFO")

            test_reports = [r.get("test_report", {}) for r in all_run_reports]
            if any(test_reports):
                plot_averaged_run_metrics(
                    reports=test_reports,
                    behaviors=task.behaviors,
                    out_dir=output_dir,
                    plot_suffix=plot_suffix
                )
                log_message(f"Per-run/replicate performance plots saved to '{output_dir}'.", "INFO")

        if os.path.normpath(output_dir) == os.path.normpath(task.dataset.path):
            ds = task.dataset
            
            # 1. Read the latest config from disk to work with
            with open(ds.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)

            # 2. Prepare the metrics block with performance scores
            metrics_block = {}
            best_run_report = all_run_reports[best_run_idx] if all_run_reports else {}
            best_val_report_dict = best_run_report.get("validation_report", {})
            best_test_report_dict = best_run_report.get("test_report", {})

            for b in task.behaviors:
                val_metrics = best_val_report_dict.get(b, {})
                test_metrics = best_test_report_dict.get(b, {})
                metrics_block[b] = {
                    "Precision": round(float(val_metrics.get("precision", 0.0)), 2),
                    "Recall":    round(float(val_metrics.get("recall", 0.0)), 2),
                    "F1 Score":  round(float(val_metrics.get("f1-score", 0.0)), 2),
                    "Test F1":   "N/A" if not task.use_test else round(float(test_metrics.get("f1-score", 0.0)), 2)
                }
            
            # 3. Calculate instance counts using the FULL dataset, not a subset
            from collections import Counter
            all_instances = [inst for b_labels in ds.labels.get("labels", {}).values() for inst in b_labels]
            all_subjects = list(set(os.path.dirname(inst['video']).replace('\\','/') for inst in all_instances))
            
            # Use the same splitting logic as the official count function to be consistent
            provider = RandomSplitProvider(seed=42, split_ratios=(0.8, 0.0, 0.2), stratify=False)
            train_subjects, _, test_subjects = provider.get_split(0, all_subjects, all_instances, task.behaviors)

            train_subject_set = set(train_subjects)
            test_subject_set = set(test_subjects)
            
            # Filter all instances based on the calculated subject splits
            all_train_insts = [inst for inst in all_instances if os.path.dirname(inst['video']).replace('\\','/') in train_subject_set]
            all_test_insts = [inst for inst in all_instances if os.path.dirname(inst['video']).replace('\\','/') in test_subject_set]
            
            train_instance_counts = Counter(inst['label'] for inst in all_train_insts)
            test_instance_counts = Counter(inst['label'] for inst in all_test_insts)
            train_frame_counts = Counter()
            for inst in all_train_insts: train_frame_counts[inst['label']] += (inst['end'] - inst['start'] + 1)
            test_frame_counts = Counter()
            for inst in all_test_insts: test_frame_counts[inst['label']] += (inst['end'] - inst['start'] + 1)
            
            # 4. Add the calculated counts to the metrics block
            for b in task.behaviors:
                metrics_block[b]["Train Inst (Frames)"] = f"{train_instance_counts.get(b, 0)} ({int(train_frame_counts.get(b, 0))})"
                metrics_block[b]["Test Inst (Frames)"] = f"{test_instance_counts.get(b, 0)} ({int(test_frame_counts.get(b, 0))})"

            # 5. Update the config dictionary and write to file ONCE
            config["metrics"] = metrics_block
            config["state"] = "trained"
            config["trained_model"] = model_name

            with open(ds.config_path, "w", encoding="utf-8") as f:
                yaml.dump(config, f, allow_unicode=True)
            
            # 6. Update the in-memory object to match
            ds.config = config
            log_message(f"Updated dataset metrics and counts in '{ds.config_path}'.", "INFO")

        log_message(f"Training for '{task.name}' complete. Artifacts saved.", "INFO")
        
        if not gui_state.HEADLESS_MODE:
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
    def __init__(self, name, dataset, behaviors, batch_size, learning_rate, epochs, sequence_length, 
                 training_method, patience, num_runs, num_trials, optimization_target, use_test, test_split, 
                 custom_weights=None, weight_decay=0.0, label_smoothing=0.0, lstm_hidden_size=64, lstm_layers=1):
        self.name = name
        self.dataset = dataset
        self.behaviors = behaviors
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.sequence_length = sequence_length
        self.training_method = training_method
        self.patience = patience
        self.num_runs = num_runs
        self.num_trials = num_trials
        self.optimization_target = optimization_target
        self.use_test = bool(use_test)
        self.test_split = float(test_split)
        self.custom_weights = custom_weights
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers

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

def plot_averaged_run_metrics(reports: list, behaviors: list, out_dir: str, plot_suffix: str = 'runs'):
    """
    Creates bar charts for precision, recall, and f1-score, showing the mean,
    standard deviation, and all individual data points across all training runs.
    """
    if not reports or not any(reports): return

    metrics = ['precision', 'recall', 'f1-score']
    for metric in metrics:
        plt.figure(figsize=(max(8, len(behaviors) * 0.6), 6))
        
        means = []
        stds = []
        all_run_values = []
        
        for b_name in behaviors:
            run_values = [r.get(b_name, {}).get(metric, 0) for r in reports]
            means.append(np.mean(run_values))
            stds.append(np.std(run_values))
            all_run_values.append(run_values)

        x_pos = np.arange(len(behaviors))
        
        plt.bar(x_pos, means, yerr=stds, align='center', alpha=0.7, ecolor='black', capsize=10, label=f'Mean (n={len(reports)})')
        
        for i, run_vals in enumerate(all_run_values):
            jitter = np.random.normal(0, 0.04, size=len(run_vals))
            plt.scatter([i + j for j in jitter], run_vals, color='black', alpha=0.6, zorder=3, label='Individual Run' if i == 0 else "")
        
        plt.ylabel(metric.replace('-', ' ').title())
        plt.xticks(x_pos, behaviors, rotation='vertical')
        plt.title(f"Average {metric.replace('-', ' ').title()} Across {len(reports)} {plot_suffix.capitalize()}")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        plt.tight_layout()
        
        plt.savefig(os.path.join(out_dir, f"{metric.replace(' ', '_')}_{plot_suffix}_plot.png"))
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
    eel.spawn(sync_labels_worker, source_dataset_name, target_dataset_name)


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
    
    # Start the ffmpeg self-healing monitor
    monitor = threading.Thread(target=_recording_monitor_worker, daemon=True)
    monitor.start()
    
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
    """(LAUNCHER) Spawns the label synchronization worker."""
    log_message(f"Spawning worker to sync labels from '{source_dataset_name}' to '{target_dataset_name}'", "INFO")
    # Use correct eel.spawn pattern and call the single, correct worker function
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