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


# =================================================================
# LOGGING HELPER
# =================================================================

def log_message(message: str, level: str = "INFO"):
    """
    Puts a formatted message into the global log queue for the UI and
    also prints it to the console for developer debugging.
    """
    timestamp = datetime.now().strftime('%H:%M:%S')
    # The format [LEVEL] is specifically used by the frontend for color-coding.
    formatted_message = f"[{timestamp}] [{level}] {message}"
    print(formatted_message) # Keep console logging for developers
    if gui_state.log_queue:
        gui_state.log_queue.put(formatted_message)


# =================================================================
# WORKER THREAD CLASSES
# =================================================================

class EncodeThread(threading.Thread):
    """A background thread that continuously processes video encoding tasks."""
    def __init__(self, device_str: str):
        super().__init__()
        self.device = torch.device(device_str)
        self.cuda_stream = torch.cuda.Stream(device=self.device) if self.device.type == 'cuda' else None
        self.encoder = cbas.DinoEncoder(device_str)

    def run(self):
        """The main loop for the thread. Pulls tasks from the global queue."""
        while True:
            file_to_encode = None
            with gui_state.encode_lock:
                if gui_state.encode_tasks:
                    file_to_encode = gui_state.encode_tasks.pop(0)

            if file_to_encode:
                log_message(f"Starting encoding for: {os.path.basename(file_to_encode)}", "INFO")
                if self.cuda_stream:
                    with torch.cuda.stream(self.cuda_stream):
                        out_file = cbas.encode_file(self.encoder, file_to_encode)
                else:
                    out_file = cbas.encode_file(self.encoder, file_to_encode)
                
                if out_file:
                    log_message(f"Finished encoding: {os.path.basename(file_to_encode)}", "INFO")
                    with gui_state.classify_lock:
                        gui_state.classify_tasks.append(out_file)
                    log_message(f"Added '{os.path.basename(out_file)}' to classification queue.", "INFO")
                else:
                    log_message(f"Failed to encode: {os.path.basename(file_to_encode)}", "WARN")
            else:
                time.sleep(1)  # Sleep if queue is empty to prevent busy-waiting

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
        self.running = False
        self.torch_model = None
        self.model_meta = None  # Stores the cbas.Model object

    def start_inferring(self, model_obj: cbas.Model, whitelist: list[str]):
        """
        Loads a model and prepares the thread to start processing classification tasks.
        
        Args:
            model_obj (cbas.Model): The model object containing config and weights path.
            whitelist (list[str]): A list of directories to restrict inference to (currently unused).
        """
        with gui_state.classify_lock:
            self.model_meta = model_obj
            self.whitelist = whitelist
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
                self.running = True
                log_message(f"Model '{model_obj.name}' loaded successfully.", "INFO")
            except Exception as e:
                log_message(f"Error loading model '{model_obj.name}': {e}", "ERROR")
                self.running = False

    def run(self):
        """The main loop for the thread. Pulls tasks from the global queue."""
        while True:
            file_to_classify = None
            can_run_now = False
            
            with gui_state.classify_lock:
                can_run_now = self.running
                if can_run_now and gui_state.classify_tasks:
                    file_to_classify = gui_state.classify_tasks.pop(0)

            if can_run_now and file_to_classify:
                log_message(f"Classifying: {os.path.basename(file_to_classify)} with model '{self.model_meta.name}'", "INFO")
                if self.torch_model and self.model_meta:
                    if self.cuda_stream:
                        with torch.cuda.stream(self.cuda_stream):
                            out_file = cbas.infer_file(file_to_classify, self.torch_model, self.model_meta.name, self.model_meta.config["behaviors"], self.model_meta.config["seq_len"], device=self.device)
                    else:
                        out_file = cbas.infer_file(file_to_classify, self.torch_model, self.model_meta.name, self.model_meta.config["behaviors"], self.model_meta.config["seq_len"], device=self.device)
                    
                    if out_file:
                        log_message(f"Finished classifying: {os.path.basename(file_to_classify)}", "INFO")
                    else:
                        log_message(f"Failed to classify: {os.path.basename(file_to_classify)}", "WARN")
                    
                    # Notify UI upon queue completion
                    with gui_state.classify_lock:
                        if not gui_state.classify_tasks:
                            log_message("Inference queue is empty. Classification complete.", "INFO")
                            eel.updateTrainingStatusOnUI(self.model_meta.name, "Inference complete.")()
                else:
                    log_message(f"Model not ready, re-queuing: {os.path.basename(file_to_classify)}", "WARN")
                    with gui_state.classify_lock:
                        gui_state.classify_tasks.insert(0, file_to_classify)
                    time.sleep(5)
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

    def queue_task(self, task: "TrainingTask"):
        """Adds a new training task to this thread's queue."""
        with self.training_queue_lock:
            self.training_queue.append(task)
        log_message(f"Queued training task for dataset: {task.name}", "INFO")

    def run(self):
        """The main loop for the thread. Pulls tasks from its internal queue."""
        while True:
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
            eel.updateTrainingStatusOnUI(task.name, "Loading dataset...")()
            log_message(f"Loading and processing dataset '{task.name}' for training...", "INFO")
            def update_progress(p): eel.updateDatasetLoadProgress(task.name, p)()
            
            train_insts, test_insts = None, None # Initialize
            if task.training_method == "weighted_loss":
                train_ds, test_ds, weights, train_insts, test_insts = gui_state.proj.load_dataset_for_weighted_loss(task.name, seq_len=task.sequence_length, progress_callback=update_progress)
            else:
                train_ds, test_ds, train_insts, test_insts = gui_state.proj.load_dataset(task.name, seq_len=task.sequence_length, progress_callback=update_progress)
                weights = None
            
            # Error check if loading failed
            if train_ds is None or test_ds is None or train_insts is None or test_insts is None:
                raise ValueError("Dataset loading returned None. Check for empty labels or other data issues.")
            
            eel.updateDatasetLoadProgress(task.name, 100)() # Signal completion
            log_message(f"Dataset '{task.name}' loaded successfully.", "INFO")
            if not train_ds or len(train_ds) == 0:
                raise ValueError("Dataset is empty or failed to load.")
        except Exception as e:
            log_message(f"Critical error loading dataset {task.name}: {e}", "ERROR")
            eel.updateTrainingStatusOnUI(task.name, f"Error loading dataset: {e}")()
            eel.updateDatasetLoadProgress(task.name, -1)(); return

        best_model, best_reports, best_epoch = None, None, -1
        best_f1 = -1.0
        NUM_TRIALS = 10

        for i in range(NUM_TRIALS):
            eel.updateTrainingStatusOnUI(task.name, f"Training Trial {i + 1}/{NUM_TRIALS}...")()
            log_message(f"Starting training trial {i + 1}/{NUM_TRIALS} for '{task.name}'.", "INFO")
            trial_model, trial_reports, trial_best_epoch = cbas.train_lstm_model(
                train_ds, test_ds, task.sequence_length, task.behaviors,
                lr=task.learning_rate, batch_size=task.batch_size,
                epochs=task.epochs, device=self.device, class_weights=weights,
                patience=task.patience
            )

            if trial_model and trial_reports and trial_best_epoch != -1:
                # We need to access the VALIDATION report (val_report) to get the F1 score.
                
                # Get the PerformanceReport object for the best epoch of this trial
                best_epoch_report = trial_reports[trial_best_epoch]
                
                # Access the f1-score from the validation report inside that object
                f1 = best_epoch_report.val_report.get("weighted avg", {}).get("f1-score", -1.0)
                                              
                eel.updateTrainingStatusOnUI(task.name, f"Trial {i + 1} F1: {f1:.4f}")()
                if f1 > best_f1:
                    log_message(f"New best model in Trial {i + 1} with F1: {f1:.4f}", "INFO")
                    best_f1, best_model, best_reports, best_epoch = f1, trial_model, trial_reports, trial_best_epoch

        if best_model:
            # Pass the raw instance lists down to the saving function
            self._save_training_results(task, best_model, best_reports, best_epoch, train_insts, test_insts)
        else:
            log_message(f"Training failed for '{task.name}' after {NUM_TRIALS} trials.", "ERROR")
            eel.updateTrainingStatusOnUI(task.name, f"Training failed after {NUM_TRIALS} trials.")

    def _save_training_results(self, task, model, reports, best_epoch_idx, train_insts, test_insts):
        """Saves the best model, performance reports, and plots."""
        # Get the specific report for the best epoch
        best_report_obj = reports[best_epoch_idx]
        
        # The report to save to disk and display should be from the VALIDATION set
        best_validation_report = best_report_obj.val_report
        best_validation_cm = best_report_obj.val_cm
        
        log_message(f"Saving results for best model (Validation F1: {best_validation_report['weighted avg']['f1-score']:.4f})...", "INFO")
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
        # The UI metrics should reflect the model's performance on unseen (validation) data
        self._update_metrics_in_ui(task.name, best_validation_report, task.behaviors, task.dataset, train_insts, test_insts)
        log_message(f"Training for '{task.name}' complete. Model and reports saved.", "INFO")
        eel.updateTrainingStatusOnUI(task.name, f"Training complete. Best Validation F1: {best_validation_report['weighted avg']['f1-score']:.4f}")

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
    """A watchdog handler that queues newly created video files for encoding."""
    def on_created(self, event):
        if event.is_directory or not event.src_path.endswith('.mp4'): return

        video_path = event.src_path
        log_message(f"Watcher detected new file: {os.path.basename(video_path)}", "INFO")

        # =========================================================================
        # NEW LOGIC TO PREVENT WATCHER FROM ACTING ON IMPORTS
        # =========================================================================
        # Check if the camera name from the file path is in the active recording list.
        # If it's not an active stream, it's a manual copy/import, so we do nothing.
        try:
            # e.g., "C:\...\Mouse 1_00000.mp4" -> "Mouse 1"
            camera_name = os.path.basename(video_path).rsplit('_', 1)[0]
            # gui_state.proj might not exist on initial startup, so check it.
            if not gui_state.proj or camera_name not in gui_state.proj.active_recordings:
                log_message(f"Ignoring file '{os.path.basename(video_path)}' as it's not from an active recording stream.", "INFO")
                return
        except (IndexError, AttributeError):
             # If parsing fails or proj isn't loaded, safely exit.
            return
        # =========================================================================
        
        # This logic now ONLY runs for active, live recordings.
        dirname, basename = os.path.split(video_path)
        try:
            name_part, num_str = os.path.splitext(basename)[0].rsplit('_', 1)
            current_seg_num = int(num_str)
        except (ValueError, IndexError) as e:
            log_message(f"Could not parse segment number from {basename}: {e}", "WARN"); return

        if current_seg_num > 0:
            prev_seg_num_str = str(current_seg_num - 1).zfill(len(num_str))
            prev_file = os.path.join(dirname, f"{name_part}_{prev_seg_num_str}.mp4")
            
            if os.path.exists(prev_file):
                with gui_state.encode_lock:
                    if prev_file not in gui_state.encode_tasks:
                        gui_state.encode_tasks.append(prev_file)
                        log_message(f"Watcher queued for encoding: '{os.path.basename(prev_file)}'", "INFO")


# =================================================================
# GLOBAL THREAD MANAGEMENT FUNCTIONS
# =================================================================

@eel.expose
def start_threads():
    """Initializes and starts all background worker threads."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing worker threads on device: {device}")

    gui_state.encode_lock = threading.Lock()
    gui_state.encode_thread = EncodeThread(device)
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


@eel.expose
def start_recording_watcher():
    """Initializes and starts the file system watcher."""
    if not gui_state.proj or not os.path.exists(gui_state.proj.recordings_dir): return
    if gui_state.recording_observer and gui_state.recording_observer.is_alive(): return

    event_handler = VideoFileWatcher()
    gui_state.recording_observer = Observer()
    gui_state.recording_observer.schedule(event_handler, gui_state.proj.recordings_dir, recursive=True)
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