import threading
import gui_state
import time
import ctypes
import eel
import os
import torch
import cbas
import yaml

import matplotlib
matplotlib.use("Agg")  # Set the backend to non-GUI mode

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

import classifier_head

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class EncodeThread(threading.Thread):
    def __init__(self, device):
        threading.Thread.__init__(self)
        self.device = device
        # This is now a fast operation thanks to our change in cbas.py
        self.encoder = cbas.DinoEncoder(device=self.device)

        if self.device.type == 'cuda':
            self.cuda_stream = torch.cuda.Stream(device=self.device)
        else:
            self.cuda_stream = None

    def run(self):
        while True:
            gui_state.encode_lock.acquire()
            num_encode_tasks = len(gui_state.encode_tasks)

            if num_encode_tasks > 0:
                file = gui_state.encode_tasks.pop(0)
                gui_state.encode_lock.release()

                # The slow model download will happen here, inside the background thread, on the first call
                if self.cuda_stream:
                    with torch.cuda.stream(self.cuda_stream):
                        out_file = cbas.encode_file(self.encoder, file)
                else:
                    out_file = cbas.encode_file(self.encoder, file)

                if out_file is None:
                    print(f"Failed to encode unfinished video file '{file}'")
                else:
                    print(f"Encoded '{file}' as '{out_file}'")
                    gui_state.classify_lock.acquire()
                    gui_state.classify_tasks.append(out_file)
                    gui_state.classify_lock.release()

            else:
                gui_state.encode_lock.release()

            time.sleep(1)

    def get_id(self):
        if hasattr(self, "_thread_id"):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

    def raise_exception(self):
        thread_id = self.get_id()
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            thread_id, ctypes.py_object(SystemExit)
        )
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            print("Exception raise failure")

class ClassificationThread(threading.Thread):
    def __init__(self, device):
        threading.Thread.__init__(self)
        self.device = device
        self.running = False

        if self.device.type == 'cuda':
            self.cuda_stream = torch.cuda.Stream(device=self.device)
        else:
            self.cuda_stream = None
    
    def start_inferring(self, model, whitelist):
        gui_state.classify_lock.acquire()
        self.model = model
        self.whitelist = whitelist

        weights = torch.load(model.weights_path, map_location=self.device) # Use map_location for robustness

        torch_model = classifier_head.classifier(
            in_features=768,
            out_features=len(model.config["behaviors"]),
            seq_len=model.config["seq_len"],
        )
        torch_model.load_state_dict(state_dict=weights)
        torch_model.eval()

        self.torch_model = torch_model.to(self.device)
        self.running = True
        gui_state.classify_lock.release()

        print(f"Loaded model '{model.name}' for classification on device '{self.device}'")

    def run(self):
        while True:
            if self.running:
                gui_state.classify_lock.acquire()
                num_classify_tasks = len(gui_state.classify_tasks)
                if num_classify_tasks > 0:
                    file = gui_state.classify_tasks.pop(0)
                    gui_state.classify_lock.release()

                    if self.cuda_stream:
                        with torch.cuda.stream(self.cuda_stream):
                            out_file = cbas.infer_file(file, self.torch_model, self.model.name, self.model.config["behaviors"], self.model.config["seq_len"], str(self.device))
                    else:
                        out_file = cbas.infer_file(file, self.torch_model, self.model.name, self.model.config["behaviors"], self.model.config["seq_len"], str(self.device))
                    
                    if out_file is None:
                        print(f"Failed to classify file '{file}")
                    else:
                        print(f"Classified file '{file}' to '{out_file}'")
                else:
                    gui_state.classify_lock.release()

            time.sleep(1)

    def get_id(self):
        if hasattr(self, "_thread_id"):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

    def raise_exception(self):
        thread_id = self.get_id()
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            thread_id, ctypes.py_object(SystemExit)
        )
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            print("Exception raise failure")

class TrainingTask():
    def __init__(self, name, dataset, behaviors, batch_size, learning_rate, epochs, sequence_length):
        self.name = name
        self.dataset = dataset
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.sequence_length = sequence_length
        self.behaviors = behaviors
    

def save_confusion_matrix_plot(confusion_matrix, path):
    disp = ConfusionMatrixDisplay(confusion_matrix)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)

    plt.savefig(path)
    plt.close(fig)

def plot_report_list_metric(report_list, metric, behaviors, path):
    n_epochs = len(report_list)
    epochs = list(range(1, n_epochs + 1))

    plt.figure(figsize=(8, 6))

    for behavior in behaviors:
        data = [report.sklearn_report[behavior][metric] for report in report_list]
        plt.plot(epochs, data, marker='o', linestyle='-', label=behavior)

    plt.xlabel("Epochs")
    plt.ylabel(metric.capitalize())
    plt.title(f"{metric.capitalize()} Over Epochs")
    plt.legend(title="Behaviors")
    plt.grid(True)

    os.makedirs(path, exist_ok=-1)

    filename = os.path.join(path, f"{metric}-report.png")
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()

class TrainingThread(threading.Thread):
    def __init__(self, device):
        threading.Thread.__init__(self)
        self.device = device

        if self.device.type == 'cuda':
            self.cuda_stream = torch.cuda.Stream(device=device)
        else:
            self.cuda_stream = None

        self.training_queue_lock = threading.Lock()
        self.training_queue: list[TrainingTask] = []

    def run(self):
        while True:
            self.training_queue_lock.acquire()
            num_training_tasks = len(self.training_queue)
            if num_training_tasks > 0:
                task = self.training_queue.pop(0)
                self.training_queue_lock.release()

                if self.cuda_stream:
                    with torch.cuda.stream(self.cuda_stream):
                        self._run_task(task)
                else:
                    self._run_task(task)
            else:
                self.training_queue_lock.release()

            time.sleep(1)
    
    def _run_task(self, task):
        """Helper method to contain the training logic."""
        train_ds, test_ds = gui_state.proj.load_dataset(task.name)

        model_dir = os.path.join(gui_state.proj.models_dir, task.name)
        model_path = os.path.join(model_dir, "model.pth")
        model_config_path = os.path.join(model_dir, "config.yaml")

        model, report_list, best_epoch = cbas.train_lstm_model(train_ds, test_ds, task.sequence_length, task.behaviors, lr=task.learning_rate, batch_size=task.batch_size, epochs=task.epochs, device=self.device)

        best_report = report_list[best_epoch]
        report_dict = best_report.sklearn_report

        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        torch.save(model.state_dict(), model_path)
        task.dataset.config["model"] = task.name

        with open(task.dataset.config_path, "w+") as file:
            yaml.dump(task.dataset.config, file, allow_unicode=True)

        behaviors = task.dataset.config["behaviors"]

        for b in behaviors:
            task.dataset.update_metric(b, "F1 Score", round(report_dict[b]["f1-score"], 2))
            task.dataset.update_metric(b, "Recall", round(report_dict[b]["recall"], 2))
            task.dataset.update_metric(b, "Precision", round(report_dict[b]["precision"], 2))

        model_config = {"seq_len": task.sequence_length, "behaviors": task.behaviors}

        performance_path = os.path.join(task.dataset.path, "performance.yaml")
        with open(performance_path, 'w+') as file:
            yaml.dump(best_report.sklearn_report, file, allow_unicode=True)

        save_confusion_matrix_plot(best_report.confusion_matrix, os.path.join(task.dataset.path, "confusion_matrix.png"))

        if not os.path.exists(os.path.join(task.dataset.path, "confusion_matrices")):
            os.mkdir(os.path.join(task.dataset.path, "confusion_matrices"))

        for i, report in enumerate(report_list):
            save_confusion_matrix_plot(report.confusion_matrix, os.path.join(task.dataset.path, "confusion_matrices", f"epoch_{i}.png"))

        plot_report_list_metric(report_list, "f1-score", behaviors, task.dataset.path)
        plot_report_list_metric(report_list, "recall", behaviors, task.dataset.path)
        plot_report_list_metric(report_list, "precision", behaviors, task.dataset.path)

        with open(model_config_path, "w+") as file:
            yaml.dump(model_config, file, allow_unicode=True)
        
        gui_state.proj.models[task.name] = cbas.Model(model_dir)

    def queue_task(self, task):
        self.training_queue_lock.acquire()
        self.training_queue.append(task)
        self.training_queue_lock.release()

    def get_id(self):
        if hasattr(self, "_thread_id"):
            return self._thread_id
        for id, thread in threading._active.items():
            if thread is self:
                return id

    def raise_exception(self):
        thread_id = self.get_id()
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            thread_id, ctypes.py_object(SystemExit)
        )
        if res > 1:
            ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
            print("Exception raise failure")

class VideoFileWatcher(FileSystemEventHandler):
    def __init__(self):
        super().__init__()

    def on_any_event(self, event):
        if event.event_type == 'created' and event.is_directory == False:
            video_path = event.src_path
            dirname, basename = os.path.split(video_path)
            if '.mp4' not in basename:
                return

            number = int(basename.split('.')[0].split('_')[-1])

            if number != 0:
                prev_file_path = os.path.join(dirname, basename.split('_')[0] + '_' + str(number - 1).zfill(5) + '.mp4')

                gui_state.encode_lock.acquire()
                gui_state.encode_tasks.append(prev_file_path)
                gui_state.encode_lock.release()
                print(f"Sent encode thread '{prev_file_path}'")


@eel.expose
def start_threads():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"--- CBAS is running on device: {device} ---")

    gui_state.encode_lock = threading.Lock()
    gui_state.encode_thread = EncodeThread(device)
    gui_state.encode_thread.start()

    gui_state.training_thread = TrainingThread(device)
    gui_state.training_thread.start()

    gui_state.classify_lock = threading.Lock()
    gui_state.classify_thread = ClassificationThread(device)
    gui_state.classify_thread.start()

@eel.expose
def start_recording_watcher():
    event_handler = VideoFileWatcher()
    gui_state.recording_observer = Observer()
    gui_state.recording_observer.schedule(event_handler, gui_state.proj.recordings_dir, recursive=True)
    gui_state.recording_observer.start()

def stop_threads():
    if gui_state.encode_thread:
        gui_state.encode_thread.raise_exception()
        gui_state.encode_thread.join()

    if gui_state.classify_thread:
        gui_state.classify_thread.raise_exception()
        gui_state.classify_thread.join()

    if gui_state.training_thread:
        gui_state.training_thread.raise_exception()
        gui_state.training_thread.join()
    
    if gui_state.recording_observer:
        gui_state.recording_observer.stop()
        gui_state.recording_observer.join()