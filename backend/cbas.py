"""
Core backend logic and data models for the CBAS application.

This module defines the primary classes that represent the project structure:
- Project: The top-level container for all data and configurations.
- Camera: Manages settings and recording for a single video source.
- Recording: Represents a single recording session with its associated video and data files.
- Model: A wrapper for a trained machine learning model.
- Dataset: Manages labeled data, training/testing splits, and dataset configurations.
- Actogram: Handles the generation and plotting of actogram visualizations.

It also includes standalone utility functions for video processing (encoding),
model inference, and data handling (PyTorch Datasets, collation).
"""

# Standard library imports
import os
import io
import time
import base64
import math
import shutil
import subprocess
from datetime import datetime
import random
import yaml

# Third-party imports
import cv2
import decord
import h5py
import numpy as np
import pandas as pd
import torch
import transformers
from torch import nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg') # Essential for non-GUI thread plotting
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Local application imports
import classifier_head


# =================================================================
# EXCEPTIONS
# =================================================================

class InvalidProject(Exception):
    """Custom exception raised when a directory is not a valid CBAS project."""
    def __init__(self, path):
        super().__init__(f"Path '{path}' is not a valid CBAS project directory.")


# =================================================================
# CORE PROCESSING FUNCTIONS
# =================================================================

def encode_file(encoder: nn.Module, path: str) -> str | None:
    """
    Extracts DINOv2 embeddings from a video file and saves them to an HDF5 file.

    Args:
        encoder (nn.Module): The pre-loaded DinoEncoder model.
        path (str): The absolute path to the input video (.mp4) file.

    Returns:
        str | None: The path to the created HDF5 file, or None if encoding failed.
    """
    try:
        reader = decord.VideoReader(path, ctx=decord.cpu(0))
    except Exception as e:
        print(f"Error reading video {path} with decord: {e}")
        return None

    frames_np = reader.get_batch(range(len(reader))).asnumpy()
    if frames_np.size == 0:
        print(f"Warning: Video {path} contains no frames.")
        return None
    
    # Use green channel for B/W video, normalize, and set precision
    frames = torch.from_numpy(frames_np[:, :, :, 1] / 255).half()

    batch_size = 256
    embeddings = []
    for i in range(0, len(frames), batch_size):
        batch = frames[i: i + batch_size]
        with torch.no_grad(), torch.amp.autocast(device_type=encoder.device.type if encoder.device.type != 'mps' else 'cpu'):
            out = encoder(batch.unsqueeze(1).to(encoder.device))
        embeddings.extend(out.squeeze(1).cpu())

    out_file_path = os.path.splitext(path)[0] + "_cls.h5"
    with h5py.File(out_file_path, "w") as file:
        file.create_dataset("cls", data=torch.stack(embeddings).numpy())
    return out_file_path


def infer_file(file_path: str, model: nn.Module, dataset_name: str, behaviors: list[str], seq_len: int, device=None) -> str | None:
    """
    Runs a trained classifier on an HDF5 embedding file to produce behavior probabilities.

    Args:
        file_path (str): Path to the _cls.h5 file.
        model (nn.Module): The trained classifier head model.
        dataset_name (str): The name of the model, used for the output filename.
        behaviors (list[str]): List of behavior names for the output columns.
        seq_len (int): The sequence length the model was trained on.
        device: The torch.device to run inference on.

    Returns:
        str | None: The path to the output CSV file, or None on failure.
    """
    output_file = file_path.replace("_cls.h5", f"_{dataset_name}_outputs.csv")
    try:
        with h5py.File(file_path, "r") as f:
            cls_np = np.array(f["cls"][:])
    except Exception as e:
        print(f"Error reading HDF5 file {file_path}: {e}")
        return None
    
    if cls_np.ndim < 2 or cls_np.shape[0] < seq_len:
        print(f"Warning: HDF5 file {file_path} is too short for inference. Skipping.")
        return None

    # This handles the data loading robustly.
    cls_np_f32 = cls_np.astype(np.float32)
    cls = torch.from_numpy(cls_np_f32 - np.mean(cls_np_f32, axis=0))
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    predictions = []
    batch_windows = []
    half_seqlen = seq_len // 2

    # Create sliding windows and process in batches
    for i in range(half_seqlen, len(cls) - half_seqlen):
        window = cls[i - half_seqlen: i + half_seqlen + 1]
        batch_windows.append(window)

        if len(batch_windows) >= 4096 or i == len(cls) - half_seqlen - 1:
            if not batch_windows: continue
            
            batch_tensor = torch.stack(batch_windows)
            
            # Remove the `torch.amp.autocast` context manager.
            # We only need `torch.no_grad()` for inference. This forces all
            # operations to use the model's native float32 precision,
            # preventing the BFloat16 type error.
            with torch.no_grad():
                logits = model.forward_nodrop(batch_tensor.to(device))
                
            predictions.extend(torch.softmax(logits, dim=1).cpu().numpy())
            batch_windows = []

    if not predictions: return None

    # Pad predictions for frames at the beginning and end that don't have a full window
    padded_predictions = []
    for i in range(len(cls)):
        if i < half_seqlen:
            padded_predictions.append(predictions[0])
        elif i >= len(cls) - half_seqlen:
            padded_predictions.append(predictions[-1])
        else:
            padded_predictions.append(predictions[i - half_seqlen])

    pd.DataFrame(np.array(padded_predictions), columns=behaviors).to_csv(output_file, index=False)
    return output_file

def _create_matplotlib_actogram(binned_activity, light_cycle_booleans, tau, bin_size_minutes, plot_title, start_hour_offset, plot_acrophase=False):
    """
    Generates a double-plotted actogram using Matplotlib.
    This function contains all data preparation and plotting logic.
    """
    bins_per_period = int((tau * 60) / bin_size_minutes)
    if bins_per_period == 0: return None

    padding_bins = int(start_hour_offset * 60 / bin_size_minutes)
    padded_activity = np.pad(binned_activity, (padding_bins, 0), 'constant')

    num_days = math.ceil(len(padded_activity) / bins_per_period)
    if num_days < 1: return None
    
    required_len = num_days * bins_per_period
    padded_for_reshape = np.pad(padded_activity, (0, required_len - len(padded_activity)), 'constant', constant_values=np.nan)
    daily_data = padded_for_reshape.reshape(num_days, bins_per_period)

    acrophase_points = []
    if plot_acrophase:
        t = np.linspace(0, 2 * np.pi, bins_per_period, endpoint=False)
        for day_idx, day_activity in enumerate(daily_data):
            if np.isnan(day_activity).any() or np.sum(np.nan_to_num(day_activity)) == 0: continue
            day_activity = np.nan_to_num(day_activity)
            phase_rad = math.atan2(np.sum(day_activity * np.sin(t)), np.sum(day_activity * np.cos(t)))
            acrophase_hour_rel = (phase_rad / (2 * np.pi)) * 24
            acrophase_hour_abs = (acrophase_hour_rel + 24 + start_hour_offset) % 24
            acrophase_points.append((day_idx, acrophase_hour_abs))
    
    right_half = np.full_like(daily_data, np.nan)
    if num_days > 1: right_half[:-1, :] = daily_data[1:, :]
    double_plotted_events = np.concatenate([daily_data, right_half], axis=1)
  
    # Setup background colormap based on light cycle
    light_yellow, dark_yellow, light_grey, dark_grey = '#FEFDE3', '#E8D570', '#D3D3D3', '#A9A9A9'
    if all(light_cycle_booleans): # LL
        pattern = [1]*int(12*60/bin_size_minutes) + [0]*int(12*60/bin_size_minutes)
        cmap = LinearSegmentedColormap.from_list("light_cmap", [dark_yellow, light_yellow])
    elif not any(light_cycle_booleans): # DD
        pattern = [1]*int(12*60/bin_size_minutes) + [0]*int(12*60/bin_size_minutes)
        cmap = LinearSegmentedColormap.from_list("light_cmap", [dark_grey, light_grey])
    else: # LD
        pattern = np.repeat([b for b in light_cycle_booleans], (60 / bin_size_minutes))
        cmap = LinearSegmentedColormap.from_list("light_cmap", [dark_grey, light_yellow])

    double_plotted_light = np.array([np.concatenate([pattern, pattern]) for _ in range(num_days)])

    # Setup activity colormap (viridis with transparency)
    cmap_viridis = plt.get_cmap('viridis')
    activity_colors = cmap_viridis(np.arange(cmap_viridis.N))
    activity_colors[0, 3] = 0
    transparent_viridis = LinearSegmentedColormap.from_list('transparent_viridis', activity_colors)
    transparent_viridis.set_bad(color=(0,0,0,0))

    # Setup Figure
    fig, ax = plt.subplots(figsize=(10, max(4, num_days * 0.4)), dpi=120)
    fig.patch.set_facecolor('#343a40')
    ax.set_facecolor('#343a40')
    
    plot_extent = [0, 2 * tau, num_days, 0]
    
    # Plotting
    ax.imshow(double_plotted_light, aspect='auto', cmap=cmap, interpolation='none', extent=plot_extent, vmin=0, vmax=1)
    non_zero_activity = [v for v in binned_activity if v > 0]
    vmax = np.percentile(non_zero_activity, 90) + 1e-6 if non_zero_activity else 1
    cax = ax.imshow(double_plotted_events, aspect='auto', cmap=transparent_viridis, interpolation='none', extent=plot_extent, vmin=0, vmax=vmax)
    
    if acrophase_points:
        for day_idx, hour in acrophase_points:
            ax.plot(hour, day_idx + 0.5, 'o', color='red', markersize=8, markeredgecolor='black')
            ax.plot(hour + tau, day_idx + 0.5, 'o', color='red', markersize=8, markeredgecolor='black')

    # Styling and Colorbar
    cbar = fig.colorbar(cax, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Event Count', color='white')
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.outline.set_edgecolor('white')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    
    ax.set_title(plot_title, color='white', pad=20)
    ax.set_xlabel('Time of Day (Double Plotted)', color='white')
    ax.set_ylabel('Day', color='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values(): spine.set_edgecolor('white')

    ax.set_xlim(0, 2*tau); ax.set_ylim(num_days, 0)
    ax.set_xticks(np.arange(0, 2 * tau + 1, 4))
    ax.set_xticklabels([f"{int(tick % 24):02d}" for tick in np.arange(0, 2 * tau + 1, 4)])
    ax.set_yticks(np.arange(0.5, num_days, 1))
    ax.set_yticklabels([f"{i+1}" for i in range(num_days)])
    
    fig.tight_layout()
    return fig


# =================================================================
# DATA MODEL CLASSES
# =================================================================

class DinoEncoder(nn.Module):
    """Wraps a frozen DINOv2 model for feature extraction."""
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = torch.device(device)
        self.model = transformers.AutoModel.from_pretrained("facebook/dinov2-base").to(self.device)
        self.model.eval()
        for param in self.model.parameters(): param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Processes a batch of grayscale frames into embeddings."""
        B, S, H, W = x.shape
        x = x.to(self.device).unsqueeze(2).repeat(1, 1, 3, 1, 1).reshape(B * S, 3, H, W)
        with torch.no_grad():
            out = self.model(x)
        return out.last_hidden_state[:, 0, :].reshape(B, S, 768)


class Recording:
    """Represents a single recording session folder containing video segments and their data."""
    def __init__(self, path: str):
        if not os.path.isdir(path): raise FileNotFoundError(path)
        self.path = path
        self.name = os.path.basename(path)
        
        all_files = [f.path for f in os.scandir(self.path) if f.is_file()]
        self.video_files = sorted([f for f in all_files if f.endswith(".mp4")], key=lambda x: int(x.split("_")[-1].split(".")[0]))
        self.encoding_files = [f for f in all_files if f.endswith("_cls.h5")]
        self.unencoded_files = [vf for vf in self.video_files if vf.replace(".mp4", "_cls.h5") not in self.encoding_files]

        self.classifications = {}
        for csv_file in [f for f in all_files if f.endswith(".csv")]:
            try:
                model_name = os.path.basename(csv_file).split("_")[-2]
                self.classifications.setdefault(model_name, []).append(csv_file)
            except IndexError: continue


class Camera:
    """Manages configuration and FFMPEG process for a single camera."""
    def __init__(self, config: dict, project: "Project"):
        self.config = config
        self.project = project
        self.name = config.get("name", "Unnamed")
        self.path = os.path.join(self.project.cameras_dir, self.name)
        self.update_settings(config, write_to_disk=False)

    def settings_to_dict(self) -> dict:
        return {
            "name": self.name, "rtsp_url": self.rtsp_url, "framerate": self.framerate,
            "resolution": self.resolution, "crop_left_x": self.crop_left_x,
            "crop_top_y": self.crop_top_y, "crop_width": self.crop_width, "crop_height": self.crop_height,
        }

    def update_settings(self, settings: dict, write_to_disk: bool = True):
        self.rtsp_url = str(settings.get("rtsp_url", ""))
        self.framerate = int(settings.get("framerate", 10))
        self.resolution = int(settings.get("resolution", 256))
        self.crop_left_x = float(settings.get("crop_left_x", 0.0))
        self.crop_top_y = float(settings.get("crop_top_y", 0.0))
        self.crop_width = float(settings.get("crop_width", 1.0))
        self.crop_height = float(settings.get("crop_height", 1.0))
        if write_to_disk: self.write_settings_to_config()

    def write_settings_to_config(self):
        with open(os.path.join(self.path, "config.yaml"), "w") as file:
            yaml.dump(self.settings_to_dict(), file, allow_unicode=True)

    def create_recording_dir(self) -> str | bool:
        date_path = os.path.join(self.project.recordings_dir, datetime.now().strftime("%Y%m%d"))
        os.makedirs(date_path, exist_ok=True)
        cam_path = os.path.join(date_path, f"{self.name}-{datetime.now().strftime('%I%M%S-%p')}")
        if not os.path.exists(cam_path):
            os.makedirs(cam_path); return cam_path
        return False

    def start_recording(self, destination: str, segment_time: int) -> bool:
        if self.name in self.project.active_recordings: return False
        os.makedirs(destination, exist_ok=True)
        
        dest_pattern = os.path.join(destination, f"{self.name}_%05d.mp4")
        command = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-rtsp_transport", "tcp",
            "-i", str(self.rtsp_url), "-r", str(self.framerate),
            "-filter_complex", f"[0:v]crop=iw*{self.crop_width}:ih*{self.crop_height}:iw*{self.crop_left_x}:ih*{self.crop_top_y},scale={self.resolution}:{self.resolution}[cropped]",
            "-map", "[cropped]", "-c:v", "libx264", "-preset", "fast", "-f", "segment",
            "-segment_time", str(segment_time), "-reset_timestamps", "1",
            "-force_key_frames", f"expr:gte(t,n_forced*{segment_time})", "-y", dest_pattern,
        ]
        
        try:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.project.active_recordings[self.name] = process
            return True
        except Exception as e:
            print(f"Failed to start ffmpeg for {self.name}: {e}"); return False

    def stop_recording(self) -> bool:
        if self.name in self.project.active_recordings:
            process = self.project.active_recordings.pop(self.name)
            try:
                process.stdin.write(b'q\n'); process.stdin.flush()
                process.communicate(timeout=10)
            except (subprocess.TimeoutExpired, Exception):
                process.terminate(); process.wait(timeout=5)
            return True
        return False


class Model:
    """Wraps a trained classifier model, holding its configuration and weights path."""
    def __init__(self, path: str):
        self.path = path
        self.name = os.path.basename(path)
        self.config_path = os.path.join(path, "config.yaml")
        self.weights_path = os.path.join(path, "model.pth")
        
        if not os.path.exists(self.config_path): raise FileNotFoundError(f"Model config not found: {self.config_path}")
        with open(self.config_path) as f: self.config = yaml.safe_load(f)
        if not os.path.exists(self.weights_path): raise FileNotFoundError(f"Model weights not found: {self.weights_path}")


class Dataset:
    """Manages a dataset's configuration, labeled instances, and data loading."""
    def __init__(self, path: str):
        self.path = path
        self.name = os.path.basename(path)
        self.config_path = os.path.join(path, "config.yaml")
        self.labels_path = os.path.join(path, "labels.yaml")

        if not os.path.exists(self.config_path): raise FileNotFoundError(f"Dataset config not found: {self.config_path}")
        with open(self.config_path) as f: self.config = yaml.safe_load(f)

        if not os.path.exists(self.labels_path):
            behaviors = self.config.get("behaviors", [])
            default_labels = {"behaviors": behaviors, "labels": {b: [] for b in behaviors}}
            with open(self.labels_path, "w") as f: yaml.dump(default_labels, f, allow_unicode=True)
            self.labels = default_labels
        else:
            with open(self.labels_path) as f: self.labels = yaml.safe_load(f)

    def update_metric(self, behavior: str, group: str, value):
        self.config.setdefault("metrics", {}).setdefault(behavior, {})[group] = value
        with open(self.config_path, "w") as file:
            yaml.dump(self.config, file, allow_unicode=True)
            
    def predictions_to_instances(self, csv_path: str, model_name: str, threshold: float = 0.7) -> list:
        try: df = pd.read_csv(csv_path)
        except FileNotFoundError: return []

        instances, behaviors = [], self.config.get("behaviors", [])
        if not behaviors or any(b not in df.columns for b in behaviors): return []

        df['predicted_label'] = df[behaviors].idxmax(axis=1)
        df['max_prob'] = df[behaviors].max(axis=1)
        
        in_event, current_event = False, {}
        for i, row in df.iterrows():
            is_above_thresh = row['max_prob'] >= threshold
            if not in_event and is_above_thresh:
                in_event = True
                current_event = {"video": csv_path.replace(f"_{model_name}_outputs.csv", ".mp4"), "start": i, "label": row['predicted_label']}
            elif in_event and (not is_above_thresh or row['predicted_label'] != current_event['label']):
                in_event = False
                current_event['end'] = i - 1
                if current_event['end'] >= current_event['start']: instances.append(current_event)
                if is_above_thresh: # Start a new event immediately
                    in_event = True
                    current_event = {"video": csv_path.replace(f"_{model_name}_outputs.csv", ".mp4"), "start": i, "label": row['predicted_label']}
        if in_event:
            current_event['end'] = len(df) - 1
            if current_event['end'] >= current_event['start']: instances.append(current_event)
        return instances


class Actogram:
    """Generates and holds data for an actogram visualization."""
    def __init__(self, directory: str, model: str, behavior: str, framerate: float, start: float, binsize_minutes: int, threshold: float, lightcycle: str, plot_acrophase: bool = False):
        self.directory, self.model, self.behavior = directory, model, behavior
        self.framerate, self.start_hour_on_plot = float(framerate), float(start)
        self.threshold, self.bin_size_minutes = float(threshold), int(binsize_minutes)
        self.plot_acrophase = plot_acrophase
        self.lightcycle_str = {"LL": "1"*24, "DD": "0"*24}.get(lightcycle, "1"*12 + "0"*12)

        if self.framerate <= 0 or self.bin_size_minutes <= 0: return
        self.binsize_frames = int(self.bin_size_minutes * self.framerate * 60)
        if self.binsize_frames <= 0: return

        # Calculate raw activity
        activity_per_frame = []
        valid_csvs = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(f"_{model}_outputs.csv")]
        if not valid_csvs: return
        
        valid_csvs.sort(key=lambda p: int(os.path.basename(p).split('_')[-3]) if os.path.basename(p).split('_')[-3].isdigit() else -1)
        
        for file_path in valid_csvs:
            df = pd.read_csv(file_path)
            if df.empty or self.behavior not in df.columns: continue
            probs = df[self.behavior].to_numpy()
            is_max = (df[df.columns.drop(self.behavior)].max(axis=1) < probs).to_numpy() # More robust is_max
            activity_per_frame.extend((probs * is_max >= self.threshold).astype(float).tolist())
        
        if not activity_per_frame: return

        # Bin activity data
        binned_activity = [np.sum(activity_per_frame[i:i + self.binsize_frames]) for i in range(0, len(activity_per_frame), self.binsize_frames)]
        if not binned_activity: return

        # Plot and encode
        fig = _create_matplotlib_actogram(binned_activity, [c=="1" for c in self.lightcycle_str], 24.0, self.bin_size_minutes, f"{model} - {behavior}", self.start_hour_on_plot, self.plot_acrophase)
        if fig:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#343a40')
            buf.seek(0)
            self.blob = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)


# =================================================================
# PYTORCH DATASET & TRAINING CLASSES
# =================================================================

class StandardDataset(torch.utils.data.Dataset):
    """A standard PyTorch dataset without any balancing."""
    def __init__(self, sequences, labels): self.sequences, self.labels = sequences, labels
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx], self.labels[idx]


class BalancedDataset(torch.utils.data.Dataset):
    """A PyTorch dataset that balances classes via oversampling during batch creation."""
    def __init__(self, sequences: list, labels: list, behaviors: list):
        self.behaviors, self.num_behaviors = behaviors, len(behaviors)
        self.buckets = {b: [] for b in self.behaviors}
        for seq, label in zip(sequences, labels):
            if 0 <= label.item() < self.num_behaviors:
                self.buckets[self.behaviors[label.item()]].append(seq)
        self.total_sequences = sum(len(b) for b in self.buckets.values())
        self.counter = 0

    def __len__(self):
        if self.num_behaviors == 0: return 0
        return self.total_sequences + (self.num_behaviors - self.total_sequences % self.num_behaviors) % self.num_behaviors

    def __getitem__(self, idx: int):
        if self.num_behaviors == 0: raise IndexError("No behaviors defined.")
        b_idx = self.counter % self.num_behaviors
        self.counter += 1
        b_name = self.behaviors[b_idx]
        if not self.buckets[b_name]: raise IndexError(f"Behavior '{b_name}' has no samples.")
        sample_idx = idx % len(self.buckets[b_name])
        return self.buckets[b_name][sample_idx], torch.tensor(b_idx).long()


class Project:
    """
    Top-level class representing the entire CBAS project structure on disk.
    This class loads and manages all cameras, recordings, models, and datasets.
    """
    def __init__(self, path: str):
        if not os.path.isdir(path):
            raise InvalidProject(path)
        self.path = path
        
        # Define and ensure existence of project subdirectories
        self.cameras_dir = os.path.join(path, "cameras")
        self.recordings_dir = os.path.join(path, "recordings")
        self.models_dir = os.path.join(path, "models")
        self.datasets_dir = os.path.join(path, "data_sets")

        for subdir in [self.cameras_dir, self.recordings_dir, self.models_dir, self.datasets_dir]:
            os.makedirs(subdir, exist_ok=True)
            
        self.active_recordings: dict[str, subprocess.Popen] = {}
        
        # Load all project components from disk
        self._load_cameras()
        self._load_recordings()
        self._load_models()
        self._load_datasets()

    def _load_cameras(self):
        """Loads all camera configurations from the 'cameras' directory."""
        self.cameras = {}
        for cam_dir in [d for d in os.scandir(self.cameras_dir) if d.is_dir()]:
            config_path = os.path.join(cam_dir.path, "config.yaml")
            if os.path.exists(config_path):
                try:
                    with open(config_path) as f:
                        config = yaml.safe_load(f)
                    if "name" in config:
                        self.cameras[config["name"]] = Camera(config, self)
                except Exception as e:
                    print(f"Error loading camera config {config_path}: {e}")

    def _load_recordings(self):
        """Loads all recording sessions from the 'recordings' directory."""
        self.recordings = {}
        for day_dir in [d for d in os.scandir(self.recordings_dir) if d.is_dir()]:
            self.recordings[day_dir.name] = {}
            for session_dir in [d for d in os.scandir(day_dir.path) if d.is_dir()]:
                try:
                    rec = Recording(session_dir.path)
                    self.recordings[day_dir.name][rec.name] = rec
                except Exception as e:
                    print(f"Error loading recording {session_dir.path}: {e}")

    def _load_models(self):
        """Loads all trained models from the 'models' directory."""
        self.models = {}
        for model_dir in [d for d in os.scandir(self.models_dir) if d.is_dir()]:
            try:
                self.models[model_dir.name] = Model(model_dir.path)
            except Exception as e:
                print(f"Error loading model {model_dir.path}: {e}")
        # Also load the built-in JonesLabModel if it exists
        jl_model_path = os.path.join(os.path.dirname(__file__), "models", "JonesLabModel")
        if os.path.isdir(jl_model_path) and "JonesLabModel" not in self.models:
            try:
                self.models["JonesLabModel"] = Model(jl_model_path)
            except Exception as e:
                print(f"Warning: Could not load built-in JonesLabModel: {e}")

    def _load_datasets(self):
        """Loads all datasets from the 'data_sets' directory."""
        self.datasets = {}
        for ds_dir in [d for d in os.scandir(self.datasets_dir) if d.is_dir()]:
            try:
                self.datasets[ds_dir.name] = Dataset(ds_dir.path)
            except Exception as e:
                print(f"Error loading dataset {ds_dir.path}: {e}")

    @staticmethod
    def create_project(parent_directory: str, project_name: str) -> "Project | None":
        """Creates a new project directory structure and returns a Project instance."""
        project_path = os.path.join(parent_directory, project_name)
        if os.path.exists(project_path):
            print(f"Project '{project_name}' already exists. Cannot create.")
            return None
        try:
            for sub in ["cameras", "recordings", "models", "data_sets"]:
                os.makedirs(os.path.join(project_path, sub))
            return Project(project_path)
        except OSError as e:
            print(f"Error creating project directories: {e}")
            return None
            
    def create_camera(self, name: str, settings: dict) -> Camera | None:
        """Creates a new camera configuration and folder within the project."""
        camera_path = os.path.join(self.cameras_dir, name)
        if os.path.exists(camera_path):
            print(f"Camera '{name}' already exists.")
            return None
        os.makedirs(camera_path, exist_ok=True)
        
        settings_with_name = settings.copy()
        settings_with_name["name"] = name 

        with open(os.path.join(camera_path, "config.yaml"), "w") as file:
            yaml.dump(settings_with_name, file, allow_unicode=True)

        cam = Camera(settings_with_name, self)
        self.cameras[name] = cam
        return cam
        
    def create_dataset(self, name: str, behaviors: list[str], recordings_whitelist: list[str]) -> Dataset | None:
        """Creates a new dataset configuration and folder within the project."""
        directory = os.path.join(self.datasets_dir, name)
        if os.path.exists(directory):
            print(f"Dataset '{name}' already exists.")
            return None
        os.makedirs(directory, exist_ok=True)

        config_path = os.path.join(directory, "config.yaml")
        labels_path = os.path.join(directory, "labels.yaml")

        metrics = {b: {"Train #": 0, "Test #": 0, "Precision": "N/A", "Recall": "N/A", "F1 Score": "N/A"} for b in behaviors}
        dconfig = {"name": name, "behaviors": behaviors, "whitelist": recordings_whitelist, "model": None, "metrics": metrics}
        lconfig = {"behaviors": behaviors, "labels": {b: [] for b in behaviors}}

        with open(config_path, "w") as file: yaml.dump(dconfig, file, allow_unicode=True)
        with open(labels_path, "w") as file: yaml.dump(lconfig, file, allow_unicode=True)
        
        ds = Dataset(directory)
        self.datasets[name] = ds
        return ds

    def convert_instances(self, insts: list, seq_len: int, behaviors: list, progress_callback=None) -> tuple:
        """Converts labeled instances from dicts into training-ready tensors."""
        seqs, labels = [], []
        half_seqlen = seq_len // 2
        total_insts = len(insts)

        for i, inst in enumerate(insts):
            if progress_callback and i % 5 == 0:
                progress_callback((i + 1) / total_insts * 100)
            
            video_path = inst.get("video")
            if not video_path: continue
            cls_path = video_path.replace(".mp4", "_cls.h5")
            if not os.path.exists(cls_path): continue

            try:
                with h5py.File(cls_path, "r") as f: cls_arr = f["cls"][:]
            except Exception:
                continue
            
            if cls_arr.ndim < 2 or cls_arr.shape[0] < seq_len: continue
            
            cls_centered = cls_arr - np.mean(cls_arr, axis=0)
            
            # Ensure that start and end frames are always treated as integers.
            start = int(inst.get("start", -1))
            end = int(inst.get("end", -1))
            
            if start == -1 or end == -1: continue

            valid_start = max(half_seqlen, start)
            # Ensure valid_end is also an integer before use
            valid_end = int(min(cls_centered.shape[0] - half_seqlen, end))

            # The range function requires integer arguments.
            for frame_idx in range(valid_start, valid_end):
                window_start = frame_idx - half_seqlen
                window_end = frame_idx + half_seqlen + 1
                
                # This check is redundant if the loop range is correct, but safe to keep.
                if window_end > cls_centered.shape[0]: continue

                window = cls_centered[window_start:window_end]
                if window.shape[0] != seq_len: continue
                
                seqs.append(torch.from_numpy(window).float())
                try:
                    labels.append(torch.tensor(behaviors.index(inst["label"])).long())
                except ValueError:
                    seqs.pop()
        
        if not seqs: return [], []
        
        shuffled_pairs = list(zip(seqs, labels))
        random.shuffle(shuffled_pairs)
        seqs, labels = zip(*shuffled_pairs)
        
        return list(seqs), list(labels)

    def _load_dataset_common(self, name, split):
        """Helper to reduce code duplication between dataset loading methods."""
        dataset_path = os.path.join(self.datasets_dir, name)
        if not os.path.isdir(dataset_path): raise FileNotFoundError(dataset_path)
        with open(os.path.join(dataset_path, "labels.yaml"), "r") as f:
            label_config = yaml.safe_load(f)
        
        behaviors = label_config.get("behaviors", [])
        if not behaviors: return None, None, None
        
        all_insts = [inst for b in behaviors for inst in label_config.get("labels", {}).get(b, [])]
        if not all_insts: return [], [], behaviors
        random.shuffle(all_insts)

        inst_groups = {}
        for inst in all_insts:
            if 'video' in inst:
                inst_groups.setdefault(os.path.dirname(inst["video"]), []).append(inst)
        
        group_keys = list(inst_groups.keys()); random.shuffle(group_keys)
        split_idx = int((1 - split) * len(group_keys))
        train_keys, test_keys = group_keys[:split_idx], group_keys[split_idx:]
        
        train_insts = [inst for key in train_keys for inst in inst_groups[key]]
        test_insts = [inst for key in test_keys for inst in inst_groups[key]]
        
        return train_insts, test_insts, behaviors

    def load_dataset(self, name: str, seed: int = 42, split: float = 0.2, seq_len: int = 15, progress_callback=None) -> tuple:
        random.seed(seed)
        train_insts, test_insts, behaviors = self._load_dataset_common(name, split)
        if train_insts is None: return None, None
        
        def train_prog(p): progress_callback(p*0.5) if progress_callback else None
        def test_prog(p): progress_callback(50 + p*0.5) if progress_callback else None
        
        train_seqs, train_labels = self.convert_instances(train_insts, seq_len, behaviors, train_prog)
        test_seqs, test_labels = self.convert_instances(test_insts, seq_len, behaviors, test_prog)
        
        return BalancedDataset(train_seqs, train_labels, behaviors), BalancedDataset(test_seqs, test_labels, behaviors)

    def load_dataset_for_weighted_loss(self, name, seed=42, split=0.2, seq_len=15, progress_callback=None) -> tuple:
        random.seed(seed)
        train_insts, test_insts, behaviors = self._load_dataset_common(name, split)
        if train_insts is None: return None, None, None

        def train_prog(p): progress_callback(p*0.5) if progress_callback else None
        def test_prog(p): progress_callback(50 + p*0.5) if progress_callback else None
        
        train_seqs, train_labels = self.convert_instances(train_insts, seq_len, behaviors, train_prog)
        if not train_labels: return None, None, None
        
        class_counts = np.bincount([lbl.item() for lbl in train_labels], minlength=len(behaviors))
        weights = [sum(class_counts) / (len(behaviors) * c) if c > 0 else 0 for c in class_counts]

        test_seqs, test_labels = self.convert_instances(test_insts, seq_len, behaviors, test_prog)

        return StandardDataset(train_seqs, train_labels), StandardDataset(test_seqs, test_labels), weights


# =================================================================
# MODEL TRAINING & UTILITIES
# =================================================================

def collate_fn(batch):
    """Custom collate function for PyTorch DataLoader to stack sequences and labels."""
    dcls, lbls = zip(*batch)
    return torch.stack(dcls), torch.stack(lbls)

def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    """Returns the off-diagonal elements of a square matrix for covariance loss."""
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class PerformanceReport:
    """Simple container for scikit-learn classification report and confusion matrix."""
    def __init__(self, report: dict, cm: np.ndarray):
        self.sklearn_report = report
        self.confusion_matrix = cm

def train_lstm_model(train_set, test_set, seq_len: int, behaviors: list, batch_size=512, lr=1e-4, epochs=10, device=None, class_weights=None) -> tuple:
    """Main function to train the classifier head model."""
    if len(train_set) == 0: return None, None, -1
    if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle=False, collate_fn=collate_fn) if len(test_set) > 0 else None

    model = classifier_head.classifier(in_features=768, out_features=len(behaviors), seq_len=seq_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_weights = torch.tensor(class_weights, dtype=torch.float).to(device) if class_weights else None
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    
    best_f1, best_model_state, best_epoch = -1.0, None, -1
    epoch_reports = []

    for e in range(epochs):
        model.train()
        for i, (d, l) in enumerate(train_loader):
            d, l = d.to(device).float(), l.to(device)
            optimizer.zero_grad()
            lstm_logits, linear_logits, rawm = model(d)
            
            inv_loss = criterion(lstm_logits + linear_logits, l)
            
            rawm_centered = rawm - rawm.mean(dim=0)
            covm_loss = torch.tensor(0.0).to(device)
            if rawm_centered.ndim == 2 and rawm_centered.shape[0] > 1:
                # Original formulation for (Features x Features) covariance
                covm = (rawm_centered.T @ rawm_centered) / (rawm_centered.shape[0] - 1)
                # To regularize feature dimensions, not batch samples
                covm_loss = torch.sum(torch.pow(off_diagonal(covm), 2))
            
            loss = inv_loss + covm_loss
            loss.backward(); optimizer.step()
            if i % 50 == 0: print(f"[Epoch {e+1}/{epochs} Batch {i}/{len(train_loader)}] Loss: {loss.item():.4f}")

        if test_loader:
            model.eval()
            actuals, predictions = [], []
            with torch.no_grad():
                for d, l in test_loader:
                    logits = model.forward_nodrop(d.to(device).float())
                    actuals.extend(l.cpu().numpy())
                    predictions.extend(logits.argmax(1).cpu().numpy())
            
            if actuals:
                report = classification_report(actuals, predictions, target_names=behaviors, output_dict=True, zero_division=0)
                cm = confusion_matrix(actuals, predictions, labels=range(len(behaviors)))
                epoch_reports.append(PerformanceReport(report, cm))
                wf1 = report.get("weighted avg", {}).get("f1-score", -1.0)
                print(f"--- Epoch {e+1} Validation F1: {wf1:.4f} ---")
                if wf1 > best_f1:
                    best_f1, best_epoch, best_model_state = wf1, e, model.state_dict().copy()

    if best_model_state is None and epochs > 0: # Save last state if no improvement
        best_model_state = model.state_dict().copy()
        best_epoch = epochs - 1

    if best_model_state:
        final_model = classifier_head.classifier(in_features=768, out_features=len(behaviors), seq_len=seq_len)
        final_model.load_state_dict(best_model_state)
        return final_model.eval(), epoch_reports, best_epoch
    
    return None, None, -1