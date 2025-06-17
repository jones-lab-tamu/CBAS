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
import re
import threading

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
from scipy.signal import medfilt # Or use pandas rolling median

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

def encode_file(encoder: nn.Module, path: str, progress_callback=None) -> str | None:
    """
    Extracts DINOv2 embeddings from a video file in chunks to conserve memory
    and saves them to a resizable HDF5 file.

    Args:
        encoder (nn.Module): The pre-loaded DinoEncoder model.
        path (str): The absolute path to the input video (.mp4) file.

    Returns:
        str | None: The path to the created HDF5 file, or None if encoding failed.
    """
    try:
        # Use decord for efficient, hardware-accelerated video reading
        reader = decord.VideoReader(path, ctx=decord.cpu(0))
        video_len = len(reader)
    except Exception as e:
        print(f"Error reading video {path} with decord: {e}")
        return None

    if video_len == 0:
        print(f"Warning: Video {path} contains no frames. Skipping.")
        return None

    out_file_path = os.path.splitext(path)[0] + "_cls.h5"
    
    # Define a reasonable batch size for processing
    # 512 is a good starting point, adjust based on VRAM if needed
    CHUNK_SIZE = 512  

    # Create the HDF5 file and an infinitely resizable dataset
    with h5py.File(out_file_path, "w") as h5f:
        # Create dataset with initial shape, but allow for unlimited growth on the first axis
        dset = h5f.create_dataset(
            "cls", 
            (0, 768),  # Initial shape (0 frames, 768 features)
            maxshape=(None, 768), # (unlimited frames, 768 features)
            dtype='f4' # Use 32-bit float for compatibility
        )

        # Loop through the video in chunks
        for i in range(0, video_len, CHUNK_SIZE):
            # 1. Read a chunk of frames from the video
            end_index = min(i + CHUNK_SIZE, video_len)
            frames_np = reader.get_batch(range(i, end_index)).asnumpy()

            # 2. Pre-process the frames
            # Use green channel for B/W video, normalize to [0,1], set precision
            frames_tensor = torch.from_numpy(frames_np[:, :, :, 1] / 255.0).float()
            
            # 3. Get embeddings from the encoder
            # The .half() precision is now handled inside the amp autocast context for safety
            with torch.no_grad(), torch.amp.autocast(device_type=encoder.device.type if encoder.device.type != 'mps' else 'cpu'):
                embeddings_batch = encoder(frames_tensor.unsqueeze(1).to(encoder.device))

            # Squeeze out the extra dimension and move to CPU
            embeddings_out = embeddings_batch.squeeze(1).cpu().numpy()

            # 4. Append the results to the HDF5 dataset
            # Resize the dataset to accommodate the new chunk
            dset.resize(dset.shape[0] + len(embeddings_out), axis=0)
            # Write the new data into the newly created space
            dset[-len(embeddings_out):] = embeddings_out
            
            # Optional: Print progress to the console
            print(f"  - Encoded frames {i} to {end_index} of {video_len} for {os.path.basename(path)}")

    print(f"Successfully encoded {os.path.basename(path)} to {os.path.basename(out_file_path)}")
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

    # This handles the data loading robustly, which is good to keep.
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


def _create_matplotlib_actogram(binned_activity, light_cycle_booleans, tau, bin_size_minutes, plot_title, start_hour_offset, plot_acrophase=False, base_color=None):
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

    # Setup activity colormap (monochromatic or viridis)
    if base_color:
        # Create a transparent-to-solid colormap from the provided base color
        activity_cmap = LinearSegmentedColormap.from_list(
            'monochromatic_cmap', 
            [(0,0,0,0), base_color] # Goes from transparent to the specified color
        )
    else:
        # Fallback to the original viridis colormap
        cmap_viridis = plt.get_cmap('viridis')
        activity_colors = cmap_viridis(np.arange(cmap_viridis.N))
        activity_colors[0, 3] = 0 # Make the 'zero' value transparent
        activity_cmap = LinearSegmentedColormap.from_list('transparent_viridis', activity_colors)
    
    activity_cmap.set_bad(color=(0,0,0,0))

    # Setup Figure
    fig, ax = plt.subplots(figsize=(10, max(4, num_days * 0.4)), dpi=120)
    fig.patch.set_facecolor('#343a40')
    ax.set_facecolor('#343a40')
    
    plot_extent = [0, 2 * tau, num_days, 0]
    
    # Plotting
    ax.imshow(double_plotted_light, aspect='auto', cmap=cmap, interpolation='none', extent=plot_extent, vmin=0, vmax=1)
    non_zero_activity = [v for v in binned_activity if v > 0]
    vmax = np.percentile(non_zero_activity, 90) + 1e-6 if non_zero_activity else 1
    cax = ax.imshow(double_plotted_events, aspect='auto', cmap=activity_cmap, interpolation='none', extent=plot_extent, vmin=0, vmax=vmax)
    
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

        def sort_key(filepath):
            # Use regex to find the number sequence before the extension or _aug
            # This looks for one or more digits (\d+) that are followed by
            # either "_aug" or the ".mp4" extension.
            match = re.search(r'_(\d+)(?:_aug)?\.mp4', os.path.basename(filepath))
            if match:
                return int(match.group(1))
            return -1 # Fallback for non-matching names, sorts them first

        self.video_files = sorted(
            [f for f in all_files if f.endswith(".mp4")],
            key=sort_key
        )
        
        self.encoding_files = [f for f in all_files if f.endswith("_cls.h5")]
        self.unencoded_files = [vf for vf in self.video_files if vf.replace(".mp4", "_cls.h5") not in self.encoding_files]

        self.classifications = {}
        for csv_file in [f for f in all_files if f.endswith(".csv")]:
            try:
                # This part is for parsing model names from output files, it's unrelated
                # to the video sorting issue and should still work fine.
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

    def start_recording(self, session_name: str, segment_time: int) -> bool:
        if self.name in self.project.active_recordings: return False

        # The session_name is the top-level folder.
        # Path: .../recordings/Session Name/
        session_path = os.path.join(self.project.recordings_dir, session_name)
        # Path: .../recordings/Session Name/Camera Name/
        final_dest_dir = os.path.join(session_path, self.name)
        
        os.makedirs(final_dest_dir, exist_ok=True)
        dest_pattern = os.path.join(final_dest_dir, f"{self.name}_%05d.mp4")

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
    
    def update_instance_counts_in_config(self, project: 'Project'):
        """
        Recalculates train/test instance counts from labels.yaml and saves them
        to config.yaml. This is used after syncing or manual changes.
        """
        print(f"Updating instance counts for dataset: {self.name}")
        # We need to re-run the split logic to get the correct counts
        train_insts, test_insts, _ = project._load_dataset_common(self.name, 0.2)
        
        if train_insts is None or test_insts is None:
            print(f"Warning: Could not load instances to update counts for {self.name}.")
            return
            
        from collections import Counter # Local import is fine here
        
        train_instance_counts = Counter(inst['label'] for inst in train_insts)
        test_instance_counts = Counter(inst['label'] for inst in test_insts)
        train_frame_counts = Counter()
        for inst in train_insts:
            train_frame_counts[inst['label']] += (inst['end'] - inst['start'] + 1)
        test_frame_counts = Counter()
        for inst in test_insts:
            test_frame_counts[inst['label']] += (inst['end'] - inst['start'] + 1)
            
        # Update the config with the new counts
        for behavior_name in self.config.get("behaviors", []):
            train_n_inst = train_instance_counts.get(behavior_name, 0)
            train_n_frame = train_frame_counts.get(behavior_name, 0)
            test_n_inst = test_instance_counts.get(behavior_name, 0)
            test_n_frame = test_frame_counts.get(behavior_name, 0)
            
            self.update_metric(behavior_name, "Train #", f"{train_n_inst} ({int(train_n_frame)})")
            self.update_metric(behavior_name, "Test #", f"{test_n_inst} ({int(test_n_frame)})")
            # We can reset the performance metrics as they are now stale
            self.update_metric(behavior_name, "F1 Score", "N/A")
            self.update_metric(behavior_name, "Recall", "N/A")
            self.update_metric(behavior_name, "Precision", "N/A")
            
        print(f"Successfully updated and saved instance counts for {self.name}.")
    
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

    def predictions_to_instances_with_confidence(self, csv_path: str, model_name: str, threshold: float = 0.5, smoothing_window: int = 1) -> tuple[list, pd.DataFrame]:
        """
        Processes a prediction CSV to find behavior instances and their average confidence.
        Optionally applies a median filter to smooth the predictions before creating instances.
        
        Args:
            csv_path (str): Path to the prediction CSV file.
            model_name (str): The name of the model used for prediction.
            threshold (float): The probability threshold to consider a prediction valid.
            smoothing_window (int): The kernel size for the median filter. A value of 1 or less
                                    disables smoothing. Must be an odd number.
                                    
        Returns:
            A tuple containing:
            - A list of instance dictionaries, each with a 'confidence' key.
            - The full pandas DataFrame of probabilities.
        """
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            return [], None

        instances, behaviors = [], self.config.get("behaviors", [])
        if not behaviors or any(b not in df.columns for b in behaviors):
            return [], df

        # Get the raw predicted label and its probability for each frame
        df['predicted_label'] = df[behaviors].idxmax(axis=1)
        df['max_prob'] = df[behaviors].max(axis=1)

        # --- NEW: Conditional Smoothing Logic ---
        if smoothing_window > 1:
            # Ensure the window size is odd for the median filter
            if smoothing_window % 2 == 0:
                smoothing_window += 1
            
            # 1. Map string labels to integer indices for filtering
            behavior_map = {name: i for i, name in enumerate(behaviors)}
            df['predicted_index'] = df['predicted_label'].map(behavior_map).fillna(-1).astype(int)

            # 2. Apply the median filter on the integer indices
            df['smoothed_index'] = medfilt(df['predicted_index'], kernel_size=smoothing_window)

            # 3. Map the smoothed integer indices back to string labels
            index_to_behavior_map = {i: name for name, i in behavior_map.items()}
            df['label_for_grouping'] = df['smoothed_index'].map(index_to_behavior_map)
        else:
            # If no smoothing, use the original predictions for grouping
            df['label_for_grouping'] = df['predicted_label']
        # --- END of new logic ---

        # The rest of the function now iterates using the 'label_for_grouping' column
        in_event, current_event = False, {}
        for i, row in df.iterrows():
            # Use the original max_prob for the threshold check
            is_above_thresh = row['max_prob'] >= threshold
            
            # Check if the label for grouping is valid (not NaN from the map)
            if pd.isna(row['label_for_grouping']):
                if in_event: # End the current event if we hit an invalid label
                    in_event = False
                    current_event['end'] = i - 1
                    current_event['confidence'] = np.mean(current_event.pop('confidences', [0]))
                    if current_event['end'] >= current_event['start']:
                        instances.append(current_event)
                continue

            if not in_event and is_above_thresh:
                in_event = True
                current_event = {"video": csv_path.replace(f"_{model_name}_outputs.csv", ".mp4"), 
                                 "start": i, 
                                 "label": row['label_for_grouping'], # Use the potentially smoothed label
                                 "confidences": [row['max_prob']]}
            elif in_event:
                # End the event if threshold is not met OR the label changes
                if not is_above_thresh or row['label_for_grouping'] != current_event['label']:
                    in_event = False
                    current_event['end'] = i - 1
                    current_event['confidence'] = np.mean(current_event.pop('confidences', [0]))
                    if current_event['end'] >= current_event['start']:
                        instances.append(current_event)
                    
                    # Start a new event immediately if conditions are met
                    if is_above_thresh:
                        in_event = True
                        current_event = {"video": csv_path.replace(f"_{model_name}_outputs.csv", ".mp4"), 
                                         "start": i, 
                                         "label": row['label_for_grouping'], 
                                         "confidences": [row['max_prob']]}
                else: # Still in the same event, just append the confidence
                    current_event['confidences'].append(row['max_prob'])

        # Finalize the last event if the video ends during it
        if in_event and 'start' in current_event:
            current_event['end'] = len(df) - 1
            current_event['confidence'] = np.mean(current_event.pop('confidences', [0]))
            if current_event['end'] >= current_event['start']:
                instances.append(current_event)
                
        return instances, df


class Actogram:
    """Generates and holds data for an actogram visualization."""
    def __init__(self, directory: str, model: str, behavior: str, framerate: float, start: float, binsize_minutes: int, threshold: float, lightcycle: str, plot_acrophase: bool = False, base_color: str = None):
        self.directory, self.model, self.behavior = directory, model, behavior
        self.framerate, self.start_hour_on_plot = float(framerate), float(start)
        self.threshold, self.bin_size_minutes = float(threshold), int(binsize_minutes)
        self.plot_acrophase = plot_acrophase
        self.lightcycle_str = {"LL": "1"*24, "DD": "0"*24}.get(lightcycle, "1"*12 + "0"*12)
        self.blob = None # Initialize blob as None
        self.binned_activity = [] # <-- ADD THIS LINE

        if self.framerate <= 0 or self.bin_size_minutes <= 0: return
        self.binsize_frames = int(self.bin_size_minutes * self.framerate * 60)
        if self.binsize_frames <= 0: return

        # --- Correct data loading logic from before ---
        activity_per_frame = []
        all_csvs_for_model = [os.path.join(self.directory, f) for f in os.listdir(self.directory) if f.endswith(f"_{self.model}_outputs.csv")]
        if not all_csvs_for_model: return
        
        try:
            all_csvs_for_model.sort(key=lambda p: int(re.search(r'_(\d+)_' + self.model, os.path.basename(p)).group(1)))
        except (AttributeError, ValueError):
            all_csvs_for_model.sort()

        for file_path in all_csvs_for_model:
            df = pd.read_csv(file_path)
            if df.empty or self.behavior not in df.columns: continue
            probs = df[self.behavior].to_numpy()
            is_max = (df[df.columns.drop(self.behavior)].max(axis=1) < probs).to_numpy()
            activity_per_frame.extend((probs * is_max >= self.threshold).astype(float).tolist())
        
        if not activity_per_frame: return

        # Bin the complete, concatenated activity data
        binned_activity_result = [np.sum(activity_per_frame[i:i + self.binsize_frames]) for i in range(0, len(activity_per_frame), self.binsize_frames)]
        
        self.binned_activity = binned_activity_result

        if not self.binned_activity: return

        # Plot and encode
        fig = _create_matplotlib_actogram(self.binned_activity, [c=="1" for c in self.lightcycle_str], 24.0, self.bin_size_minutes, f"{model} - {behavior}", self.start_hour_on_plot, self.plot_acrophase, base_color)
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
        
    def reload(self):
        """
        Reloads all project components (datasets, models, etc.) from disk.
        """
        print("Project data reload requested. Re-scanning all directories...")
        # Simply re-run the same loading logic from the constructor
        self._load_cameras()
        self._load_recordings()
        self._load_models()
        self._load_datasets()
        print("Project data reloaded successfully.")
  
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

    def reload_recordings(self):
        """
        Re-scans the recordings directory and updates the project's internal state.
        This is useful after manual file operations like importing.
        """
        print("Reloading recording sessions from disk...")
        self._load_recordings()

    def _load_models(self):
        """
        Loads all models from the project's 'models' directory and also loads
        the bundled 'JonesLabModel' if it exists in the application's source.
        """
        self.models = {}
        
        # 1. Load user-specific models from the current project directory.
        #    These models are created by the user via the "Train" functionality.
        for model_dir in [d for d in os.scandir(self.models_dir) if d.is_dir()]:
            try:
                self.models[model_dir.name] = Model(model_dir.path)
            except Exception as e:
                print(f"Error loading project model {model_dir.path}: {e}")
        
        # 2. Load the bundled default model if it exists and isn't overridden by a user model.
        #    To disable this, a user can simply delete the 'JonesLabModel' folder from the
        #    application's 'models' directory.
        try:
            # Check if a user model isn't already named "JonesLabModel" to give user models priority.
            if "JonesLabModel" not in self.models:
                # Find the application's root directory relative to this script file.
                app_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                bundled_model_path = os.path.join(app_root_dir, "models", "JonesLabModel")
                
                if os.path.isdir(bundled_model_path):
                    print(f"Found bundled JonesLabModel at: {bundled_model_path}")
                    self.models["JonesLabModel"] = Model(bundled_model_path)
                # If the folder doesn't exist, we silently skip it. This is the
                # expected behavior if the user has intentionally deleted it.

        except Exception as e:
            # This will only run if there's an unexpected error, like a corrupted file.
            print(f"Warning: Could not load the bundled JonesLabModel: {e}")

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

    def convert_instances(self, project_root_path: str, insts: list, seq_len: int, behaviors: list, progress_callback=None) -> tuple:
        """Converts labeled instances from dicts into training-ready tensors."""
        seqs, labels = [], []
        half_seqlen = seq_len // 2

        instances_by_video = {}
        for inst in insts:
            video_path = inst.get("video")
            if video_path:
                instances_by_video.setdefault(video_path, []).append(inst)
        
        total_videos = len(instances_by_video)
        
        for i, (relative_video_path, video_instances) in enumerate(instances_by_video.items()):
            if progress_callback:
                progress_callback((i + 1) / total_videos * 100)
            
            # Use the passed-in project_root_path instead of gui_state
            absolute_video_path = os.path.join(project_root_path, relative_video_path)
            cls_path = os.path.splitext(absolute_video_path)[0] + "_cls.h5"
            
            if not os.path.exists(cls_path):
                print(f"Warning: H5 file not found, skipping instances for {relative_video_path}")
                continue

            try:
                with h5py.File(cls_path, "r") as f:
                    cls_arr = f["cls"][:]
            except Exception as e:
                print(f"Warning: Could not read H5 file {cls_path}: {e}")
                continue
            
            if cls_arr.ndim < 2 or cls_arr.shape[0] < seq_len:
                continue
            
            cls_centered = cls_arr - np.mean(cls_arr, axis=0)
            
            for inst in video_instances:
                start = int(inst.get("start", -1))
                end = int(inst.get("end", -1))
                
                if start == -1 or end == -1:
                    continue

                valid_start = max(half_seqlen, start)
                valid_end = int(min(cls_centered.shape[0] - half_seqlen, end))

                for frame_idx in range(valid_start, valid_end):
                    window_start = frame_idx - half_seqlen
                    window_end = frame_idx + half_seqlen + 1
                    
                    if window_end > cls_centered.shape[0]:
                        continue

                    window = cls_centered[window_start:window_end]
                    if window.shape[0] != seq_len:
                        continue
                    
                    try:
                        seqs.append(torch.from_numpy(window).float())
                        labels.append(torch.tensor(behaviors.index(inst["label"])).long())
                    except ValueError:
                        if seqs: seqs.pop()
        
        if not seqs:
            return [], []
        
        shuffled_pairs = list(zip(seqs, labels))
        random.shuffle(shuffled_pairs)
        seqs, labels = zip(*shuffled_pairs)
        
        return list(seqs), list(labels)
        
    def _load_dataset_common(self, name, split):
        """
        Helper to load a dataset, using a Stratified Group Split strategy.
        This ensures no data leakage (groups are not split between train/test)
        while guaranteeing all behaviors are represented in the test set to prevent crashes.
        """
        dataset_path = os.path.join(self.datasets_dir, name)
        if not os.path.isdir(dataset_path): raise FileNotFoundError(dataset_path)
        with open(os.path.join(dataset_path, "labels.yaml"), "r") as f:
            label_config = yaml.safe_load(f)
        
        behaviors = label_config.get("behaviors", [])
        if not behaviors: return None, None, None
        
        all_insts = [inst for b in behaviors for inst in label_config.get("labels", {}).get(b, [])]
        if not all_insts: return [], [], behaviors
        random.shuffle(all_insts)

        # --- New Stratified Group Splitting Logic ---

        # 1. Map groups (individual animals/cameras) to the behaviors and instances they contain.
        #    A "group" is now correctly identified by parsing the video's parent directory name.
        group_to_behaviors = {}
        group_to_instances = {}
        for inst in all_insts:
            if 'video' in inst and inst['video']:
                # The "group" is now the parent directory of the video file (e.g., "WTM1", "OVX2").
                group_key = os.path.basename(os.path.dirname(inst['video']))
                
                group_to_instances.setdefault(group_key, []).append(inst)
                group_to_behaviors.setdefault(group_key, set()).add(inst['label'])

        all_groups = list(group_to_instances.keys())
        random.shuffle(all_groups) # Shuffle to break ties randomly

        test_groups = set()
        behaviors_needed_in_test = set(behaviors)
        
        # 2. Greedily build a test set that covers all behaviors.
        # In each step, pick the group that adds the most *new* behaviors to the test set.
        while behaviors_needed_in_test:
            best_group = None
            best_group_coverage = -1
            
            available_groups = [g for g in all_groups if g not in test_groups]
            if not available_groups:
                print(f"Warning: Could not find groups to cover all behaviors. Missing: {behaviors_needed_in_test}")
                break

            for group in available_groups:
                newly_covered = len(behaviors_needed_in_test.intersection(group_to_behaviors.get(group, set())))
                if newly_covered > best_group_coverage:
                    best_group_coverage = newly_covered
                    best_group = group
            
            if best_group:
                test_groups.add(best_group)
                behaviors_needed_in_test.difference_update(group_to_behaviors.get(best_group, set()))
            else:
                break
        
        # 3. Fill the rest of the test set up to the desired split percentage.
        # Add remaining groups, prioritizing those with more instances.
        remaining_groups = [g for g in all_groups if g not in test_groups]
        remaining_groups.sort(key=lambda g: len(group_to_instances.get(g, [])), reverse=True)
        
        current_test_size = sum(len(group_to_instances.get(g, [])) for g in test_groups)
        total_size = len(all_insts)

        for group in remaining_groups:
            if current_test_size / total_size >= split:
                break
            if group not in test_groups:
                test_groups.add(group)
                current_test_size += len(group_to_instances.get(group, []))

        # 4. All other groups go to the training set.
        train_groups = [g for g in all_groups if g not in test_groups]

        # 5. Assemble the final instance lists.
        train_insts = [inst for group in train_groups for inst in group_to_instances[group]]
        test_insts = [inst for group in test_groups for inst in group_to_instances[group]]
        
        random.shuffle(train_insts)
        random.shuffle(test_insts)
        
        print(f"Stratified Group Split: {len(train_insts)} train instances, {len(test_insts)} test instances.")
        print(f"  - Train groups: {len(train_groups)}, Test groups: {len(test_groups)}")

        if not test_insts and train_insts:
            print("  - Warning: Stratified split resulted in an empty test set. Falling back to 80/20 instance split.")
            split_idx = int(len(all_insts) * (1 - split))
            return all_insts[:split_idx], all_insts[split_idx:], behaviors

        return train_insts, test_insts, behaviors

    def load_dataset(self, name: str, seed: int = 42, split: float = 0.2, seq_len: int = 15, progress_callback=None) -> tuple:
        random.seed(seed)
        train_insts, test_insts, behaviors = self._load_dataset_common(name, split)
        if train_insts is None: return None, None, None, None
        
        def train_prog(p): progress_callback(p*0.5) if progress_callback else None
        def test_prog(p): progress_callback(50 + p*0.5) if progress_callback else None
        
        # Pass self.path to the function call
        train_seqs, train_labels = self.convert_instances(self.path, train_insts, seq_len, behaviors, train_prog)
        test_seqs, test_labels = self.convert_instances(self.path, test_insts, seq_len, behaviors, test_prog)
        
        return BalancedDataset(train_seqs, train_labels, behaviors), StandardDataset(test_seqs, test_labels), train_insts, test_insts

    def load_dataset_for_weighted_loss(self, name, seed=42, split=0.2, seq_len=15, progress_callback=None) -> tuple:
        random.seed(seed)
        train_insts, test_insts, behaviors = self._load_dataset_common(name, split)
        if train_insts is None: return None, None, None, None, None

        def train_prog(p): progress_callback(p*0.5) if progress_callback else None
        def test_prog(p): progress_callback(50 + p*0.5) if progress_callback else None
        
        # Pass self.path to the function call
        train_seqs, train_labels = self.convert_instances(self.path, train_insts, seq_len, behaviors, train_prog)
        if not train_labels: return None, None, None, None, None
        
        class_counts = np.bincount([lbl.item() for lbl in train_labels], minlength=len(behaviors))
        weights = [sum(class_counts) / (len(behaviors) * c) if c > 0 else 0 for c in class_counts]

        # Pass self.path to the function call
        test_seqs, test_labels = self.convert_instances(self.path, test_insts, seq_len, behaviors, test_prog)

        return StandardDataset(train_seqs, train_labels), StandardDataset(test_seqs, test_labels), weights, train_insts, test_insts


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
    """Simple container for training and validation reports for a single epoch."""
    def __init__(self, train_report: dict, train_cm: np.ndarray, val_report: dict, val_cm: np.ndarray):
        self.train_report = train_report
        self.train_cm = train_cm
        self.val_report = val_report
        self.val_cm = val_cm

# In cbas.py

def train_lstm_model(train_set, test_set, seq_len: int, behaviors: list, cancel_event: threading.Event, batch_size=512, lr=1e-4, epochs=10, device=None, class_weights=None, patience=3) -> tuple:
    """Main function to train the classifier head model, now with cancellation support."""
    if len(train_set) == 0: return None, None, -1
    if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Use drop_last=True for the train loader to ensure consistent batch sizes for some calculations
    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=(device.type == 'cuda'), drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle=False, collate_fn=collate_fn) if len(test_set) > 0 else None

    model = classifier_head.classifier(in_features=768, out_features=len(behaviors), seq_len=seq_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_weights = torch.tensor(class_weights, dtype=torch.float).to(device) if class_weights else None
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    
    best_f1, best_model_state, best_epoch = -1.0, None, -1
    epoch_reports = []
    epochs_no_improve = 0

    for e in range(epochs):
        # Check for cancellation signal at the start of each epoch
        if cancel_event.is_set():
            print(f"Cancellation detected within training loop at epoch {e+1}.")
            return None, epoch_reports, best_epoch

        # --- Training Phase ---
        model.train()
        for i, (d, l) in enumerate(train_loader):
            d, l = d.to(device).float(), l.to(device)
            optimizer.zero_grad()
            lstm_logits, linear_logits, rawm = model(d)
            
            inv_loss = criterion(lstm_logits + linear_logits, l)
            
            rawm_centered = rawm - rawm.mean(dim=0)
            covm_loss = torch.tensor(0.0).to(device)
            if rawm_centered.ndim == 2 and rawm_centered.shape[0] > 1:
                covm = (rawm_centered.T @ rawm_centered) / (rawm_centered.shape[0] - 1)
                covm_loss = torch.sum(torch.pow(off_diagonal(covm), 2))
            
            loss = inv_loss + covm_loss
            loss.backward()
            optimizer.step()
            if i % 50 == 0: print(f"[Epoch {e+1}/{epochs} Batch {i}/{len(train_loader)}] Loss: {loss.item():.4f}")

        # --- Evaluation Phase ---
        model.eval()
        
        # 1. Evaluate on Training Set
        train_actuals, train_predictions = [], []
        with torch.no_grad():
            for d, l in train_loader:
                logits = model.forward_nodrop(d.to(device).float())
                train_actuals.extend(l.cpu().numpy())
                train_predictions.extend(logits.argmax(1).cpu().numpy())
        
        train_report = classification_report(train_actuals, train_predictions, target_names=behaviors, output_dict=True, zero_division=0)
        train_cm = confusion_matrix(train_actuals, train_predictions, labels=range(len(behaviors)))

        # 2. Evaluate on Validation (Test) Set
        val_report, val_cm = {}, np.array([])
        if test_loader:
            val_actuals, val_predictions = [], []
            with torch.no_grad():
                for d, l in test_loader:
                    logits = model.forward_nodrop(d.to(device).float())
                    val_actuals.extend(l.cpu().numpy())
                    val_predictions.extend(logits.argmax(1).cpu().numpy())
            
            if val_actuals:
                val_report = classification_report(val_actuals, val_predictions, target_names=behaviors, output_dict=True, zero_division=0)
                val_cm = confusion_matrix(val_actuals, val_predictions, labels=range(len(behaviors)))
        
        epoch_reports.append(PerformanceReport(train_report, train_cm, val_report, val_cm))

        # --- Early Stopping Logic (based on VALIDATION F1 score) ---
        current_val_f1 = val_report.get("weighted avg", {}).get("f1-score", -1.0)
        print(f"--- Epoch {e+1} | Train F1: {train_report.get('weighted avg', {}).get('f1-score', -1.0):.4f} | Val F1: {current_val_f1:.4f} ---")
        
        if current_val_f1 > best_f1:
            best_f1, best_epoch, best_model_state = current_val_f1, e, model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if test_loader and epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            from workthreads import log_message
            log_message(f"Early stopping triggered at epoch {e + 1}.", "INFO")
            break
            
    if best_model_state is None and epochs > 0:
        if not test_loader: # Only save last state if there was no validation set at all
             best_model_state = model.state_dict().copy()
             best_epoch = epochs - 1

    if best_model_state:
        final_model = classifier_head.classifier(in_features=768, out_features=len(behaviors), seq_len=seq_len)
        final_model.load_state_dict(best_model_state)
        return final_model.eval(), epoch_reports, best_epoch
    
    return None, None, -1