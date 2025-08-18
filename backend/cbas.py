"""
Core backend logic and data models for the CBAS application.
This version uses the proven v2-stable logic for recording.
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
import random # Other parts of the code still use it
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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import medfilt

# Local application imports
import classifier_head
import gui_state
from subprocess import Popen, PIPE, DEVNULL 
import sys

class InvalidProject(Exception):
    def __init__(self, path):
        super().__init__(f"Path '{path}' is not a valid CBAS project directory.")

def encode_file(encoder: nn.Module, path: str, progress_callback=None) -> str | None:
    try:
        reader = decord.VideoReader(path, ctx=decord.cpu(0))
        video_len = len(reader)
    except Exception as e:
        print(f"Error reading video {path} with decord: {e}")
        return None
    if video_len == 0:
        print(f"Warning: Video {path} contains no frames. Skipping.")
        return None
        
    out_file_path = os.path.splitext(path)[0] + "_cls.h5"
    CHUNK_SIZE = 512
    
    with h5py.File(out_file_path, "w") as h5f:

        # Add the encoder's model identifier as a file-level attribute.
        # This "stamps" the file with its creator's identity.
        if gui_state.proj:
            h5f.attrs['encoder_model_identifier'] = gui_state.proj.encoder_model_identifier

        dset = h5f.create_dataset("cls", (0, 768), maxshape=(None, 768), dtype='f4')
        for i in range(0, video_len, CHUNK_SIZE):
            end_index = min(i + CHUNK_SIZE, video_len)
            frames_np = reader.get_batch(range(i, end_index)).asnumpy()
            
            if progress_callback:
                percent_complete = (end_index / video_len) * 100
                progress_callback(percent_complete)         
                       
            frames_tensor = torch.from_numpy(frames_np[:, :, :, 1] / 255.0).float()
            with torch.no_grad(), torch.amp.autocast(device_type=encoder.device.type if encoder.device.type != 'mps' else 'cpu'):
                embeddings_batch = encoder(frames_tensor.unsqueeze(1).to(encoder.device))
            embeddings_out = embeddings_batch.squeeze(1).cpu().numpy()
            dset.resize(dset.shape[0] + len(embeddings_out), axis=0)
            dset[-len(embeddings_out):] = embeddings_out
            
    print(f"Successfully encoded {os.path.basename(path)} to {os.path.basename(out_file_path)}")
    return out_file_path

def infer_file(file_path: str, model: nn.Module, dataset_name: str, behaviors: list[str], seq_len: int, device=None) -> str | None:
    """
    Runs inference on a single HDF5 file and saves the results to a CSV.
    This version is updated to be compatible with the final model API.
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
        
    cls_np_f32 = cls_np.astype(np.float32)
    cls = torch.from_numpy(cls_np_f32) # No need to mean-center here, model does it
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # The calling function (ClassificationThread.run) is responsible for setting model.eval()
    predictions, batch_windows = [], []
    half_seqlen = seq_len // 2
    for i in range(half_seqlen, len(cls) - half_seqlen):
        window = cls[i - half_seqlen: i + half_seqlen + 1]
        batch_windows.append(window)
        if len(batch_windows) >= 4096 or i == len(cls) - half_seqlen - 1:
            if not batch_windows: continue
            batch_tensor = torch.stack(batch_windows)
            with torch.no_grad():
                # Call the unified forward method and unpack correctly
                # The model is already in eval mode, so dropout is off.
                logits, _ = model(batch_tensor.to(device))
            predictions.extend(torch.softmax(logits, dim=1).cpu().numpy())
            batch_windows = []
            
    if not predictions: return None
    
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
    if base_color:
        activity_cmap = LinearSegmentedColormap.from_list('monochromatic_cmap', [(0,0,0,0), base_color])
    else:
        cmap_viridis = plt.get_cmap('viridis')
        activity_colors = cmap_viridis(np.arange(cmap_viridis.N))
        activity_colors[0, 3] = 0
        activity_cmap = LinearSegmentedColormap.from_list('transparent_viridis', activity_colors)
    activity_cmap.set_bad(color=(0,0,0,0))
    fig, ax = plt.subplots(figsize=(10, max(4, num_days * 0.4)), dpi=120)
    fig.patch.set_facecolor('#343a40')
    ax.set_facecolor('#343a40')
    plot_extent = [0, 2 * tau, num_days, 0]
    ax.imshow(double_plotted_light, aspect='auto', cmap=cmap, interpolation='none', extent=plot_extent, vmin=0, vmax=1)
    non_zero_activity = [v for v in binned_activity if v > 0]
    vmax = np.percentile(non_zero_activity, 90) + 1e-6 if non_zero_activity else 1
    cax = ax.imshow(double_plotted_events, aspect='auto', cmap=activity_cmap, interpolation='none', extent=plot_extent, vmin=0, vmax=vmax)
    if acrophase_points:
        for day_idx, hour in acrophase_points:
            ax.plot(hour, day_idx + 0.5, 'o', color='red', markersize=8, markeredgecolor='black')
            ax.plot(hour + tau, day_idx + 0.5, 'o', color='red', markersize=8, markeredgecolor='black')
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

class DinoEncoder(nn.Module):
    def __init__(self, model_identifier: str, device="cuda"):
        super().__init__()
        self.device = torch.device(device)
        
        print(f"Loading DINO encoder model: {model_identifier}")
        try:
            self.model = transformers.AutoModel.from_pretrained(model_identifier).to(self.device)
        except Exception as e:
            # Provide a helpful error message to the user via the log
            from workthreads import log_message
            log_message("--- MODEL LOADING FAILED ---", "ERROR")
            log_message(f"Could not load the encoder model: '{model_identifier}'.", "ERROR")
            log_message("If you are trying to use a new or gated model (like DINOv3), please ensure you have:", "ERROR")
            log_message("1. Logged into the Hugging Face Hub ('huggingface-cli login').", "ERROR")
            log_message("2. Agreed to the model's terms of use on its Hugging Face page.", "ERROR")
            log_message("3. Installed the latest version of 'transformers' from source if required.", "ERROR")
            log_message(f"Original error: {e}", "ERROR")
            # Re-raise the exception to stop the application startup
            raise e

        self.model.eval()
        for param in self.model.parameters(): param.requires_grad = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, H, W = x.shape
        x = x.to(self.device).unsqueeze(2).repeat(1, 1, 3, 1, 1).reshape(B * S, 3, H, W)
        with torch.no_grad():
            out = self.model(x)
        return out.last_hidden_state[:, 0, :].reshape(B, S, 768)

class Recording:
    def __init__(self, path: str):
        if not os.path.isdir(path): raise FileNotFoundError(path)
        self.path = path
        self.name = os.path.basename(path)
        all_files = [f.path for f in os.scandir(self.path) if f.is_file()]
        
        def sort_key(filepath):
            match = re.search(r'_(\d+)(?:_aug)?\.mp4', os.path.basename(filepath))
            if match: return int(match.group(1))
            return -1
            
        self.video_files = sorted([f for f in all_files if f.endswith(".mp4")], key=sort_key)
        self.encoding_files = [f for f in all_files if f.endswith("_cls.h5")]
        self.unencoded_files = [vf for vf in self.video_files if vf.replace(".mp4", "_cls.h5") not in self.encoding_files]
        
        self.classifications = {}
        for csv_file_path in [f for f in all_files if f.endswith(".csv")]:
            base_name = os.path.basename(csv_file_path)
            
            # Ensure the file is a classification output file
            if base_name.endswith("_outputs.csv"):
                # Isolate the part of the filename that contains the video name and model name
                # e.g., "OVX1_00045_cbas_aug" from "OVX1_00045_cbas_aug_outputs.csv"
                name_part = base_name[:-12] # len("_outputs.csv") is 12
                
                # Find which video file this CSV corresponds to by checking prefixes
                matched_video_base = None
                for vf in self.video_files:
                    vf_base = os.path.splitext(os.path.basename(vf))[0]
                    if name_part.startswith(vf_base):
                        # We found a match, e.g., "OVX1_00045_cbas_aug" starts with "OVX1_00045"
                        matched_video_base = vf_base
                        break
                
                if matched_video_base:
                    # The model name is what's left after removing the video name and the connecting underscore
                    model_name = name_part[len(matched_video_base) + 1:]
                    self.classifications.setdefault(model_name, []).append(csv_file_path)

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
            "segment_seconds": self.segment_seconds,
        }

    def update_settings(self, settings: dict, write_to_disk: bool = True):
        self.rtsp_url = str(settings.get("rtsp_url", ""))
        self.framerate = int(settings.get("framerate", 10))
        self.resolution = int(settings.get("resolution", 256))
        self.segment_seconds = int(settings.get("segment_seconds", 600))
        self.crop_left_x = float(settings.get("crop_left_x", 0.0))
        self.crop_top_y = float(settings.get("crop_top_y", 0.0))
        self.crop_width = float(settings.get("crop_width", 1.0))
        self.crop_height = float(settings.get("crop_height", 1.0))

        # This ensures we always have the high-quality stream URL available,
        # regardless of what the user entered.
        if "/profile1" in self.rtsp_url:
            self.profile0_url = self.rtsp_url.replace("/profile1", "/profile0")
        else:
            self.profile0_url = self.rtsp_url # Assume it's already profile0 or a different URL structure

        if write_to_disk: self.write_settings_to_config()

    def write_settings_to_config(self):
        """Saves the current camera object's settings to its config.yaml file."""
        with open(os.path.join(self.path, "config.yaml"), "w") as file:
            yaml.dump(self.settings_to_dict(), file, allow_unicode=True)

    def start_recording(self, session_name: str) -> bool:
        if self.name in self.project.active_recordings:
            print(f"[{self.name}] is already recording.")
            return False

        self.project.current_session_name = session_name
        recording_url = self.profile0_url
        print(f"[{self.name}] Using high-quality stream '{recording_url}' for recording.")
        
        session_path = os.path.join(self.project.recordings_dir, session_name)
        final_dest_dir = os.path.join(session_path, self.name)
        os.makedirs(final_dest_dir, exist_ok=True)
        
        playlist_file = os.path.join(final_dest_dir, f"{self.name}_playlist.m3u8")
        ffmpeg_log_path = os.path.join(final_dest_dir, f"{self.name}_ffmpeg_err.log")
        dest_pattern = os.path.join(final_dest_dir, f"{self.name}_%05d.mp4")

        filter_string = (
            f"crop=iw*{self.crop_width}:ih*{self.crop_height}:iw*{self.crop_left_x}:ih*{self.crop_top_y},"
            f"scale={self.resolution}:{self.resolution}:force_original_aspect_ratio=decrease,"
            f"pad={self.resolution}:{self.resolution}:(ow-iw)/2:(oh-ih)/2"
        )

        command = [
            'ffmpeg', '-hide_banner', '-loglevel', 'warning',
            '-rtsp_transport', 'tcp', '-timeout', '15000000',
            '-stream_loop', '-1', # Using the most stable flag for your ffmpeg version
            '-i', recording_url,
            '-vf', filter_string, '-r', str(self.framerate), '-an', '-c:v', 'libx264',
            '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', '-g', str(self.framerate * 2),
            '-sc_threshold', '0', '-f', 'hls', '-hls_time', str(self.segment_seconds),
            '-hls_list_size', '0', '-hls_flags', 'delete_segments+program_date_time',
            '-hls_segment_filename', dest_pattern, '-y', playlist_file
        ]
        
        try:
            print(f"Starting recording for {self.name} with command: {' '.join(command)}")
            creation_flags = 0
            if sys.platform == "win32": creation_flags = subprocess.CREATE_NO_WINDOW
            ffmpeg_log_file = open(ffmpeg_log_path, "a")
            process = Popen(
                command, stdin=PIPE, stdout=DEVNULL, stderr=ffmpeg_log_file,
                shell=False, creationflags=creation_flags
            )
            # Store the process, start time, AND session name for restarts
            self.project.active_recordings[self.name] = (process, time.time(), session_name)
            return True
        except Exception as e:
            print(f"Failed to start ffmpeg for {self.name}: {e}")
            return False
            
    def stop_recording(self) -> bool:
        if self.name in self.project.active_recordings:
            process, _, _ = self.project.active_recordings.pop(self.name) # Unpack the 3-item tuple
            try:
                if process.stdin:
                    process.stdin.write(b'q')
                    process.stdin.flush()
                    process.stdin.close()
                process.wait(timeout=5)
            except Exception as e:
                print(f"Error while stopping process for {self.name}: {e}. Killing process.")
                process.kill()

            # After stopping, find the very last created video file and queue it.
            # This ensures the final piece of the recording is always processed.
            try:
                # Check if a session name has been set
                if self.project.current_session_name:
                    camera_folder = os.path.join(self.project.recordings_dir, self.project.current_session_name, self.name)
                    if os.path.isdir(camera_folder):
                        # Find all .mp4 files and get the one with the latest modification time.
                        video_files = [os.path.join(camera_folder, f) for f in os.listdir(camera_folder) if f.endswith('.mp4')]
                        if video_files:
                            latest_file = max(video_files, key=os.path.getmtime)
                            with gui_state.encode_lock:
                                if latest_file not in gui_state.encode_tasks:
                                    gui_state.encode_tasks.append(latest_file)
                                    from workthreads import log_message # Local import to avoid circular dependency
                                    log_message(f"Queued final segment on stop: '{os.path.basename(latest_file)}'", "INFO")
            except Exception as e:
                # Use a local import here too to be safe
                from workthreads import log_message
                log_message(f"Could not queue final segment for {self.name}: {e}", "ERROR")

            return True
        return False

class Model:
    def __init__(self, path: str):
        self.path = path
        self.name = os.path.basename(path)
        self.config_path = os.path.join(path, "config.yaml")
        self.weights_path = os.path.join(path, "model.pth")
        if not os.path.exists(self.config_path): raise FileNotFoundError(f"Model config not found: {self.config_path}")
        with open(self.config_path) as f: self.config = yaml.safe_load(f)
        if not os.path.exists(self.weights_path): raise FileNotFoundError(f"Model weights not found: {self.weights_path}")

class Dataset:
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
        from collections import Counter
        train_insts, test_insts, _ = project._load_dataset_common(self.name, 0.2, seed=42) # Provide a default seed
        if train_insts is None or test_insts is None: return
        train_instance_counts = Counter(inst['label'] for inst in train_insts)
        test_instance_counts = Counter(inst['label'] for inst in test_insts)
        train_frame_counts = Counter()
        for inst in train_insts:
            train_frame_counts[inst['label']] += (inst['end'] - inst['start'] + 1)
        test_frame_counts = Counter()
        for inst in test_insts:
            test_frame_counts[inst['label']] += (inst['end'] - inst['start'] + 1)
        for behavior_name in self.config.get("behaviors", []):
            train_n_inst = train_instance_counts.get(behavior_name, 0)
            train_n_frame = train_frame_counts.get(behavior_name, 0)
            test_n_inst = test_instance_counts.get(behavior_name, 0)
            test_n_frame = test_frame_counts.get(behavior_name, 0)
            self.update_metric(behavior_name, "Train #", f"{train_n_inst} ({int(train_n_frame)})")
            self.update_metric(behavior_name, "Test #", f"{test_n_inst} ({int(test_n_frame)})")
            self.update_metric(behavior_name, "F1 Score", "N/A")
            self.update_metric(behavior_name, "Recall", "N/A")
            self.update_metric(behavior_name, "Precision", "N/A")
            
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
                if is_above_thresh:
                    in_event = True
                    current_event = {"video": csv_path.replace(f"_{model_name}_outputs.csv", ".mp4"), "start": i, "label": row['predicted_label']}
        if in_event:
            current_event['end'] = len(df) - 1
            if current_event['end'] >= current_event['start']: instances.append(current_event)
        return instances
        
    def predictions_to_instances_with_confidence(self, csv_path: str, model_name: str, threshold: float = 0.5, smoothing_window: int = 1) -> tuple[list, pd.DataFrame]:
        try: df = pd.read_csv(csv_path)
        except FileNotFoundError: return [], None
        instances, behaviors = [], self.config.get("behaviors", [])
        if not behaviors or any(b not in df.columns for b in behaviors): return [], df
        df['predicted_label'] = df[behaviors].idxmax(axis=1)
        df['max_prob'] = df[behaviors].max(axis=1)
        if smoothing_window > 1:
            if smoothing_window % 2 == 0: smoothing_window += 1
            behavior_map = {name: i for i, name in enumerate(behaviors)}
            df['predicted_index'] = df['predicted_label'].map(behavior_map).fillna(-1).astype(int)
            df['smoothed_index'] = medfilt(df['predicted_index'], kernel_size=smoothing_window)
            index_to_behavior_map = {i: name for name, i in behavior_map.items()}
            df['label_for_grouping'] = df['smoothed_index'].map(index_to_behavior_map)
        else:
            df['label_for_grouping'] = df['predicted_label']
        df['block_start'] = df['label_for_grouping'].ne(df['label_for_grouping'].shift())
        block_start_indices = df[df['block_start']].index.tolist()
        if len(df) not in block_start_indices: block_start_indices.append(len(df))
        for i in range(len(block_start_indices) - 1):
            start_idx, end_idx = block_start_indices[i], block_start_indices[i+1] - 1
            block_label = df['label_for_grouping'].iloc[start_idx]
            if pd.isna(block_label): continue
            block_confidence = df['max_prob'].iloc[start_idx:end_idx + 1].mean()
            absolute_video_path = csv_path.replace(f"_{model_name}_outputs.csv", ".mp4")
            relative_video_path = os.path.relpath(absolute_video_path, start=gui_state.proj.path).replace('\\', '/')
            new_instance = {"video": relative_video_path, "start": start_idx,"end": end_idx,"label": block_label,"confidence": block_confidence}
            instances.append(new_instance)
        return instances, df

class Actogram:
    def __init__(self, behavior: str, framerate: float, start: float, binsize_minutes: int, threshold: float, lightcycle: str, plot_acrophase: bool = False, base_color: str = None, directory: str = None, model: str = None, preloaded_df: pd.DataFrame = None):
        
        self.behavior = behavior
        self.framerate, self.start_hour_on_plot = float(framerate), float(start)
        self.threshold, self.bin_size_minutes = float(threshold), int(binsize_minutes)
        self.plot_acrophase = plot_acrophase
        self.lightcycle_str = {"LL": "1"*24, "DD": "0"*24}.get(lightcycle, "1"*12 + "0"*12)
        self.blob = None
        self.binned_activity = []

        if self.framerate <= 0 or self.bin_size_minutes <= 0: return
        self.binsize_frames = int(self.bin_size_minutes * self.framerate * 60)
        if self.binsize_frames <= 0: return

        activity_per_frame = []
        
        # Use preloaded_df if available 
        if preloaded_df is not None:
            if self.behavior in preloaded_df.columns:
                probs = preloaded_df[self.behavior].to_numpy()
                is_max = (preloaded_df[preloaded_df.columns.drop(self.behavior)].max(axis=1) < probs).to_numpy()
                activity_per_frame.extend((probs * is_max >= self.threshold).astype(float).tolist())
        # Original logic if no preloaded_df
        elif directory and model:
            all_csvs_for_model = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(f"_{model}_outputs.csv")]
            if not all_csvs_for_model: return
            try:
                all_csvs_for_model.sort(key=lambda p: int(re.search(r'_(\d+)_' + model, os.path.basename(p)).group(1)))
            except (AttributeError, ValueError):
                all_csvs_for_model.sort()
            
            for file_path in all_csvs_for_model:
                df = pd.read_csv(file_path)
                if df.empty or self.behavior not in df.columns: continue
                probs = df[self.behavior].to_numpy()
                is_max = (df[df.columns.drop(self.behavior)].max(axis=1) < probs).to_numpy()
                activity_per_frame.extend((probs * is_max >= self.threshold).astype(float).tolist())
        else:
             # Not enough information to proceed
             return
            
        if not activity_per_frame: return
        
        # Binning and plotting
        self.binned_activity = [np.sum(activity_per_frame[i:i + self.binsize_frames]) for i in range(0, len(activity_per_frame), self.binsize_frames)]
        if not self.binned_activity: return
        fig = _create_matplotlib_actogram(self.binned_activity, [c=="1" for c in self.lightcycle_str], 24.0, self.bin_size_minutes, f"{model} - {behavior}", self.start_hour_on_plot, self.plot_acrophase, base_color)
        if fig:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#343a40')
            buf.seek(0)
            self.blob = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)

class StandardDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels): self.sequences, self.labels = sequences, labels
    def __len__(self): return len(self.sequences)
    def __getitem__(self, idx): return self.sequences[idx], self.labels[idx]

class BalancedDataset(torch.utils.data.Dataset):
    def __init__(self, sequences: list, labels: list, behaviors: list):
        self.behaviors, self.num_behaviors = behaviors, len(behaviors)
        self.buckets = {b: [] for b in self.behaviors}
        for seq, label in zip(sequences, labels):
            if 0 <= label.item() < self.num_behaviors:
                self.buckets[self.behaviors[label.item()]].append(seq)
        
        # Filter out behaviors that have no samples to prevent errors.
        self.available_behaviors = [b for b in self.behaviors if self.buckets[b]]
        self.num_available_behaviors = len(self.available_behaviors)

        self.total_sequences = sum(len(b) for b in self.buckets.values())
        self.counter = 0
        
    def __len__(self):
        # Base the length on the number of *available* behaviors.
        if self.num_available_behaviors == 0: return 0
        return self.total_sequences + (self.num_available_behaviors - self.total_sequences % self.num_available_behaviors) % self.num_available_behaviors
        
    def __getitem__(self, idx: int):
        if self.num_available_behaviors == 0:
            raise IndexError("No behaviors with samples available in this dataset split.")

        # Cycle through only the behaviors that actually have samples.
        b_idx_in_available = self.counter % self.num_available_behaviors
        b_name = self.available_behaviors[b_idx_in_available]
        
        # Get the original index of this behavior to return the correct label tensor.
        original_b_idx = self.behaviors.index(b_name)
        
        self.counter += 1
        
        # The original check is now implicitly handled by the loop.
        # This line is no longer needed: if not self.buckets[b_name]: raise IndexError(f"Behavior '{b_name}' has no samples.")
        
        sample_idx = idx % len(self.buckets[b_name])
        return self.buckets[b_name][sample_idx], torch.tensor(original_b_idx).long()

class Project:
    def __init__(self, path: str):
        if not os.path.isdir(path): raise InvalidProject(path)
        self.path = path
        self.cameras_dir = os.path.join(path, "cameras")
        self.recordings_dir = os.path.join(path, "recordings")
        self.models_dir = os.path.join(path, "models")
        self.datasets_dir = os.path.join(path, "data_sets")
        for subdir in [self.cameras_dir, self.recordings_dir, self.models_dir, self.datasets_dir]:
            os.makedirs(subdir, exist_ok=True)
        

        # Load the project-specific config file if it exists
        self.project_config = {}
        config_path = os.path.join(self.path, "cbas_config.yaml")
        if os.path.exists(config_path):
            print(f"Found project-specific config file: {config_path}")
            try:
                with open(config_path, 'r') as f:
                    self.project_config = yaml.safe_load(f)
            except Exception as e:
                print(f"WARNING: Could not read or parse cbas_config.yaml. Using defaults. Error: {e}")
        
        # Determine the encoder model to use for this project
        self.encoder_model_identifier = self.project_config.get(
            "encoder_model_identifier", 
            "facebook/dinov2-with-registers-base" # DINOv2 with registers, the safe default
        )

        self.active_recordings: dict[str, tuple[subprocess.Popen, float, str]] = {}
        self.current_session_name: str | None = None
        self._load_cameras()
        self._load_recordings()
        self._load_models()
        self._load_datasets()
        
    def reload(self):
        print("Project data reload requested. Re-scanning all directories...")
        self._load_cameras()
        self._load_recordings()
        self._load_models()
        self._load_datasets()
        print("Project data reloaded successfully.")
        
    def _load_cameras(self):
        self.cameras = {}
        for cam_dir in [d for d in os.scandir(self.cameras_dir) if d.is_dir()]:
            config_path = os.path.join(cam_dir.path, "config.yaml")
            if os.path.exists(config_path):
                try:
                    with open(config_path) as f: config = yaml.safe_load(f)
                    if "name" in config:
                        self.cameras[config["name"]] = Camera(config, self)
                except Exception as e: print(f"Error loading camera config {config_path}: {e}")
                
    def _load_recordings(self):
        self.recordings = {}
        for day_dir in [d for d in os.scandir(self.recordings_dir) if d.is_dir()]:
            self.recordings[day_dir.name] = {}
            for session_dir in [d for d in os.scandir(day_dir.path) if d.is_dir()]:
                try:
                    rec = Recording(session_dir.path)
                    self.recordings[day_dir.name][rec.name] = rec
                except Exception as e: print(f"Error loading recording {session_dir.path}: {e}")
                
    def reload_recordings(self):
        print("Reloading recording sessions from disk...")
        self._load_recordings()
        
    def _load_models(self):
        self.models = {}
        for model_dir in [d for d in os.scandir(self.models_dir) if d.is_dir()]:
            try: self.models[model_dir.name] = Model(model_dir.path)
            except Exception as e: print(f"Error loading project model {model_dir.path}: {e}")
        
    def _load_datasets(self):
        self.datasets = {}
        for ds_dir in [d for d in os.scandir(self.datasets_dir) if d.is_dir()]:
            try: self.datasets[ds_dir.name] = Dataset(ds_dir.path)
            except Exception as e: print(f"Error loading dataset {ds_dir.path}: {e}")
    @staticmethod
    
    def create_project(parent_directory: str, project_name: str) -> "Project | None":
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
        directory = os.path.join(self.datasets_dir, name)
        if os.path.exists(directory):
            print(f"Dataset '{name}' already exists.")
            return None
        os.makedirs(directory, exist_ok=True)
        config_path = os.path.join(directory, "config.yaml")
        labels_path = os.path.join(directory, "labels.yaml")
        
        # A new dataset should have a minimal config. 
        # The 'metrics' dictionary should not exist yet.
        dconfig = {
            "name": name, 
            "behaviors": behaviors, 
            "whitelist": recordings_whitelist, 
            "model": None
            # The 'metrics' key is intentionally omitted.
        }

        lconfig = {"behaviors": behaviors, "labels": {b: [] for b in behaviors}}
        with open(config_path, "w") as file: yaml.dump(dconfig, file, allow_unicode=True)
        with open(labels_path, "w") as file: yaml.dump(lconfig, file, allow_unicode=True)
        ds = Dataset(directory)
        self.datasets[name] = ds
        return ds
    
    def delete_dataset(self, name: str) -> bool:
        """
        Deletes a dataset's folder, its corresponding model folder, and removes
        them from the in-memory project state.
        """
        if name not in self.datasets:
            print(f"Attempted to delete non-existent dataset: {name}")
            return False

        print(f"Deleting dataset '{name}' and its associated model...")
        
        dataset_path = self.datasets[name].path
        model_path = os.path.join(self.models_dir, name)

        try:
            # Delete the dataset folder
            if os.path.isdir(dataset_path):
                shutil.rmtree(dataset_path)
                print(f"  - Successfully removed dataset folder: {dataset_path}")

            # Delete the model folder if it exists
            if os.path.isdir(model_path):
                shutil.rmtree(model_path)
                print(f"  - Successfully removed model folder: {model_path}")

            # Remove from the in-memory dictionaries
            self.datasets.pop(name, None)
            self.models.pop(name, None)
            
            print(f"Deletion of '{name}' complete.")
            return True
        except Exception as e:
            print(f"An error occurred during deletion of '{name}': {e}")
            # As a safety measure, reload the project state from disk to ensure consistency
            self.reload()
            return False
    
    def convert_instances(self, project_root_path: str, insts: list, seq_len: int, behaviors: list, progress_callback=None) -> tuple:
        seqs, labels = [], []
        half_seqlen = seq_len // 2
        instances_by_video = {}
        for inst in insts:
            video_path = inst.get("video")
            if video_path: instances_by_video.setdefault(video_path, []).append(inst)
        total_videos = len(instances_by_video)
        for i, (relative_video_path, video_instances) in enumerate(instances_by_video.items()):
            if progress_callback: progress_callback((i + 1) / total_videos * 100)
            absolute_video_path = os.path.join(project_root_path, relative_video_path)
            cls_path = os.path.splitext(absolute_video_path)[0] + "_cls.h5"
            if not os.path.exists(cls_path):
                print(f"Warning: H5 file not found, skipping instances for {relative_video_path}")
                continue
            try:
                with h5py.File(cls_path, "r") as f: cls_arr = f["cls"][:]
            except Exception as e:
                print(f"Warning: Could not read H5 file {cls_path}: {e}")
                continue
            if cls_arr.ndim < 2 or cls_arr.shape[0] < seq_len: continue
            cls_centered = cls_arr - np.mean(cls_arr, axis=0)
            for inst in video_instances:
                start = int(inst.get("start", -1))
                end = int(inst.get("end", -1))
                if start == -1 or end == -1: continue
                
                # Iterates over every single labeled frame, from the
                # true start to the true end of the instance.
                valid_start = start
                valid_end = end

                # The loop includes the final frame by using "+ 1".
                for frame_idx in range(valid_start, valid_end + 1):
                    window_start = frame_idx - half_seqlen
                    window_end = frame_idx + half_seqlen + 1
                    
                    # Elegant safety checks prevent out-of-bounds errors at the absolute
                    # beginning or end of a video file, discarding the minimum data.
                    if window_start < 0: continue
                    if window_end > cls_centered.shape[0]: continue
                                        
                    window = cls_centered[window_start:window_end]
                    if window.shape[0] != seq_len: continue
                    
                    try:
                        label_index = behaviors.index(inst["label"].strip())
                        seqs.append(torch.from_numpy(window).float())
                        labels.append(torch.tensor(label_index).long())
                    except ValueError:
                        print(f"WARNING: The label '{inst['label']}' in video '{inst['video']}' is not in the master behavior list. This instance will be SKIPPED. Please check for typos in your labels.yaml file.")

        if not seqs: return [], []
        shuffled_pairs = list(zip(seqs, labels))
        random.shuffle(shuffled_pairs)
        seqs, labels = zip(*shuffled_pairs)
        return list(seqs), list(labels)
        
    def _load_dataset_common(self, name, split, seed):
        """
        Internal method to load and split datasets with heavy debugging.
        """
        rng = np.random.default_rng(seed)

        dataset_path = os.path.join(self.datasets_dir, name)
        if not os.path.isdir(dataset_path): raise FileNotFoundError(dataset_path)
        
        with open(os.path.join(dataset_path, "labels.yaml"), "r") as f: 
            label_config = yaml.load(f, Loader=yaml.FullLoader)
            
        behaviors = label_config.get("behaviors", [])
        if not behaviors: return None, None, None
        
        all_insts = [inst for b in behaviors for inst in label_config.get("labels", {}).get(b, [])]
        if not all_insts: return [], [], behaviors
        
        group_to_behaviors, group_to_instances = {}, {}
        for inst in all_insts:
            if 'video' in inst and inst['video']:

                # The unique key for a subject/group must be its full relative path
                # to distinguish between subjects with the same name in different sessions.
                group_key = os.path.dirname(inst['video']).replace('\\', '/')
                group_to_instances.setdefault(group_key, []).append(inst)
                group_to_behaviors.setdefault(group_key, set()).add(inst['label'])
        
        all_groups = sorted(list(group_to_instances.keys()))
        rng.shuffle(all_groups)
        
        test_groups, behaviors_needed_in_test = set(), set(behaviors)
   
        while behaviors_needed_in_test:
            group_found_in_pass = False
            for group in all_groups:
                if group in test_groups:
                    continue

                behaviors_in_group = group_to_behaviors.get(group, set())
                if behaviors_needed_in_test.intersection(behaviors_in_group):
                    test_groups.add(group)
                    behaviors_needed_in_test.difference_update(behaviors_in_group)
                    group_found_in_pass = True
                    break 
            
            if not group_found_in_pass:
                if behaviors_needed_in_test:
                    print(f"Warning: Could not find groups to cover all behaviors. Missing: {behaviors_needed_in_test}")
                break

        total_size = len(all_insts)
        if total_size > 0:
            current_test_size = sum(len(group_to_instances.get(g, [])) for g in test_groups)
            while current_test_size / total_size < split:
                group_added_in_pass = False
                for group in all_groups:
                    if group not in test_groups:
                        test_groups.add(group)
                        current_test_size += len(group_to_instances.get(group, []))
                        group_added_in_pass = True
                        break
                
                if not group_added_in_pass:
                    break
        
        train_groups = [g for g in all_groups if g not in test_groups]
        train_insts = [inst for group in train_groups for inst in group_to_instances.get(group, [])]
        test_insts = [inst for group in test_groups for inst in group_to_instances.get(group, [])]
        
        rng.shuffle(train_insts)
        rng.shuffle(test_insts)
        
        print(f"Stratified Group Split (Seed: {seed}): {len(train_insts)} train instances, {len(test_insts)} test instances.")
        print(f"  - Train groups: {len(train_groups)}, Test groups: {len(test_groups)}")
        
        if not train_insts and test_insts:
            print("  - [WARN] All labeled data belongs to a single group. Subject-level split is not possible. Falling back to a random 80/20 instance split.")
            all_insts = test_insts
            rng.shuffle(all_insts)
            split_idx = int(len(all_insts) * (1 - split))
            return all_insts[:split_idx], all_insts[split_idx:], behaviors        
        
        if not test_insts and train_insts:
            print("  - Warning: Stratified split resulted in an empty test set. Falling back to 80/20 instance split.")
            all_insts = train_insts
            rng.shuffle(all_insts)
            split_idx = int(len(all_insts) * (1 - split))
            return all_insts[:split_idx], all_insts[split_idx:], behaviors

        return train_insts, test_insts, behaviors
        
    def load_dataset(self, name: str, seed: int = 42, split: float = 0.2, seq_len: int = 15, progress_callback=None) -> tuple:
        train_insts, test_insts, behaviors = self._load_dataset_common(name, split, seed)
        if train_insts is None: return None, None, None, None
        
        def train_prog(p): progress_callback(p*0.5) if progress_callback else None
        def test_prog(p): progress_callback(50 + p*0.5) if progress_callback else None
        
        train_seqs, train_labels = self.convert_instances(self.path, train_insts, seq_len, behaviors, train_prog)
        test_seqs, test_labels = self.convert_instances(self.path, test_insts, seq_len, behaviors, test_prog)
        
        return BalancedDataset(train_seqs, train_labels, behaviors), StandardDataset(test_seqs, test_labels), train_insts, test_insts
        
    def load_dataset_for_weighted_loss(self, name, seed=42, split=0.2, seq_len=15, progress_callback=None) -> tuple:
        train_insts, test_insts, behaviors = self._load_dataset_common(name, split, seed)
        if train_insts is None: return None, None, None, None, None
        
        def train_prog(p): progress_callback(p*0.5) if progress_callback else None
        def test_prog(p): progress_callback(50 + p*0.5) if progress_callback else None
        
        train_seqs, train_labels = self.convert_instances(self.path, train_insts, seq_len, behaviors, train_prog)
        if not train_labels: return None, None, None, None, None
        
        class_counts = np.bincount([lbl.item() for lbl in train_labels], minlength=len(behaviors))
        weights = [sum(class_counts) / (len(behaviors) * c) if c > 0 else 0 for c in class_counts]
        test_seqs, test_labels = self.convert_instances(self.path, test_insts, seq_len, behaviors, test_prog)
        
        return StandardDataset(train_seqs, train_labels), StandardDataset(test_seqs, test_labels), weights, train_insts, test_insts

def collate_fn(batch):
    dcls, lbls = zip(*batch)
    return torch.stack(dcls), torch.stack(lbls)
    
def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
class PerformanceReport:
    def __init__(self, train_report: dict, train_cm: np.ndarray, val_report: dict, val_cm: np.ndarray):
        self.train_report = train_report
        self.train_cm = train_cm
        self.val_report = val_report
        self.val_cm = val_cm
        
def train_lstm_model(train_set, test_set, seq_len: int, behaviors: list, cancel_event: threading.Event, batch_size=512, lr=1e-4, epochs=10, device=None, class_weights=None, patience=3, progress_callback=None, optimization_target="weighted avg") -> tuple:
    """
    Trains the classifier head. This is the final, corrected version compatible
    with the production-ready ClassifierLSTMDeltas class.
    """
    from workthreads import log_message

    if len(train_set) == 0: return None, None, -1
    if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=(device.type == 'cuda'), drop_last=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0) if len(test_set) > 0 else None
    
    model = classifier_head.ClassifierLSTMDeltas(
        in_features=768,
        out_features=len(behaviors),
        seq_len=seq_len
    ).to(device)
    
    log_message(f"Successfully instantiated model architecture: {type(model).__name__}", "INFO")
    
    optimizer = optim.Adam([
        {'params': [p for name, p in model.named_parameters() if name != 'gate']},
        {'params': model.gate, 'weight_decay': 1e-3}
    ], lr=lr)
    loss_weights = torch.tensor(class_weights, dtype=torch.float).to(device) if class_weights else None
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    
    best_f1, best_model_state, best_epoch = -1.0, None, -1
    epoch_reports, epochs_no_improve = [], 0
    
    for e in range(epochs):
        if cancel_event.is_set():
            print(f"Cancellation detected within training loop at epoch {e+1}.")
            return None, epoch_reports, best_epoch
        
        if progress_callback:
            progress_callback(f"Training Epoch {e + 1}/{epochs}...")
        
        model.train()
        for i, (d, l) in enumerate(train_loader):
            d, l = d.to(device).float(), l.to(device)
            optimizer.zero_grad()
            
            final_logits, rawm = model(d)
            inv_loss = criterion(final_logits, l)
            
            rawm_centered = rawm - rawm.mean(dim=0)
            covm_loss = torch.tensor(0.0).to(device)
            if rawm_centered.ndim == 2 and rawm_centered.shape[0] > 1:
                covm = (rawm_centered.T @ rawm_centered) / (rawm_centered.shape[0] - 1)
                covm_loss = torch.sum(torch.pow(off_diagonal(covm), 2))
                
            loss = inv_loss + covm_loss
            loss.backward()
            optimizer.step()
            if i % 50 == 0: print(f"[Epoch {e+1}/{epochs} Batch {i}/{len(train_loader)}] Loss: {loss.item():.4f}")
            
        model.eval()
        train_actuals, train_predictions = [], []
        with torch.no_grad():
            for d, l in train_loader:
                logits, _ = model(d.to(device).float())
                train_actuals.extend(l.cpu().numpy())
                train_predictions.extend(logits.argmax(1).cpu().numpy())
        
        if not train_actuals:
            epochs_no_improve += 1 
            if epochs_no_improve >= patience: break
            continue

        train_report = classification_report(train_actuals, train_predictions, target_names=behaviors, output_dict=True, zero_division=0, labels=range(len(behaviors)))
        train_cm = confusion_matrix(train_actuals, train_predictions, labels=range(len(behaviors)))
        val_report, val_cm = {}, np.array([])
        
        if test_loader:
            val_actuals, val_predictions = [], []
            with torch.no_grad():
                for d, l in test_loader:
                    logits, _ = model(d.to(device).float())
                    val_actuals.extend(l.cpu().numpy())
                    val_predictions.extend(logits.argmax(1).cpu().numpy())
            if val_actuals:
                val_report = classification_report(val_actuals, val_predictions, target_names=behaviors, output_dict=True, zero_division=0, labels=range(len(behaviors)))
                val_cm = confusion_matrix(val_actuals, val_predictions, labels=range(len(behaviors)))
                
        epoch_reports.append(PerformanceReport(train_report, train_cm, val_report, val_cm))
        
        optimization_key = optimization_target
        current_val_f1 = val_report.get(optimization_key, {}).get("f1-score", -1.0)
        current_train_f1 = train_report.get(optimization_key, {}).get("f1-score", -1.0)
        
        if progress_callback:
            progress_callback(f"Epoch {e + 1} Val F1: {current_val_f1:.4f}")
        
        print(f"--- Epoch {e+1} | Train F1: {current_train_f1:.4f} | Val F1: {current_val_f1:.4f} ({optimization_key}) ---")
        
        if current_val_f1 > best_f1:
            best_f1, best_epoch, best_model_state = current_val_f1, e, model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            
        if test_loader and epochs_no_improve >= patience:
            log_message(f"Early stopping triggered at epoch {e + 1}.", "INFO")
            break                      
            
    if best_model_state is None and epochs > 0 and not test_loader:
        best_model_state = model.state_dict().copy()
        best_epoch = epochs - 1
             
    if best_model_state:
        final_model = classifier_head.ClassifierLSTMDeltas(
            in_features=768,
            out_features=len(behaviors),
            seq_len=seq_len
        )
        final_model.load_state_dict(best_model_state)
        return final_model.eval(), epoch_reports, best_epoch
        
    return None, None, -1