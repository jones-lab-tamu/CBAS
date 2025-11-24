"""
Core backend logic and data models for the CBAS application.
This version uses the proven v2-stable logic for recording.
"""
from __future__ import annotations

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
import json
from collections import defaultdict
import atexit

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
from backend.splits import RandomSplitProvider

CHUNK_SIZE = 512

# =========================================================================
# LAZY DATA LOADING
# =========================================================================

# This dictionary will be created as a fresh, empty dict in each DataLoader
# worker process. It is not shared between them. It caches open H5 file handles.
_worker_h5_handles = {}

def _cleanup_worker_handles():
    """Function to be called when a worker process exits to close H5 files."""
    for handle in _worker_h5_handles.values():
        try:
            handle.close()
        except Exception:
            pass # Ignore errors on close, the process is terminating anyway
    _worker_h5_handles.clear()

def cleanup_global_handles():
    """
    Explicitly closes all cached HDF5 handles in the main process.
    This must be called by the TrainingThread after a job completes.
    """
    global _worker_h5_handles
    if _worker_h5_handles:
        print(f"Cleaning up {len(_worker_h5_handles)} open HDF5 file handles...")
        for path, handle in list(_worker_h5_handles.items()):
            try:
                handle.close()
            except Exception:
                pass # Ignore errors on close (file might be already closed)
        _worker_h5_handles.clear()

def worker_init_fn(worker_id):
    """Initializes a DataLoader worker by registering the cleanup function."""
    atexit.register(_cleanup_worker_handles)

def _validate_lazy_vs_eager(project_root_path, instances_subset, seq_len, behaviors):
    """
    A debug function to compare the output of the old eager loading method
    with the new lazy loading method to ensure byte-level equivalence.
    """
    print(f"Validating on a subset of {len(instances_subset)} instances...")
    half_seqlen = seq_len // 2

    # --- 1. Eager Path (Old Logic, re-implemented for validation) ---
    eager_seqs = []
    eager_labels = []
    eager_instances_by_video = defaultdict(list)
    for inst in instances_subset:
        eager_instances_by_video[inst.get("video")].append(inst)

    for relative_video_path, video_instances in eager_instances_by_video.items():
        absolute_video_path = os.path.join(project_root_path, relative_video_path)
        cls_path = os.path.splitext(absolute_video_path)[0] + "_cls.h5"
        if not os.path.exists(cls_path): continue
        try:
            with h5py.File(cls_path, "r") as f: cls_arr = f["cls"][:]
        except Exception: continue
        if cls_arr.ndim < 2 or cls_arr.shape[0] < seq_len: continue
        
        for inst in video_instances:
            start, end = int(inst.get("start", -1)), int(inst.get("end", -1))
            if start == -1 or end == -1: continue
            try:
                label_index = behaviors.index(inst["label"].strip())
            except ValueError: continue

            for frame_idx in range(start, end + 1):
                window_start, window_end = frame_idx - half_seqlen, frame_idx + half_seqlen + 1
                if window_start < 0 or window_end > cls_arr.shape[0]: continue
                window = cls_arr[window_start:window_end]
                if window.shape[0] != seq_len: continue
                eager_seqs.append(torch.from_numpy(window).float())
                eager_labels.append(torch.tensor(label_index).long())

    # --- 2. Lazy Path (New Logic) ---
    lazy_manifest = []
    lazy_instances_by_video = defaultdict(list)
    for inst in instances_subset:
        lazy_instances_by_video[inst.get("video")].append(inst)
    
    for relative_video_path, video_instances in lazy_instances_by_video.items():
        absolute_video_path = os.path.join(project_root_path, relative_video_path)
        cls_path = os.path.splitext(absolute_video_path)[0] + "_cls.h5"
        if not os.path.exists(cls_path): continue
        try:
            with h5py.File(cls_path, "r") as f: num_frames = f["cls"].shape[0]
        except Exception: continue
        if num_frames < seq_len: continue
        
        for inst in video_instances:
            start, end = int(inst.get("start", -1)), int(inst.get("end", -1))
            if start == -1 or end == -1: continue
            try:
                label_index = behaviors.index(inst["label"].strip())
            except ValueError: continue
            for frame_idx in range(start, end + 1):
                if (frame_idx - half_seqlen >= 0) and (frame_idx + half_seqlen < num_frames):
                    lazy_manifest.append((cls_path, frame_idx, label_index))

    # --- 3. Comparison ---
    if len(eager_seqs) != len(lazy_manifest):
        raise ValueError(f"Mismatch in number of examples: Eager={len(eager_seqs)}, Lazy={len(lazy_manifest)}")

    print(f"Generated {len(eager_seqs)} windows for comparison. Checking for equivalence...")

    h5_handles = {}
    try:
        for i in range(len(lazy_manifest)):
            eager_tensor, eager_label = eager_seqs[i], eager_labels[i]

            h5_path, center_frame, label_index = lazy_manifest[i]
            if h5_path not in h5_handles:
                h5_handles[h5_path] = h5py.File(h5_path, 'r')
            h5_file = h5_handles[h5_path]
            
            window_start = center_frame - half_seqlen
            window_end = center_frame + half_seqlen + 1
            window_data = h5_file['cls'][window_start:window_end]
            lazy_tensor = torch.from_numpy(window_data).float()
            lazy_label = torch.tensor(label_index).long()

            if not torch.equal(eager_tensor, lazy_tensor) or not torch.equal(eager_label, lazy_label):
                raise ValueError(f"Data mismatch at index {i} for window centered at frame {center_frame} in {h5_path}")
    finally:
        for handle in h5_handles.values():
            handle.close()

    print("Validation successful: All generated tensors are byte-for-byte identical.")


class LazyStandardDataset(torch.utils.data.Dataset):
    """
    A lazy-loading PyTorch Dataset. It stores a manifest of "pointers" to
    data windows and only loads a window from disk when requested by __getitem__.
    """
    def __init__(self, manifest: list, seq_len: int):
        self.manifest = manifest
        self.seq_len = seq_len
        self.half_seqlen = seq_len // 2

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        # 1. Get the pointer for this specific training example
        h5_path, center_frame, label_index = self.manifest[idx]

        # 2. Use the process-local dictionary to get/cache the H5 file handle
        if h5_path not in _worker_h5_handles:
            try:
                # Open the file in read-only mode and cache the handle
                _worker_h5_handles[h5_path] = h5py.File(h5_path, 'r')
            except Exception as e:
                # If a file is corrupt or missing, this prevents a crash.
                # We return a dummy tensor that can be filtered out later.
                print(f"WORKER-ERROR: Could not open H5 file {h5_path}. {e}")
                return torch.zeros(self.seq_len, 768), torch.tensor(-1).long()
        
        h5_file = _worker_h5_handles[h5_path]

        # 3. Read ONLY the required slice of data from disk
        window_start = center_frame - self.half_seqlen
        window_end = center_frame + self.half_seqlen + 1
        
        try:
            window_data = h5_file['cls'][window_start:window_end]
            
            # 4. Convert to tensor and return
            window_tensor = torch.from_numpy(window_data).float()
            
            # Defensive check for corrupted data reads
            if window_tensor.shape[0] != self.seq_len:
                 return torch.zeros(self.seq_len, 768), torch.tensor(-1).long()

            return window_tensor, torch.tensor(label_index).long()
        except Exception as e:
            print(f"WORKER-ERROR: Could not read slice from {h5_path}. {e}")
            return torch.zeros(self.seq_len, 768), torch.tensor(-1).long()


class LazyBalancedDataset(torch.utils.data.Dataset):
    """
    A lazy-loading version of the BalancedDataset. It buckets the *indices*
    of the manifest, not the tensors themselves, and uses LazyStandardDataset's
    __getitem__ logic to load data on the fly.
    """
    def __init__(self, manifest: list, seq_len: int, behaviors: list):
        self.manifest = manifest
        self.seq_len = seq_len
        self.behaviors = behaviors
        self.num_behaviors = len(behaviors)
        self.half_seqlen = seq_len // 2

        # Bucket the indices of the manifest, not the data itself
        self.buckets = {b: [] for b in self.behaviors}
        for i, (_, _, label_index) in enumerate(manifest):
            if 0 <= label_index < self.num_behaviors:
                behavior_name = self.behaviors[label_index]
                self.buckets[behavior_name].append(i)
        
        self.available_behaviors = [b for b in self.behaviors if self.buckets[b]]
        self.num_available_behaviors = len(self.available_behaviors)

        self.total_sequences = len(manifest)
        self.counter = 0

    def __len__(self):
        if self.num_available_behaviors == 0: return 0
        # This logic ensures the length is a multiple of the number of available classes,
        # which helps the sampler see each class roughly equally.
        return self.total_sequences + (self.num_available_behaviors - self.total_sequences % self.num_available_behaviors) % self.num_available_behaviors

    def __getitem__(self, idx: int):
        if self.num_available_behaviors == 0:
            raise IndexError("No behaviors with samples available in this dataset split.")

        # Oversampling logic: pick a behavior class to sample from
        b_idx_in_available = self.counter % self.num_available_behaviors
        b_name = self.available_behaviors[b_idx_in_available]
        self.counter += 1
        
        # Get the list of manifest indices for the chosen behavior
        indices_for_behavior = self.buckets[b_name]
        
        # Pick a random sample from that list
        manifest_idx = indices_for_behavior[idx % len(indices_for_behavior)]
        
        # --- The rest is identical to LazyStandardDataset.__getitem__ ---
        h5_path, center_frame, label_index = self.manifest[manifest_idx]

        if h5_path not in _worker_h5_handles:
            try:
                _worker_h5_handles[h5_path] = h5py.File(h5_path, 'r')
            except Exception as e:
                print(f"WORKER-ERROR: Could not open H5 file {h5_path}. {e}")
                return torch.zeros(self.seq_len, 768), torch.tensor(-1).long()
        
        h5_file = _worker_h5_handles[h5_path]

        window_start = center_frame - self.half_seqlen
        window_end = center_frame + self.half_seqlen + 1
        
        try:
            window_data = h5_file['cls'][window_start:window_end]
            window_tensor = torch.from_numpy(window_data).float()
            if window_tensor.shape[0] != self.seq_len:
                 return torch.zeros(self.seq_len, 768), torch.tensor(-1).long()
            return window_tensor, torch.tensor(label_index).long()
        except Exception as e:
            print(f"WORKER-ERROR: Could not read slice from {h5_path}. {e}")
            return torch.zeros(self.seq_len, 768), torch.tensor(-1).long()

# =========================================================================

def _reflect_indices(idx: np.ndarray, N: int) -> np.ndarray:
    """Helper to safely reflect indices for padding."""
    if N == 1:
        return np.zeros_like(idx)
    # fold into [0, 2N)
    m = np.mod(idx, 2 * N)
    m = np.where(m < 0, m + 2 * N, m)
    # reflect the second half back to [0, N-1]
    return np.where(m >= N, 2 * N - m - 1, m)

### DATA LOADING ###

def create_datasets_from_splits(project, dataset_name: str, train_subjects: list, val_subjects: list, test_subjects: list, seq_len: int):
    """
    Creates PyTorch datasets from pre-defined lists of subject IDs.
    This is the core data handling function used by both GUI and headless runs.
    (REFACTORED FOR LAZY LOADING AND VALIDATION)
    """
    dataset = project.datasets.get(dataset_name)
    if not dataset:
        return None, None, None, [], [], [], []

    all_instances = [inst for b_labels in dataset.labels.get("labels", {}).values() for inst in b_labels]
    behaviors = dataset.config.get("behaviors", [])

    def get_insts_for_subjects(subjects):
        subject_set = set(subjects)
        return [inst for inst in all_instances if os.path.dirname(inst['video']) in subject_set]

    train_insts = get_insts_for_subjects(train_subjects)
    val_insts = get_insts_for_subjects(val_subjects)
    test_insts = get_insts_for_subjects(test_subjects)
    
    # --- START OF PHASE 3 VALIDATION LOGIC ---
    if os.environ.get('CBAS_VALIDATE_LAZY_LOADER') == '1':
        print("--- RUNNING LAZY LOADER VALIDATION ---")
        try:
            # Pass a small subset of train_insts to the validation function
            _validate_lazy_vs_eager(
                project.path,
                train_insts[:100], # Limit to first 100 instances for speed
                seq_len,
                behaviors
            )
            print("--- LAZY LOADER VALIDATION PASSED ---")
        except Exception as e:
            print(f"--- FATAL: LAZY LOADER VALIDATION FAILED: {e} ---")
            traceback.print_exc()
            raise e # Re-raise to stop the training process
    # --- END OF PHASE 3 VALIDATION LOGIC ---

    # Generate the lightweight manifests (lists of pointers)
    train_manifest = project.convert_instances(project.path, train_insts, seq_len, behaviors)
    val_manifest = project.convert_instances(project.path, val_insts, seq_len, behaviors)
    test_manifest = project.convert_instances(project.path, test_insts, seq_len, behaviors)

    # Instantiate the new lazy dataset classes
    train_ds = LazyBalancedDataset(train_manifest, seq_len, behaviors) if train_manifest else None
    val_ds = LazyStandardDataset(val_manifest, seq_len) if val_manifest else None
    test_ds = LazyStandardDataset(test_manifest, seq_len) if test_manifest else None

    return train_ds, val_ds, test_ds, train_insts, val_insts, test_insts, behaviors

def compute_class_weights_from_instances(
    train_insts: list,
    behaviors: list,
    epsilon: float = 1e-6
):
    """
    Compute per-class weights based on training instance counts.
    """
    counts = {b: 0 for b in behaviors}
    for inst in train_insts:
        lbl = inst.get("label")
        if lbl in counts:
            counts[lbl] += 1

    raw_weights = []
    for b in behaviors:
        c = counts[b]
        if c == 0:
            raw_weights.append(1.0 / epsilon)
        else:
            raw_weights.append(1.0 / c)

    weights = np.array(raw_weights, dtype=np.float32)
    weights = weights / weights.sum() * len(behaviors)

    return torch.tensor(weights, dtype=torch.float32)
    
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
    tmp_file_path = out_file_path + ".tmp"

    try:
        with h5py.File(tmp_file_path, "w") as h5f:
            if gui_state.proj:
                h5f.attrs['encoder_model_identifier'] = gui_state.proj.encoder_model_identifier
                h5f.attrs['schema_version'] = "1.0"
            
            dset = h5f.create_dataset(
                "cls", shape=(0, 768), maxshape=(None, 768),
                dtype='f2', chunks=(8192, 768)
            )

            for i in range(0, video_len, CHUNK_SIZE):
                end_index = min(i + CHUNK_SIZE, video_len)
                frames_np = reader.get_batch(range(i, end_index)).asnumpy()

                if progress_callback:
                    percent_complete = (end_index / video_len) * 100
                    progress_callback(percent_complete)

                frames_tensor = torch.from_numpy(frames_np[:, :, :, 1] / 255.0).float()

                device_type = "cuda" if encoder.device.type == "cuda" else "cpu"
                with torch.no_grad(), torch.autocast(device_type=device_type, enabled=(device_type == "cuda")):
                    embeddings_batch = encoder(frames_tensor.unsqueeze(1).to(encoder.device))
                    embeddings_out = embeddings_batch.squeeze(1).cpu().numpy()
                    dset.resize(dset.shape[0] + len(embeddings_out), axis=0)
                    dset[-len(embeddings_out):] = embeddings_out

                h5f.flush()

        os.replace(tmp_file_path, out_file_path)

        print(f"Successfully encoded {os.path.basename(path)} to {os.path.basename(out_file_path)}")
        return out_file_path

    except Exception as e:
        print(f"ERROR during encoding for {path}: {e}")
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
        return None

def infer_file(
    file_path: str,
    model: nn.Module,
    dataset_name: str,
    behaviors: list[str],
    seq_len: int,
    device=None,
    temperature=1.0,
) -> str | None:
    """
    Runs inference on a single HDF5 file and saves the results to a CSV.
    Uses buffered reading (chunks) to prevent OOM on large videos.
    """
    output_file = file_path.replace("_cls.h5", f"_{dataset_name}_outputs.csv")
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure model is on correct device
    param = next(model.parameters(), None)
    if param is not None and param.device != device:
        model = model.to(device)

    half_seqlen = seq_len // 2
    INFERENCE_CHUNK_SIZE = 20000  # Process ~20k frames at a time to save RAM
    
    try:
        with h5py.File(file_path, "r") as f:
            dset = f["cls"]
            total_frames = dset.shape[0]
            
            if total_frames == 0:
                print(f"Warning: HDF5 file {file_path} is empty.")
                return None

            # We collect results in a list. A list of 1M float arrays is RAM-efficient (~30MB for 24h).
            # The heavy feature vectors (GBs) are discarded after each chunk.
            all_probs = []

            for start_idx in range(0, total_frames, INFERENCE_CHUNK_SIZE):
                end_idx = min(start_idx + INFERENCE_CHUNK_SIZE, total_frames)
                
                # Determine read bounds with context
                # To predict frames [start_idx ... end_idx], we need features 
                # from (start_idx - half) to (end_idx + half).
                read_start = max(0, start_idx - half_seqlen)
                read_end = min(total_frames, end_idx + half_seqlen)
                
                # Load ONLY the necessary chunk into RAM
                feature_chunk = dset[read_start:read_end] 
                chunk_tensor = torch.from_numpy(feature_chunk).float()
                
                # --- Edge Padding Logic ---
                # If we are at the very start of the video and couldn't read past context:
                if start_idx < half_seqlen:
                    pad_qty = half_seqlen - start_idx
                    if pad_qty > 0:
                        # Replicate the first frame's features 'pad_qty' times
                        front_pad = chunk_tensor[0:1].repeat(pad_qty, 1)
                        chunk_tensor = torch.cat([front_pad, chunk_tensor], dim=0)

                # If we are at the very end of the video and couldn't read future context:
                if end_idx > total_frames - half_seqlen:
                    pad_qty = half_seqlen - (total_frames - end_idx)
                    if pad_qty > 0:
                        # Replicate the last frame's features 'pad_qty' times
                        end_pad = chunk_tensor[-1:].repeat(pad_qty, 1)
                        chunk_tensor = torch.cat([chunk_tensor, end_pad], dim=0)
                
                # --- Inference Loop ---
                # chunk_tensor is now padded to allow sliding exactly 'num_targets' windows
                num_targets = end_idx - start_idx
                batch_buffer = []
                chunk_predictions = []
                
                for i in range(num_targets):
                    # Slice window of size seq_len
                    window = chunk_tensor[i : i + seq_len]
                    batch_buffer.append(window)
                    
                    # Run batch when full or at end of chunk
                    if len(batch_buffer) >= 512 or i == num_targets - 1:
                        if not batch_buffer: continue
                        
                        batch_stack = torch.stack(batch_buffer).to(device)
                        with torch.no_grad():
                            logits, _ = model(batch_stack)
                            scaled_logits = logits / max(1e-3, temperature)
                            probs = torch.softmax(scaled_logits, dim=1).cpu().numpy()
                            chunk_predictions.extend(probs)
                        
                        batch_buffer = []
                
                all_probs.extend(chunk_predictions)
                
                # Explicitly free memory
                del chunk_tensor
                del feature_chunk

        # Save to CSV
        if not all_probs:
            return None
            
        # Check for length mismatch (sanity check)
        if len(all_probs) != total_frames:
            print(f"Warning: Prediction count ({len(all_probs)}) != Frame count ({total_frames}).")
            
        pd.DataFrame(np.array(all_probs), columns=behaviors).to_csv(output_file, index=False)
        return output_file

    except Exception as e:
        print(f"Error during buffered inference on {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

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
        pattern = np.repeat([int(b) for b in light_cycle_booleans], int(60 // bin_size_minutes))
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

# =========================================================================
# CORE DATA MODEL CLASSES (RESTORED AND REORDERED)
# =========================================================================

class DinoEncoder(nn.Module):
    def __init__(self, model_identifier: str, device="cuda"):
        super().__init__()
        self.device = torch.device(device)
        
        print(f"Loading DINO encoder model: {model_identifier}")
        try:
            self.model = transformers.AutoModel.from_pretrained(model_identifier).to(self.device)
        except Exception as e:
            from workthreads import log_message
            log_message("--- MODEL LOADING FAILED ---", "ERROR")
            log_message(f"Could not load the encoder model: '{model_identifier}'.", "ERROR")
            log_message("If you are trying to use a new or gated model (like DINOv3), please ensure you have:", "ERROR")
            log_message("1. Logged into the Hugging Face Hub ('huggingface-cli login').", "ERROR")
            log_message("2. Agreed to the model's terms of use on its Hugging Face page.", "ERROR")
            log_message("3. Installed the latest version of 'transformers' from source if required.", "ERROR")
            log_message(f"Original error: {e}", "ERROR")
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
            
            if base_name.endswith("_outputs.csv"):
                name_part = base_name[:-12]
                
                matched_video_base = None
                for vf in self.video_files:
                    vf_base = os.path.splitext(os.path.basename(vf))[0]
                    if name_part.startswith(vf_base):
                        matched_video_base = vf_base
                        break
                
                if matched_video_base:
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

        if "/profile1" in self.rtsp_url:
            self.profile0_url = self.rtsp_url.replace("/profile1", "/profile0")
        else:
            self.profile0_url = self.rtsp_url

        if write_to_disk: self.write_settings_to_config()

    def write_settings_to_config(self):
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
            '-stream_loop', '-1',
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
            self.project.active_recordings[self.name] = (process, time.time(), session_name)
            return True
        except Exception as e:
            print(f"Failed to start ffmpeg for {self.name}: {e}")
            return False
            
    def stop_recording(self) -> bool:
        if self.name in self.project.active_recordings:
            process, _, _ = self.project.active_recordings.pop(self.name)
            try:
                if process.stdin:
                    process.stdin.write(b'q')
                    process.stdin.flush()
                    process.stdin.close()
                process.wait(timeout=5)
            except Exception as e:
                print(f"Error while stopping process for {self.name}: {e}. Killing process.")
                process.kill()

            try:
                if self.project.current_session_name:
                    camera_folder = os.path.join(self.project.recordings_dir, self.project.current_session_name, self.name)
                    if os.path.isdir(camera_folder):
                        video_files = [os.path.join(camera_folder, f) for f in os.listdir(camera_folder) if f.endswith('.mp4')]
                        if video_files:
                            latest_file = max(video_files, key=os.path.getmtime)
                            with gui_state.encode_lock:
                                if latest_file not in gui_state.encode_tasks:
                                    gui_state.encode_tasks.append(latest_file)
                                    from workthreads import log_message
                                    log_message(f"Queued final segment on stop: '{os.path.basename(latest_file)}'", "INFO")
            except Exception as e:
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
        
        all_instances = [inst for b_labels in self.labels.get("labels", {}).values() for inst in b_labels]
        if not all_instances:
            for behavior_name in self.config.get("behaviors", []):
                self.update_metric(behavior_name, "Train Inst (Frames)", "0 (0)")
                self.update_metric(behavior_name, "Test Inst (Frames)", "0 (0)")
            return

        all_subjects = list(set(os.path.dirname(inst['video']) for inst in all_instances))
        behaviors = self.config.get("behaviors", [])

        provider = RandomSplitProvider(seed=42, split_ratios=(0.8, 0.0, 0.2), stratify=False)
        train_subjects, _, test_subjects = provider.get_split(0, all_subjects, all_instances, behaviors)

        train_subject_set = set(train_subjects)
        test_subject_set = set(test_subjects)
        train_insts = [inst for inst in all_instances if os.path.dirname(inst['video']) in train_subject_set]
        test_insts = [inst for inst in all_instances if os.path.dirname(inst['video']) in test_subject_set]

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
            
            self.update_metric(behavior_name, "Train Inst (Frames)", f"{train_n_inst} ({int(train_n_frame)})")
            self.update_metric(behavior_name, "Test Inst (Frames)", f"{test_n_inst} ({int(test_n_frame)})")
            
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
        
        if preloaded_df is not None:
            if self.behavior in preloaded_df.columns:
                probs = preloaded_df[self.behavior].to_numpy()
                is_max = (preloaded_df[preloaded_df.columns.drop(self.behavior)].max(axis=1) < probs).to_numpy()
                activity_per_frame.extend((probs * is_max >= self.threshold).astype(float).tolist())
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
             return
            
        if not activity_per_frame: return
        
        self.binned_activity = [np.sum(activity_per_frame[i:i + self.binsize_frames]) for i in range(0, len(activity_per_frame), self.binsize_frames)]
        if not self.binned_activity: return
        fig = _create_matplotlib_actogram(self.binned_activity, [c=="1" for c in self.lightcycle_str], 24.0, self.bin_size_minutes, f"{model} - {behavior}", self.start_hour_on_plot, self.plot_acrophase, base_color)
        if fig:
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight', facecolor='#343a40')
            buf.seek(0)
            self.blob = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)

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
        
        self.project_config = {}
        config_path = os.path.join(self.path, "cbas_config.yaml")
        if os.path.exists(config_path):
            print(f"Found project-specific config file: {config_path}")
            try:
                with open(config_path, 'r') as f:
                    self.project_config = yaml.safe_load(f)
            except Exception as e:
                print(f"WARNING: Could not read or parse cbas_config.yaml. Using defaults. Error: {e}")
        
        self.encoder_model_identifier = self.project_config.get(
            "encoder_model_identifier", 
            "facebook/dinov2-with-registers-base"
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
        
        dconfig = {
            "name": name, 
            "behaviors": behaviors, 
            "whitelist": recordings_whitelist, 
            "model": None
        }

        lconfig = {"behaviors": behaviors, "labels": {b: [] for b in behaviors}}
        with open(config_path, "w") as file: yaml.dump(dconfig, file, allow_unicode=True)
        with open(labels_path, "w") as file: yaml.dump(lconfig, file, allow_unicode=True)
        ds = Dataset(directory)
        self.datasets[name] = ds
        return ds
    
    def delete_dataset(self, name: str) -> bool:
        if name not in self.datasets:
            print(f"Attempted to delete non-existent dataset: {name}")
            return False

        print(f"Deleting dataset '{name}' and its associated model...")
        
        dataset_path = self.datasets[name].path
        model_paths = [
            os.path.join(self.models_dir, name),
            os.path.join(self.models_dir, f"{name}_model"),
        ]

        try:
            if os.path.isdir(dataset_path):
                shutil.rmtree(dataset_path)
                print(f"  - Successfully removed dataset folder: {dataset_path}")

            for mp in model_paths:
                if os.path.isdir(mp):
                    shutil.rmtree(mp)
                    print(f"  - Successfully removed model folder: {mp}")

            self.datasets.pop(name, None)
            self.models.pop(name, None)
            
            print(f"Deletion of '{name}' complete.")
            return True
        except Exception as e:
            print(f"An error occurred during deletion of '{name}': {e}")
            self.reload()
            return False
    
    def convert_instances(self, project_root_path: str, insts: list, seq_len: int, behaviors: list, progress_callback=None) -> list:
        """
        Generates a memory-efficient "manifest" of training examples.
        (REFACTORED FOR LAZY LOADING)
        """
        manifest = []
        half_seqlen = seq_len // 2
        
        instances_by_video = defaultdict(list)
        for inst in insts:
            instances_by_video[inst.get("video")].append(inst)

        total_videos = len(instances_by_video)
        for i, (relative_video_path, video_instances) in enumerate(instances_by_video.items()):
            if progress_callback: progress_callback((i + 1) / total_videos * 100)
            
            if not relative_video_path: continue

            absolute_video_path = os.path.join(project_root_path, relative_video_path)
            cls_path = os.path.splitext(absolute_video_path)[0] + "_cls.h5"
            if not os.path.exists(cls_path):
                print(f"Warning: H5 file not found, skipping instances for {relative_video_path}")
                continue
            
            try:
                with h5py.File(cls_path, "r") as f:
                    num_frames = f["cls"].shape[0]
            except Exception as e:
                print(f"Warning: Could not read H5 file {cls_path}: {e}")
                continue

            if num_frames < seq_len: continue
            
            for inst in video_instances:
                start = int(inst.get("start", -1))
                end = int(inst.get("end", -1))
                if start == -1 or end == -1: continue
                
                try:
                    label_index = behaviors.index(inst["label"].strip())
                except ValueError:
                    print(f"WARNING: The label '{inst['label']}' in video '{inst['video']}' is not in the master behavior list. This instance will be SKIPPED.")
                    continue

                for frame_idx in range(start, end + 1):
                    if (frame_idx - half_seqlen >= 0) and (frame_idx + half_seqlen < num_frames):
                        manifest.append((cls_path, frame_idx, label_index))

        return manifest


def evaluate_on_split(model, dataset, behaviors, device=None):
    """
    Performs a one-time evaluation of a trained model on a given dataset split.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=False,
                                         num_workers=0, pin_memory=(device.type=='cuda'),
                                         collate_fn=collate_fn, worker_init_fn=worker_init_fn)
    y_true, y_pred = [], []
    model.to(device).eval()
    
    with torch.no_grad():
        for x, y in loader:
            valid = (y != -1)
            if not valid.any(): continue
            
            logits, _ = model(x[valid].to(device))
            y_true.extend(y[valid].cpu().numpy())
            y_pred.extend(logits.argmax(1).cpu().numpy())

    if not y_true:
        return {"report": {}, "cm": np.array([])}

    rep = classification_report(y_true, y_pred, target_names=behaviors,
                                output_dict=True, zero_division=0,
                                labels=range(len(behaviors)))
    cm  = confusion_matrix(y_true, y_pred, labels=range(len(behaviors)))
    return {"report": rep, "cm": cm}

def collate_fn(batch):
    # Filter out samples that failed to load
    batch = [b for b in batch if b[1] != -1]
    if not batch:
        return torch.tensor([]), torch.tensor([])
    
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
        
def train_lstm_model(train_set, test_set, seq_len: int, behaviors: list, cancel_event: threading.Event, 
                     batch_size=512, lr=1e-4, epochs=10, device=None, class_weights=None, patience=3, 
                     progress_callback=None, optimization_target="weighted avg",
                     weight_decay=0.0, label_smoothing=0.0, lstm_hidden_size=64, lstm_layers=1
                     ) -> tuple:    
                     
    from workthreads import log_message

    if len(train_set) == 0: return None, None, -1
    if device is None: device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0, pin_memory=(device.type == 'cuda'), drop_last=False, worker_init_fn=worker_init_fn)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0, worker_init_fn=worker_init_fn) if test_set and len(test_set) > 0 else None
    
    model = classifier_head.ClassifierLSTMDeltas(
        in_features=768,
        out_features=len(behaviors),
        seq_len=seq_len,
        lstm_hidden_size=lstm_hidden_size,
        lstm_layers=lstm_layers
    ).to(device)
    
    log_message(f"Successfully instantiated model architecture: {type(model).__name__}", "INFO")
    
    log_message("--- Training Trial Hyperparameters ---", "INFO")
    log_message(f"  Learning Rate: {lr}", "INFO")
    log_message(f"  Weight Decay: {weight_decay}", "INFO")
    log_message(f"  Label Smoothing: {label_smoothing}", "INFO")
    log_message(f"  LSTM Hidden Size: {lstm_hidden_size}", "INFO")
    log_message(f"  LSTM Layers: {lstm_layers}", "INFO")
    log_message("------------------------------------", "INFO")
    
    optimizer = optim.Adam([
        {'params': [p for name, p in model.named_parameters() if name != 'gate']},
        {'params': model.gate, 'weight_decay': 1e-3}
    ], lr=lr, weight_decay=weight_decay)
    
    loss_weights = torch.tensor(class_weights, dtype=torch.float).to(device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=label_smoothing)
    
    best_f1, best_model_state, best_epoch = -1.0, None, -1
    epoch_reports, epochs_no_improve = [], 0
    
    for e in range(epochs):
        if cancel_event.is_set():
            return None, epoch_reports, best_epoch
        
        if progress_callback:
            progress_callback(f"Training Epoch {e + 1}/{epochs}...")
        
        model.train()
        for i, (d, l) in enumerate(train_loader):
        
            # Check for cancel every batch for responsive stopping
            if cancel_event.is_set():
                break
                
            if d.numel() == 0: continue
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
                if d.numel() == 0: continue
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
                
                # Check for cancel in validation
                    if cancel_event.is_set():
                        break
                
                    if d.numel() == 0: continue
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
            val_f1_str = f"{current_val_f1:.4f}" if test_loader else "N/A"
            progress_callback(f"Epoch {e + 1} Val F1: {val_f1_str}")
        
        val_f1_print_str = f"{current_val_f1:.4f}" if test_loader else "N/A"
        print(f"--- Epoch {e+1} | Train F1: {current_train_f1:.4f} | Val F1: {val_f1_print_str} ({optimization_key}) ---")
        
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
            seq_len=seq_len,
            lstm_hidden_size=lstm_hidden_size, 
            lstm_layers=lstm_layers            
        )
        final_model.load_state_dict(best_model_state)
        return final_model.eval(), epoch_reports, best_epoch
        
    return None, None, -1