# check_h5_length.py
import h5py
import os

# --- CONFIGURE THIS ---
PROJECT_ROOT = r"C:\Users\Jones-Lab\Documents\CBAS_Projects\cbas"
VIDEO_FILE_REL_PATH = r"recordings/prerecorded/OVX3/OVX3_00100.mp4"


# --- END CONFIGURATION ---

base, _ = os.path.splitext(VIDEO_FILE_REL_PATH)
h5_file_rel_path = base + "_cls.h5"
h5_file_abs_path = os.path.join(PROJECT_ROOT, h5_file_rel_path)

print(f"Checking file: {h5_file_abs_path}")

try:
    with h5py.File(h5_file_abs_path, 'r') as f:
        nframes = f['cls'].shape[0]
        print(f"SUCCESS: The file contains exactly {nframes} frames (indices 0 to {nframes - 1}).")
except Exception as e:
    print(f"ERROR: Could not read the file. {e}")