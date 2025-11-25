# Troubleshooting CBAS v3

This guide covers common issues encountered during installation, recording, training, and analysis.

---

## 1. Installation & Startup

### Error: "No matching distribution found for torch"
*   **Cause:** You are likely using an unsupported version of Python (e.g., Python 3.13, 3.10, or a 32-bit version).
*   **Fix:** CBAS requires a **64-bit** installation of **Python 3.11** or **Python 3.12**. Uninstall your current Python version and download the correct installer from python.org.

### App Hangs on Splash Screen / "Loading DINO encoder..."
*   **Cause:** The application is trying to download the DINOv2 model from Hugging Face, but authentication is failing or terms haven't been accepted.
*   **Fix:**
    1.  Ensure you have a Hugging Face account.
    2.  Accept the terms of use on the model's page (e.g., `facebook/dinov3...`).
    3.  Log in via the command line: `huggingface-cli login`.

### Error: "Port 8000 is already in use"
*   **Cause:** The previous CBAS session didn't close cleanly, leaving the Python backend running in the background ("Zombie process").
*   **Fix:**
    *   **Windows:** Open Task Manager, find `python.exe` or `ffmpeg.exe` processes, and click **End Task**.
    *   **macOS/Linux:** Open a terminal and run `pkill python` or `pkill ffmpeg`.

### Backend Crash on Startup
*   **Cause:** Often caused if `ffmpeg` is not found in your system's PATH.
*   **Fix:** Follow the detailed FFmpeg installation instructions in `Installation.md` to ensure the `bin` folder is added to your Windows Environment Variables.

### Console Error: "ElectronAPI" or "IPC" not found
*   **Cause:** The `preload.js` script failed to load, usually due to a file structure issue during installation.
*   **Fix:** Ensure you are running `npm start` from the root `CBAS` directory.

---

## 2. Recording & Cameras

### Camera Preview is Black / Loading Forever
*   **Cause:**
    *   Incorrect RTSP credentials (username/password).
    *   The computer is not configured to be on the correct subnet (e.g., `192.168.1.x`).
*   **Fix:** Verify the camera URL in VLC Media Player. Check your computer's static IP settings (see `Hardware_Setup.md`).

### Visual Artifacts (Green smearing, Grey blocks, "Melting")
*   **Cause:** Network packet loss. This happens if too many cameras are recording at high resolution/framerate simultaneously, saturating the network switch or the computer's network card.
*   **Fix:**
    *   Reduce the **Framerate** (e.g., 15 -> 10) in the camera's web settings.
    *   Reduce the **Bitrate** in the camera's web settings.
    *   Ensure you are using Gigabit (or 10Gb) switches and cables (Cat6+).

### Error: "FFmpeg not recognized"
*   **Cause:** Windows cannot find the FFmpeg executable.
*   **Fix:** Revisit `Installation.md` and double-check the System PATH steps. You must restart your computer or command prompt after changing the PATH.

---

## 3. The "Missing Files" / Pipeline Confusion
*Common issues regarding the "No new files found" error.*

### Error: "No new files found to process" (Inference)
*   **Cause:** The raw video (`.mp4`) exists, but the **Encoding** step hasn't finished yet. The inference engine cannot see videos until they are converted into `.h5` feature files.
*   **Fix:** Check the footer of the app. Is the **Encoding** progress bar moving? If not, **restart the app**. On startup, CBAS scans for missing `.h5` files and automatically queues them for encoding.

### Inference finishes instantly / "No files" (But files exist)
*   **Cause:** The **"Zombie CSV"** issue. If a previous inference run failed or crashed, it may have left behind empty (0-byte) or corrupt `_outputs.csv` files. CBAS sees these files and assumes the job is already done.
*   **Fix:** Manually open the recording folder and delete any `_outputs.csv` files for the specific model you are trying to run.

### "Review & Correct" shows a blank timeline
*   **Cause A:** The `.h5` file is missing (see above).
*   **Cause B:** The **Confidence Slider** is set too high. If your model is weak, it might only be 30% confident. If the slider is at 70%, nothing will show.
*   **Fix:** Drag the slider down to 0% to see all predictions.

---

## 4. Model Training & Performance

### Inference output is 1.0 (100%) for everything
*   **Cause:** The **"Single Class" Error**. You only defined one behavior (e.g., "Rearing") in your dataset. The math (Softmax) forces probabilities to sum to 100%, so if there is only one option, it is always 100%.
*   **Fix:** Create a new dataset with at least two behaviors: your target (e.g., `Rearing`) and a `Background` (or `Other`) class.

### Training stops after 3-5 Epochs ("Early Stopping")
*   **Cause:** Your dataset is too small. The model "memorized" the answers immediately and stopped learning.
*   **Fix:** Label more data. Aim for **50–100 instances** per behavior.

### Dataset metrics show "N/A"
*   **Cause:** The behavior is so rare that it didn't end up in the **Validation Set** or **Test Set** during the random split. If there are 0 examples to test on, the score is undefined.
*   **Fix:** Label more examples of that specific behavior to ensure it gets distributed across all data splits.

### Error: "CUDA Out of Memory"
*   **Cause:** The **Batch Size** is too high for your GPU's VRAM.
*   **Fix:** In the training settings, reduce the Batch Size (e.g., from 512 to 256 or 128).

### Model fails to load / Shape Mismatch
*   **Cause:**
    *   Trying to load a **v2 `JonesLabModel`** into v3 (incompatible architecture).
    *   Trying to load a model trained with a different hidden size (e.g., 128) without its accompanying `model_meta.json` file.
*   **Fix:** Train new models within CBAS v3.

---

## 5. Visualization & Video Management

### Actogram is Empty/Black
*   **Cause:** The **Threshold** is set too high (e.g., 90%) for a model that is only moderately confident (e.g., 70%).
*   **Fix:** Lower the Threshold slider in the Visualize tab.

### Imported videos look "squashed" or distorted
*   **Cause:** The **"Standardize"** checkbox was checked during import. This forces videos to `256x256` pixels (square) for AI compatibility.
*   **Note:** This is normal behavior. The AI learns from the distorted view perfectly fine.

### Timeline doesn't match the video
*   **Cause:** Variable framerate videos that were **not** standardized.
*   **Fix:** Always check "Standardize" when importing videos to ensure a locked 10 FPS.

---

## 6. General "How do I...?"

### How do I delete a dataset?
Go to the **Label/Train** page -> Click **Manage** on the dataset card -> Click **Delete this Dataset** (at the bottom).

### How do I recover if the app crashes mid-recording?
CBAS uses a file watcher. If the app crashes, the `ffmpeg` process usually keeps recording in the background.
*   **To recover:** Restart CBAS.
*   **If the recording stopped:** The logic is designed to split files into chunks. You will likely have valid video files up until the moment of the crash.

### Git Pull Fails ("Local changes would be overwritten")
*   **Cause:** You modified a system file (like `cbas_config.yaml.example` or `requirements.txt`) that an update is trying to change.
*   **Fix:**
    1.  Stash your changes: `git stash`
    2.  Pull the update: `git pull`
    3.  Restore your changes (optional): `git stash pop`