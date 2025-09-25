<p align="center">
    <img src="./frontend/assets/cbas_logo.svg" alt="CBAS Logo" style="width: 500px; height: auto;">
</p>
<details align="center">
  <summary>What's with the old fish logo in CBASv2?</summary>
  <br>
  <p>The original CBAS logo featured fish, which confused some of our rodent-focused users! The name CBAS is pronounced like "sea bass," which was the source of the original pun.</p>
  <img src="./frontend/assets/cbas_mouseover.svg" alt="CBAS Easter Egg Logo" style="width: 500px; height: auto;">
</details>

# CBAS v3 (BETA) - *Recommended for most users*

> [!NOTE]
> **CBAS v3 is the actively developed version and is recommended for new projects and large videos.**
> It introduces a streamed/chunked encoder (prevents multi-GB RAM spikes), a more robust backend,
> and a simpler labeling/training workflow. Please report issues! v3 is under active development.

> [!IMPORTANT]
> **Need to reproduce a published result exactly?** Use the **`v2-stable`** branch that matches the paper.
> * [**Browse `v2-stable`**](https://github.com/jones-lab-tamu/CBAS/tree/v2-stable)
> * [**Download `.zip`**](https://github.com/jones-lab-tamu/CBAS/archive/refs/heads/v2-stable.zip)

CBAS (Circadian Behavioral Analysis Suite) is a full-featured, open-source application for phenotyping complex animal behaviors. It automates behavior classification from video and provides a streamlined interface for labeling, training, visualization, and analysis.

*Originally created by Logan Perry and now maintained by the Jones Lab at Texas A&M University.*

## Key Features at a Glance (v3)

*   **Standalone Desktop App:** Robust, cross-platform (Windows/macOS/Linux) Electron app. No browser required.
*   **Real-time Video Acquisition:** Record and process streams from any number of RTSP cameras.
*   **High-Performance AI:** Supports state-of-the-art **DINOv3** Vision Transformer backbones (gated via Hugging Face).
*   **Active Learning Workflow:** Pre-label with an existing model, then rapidly “Review & Correct” using the interactive timeline.
*   **Confidence-Based Filtering:** Jump directly to uncertain segments to spend time where it matters most.
*   **Automated Model Training:** Create custom classifiers with balanced/weighted options and detailed performance reports.
*   **Rich Visualization:** Multi-plot actograms, side-by-side behavior comparisons, adjustable binning/light cycles, optional acrophase.
*   **Streamed Encoding (No OOM):** Videos are processed in **chunks**, not loaded entirely into RAM-fixes v2’s large-video memory failures.

## What’s New in CBAS v3?

*   **Standalone Desktop Application:** Cross-platform Electron app; runs fully offline once models are cached.
*   **Supercharged Labeling:** Active-learning with confidence-guided review and fast boundary correction (keyboard-first workflow).
*   **Enhanced Visualization:** Tiled, side-by-side actograms for direct behavior comparison.
*   **Modern, Stable Backend:** Dedicated worker threads keep the UI responsive during long encodes/training.
*   **Self-Describing Models:** Bundled metadata ensures trained heads reload with the correct dimensions (prevents “shape mismatch” errors).

### Which version should I use?

| Scenario | Recommended |
|---|---|
| New project, large videos (≥ 5–10 min) | **v3 (beta)** – streamed encoder prevents RAM exhaustion |
| Active labeling/training with confidence-guided review | **v3 (beta)** |
| Exact reproduction of published results | **v2-stable** |
| Very old machines with a known v2 workflow | **v2-stable** (use shorter/segmented clips) |

> **Seeing “Unable to allocate N GiB” in v2?** Switch to **v3**-it streams frames and eliminates whole-video RAM spikes.

<p align="center">
    <img src=".//assets/realtime.gif" alt="CBAS actograms" style="width: 600px; height: auto;">
</p>
<p align="center"> 

###### *(Left) Real-time video recording of an individual mouse. (Right) Real-time actogram generation of nine distinct home cage behaviors.* 

</p>

## Core Modules

---
### Module 1: Acquisition

The acquisition module is capable of batch processing streaming video data from any number of network-configured real-time streaming protocol (RTSP) IP cameras. This module's core functionality remains consistent with v2.

<p align="center">
    <img src=".//assets/acquisition_1.png" alt="CBAS Acquisition Diagram" style="width: 500px; height: auto;">
</p>

---
### Module 2: Classification and Visualization (Majorly upgraded in v3)

This module uses a powerful machine learning model to automatically classify behaviors and provides tools to analyze the results.

*   **High-Performance ML Backend:** CBAS supports DINOv3 vision transformers as feature backbones with a custom LSTM head for time-series classification.
*   **Multi-Actogram Analysis:** Tiled, side-by-side behavior plots with distinct colors for clear analysis.
*   **Interactive Plotting:** Adjust bin size, start time, thresholds, light cycles; optionally plot acrophase.

<p align="center">
    <img src=".//assets/classification_1.png" alt="CBAS Classification Diagram" style="width: 500px; height: auto;">
</p>
<p align="center"> 
    <img src=".//assets/classification_2.png" alt="CBAS Classification Diagram" style="width: 500px; height: auto;">
</p>

---
### Module 3: Training (Majorly Upgraded in v3)

The training module in v3 introduces a modern, efficient workflow for creating high-quality, custom datasets and models.

*   **Active Learning Interface:** Pre-label, then “Review & Correct” using confidence filters and an interactive timeline.
*   **Flexible Training Options:** Balanced oversampling or weighted loss for rare behaviors.
*   **Automated Performance Reports:** F1/precision/recall plots and confusion matrices generated at the end of training.

<p align="center">
    <img src=".//assets/training_1.png" alt="CBAS Training Diagram" style="width: 500px; height: auto;">
</p>
<p align="center">
    <img src=".//assets/training_2.png" alt="CBAS Training Diagram" style="width: 500px; height: auto;">
</p>
<p align="center">
    <img src=".//assets/training_3.png" alt="CBAS Training Diagram" style="width: 500px; height: auto;">
</p>

-------

## Installation

We have tested the installation instructions to be as straightforward and user-friendly as possible, even for users with limited programming experience.

[**Click here for step-by-step instructions on how to install CBAS v3 from source.**](Installation.md)

------

## Updating to the Latest Version

As CBAS v3 is in active development, we recommend updating frequently to get the latest features and bug fixes. Because you installed CBAS from source using Git, updating is simple.

1.  **Open a Command Prompt** (Windows) or **Terminal** (macOS/Linux).
2.  **Navigate to your CBAS directory.** This is the folder where you originally ran the `git clone` command.
    ```bash
    # Example for Windows
    cd C:\Users\YourName\Documents\CBAS
    ```
3.  **Pull the latest changes** from the main branch on GitHub. This is the core update command:
    ```bash
    git pull origin main
    ```
4.  **Update dependencies.** Occasionally, we may add or update the required Python or Node.js packages. It's good practice to run these commands after updating:
    
    > [!NOTE]
    > Remember to activate your virtual environment before running the `pip` command.

    ```bash
    # Activate your virtual environment first
    # On Windows: .\venv\Scripts\activate
    # On macOS/Linux: source venv/bin/activate

    # Update Python packages
    pip install -r requirements.txt

    # Update Node.js packages
    npm install
    ```

After these steps, you can launch the application as usual with `npm start`, and you will be running the latest version.

------

## Setup & Use

The documentation is organized to follow the logical workflow of a typical project.

*   **1. Hardware & Project Setup**
    *   [**Hardware & Network Setup:** For instructions on configuring RTSP IP cameras and the recording rig.](Hardware_Setup.md)
    *   [**Software Installation:** For step-by-step instructions on how to install CBAS.](Installation.md)
    *   [**Understanding the Project Directory:** For a detailed breakdown of the file and folder structure.](ProjectDirectory.md)

*   **2. Core Workflows**
    *   [**Recording Video:** For a guide on adding cameras and managing recording sessions in CBAS.](Recording.md)
    *   [**Training a Custom Model:** For a detailed guide on creating a new dataset and training a model.](Training.md)
    *   [**Visualizing & Analyzing Data:** For a guide on using your trained model to analyze videos and interpret the results.](Visualization_and_Analysis.md)
	
### Optional: Using the Default `JonesLabModel`

CBAS includes a pre-trained model, the `JonesLabModel`, which can serve as a demonstration or as a starting point for the "Guided Labeling" workflow. Please be aware that this model was trained on a specific hardware and environmental setup, and its performance will vary on your own data.

This model is not loaded by default. To use it in your project:

1.  **Locate the Model:** Find the `JonesLabModel` folder inside the application's source code directory (it is located at `CBAS/models/JonesLabModel`).
2.  **Copy to Your Project:** Copy this entire `JonesLabModel` folder.
3.  **Paste into Your Project:** Paste the folder into your own project's `models/` directory.

The next time you open the "Label/Train" page or click the "Refresh Datasets" button, the `JonesLabModel` card will appear, and it will be available for inference.

### Advanced: Using Experimental Encoder Models

CBAS allows power users to experiment with different feature encoder models on a per-project basis. Copy `cbas_config.yaml.example` into your project root as `cbas_config.yaml` and edit the `encoder_model_identifier`.

> [!NOTE]
> Using **DINOv3** requires a one-time Hugging Face authentication (read token) and accepting the model’s terms. Switching encoders requires **re-encoding** videos in that project.

#### Instructions for Using Gated Models (like DINOv3)

Some state-of-the-art models require you to agree to their terms of use and authenticate with Hugging Face before you can download them. This is a one-time setup per computer.

**Step 1: Get Your Hugging Face Access Token**

1.  Log into your account on [huggingface.co](https://huggingface.co).
2.  Go to your **Settings** page by clicking your profile picture in the top-right.
3.  Navigate to the **Access Tokens** tab on the left.
4.  Create a **New token**. Give it a name (e.g., `cbas-access`) and assign it the **`read`** role.
5.  Copy the generated token (`hf_...`) to your clipboard.

**Step 2: Log In from the Command Line**

You must log in from the same terminal environment you use to run CBAS.

1.  Open a Command Prompt (Windows) or Terminal (macOS/Linux).
2.  Navigate to your main CBAS source code directory (e.g., `cd C:\Users\YourName\Documents\CBAS`).
3.  **Activate your Python virtual environment:**
    *   On Windows: `.\venv\Scripts\activate`
    *   On macOS/Linux: `source venv/bin/activate`
4.  Run the login command:
    ```bash
    huggingface-cli login
    ```
5.  Paste your access token when prompted and press Enter. Your terminal is now authenticated.

**Step 3: Agree to the Model's Terms**

Before you can download the model, you must accept its terms on the model's page.
1.  Go to the model's page on Hugging Face, for example: [facebook/dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m).
2.  If prompted, click the button to review and accept the terms of use.

**Step 4: Configure Your Project and Run CBAS**

1.  In your specific CBAS project folder, edit your `cbas_config.yaml` file to uncomment the line for the DINOv3 model.
    ```yaml
    encoder_model_identifier: "facebook/dinov3-vitb16-pretrain-lvd1689m"
    ```
2.  Save the file and run CBAS normally (`npm start`).

The first time you launch a project with the new model, the backend will download the model files, which may take several minutes. All subsequent launches will be fast. Because this is a new encoder, all videos in this project will be automatically re-queued for encoding.
--------------

## Troubleshooting

<details>
  <summary>Stuck on “Loading DINO encoder…” (Windows)</summary>
  
  1) Seed the model once from the SAME venv CBAS uses:
     > .\venv\Scripts\activate
     > huggingface-cli login
     > python -c "from huggingface_hub import snapshot_download as d; d('facebook/dinov3-vitb16-pretrain-lvd1689m', local_files_only=False, local_dir_use_symlinks=False)"
  2) Launch with offline cache so CBAS doesn’t make network calls:
     > set HF_HUB_OFFLINE=1
     > npm start
  3) If it still stalls:
     - Ensure the token is set in this venv:  `huggingface-cli whoami`
     - Accept model terms in your browser while logged in (model page on Hugging Face).
     - Upgrade hub libs in this venv:  `pip install -U huggingface_hub transformers timm`
     - Clear stale locks and re-download:
       > powershell -NoProfile -Command "Get-ChildItem -Recurse \"$env:USERPROFILE\.cache\huggingface\hub\" -Filter *.lock | Remove-Item -Force"
       > rmdir /S /Q "%USERPROFILE%\.cache\huggingface\hub\models--facebook--dinov3-vitb16-pretrain-lvd1689m"
       > python -c "from huggingface_hub import snapshot_download as d; d('facebook/dinov3-vitb16-pretrain-lvd1689m', local_files_only=False, local_dir_use_symlinks=False)"
</details>

<details>
  <summary>PyTorch size-mismatch when loading a trained model</summary>
  
  - Cause: head rebuilt with wrong dims (e.g., hidden_size 64) vs checkpoint (e.g., 128).
  - Fix (v3): loader must read from model_meta.json and pass:
      • lstm_hidden_size, lstm_layers, seq_len, behaviors
    Also verify:
      • encoder_model_identifier matches the project encoder
      • *_cls.h5 embedding width (e.g., 768 for DINOv3) matches expectations
  - If you must run an older build: add lstm_hidden_size=128 (or the trained value) when constructing ClassifierLSTMDeltas.
</details>

<details>
  <summary>v2 “Unable to allocate N GiB for an array …”</summary>
  
  - Cause: v2 loads entire video into RAM (e.g., a 10-min 720p clip can be ~90 GB).
  - Best fix: use v3 (streamed/chunked encoder; no whole-video RAM spikes).
  - If staying on v2 temporarily:
      • Split videos into short segments (e.g., 60–120 s)
      • Downsample resolution/FPS before import
      • Test with short clips first
</details>

## Hardware Requirements

While not required, we **strongly** recommend using a modern NVIDIA GPU (RTX 20-series or newer) to allow for GPU-accelerated training and inference.

Our lab's test machines:
- **CPU:** AMD Ryzen 9 5900X / 7900X
- **RAM:** 32 GB DDR4/DDR5
- **SSD:** 1TB+ NVMe SSD
- **GPU:** NVIDIA GeForce RTX 3090 24GB / 4090 24GB

-----

## Feedback

As this is a beta, feedback and bug reports are highly encouraged! Please open an [Issue](https://github.com/jones-lab-tamu/CBAS/issues) to report any problems you find.

-----

###### MIT License

###### Copyright (c) 2025 Jones Lab, Texas A&M University

###### Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

###### The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

###### THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.