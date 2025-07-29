# Installing CBAS from Source (Windows)

This guide provides step-by-step instructions for installing the CBAS v3 (Beta) desktop application.

## Step 1: Install Primary Dependencies

1.  **Git:** Download from [https://git-scm.com/download/win](https://git-scm.com/download/win) and install using the default settings.

2.  **Python (64-bit, version 3.11 or 3.12):**

    > [!IMPORTANT]
    > **You must use a 64-bit version of either Python 3.11 or Python 3.12.**
    > Due to strict dependencies for the GPU-accelerated libraries, using an unsupported Python version (like 3.10 or 3.13+) or a 32-bit installation will cause the installation to fail with a `No matching distribution found for torch` error.

    *   **Uninstall any other Python versions** from your system via "Add or remove programs" to avoid conflicts.
    *   Download the **"Windows installer (64-bit)"** for your chosen version from the official site:
        *   [**Python 3.11.9** (Recommended for maximum compatibility)](https://www.python.org/downloads/release/python-3119/)
        *   [**Python 3.12.6** (Known to be working)](https://www.python.org/downloads/release/python-3126/)
    *   Run the installer. On the first screen, **check the box that says "Add python.exe to PATH"**.

3.  **Node.js (LTS version):** Download the LTS version from [https://nodejs.org/](https://nodejs.org/) and install with default settings.

4.  **FFmpeg:** Required for video recording.
    *   Download the "essentials" build from: [https://gyan.dev/ffmpeg/builds/](https://gyan.dev/ffmpeg/builds/).
    *   Unzip the file and move the `ffmpeg-essentials_build` folder to a permanent location like `C:\`.
    *   Add the `bin` subfolder to your Windows PATH environment variable (e.g., `C:\ffmpeg-essentials_build\bin`).

## Step 2: Install and Run CBAS v3

1.  **Open a NEW Command Prompt** to ensure it recognizes the newly installed Python 3.11.

2.  **Verify Python Version:** Run the following command. The output must say `Python 3.11.9`.
    ```
    python --version
    ```
    > If it shows a different version, the PATH is incorrect. Please uninstall other Python versions and reinstall 3.11, ensuring the "Add to PATH" box is checked.

3.  **Clone the Repository and Checkout the `main` Branch:**
    ```
    cd C:\Users\YourName\Documents
    git clone https://github.com/jones-lab-tamu/CBAS.git
    cd CBAS
    git checkout main
    ```

4.  **Create and Activate a Python Virtual Environment:**
    ```
    python -m venv venv
    .\venv\Scripts\activate
    ```

5.  **Install All Python Dependencies:**

    *   First, upgrade `pip` within the virtual environment:
        ```bash
        python.exe -m pip install --upgrade pip
        ```

    *   Next, install the specific PyTorch version required for GPU acceleration.
        The following command is tailored for systems with an NVIDIA GPU and CUDA 12.1 drivers:
        ```bash
        pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
        ```

    *   Finally, install the remaining application dependencies from the `requirements.txt` file:
        ```bash
        pip install -r requirements.txt
        ```

    > **Note for Non-GPU Systems:**
	> If you do not have an NVIDIA GPU, or if the command above fails, you can install the CPU-only version of PyTorch. **Please be aware that all AI-related tasks will be significantly slower.**
	> ```bash
	> pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
	> ```

6.  **Install Node.js Dependencies:**
    ```
    npm install
    ```

7.  **Run the Application:**
    ```
    npm start
    ```

---
## Legacy v2 Installation

**Note:** The legacy v2 version is a browser-based application with different dependencies. It is recommended to install v2 in a completely separate folder and virtual environment to avoid conflicts with the modern v3 application. These instructions require installing the older CUDA Toolkit 11.8.

<details>
<summary>Click to expand v2 Installation Instructions</summary>

1.  **Install NVIDIA CUDA Toolkit 11.8:** Download and install from [NVIDIA's archive](https://developer.nvidia.com/cuda-11-8-0-download-archive). Use the "Express" installation.
2.  **Install Visual Studio Build Tools:** Download from [here](https://visualstudio.microsoft.com/visual-cpp-build-tools/) and select the "Desktop development with C++" workload.
3.  **Clone and Checkout `v2-stable` Branch:**
    ```
    git clone https://github.com/jones-lab-tamu/CBAS.git cbas-v2
    cd cbas-v2
    git checkout v2-stable
    ```
4.  **Create and Activate a Separate Virtual Environment:**
    ```
    python -m venv venv-v2
    .\venv-v2\Scripts\activate
    ```
5.  **Install Python Dependencies for v2:**
    ```
    pip install -r requirements.txt
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
6.  **Run the Application:**
    ```
    python backend/app.py
    ```

</details>