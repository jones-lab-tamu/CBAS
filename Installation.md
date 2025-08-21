# Installing CBAS from Source

This guide provides step-by-step instructions for installing the CBAS v3 (Beta) desktop application. These instructions are tailored for users with limited programming experience.

## Step 1: Install Primary Dependencies

1.  **Git:** Download from [https://git-scm.com/](https://git-scm.com/) and install using the default settings. This is required to download the CBAS source code.

2.  **Python (64-bit, version 3.11 or 3.12):**

    > [!IMPORTANT]
    > **You must use a 64-bit version of either Python 3.11 or Python 3.12.**
    > Due to strict dependencies for the GPU-accelerated libraries, using an unsupported Python version (like 3.10 or 3.13+) or a 32-bit installation will cause the installation to fail.

    *   **Uninstall any other Python versions** from your system via "Add or remove programs" (Windows) or your system's package manager to avoid conflicts.
    *   Download the **"Windows installer (64-bit)"** or **"macOS 64-bit universal installer"** for your chosen version from the official site:
        *   [**Python 3.11.9** (Recommended for maximum compatibility)](https://www.python.org/downloads/release/python-3119/)
        *   [**Python 3.12.6** (Known to be working)](https://www.python.org/downloads/release/python-3126/)
    *   Run the installer. On the first screen, **check the box that says "Add python.exe to PATH"** (Windows) or ensure you install it for your user.

3.  **Node.js (LTS version):** Download the LTS version from [https://nodejs.org/](https://nodejs.org/) and install with default settings.

4.  **FFmpeg:** Required for video recording and processing.
    *   **Windows:** Download the "essentials" build from: [https://gyan.dev/ffmpeg/builds/](https://gyan.dev/ffmpeg/builds/). Unzip the file and move the `ffmpeg-essentials_build` folder to a permanent location like `C:\`. Add the `bin` subfolder to your Windows PATH environment variable (e.g., `C:\ffmpeg-essentials_build\bin`).
    *   **macOS (using Homebrew):** Open Terminal and run: `brew install ffmpeg`
    *   **Linux (using apt):** Open a terminal and run: `sudo apt update && sudo apt install ffmpeg`

## Step 2: Install and Run CBAS v3

1.  **Open a NEW Command Prompt (Windows) or Terminal (macOS/Linux)** to ensure it recognizes the newly installed software.

2.  **Clone the Repository:** This downloads the CBAS source code to your computer.
    ```bash
    # We recommend cloning into your Documents folder
    cd Documents
    git clone https://github.com/jones-lab-tamu/CBAS.git
    cd CBAS
    ```

3.  **Create and Activate a Python Virtual Environment:** This creates a self-contained space for CBAS's Python packages.

    **On Windows (Command Prompt):**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```
    **On macOS / Linux (Terminal):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

4.  **Install All Python Dependencies:**

    *   First, upgrade `pip` within the virtual environment:
        ```bash
        python -m pip install --upgrade pip
        ```

    *   Next, install the specific PyTorch version required for GPU acceleration.
        > [!NOTE]
        > The following command is tailored for systems with an NVIDIA GPU and CUDA 12.1 drivers. A modern NVIDIA driver is required.
        ```bash
        pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --extra-index-url https://download.pytorch.org/whl/cu121
        ```

    *   Finally, install the remaining application dependencies from the `requirements.txt` file:
        ```bash
        pip install -r requirements.txt
        ```

    > **For Non-GPU or AMD/Apple Silicon Systems:**
	> If you do not have an NVIDIA GPU, or if the command above fails, you can install the CPU-only version of PyTorch. **Please be aware that all AI-related tasks will be significantly slower.**
	> ```bash
	> pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1
	> ```

5.  **Install Node.js Dependencies:**
    ```bash
    npm install
    ```

6.  **Run the Application:**
    ```bash
    npm start
    ```
    The CBAS application window should now open.

---
## Troubleshooting

> **Error: `No matching distribution found for torch`**
> This is the most common installation error. It almost always means you are using an unsupported version of Python.
> *   **Solution:** Ensure you have completely uninstalled all other versions of Python and have installed the **64-bit** version of Python 3.11 or 3.12. In your terminal, run `python --version` (Windows) or `python3 --version` (macOS/Linux). The output *must* be one of the supported versions.
>
> **Error: `git is not recognized as an internal or external command`**
> *   **Solution:** Git was not installed correctly or was not added to your system's PATH. Re-install Git from [https://git-scm.com/](https://git-scm.com/) and ensure you use the default installation settings, which include adding it to the PATH.