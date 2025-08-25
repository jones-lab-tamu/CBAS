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

    *   **macOS (using Homebrew):** Open Terminal and run: `brew install ffmpeg`
    *   **Linux (using apt):** Open a terminal and run: `sudo apt update && sudo apt install ffmpeg`
    *   **Windows:** This requires a few manual steps. Please follow the detailed instructions below.

    > [!NOTE]
    > **Detailed Windows Instructions for FFmpeg**
    >
    > 1.  **Download:** Go to [https://gyan.dev/ffmpeg/builds/](https://gyan.dev/ffmpeg/builds/) and download the latest "essentials" release `.zip` file.
    > 2.  **Unzip and Place:** Unzip the downloaded file. You will have a folder like `ffmpeg-7.0-essentials_build`. For simplicity, rename this folder to just `ffmpeg` and move it to a permanent location, like the root of your `C:\` drive. The final path should be `C:\ffmpeg`.
    > 3.  **Find the `bin` Folder:** Inside `C:\ffmpeg`, you will see a folder named `bin`. The full path to this folder is `C:\ffmpeg\bin`. This is the path we need to add to the Windows PATH.
    > 4.  **Open Environment Variables:**
    >     *   Click the **Start Menu** (or press the Windows key) and type `env`.
    >     *   Click on **"Edit the system environment variables"**. A "System Properties" window will open.
    >     *   Click the **"Environment Variables..."** button at the bottom.
    > 5.  **Edit the Path Variable:**
    >     *   In the new window, look in the top box titled "User variables for [YourUsername]".
    >     *   Find the variable named **`Path`** in the list and click on it to select it.
    >     *   Click the **"Edit..."** button.
    > 6.  **Add the New Path:**
    >     *   A new window will open showing a list of paths. Click the **"New"** button.
    >     *   A new, empty line will appear. Type or paste the full path to the `bin` folder: `C:\ffmpeg\bin`
    >     *   Click **OK** to close the path editor.
    > 7.  **Confirm Changes:** Click **OK** on the "Environment Variables" window, and then **OK** on the "System Properties" window to save all your changes.
    > 8.  **Verify the Installation:**
    >     *   **Close all open Command Prompt windows.** You must open a new one for the changes to take effect.
    >     *   Open a **new** Command Prompt and type: `ffmpeg -version`
    >     *   If it was successful, you will see version information printed. If you see an error like `'ffmpeg' is not recognized...`, please carefully repeat the steps above.

## Step 2: Install and Run CBAS v3

1.  **Open a NEW Command Prompt (Windows) or Terminal (macOS/Linux)** to ensure it recognizes the newly installed software.

2.  **Clone the Repository:** This downloads the CBAS source code to your computer.
    ```bash
    # We recommend cloning into your Documents folder
    cd Documents
    git clone https://github.com/jones-lab-tamu/CBAS.git
    cd CBAS
    ```
    > [!NOTE]
    > The `git clone` command will print several lines of progress to your terminal as it downloads the project. This is normal. As long as you do not see a specific message that says "ERROR" or "FATAL", the command was successful.

3.  **Create and Activate a Python Virtual Environment:**

    > [!NOTE]
    > This step creates a self-contained "bubble" for all of CBAS's specific Python packages. This is a best practice that prevents CBAS from interfering with any other Python software on your system. After you run the `activate` command, you will see `(venv)` appear at the beginning of your command prompt line. This is normal and indicates that the virtual environment is active.

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
>
> **Error: `pip install` fails with a network error (e.g., "Timeout" or "Could not connect")**
> *   **Solution:** The `pip` command needs to download files from the internet. Ensure your computer has an active internet connection. If you are on a university or corporate network, you may be behind a firewall that is blocking access. You may need to consult your IT department or try the installation on a different network.
>
> **Error: `pip` is not recognized... (or similar)**
> *   **Solution:** This almost always means your virtual environment is not active. Look for the `(venv)` at the beginning of your command prompt line. If it's not there, re-run the activation command (`.\venv\Scripts\activate` on Windows or `source venv/bin/activate` on macOS/Linux).