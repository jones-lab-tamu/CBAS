# Recording with CBAS

This guide explains how to add cameras to a CBAS project and manage recording sessions.

> **Prerequisite:** This guide assumes your camera hardware, computer, and network switch have already been configured according to the [**Hardware & Network Setup Guide**](Hardware_Setup.md).

---

## 1. Create or Open a Project

First, launch the CBAS application. You will be prompted to either **Create a project** or **Open a project**. All recordings and data will be saved within this project directory.

## 2. Add a Camera to the Project

1.  Navigate to the **Record** tab in the CBAS interface.
2.  Click the large **blue `+` button** in the bottom-right corner to open the "Add Camera" modal.
3.  Fill in the camera's details:
    -   **Camera Name:** A unique name for this camera (e.g., `Cage_01`, `Mouse_A123`).
    -   **RTSP URL:** The full Real-Time Streaming Protocol URL for the camera. This includes the username and password you set during the hardware setup.
        -   **Format:** `rtsp://<username>:<password>@<camera_ip_address>:8554/profile0`
        -   **Example:** `rtsp://admin:MySecurePwd@192.168.1.51:8554/profile0`
4.  Click **Add**.

A new card for your camera will appear on the Record page. CBAS will attempt to fetch a thumbnail image.

## 3. Configure Camera Settings in CBAS

Before you record, you can set the cropping and final resolution for your analysis videos.

1.  Click the **Settings/Crop** button (<i class="bi bi-crop"></i>) on the camera's card.
2.  In the settings window, you can draw a rectangle directly on the video preview to crop the image to your region of interest (e.g., the animal's home cage).
3.  You can also adjust the final video settings:
    -   **Record FPS:** The framerate for the saved video file. 10 FPS is recommended.
    -   **Analysis Res.:** The resolution (in pixels) for the final analysis video. `256` is recommended for the DINOv2 model.
    -   **Segment (sec):** The length of each video segment in seconds (e.g., `600` for 10-minute files).
4.  Click **Save**. Repeat for all cameras.

<p align="center">
    <img src=".//assets/acquisition_1.png" alt="CBAS Acquisition Diagram" style="width: 500px; height: auto;">
</p>

## 4. Start and Stop Recordings

1.  At the top of the Record page, enter a **Current Session Name** (e.g., `Experiment1_Baseline`). This will be the name of the folder where your recordings are saved.
2.  To start an individual camera, click the green **Start** button on its card.
3.  To start all cameras at once, click the **large video camera button** in the bottom-right.
4.  The camera card will update to show a timer and the status dot will blink.
5.  To stop recording, click the red **Stop** button on the card, or the large square button in the bottom-right to stop all cameras.

All recorded video files will be automatically saved and organized in your project directory under `My_Project/recordings/<Session_Name>/<Camera_Name>/`.