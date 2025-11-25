# Visualizing and Analyzing Data

Congratulations on training a custom model! This guide explains how to use your model to automatically classify behavior and how to use the two main analysis tools in CBAS: **Actograms** for session-level patterns and **Ethograms** with **Interactive Playback** for single-video inspection.

---

## 1. Understanding the Analysis Pipeline (Crucial)

CBAS v3 uses a high-performance **Two-Step Pipeline** to analyze videos. Understanding this distinction is the key to avoiding common errors like "No new files found."

### Step 1: Encoding (Automatic Background Task)
When you import or record a video, CBAS immediately starts a background process called **Encoding**.
*   **What it does:** It converts the raw video pixels (`.mp4`) into mathematical feature vectors using the DINO AI.
*   **The Output:** It creates a file ending in **`_cls.h5`** next to your video.
*   **How to check:** Look at the footer of the application. If the **Encoding Progress Bar** is moving, this step is active.

### Step 2: Inference (Manual Action)
This is what you trigger via the **Inference** button.
*   **What it does:** It takes the trained model and applies it to the **`.h5` feature files** created in Step 1.
*   **The Output:** It creates a CSV file ending in **`_outputs.csv`** containing the behavior probabilities.

> [!IMPORTANT]
> **"No new files found to process"?**
> If you try to run Inference and get this error, it usually means **Step 1 (Encoding) hasn't finished yet.** The inference engine cannot see your videos until they have been converted to `.h5` files.
>
> *   **Fix:** Check the bottom footer. If the encoding bar is idle but files are missing, **restart the app**. On startup, CBAS scans for missing `.h5` files and restarts the encoder automatically.

---

## 2. Classify Videos with Your Model (Inference)

After a model has been successfully trained—and your videos have been encoded—you can run inference.

1.  Navigate to the **Inference** tab in the top navigation bar.
2.  **Select Model:** Click on the card for the trained model you want to use.
3.  **Select Videos:** Check the boxes for the recording session(s) or specific subject folders you want to analyze.
4.  Click **Start Inference**.

<p align="center">
    <img src="./assets/v3-inference-modal.png" alt="The CBAS inference modal." style="width: 500px; height: auto;">
</p>

The status bar will update to show progress. In the background, CBAS is creating a new `_outputs.csv` file for each video segment.

---

## 3. Session Analysis: Generating Actograms

Once inference is complete, navigate to the **Visualize** page. By default, you are in "Session Analysis" mode. This mode is for analyzing entire recording sessions to see long-term behavioral patterns.

1.  The "Select Recording" panel on the left will be populated. Expand the tree to find the session, subject, and the model you just used.
2.  Click the checkbox next to one or more behaviors.
3.  An **Actogram** will automatically be generated on the right. You can select multiple behaviors to view them side-by-side.

<p align="center">
    <img src="./assets/v3-visualize-actogram.png" alt="The CBAS visualization interface." style="width: 700px; height: auto;">
</p>

The actogram is a powerful tool for circadian biology. You can use the controls at the top to adjust the plot in real-time:
*   **Bin Size:** The number of minutes to group behavioral events into. Larger bins smooth the data.
*   **Start Time:** The Zeitgeber Time (ZT) or Circadian Time (CT) to start the plot on the left axis.
*   **Threshold:** The model confidence level required to count an event.
*   **Light Cycle:** Changes the background shading to match LD, DD, or LL conditions.

---

## 4. Single Video Inspection: Ethograms and Interactive Playback

This mode allows you to closely examine a model's performance on a single video file. You can either generate a static **Ethogram** plot for a quick overview or launch an **Interactive Playback** session to watch the video with the model's predictions overlaid on the timeline.

1.  On the **Visualize** page, click the **"Single Video"** toggle at the top of the right-hand panel.
2.  The selection tree on the left will now show individual classified video files.
3.  From here, you have two options for any video in the list:
    *   **To Generate an Ethogram:** Simply **click on the video file name**. A static plot will appear on the right, showing the sequence and duration of each classified behavior.
    *   **To Start Interactive Playback:** Click the **play button icon** (<i class="bi bi-play-circle-fill"></i>) next to the video file name.

### Interactive Playback Mode

Launching the interactive playback will take you to a read-only version of the labeling interface. Here, you can watch the video frame-by-frame while seeing the model's top prediction for each frame visualized on the timeline. This is the most direct way to visually verify your model's performance and identify specific moments of confusion that you may want to correct later in the "Label/Train" page.

---

## 5. The Analysis & Refinement Loop (Improving Your Model)

Now that you have classification results, the final step is to use them to make your model even better. This is the "active learning" part of CBAS.

-   **Find areas of weakness:** By looking at the actograms, ethograms, or using the interactive playback, you might notice times when the model is confused (e.g., mislabeling "drinking" as "rearing").
-   **Correct the mistakes:** Go back to the **Label/Train** page. Use the **"Review & Correct"** workflow (described in `Training.md`) on the videos where the model performed poorly. This is much faster than labeling from scratch.
-   **Add more data:** By correcting the model's mistakes, you are adding high-quality, targeted examples to your dataset.
-   **Re-Train:** Once you've added a good number of new corrections, train a new version of your model. Its F1-score should improve!

By repeating this **Classify -> Visualize -> Correct -> Re-Train** cycle, you can quickly create highly accurate models tailored to your specific experimental conditions.