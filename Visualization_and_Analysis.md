# Visualizing and Analyzing Data

Congratulations on training a custom model! This guide explains how to use your model to automatically classify behavior and how to use the two main analysis tools in CBAS: **Actograms** for session-level patterns and **Ethograms** with **Interactive Playback** for single-video inspection.

---

## 1. Classify Videos with Your Model (Inference)

After a model has been successfully trained, the "Infer" button on its card becomes active. This allows you to run your model on any set of recordings.

1.  On the **Label/Train** page, find the card for the model you want to use.
2.  Click the **Infer** button. This will open the "Start Classification" modal.
3.  Select the recording session(s) or specific subject folders you want to analyze.
4.  Click **Start**.

<p align="center">
    <img src="./assets/v3-inference-modal.png" alt="The CBAS inference modal." style="width: 500px; height: auto;">
</p>

The status on the model's card will update to show "Inference tasks queued..." and a progress bar will appear as the videos are processed. In the background, CBAS is creating a new `_outputs.csv` file for each video segment. These files contain the frame-by-frame predictions from your model.

---

## 2. Session Analysis: Generating Actograms

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

## 3. Single Video Inspection: Ethograms and Interactive Playback

This mode allows you to closely examine a model's performance on a single video file. You can either generate a static **Ethogram** plot for a quick overview or launch an **Interactive Playback** session to watch the video with the model's predictions overlaid on the timeline.

1.  On the **Visualize** page, click the **"Single Video"** toggle at the top of the right-hand panel.
2.  The selection tree on the left will now show individual classified video files.
3.  From here, you have two options for any video in the list:
    *   **To Generate an Ethogram:** Simply **click on the video file name**. A static plot will appear on the right, showing the sequence and duration of each classified behavior.
    *   **To Start Interactive Playback:** Click the **play button icon** (<i class="bi bi-play-circle-fill"></i>) next to the video file name.

### Interactive Playback Mode

Launching the interactive playback will take you to a read-only version of the labeling interface. Here, you can watch the video frame-by-frame while seeing the model's top prediction for each frame visualized on the timeline. This is the most direct way to visually verify your model's performance and identify specific moments of confusion that you may want to correct later in the "Label/Train" page.

---

## 4. The Analysis & Refinement Loop (Improving Your Model)

Now that you have classification results, the final step is to use them to make your model even better. This is the "active learning" part of CBAS.

-   **Find areas of weakness:** By looking at the actograms, ethograms, or using the interactive playback, you might notice times when the model is confused (e.g., mislabeling "drinking" as "rearing").
-   **Correct the mistakes:** Go back to the **Label/Train** page. Use the **"Review & Correct"** workflow (described in `Training.md`) on the videos where the model performed poorly. This is much faster than labeling from scratch.
-   **Add more data:** By correcting the model's mistakes, you are adding high-quality, targeted examples to your dataset.
-   **Re-Train:** Once you've added a good number of new corrections, train a new version of your model. Its F1-score should improve!

By repeating this **Classify -> Visualize -> Correct -> Re-Train** cycle, you can quickly create highly accurate models tailored to your specific experimental conditions.