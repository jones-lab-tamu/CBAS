# Classifying and Visualizing Data

Congratulations on training a custom model! This guide explains the next steps in the CBAS workflow: using your new model to automatically classify behavior in your videos and visualizing the results.

This is the core analysis loop of CBAS. You will typically repeat this process: classify with your best model, visualize the results to find where it struggles, and then go back to add more labels to improve it further.

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

## 2. Visualize the Results

Once inference is complete, you can view the results on the **Visualize** page.

1.  Navigate to the **Visualize** tab.
2.  The "Select Recording" panel on the left will now be populated. You can expand the tree to find the session, subject, and the model you just used for classification.
3.  Click the checkbox next to one or more behaviors.
4.  An **Actogram** will automatically be generated on the right.

<p align="center">
    <img src="./assets/v3-visualize-actogram.png" alt="The CBAS visualization interface." style="width: 700px; height: auto;">
</p>

The actogram is a powerful tool for circadian biology. It shows the intensity of a behavior over time, double-plotted to make daily patterns easy to see. You can use the controls at the top to adjust the bin size, start time, and light cycle to customize the plot for your experiment.

---

## 3. The Analysis & Refinement Loop (Improving Your Model)

Now that you have classification results, the final step is to use them to make your model even better. This is the "active learning" part of CBAS.

-   **Find areas of weakness:** By looking at the actograms or even watching the raw video, you might notice times when the model is confused (e.g., mislabeling "drinking" as "rearing").
-   **Correct the mistakes:** Go back to the **Label/Train** page. Use the **"Review & Correct"** workflow (described in `Training.md`) on the videos where the model performed poorly. This is much faster than labeling from scratch.
-   **Add more data:** By correcting the model's mistakes, you are adding high-quality, targeted examples to your dataset.
-   **Re-Train:** Once you've added a good number of new corrections, train a new version of your model. Its F1-score should improve!

By repeating this **Classify -> Visualize -> Correct -> Re-Train** cycle, you can quickly create highly accurate models tailored to your specific experimental conditions.