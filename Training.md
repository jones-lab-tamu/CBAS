# Training a Custom Model in CBAS v3

CBAS v3 introduces a powerful and flexible workflow for creating custom datasets and training high-performance classification models. This guide will walk you through the entire process, from creating a dataset to initiating training.

The core of the v3 workflow is the **Label/Train** page.

## 1. Create a New Dataset

A **Dataset** in CBAS is a container for your project. It holds two key things:
1.  The **list of behaviors** you want to classify (e.g., `eating`, `drinking`, `resting`).
2.  All the **labeled examples** (instances) of those behaviors that you will use to train your model.

To create your first dataset:

1.  Navigate to the **Label/Train** page.
2.  Click the large **blue `+` button** in the bottom-right corner to open the "Create Dataset" modal.
3.  Fill out the form with the following in mind:
    *   **Name:** Give your dataset a descriptive, single-word name. *Best Practice: Use underscores instead of spaces (e.g., `MyCircadianStudy`).*
    *   **Behaviors:** List the behaviors you want to classify, separated by a semicolon (`;`). You can define up to 20 behaviors. *Example: `rearing; resting; grooming`*
    *   **Directories to label:** Select the parent recording session folders that contain the videos you want to access for labeling. This helps keep your video selection list clean and relevant.
4.  Click **Create**. A new card representing your dataset will appear on the page.

> [!WARNING]
> **CRITICAL: The "Two Behavior" Rule**
> You **must** define at least two behaviors for a model to work.
>
> *   **Why?** CBAS uses a probability function (Softmax) that forces all predictions to sum to 100%. If you only define one behavior (e.g., `Rearing`), the model is mathematically forced to predict "Rearing" with 100% confidence for every single frame, regardless of what the animal is doing.
> *   **The Solution:** If you are only interested in one specific behavior, you must create a **"Background"** (or "Other") class.
>     *   *Example:* Define behaviors as `Rearing; Background`.
>     *   Label your Rearing instances.
>     *   **Also label** instances of the animal *not* rearing (sleeping, walking, etc.) and assign them to "Background."

<p align="center">
    <img src="./assets/v3-training-create-dataset.png" alt="Creating a new dataset in CBAS v3" style="width: 600px; height: auto;">
</p>

## 2. Label Your Data

This is the most critical step. High-quality labels produce high-quality models. To begin, click the **Label** button on your dataset's card. This will open the "Labeling Options" modal, presenting you with two powerful workflows.

*   **Label from Scratch** is the traditional method, essential when you have no existing model to help you.
*   **Review & Correct** is the modern, "active learning" method. It is dramatically faster for improving a dataset once you have a baseline model to work with.

### Method A: Label from Scratch (The Classic Approach)

Use this method when you are starting a brand new project and have no existing model.

1.  In the "Labeling Options" modal, click **Start From Scratch**.
2.  Select a video from the dropdown list and click **Label Selected Video**.
3.  The labeling interface will open. Use the following controls:
    *   **Navigate:** Use the arrow keys (`←` / `→`) to move through the video one frame at a time. Use `↑` and `↓` to double or halve the navigation speed.
    *   **Label an Instance:**
        1.  Go to the frame where the behavior *begins*.
        2.  Press the number key corresponding to that behavior (e.g., `1` for your first behavior). This marks the start.
        3.  Go to the frame where the behavior *ends*.
        4.  Press the **same number key again**. A colored bar will appear on the timeline, representing your new label.
    *   **Delete/Undo:** Press `Delete` to remove the instance under the playhead. Press `Backspace` to undo the last instance you *added*.
4.  When you are finished with a video, click the green **Commit Corrections** button. This saves your labels and overwrites any previous labels *for that specific video file*.

### Method B: Review & Correct (The Active Learning Workflow)

This is the fastest way to expand and refine a dataset. The workflow is simple: let an existing model do a "first pass" at labeling, then you only have to fix its mistakes.

1.  In the "Labeling Options" modal, choose an existing model (like the default `JonesLabModel` or another one you've trained) from the dropdown.
2.  Select a recording session and a specific video to pre-label.
3.  Click **Pre-Label & Correct Selected Video**. CBAS will run inference in the background and then open the labeling interface, pre-filled with the model's predictions.

<p align="center">
    <img src="./assets/v3-training-prelabel-ui.png" alt="The v3 labeling interface in Review & Correct mode" style="width: 700px; height: auto;">
</p>

**The "Review & Correct" Workflow:**

*   **Navigate Instances:** Use `Tab` and `Shift+Tab` to instantly jump between the pre-labeled behavior instances.
*   **Filter by Confidence:** The model is least certain about low-confidence predictions. Use the **Filter Confidence** slider at the bottom to hide high-confidence predictions, allowing you to focus your time on the instances the model is most likely to have gotten wrong.
*   **Correct Boundaries:** If an instance is too long or too short, move the playhead to the correct frame and use the `[` (start) and `]` (end) keys to adjust the boundary.
*   **Change Labels:** If an instance is mislabeled (e.g., the model labeled `grooming` but it's actually `resting`), move the playhead inside it and press the number key for the *correct* behavior.
*   **Confirm a Prediction:** When you are satisfied that an instance is correct, press `Enter`. This "locks" the instance and marks it with a white border.
*   **Save Your Work:** When you are finished reviewing, click the green **Commit Corrections** button. **This is very important:** this action saves **only** the instances you manually added, modified, or "locked" by pressing Enter. All other unconfirmed predictions are discarded. This ensures your dataset only contains high-quality, human-verified labels.

> ### **Pro-Tip: What Makes a Good Dataset?**
> *   **Quantity:** Aim for at least **100-200** high-quality labeled instances for each behavior. For rare or subtle behaviors, more is always better.
> *   **Signs of Too Little Data:** If your training finishes very quickly (e.g., in fewer than 5 epochs) or hits "Early Stopping" almost immediately, the model has likely "memorized" your small dataset rather than learning general patterns.
>     *   **Rule of Thumb:** 15 examples is not enough. Aim for **50–100 instances** per behavior as a starting point.
> *   **Variety:** The most important factor is variety. Ensure your labeled examples come from:
>     *   Multiple different animals.
>     *   Different times of day (and different lighting conditions).
>     *   Different days of recording.
> A model trained on a wide variety of examples will be much more robust and generalize better to new, unseen videos.

## 3. Train the Model

Once you have a sufficient number of labeled and confirmed instances (>100 per behavior is a good starting point), you can train your model.

1.  On the Label/Train page, click the **Train** button on your dataset's card. This will open the training parameter modal.

<p align="center">
    <img src="./assets/v3-training-modal.png" alt="The training parameter modal" style="width: 400px; height: auto;">
</p>

2.  Adjust the training parameters if needed. While the defaults are a good starting point, understanding these options can help you fine-tune your model.

### Understanding the Training Parameters

#### Training Methods: Handling Unbalanced Data

Real-world behavioral data is often unbalanced (e.g., a mouse might spend thousands of frames `resting` but only a few hundred `rearing`). These methods help the model learn effectively from all behaviors, even the rare ones.

*   **Balanced Sampling (Oversampling) - `Default & Recommended`**
    *   **What it is:** This method ensures the model sees a balanced mix of all behaviors during training. It does this by showing examples of rare behaviors more frequently.
    *   **When to use it:** This is the **best choice for most situations**, especially when you have one or more behaviors with significantly fewer labeled examples than others. It is the recommended default.

*   **Weighted Loss**
    *   **What it is:** Instead of showing more examples, this method tells the model to "pay more attention" to its mistakes on rare behaviors. It applies a higher penalty or "cost" when it misclassifies a rare behavior, forcing it to learn them better.
    *   **When to use it:** This is a more advanced technique. It can be effective if balanced sampling doesn't yield good results.

---
#### Advanced Parameters

*   **Sequence Length**
    *   **What it is:** The number of video frames (the "context window") the model looks at to classify the single frame in the center. It must be an odd number.
    *   **When to change it:** The default of `31` is a good general-purpose value. If you are focused on very long-duration behaviors, you could try increasing it (e.g., to `61`). If you are focused on very fast, twitch-like behaviors, you could try decreasing it (e.g., to `15`).

*   **Batch Size**
    *   **What it is:** How many training examples the model processes at once before updating its knowledge.
    *   **When to change it:** **If you see a `CUDA out of memory` error** in the command line window running the backend, **this is the first parameter you should lower**. Try halving it (e.g., from `512` to `256`). If you have a very powerful GPU (e.g., 24GB+ VRAM), you could try increasing it.

*   **Learning Rate**
    *   **What it is:** How big of a "step" the model takes when it learns from its mistakes.
    *   **When to change it:** You should generally **not need to change this**. The default value is chosen to be small and stable.

*   **Epochs**
    *   **What it is:** One "epoch" represents one full pass through your entire training dataset.
    *   **When to change it:** After training, check the `f1-score_epochs_plot.png`. If the "Weighted Avg" line was still steadily improving at the end, you can increase this value to `20` or more to give it more time to learn.

---
#### Advanced: Using a Held-Out Test Set for Unbiased Evaluation

For rigorous scientific validation, it's crucial to evaluate your final model on data it has never seen during training or model selection. CBAS supports this through a **held-out test set**.

**How it Works:**
When you enable this option, CBAS performs a 3-way data split based on your subjects/animals:
1.  **Training Set:** Used to train the model.
2.  **Validation Set:** Used during training to select the best-performing model from all the trials.
3.  **Test Set:** A completely separate set of subjects that is "held out". It is used only *once* at the very end to provide a final, unbiased F1 score of the winning model.

**How to Use It:**
1.  In the training parameter modal, expand the "Advanced Options" section.
2.  Check the box for **"Enable Held-Out Test Set"**.
3.  Use the slider to determine what percentage of your subjects to reserve for the test set (20% is a good starting point).
4.  Before you can train, CBAS will run a "pre-flight check" to ensure you have enough unique subjects to create all three data splits without losing any behaviors. If the check fails, the "Train" button will be disabled.

After training is complete, your dataset card will display a **"Test F1"** score. This score is the most accurate representation of how your model will likely perform on new, unseen data from your lab.

---
3.  Click **Train** to begin.

The training process will start in the background. You can safely continue to use other parts of the application while a model trains.

### Understanding the Training Results

Once training is complete, the performance metrics on the dataset card will update. More importantly, CBAS automatically generates a set of detailed performance reports inside your dataset's folder.

To view these reports:
1.  Click the **Manage** button on your dataset's card.
2.  Click **Show Dataset Files in Explorer**.

Inside this folder, you will find several new `.png` image files:

*   **`confusion_matrix_BEST.png`:** This shows you exactly where your model is getting confused. The diagonal shows correct predictions. Any numbers *off* the diagonal show you which behaviors are being mistaken for others.
    <p align="center">
        <img src="./assets/v3-training-cm.png" alt="Example Confusion Matrix" style="width: 400px; height: auto;">
    </p>

*   **`f1-score_epochs_plot.png`:** This plot shows how the model's performance on each behavior changed over the training epochs. This is useful for diagnosing training issues.
    <p align="center">
        <img src="./assets/v3-training-f1-plot.png" alt="Example F1 Score Plot" style="width: 500px; height: auto;">
    </p>

## 4. Next Steps: Using Your Model

After training is complete, you are now ready to use your model to analyze your video data.

➡️ [**Click here for the guide on Visualizing and Analyzing Data**](Visualization_and_Analysis.md)