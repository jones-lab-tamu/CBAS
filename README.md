<p align="center">
    <img src="./frontend/assets/cbas_logo.svg" alt="CBAS Logo" style="width: 500px; height: auto;">
</p>
<details align="center">
  <summary>What's with the old fish logo in CBASv2?</summary>
  <br>
  <p>The original CBAS logo featured fish, which confused some of our rodent-focused users! The name CBAS is pronounced like "sea bass," which was the source of the original pun.</p>
  <img src="./frontend/assets/cbas_mouseover.svg" alt="CBAS Easter Egg Logo" style="width: 500px; height: auto;">
</details>

# CBAS v3 (BETA)

> [!WARNING]
> **This is the development branch for the upcoming CBAS v3.** This version is a complete architectural and functional overhaul and should be considered **experimental**. It may contain bugs or undergo significant changes.

> For the current stable version (v2), please see the [**`v2-stable` branch**](https://github.com/jones-lab-tamu/CBAS/tree/v2-stable).

---

CBAS (Circadian Behavioral Analysis Suite) is a full-featured, open-source application for phenotyping complex animal behaviors. It is designed to automate the classification of behaviors from video data and provide a simple, powerful interface for visualizing and analyzing the results.

*Originally created by Logan Perry and now maintained by the Jones Lab at Texas A&M University.*

## Key Features at a Glance

*   **Standalone Desktop App:** A robust, cross-platform application for Windows, macOS, and Linux. No web browser required.
*   **Real-time Video Acquisition:** Record and process video from any number of network-based RTSP cameras simultaneously.
*   **High-Performance AI:** Uses a state-of-the-art DINOv2 Vision Transformer backbone for powerful and accurate feature extraction.
*   **Active Learning Workflow:** Dramatically accelerate labeling by pre-labeling videos with an existing model and using the "Review & Correct" interface to rapidly fix mistakes.
*   **Advanced Labeling Interface:** Features an interactive, zoomable timeline, confidence-based filtering, and keyboard shortcuts for efficient instance correction.
*   **Automated Model Training:** Create custom, high-performance behavior classifiers with just a few clicks. Includes tools for handling rare behaviors and generates detailed performance reports.
*   **Rich Data Visualization:** Generate multi-plot actograms, compare behaviors side-by-side, and analyze circadian patterns with interactive controls.

## What's New in CBAS v3?

CBAS v3 is a ground-up rebuild of the original application, focused on stability, performance, and a dramatically improved user workflow.

*   **Standalone Desktop Application:** v3 is now a robust, cross-platform desktop app powered by Electron. It no longer runs in a web browser and can be used completely offline.
*   **Advanced "Active Learning" Workflow:** The labeling process has been supercharged. Pre-label videos with any model, then use `Tab` to instantly jump between the model's predictions and use the interactive timeline to rapidly correct boundaries.
*   **Confidence-Based Filtering:** Focus your efforts where they matter most. The new interface allows you to filter and view only the behavioral instances the model is least certain about, making your labeling time more efficient.
*   **Enhanced Visualization:** Generate side-by-side actograms for direct comparison of multiple behaviors.
*   **Modern, Stable Backend:** The application's backend has been re-engineered with dedicated worker threads for a stable, responsive, and crash-free experience during intensive tasks like training and inference.

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
### Module 2: Classification and Visualization (Majorly Upgraded in v3)

This module uses a powerful machine learning model to automatically classify behaviors and provides tools to analyze the results.

*   **High-Performance ML Backend:** CBAS uses a frozen [DINOv2 vision transformer](https://arxiv.org/abs/2304.07193) as its feature-extracting backbone, with a custom LSTM-based model head for time-series classification.
*   **Multi-Actogram Analysis:** The new visualization interface allows for **tiled, side-by-side comparison of multiple behaviors**, each with a distinct color for clear analysis.
*   **Interactive Plotting:** All actogram parameters (bin size, start time, thresholds, light cycles) can be adjusted in real-time. You can also toggle the plotting of the acrophase to analyze circadian periodicity.

<p align="center">
    <img src=".//assets/classification_1.png" alt="CBAS Classification Diagram" style="width: 500px; height: auto;">
</p>
<p align="center"> 
    <img src=".//assets/classification_2.png" alt="CBAS Classification Diagram" style="width: 500px; height: auto;">
</p>

---
### Module 3: Training (Majorly Upgraded in v3)

The training module in v3 introduces a modern, efficient workflow for creating high-quality, custom datasets and models.

*   **Active Learning Interface:** The new "Review & Correct" mode allows you to pre-label videos with an existing model. You can then use the new **confidence-based filtering and interactive timeline** to rapidly find and correct the model's mistakes, dramatically reducing manual labeling time.
*   **Flexible Training Options:** Train models using balanced oversampling or a weighted-loss function to handle rare behaviors.
*   **Automated Performance Reports:** After training, CBAS automatically generates detailed performance reports, including F1/precision/recall plots and confusion matrices, to help you evaluate and trust your new custom model.

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

## Setup & Use

The documentation is organized to follow the logical workflow of a typical project.

*   **1. Hardware & Project Setup**
    *   [**Hardware & Network Setup:** For instructions on configuring RTSP IP cameras and the recording rig.](Hardware_Setup.md)
    *   [**Software Installation:** For step-by-step instructions on how to install CBAS.](Installation.md)
    *   [**Understanding the Project Directory:** For a detailed breakdown of the file and folder structure.](ProjectDirectory.md)

*   **2. Core Workflows**
    *   [**Recording Video:** For a guide on adding cameras and managing recording sessions in CBAS.](Recording.md)
    *   [**Training a Custom Model:** For a detailed guide on creating a new dataset and training a model.](Training.md)
    *   [**Classifying & Visualizing Data:** For a guide on using your trained model to analyze videos and interpret the results.](Analysis.md)

--------------
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

###### THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.