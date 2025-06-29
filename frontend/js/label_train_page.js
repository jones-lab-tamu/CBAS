/**
 * @file Manages the Label & Train page UI and interactions.
 * @description This file handles dataset listing, model training, inference, and the entire
 * advanced labeling workflow including pre-labeling and instance-based navigation.
 */

let cardRefreshTimeoutId = null;

// =================================================================
// EEL-EXPOSED & LOG PANEL FUNCTIONS
// =================================================================

eel.expose(updateDatasetLoadProgress);
function updateDatasetLoadProgress(datasetName, percent) {
    const container = document.getElementById(`progress-container-${datasetName}`);
    const bar = document.getElementById(`progress-bar-${datasetName}`);
    if (!container || !bar) return;

    if (percent < 0) { // Negative value hides the bar
        container.style.display = 'none';
    } else if (percent >= 100) {
        bar.style.width = '100%';
        bar.innerText = 'Complete!';
        // Hide the bar after a short delay
        setTimeout(() => {
            container.style.display = 'none';
            // Reset for next time
            bar.style.width = '0%';
            bar.innerText = '';
        }, 2000);
    } else {
        container.style.display = 'block'; // Make sure it's visible
        const displayPercent = Math.round(percent);
        bar.style.width = `${displayPercent}%`;
        bar.innerText = `Processing: ${displayPercent}%`;
    }
    return true;
}

eel.expose(update_augmentation_progress);
function update_augmentation_progress(percent, label = "Augmenting Dataset...") {
    const overlay = document.getElementById('progress-bar-overlay');
    const bar = document.getElementById('progress-bar-element');
    const barLabel = document.getElementById('progress-bar-label');

    if (!overlay || !bar || !barLabel) return;

    if (percent < 0) { // A negative value signals that the task is done/failed.
        overlay.style.display = 'none';
        return;
    }

    overlay.style.display = 'block';
    barLabel.innerText = label;

    const displayPercent = Math.round(percent);
    bar.style.width = `${displayPercent}%`;
    bar.innerText = `${displayPercent}%`;
    bar.setAttribute('aria-valuenow', displayPercent);

    if (displayPercent >= 100) {
        barLabel.innerText = "Finalizing...";
        // Optional: Hide after a short delay
        setTimeout(() => {
            overlay.style.display = 'none';
        }, 2000);
    }
    return true;	
}


async function startAugmentation(sourceDatasetName, newDatasetName) {
    if (!sourceDatasetName || !newDatasetName) return;

    augmentDatasetBsModal.hide();
    
    // Do NOT show the full-screen 'cover-spin'.
    // ONLY show the bottom progress bar. This leaves the UI responsive.
    update_augmentation_progress(0, `Augmenting '${sourceDatasetName}'...`);

    try {
        // This is a "fire-and-forget" call.
        // The JavaScript thread is immediately free to receive other events.
        eel.create_augmented_dataset(sourceDatasetName, newDatasetName)();
    } catch (error) {
        // This catch block will now only catch errors if the eel call itself fails to launch.
        showErrorOnLabelTrainPage("An error occurred while trying to start the augmentation task: " + error.message);
        update_augmentation_progress(-1); // Hide progress bar on launch error.
    }
}

// =================================================================
// GLOBAL STATE & VARIABLES
// =================================================================

let routingInProgress = false;
let labelingInterfaceActive = false;
let scrubSpeedMultiplier = 1;
let confidenceFilterDebounceTimer;
let recordingDirTree = {};
let selectedVideoPathsForImport = [];

const addDatasetModalElement = document.getElementById('addDataset');
const trainModalElement = document.getElementById('trainModal');
const inferenceModalElement = document.getElementById('inferenceModal');
const errorModalElement = document.getElementById('errorModal');
const preLabelModalElement = document.getElementById('preLabelModal');
const importVideosModalElement = document.getElementById('importVideosModal');
const manageDatasetModalElement = document.getElementById('manageDatasetModal');
const augmentDatasetModalElement = document.getElementById('augmentDatasetModal');
const syncDatasetModalElement = document.getElementById('syncDatasetModal');

let addDatasetBsModal = addDatasetModalElement ? new bootstrap.Modal(addDatasetModalElement) : null;
let trainBsModal = trainModalElement ? new bootstrap.Modal(trainModalElement) : null;
let inferenceBsModal = inferenceModalElement ? new bootstrap.Modal(inferenceModalElement) : null;
let generalErrorBsModal = errorModalElement ? new bootstrap.Modal(errorModalElement) : null;
let preLabelBsModal = preLabelModalElement ? new bootstrap.Modal(preLabelModalElement) : null;
let importVideosBsModal = importVideosModalElement ? new bootstrap.Modal(importVideosModalElement) : null;
let manageDatasetBsModal = manageDatasetModalElement ? new bootstrap.Modal(manageDatasetModalElement) : null;
let augmentDatasetBsModal = augmentDatasetModalElement ? new bootstrap.Modal(augmentDatasetModalElement) : null;
let syncDatasetBsModal = syncDatasetModalElement ? new bootstrap.Modal(syncDatasetModalElement) : null;

let currentLabelingMode = 'scratch'; // Default to 'scratch'

// =================================================================
// ROUTING & UTILITY FUNCTIONS
// =================================================================

function routeToRecordPage() { routingInProgress = true; window.location.href = './record.html'; }
function routeToLabelTrainPage() { routingInProgress = true; window.location.href = './label-train.html'; }
function routeToVisualizePage() { routingInProgress = true; window.location.href = './visualize.html'; }

function getTextColorForBg(hexColor) {
    if (!hexColor) return '#000000';
    const cleanHex = hexColor.startsWith('#') ? hexColor.slice(1) : hexColor;
    if (cleanHex.length !== 6) return '#000000';
    const r = parseInt(cleanHex.substring(0, 2), 16);
    const g = parseInt(cleanHex.substring(2, 4), 16);
    const b = parseInt(cleanHex.substring(4, 6), 16);
    const luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
    return luminance > 0.5 ? '#000000' : '#FFFFFF';
}

function showManageDatasetModal(datasetName) {
    if (!manageDatasetBsModal) return;
    document.getElementById('md-dataset-name').innerText = datasetName;

    // Attach event for the "Show Files" button
    const revealBtn = document.getElementById('revealFilesButton');
    if (revealBtn) {
        revealBtn.onclick = () => {
            eel.reveal_dataset_files(datasetName)();
            manageDatasetBsModal.hide();
        };
    }

    // Attach event for the new "Recalculate Stats" button
    const recalcBtn = document.getElementById('recalculateStatsButton');
    if (recalcBtn) {
        recalcBtn.onclick = async () => { // Make this function async
            if (confirm(`Are you sure you want to recalculate stats for '${datasetName}'? ...`)) {
                
                // Show a spinner while waiting
                document.getElementById('cover-spin').style.visibility = 'visible';
                
                // Call the Eel function and AWAIT its response
                const updatedDatasets = await eel.recalculate_dataset_stats(datasetName)();
                
                if (updatedDatasets) {
                    // Pass the new data directly to the rendering function
                    loadInitialDatasetCards(updatedDatasets);
                } else {
                    showErrorOnLabelTrainPage("Failed to get updated stats from the backend.");
                }
                
                document.getElementById('cover-spin').style.visibility = 'hidden';
                manageDatasetBsModal.hide();
            }
        };
    }

    manageDatasetBsModal.show();
}

function createCountCellHTML(data) {
    if (!data || typeof data.inst === 'undefined') {
        return data || 'N/A';
    }

    let instHtml = `${data.inst}`;
    let frameHtml = `(${data.frame})`;
    let instTooltip = 'Instances';
    let frameTooltip = 'Frames';

    if (data.inst_status === 'low') {
        instHtml = `<span class="text-danger fw-bold">${data.inst}</span>`;
        instTooltip = 'Warning: Low instance count.';
    } else if (data.inst_status === 'high') {
        instHtml = `<span class="text-info fw-bold">${data.inst}</span>`;
        instTooltip = 'Note: High instance count.';
    }

    if (data.frame_status === 'low') {
        frameHtml = `<span class="text-danger fw-bold">(${data.frame})</span>`;
        frameTooltip = 'Warning: Low frame count.';
    }

    return `<span title="${instTooltip}">${instHtml}</span> <span title="${frameTooltip}">${frameHtml}</span>`;
}

function showAugmentModal(datasetName) {
    if (!augmentDatasetBsModal) return;
    const newName = `${datasetName}_aug`;
    document.getElementById('aug-dataset-name').innerText = datasetName;
    document.getElementById('aug-new-dataset-name').innerText = newName;
    
    // Attach the onclick event here to ensure we have the correct datasetName
    const startBtn = document.getElementById('startAugmentationButton');
    startBtn.onclick = () => startAugmentation(datasetName, newName);

    augmentDatasetBsModal.show();
}

function showSyncModal(sourceDatasetName, targetDatasetName) {
    if (!syncDatasetBsModal) return;

    // Populate the modal with the correct dataset names
    document.getElementById('sync-target-dataset-name').innerText = targetDatasetName;
    document.getElementById('sync-target-dataset-name-body').innerText = targetDatasetName;
    document.getElementById('sync-source-dataset-name').innerText = sourceDatasetName;
    
    // Attach the onclick event to the button
    const startBtn = document.getElementById('startSyncButton');
    startBtn.onclick = () => startSync(sourceDatasetName, targetDatasetName);

    syncDatasetBsModal.show();
}

async function startSync(sourceDatasetName, targetDatasetName) {
    if (!sourceDatasetName || !targetDatasetName) return;

    syncDatasetBsModal.hide();
    document.getElementById('cover-spin').style.visibility = 'visible';
    update_log_panel(`Syncing labels from '${sourceDatasetName}' to '${targetDatasetName}'...`);

    try {
        // We'll create this new eel function in the next step
        await eel.sync_augmented_dataset(sourceDatasetName, targetDatasetName)();
    } catch (error) {
        showErrorOnLabelTrainPage("An error occurred while trying to start the sync task: " + error.message);
    } finally {
        // Always hide the spinner
        document.getElementById('cover-spin').style.visibility = 'hidden';
    }
}

// =================================================================
// EEL-EXPOSED FUNCTIONS (Called FROM Python)
// =================================================================

eel.expose(notify_import_complete);
function notify_import_complete(success, message) {
    document.getElementById('cover-spin').style.visibility = 'hidden';
    if (success) {
        console.log("Import complete, refreshing dataset list.");
        loadInitialDatasetCards();
        alert(message);
    } else {
        showErrorOnLabelTrainPage(message);
    }
    return true;	
}

eel.expose(showErrorOnLabelTrainPage);
function showErrorOnLabelTrainPage(message) {
    const errorMessageElement = document.getElementById("error-message");
    if (errorMessageElement && generalErrorBsModal) {
        errorMessageElement.innerText = message;
        generalErrorBsModal.show();
    } else {
        alert(message);
    }
    return true;	
}

eel.expose(updateLabelImageSrc);
function updateLabelImageSrc(mainFrameBlob, timelineBlob, zoomBlob) {
    const mainFrameImg = document.getElementById('label-image');
    const fullTimelineImg = document.getElementById('full-timeline-image');
    const zoomImg = document.getElementById('zoom-bar-image');
    const blankGif = "data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=";

    if (mainFrameImg) mainFrameImg.src = mainFrameBlob ? "data:image/jpeg;base64," + mainFrameBlob : "assets/noVideo.png";
    if (fullTimelineImg) fullTimelineImg.src = timelineBlob ? "data:image/jpeg;base64," + timelineBlob : blankGif;
    if (zoomImg) zoomImg.src = zoomBlob ? "data:image/jpeg;base64," + zoomBlob : blankGif;
    return true;	
}

eel.expose(updateFileInfo);
function updateFileInfo(filenameStr) {
    const elem = document.getElementById('file-info');
    if (elem) elem.innerText = filenameStr || "No video loaded";
    return true;	
}

eel.expose(updateLabelingStats);
function updateLabelingStats(behaviorName, instanceCount, frameCount) {
    const elem = document.getElementById(`controls-${behaviorName}-count`);
    if (elem) elem.innerHTML = `${instanceCount} / ${frameCount}`;
    return true;	
}

eel.expose(updateMetricsOnPage);
function updateMetricsOnPage(datasetName, behaviorName, metricGroupKey, metricValue) {
    const idSuffixMap = {
        'Train #': 'train-count',
        'Test #': 'test-count',
        'Precision': 'precision',
        'Recall': 'recall',
        'F1 Score': 'fscore'
    };
    const suffix = idSuffixMap[metricGroupKey];
    if (!suffix) return;
    const elem = document.getElementById(`${datasetName}-${behaviorName}-${suffix}`);
    if (!elem) return;

    if (typeof metricValue === 'object' && metricValue !== null) {
        elem.innerHTML = createCountCellHTML(metricValue);
    } else {
        elem.innerText = metricValue;
    }

    elem.classList.add('bg-success', 'text-white');
    setTimeout(() => {
        elem.classList.remove('bg-success', 'text-white');
    }, 2000);
    return true;	
}

eel.expose(updateTrainingStatusOnUI);
function updateTrainingStatusOnUI(datasetName, message) {
    const statusElem = document.getElementById(`dataset-status-${datasetName}`);
    if (!statusElem) return;

    // A more robust check for any message that indicates a final state.
    const isFinalState = /complete|failed|cancelled|error/i.test(message);

    if (isFinalState) {
        // This part is correct: show the final message, then reload the cards.
        statusElem.innerHTML = message;
        statusElem.style.display = 'block';
        
        // Store the ID of the timeout before starting it
        cardRefreshTimeoutId = setTimeout(() => {
            loadInitialDatasetCards();
        }, 3000); 
    } else {
        // For ANY message that is not a final state, we assume it's an
        // in-progress update and should include the Cancel button.
        statusElem.innerHTML = `
            <div class="d-flex justify-content-between align-items-center">
                <span>${message}</span>
                <button class="btn btn-xs btn-outline-danger py-0" onclick="cancelTraining('${datasetName}')">Cancel</button>
            </div>`;
        statusElem.style.display = 'block';
    }
    return true;
}

// Function to call the backend to cancel
function cancelTraining(datasetName) {
    if (confirm(`Are you sure you want to cancel the training task for '${datasetName}'?`)) {
        console.log(`Requesting cancellation for training task: ${datasetName}`);
        eel.cancel_training_task(datasetName)(); // We will create this Eel function next
    }
}

eel.expose(updateDatasetLoadProgress);
function updateDatasetLoadProgress(datasetName, percent) {
    const container = document.getElementById(`progress-container-${datasetName}`);
    const bar = document.getElementById(`progress-bar-${datasetName}`);
    if (!container || !bar) return;

    if (percent < 0) {
        container.style.display = 'none';
    } else if (percent >= 100) {
        bar.style.width = '100%';
        bar.innerText = 'Loaded!';
        setTimeout(() => {
            container.style.display = 'none';
            bar.style.width = '0%';
            bar.innerText = '';
        }, 1500);
    } else {
        container.style.display = 'block';
        const displayPercent = Math.round(percent);
        bar.style.width = `${displayPercent}%`;
        bar.innerText = `Processing: ${displayPercent}%`;
    }
}

eel.expose(buildLabelingUI);
function buildLabelingUI(behaviors, colors) {
    labelingInterfaceActive = true;
    const controlsContainer = document.getElementById('controls');
    if (!controlsContainer) return;

    let controlsHTML = `<div class="card bg-dark text-light h-100"><div class="card-header"><h5></h5></div><ul class="list-group list-group-flush">`;
    controlsHTML += `<li class="list-group-item bg-dark text-light d-flex justify-content-between"><strong>Behavior</strong><span><strong>Confidence</strong></span><span><strong>Key</strong></span><span><strong>Count</strong></span></li>`;

    if (behaviors && colors && behaviors.length === colors.length) {
        behaviors.forEach((behaviorName, index) => {
            const key = (index < 9) ? (index + 1) : String.fromCharCode('a'.charCodeAt(0) + (index - 9));
            const bgColor = colors[index];
            const textColor = getTextColorForBg(bgColor);
            controlsHTML += `
                <li class="list-group-item bg-dark text-light d-flex justify-content-between align-items-center" 
                    id="behavior-row-${behaviorName.replace(/[\W_]+/g, '-')}"
                    onclick="eel.label_frame(${index})()" style="cursor: pointer;" title="Click or press '${key}' to label '${behaviorName}'">
                    <span style="flex-basis: 40%;">${behaviorName}</span>
                    <span class="confidence-badge-placeholder" style="flex-basis: 20%;"></span> 
                    <span class="badge rounded-pill" style="flex-basis: 15%; background-color: ${bgColor}; color: ${textColor};">${key}</span>
                    <span id="controls-${behaviorName}-count" class="badge bg-secondary rounded-pill" style="flex-basis: 25%;">0 / 0</span>
                </li>`;
        });
    }
    controlsHTML += `</ul></div>`;
    controlsContainer.innerHTML = controlsHTML;

    document.getElementById('datasets').style.display = 'none';
    document.getElementById('label').style.display = 'flex';
    document.getElementById('labeling-cheat-sheet').style.display = 'block';
	
    const confidenceSlider = document.getElementById('confidence-slider');
    const sliderValueDisplay = document.getElementById('slider-value-display');
    if (confidenceSlider && sliderValueDisplay) {
        confidenceSlider.value = 100;
        sliderValueDisplay.textContent = '100%';
    }	
    return true;	
}

eel.expose(setLabelingModeUI);
function setLabelingModeUI(mode, modelName = '') {
    // Set the global state variable for later use
    currentLabelingMode = mode;

    const controlsHeader = document.querySelector('#controls .card-header');
    const cheatSheet = document.getElementById('labeling-cheat-sheet');
    const saveBtn = document.getElementById('save-labels-btn'); // Get the save button

    if (!controlsHeader || !cheatSheet || !saveBtn) return true;

    if (mode === 'review') {
        controlsHeader.classList.remove('bg-dark');
        controlsHeader.classList.add('bg-success');
        controlsHeader.querySelector('h5').innerHTML = `Reviewing: <span class="badge bg-light text-dark">${modelName}</span>`;
        
        cheatSheet.innerHTML = `
            <div class="card bg-dark">
              <div class="card-header d-flex justify-content-between align-items-center">
                <h5 class="mb-0 text-success"><i class="bi bi-robot me-2"></i>Review Mode Controls</h5>
                <div id="confidence-filter-container" class="d-flex align-items-center w-50">
                  <label for="confidence-slider" class="form-label me-3 mb-0 text-nowrap text-light">Filter Confidence < </label>
                  <input type="range" class="form-range" min="0" max="100" step="1" value="100" id="confidence-slider">
                  <span id="slider-value-display" class="ms-3 badge bg-info" style="width: 55px;">100%</span>
                  <button id="reset-slider-btn" class="btn btn-sm btn-outline-secondary ms-2">Reset</button>
                </div>
              </div>
              <div class="card-body" style="font-size: 0.9rem;">
                <div class="row">
                  <div class="col-md-6">
                    <ul class="list-unstyled">
                      <li><kbd>Tab</kbd> / <kbd>Shift+Tab</kbd> : Next/Prev Instance</li>
                      <li><kbd>←</kbd> / <kbd>→</kbd> : Step one frame</li>
                      <li><kbd>[</kbd> / <kbd>]</kbd> : Set Start/End of Instance</li>
                      <li><kbd>Enter</kbd> : Confirm / Lock / Unlock Instance</li>
                    </ul>
                  </div>
                  <div class="col-md-6">
                    <ul class="list-unstyled">
                        <li><kbd>1</kbd> - <kbd>9</kbd> : Change Instance Label</li>
                        <li><kbd>Delete</kbd> : Delete instance at current frame</li>
                        <li><kbd>Backspace</kbd> : Undo last added label</li>
                        <li><kbd>Ctrl</kbd> + <kbd>S</kbd> : Commit Corrections</li>
                        <li><kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>←</kbd>/<kbd>→</kbd> : Prev/Next video</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>`;
        // Set button text for review mode
        saveBtn.innerHTML = '<i class="bi bi-save-fill me-2"></i>Commit Corrections';
    } else { // 'scratch' mode
        controlsHeader.classList.remove('bg-success');
        controlsHeader.classList.add('bg-dark');
        controlsHeader.querySelector('h5').innerHTML = `Behaviors (Click to label)`;
        
        cheatSheet.innerHTML = `
            <div class="card bg-dark">
              <div class="card-header"><h5>Labeling Controls</h5></div>
              <div class="card-body" style="font-size: 0.9rem;">
                <div class="row">
                  <div class="col-md-6">
                    <ul class="list-unstyled">
                      <li><kbd>←</kbd> / <kbd>→</kbd> : Step one frame</li>
                      <li><kbd>↑</kbd> / <kbd>↓</kbd> : Double / Halve scrub speed</li>
                      <li><kbd>Click Timeline</kbd> : Jump to frame</li>
                      <li><kbd>Ctrl</kbd> + <kbd>S</kbd> : Save Labels</li>
                    </ul>
                  </div>
                  <div class="col-md-6">
                    <ul class="list-unstyled">
                      <li><kbd>1</kbd> - <kbd>9</kbd> : Start / End a new label</li>
                      <li><kbd>Delete</kbd> : Delete instance at current frame</li>
                      <li><kbd>Backspace</kbd> : Undo last added label</li>
                      <li><kbd>Ctrl</kbd>+<kbd>Shift</kbd>+<kbd>←</kbd>/<kbd>→</kbd> : Prev/Next video</li>
                    </ul>
                  </div>
                </div>
              </div>
            </div>`;
        // Set button text for scratch/manual mode
        saveBtn.innerHTML = '<i class="bi bi-save-fill me-2"></i>Save New Labels';       
    }

    // This part for the slider remains the same and is correct.
    const confidenceSlider = document.getElementById('confidence-slider');
    const sliderValueDisplay = document.getElementById('slider-value-display');
    const resetSliderBtn = document.getElementById('reset-slider-btn');
    const timelineContainer = document.getElementById('full-timeline-section');
    if (confidenceSlider && sliderValueDisplay) {
        confidenceSlider.addEventListener('input', function() {
            sliderValueDisplay.textContent = `${this.value}%`;
            if (timelineContainer) {
                timelineContainer.classList.toggle('timeline-filtered', parseInt(this.value) < 100);
            }
            clearTimeout(confidenceFilterDebounceTimer);
            confidenceFilterDebounceTimer = setTimeout(() => {
                eel.refilter_instances(parseInt(this.value))();
            }, 400);
        });
    }
    if (resetSliderBtn) {
        resetSliderBtn.addEventListener('click', function() {
            if (confidenceSlider && sliderValueDisplay) {
                confidenceSlider.value = 100;
                sliderValueDisplay.textContent = '100%';
                if (timelineContainer) {
                    timelineContainer.classList.remove('timeline-filtered');
                }
                eel.refilter_instances(100)();
            }
        });
    }
    return true;
}


eel.expose(highlightBehaviorRow);
function highlightBehaviorRow(behaviorNameToHighlight) {
    document.querySelectorAll('#controls .list-group-item').forEach(row => {
        row.classList.remove('highlight-selected');
    });
    if (behaviorNameToHighlight) {
        const behaviorSpans = document.querySelectorAll('#controls .list-group-item span:first-child');
        const targetSpan = Array.from(behaviorSpans).find(span => span.textContent === behaviorNameToHighlight);
        targetSpan?.closest('.list-group-item')?.classList.add('highlight-selected');
    }
    return true;	
}

eel.expose(updateConfidenceBadge);
function updateConfidenceBadge(behaviorName, confidence) {
    document.querySelectorAll('.confidence-badge-placeholder').forEach(el => el.innerHTML = '');

    if (behaviorName && confidence !== null && typeof confidence !== 'undefined') {
        const rowId = `behavior-row-${behaviorName.replace(/[\W_]+/g, '-')}`;
        const row = document.getElementById(rowId);
        if (!row) return;

        const placeholder = row.querySelector('.confidence-badge-placeholder');
        if (!placeholder) return;

        let badgeClass = 'bg-danger';
        if (confidence >= 0.9) badgeClass = 'bg-success';
        else if (confidence >= 0.7) badgeClass = 'bg-warning text-dark';

        const confidencePercent = (confidence * 100).toFixed(0);
        placeholder.innerHTML = `<span class="badge ${badgeClass}">${confidencePercent}%</span>`;
    }
    return true;	
}

eel.expose(setConfirmationModeUI);
function setConfirmationModeUI(isConfirming) {
    const commitBtn = document.getElementById('save-labels-btn');
    const cancelBtn = document.getElementById('cancel-commit-btn');
    if (!commitBtn || !cancelBtn) return;

    if (isConfirming) {
        commitBtn.innerHTML = '<i class="bi bi-check-circle-fill me-2"></i>Confirm & Save';
        commitBtn.classList.replace('btn-success', 'btn-primary');
        cancelBtn.style.display = 'inline-block';
    } else {
        // Use the state variable to reset the text properly.
        if (currentLabelingMode === 'review') {
            commitBtn.innerHTML = '<i class="bi bi-save-fill me-2"></i>Commit Corrections';
        } else {
            commitBtn.innerHTML = '<i class="bi bi-save-fill me-2"></i>Save New Labels';
        }
        commitBtn.classList.replace('btn-primary', 'btn-success');
        cancelBtn.style.display = 'none';
    }
    return true;
}


// =================================================================
// UI INTERACTION & EVENT HANDLERS
// =================================================================

function jumpToFrame() {
    const input = document.getElementById('frame-jump-input');
    if (input && input.value) {
        const frameNum = parseInt(input.value);
        if (!isNaN(frameNum)) eel.jump_to_frame(frameNum)();
    }
}

function handleCommitClick() {
    const commitBtn = document.getElementById('save-labels-btn');
    const isConfirming = commitBtn.innerText.includes("Confirm & Save");

    // Define messages based on the current mode
    const confirmationMessage = currentLabelingMode === 'review' 
        ? "Are you sure you want to commit these verified labels? This will overwrite previous labels for this video file."
        : "Are you sure you want to save these new labels? This will overwrite any previous labels for this video file.";

    if (isConfirming) {
        if (confirm(confirmationMessage)) {
            eel.save_session_labels()();
            
            commitBtn.innerHTML = '<i class="bi bi-check-lg"></i> Saved!';
            commitBtn.classList.add('btn-info');
            commitBtn.classList.remove('btn-primary');
            setTimeout(() => {
                document.getElementById('label').style.display = 'none';
                document.getElementById('labeling-cheat-sheet').style.display = 'none';
                document.getElementById('datasets').style.display = 'block';
                labelingInterfaceActive = false;
                loadInitialDatasetCards(); 
            }, 1500);
        }
    } else {
        eel.stage_for_commit()();
    }
}

function jumpToInstance(direction) {
    eel.jump_to_instance(direction)();
}

async function startAugmentation(sourceDatasetName, newDatasetName) {
    if (!sourceDatasetName || !newDatasetName) return;

    augmentDatasetBsModal.hide();
    document.getElementById('cover-spin').style.visibility = 'visible';
    update_log_panel(`Starting augmentation for dataset '${sourceDatasetName}'. A new dataset will be created at '${newDatasetName}'.`);

    try {
        await eel.create_augmented_dataset(sourceDatasetName, newDatasetName)();
    } catch (error) {
        showErrorOnLabelTrainPage("An error occurred while trying to start the augmentation task: " + error.message);
        document.getElementById('cover-spin').style.visibility = 'hidden';
    }
}

async function showPreLabelOptions(datasetName) {
    document.getElementById('pl-dataset-name').innerText = datasetName;
    const videoSelect = document.getElementById('pl-video-select');
    videoSelect.innerHTML = '<option selected disabled>Loading videos...</option>';
    
    const videos = await eel.get_videos_for_dataset(datasetName)();
    let videoOptions = '<option selected disabled>Choose a video...</option>';
    if (videos?.length) {
        videos.forEach(v => videoOptions += `<option value="${v[0]}">${v[1]}</option>`);
    } else {
        videoOptions = '<option>No videos found in dataset</option>';
    }
    videoSelect.innerHTML = videoOptions;

    preLabelBsModal?.show();
}

async function onModelSelectChange(event) {
    const modelName = event.target.value;
    const datasetName = document.getElementById('pl-dataset-name').innerText;
    
    const infoDiv = document.getElementById('pl-behavior-match-info');
    infoDiv.innerHTML = '';
    try {
        const [allModelConfigs, allDatasetConfigs] = await Promise.all([eel.get_model_configs()(), eel.load_dataset_configs()()]);
        const targetBehaviors = new Set(allDatasetConfigs[datasetName].behaviors);
        const modelConfig = allModelConfigs[modelName];

        if (modelConfig?.behaviors) {
            const modelBehaviors = new Set(modelConfig.behaviors);
            const matching = [...targetBehaviors].filter(b => modelBehaviors.has(b)).join(', ');
            const nonMatching = [...targetBehaviors].filter(b => !modelBehaviors.has(b)).join(', ');
            infoDiv.innerHTML = `Will pre-label for: <strong>${matching}</strong>.`;
            if (nonMatching) infoDiv.innerHTML += `<br><span class="text-warning">Will ignore: ${nonMatching}</span>`;
        }
    } catch(e) { console.error(e); }

    const sessionSelect = document.getElementById('pl-session-select');
    sessionSelect.disabled = true;
    sessionSelect.innerHTML = '<option>Loading sessions...</option>';
    const sessions = await eel.get_inferred_session_dirs(datasetName, modelName)();
    
    sessionSelect.innerHTML = '<option selected disabled>Choose a session...</option>';
    if (sessions?.length > 0) {
        sessions.forEach(dir => sessionSelect.innerHTML += `<option value="${dir}">${dir}</option>`);
        sessionSelect.disabled = false;
    } else {
        sessionSelect.innerHTML = '<option selected disabled>No inferred sessions</option>';
    }
}

async function onSessionSelectChange(event) {
    const sessionDir = event.target.value;
    const modelName = document.getElementById('pl-model-select').value;
    const videoSelect = document.getElementById('pl-video-select');

    videoSelect.disabled = true;
    videoSelect.innerHTML = '<option>Loading videos...</option>';
    const videos = await eel.get_inferred_videos_for_session(sessionDir, modelName)();

    videoSelect.innerHTML = '<option selected disabled>Choose a video...</option>';
    if (videos?.length > 0) {
        videos.forEach(v => videoSelect.innerHTML += `<option value="${v[0]}">${v[1]}</option>`);
        videoSelect.disabled = false;
    } else {
        videoSelect.innerHTML = '<option selected disabled>No videos found</option>';
    }
}

eel.expose(refreshAllDatasets);
function refreshAllDatasets() {
    console.log("Refreshing datasets from disk...");
    document.getElementById('cover-spin').style.visibility = 'visible';
    
    // Call the backend to reload data. This is a "fire and forget" call from JS.
    eel.reload_project_data()().then(() => {
        // This 'then' block executes AFTER the Python function returns.
        // Now we can safely reload the UI cards.
        loadInitialDatasetCards().then(() => {
            // After the cards are loaded, hide the spinner.
            document.getElementById('cover-spin').style.visibility = 'hidden';
        });
    }).catch(error => {
        // If anything goes wrong, make sure to hide the spinner.
        console.error("Failed to refresh datasets:", error);
        showErrorOnLabelTrainPage("An error occurred while trying to refresh the datasets.");
        document.getElementById('cover-spin').style.visibility = 'hidden';
    });
    return true;	
}

async function showVideoSelectionForScratch() {
    const datasetName = document.getElementById('pl-dataset-name').innerText;
    document.querySelector('label[for="pl-model-select"]').style.display = 'none';
    document.getElementById('pl-model-select').style.display = 'none';
    document.querySelector('label[for="pl-session-select"]').style.display = 'none';
    document.getElementById('pl-session-select').style.display = 'none';
    document.querySelector('#preLabelModal .btn-outline-primary').style.display = 'none';
    document.querySelector('#preLabelModal p.text-center').style.display = 'none';

    const mainButton = document.querySelector('#preLabelModal .btn-outline-success');
    mainButton.innerHTML = '<i class="bi bi-pen me-2"></i>Label Selected Video';
    mainButton.onclick = function() {
        const videoPath = document.getElementById('pl-video-select').value;
        if (!videoPath || videoPath.includes("...")) {
            showErrorOnLabelTrainPage("Please select a video to label.");
            return;
        }
        preLabelBsModal?.hide();
        
        // Before showing the label modal, ensure the button text is correct for scratch mode.
        const saveBtn = document.getElementById('save-labels-btn');
        if (saveBtn) {
            saveBtn.innerHTML = '<i class="bi bi-save-fill me-2"></i>Save New Labels';
        }
        
        prepareAndShowLabelModal(datasetName, videoPath);
    };
    
    const videoSelect = document.getElementById('pl-video-select');
    videoSelect.disabled = false;
    videoSelect.innerHTML = '<option>Loading videos...</option>';
    const videos = await eel.get_videos_for_dataset(datasetName)();
    videoSelect.innerHTML = '<option selected disabled>Choose a video...</option>';
    if (videos?.length > 0) {
        videos.forEach(v => videoSelect.innerHTML += `<option value="${v[0]}">${v[1]}</option>`);
    } else {
        videoSelect.innerHTML = '<option>No videos found in dataset</option>';
    }
}

async function startPreLabeling() {
    const datasetName = document.getElementById('pl-dataset-name').innerText;
    const modelName = document.getElementById('pl-model-select').value;
    const videoPath = document.getElementById('pl-video-select').value;

    // --- Get the value from the smoothing window input ---
    const smoothingWindow = parseInt(document.getElementById('pl-smoothing-window').value) || 1;

    if (!modelName || modelName.includes("...")) { showErrorOnLabelTrainPage("Please select a model."); return; }
    if (!videoPath || videoPath.includes("...")) { showErrorOnLabelTrainPage("Please select a video."); return; }

    preLabelBsModal?.hide();
    document.getElementById('cover-spin').style.visibility = 'visible';

    try {
        // --- Pass the new smoothingWindow value to the backend ---
        const success = await eel.start_labeling_with_preload(datasetName, modelName, videoPath, smoothingWindow)();
        if (!success) showErrorOnLabelTrainPage("Pre-labeling failed. The backend task could not be started.");
    } catch (e) {
        showErrorOnLabelTrainPage(`An error occurred: ${e.message || e}`);
    } finally {
        document.getElementById('cover-spin').style.visibility = 'hidden';
    }
}

async function prepareAndShowLabelModal(datasetName, videoToOpen) {
    try {
        // This function is now only used for scratch/edit, so always set mode to 'scratch'
        setLabelingModeUI('scratch'); 
        
        const success = await eel.start_labeling(datasetName, videoToOpen, null)();
        if (!success) showErrorOnLabelTrainPage('Backend failed to start the labeling task.');
    } catch (error) {
        showErrorOnLabelTrainPage(`Error initializing labeling interface: ${error.message || 'Unknown error'}`);
    }
}

async function showImportVideosDialog() {
    if (!window.electronAPI) {
        showErrorOnLabelTrainPage("The file system API is not available.");
        return;
    }
    try {
        const filePaths = await window.electronAPI.invoke('show-open-video-dialog');
        if (filePaths?.length > 0) {
            selectedVideoPathsForImport = filePaths;
            document.getElementById('import-file-count').textContent = filePaths.length;
            document.getElementById('import-session-name').value = '';
            document.getElementById('import-subject-name').value = '';
            
            const sessionDatalist = document.getElementById('session-names-list');
            if (sessionDatalist) {
                sessionDatalist.innerHTML = ''; 
                const sessionNames = await eel.get_existing_session_names()();
                sessionNames.forEach(name => {
                    const option = document.createElement('option');
                    option.value = name;
                    sessionDatalist.appendChild(option);
                });
            }
            importVideosBsModal?.show();
        } else {
            console.log("User cancelled video selection.");
        }
    } catch (err) {
        showErrorOnLabelTrainPage("Could not open file dialog: " + err.message);
    }
}

async function handleImportSubmit() {
    const sessionName = document.getElementById('import-session-name').value;
    const subjectName = document.getElementById('import-subject-name').value;
	const shouldStandardize = document.getElementById('standardize-video-toggle').checked;

	
    if (!sessionName.trim() || !subjectName.trim()) {
        showErrorOnLabelTrainPage("Session Name and Subject Name cannot be empty.");
        return;
    }
    if (selectedVideoPathsForImport.length === 0) {
        showErrorOnLabelTrainPage("No video files were selected.");
        return;
    }

    importVideosBsModal?.hide();
    document.getElementById('cover-spin').style.visibility = 'visible';

    try {
        // Pass the new boolean flag to the backend
        await eel.import_videos(sessionName, subjectName, selectedVideoPathsForImport, shouldStandardize)();
    } catch (error) {
        showErrorOnLabelTrainPage("An error occurred while trying to start the import task: " + error.message);
        document.getElementById('cover-spin').style.visibility = 'hidden';
    }
}
async function showAddDatasetModal() {
    try {
        const treeContainer = document.getElementById('ad-recording-tree');
        treeContainer.innerHTML = '';
        const fetchedRecordingTree = await eel.get_record_tree()();
        recordingDirTree = fetchedRecordingTree || {};
        if (fetchedRecordingTree && Object.keys(fetchedRecordingTree).length > 0) {
            for (const dateDir in fetchedRecordingTree) {
                let dateHTML = `<div class="form-check"><input class="form-check-input" type="checkbox" id="${dateDir}" onchange="updateChildrenCheckboxes('${dateDir}')"><label class="form-check-label" for="${dateDir}">${dateDir}</label></div>`;
                let sessionsHTML = "<div style='margin-left:20px'>";
                fetchedRecordingTree[dateDir].forEach(sessionDir => {
                    sessionsHTML += `<div class="form-check"><input class="form-check-input" type="checkbox" id="${dateDir}-${sessionDir}"><label class="form-check-label" for="${dateDir}-${sessionDir}">${sessionDir}</label></div>`;
                });
                sessionsHTML += `</div>`;
                treeContainer.innerHTML += dateHTML + sessionsHTML;
            }
            addDatasetBsModal?.show();
        } else {
            showErrorOnLabelTrainPage('No recordings found to create a dataset from.');
        }
    } catch (e) {
        showErrorOnLabelTrainPage("Failed to load recording tree.");
    }
}

async function submitCreateDataset() {
    const selectedRecordings = [];
    Object.keys(recordingDirTree).forEach(dir => {
        const dirCheckbox = document.getElementById(dir);
        if (dirCheckbox?.checked) {
            selectedRecordings.push(dir);
        } else {
            recordingDirTree[dir]?.forEach(subdir => {
                const subdirCheckbox = document.getElementById(`${dir}-${subdir}`);
                if (subdirCheckbox?.checked) {
                    selectedRecordings.push(`${dir}/${subdir}`);
                }
            });
        }
    });

    if (selectedRecordings.length === 0 && !confirm("No recordings selected. Create dataset with empty whitelist?")) return;

    const name = document.getElementById('dataset-name-modal-input').value;
    const behaviorsStr = document.getElementById('dataset-behaviors-modal-input').value;
    if (!name.trim() || !behaviorsStr.trim()) { showErrorOnLabelTrainPage("Name and behaviors are required."); return; }
    const behavior_array = behaviorsStr.split(';').map(b => b.trim()).filter(Boolean);
    if (behavior_array.length < 1 || behavior_array.length > 20) { showErrorOnLabelTrainPage('Must have between 1 and 20 behaviors.'); return; }

    const success = await eel.create_dataset(name, behavior_array, selectedRecordings)();
    if (success) {
        addDatasetBsModal?.hide();
        document.getElementById('dataset-name-modal-input').value = '';
        document.getElementById('dataset-behaviors-modal-input').value = '';
        loadInitialDatasetCards();
    } else {
        showErrorOnLabelTrainPage('Failed to create dataset. A dataset with that name may already exist.');
    }
}

function showTrainModal(datasetName) {
    const tmDatasetElement = document.getElementById('tm-dataset');
    if (tmDatasetElement) tmDatasetElement.innerText = datasetName;
    trainBsModal?.show();
}

async function submitTrainModel() {
    const datasetName = document.getElementById('tm-dataset').innerText;
    const batchSize = document.getElementById('tm-batchsize').value;
    const seqLen = document.getElementById('tm-seqlen').value;
    const learningRate = document.getElementById('tm-lrate').value;
    const epochsCount = document.getElementById('tm-epochs').value;
    const trainMethod = document.getElementById('tm-method').value;
    const patience = document.getElementById('tm-patience').value;

    if (!batchSize || !seqLen || !learningRate || !epochsCount || !patience) {
        showErrorOnLabelTrainPage("All training parameters must be filled.");
        return;
    }
    
    // Immediately hide the modal so the user knows their action was accepted.
    trainBsModal?.hide();

    // If there is a pending card refresh from a previous run, cancel it.
    // This prevents the old run's refresh from wiping out the UI state of this new run.
    if (cardRefreshTimeoutId) {
        clearTimeout(cardRefreshTimeoutId);
        cardRefreshTimeoutId = null;
        console.log("Cancelled a pending UI refresh to start a new training task.");
    }

    // Immediately update the UI to show the new task has been queued.
    updateTrainingStatusOnUI(datasetName, "Training task queued...");
    
    // Call the backend to start the training process. This is a non-blocking,
    // "fire-and-forget" call. The UI thread is now free.
    eel.train_model(datasetName, batchSize, learningRate, epochsCount, seqLen, trainMethod, patience)();
}

async function showInferenceModal(datasetName) {
    const imDatasetElem = document.getElementById('im-dataset');
    if (imDatasetElem) imDatasetElem.innerText = datasetName;
    
    const treeContainer = document.getElementById('im-recording-tree');
    treeContainer.innerHTML = '';
    const fetchedRecordingTree = await eel.get_record_tree()();
    recordingDirTree = fetchedRecordingTree || {};
    
    if (fetchedRecordingTree && Object.keys(fetchedRecordingTree).length > 0) {
        for (const dateDir in fetchedRecordingTree) {
            let dateHTML = `<div class="form-check"><input class="form-check-input" type="checkbox" id="${dateDir}-im" onchange="updateChildrenCheckboxes('${dateDir}-im', true)"><label class="form-check-label" for="${dateDir}-im">${dateDir}</label></div>`;
            let sessionsHTML = "<div style='margin-left:20px'>";
            fetchedRecordingTree[dateDir].forEach(sessionDir => {
                sessionsHTML += `<div class="form-check"><input class="form-check-input" type="checkbox" id="${dateDir}-${sessionDir}-im"><label class="form-check-label" for="${dateDir}-${sessionDir}-im">${sessionDir}</label></div>`;
            });
            sessionsHTML += `</div>`;
            treeContainer.innerHTML += dateHTML + sessionsHTML;
        }
        inferenceBsModal?.show();
    } else {
        showErrorOnLabelTrainPage('No recordings found to run inference on.');
    }
}

async function submitStartClassification() {
    const datasetNameForModel = document.getElementById('im-dataset').innerText;
    const selectedRecs = [];
    Object.keys(recordingDirTree).forEach(dir => {
        const dirCheckbox = document.getElementById(`${dir}-im`);
        if (dirCheckbox?.checked) {
            selectedRecs.push(dir);
        } else {
            recordingDirTree[dir]?.forEach(subdir => {
                const subdirCheckbox = document.getElementById(`${dir}-${subdir}-im`);
                if (subdirCheckbox?.checked) {
                    selectedRecs.push(`${dir}/${subdir}`);
                }
            });
        }
    });

    if (selectedRecs.length === 0) { showErrorOnLabelTrainPage('No recordings selected for inference.'); return; }
    
    updateTrainingStatusOnUI(datasetNameForModel, "Inference tasks queued...");
    await eel.start_classification(datasetNameForModel, selectedRecs)();
    inferenceBsModal?.hide();
}

/**
 * Fetches and renders the initial list of dataset cards.
 */
async function loadInitialDatasetCards(datasets = null) {
    try {
        if (datasets === null) {
            datasets = await eel.load_dataset_configs()();
        }
        
        const container = document.getElementById('dataset-container');
        if (!container) return;
        container.className = 'row g-3'; // Use g-3 for gutter/spacing
        let htmlContent = '';

        // --- Template for the "JonesLabModel" default card ---
        if (await eel.model_exists("JonesLabModel")()) {
            htmlContent += `
                <div class="col-md-6 col-lg-4 d-flex">
                    <div class="card shadow h-100 flex-fill">
                        <div class="card-header bg-dark text-white">
                            <h5 class="card-title mb-0">JonesLabModel <span class="badge bg-info">Default</span></h5>
                        </div>
                        <div class="card-body d-flex flex-column">
                            <p class="card-text small text-muted mt-auto">Pre-trained model for general inference.</p>
                        </div>
                        <div class="card-footer text-end">
                            <button class="btn btn-sm btn-outline-warning" type="button" onclick="showInferenceModal('JonesLabModel')" data-bs-toggle="tooltip" data-bs-placement="top" title="Use this model to classify unlabeled videos">Infer</button>
                        </div>
                    </div>
                </div>`;
        }

        // --- Loop through user-created datasets ---
        if (datasets) {
            for (const datasetName in datasets) {
                if (datasetName === "JonesLabModel") continue;
                
                const config = datasets[datasetName];
                const state = config.state || 'new'; // Default to 'new' if state is missing
                const behaviors = config.behaviors || [];
                const metrics = config.metrics || {};

                // --- Card Shell ---
                htmlContent += `
                    <div class="col-md-6 col-lg-4 d-flex">
                        <div class="card shadow h-100 flex-fill">
                            <div class="card-header bg-dark text-white">
                                <h5 class="card-title mb-0">
                                    ${datasetName}
                                    ${datasetName.endsWith('_aug') ? '<span class="badge bg-info ms-2">Augmented</span>' : ''}
                                </h5>
                            </div>
                            <div class="card-body d-flex flex-column">`;

                // --- STATE: NEW (Dataset is empty) ---
                htmlContent += `
                    <div class="card-state-view" id="state-view-new-${datasetName}" style="display: ${state === 'new' ? 'flex' : 'none'};">
                        <div class="text-center my-auto">
                            <h6 class="text-muted">Empty Dataset</h6>
                            <p class="small text-muted">Start by labeling your first video.</p>
                            <button class="btn btn-primary" onclick="showPreLabelOptions('${datasetName}')">
                                <i class="bi bi-pen-fill me-2"></i>Label First Video
                            </button>
                        </div>
                    </div>`;

                // --- STATE: LABELED (Has labels, not yet trained) ---
                htmlContent += `
                    <div class="card-state-view" id="state-view-labeled-${datasetName}" style="display: ${state === 'labeled' ? 'flex' : 'none'};">
                        <div>
                            <p class="small text-muted mb-1">Instance Counts (Train/Test):</p>
                            <div class="table-responsive" style="max-height: 150px;">
                                <table class="table table-sm table-hover small">
                                    <tbody>
                                        ${behaviors.map(b => `<tr><td>${b}</td><td class="text-end">${(metrics[b] || {})['Train #'] || '0 (0)'}</td><td class="text-end">${(metrics[b] || {})['Test #'] || '0 (0)'}</td></tr>`).join('')}
                                    </tbody>
                                </table>
                            </div>
                            <div class="mt-3 text-center">
                                <p class="small text-muted">Ready to train, or add more labels.</p>
                                <div class="btn-group">
                                    <button class="btn btn-outline-primary" onclick="showPreLabelOptions('${datasetName}')">Label More</button>
                                    <button class="btn btn-success" onclick="checkAndShowTrainModal('${datasetName}')">Train Model</button>
                                </div>
                            </div>
                        </div>
                    </div>`;
                
                // --- STATE: TRAINED (Has labels and a model) ---
                const metricHeaders = ['Train Inst<br><small>(Frames)</small>', 'Test Inst<br><small>(Frames)</small>', 'Precision', 'Recall', 'F1 Score'];
                htmlContent += `
                    <div class="card-state-view" id="state-view-trained-${datasetName}" style="display: ${state === 'trained' ? 'flex' : 'none'}; font-size: 0.85rem;">
                        <div class="table-responsive">
                            <table class="table table-sm table-hover small">
                                <thead><tr><th>Behavior</th>${metricHeaders.map(h => `<th class="text-center">${h}</th>`).join('')}</tr></thead>
                                <tbody>
                                    ${behaviors.map(b => {
                                        const bMetrics = metrics[b] || {};
                                        return `<tr>
                                            <td>${b}</td>
                                            <td class="text-center">${bMetrics['Train #'] || 'N/A'}</td>
                                            <td class="text-center">${bMetrics['Test #'] || 'N/A'}</td>
                                            <td class="text-center">${bMetrics['Precision'] || 'N/A'}</td>
                                            <td class="text-center">${bMetrics['Recall'] || 'N/A'}</td>
                                            <td class="text-center">${bMetrics['F1 Score'] || 'N/A'}</td>
                                        </tr>`;
                                    }).join('')}
                                </tbody>
                            </table>
                        </div>
                    </div>`;

                // --- Common Elements for all states ---
                htmlContent += `
                    <div class="mt-auto">
                        <div class="progress mt-2" id="progress-container-${datasetName}" style="height: 20px; display: none;">
                            <div class="progress-bar progress-bar-striped progress-bar-animated" id="progress-bar-${datasetName}" role="progressbar" style="width: 0%;"></div>
                        </div>
                        <div id="dataset-status-${datasetName}" class="mt-2 small text-info"></div>
                    </div>`;

                // --- Footer with State-Dependent Buttons ---
                htmlContent += `
                            </div> <!-- End card-body -->
                            <div class="card-footer d-flex justify-content-end align-items-center">
                                <button class="btn btn-sm btn-outline-secondary me-auto" type="button" onclick="showManageDatasetModal('${datasetName}')" data-bs-toggle="tooltip" title="View dataset files"><i class="bi bi-folder2-open"></i> Manage</button>`;
                
                // Augment/Sync buttons
                if (datasetName.endsWith('_aug')) {
                    const sourceName = datasetName.replace('_aug', '');
                    htmlContent += `<button class="btn btn-sm btn-outline-info me-1" type="button" onclick="showSyncModal('${sourceName}', '${datasetName}')" data-bs-toggle="tooltip" title="Re-sync labels from the original '${sourceName}' dataset."><i class="bi bi-arrow-repeat"></i> Sync from Source</button>`;
                } else {
                    htmlContent += `<button class="btn btn-sm btn-outline-info me-1" type="button" onclick="showAugmentModal('${datasetName}')" data-bs-toggle="tooltip" title="Create an augmented version of this dataset."><i class="bi bi-images"></i> Augment</button>`;
                }
                
                // Label button (only for non-augmented datasets and if not in 'new' state)
                if (!datasetName.endsWith('_aug') && state !== 'new') {
                     htmlContent += `<button class="btn btn-sm btn-outline-primary me-1" type="button" onclick="showPreLabelOptions('${datasetName}')">Label</button>`;
                }
                
                // Train button (only if labeled or already trained)
                if (state === 'labeled' || state === 'trained') {
                    const trainText = state === 'trained' ? 'Re-Train' : 'Train';
                    htmlContent += `<button class="btn btn-sm btn-outline-success me-1" type="button" onclick="checkAndShowTrainModal('${datasetName}')">${trainText}</button>`;
                }
                
                // Infer button (only if trained)
                if (state === 'trained') {
                    htmlContent += `<button class="btn btn-sm btn-outline-warning" type="button" onclick="showInferenceModal('${datasetName}')">Infer</button>`;
                }
                
                htmlContent += `</div></div></div>`; // End card, col
            }
        }

        container.innerHTML = htmlContent || "<p class='text-light'>No datasets found. Click '+' to create one.</p>";

        var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });

    } catch (error) {
        console.error("Error loading initial dataset configs:", error);
    }
}

/**
 * Checks if all necessary H5 files for a dataset exist on the backend
 * before showing the training modal. This acts as a pre-flight check
 * to prevent training from starting prematurely after augmentation.
 * @param {string} datasetName - The name of the dataset to check.
 */
async function checkAndShowTrainModal(datasetName) {
    document.getElementById('cover-spin').style.visibility = 'visible';
    try {
        // We will create this new eel function in the backend.
        const [isReady, message] = await eel.check_dataset_files_ready(datasetName)();

        if (isReady) {
            // All files are present, proceed to show the training modal.
            showTrainModal(datasetName);
        } else {
            // Files are missing, show a helpful error message.
            showErrorOnLabelTrainPage(
                `Dataset Not Ready for Training\n\n` +
                `${message}\n\n` +
                `This is normal after augmenting a dataset. Please wait a few moments for the background encoding process to finish and then try again.`
            );
        }
    } catch (error) {
        console.error("Error during pre-training check:", error);
        showErrorOnLabelTrainPage("An error occurred while verifying the dataset.");
    } finally {
        document.getElementById('cover-spin').style.visibility = 'hidden';
    }
}

function handleMouseMoveForLabelScrub(event) {
    const imageElement = event.target;
    if (!imageElement) return;
    const imageRect = imageElement.getBoundingClientRect();
    const x = event.clientX - imageRect.left;
    eel.handle_click_on_label_image(x, 0)?.();
}

function handleMouseUpForLabelScrub() {
    document.removeEventListener('mousemove', handleMouseMoveForLabelScrub);
    document.removeEventListener('mouseup', handleMouseUpForLabelScrub);
}

function updateChildrenCheckboxes(parentCheckboxId, isInfModalSuffix = false) {
    const baseParentDirId = isInfModalSuffix ? parentCheckboxId.slice(0, -3) : parentCheckboxId;
    const subdirs = recordingDirTree[baseParentDirId];
    const parentCheckbox = document.getElementById(parentCheckboxId);
    if (subdirs && parentCheckbox) {
        subdirs.forEach(subdir => {
            const childCheckboxId = isInfModalSuffix ? `${baseParentDirId}-${subdir}-im` : `${baseParentDirId}-${subdir}`;
            const childCheckbox = document.getElementById(childCheckboxId);
            if (childCheckbox) childCheckbox.checked = parentCheckbox.checked;
        });
    }
}

function waitForEelConnection() {
    return new Promise(resolve => {
        if (eel._websocket && eel._websocket.readyState === 1) {
            resolve(); return;
        }
        const interval = setInterval(() => {
            if (eel._websocket && eel._websocket.readyState === 1) {
                clearInterval(interval);
                resolve();
            }
        }, 100);
    });
}


// --- Global Event Listeners ---

document.addEventListener('DOMContentLoaded', async () => {
    await waitForEelConnection();
    loadInitialDatasetCards();

    // --- Listeners for various modal/page buttons ---
    document.getElementById('createDatasetButton')?.addEventListener('click', submitCreateDataset);
    document.getElementById('modal-import-button-final')?.addEventListener('click', handleImportSubmit);
    document.getElementById('trainModelButton')?.addEventListener('click', submitTrainModel);
	document.getElementById('startClassificationButton')?.addEventListener('click', submitStartClassification);
    
    // --- Listeners for the Labeling UI ---
    const fullTimelineElement = document.getElementById('full-timeline-image');
    if (fullTimelineElement) {
        fullTimelineElement.addEventListener('mousedown', function (event) {
            handleMouseMoveForLabelScrub(event);
            document.addEventListener('mousemove', handleMouseMoveForLabelScrub);
            document.addEventListener('mouseup', handleMouseUpForLabelScrub, { once: true });
        });
    }
    const zoomBarImageElement = document.getElementById('zoom-bar-image');
    if (zoomBarImageElement) {
        zoomBarImageElement.addEventListener('mousedown', function (event) {
            get_zoom_range_for_click(event.offsetX); // Immediately jump on click
        });
    }
    
    // --- Logic for the "Labeling Options" Modal ---
    const manualBtn = document.getElementById('manual-label-btn');
    const guidedBtn = document.getElementById('guided-label-btn');
    const guidedOptionsPanel = document.getElementById('guided-options-panel');
    const startPreloadBtn = document.getElementById('start-preload-btn');
    const videoSelectModal = document.getElementById('pl-video-select');
    const modelSelectModal = document.getElementById('pl-model-select');

    if (manualBtn) {
        manualBtn.onclick = () => {
            const videoPath = videoSelectModal.value;
            if (!videoPath || videoSelectModal.selectedOptions[0].disabled) {
                showErrorOnLabelTrainPage("Please select a video first.");
                return;
            }
            const datasetName = document.getElementById('pl-dataset-name').innerText;
            preLabelBsModal.hide();
            prepareAndShowLabelModal(datasetName, videoPath);
        };
    }
    
    if (guidedBtn) {
        guidedBtn.onclick = async () => {
            const videoPath = videoSelectModal.value;
            if (!videoPath || videoSelectModal.selectedOptions[0].disabled) {
                showErrorOnLabelTrainPage("Please select a video first.");
                return;
            }
            guidedOptionsPanel.style.display = 'block';
            
            // Get references to the elements we need to manage
            const smoothingInput = document.getElementById('pl-smoothing-window');
            const prelabelBtn = document.getElementById('start-preload-btn');

            // 1. Immediately disable the controls to prevent interaction during loading
            modelSelectModal.innerHTML = '<option>Loading compatible models...</option>';
            modelSelectModal.disabled = true;
            if (smoothingInput) smoothingInput.disabled = true;
            if (prelabelBtn) prelabelBtn.disabled = true;

            const datasetName = document.getElementById('pl-dataset-name').innerText;
            
            try {
                const [allDatasetConfigs, allModelConfigs, allModels] = await Promise.all([
                    eel.load_dataset_configs()(), eel.get_model_configs()(), eel.get_available_models()()
                ]);
                const targetBehaviors = new Set(allDatasetConfigs[datasetName].behaviors);
                const compatibleModels = allModels.filter(modelName => {
                    const modelConfig = allModelConfigs[modelName];
                    return modelConfig?.behaviors && [...targetBehaviors].some(b => new Set(modelConfig.behaviors).has(b));
                });

                // 2. Populate the dropdown and re-enable controls based on the result
                if (compatibleModels.length > 0) {
                    modelSelectModal.innerHTML = '<option selected disabled>Choose a model...</option>' + compatibleModels.map(name => `<option value="${name}">${name}</option>`).join('');
                    modelSelectModal.disabled = false;
                    if (smoothingInput) smoothingInput.disabled = false;
                    if (prelabelBtn) prelabelBtn.disabled = false; // The button is now useful
                } else {
                    modelSelectModal.innerHTML = '<option selected disabled>No compatible models found</option>';
                    // Keep other controls disabled as there's nothing to do
                }

            } catch (e) {
                modelSelectModal.innerHTML = '<option selected disabled>Error loading models</option>';
                // Keep controls disabled on error
            }
        };
    }

    if (startPreloadBtn) {
        // This is the corrected line.
        startPreloadBtn.onclick = startPreLabeling;
    }
    
    // Reset the modal when it's closed
    preLabelModalElement?.addEventListener('hidden.bs.modal', () => {
        if(guidedOptionsPanel) guidedOptionsPanel.style.display = 'none';
        if(videoSelectModal) videoSelectModal.selectedIndex = 0;
        if(modelSelectModal) modelSelectModal.innerHTML = '';
    });
});

window.addEventListener("keydown", (event) => {
    if (document.querySelector('.modal.show') || !labelingInterfaceActive || document.getElementById('label')?.style.display !== 'flex') return;

    if (event.ctrlKey && !event.shiftKey && event.key.toLowerCase() === 's') {
        event.preventDefault(); 
        document.getElementById('save-labels-btn')?.click();
        return; 
    }

    if (document.activeElement === document.getElementById('frame-jump-input')) {
        if (event.key === 'Enter') jumpToFrame();
        return;
    }
    
    let handled = true;
    if (event.key === "Tab") {
        event.preventDefault();
        jumpToInstance(event.shiftKey ? -1 : 1);
        return;
    }

    switch (event.key) {
        case "ArrowLeft":
            if (event.ctrlKey && event.shiftKey) { eel.next_video(-1)(); }
            else { eel.next_frame(-scrubSpeedMultiplier)(); }
            break;
        case "ArrowRight":
            if (event.ctrlKey && event.shiftKey) { eel.next_video(1)(); }
            else { eel.next_frame(scrubSpeedMultiplier)(); }
            break;
        case "ArrowUp": scrubSpeedMultiplier = Math.min(scrubSpeedMultiplier * 2, 128); break;
        case "ArrowDown": scrubSpeedMultiplier = Math.max(1, Math.trunc(scrubSpeedMultiplier / 2)); break;
        case "Delete": eel.delete_instance_from_buffer()(); break;
        case "Backspace": eel.pop_instance_from_buffer()(); break;
        case "[": eel.update_instance_boundary('start')(); break;
        case "]": eel.update_instance_boundary('end')(); break;
        case "Enter": eel.confirm_selected_instance()(); break;
        default:
            let bIdx = -1;
            const keyNum = parseInt(event.key);
            if (!isNaN(keyNum) && keyNum >= 1 && keyNum <= 9) {
                bIdx = keyNum - 1;
            } else if (event.key.length === 1 && event.key.match(/[a-z]/i)) {
                bIdx = event.key.toLowerCase().charCodeAt(0) - 'a'.charCodeAt(0) + 9;
            }
            
            if (bIdx !== -1) {
                eel.label_frame(bIdx)();
            } else {
                handled = false;
            }
            break;
    }
    if (handled) event.preventDefault();
});