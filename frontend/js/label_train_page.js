/**
 * @file Manages the Label & Train page UI and interactions.
 * @description This file handles dataset listing, model training, inference, and the entire
 * advanced labeling workflow including pre-labeling and instance-based navigation.
 */

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
}


// 2. Find and REPLACE the existing startAugmentation function with this new version.
async function startAugmentation(sourceDatasetName, newDatasetName) {
    if (!sourceDatasetName || !newDatasetName) return;

    augmentDatasetBsModal.hide();
    
    // HIDE the spinner and SHOW the progress bar.
    document.getElementById('cover-spin').style.visibility = 'hidden';
    update_augmentation_progress(0, `Augmenting '${sourceDatasetName}'...`);

    try {
        await eel.create_augmented_dataset(sourceDatasetName, newDatasetName)();
    } catch (error) {
        showErrorOnLabelTrainPage("An error occurred while trying to start the augmentation task: " + error.message);
        // Hide the progress bar on error
        update_augmentation_progress(-1);
    }
}

eel.expose(update_log_panel);
function update_log_panel(message) {
    const logContainer = document.getElementById('log-panel-content');
    if (!logContainer) return;

    let logHistory = JSON.parse(sessionStorage.getItem('logHistory') || '[]');
    logHistory.push(message);
    while (logHistory.length > 500) {
        logHistory.shift();
    }
    sessionStorage.setItem('logHistory', JSON.stringify(logHistory));

    renderLogMessage(message, logContainer);
    logContainer.scrollTop = logContainer.scrollHeight;
}

function renderLogMessage(message, container) {
    const logEntry = document.createElement('div');
    logEntry.className = 'log-message';

    if (message.includes('[ERROR]')) {
        logEntry.classList.add('log-level-ERROR');
    } else if (message.includes('[WARN]')) {
        logEntry.classList.add('log-level-WARN');
    } else {
        logEntry.classList.add('log-level-INFO');
    }
    
    logEntry.textContent = message;
    container.appendChild(logEntry);
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
}

eel.expose(updateFileInfo);
function updateFileInfo(filenameStr) {
    const elem = document.getElementById('file-info');
    if (elem) elem.innerText = filenameStr || "No video loaded";
}

eel.expose(updateLabelingStats);
function updateLabelingStats(behaviorName, instanceCount, frameCount) {
    const elem = document.getElementById(`controls-${behaviorName}-count`);
    if (elem) elem.innerHTML = `${instanceCount} / ${frameCount}`;
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
}

eel.expose(updateTrainingStatusOnUI);
function updateTrainingStatusOnUI(datasetName, message) {
    const statusElem = document.getElementById(`dataset-status-${datasetName}`);
    if (statusElem) {
        // If the message is about training, add a cancel button
        if (message.toLowerCase().includes('training') || message.toLowerCase().includes('loading dataset')) {
            statusElem.innerHTML = `
                <div class="d-flex justify-content-between align-items-center">
                    <span>${message}</span>
                    <button class="btn btn-xs btn-outline-danger py-0" onclick="cancelTraining('${datasetName}')">
                        Cancel
                    </button>
                </div>`;
        } else {
            // For other messages like "complete" or "failed", just show the text
            statusElem.innerHTML = message;
        }
        
        statusElem.style.display = message ? 'block' : 'none';
    }
}

// Add this new function to call the backend to cancel
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
}

eel.expose(setLabelingModeUI);
function setLabelingModeUI(mode, modelName = '') {
    const controlsHeader = document.querySelector('#controls .card-header');
    const cheatSheet = document.getElementById('labeling-cheat-sheet');
    if (!controlsHeader || !cheatSheet) return;

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
                      <li><kbd>Ctrl</kbd> + <kbd>S</kbd> : Commit Corrections</li>
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
    }

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
        commitBtn.innerHTML = '<i class="bi bi-save-fill me-2"></i>Commit Corrections';
        commitBtn.classList.replace('btn-primary', 'btn-success');
        cancelBtn.style.display = 'none';
    }
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
    if (commitBtn.innerText.includes("Commit Corrections")) {
        eel.stage_for_commit()();
    } else {
        if (confirm("Are you sure you want to commit these verified labels? This will overwrite previous labels for this video file.")) {
            eel.save_session_labels()();
        }
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
    const modelSelect = document.getElementById('pl-model-select');
    const sessionSelect = document.getElementById('pl-session-select');
    const videoSelect = document.getElementById('pl-video-select');
    const infoDiv = document.getElementById('pl-behavior-match-info');

    document.getElementById('pl-dataset-name').innerText = datasetName;
    modelSelect.innerHTML = '<option selected disabled>Loading...</option>';
    sessionSelect.innerHTML = '<option selected disabled>First, choose a model...</option>';
    videoSelect.innerHTML = '<option selected disabled>First, choose a session...</option>';
    infoDiv.innerHTML = '';
    [modelSelect, sessionSelect, videoSelect].forEach(el => el.disabled = true);

    try {
        const [allDatasetConfigs, allModelConfigs, allModels] = await Promise.all([
            eel.load_dataset_configs()(),
            eel.get_model_configs()(),
            eel.get_available_models()()
        ]);

        const targetDatasetConfig = allDatasetConfigs[datasetName];
        if (!targetDatasetConfig?.behaviors) {
            showErrorOnLabelTrainPage(`Could not load behaviors for dataset '${datasetName}'.`);
            return;
        }
        const targetBehaviors = new Set(targetDatasetConfig.behaviors);
        
        const compatibleModels = allModels.filter(modelName => {
            const modelConfig = allModelConfigs[modelName];
            if (!modelConfig?.behaviors) return false;
            const modelBehaviors = new Set(modelConfig.behaviors);
            return [...targetBehaviors].some(b => modelBehaviors.has(b));
        });

        if (compatibleModels.length > 0) {
            modelSelect.innerHTML = '<option selected disabled>Choose a model...</option>' + compatibleModels.map(name => `<option value="${name}">${name}</option>`).join('');
            modelSelect.disabled = false;
        } else {
            modelSelect.innerHTML = '<option selected disabled>No compatible models found</option>';
        }
    } catch (e) {
        showErrorOnLabelTrainPage("Error preparing labeling options: " + e.message);
    }

    preLabelBsModal.show();
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
async function refreshAllDatasets() {
    console.log("Refreshing datasets from disk...");
    document.getElementById('cover-spin').style.visibility = 'visible';
    try {
        // 1. Tell Python to reload its data.
        await eel.reload_project_data()();
        // 2. NOW, call loadInitialDatasetCards with no arguments. It will fetch
        //    the newly reloaded data from the backend.
        await loadInitialDatasetCards();
    } catch (error) {
        console.error("Failed to refresh datasets:", error);
        showErrorOnLabelTrainPage("An error occurred while trying to refresh the datasets.");
    } finally {
        document.getElementById('cover-spin').style.visibility = 'hidden';
    }
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
        await eel.import_videos(sessionName, subjectName, selectedVideoPathsForImport)();
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
    const patience = document.getElementById('tm-patience').value; // Get the new value

    if (!batchSize || !seqLen || !learningRate || !epochsCount || !patience) { // Add patience to the check
        showErrorOnLabelTrainPage("All training parameters must be filled."); 
        return; 
    }
    
    updateTrainingStatusOnUI(datasetName, "Training task queued...");
    // Add patience to the eel.train_model call
    await eel.train_model(datasetName, batchSize, learningRate, epochsCount, seqLen, trainMethod, patience)();
    trainBsModal?.hide();
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
        // If no data is passed in, fetch it from the backend as before.
        if (datasets === null) {
            datasets = await eel.load_dataset_configs()();
        }
        
        const container = document.getElementById('dataset-container');
        if (!container) return;
        container.className = 'row g-3';
        let htmlContent = '';

        if (await eel.model_exists("JonesLabModel")()) {
            htmlContent += `
                <div class="col-md-6 col-lg-4">
                    <div class="card shadow h-100">
                        <div class="card-header bg-dark text-white">
                            <h5 class="card-title mb-0">JonesLabModel <span class="badge bg-info">Default</span></h5>
                        </div>
                        <div class="card-body d-flex flex-column">
                            <p class="card-text small text-muted">Pre-trained model for general inference.</p>
                        </div>
                        <div class="card-footer text-end">
                            <button class="btn btn-sm btn-outline-warning" type="button" onclick="showInferenceModal('JonesLabModel')" data-bs-toggle="tooltip" data-bs-placement="top" title="Use this model to classify unlabeled videos">Infer</button>
                        </div>
                    </div>
                </div>`;
        }

        if (datasets) {
            for (const datasetName in datasets) {
                if (datasetName === "JonesLabModel") continue;
                
                const config = datasets[datasetName];
                const behaviors = config.behaviors || [];
                const metrics = config.metrics || {};
                const modelExists = !!config.model;
                const metricHeaders = [
                    'Train Inst<br><small>(Frames)</small>', 
                    'Test Inst<br><small>(Frames)</small>', 
                    'Precision', 'Recall', 'F1 Score'
                ];

                htmlContent += `
                    <div class="col-md-6 col-lg-4">
                        <div class="card shadow h-100">
                            <div class="card-header bg-dark text-white">
                                <h5 class="card-title mb-0">
                                    ${datasetName}
                                    ${datasetName.endsWith('_aug') ? '<span class="badge bg-info ms-2">Augmented</span>' : ''}
                                </h5>
                            </div>
                            <div class="card-body" style="font-size: 0.85rem;">`;

                if (behaviors.length > 0) {
                    htmlContent += `
                        <div class="table-responsive">
                            <table class="table table-sm table-hover small">
                                <thead>
                                    <tr>
                                        <th>Behavior</th>
                                        ${metricHeaders.map(h => `<th class="text-center">${h}</th>`).join('')}
                                    </tr>
                                </thead>
                                <tbody>`;

                    behaviors.forEach(behaviorName => {
                        const bMetrics = metrics[behaviorName] || {};
                        const trainValue = bMetrics['Train #'] ?? 'N/A';
                        const testValue = bMetrics['Test #'] ?? 'N/A';
                        const precisionValue = bMetrics['Precision'] ?? 'N/A';
                        const recallValue = bMetrics['Recall'] ?? 'N/A';
                        const f1Value = bMetrics['F1 Score'] ?? 'N/A';
                        
                        htmlContent += `
                            <tr>
                                <td>${behaviorName}</td>
                                <td class="text-center" id="${datasetName}-${behaviorName}-train-count">${trainValue}</td>
                                <td class="text-center" id="${datasetName}-${behaviorName}-test-count">${testValue}</td>
                                <td class="text-center" id="${datasetName}-${behaviorName}-precision">${precisionValue}</td>
                                <td class="text-center" id="${datasetName}-${behaviorName}-recall">${recallValue}</td>
                                <td class="text-center" id="${datasetName}-${behaviorName}-fscore">${f1Value}</td>
                            </tr>`;
                    });

                    htmlContent += `</tbody></table></div>`;
                } else {
                    htmlContent += `<p class="text-muted">No behaviors defined yet.</p>`;
                }

                htmlContent += `
                    <div class="progress mt-2" id="progress-container-${datasetName}" style="height: 20px; display: none;">
                        <div class="progress-bar progress-bar-striped progress-bar-animated" id="progress-bar-${datasetName}" role="progressbar" style="width: 0%;"></div>
                    </div>
                    <div id="dataset-status-${datasetName}" class="mt-2 small text-info"></div>
                    
					<div id="progress-container-${datasetName}" class="progress mt-2" style="height: 20px; display: none;">
                        <div class="progress-bar progress-bar-striped progress-bar-animated" id="progress-bar-${datasetName}" role="progressbar" style="width: 0%;"></div>
                    </div>					
														
                </div>`;
                
                htmlContent += `
                    <div class="card-footer d-flex justify-content-end align-items-center">
                        <button class="btn btn-sm btn-outline-secondary me-auto" type="button" onclick="showManageDatasetModal('${datasetName}')" data-bs-toggle="tooltip" data-bs-placement="top" title="View dataset files on disk">
                            <i class="bi bi-folder2-open"></i> Manage
                        </button>`;
                
                if (datasetName.endsWith('_aug')) {
                    const sourceName = datasetName.replace('_aug', '');
                    htmlContent += `
                        <button class="btn btn-sm btn-outline-info me-1" type="button" onclick="showSyncModal('${sourceName}', '${datasetName}')" data-bs-toggle="tooltip" data-bs-placement="top" title="Re-sync labels from the original '${sourceName}' dataset.">
                            <i class="bi bi-arrow-repeat"></i> Sync from Source
                        </button>`;
                } else {
                    htmlContent += `
                        <button class="btn btn-sm btn-outline-info me-1" type="button" onclick="showAugmentModal('${datasetName}')" data-bs-toggle="tooltip" data-bs-placement="top" title="Create a new dataset with augmented videos. Required if you only have one labeled video. This creates a virtual 'second video' so the model has data to train on."">
                            <i class="bi bi-images"></i> Augment
                        </button>`;
                }

                if (datasetName.endsWith('_aug')) {
                    // If it's an augmented dataset, wrap the disabled button in a span for the tooltip
                    htmlContent += `
                        <span class="d-inline-block" tabindex="0" data-bs-toggle="tooltip" data-bs-placement="top" title="Labeling should be done on the original source dataset">
                            <button class="btn btn-sm btn-outline-primary me-1" type="button" disabled>
                                Label
                            </button>
                        </span>`;
                } else {
                    // Otherwise, render the normal, enabled button
                    htmlContent += `
                        <button class="btn btn-sm btn-outline-primary me-1" type="button" onclick="showPreLabelOptions('${datasetName}')" data-bs-toggle="tooltip" data-bs-placement="top" title="Label videos for this dataset">
                            Label
                        </button>`;
                }

                // The "Train" button now calls a new validation function
                htmlContent += `
						<button class="btn btn-sm btn-outline-success me-1" type="button" 
								onclick="checkAndShowTrainModal('${datasetName}')" ...>
							Train
						</button>`;
                
                if (modelExists) {
                    htmlContent += `<button class="btn btn-sm btn-outline-warning" type="button" onclick="showInferenceModal('${datasetName}')" data-bs-toggle="tooltip" data-bs-placement="top" title="Use this dataset's trained model to classify unlabeled videos">Infer</button>`;
                }

                htmlContent += `</div></div></div>`;
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
 * Checks if a dataset has any training instances before attempting to show the training modal.
 * If the training set is empty, it shows a helpful error message instead.
 * @param {string} datasetName - The name of the dataset to check.
 */
function checkAndShowTrainModal(datasetName) {
    // Find all the 'Train #' cells for this specific dataset card
    const trainCountCells = document.querySelectorAll(`[id^="${datasetName}-"][id$="-train-count"]`);
    
    let totalTrainInstances = 0;
    trainCountCells.forEach(cell => {
        // The text is formatted like "65 (1950)". We want the first number.
        const instanceCount = parseInt(cell.innerText.split(' ')[0], 10);
        if (!isNaN(instanceCount)) {
            totalTrainInstances += instanceCount;
        }
    });

    // If there are instances to train on, proceed as normal
    if (totalTrainInstances > 0) {
        showTrainModal(datasetName); // This is your original function
    } else {
        // Otherwise, show the specific, actionable error message
        showErrorOnLabelTrainPage(
            `Training Failed: No Training Data\n\n` +
            `CBAS must reserve some labeled data for testing, but you have only labeled one video. This forces the entire video into the test set, leaving nothing to train on.\n\n` +
            `Solution: Click the "Augment" button on the '${datasetName}' card. This will automatically create a second, virtual video, allowing the training to proceed.`
        );
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

    document.getElementById('createDatasetButton')?.addEventListener('click', submitCreateDataset);
    document.getElementById('trainModelButton')?.addEventListener('click', submitTrainModel);
    document.getElementById('startClassificationButton')?.addEventListener('click', submitStartClassification);
    document.getElementById('modal-import-button-final')?.addEventListener('click', handleImportSubmit);
	document.getElementById('pl-model-select')?.addEventListener('change', onModelSelectChange);
    document.getElementById('pl-session-select')?.addEventListener('change', onSessionSelectChange);
	document.getElementById('startAugmentationButton')?.addEventListener('click', startAugmentation);

    const fullTimelineElement = document.getElementById('full-timeline-image');
    if (fullTimelineElement) {
        fullTimelineElement.addEventListener('mousedown', function (event) {
            const x = event.offsetX;
            eel.handle_click_on_label_image(x, 0)?.();
            document.addEventListener('mousemove', handleMouseMoveForLabelScrub);
            document.addEventListener('mouseup', handleMouseUpForLabelScrub);
        });
    }

    const zoomBarImageElement = document.getElementById('zoom-bar-image');
    if (zoomBarImageElement) {
        zoomBarImageElement.addEventListener('mousedown', function (event) {
            const imageRect = event.target.getBoundingClientRect();
            const x = event.clientX - imageRect.left;
            eel.get_zoom_range_for_click(x)();
        });
    }

    const logCollapseElement = document.getElementById('log-panel-collapse');
    if (logCollapseElement) {
        const fabLeft = document.querySelector('.fab-container-left');
        const fabRight = document.querySelector('.fab-container-right');
        const fabUpPosition = `${200 + 45 + 5}px`; 
        const fabDownPosition = '65px';

        logCollapseElement.addEventListener('show.bs.collapse', () => {
            if (fabLeft) fabLeft.style.bottom = fabUpPosition;
            if (fabRight) fabRight.style.bottom = fabUpPosition;
        });
        logCollapseElement.addEventListener('hide.bs.collapse', () => {
            if (fabLeft) fabLeft.style.bottom = fabDownPosition;
            if (fabRight) fabRight.style.bottom = fabDownPosition;
        });
    }
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
            event.ctrlKey && event.shiftKey ? eel.next_video(-1)() : eel.next_frame(-scrubSpeedMultiplier)();
            break;
        case "ArrowRight":
            event.ctrlKey && event.shiftKey ? eel.next_video(1)() : eel.next_frame(scrubSpeedMultiplier)();
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
            if (event.keyCode >= 49 && event.keyCode <= 57) bIdx = event.keyCode - 49;
            else if (event.keyCode >= 65 && event.keyCode <= 90) bIdx = event.keyCode - 65 + 9;
            if (bIdx !== -1) eel.label_frame(bIdx)();
            else handled = false;
            break;
    }
    if (handled) event.preventDefault();
});

window.addEventListener("unload", () => { if (!routingInProgress) eel.kill_streams()?.catch(console.error); });
window.onbeforeunload = () => { if (!routingInProgress) eel.kill_streams()?.catch(console.error); };