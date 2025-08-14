/**
 * @file Manages the Label & Train page UI and interactions.
 * @description This file handles dataset listing, model training, inference, and the entire
 * advanced labeling workflow including pre-labeling and instance-based navigation.
 */

let cardRefreshTimeoutId = null;
let editWhitelistBsModal = document.getElementById('editWhitelistModal') ? new bootstrap.Modal(document.getElementById('editWhitelistModal')) : null;
let conflictResolutionBsModal = document.getElementById('conflictResolutionModal') ? new bootstrap.Modal(document.getElementById('conflictResolutionModal')) : null;
let isReviewByBehaviorMode = false;

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
const cropOnImportModalElement = document.getElementById('cropOnImportModal');
const categoryReviewModalElement = document.getElementById('categoryReviewModal');
const behaviorSelectModalElement = document.getElementById('behaviorSelectModal');
const disagreementReviewModalElement = document.getElementById('disagreementReviewModal');
const labelCoverageModalElement = document.getElementById('labelCoverageModal');

let addDatasetBsModal = addDatasetModalElement ? new bootstrap.Modal(addDatasetModalElement) : null;
let trainBsModal = trainModalElement ? new bootstrap.Modal(trainModalElement) : null;
let inferenceBsModal = inferenceModalElement ? new bootstrap.Modal(inferenceModalElement) : null;
let generalErrorBsModal = errorModalElement ? new bootstrap.Modal(errorModalElement) : null;
let preLabelBsModal = preLabelModalElement ? new bootstrap.Modal(preLabelModalElement) : null;
let importVideosBsModal = importVideosModalElement ? new bootstrap.Modal(importVideosModalElement) : null;
let manageDatasetBsModal = manageDatasetModalElement ? new bootstrap.Modal(manageDatasetModalElement) : null;
let augmentDatasetBsModal = augmentDatasetModalElement ? new bootstrap.Modal(augmentDatasetModalElement) : null;
let syncDatasetBsModal = syncDatasetModalElement ? new bootstrap.Modal(syncDatasetModalElement) : null;
let cropOnImportBsModal = cropOnImportModalElement ? new bootstrap.Modal(cropOnImportModalElement) : null;
let categoryReviewBsModal = categoryReviewModalElement ? new bootstrap.Modal(categoryReviewModalElement) : null;
let behaviorSelectBsModal = behaviorSelectModalElement ? new bootstrap.Modal(behaviorSelectModalElement) : null;
let disagreementReviewBsModal = disagreementReviewModalElement ? new bootstrap.Modal(disagreementReviewModalElement) : null;
let labelCoverageBsModal = labelCoverageModalElement ? new bootstrap.Modal(labelCoverageModalElement) : null;


// --- Import Cropping State ---
let importCropData = { x: 0.0, y: 0.0, w: 1.0, h: 1.0, apply: true, stretch: false };
let importCropCanvas, importCropCtx, importImageCanvas, importImageCtx;
let importCropRect = { x: 0, y: 0, w: 0, h: 0 };
let isImportCropping_Dragging = false;
let isImportCropping_Resizing = false;
let importCrop_ResizeHandle = null;
const importCrop_HandleSize = 8;
let importPreviewImage = new Image();

let currentLabelingMode = 'scratch'; // Default to 'scratch'

// =================================================================
// ROUTING & UTILITY FUNCTIONS
// =================================================================

function routeToRecordPage() { routingInProgress = true; window.location.href = './record.html'; }
function routeToLabelTrainPage() { routingInProgress = true; window.location.href = './label-train.html'; }
function routeToVisualizePage() { routingInProgress = true; window.location.href = './visualize.html'; }
function routeToInferencePage() { window.location.href = './inference.html'; }

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

async function showDisagreementModal(datasetName) {
    if (!disagreementReviewBsModal) return;

    document.getElementById('dr-dataset-name').textContent = datasetName;
    const playlistContainer = document.getElementById('dr-playlist');
    const infoMessageContainer = document.getElementById('dr-info-message');
    playlistContainer.innerHTML = '<li class="list-group-item text-center">Searching for disagreements...</li>';
    
    if (datasetName.endsWith('_aug')) {
        const sourceName = datasetName.replace('_aug', '');
        infoMessageContainer.innerHTML = `<i class="bi bi-info-circle-fill me-2"></i>You are reviewing errors from the <strong>${datasetName}</strong> model. All corrections will be saved to the original <strong>${sourceName}</strong> dataset. Remember to 'Sync from Source' after curating.`;
        infoMessageContainer.style.display = 'block';
    } else {
        infoMessageContainer.style.display = 'none';
    }

    disagreementReviewBsModal.show();

    const disagreements = await eel.get_disagreement_playlist(datasetName)();
    playlistContainer.innerHTML = ''; 

    if (disagreements && disagreements.length > 0) {
        const projectRoot = await eel.get_project_root()();
        
        const reviewedItemsKey = `reviewedDisagreements_${datasetName}`;
        const reviewedItems = JSON.parse(sessionStorage.getItem(reviewedItemsKey) || '[]');

        for (const item of disagreements) {
            const listItem = document.createElement('li');
            listItem.className = 'list-group-item list-group-item-action';
            listItem.style.cursor = 'pointer';

            const videoToOpen = item.video_to_open;
            const correctionDataset = item.correction_dataset;
            const videoAbsPath = `${projectRoot}/${videoToOpen}`.replace(/\\/g, '/');
            
            // First, replace all backslashes with forward slashes to normalize the path.
            // Then, split by the forward slash and pop the last element (the filename).
            const displayPath = item.video_path.replace(/\\/g, '/').split('/').pop();

            const uniqueId = `${item.video_path}_${item.start_frame}`;
            const isReviewed = reviewedItems.includes(uniqueId);

            const reviewedBadge = isReviewed 
                ? '<span class="badge bg-success float-end"><i class="bi bi-check-lg"></i> Reviewed</span>' 
                : '';
            if (isReviewed) {
                listItem.classList.add('text-muted');
            }

            listItem.innerHTML = `
                <div class="d-flex w-100 justify-content-between">
                    <h6 class="mb-1">File: ${displayPath}</h6>
                    <div>
                        <small class="me-3">Confidence in wrong label: <strong>${(item.model_confidence * 100).toFixed(0)}%</strong></small>
                        ${reviewedBadge}
                    </div>
                </div>
                <p class="mb-1">
                    Human Label: <span class="badge bg-success">${item.human_label}</span>
                    <i class="bi bi-arrow-right-short"></i>
                    Model Predicted: <span class="badge bg-danger">${item.model_prediction}</span>
                </p>
                <small>Location: Frames ${item.start_frame} - ${item.end_frame}</small>
            `;

            listItem.onclick = () => {
                if (!reviewedItems.includes(uniqueId)) {
                    reviewedItems.push(uniqueId);
                    sessionStorage.setItem(reviewedItemsKey, JSON.stringify(reviewedItems));
                }
                listItem.classList.add('text-muted');
                const existingBadge = listItem.querySelector('.float-end');
                if (!existingBadge) {
                     const newBadge = document.createElement('span');
                     newBadge.className = 'badge bg-success float-end';
                     newBadge.innerHTML = '<i class="bi bi-check-lg"></i> Reviewed';
                     listItem.querySelector('.d-flex > div').appendChild(newBadge);
                }

                disagreementReviewBsModal.hide();
                const disagreementInfo = {
                    human: item.human_label,
                    model: item.model_prediction
                };
                prepareAndShowLabelModal(correctionDataset, videoAbsPath, null, item.start_frame, disagreementInfo);
            };
            playlistContainer.appendChild(listItem);
        }
    } else {
        playlistContainer.innerHTML = '<li class="list-group-item text-center text-muted">No disagreements found. Your model agrees with all your labels in the training set!</li>';
    }
}

async function showManageDatasetModal(datasetName) {
    if (!manageDatasetBsModal) return;
    document.getElementById('md-dataset-name').innerText = datasetName;

    // Populate the whitelist section
    const whitelistContainer = document.getElementById('md-whitelist-container');
    const editWhitelistBtn = document.getElementById('editWhitelistButton');
    
    if(whitelistContainer) whitelistContainer.innerHTML = 'Loading...'; 

    try {
        const datasets = await eel.load_dataset_configs()();
        const currentWhitelist = datasets[datasetName]?.whitelist || [];
        
        if (whitelistContainer) {
            if (currentWhitelist.length > 0) {
                whitelistContainer.innerHTML = `<ul class="list-unstyled mb-0">${currentWhitelist.map(dir => `<li><small>${dir}</small></li>`).join('')}</ul>`;
            } else {
                whitelistContainer.innerHTML = '<p class="text-muted small m-1">No directories are currently selected.</p>';
            }
        }

        if (editWhitelistBtn) {
            editWhitelistBtn.onclick = () => showEditWhitelistModal(datasetName, currentWhitelist);
        }
    } catch (e) {
        if(whitelistContainer) whitelistContainer.innerHTML = '<p class="text-danger small m-1">Could not load directories.</p>';
        console.error("Error loading whitelist:", e);
    }

    // Attach event for the "Show Files" button
    const revealBtn = document.getElementById('revealFilesButton');
    if (revealBtn) {
        revealBtn.onclick = () => {
            eel.reveal_dataset_files(datasetName)();
            manageDatasetBsModal.hide();
        };
    }

    const cleanAndSortBtn = document.getElementById('cleanAndSortButton');
    if (cleanAndSortBtn) {
        cleanAndSortBtn.onclick = async () => {
            manageDatasetBsModal.hide();
            document.getElementById('cover-spin').style.visibility = 'visible';
            
            const report = await eel.analyze_label_conflicts(datasetName)();
            
            document.getElementById('cover-spin').style.visibility = 'hidden';

            if (report.error) {
                showErrorOnLabelTrainPage(report.error);
                return;
            }

            const modalBody = document.getElementById('cr-modal-body');
            const confirmBtn = document.getElementById('cr-confirm-button');
            
            if (report.total_duplicates === 0 && report.total_overlaps === 0) {
                modalBody.innerHTML = '<p>No conflicts found in your `labels.yaml` file. Your labels are clean!</p>';
                confirmBtn.style.display = 'none';
            } else {
                modalBody.innerHTML = `
                    <div class="alert alert-warning" role="alert">
                        <strong>Warning:</strong> This will permanently modify your <code>labels.yaml</code> file and cannot be undone.
                    </div>
                    <p>Our analysis found the following issues:</p>
                    <ul>
                        <li><strong>${report.total_duplicates}</strong> exact duplicate label(s). These will be removed.</li>
                        <li><strong>${report.total_overlaps}</strong> overlapping label(s). These will be resolved by trimming overlapping instances.</li>
                    </ul>
                    <p>After cleanup, the entire file will be sorted for clarity.</p>
                    <p>Are you sure you want to proceed?</p>
                `;
                confirmBtn.style.display = 'block';
                
                confirmBtn.onclick = async () => {
                    conflictResolutionBsModal.hide();
                    document.getElementById('cover-spin').style.visibility = 'visible';
                    const success = await eel.clean_and_sort_labels(datasetName)();
                    if (success) {
                        alert('Your labels.yaml file has been successfully cleaned and sorted.');
                        refreshAllDatasets();
                    } else {
                        showErrorOnLabelTrainPage('An error occurred during the cleanup process. Your file has not been modified.');
                    }
                    document.getElementById('cover-spin').style.visibility = 'hidden';
                };
            }
            
            if (conflictResolutionBsModal) {
                conflictResolutionBsModal.show();
            }
        };
    }

    // Attach event for the "Label Coverage Report" button
    const labelCoverageBtn = document.getElementById('labelCoverageButton');
    if (labelCoverageBtn) {
        labelCoverageBtn.onclick = () => showLabelCoverageModal(datasetName);
    }

    const recalcBtn = document.getElementById('recalculateStatsButton');
    if (recalcBtn) {
        recalcBtn.onclick = async () => {
            if (confirm(`Are you sure you want to recalculate stats for '${datasetName}'? This will reset your performance metrics to 'N/A'.`)) {
                manageDatasetBsModal.hide();
                document.getElementById('cover-spin').style.visibility = 'visible';
                try {
                    const success = await eel.recalculate_dataset_stats(datasetName)();
                    if (success) {
                        refreshAllDatasets();
                    } else {
                        showErrorOnLabelTrainPage("Failed to recalculate stats on the backend.");
                    }
                } catch (e) {
                    showErrorOnLabelTrainPage(`An error occurred: ${e.message}`);
                } finally {
                    document.getElementById('cover-spin').style.visibility = 'hidden';
                }
            }
        };
    }
    
    const deleteBtn = document.getElementById('deleteDatasetButton');
    if (deleteBtn) {
        deleteBtn.onclick = async () => {
            const confirmationMessage = `Are you sure you want to permanently delete the '${datasetName}' dataset and its trained model?\n\nThis action cannot be undone.`;
            if (confirm(confirmationMessage)) {
                document.getElementById('cover-spin').style.visibility = 'visible';
                try {
                    const success = await eel.delete_dataset(datasetName)();
                    if (success) {
                        manageDatasetBsModal.hide();
                        refreshAllDatasets();
                    } else {
                        showErrorOnLabelTrainPage(`Failed to delete dataset '${datasetName}'. Check the logs for more information.`);
                    }
                } catch (e) {
                    showErrorOnLabelTrainPage(`An error occurred: ${e.message}`);
                } finally {
                    document.getElementById('cover-spin').style.visibility = 'hidden';
                }
            }
        };
    }

    const reviewBtn = document.getElementById('reviewByCategoryButton');
    if (reviewBtn) {
        reviewBtn.onclick = () => showReviewByCategoryModal(datasetName);
    }

    manageDatasetBsModal.show();
}

async function showLabelCoverageModal(datasetName) {
    if (!labelCoverageBsModal) return;

    document.getElementById('lc-dataset-name').textContent = datasetName;
    const reportContainer = document.getElementById('lc-report-container');
    reportContainer.innerHTML = '<div class="text-center">Generating report...</div>';
    manageDatasetBsModal.hide();
    labelCoverageBsModal.show();

    const report = await eel.get_label_coverage_report(datasetName)();

    if (report.error) {
        reportContainer.innerHTML = `<div class="alert alert-danger">${report.error}</div>`;
        return;
    }

    let html = '';
    const totalBehaviors = report.master_behavior_list.length;

    // Section for Incomplete Subjects (the most important part)
    if (report.incomplete_subjects && report.incomplete_subjects.length > 0) {
        html += '<h5><i class="bi bi-exclamation-triangle-fill text-warning me-2"></i>Incomplete Subjects</h5>';
        html += '<p class="small">These subjects are missing labels for one or more behaviors and are unlikely to be included in the test set.</p>';
        html += '<ul class="list-group">';
        report.incomplete_subjects.forEach(subject => {
            html += `
                <li class="list-group-item">
                    <strong>${subject.name}</strong>
                    <span class="badge bg-warning float-end">${subject.count} / ${totalBehaviors} Behaviors Labeled</span>
                    <br>
                    <small class="text-danger">Missing: ${subject.missing.join(', ')}</small>
                </li>
            `;
        });
        html += '</ul>';
    } else {
        html += '<h5><i class="bi bi-check-circle-fill text-success me-2"></i>All Subjects are Complete</h5>';
        html += '<p>Every subject with labels has at least one instance of every behavior. Your dataset is well-balanced for the train/test split!</p>';
    }

    // Section for Complete Subjects
    if (report.complete_subjects && report.complete_subjects.length > 0) {
        html += '<h5 class="mt-4"><i class="bi bi-check-circle-fill text-success me-2"></i>Complete Subjects</h5>';
        html += '<ul class="list-group">';
        report.complete_subjects.forEach(subject => {
            html += `
                <li class="list-group-item">
                    <strong>${subject.name}</strong>
                    <span class="badge bg-success float-end">${subject.count} / ${totalBehaviors} Behaviors Labeled</span>
                </li>
            `;
        });
        html += '</ul>';
    }

    reportContainer.innerHTML = html;
}

// To show and populate the whitelist editing modal
async function showEditWhitelistModal(datasetName, currentWhitelist) {
    if (!editWhitelistBsModal) return;
    
    document.getElementById('ew-dataset-name').innerText = datasetName;
    const treeContainer = document.getElementById('ew-recording-tree');
    treeContainer.innerHTML = 'Loading recording tree...';

    // We can reuse the existing eel.get_record_tree() function
    const fetchedRecordingTree = await eel.get_record_tree()();
    recordingDirTree = fetchedRecordingTree || {}; // Update global tree

    let treeHTML = '';
    if (fetchedRecordingTree && Object.keys(fetchedRecordingTree).length > 0) {
        for (const dateDir in fetchedRecordingTree) {
            // Check if the parent directory is in the whitelist
            const isParentChecked = currentWhitelist.includes(dateDir);
            treeHTML += `<div class="form-check"><input class="form-check-input" type="checkbox" id="ew-${dateDir}" ${isParentChecked ? 'checked' : ''} onchange="updateChildrenCheckboxes('ew-${dateDir}')"><label class="form-check-label" for="ew-${dateDir}">${dateDir}</label></div>`;
            
            let sessionsHTML = "<div style='margin-left:20px'>";
            fetchedRecordingTree[dateDir].forEach(sessionDir => {
                const fullPath = `${dateDir}/${sessionDir}`;
                // Check if the specific subdirectory is in the whitelist
                const isChildChecked = currentWhitelist.includes(fullPath);
                sessionsHTML += `<div class="form-check"><input class="form-check-input" type="checkbox" id="ew-${fullPath}" ${isChildChecked ? 'checked' : ''}><label class="form-check-label" for="ew-${fullPath}">${sessionDir}</label></div>`;
            });
            sessionsHTML += `</div>`;
            treeHTML += sessionsHTML;
        }
    } else {
        treeHTML = '<p class="text-muted">No recording directories found in this project.</p>';
    }
    treeContainer.innerHTML = treeHTML;

    // Attach the save handler
    document.getElementById('saveWhitelistButton').onclick = () => saveWhitelistChanges(datasetName);

    manageDatasetBsModal.hide(); // Hide the first modal
    editWhitelistBsModal.show(); // Show the new one
}

// To collect the new whitelist and save it
async function saveWhitelistChanges(datasetName) {
    const newWhitelist = [];
    // This logic is similar to how we collect directories for inference/creation
    Object.keys(recordingDirTree).forEach(dir => {
        const dirCheckbox = document.getElementById(`ew-${dir}`);
        if (dirCheckbox?.checked) {
            newWhitelist.push(dir);
        } else {
            recordingDirTree[dir]?.forEach(subdir => {
                const fullPath = `${dir}/${subdir}`;
                const subdirCheckbox = document.getElementById(`ew-${fullPath}`);
                if (subdirCheckbox?.checked) {
                    newWhitelist.push(fullPath);
                }
            });
        }
    });

    document.getElementById('cover-spin').style.visibility = 'visible';
    try {
        const success = await eel.update_dataset_whitelist(datasetName, newWhitelist)();
        if (success) {
            editWhitelistBsModal.hide();
            // Optional: Re-open the manage modal to show the updated list, or just close.
            // For simplicity, we'll just close. The change is saved.
        } else {
            showErrorOnLabelTrainPage("Failed to save the new directory list.");
        }
    } catch (e) {
        showErrorOnLabelTrainPage(`An error occurred: ${e.message}`);
    } finally {
        document.getElementById('cover-spin').style.visibility = 'hidden';
    }
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

async function showReviewByCategoryModal(datasetName) {
    // This function now opens the behavior selection modal.
    if (!behaviorSelectBsModal) return;

    const datasets = await eel.load_dataset_configs()();
    const behaviors = datasets[datasetName]?.behaviors || [];
    
    if (behaviors.length === 0) {
        showErrorOnLabelTrainPage("This dataset has no behaviors defined.");
        return;
    }

    // Populate the modal
    document.getElementById('bs-dataset-name').textContent = datasetName;
    const behaviorSelect = document.getElementById('bs-behavior-select');
    behaviorSelect.innerHTML = behaviors.map(b => `<option value="${b}">${b}</option>`).join('');

    // Show the modal
    manageDatasetBsModal.hide(); // Hide the manage modal first
    behaviorSelectBsModal.show();

    // Wire up the confirm button inside the new modal
    document.getElementById('bs-confirm-review-btn').onclick = () => {
        const selectedBehavior = behaviorSelect.value;
        if (selectedBehavior) {
            behaviorSelectBsModal.hide();
            // Call a new helper function to build the playlist
            buildAndShowPlaylist(datasetName, selectedBehavior);
        }
    };
}

async function buildAndShowPlaylist(datasetName, behaviorToReview) {
    // This new helper function contains the logic that used to be in the second half
    // of the old showReviewByCategoryModal function.
    
    // Show the playlist modal and set its title
    if (categoryReviewBsModal) {
        document.getElementById('cr-behavior-name').textContent = behaviorToReview;
        const playlistContainer = document.getElementById('cr-video-playlist');
        playlistContainer.innerHTML = '<li class="list-group-item text-center">Searching for instances...</li>';
        categoryReviewBsModal.show();
    }

    // Fetch the instance data from the backend
    const instancesByVideo = await eel.get_instances_for_behavior(datasetName, behaviorToReview)();
    
    const playlistContainer = document.getElementById('cr-video-playlist');
    playlistContainer.innerHTML = ''; // Clear the loading message

    if (Object.keys(instancesByVideo).length > 0) {
        const projectRoot = await eel.get_project_root()(); // Get root path once
        
        // Get the list of reviewed videos ONCE before the loop
        const reviewedVideos = JSON.parse(sessionStorage.getItem('categoryReviewedVideos') || '[]');

        for (const videoPath in instancesByVideo) {
            const videoData = instancesByVideo[videoPath];
            const listItem = document.createElement('li');
            listItem.className = 'list-group-item d-flex justify-content-between align-items-center';
            
            // Construct the absolute path
            const videoAbsPath = `${projectRoot}/${videoData.display_name}`.replace(/\\/g, '/');

            // =========================================================
            // Check if this video has been reviewed
            // =========================================================
            const isReviewed = reviewedVideos.includes(videoData.display_name);
            
            let statusIndicator = '<button class="btn btn-sm btn-primary">Start Review</button>';
            if (isReviewed) {
                statusIndicator = '<span class="badge bg-success"><i class="bi bi-check-lg"></i> Reviewed</span>';
            }
            // =========================================================

            listItem.innerHTML = `
                <div>
                    <span class="fw-bold">${videoData.display_name}</span>
                    <br>
                    <small class="text-muted">${videoData.instance_count} instance(s)</small>
                </div>
                ${statusIndicator} 
            `;
            
            // Only add the click listener if the video is NOT reviewed
            if (!isReviewed) {
                listItem.querySelector('button').onclick = () => {
                    categoryReviewBsModal.hide();
                    prepareAndShowLabelModal(datasetName, videoAbsPath, behaviorToReview);
                };
            }

            playlistContainer.appendChild(listItem);
        }
    } else {
        playlistContainer.innerHTML = '<li class="list-group-item text-center text-muted">No instances of this behavior found in the dataset.</li>';
    }
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
// IMPORT & CROPPING WORKFLOW
// =================================================================

async function setupCropOnImportModal(videoPath, fileCount) {
    document.getElementById('import-batch-file-count').textContent = fileCount;
    document.getElementById('cover-spin').style.visibility = 'visible';

    try {
        const base64Frame = await eel.get_frame_from_video(videoPath)();
        if (!base64Frame) {
            showErrorOnLabelTrainPage("Could not read the first video file to generate a preview for cropping.");
            cropOnImportBsModal?.hide();
            return;
        }

        importPreviewImage.onload = () => {
            initializeImportCropper(importPreviewImage);
            document.getElementById('cover-spin').style.visibility = 'hidden';
        };
        importPreviewImage.src = `data:image/jpeg;base64,${base64Frame}`;
    } catch (e) {
        showErrorOnLabelTrainPage(`Error setting up crop preview: ${e.message}`);
        document.getElementById('cover-spin').style.visibility = 'hidden';
    }
}

function initializeImportCropper(img) {
    importImageCanvas = document.getElementById("import-crop-image-canvas");
    importCropCanvas = document.getElementById("import-crop-overlay-canvas");
    if (!importImageCanvas || !importCropCanvas) return;
    
    importImageCtx = importImageCanvas.getContext("2d");
    importCropCtx = importCropCanvas.getContext("2d");

    const container = document.getElementById('import-crop-canvas-container');
    const aspectRatio = img.naturalWidth / img.naturalHeight;
    const canvasWidth = container.clientWidth;
    const canvasHeight = canvasWidth / aspectRatio;

    importImageCanvas.width = importCropCanvas.width = canvasWidth;
    importImageCanvas.height = importCropCanvas.height = canvasHeight;
    container.style.height = `${canvasHeight}px`;

    importImageCtx.drawImage(img, 0, 0, canvasWidth, canvasHeight);

    importCropRect = { x: 0, y: 0, w: canvasWidth, h: canvasHeight };
    updateInputsFromImportCropRect();
    drawImportCropOverlay();
}

function drawImportCropOverlay() {
    if (!importCropCtx || !importCropCanvas) return;
    importCropCtx.clearRect(0, 0, importCropCanvas.width, importCropCanvas.height);
    importCropCtx.fillStyle = "rgba(0, 0, 0, 0.5)";
    importCropCtx.fillRect(0, 0, importCropCanvas.width, importCropCanvas.height);
    importCropCtx.clearRect(importCropRect.x, importCropRect.y, importCropRect.w, importCropRect.h);
    importCropCtx.strokeStyle = "rgba(255, 0, 0, 0.9)";
    importCropCtx.lineWidth = 2;
    importCropCtx.strokeRect(importCropRect.x, importCropRect.y, importCropRect.w, importCropRect.h);
    drawImportCropHandles();
}

function drawImportCropHandles() {
    if (!importCropCtx) return;
    const { x, y, w, h } = importCropRect;
    const handles = {
        tl: { x: x, y: y }, tr: { x: x + w, y: y },
        bl: { x: x, y: y + h }, br: { x: x + w, y: y + h },
    };
    importCropCtx.fillStyle = "rgba(255, 0, 0, 0.9)";
    for (const handle in handles) {
        importCropCtx.fillRect(handles[handle].x - importCrop_HandleSize / 2, handles[handle].y - importCrop_HandleSize / 2, importCrop_HandleSize, importCrop_HandleSize);
    }
}

function getImportCropHandleAt(mouseX, mouseY) {
    const { x, y, w, h } = importCropRect;
    const handles = {
        tl: { x: x, y: y }, tr: { x: x + w, y: y },
        bl: { x: x, y: y + h }, br: { x: x + w, y: y + h },
        t: { x: x + w / 2, y: y }, b: { x: x + w / 2, y: y + h },
        l: { x: x, y: y + h / 2 }, r: { x: x + w, y: y + h / 2 }
    };
    for (const handle in handles) {
        if (Math.abs(mouseX - handles[handle].x) < importCrop_HandleSize && Math.abs(mouseY - handles[handle].y) < importCrop_HandleSize) {
            return handle;
        }
    }
    return null;
}

function updateInputsFromImportCropRect() {
    if (!importCropCanvas || importCropCanvas.width === 0 || importCropCanvas.height === 0) return;
    document.getElementById('import-crop-x').value = (importCropRect.x / importCropCanvas.width).toFixed(4);
    document.getElementById('import-crop-y').value = (importCropRect.y / importCropCanvas.height).toFixed(4);
    document.getElementById('import-crop-width').value = (importCropRect.w / importCropCanvas.width).toFixed(4);
    document.getElementById('import-crop-height').value = (importCropRect.h / importCropCanvas.height).toFixed(4);
}

async function populateSessionDatalist() {
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
function updateTrainingStatusOnUI(datasetName, displayMessage, originalMessage = null) {
    const statusElem = document.getElementById(`dataset-status-${datasetName}`);
    const progressContainer = document.getElementById(`progress-container-${datasetName}`);
    const progressBar = document.getElementById(`progress-bar-${datasetName}`);

    if (!statusElem || !progressContainer || !progressBar) return;

    // The message for the final state is still the first argument
    const isFinalState = /complete|failed|cancelled|error/i.test(displayMessage);
    
    // We now parse the epoch info from the original, unmodified message
    const messageForEpochParse = originalMessage || displayMessage;
    const epochMatch = messageForEpochParse.match(/Epoch (\d+)\s*\/\s*(\d+)/);

    if (isFinalState) {
        progressContainer.style.display = 'none';
        statusElem.innerHTML = displayMessage;
        statusElem.style.display = 'block';

        if (cardRefreshTimeoutId) clearTimeout(cardRefreshTimeoutId);
        cardRefreshTimeoutId = setTimeout(() => {
            loadInitialDatasetCards();
        }, 3000);

    } else {
        // This is an in-progress message.
        progressContainer.style.display = 'block';
        statusElem.style.display = 'block';
        
        // Update the status text with the new displayMessage
        statusElem.innerHTML = `
            <div class="d-flex justify-content-between align-items-center">
                <span>${displayMessage}</span>
                <button class="btn btn-xs btn-outline-danger py-0" onclick="cancelTraining('${datasetName}')">Cancel</button>
            </div>`;

        // Logic for updating the bar itself
        if (epochMatch) {
            const currentEpoch = parseInt(epochMatch[1]);
            const totalEpochs = parseInt(epochMatch[2]);
            const percent = Math.max(1, (currentEpoch / totalEpochs) * 100);

            progressBar.classList.remove('progress-bar-animated', 'bg-info');
            progressBar.classList.add('bg-primary');
            progressBar.style.width = `${percent}%`;
            progressBar.innerText = `Epoch ${currentEpoch}/${totalEpochs}`;
        } else if (displayMessage.toLowerCase().includes('loading dataset')) {
            progressBar.classList.add('progress-bar-animated', 'bg-info');
            progressBar.classList.remove('bg-primary');
            progressBar.style.width = '100%';
            progressBar.innerText = 'Loading...';
        } else {
             // For messages like "Trial X/Y..." that don't have epoch info,
             // we can reset the bar to 0% to show a new trial is starting.
             progressBar.style.width = '0%';
             progressBar.innerText = '';
        }
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
function buildLabelingUI(behaviors, colors, filterForBehavior = null) {
    labelingInterfaceActive = true;
    const controlsContainer = document.getElementById('controls');
    if (!controlsContainer) return;

    let controlsHTML = `<div class="card bg-dark text-light h-100"><div class="card-header"><h5></h5></div><ul class="list-group list-group-flush">`;
    controlsHTML += `<li class="list-group-item bg-dark text-light d-flex justify-content-between"><strong>Behavior</strong><span><strong>Confidence</strong></span><span><strong>Key</strong></span><span><strong>Count</strong></span></li>`;

    if (behaviors && colors && behaviors.length === colors.length) {
        behaviors.forEach((behaviorName, index) => {
            const isInactive = filterForBehavior && behaviorName !== filterForBehavior;
            const inactiveClass = isInactive ? 'label-row-inactive' : '';
            const title = isInactive ? `'${behaviorName}' is read-only in this mode` : `Click or press '${index + 1}' to label '${behaviorName}'`;

            const key = (index < 9) ? (index + 1) : String.fromCharCode('a'.charCodeAt(0) + (index - 9));
            const bgColor = colors[index];
            const textColor = getTextColorForBg(bgColor);
            controlsHTML += `
                <li class="list-group-item bg-dark text-light d-flex justify-content-between align-items-center ${inactiveClass}"
                    id="behavior-row-${behaviorName.replace(/[\W_]+/g, '-')}"
                    onclick="eel.label_frame(${index})()" style="cursor: pointer;" title="${title}">
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
    const saveBtn = document.getElementById('save-labels-btn');

    if (!controlsHeader || !cheatSheet || !saveBtn) return true;

    if (mode === 'review') {
        controlsHeader.classList.remove('bg-dark');
        controlsHeader.classList.add('bg-success');
        controlsHeader.querySelector('h5').innerHTML = `Reviewing: <span class="badge bg-light text-dark">${modelName}</span>`;
        
        // The innerHTML now matches the HTML file we just changed
        cheatSheet.innerHTML = `
            <div class="card bg-dark">
              <div class="card-header d-flex justify-content-between align-items-center">
                <h5 class="mb-0 text-success"><i class="bi bi-robot me-2"></i>Review Mode Controls</h5>
                <div id="confidence-filter-container" class="d-flex align-items-center w-50">
                  <div class="btn-group btn-group-sm me-3" role="group" aria-label="Filter mode">
                    <input type="radio" class="btn-check" name="filter-mode" id="filter-mode-below" autocomplete="off" checked>
                    <label class="btn btn-outline-info" for="filter-mode-below" data-bs-toggle="tooltip" title="Show instances BELOW this confidence (for finding errors)"><</label>
                    <input type="radio" class="btn-check" name="filter-mode" id="filter-mode-above" autocomplete="off">
                    <label class="btn btn-outline-info" for="filter-mode-above" data-bs-toggle="tooltip" title="Show instances ABOVE this confidence (for curating examples)">></label>
                  </div>
                  <label for="confidence-slider" class="form-label me-2 mb-0 text-nowrap text-light">Confidence</label>
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
        saveBtn.innerHTML = '<i class="bi bi-save-fill me-2"></i>Save New Labels';       
    }

    // --- ROBUST EVENT LISTENER ATTACHMENT ---
    const filterModeBelow = document.getElementById('filter-mode-below');
    const filterModeAbove = document.getElementById('filter-mode-above');
    const confidenceSlider = document.getElementById('confidence-slider');
    const sliderValueDisplay = document.getElementById('slider-value-display');
    const resetSliderBtn = document.getElementById('reset-slider-btn');
    const timelineContainer = document.getElementById('full-timeline-section');

    // This check is the crucial fix: only add listeners if the elements exist.
    if (filterModeBelow && filterModeAbove && confidenceSlider) {
        function triggerRefilter() {
            const threshold = parseInt(confidenceSlider.value);
            const mode = filterModeAbove.checked ? 'above' : 'below';
            eel.refilter_instances(threshold, mode)();
        }

        filterModeBelow.addEventListener('change', triggerRefilter);
        filterModeAbove.addEventListener('change', triggerRefilter);

        confidenceSlider.addEventListener('input', function() {
            if (sliderValueDisplay) sliderValueDisplay.textContent = `${this.value}%`;
            if (timelineContainer) {
                timelineContainer.classList.toggle('timeline-filtered', parseInt(this.value) < 100);
            }
            clearTimeout(confidenceFilterDebounceTimer);
            confidenceFilterDebounceTimer = setTimeout(triggerRefilter, 400);
        });

        if (resetSliderBtn) {
            resetSliderBtn.addEventListener('click', function() {
                confidenceSlider.value = 100;
                if (sliderValueDisplay) sliderValueDisplay.textContent = '100%';
                if (timelineContainer) timelineContainer.classList.remove('timeline-filtered');
                filterModeBelow.checked = true;
                triggerRefilter();
            });
        }
    }

    // Re-initialize tooltips for any newly created elements
    var tooltipTriggerList = [].slice.call(cheatSheet.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

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

async function handleMarkAsReviewed() {
    // This function is for when no changes were made, but the user wants to mark the video as done.
    
    // Step 1: Ask the backend for the authoritative, correctly formatted path of the current video.
    const reviewedVideoPath = await eel.get_current_labeling_video_path()();

    if (reviewedVideoPath) {
        // Step 2: Save this canonical path to sessionStorage.
        let reviewedVideos = JSON.parse(sessionStorage.getItem('categoryReviewedVideos') || '[]');
        if (!reviewedVideos.includes(reviewedVideoPath)) {
            reviewedVideos.push(reviewedVideoPath);
        }
        sessionStorage.setItem('categoryReviewedVideos', JSON.stringify(reviewedVideos));
        
        // Step 3: Close the labeling UI and return to the main dataset view.
        document.getElementById('label').style.display = 'none';
        document.getElementById('labeling-cheat-sheet').style.display = 'none';
        document.getElementById('datasets').style.display = 'block';
        labelingInterfaceActive = false;
    } else {
        // This is a fallback in case something went wrong on the backend.
        showErrorOnLabelTrainPage("Could not identify the current video to mark as reviewed.");
    }
}


function jumpToFrame() {
    const input = document.getElementById('frame-jump-input');
    if (input && input.value) {
        const frameNum = parseInt(input.value);
        if (!isNaN(frameNum)) eel.jump_to_frame(frameNum)();
    }
}

async function handleCommitClick() {
    const commitBtn = document.getElementById('save-labels-btn');
    if (!commitBtn) return;

    const isConfirming = commitBtn.innerText.includes("Confirm");

    if (isConfirming) {
        document.getElementById('cover-spin').style.visibility = 'visible';
        try {
            const result = await eel.save_session_labels()();

            if (result.status === 'success') {
                // Mark the video as reviewed in category mode if applicable
                if (result.video_path) {
                    let reviewedVideos = JSON.parse(sessionStorage.getItem('categoryReviewedVideos') || '[]');
                    if (!reviewedVideos.includes(result.video_path)) {
                        reviewedVideos.push(result.video_path);
                    }
                    sessionStorage.setItem('categoryReviewedVideos', JSON.stringify(reviewedVideos));
                }
                
                // Animate the save button
                commitBtn.innerHTML = '<i class="bi bi-check-lg"></i> Saved!';
                commitBtn.classList.add('btn-info');
                commitBtn.classList.remove('btn-primary');
                
                // Check if the saved dataset was a base dataset that has an augmented version.
                const savedDataset = result.dataset_name;
                if (savedDataset && !savedDataset.endsWith('_aug')) {
                    const augmentedDatasetCard = document.getElementById(`state-view-new-${savedDataset}_aug`) || 
                                                 document.getElementById(`state-view-labeled-${savedDataset}_aug`) || 
                                                 document.getElementById(`state-view-trained-${savedDataset}_aug`);
                    
                    if (augmentedDatasetCard) {
                        // If an augmented card exists, show the helpful prompt.
                        const alertMessage = `Corrections Saved to '${savedDataset}'\n\n` +
                            `Next Step: To apply these improvements to your '${savedDataset}_aug' dataset, ` +
                            `please click the 'Sync from Source' button before you re-train the model.`;
                        
                        // We show this alert *before* closing the UI so the user sees it.
                        alert(alertMessage);
                    }
                }

                // Close the labeling UI and refresh the main dashboard
                setTimeout(() => {
                    document.getElementById('label').style.display = 'none';
                    document.getElementById('labeling-cheat-sheet').style.display = 'none';
                    document.getElementById('datasets').style.display = 'block';
                    labelingInterfaceActive = false;
                    loadInitialDatasetCards(); // Use the main refresh to get latest stats
                }, 1500);

            } else if (result.status === 'no_changes') {
                alert("No changes were made to the labels in this session.");
                document.getElementById('label').style.display = 'none';
                document.getElementById('labeling-cheat-sheet').style.display = 'none';
                document.getElementById('datasets').style.display = 'block';
                labelingInterfaceActive = false;
            } else {
                showErrorOnLabelTrainPage(result.message || "An unknown error occurred during the save operation.");
            }
        } catch (e) {
            showErrorOnLabelTrainPage(`Save failed: ${e.message}`);
        } finally {
            document.getElementById('cover-spin').style.visibility = 'hidden';
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
    
    const sessionSelect = document.getElementById('pl-session-select');
    const subjectSelect = document.getElementById('pl-subject-select');
    const videoSelect = document.getElementById('pl-video-select');

    // Reset and disable dropdowns
    sessionSelect.innerHTML = '<option selected disabled>Loading sessions...</option>';
    subjectSelect.innerHTML = '<option selected disabled>Select a session first...</option>';
    videoSelect.innerHTML = '<option selected disabled>Select a subject first...</option>';
    subjectSelect.disabled = true;
    videoSelect.disabled = true;

    preLabelBsModal?.show();

    // Fetch the entire hierarchical structure from the backend
    const videoHierarchy = await eel.get_hierarchical_video_list(datasetName)();

    if (Object.keys(videoHierarchy).length === 0) {
        sessionSelect.innerHTML = '<option selected disabled>No whitelisted videos found.</option>';
        return;
    }

    // --- Populate Session Dropdown ---
    sessionSelect.innerHTML = '<option selected disabled>Choose a session...</option>';
    for (const sessionName in videoHierarchy) {
        sessionSelect.innerHTML += `<option value="${sessionName}">${sessionName}</option>`;
    }

    // --- Event Listener for Session Selection ---
    sessionSelect.onchange = () => {
        const selectedSession = sessionSelect.value;
        subjectSelect.innerHTML = '<option selected disabled>Choose a subject...</option>';
        videoSelect.innerHTML = '<option selected disabled>Select a subject first...</option>';
        videoSelect.disabled = true;
        
        if (selectedSession && videoHierarchy[selectedSession]) {
            for (const subjectName in videoHierarchy[selectedSession]) {
                subjectSelect.innerHTML += `<option value="${subjectName}">${subjectName}</option>`;
            }
            subjectSelect.disabled = false;
        } else {
            subjectSelect.disabled = true;
        }
    };

    // --- Event Listener for Subject Selection ---
    subjectSelect.onchange = () => {
        const selectedSession = sessionSelect.value;
        const selectedSubject = subjectSelect.value;
        videoSelect.innerHTML = '<option selected disabled>Choose a video...</option>';

        if (selectedSession && selectedSubject && videoHierarchy[selectedSession][selectedSubject]) {
            const videos = videoHierarchy[selectedSession][selectedSubject];
            videos.forEach(video => {
                // The value will be the full path, the text will be the filename
                videoSelect.innerHTML += `<option value="${video[0]}">${video[1]}</option>`;
            });
            videoSelect.disabled = false;
        } else {
            videoSelect.disabled = true;
        }
    };
}

async function onModelSelectChange(event) {
    const modelName = event.target.value;
    const datasetName = document.getElementById('pl-dataset-name').innerText;
    
    const infoDiv = document.getElementById('pl-behavior-match-info');
    infoDiv.innerHTML = ''; // Clear previous info
    if (!modelName) return;

    try {
        const [allModelConfigs, allDatasetConfigs] = await Promise.all([
            eel.get_model_configs()(), 
            eel.load_dataset_configs()()
        ]);
        
        const targetBehaviors = new Set(allDatasetConfigs[datasetName]?.behaviors || []);
        const modelConfig = allModelConfigs[modelName];

        if (modelConfig?.behaviors) {
            const modelBehaviors = new Set(modelConfig.behaviors);
            const matching = [...targetBehaviors].filter(b => modelBehaviors.has(b));
            const nonMatching = [...targetBehaviors].filter(b => !modelBehaviors.has(b));
            
            let infoHTML = '';
            if (matching.length > 0) {
                infoHTML += `Will pre-label for: <strong>${matching.join(', ')}</strong>.`;
            }
            if (nonMatching.length > 0) {
                infoHTML += `<br><span class="text-warning small">Will ignore: ${nonMatching.join(', ')}</span>`;
            }
            infoDiv.innerHTML = infoHTML || '<span class="text-danger">No matching behaviors found.</span>';
        }
    } catch(e) { 
        console.error("Error updating behavior match info:", e); 
        infoDiv.innerHTML = '<span class="text-danger">Could not load model info.</span>';
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
    
    eel.reload_project_data()().then(() => {
        loadInitialDatasetCards().then(() => {
            document.getElementById('cover-spin').style.visibility = 'hidden';
        });
    }).catch(error => {
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
    const smoothingWindow = parseInt(document.getElementById('pl-smoothing-window').value) || 1;

    if (!modelName || modelName.includes("...")) { showErrorOnLabelTrainPage("Please select a model."); return; }
    if (!videoPath || videoPath.includes("...")) { showErrorOnLabelTrainPage("Please select a video."); return; }

    // Ask the backend if this video already has human-verified labels.
    const hasLabels = await eel.video_has_labels(datasetName, videoPath)();

    if (hasLabels) {
        
		const confirmationMessage = "This video already contains human-verified labels.\n\n" +
		"Starting a 'Review & Correct' session will load your existing labels alongside the model's new predictions.\n\n" +
		"Your original human-verified labels will be kept automatically. Any model predictions you confirm or modify will be added.\n\n" +
		"When you 'Commit Corrections', this new, combined set of labels will overwrite the previous labels for this video.\n\n" +
            "Are you sure you want to proceed?";
		
        if (!confirm(confirmationMessage)) {
            return; // User clicked cancel, so we stop here.
        }
    }

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

async function prepareAndShowLabelModal(datasetName, videoToOpen, filterForBehavior = null, startFrame = 0, disagreementInfo = null) {
    isReviewByBehaviorMode = !!filterForBehavior;
    try {
        const markReviewedBtn = document.getElementById('mark-reviewed-btn');
        const bannerContainer = document.getElementById('disagreement-banner-container');
        const bannerContent = document.getElementById('disagreement-banner-content');
        
        if (disagreementInfo && bannerContainer && bannerContent) {
            bannerContent.innerHTML = `
                <strong>Reviewing Model Error:</strong> You labeled this segment as 
                <span class="badge bg-success">${disagreementInfo.human}</span>, but the model 
                predicted it was <span class="badge bg-danger">${disagreementInfo.model}</span>. 
                Please review and correct the label if necessary.
            `;
            bannerContainer.style.display = 'block';
        } else if (bannerContainer) {
            bannerContainer.style.display = 'none';
        }

        setLabelingModeUI('scratch'); 
        if (filterForBehavior) {
            const controlsHeader = document.querySelector('#controls .card-header h5');
            if (controlsHeader) {
                controlsHeader.innerHTML = `Reviewing: <span class="badge bg-primary">${filterForBehavior}</span>`;
            }
            if (markReviewedBtn) markReviewedBtn.style.display = 'inline-block';
        } else {
            if (markReviewedBtn) markReviewedBtn.style.display = 'none';
        }
                
        // Show a spinner while the backend prepares the session.
        document.getElementById('cover-spin').style.visibility = 'visible';

        // 1. Await the completion of the backend setup. This function now
        //    blocks until the video is loaded and the initial frame is rendered.
        const success = await eel.start_labeling(datasetName, videoToOpen, null, filterForBehavior)();
        
        // Hide the spinner once setup is complete.
        document.getElementById('cover-spin').style.visibility = 'hidden';

        if (!success) {
            // If the backend reported an error, do not proceed.
            // The error message will have already been shown by the backend.
            return;
        }

        // 2. Only after the setup is confirmed to be successful,
        //    send the command to jump to the specific frame.
        if (startFrame > 0) {
            await eel.jump_to_frame(startFrame)();
        }
        
    } catch (error) {
        // Hide spinner on error as a fallback
        document.getElementById('cover-spin').style.visibility = 'hidden';
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
            
            // Reset crop data for this new batch
            importCropData = { x: 0.0, y: 0.0, w: 1.0, h: 1.0, apply: true, stretch: false };

            // Populate and show the MAIN import modal
            document.getElementById('import-file-count').textContent = filePaths.length;
            document.getElementById('import-session-name').value = '';
            document.getElementById('import-subject-name').value = '';
            await populateSessionDatalist();
            importVideosBsModal?.show();

            // Attach a listener to our new button
            document.getElementById('set-crop-for-import-btn').onclick = () => {
                importVideosBsModal.hide();
                setupCropOnImportModal(filePaths[0], filePaths.length);
                cropOnImportBsModal.show();
            };

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
        await eel.import_videos(sessionName, subjectName, selectedVideoPathsForImport, shouldStandardize, importCropData)();
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

    const methodSelect = document.getElementById('tm-method');
    const weightsPanel = document.getElementById('custom-weights-panel');
    const weightsContainer = document.getElementById('custom-weights-container');

    // Function to toggle the visibility of the custom weights panel
    const toggleWeightsPanel = () => {
        if (methodSelect.value === 'custom_weights') {
            populateCustomWeights(datasetName);
            weightsPanel.style.display = 'block';
        } else {
            weightsPanel.style.display = 'none';
        }
    };

    // Function to calculate and display the smart default weights
    const populateCustomWeights = async (dsName) => {
        weightsContainer.innerHTML = '<em>Loading suggested weights...</em>';
        const datasets = await eel.load_dataset_configs()();
        const dataset = datasets[dsName];
        if (!dataset || !dataset.behaviors) {
            weightsContainer.innerHTML = '<em class="text-danger">Could not load behaviors.</em>';
            return;
        }

        let html = '<div class="row g-2">';
        const scalingFactor = 4.0; // This can be tuned

        dataset.behaviors.forEach(behavior => {
            const metrics = dataset.metrics ? dataset.metrics[behavior] : null;
            let f1Score = 0.5; // Default F1 if none is available
            if (metrics && metrics['F1 Score'] && !isNaN(parseFloat(metrics['F1 Score']))) {
                f1Score = parseFloat(metrics['F1 Score']);
            }

            // Calculate the suggested weight
            const difficulty = 1.0 - f1Score;
            let suggestedWeight = 1.0 + (difficulty * scalingFactor);
            // Clamp the values to a reasonable range, e.g., [0.5, 5.0]
            suggestedWeight = Math.max(0.5, Math.min(5.0, suggestedWeight));

            html += `
                <div class="col-6 d-flex align-items-center">
                    <label for="cw-${behavior}" class="form-label me-2 mb-0 text-nowrap">${behavior}:</label>
                    <input type="number" class="form-control form-control-sm custom-weight-input" id="cw-${behavior}" 
                           data-behavior="${behavior}" value="${suggestedWeight.toFixed(2)}" step="0.1" min="0.1">
                </div>
            `;
        });
        html += '</div>';
        weightsContainer.innerHTML = html;
    };

    // Add event listener to the dropdown
    if (methodSelect) {
        methodSelect.addEventListener('change', toggleWeightsPanel);
    }

    // Initial check when the modal is opened
    toggleWeightsPanel();

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
    const numRuns = document.getElementById('tm-runs').value;
	const numTrials = document.getElementById('tm-trials').value;
    const optimizationTarget = document.getElementById('tm-optimization-target').value;


    let customWeights = null;
    if (trainMethod === 'custom_weights') {
        customWeights = {};
        const weightInputs = document.querySelectorAll('.custom-weight-input');
        if (weightInputs.length === 0) {
            showErrorOnLabelTrainPage("Custom weights selected, but no behaviors found to assign weights.");
            return;
        }
        weightInputs.forEach(input => {
            const behavior = input.dataset.behavior;
            const weight = parseFloat(input.value);
            if (behavior && !isNaN(weight)) {
                customWeights[behavior] = weight;
            }
        });
    }


    if (!batchSize || !seqLen || !learningRate || !epochsCount || !patience || !numRuns || !numTrials) {
        showErrorOnLabelTrainPage("All training parameters must be filled.");
        return;
    }
    
    trainBsModal?.hide();

    if (cardRefreshTimeoutId) {
        clearTimeout(cardRefreshTimeoutId);
        cardRefreshTimeoutId = null;
    }

    updateTrainingStatusOnUI(datasetName, "Training task queued...");
    
    eel.train_model(datasetName, batchSize, learningRate, epochsCount, seqLen, trainMethod, patience, numRuns, numTrials, optimizationTarget, customWeights)();
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
        container.className = 'row g-3';
        let htmlContent = '';

        if (datasets) {
            for (const datasetName in datasets) {
                
                const config = datasets[datasetName];
                const state = config.state || 'new';
                const behaviors = config.behaviors || [];
                const metrics = config.metrics || {};

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

                htmlContent += `
                    <div class="card-state-view" id="state-view-new-${datasetName}" style="display: ${state === 'new' ? 'flex' : 'none'};"><div class="text-center my-auto"><p class="text-muted">Your dataset is empty.</p><button class="btn btn-primary" onclick="showPreLabelOptions('${datasetName}')"><i class="bi bi-pen-fill me-2"></i>Label First Video</button></div></div>
                    <div class="card-state-view" id="state-view-labeled-${datasetName}" style="display: ${state === 'labeled' ? 'flex' : 'none'}; flex-direction: column;"><div><p class="small text-muted mb-2">You have labeled examples. You can add more, or train your first model.</p><div class="table-responsive" style="max-height: 150px;"><table class="table table-sm table-hover small"><tbody>${behaviors.map(b => `<tr><td>${b}</td><td class="text-end">${(metrics[b] || {})['Train #'] || '0 (0)'}</td><td class="text-end">${(metrics[b] || {})['Test #'] || '0 (0)'}</td></tr>`).join('')}</tbody></table></div></div></div>
                    <div class="card-state-view" id="state-view-trained-${datasetName}" style="display: ${state === 'trained' ? 'flex' : 'none'}; flex-direction: column; font-size: 0.85rem;"><p class="small text-muted mb-2">Your model is trained. Use it to infer on new videos, or improve it by adding more labels.</p><div class="table-responsive"><table class="table table-sm table-hover small"><thead><tr><th>Behavior</th>${['Train Inst<br><small>(Frames)</small>', 'Test Inst<br><small>(Frames)</small>', 'Precision', 'Recall', 'F1 Score'].map(h => `<th class="text-center">${h}</th>`).join('')}</tr></thead><tbody>${behaviors.map(b => { const bMetrics = metrics[b] || {}; return `<tr><td>${b}</td><td class="text-center">${bMetrics['Train #'] || 'N/A'}</td><td class="text-center">${bMetrics['Test #'] || 'N/A'}</td><td class="text-center">${bMetrics['Precision'] || 'N/A'}</td><td class="text-center">${bMetrics['Recall'] || 'N/A'}</td><td class="text-center">${bMetrics['F1 Score'] || 'N/A'}</td></tr>`; }).join('')}</tbody></table></div></div>`;

                htmlContent += `
                    <div class="mt-auto">
                        <div class="progress mt-2" id="progress-container-${datasetName}" style="height: 20px; display: none;"><div class="progress-bar progress-bar-striped progress-bar-animated" id="progress-bar-${datasetName}" role="progressbar" style="width: 0%;"></div></div>
                        <div id="dataset-status-${datasetName}" class="mt-2 small text-info"></div>
                    </div>`;

                htmlContent += `
                            </div>
                            <div class="card-footer d-flex justify-content-end align-items-center flex-wrap gap-1">
                                <button class="btn btn-sm btn-outline-secondary me-auto" type="button" onclick="showManageDatasetModal('${datasetName}')" data-bs-toggle="tooltip" title="Manage dataset files and settings"><i class="bi bi-folder2-open"></i> Manage</button>`;
                
                if (config.has_disagreements) {
                    const tooltipText = datasetName.endsWith('_aug') 
                        ? "Review the instances your model was most confused about. Corrections will be made to the original source dataset."
                        : "Review the instances your model was most confused about to improve your labels.";
                    htmlContent += `<button class="btn btn-sm btn-outline-warning" type="button" onclick="showDisagreementModal('${datasetName}')" data-bs-toggle="tooltip" title="${tooltipText}"><i class="bi bi-lightbulb-fill"></i> Review Errors</button>`;
                }

                if (datasetName.endsWith('_aug')) {
                    const sourceName = datasetName.replace('_aug', '');
                    htmlContent += `<button class="btn btn-sm btn-info" type="button" onclick="showSyncModal('${sourceName}', '${datasetName}')" data-bs-toggle="tooltip" title="Re-sync labels from the original '${sourceName}' dataset."><i class="bi bi-arrow-repeat"></i> Sync</button>`;
                } else if (state !== 'new') {
                    htmlContent += `<button class="btn btn-sm btn-outline-info" type="button" onclick="showAugmentModal('${datasetName}')" data-bs-toggle="tooltip" title="Create an augmented version of this dataset."><i class="bi bi-images"></i> Augment</button>`;
                }
                
                if (!datasetName.endsWith('_aug') && state !== 'new') {
                     htmlContent += `<button class="btn btn-sm btn-primary" type="button" onclick="showPreLabelOptions('${datasetName}')" data-bs-toggle="tooltip" title="Add or correct labels for this dataset.">Label More</button>`;
                }
                
                if (state === 'labeled' || state === 'trained') {
                    const trainText = state === 'trained' ? 'Re-Train' : 'Train';
                    const trainTooltip = state === 'trained' ? 'Re-train the model, potentially with new labels or different settings.' : 'Train a new model using the labels in this dataset.';
                    htmlContent += `<button class="btn btn-sm btn-success" type="button" onclick="checkAndShowTrainModal('${datasetName}')" data-bs-toggle="tooltip" title="${trainTooltip}">${trainText}</button>`;
                }
                
                // The "Infer" button is now removed from here.
                
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
	document.getElementById('mark-reviewed-btn')?.addEventListener('click', handleMarkAsReviewed);
    document.getElementById('trainModelButton')?.addEventListener('click', submitTrainModel);
	document.getElementById('startClassificationButton')?.addEventListener('click', submitStartClassification);
    
	// --- Listeners for the NEW Import Cropping Modal ---
	document.getElementById('confirm-crop-and-proceed-btn')?.addEventListener('click', async () => {
		importCropData = {
			x: parseFloat(document.getElementById('import-crop-x').value),
			y: parseFloat(document.getElementById('import-crop-y').value),
			w: parseFloat(document.getElementById('import-crop-width').value),
			h: parseFloat(document.getElementById('import-crop-height').value),
			apply: document.getElementById('apply-crop-to-all-toggle').checked,
			stretch: document.getElementById('stretch-to-square-toggle').checked
		};
		cropOnImportBsModal?.hide();
		importVideosBsModal?.show();
	});

	const importCropOverlay = document.getElementById("import-crop-overlay-canvas");
	if (importCropOverlay) {
		importCropOverlay.addEventListener('mousedown', (e) => {
			const { offsetX, offsetY } = e;
			importCrop_ResizeHandle = getImportCropHandleAt(offsetX, offsetY);
			if (importCrop_ResizeHandle) {
				isImportCropping_Resizing = true;
			} else if (offsetX > importCropRect.x && offsetX < importCropRect.x + importCropRect.w && offsetY > importCropRect.y && offsetY < importCropRect.y + importCropRect.h) {
				isImportCropping_Dragging = true;
			}
		});

		importCropOverlay.addEventListener('mousemove', (e) => {
			const { offsetX, offsetY, movementX, movementY } = e;
			if (isImportCropping_Dragging) {
				importCropRect.x += movementX;
				importCropRect.y += movementY;
			} else if (isImportCropping_Resizing) {
				if (importCrop_ResizeHandle.includes('l')) { importCropRect.x += movementX; importCropRect.w -= movementX; }
				if (importCrop_ResizeHandle.includes('r')) { importCropRect.w += movementX; }
				if (importCrop_ResizeHandle.includes('t')) { importCropRect.y += movementY; importCropRect.h -= movementY; }
				if (importCrop_ResizeHandle.includes('b')) { importCropRect.h += movementY; }
			} else {
				const handle = getImportCropHandleAt(offsetX, offsetY);
				if (handle) {
					if (handle.includes('n') || handle.includes('s')) importCropOverlay.style.cursor = 'ns-resize';
					else if (handle.includes('e') || handle.includes('w')) importCropOverlay.style.cursor = 'ew-resize';
				} else if (offsetX > importCropRect.x && offsetX < importCropRect.x + importCropRect.w && offsetY > importCropRect.y && offsetY < importCropRect.y + importCropRect.h) {
					importCropOverlay.style.cursor = 'move';
				} else {
					importCropOverlay.style.cursor = 'crosshair';
				}
			}
			if (isImportCropping_Dragging || isImportCropping_Resizing) {
				drawImportCropOverlay();
				updateInputsFromImportCropRect();
			}
		});

		importCropOverlay.addEventListener('mouseup', () => {
			isImportCropping_Dragging = false;
			isImportCropping_Resizing = false;
			importCrop_ResizeHandle = null;
		});

		importCropOverlay.addEventListener('mouseleave', () => {
			isImportCropping_Dragging = false;
			isImportCropping_Resizing = false;
		});
	}
	
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
    
    // MODIFICATION: Add the onchange event listener to the model selector
    if (modelSelectModal) {
        modelSelectModal.addEventListener('change', onModelSelectChange);
    }

    // Reset the modal when it's closed
    preLabelModalElement?.addEventListener('hidden.bs.modal', () => {
        if(guidedOptionsPanel) guidedOptionsPanel.style.display = 'none';
        if(videoSelectModal) videoSelectModal.selectedIndex = 0;
        if(modelSelectModal) {
            modelSelectModal.innerHTML = '';
            // Clear the info div as well
            const infoDiv = document.getElementById('pl-behavior-match-info');
            if (infoDiv) infoDiv.innerHTML = '';
        }
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
				if (isReviewByBehaviorMode && document.querySelector('#controls .list-group-item.highlight-selected')) {
					console.log("Label changing is disabled in Review by Behavior mode.");
					return; // Do nothing
				}
				eel.label_frame(bIdx)();
			} else {
				handled = false;
            }
			
            break;
    }
    if (handled) event.preventDefault();
});