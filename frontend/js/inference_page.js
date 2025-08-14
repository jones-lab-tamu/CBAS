/**
 * @file Manages the Inference page UI and interactions.
 */

// =================================================================
// GLOBAL STATE & VARIABLES
// =================================================================

let generalErrorBsModal = document.getElementById('errorModal') ? new bootstrap.Modal(document.getElementById('errorModal')) : null;
let recordingDirTree = {};
let selectedModel = null;

// =================================================================
// EEL-EXPOSED FUNCTIONS
// =================================================================

eel.expose(updateInferenceProgress);
function updateInferenceProgress(modelName, percent, message) {
    if (modelName !== selectedModel) return; // Only update for the active task

    const progressPanel = document.getElementById('inference-progress-panel');
    const startPanel = document.getElementById('inference-start-panel');
    const progressBar = document.getElementById('inference-progress-bar');
    const statusLabel = document.getElementById('inference-status-label');

    if (!progressPanel || !startPanel || !progressBar || !statusLabel) return;

    startPanel.style.display = 'none';
    progressPanel.style.display = 'block';

    statusLabel.textContent = message;
    const displayPercent = Math.round(percent);
    progressBar.style.width = `${displayPercent}%`;
    progressBar.textContent = `${displayPercent}%`;
    progressBar.setAttribute('aria-valuenow', displayPercent);

    if (percent >= 100) {
        setTimeout(() => {
            startPanel.style.display = 'block';
            progressPanel.style.display = 'none';
            document.getElementById('start-inference-btn').disabled = true;
            document.getElementById('inference-instructions').textContent = `Inference with '${selectedModel}' complete. You can now view the results on the Visualize page.`;
        }, 2000);
    }
}

// =================================================================
// ROUTING & UI FUNCTIONS
// =================================================================

function routeToRecordPage() { window.location.href = './record.html'; }
function routeToLabelTrainPage() { window.location.href = './label-train.html'; }
function routeToVisualizePage() { window.location.href = './visualize.html'; }
function routeToInferencePage() { window.location.href = './inference.html'; }

function showError(message) {
    const el = document.getElementById("error-message");
    if (el && generalErrorBsModal) { el.innerText = message; generalErrorBsModal.show(); }
    else { alert(message); }
}

function updateStartButtonState() {
    const startBtn = document.getElementById('start-inference-btn');
    const instructions = document.getElementById('inference-instructions');
    const selectedDirs = document.querySelectorAll('#video-tree-container input[type="checkbox"]:checked').length;

    if (selectedModel && selectedDirs > 0) {
        startBtn.disabled = false;
        instructions.textContent = `Ready to run inference with '${selectedModel}' on the selected directories.`;
    } else {
        startBtn.disabled = true;
        if (!selectedModel) {
            instructions.textContent = 'Please select a model.';
        } else {
            instructions.textContent = 'Please select at least one video directory.';
        }
    }
}

// =================================================================
// CORE LOGIC
// =================================================================

async function loadModels() {
    const container = document.getElementById('model-list-container');
    if (!container) return;
    container.innerHTML = '<div class="list-group-item">Loading models...</div>';

    try {
        const models = await eel.get_available_models()();
        if (models && models.length > 0) {
            let html = '';
            models.forEach(modelName => {
                html += `<button type="button" class="list-group-item list-group-item-action" data-model-name="${modelName}">${modelName}</button>`;
            });
            container.innerHTML = html;

            // Add click event listeners
            container.querySelectorAll('button').forEach(button => {
                button.addEventListener('click', () => {
                    // Remove active state from all other buttons
                    container.querySelectorAll('button').forEach(btn => btn.classList.remove('active'));
                    // Add active state to the clicked button
                    button.classList.add('active');
                    selectedModel = button.dataset.modelName;
                    updateStartButtonState();
                });
            });
        } else {
            container.innerHTML = '<div class="list-group-item text-muted">No trained models found in this project.</div>';
        }
    } catch (e) {
        container.innerHTML = '<div class="list-group-item text-danger">Failed to load models.</div>';
        console.error(e);
    }
}

async function loadVideoTree() {
    const container = document.getElementById('video-tree-container');
    if (!container) return;
    container.innerHTML = 'Loading video directories...';

    try {
        const fetchedRecordingTree = await eel.get_record_tree()();
        recordingDirTree = fetchedRecordingTree || {};
        if (Object.keys(recordingDirTree).length > 0) {
            let treeHTML = '';
            for (const sessionDir in recordingDirTree) {
                treeHTML += `
                    <div class="form-check">
                        <input class="form-check-input" type="checkbox" id="cb-sess-${sessionDir}">
                        <label class="form-check-label" for="cb-sess-${sessionDir}">${sessionDir}</label>
                    </div>
                    <div class="ms-4">`;
                recordingDirTree[sessionDir].forEach(subjectDir => {
                    treeHTML += `
                        <div class="form-check">
                            <input class="form-check-input" type="checkbox" id="cb-subj-${sessionDir}-${subjectDir}">
                            <label class="form-check-label" for="cb-subj-${sessionDir}-${subjectDir}">${subjectDir}</label>
                        </div>`;
                });
                treeHTML += `</div>`;
            }
            container.innerHTML = treeHTML;

            // Add event listeners for hierarchical checking and button state
            container.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
                checkbox.addEventListener('change', () => {
                    // Hierarchical checking logic
                    if (checkbox.id.startsWith('cb-sess-')) {
                        const sessionName = checkbox.id.replace('cb-sess-', '');
                        document.querySelectorAll(`input[id^="cb-subj-${sessionName}-"]`).forEach(child => {
                            child.checked = checkbox.checked;
                        });
                    }
                    updateStartButtonState();
                });
            });
        } else {
            container.innerHTML = '<p class="text-muted">No recordings found in this project.</p>';
        }
    } catch (e) {
        container.innerHTML = '<p class="text-danger">Failed to load video tree.</p>';
        console.error(e);
    }
}

async function startInference() {
    const selectedDirs = [];
    Object.keys(recordingDirTree).forEach(sessionDir => {
        const sessionCheckbox = document.getElementById(`cb-sess-${sessionDir}`);
        if (sessionCheckbox && sessionCheckbox.checked) {
            selectedDirs.push(sessionDir);
        } else {
            recordingDirTree[sessionDir].forEach(subjectDir => {
                const subjectCheckbox = document.getElementById(`cb-subj-${sessionDir}-${subjectDir}`);
                if (subjectCheckbox && subjectCheckbox.checked) {
                    selectedDirs.push(`${sessionDir}/${subjectDir}`);
                }
            });
        }
    });

    if (!selectedModel || selectedDirs.length === 0) {
        showError('Please select a model and at least one directory.');
        return;
    }

    updateInferenceProgress(selectedModel, 0, `Queuing tasks for model '${selectedModel}'...`);
    await eel.start_classification(selectedModel, selectedDirs)();
}

// =================================================================
// PAGE INITIALIZATION
// =================================================================

document.addEventListener('DOMContentLoaded', () => {
    loadModels();
    loadVideoTree();

    document.getElementById('start-inference-btn')?.addEventListener('click', startInference);
});