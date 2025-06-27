/**
 * @file Manages the Visualize page UI and interactions.
 * @description This file handles building the data selection tree, generating multiple
 * tiled actograms based on user input, and adjusting them in real-time.
 */


// =================================================================
// GLOBAL STATE & VARIABLES
// =================================================================

let routingInProgress = false;
const errorModalElement = document.getElementById("errorModal");
let generalErrorBsModal = errorModalElement ? new bootstrap.Modal(errorModalElement) : null;

let currentSelection = { root: null, session: null, model: null };

// --- Task & Debounce Management ---
let actogramDebounceTimer;
let latestVizTaskId = 0; // The ID of the most recent request sent from the frontend.

// =================================================================
// ROUTING & UI MODE FUNCTIONS
// =================================================================

function routeToRecordPage() { routingInProgress = true; window.location.href = './record.html'; }
function routeToLabelTrainPage() { routingInProgress = true; window.location.href = './label-train.html'; }
function routeToVisualizePage() { routingInProgress = true; window.location.href = './visualize.html'; }

/**
 * Controls the visibility of UI sections based on the selected visualization mode.
 * This function is called by the onclick events in the HTML.
 * @param {string} mode - The mode to switch to, either 'actogram' or 'ethogram'.
 */
function setVisualizationMode(mode) {
    const actogramUI = document.getElementById('actogram-mode-ui');
    const ethogramUI = document.getElementById('ethogram-mode-ui');
    const titleElement = document.getElementById('visualization-title');

    if (mode === 'actogram') {
        titleElement.textContent = 'Actogram Analysis';
        actogramUI.style.display = 'block';
        ethogramUI.style.display = 'none';
        initializeActogramUI();
    } else if (mode === 'ethogram') {
        titleElement.textContent = 'Ethogram Analysis (Single Video)';
        actogramUI.style.display = 'none';
        ethogramUI.style.display = 'block';
        document.getElementById('ethogram-container').innerHTML = '';
        initializeEthogramUI();
    }
}

function toggleVisibility(elementId) {
    const elem = document.getElementById(elementId);
    if (elem) elem.style.display = (elem.style.display === 'none' || elem.style.display === '') ? 'block' : 'none';
}

function showActogramLoadingIndicator() {
    const spinner = document.getElementById('loading-spinner-actogram');
    const container = document.getElementById('actogram-container');
    
    if (spinner) spinner.style.display = "block";
    if (container) container.innerHTML = ''; 
}

// =================================================================
// EEL-EXPOSED FUNCTIONS (Called FROM Python)
// =================================================================

eel.expose(showErrorOnVisualizePage);
function showErrorOnVisualizePage(message) {
    const errorMessageElement = document.getElementById("error-message");
    if (errorMessageElement && generalErrorBsModal) {
        errorMessageElement.innerText = message;
        generalErrorBsModal.show();
    } else { alert(message); }
}

eel.expose(updateActogramDisplay);
function updateActogramDisplay(results, taskId) {
    if (taskId !== latestVizTaskId) {
        console.log(`Ignoring obsolete UI update for task ${taskId}. Current task is ${latestVizTaskId}.`);
        return;
    }

    const container = document.getElementById('actogram-container');
    const spinner = document.getElementById('loading-spinner-actogram');

    if (spinner) spinner.style.display = "none";
    if (!container) return;

    if (results && results.length > 0) {
        let html = '';
        const colClass = results.length === 1 ? 'col-12' : 'col-xl-6';

        results.forEach(result => {
            html += `
                <div class="${colClass}">
                    <div class="card bg-dark text-light">
                        <div class="card-header text-center">
                            <h6>${result.behavior}</h6>
                        </div>
                        <div class="card-body p-1">
                            <img src="data:image/png;base64,${result.blob}" class="img-fluid" alt="Actogram for ${result.behavior}">
                        </div>
                    </div>
                </div>
            `;
        });
        container.innerHTML = html;
    } else {
        container.innerHTML = `
            <div id="actogram-placeholder" class="d-flex align-items-center justify-content-center text-muted"
                 style="border: 1px dashed #6c757d; border-radius: .375rem; height: 300px; background-color: #212529;">
              <p class="mb-0 text-light">Select a behavior to generate an actogram.</p>
            </div>
        `;
    }
}

// =================================================================
// CORE APPLICATION LOGIC
// =================================================================

async function generateAndDisplayActograms() {
    const exportBtn = document.getElementById('export-data-btn');

    const checkedBehaviors = Array.from(document.querySelectorAll('.behavior-checkbox:checked'));
    
    latestVizTaskId++;
    const currentTaskId = latestVizTaskId;
    
    if (checkedBehaviors.length === 0) {
        updateActogramDisplay([], currentTaskId);
        // Reset the title when no behaviors are selected
        document.getElementById('visualization-title').textContent = 'Actogram Analysis';
        if (exportBtn) exportBtn.disabled = true;
        return;
    }

    const firstCheckbox = checkedBehaviors[0];
    const rootDir = firstCheckbox.dataset.root;
    const sessionDir = firstCheckbox.dataset.session;
    const modelName = firstCheckbox.dataset.model;
    const behaviorNames = checkedBehaviors.map(cb => cb.dataset.behavior);

    const framerate = document.getElementById('vs-framerate').value;
    const binsize = document.getElementById('vs-binsize').value;
    const start = document.getElementById('vs-start').value;
    const threshold = document.getElementById('vs-threshold').value;
    const lightcycle = document.getElementById('vs-lcycle').value;
    const plotAcrophase = document.getElementById('vs-acrophase').checked;

    if (!framerate || !binsize || !start || !threshold) return;

    document.getElementById('visualization-title').textContent = `Actogram: ${modelName} (${sessionDir})`;
    showActogramLoadingIndicator();

    if (exportBtn) exportBtn.disabled = false;

    try {
        await eel.generate_actograms(
            rootDir, sessionDir, modelName, behaviorNames,
            framerate, binsize, start, threshold, lightcycle, plotAcrophase,
            currentTaskId
        )();
    } catch (error) {
        console.error("Error calling eel.generate_actograms:", error);
        updateActogramDisplay([], currentTaskId);
        showErrorOnVisualizePage(`Failed to generate actogram(s): ${error.message || error}`);
        if (exportBtn) exportBtn.disabled = true;
    }
}

function handleBehaviorSelection(checkbox) {
    const rootDir = checkbox.dataset.root;
    const sessionDir = checkbox.dataset.session;
    const modelName = checkbox.dataset.model;

    if (currentSelection.root !== rootDir || currentSelection.session !== sessionDir || currentSelection.model !== modelName) {
        document.querySelectorAll('.behavior-checkbox').forEach(cb => {
            if (cb.dataset.model !== modelName || cb.dataset.session !== sessionDir) {
                cb.checked = false;
            }
        });
        currentSelection = { root: rootDir, session: sessionDir, model: modelName };
    }
    
    clearTimeout(actogramDebounceTimer);
    actogramDebounceTimer = setTimeout(generateAndDisplayActograms, 200);
}

async function initializeActogramUI() {
    const container = document.getElementById('directories');
    if (!container) return;

    try {
        const recordingTree = await eel.get_recording_tree()();
        if (!recordingTree || recordingTree.length === 0) {
            container.innerHTML = "<p class='text-light p-3'>No classified recordings available for actograms.</p>";
            return;
        }

        let htmlBuilder = '';
        recordingTree.forEach((dateEntry) => {
            const [dateStr, sessions] = dateEntry;
            const dateId = `rd-${dateStr.replace(/[\W_]+/g, '-')}`;
            htmlBuilder += `<h5 class='text-light mt-2 hand-cursor' onclick="toggleVisibility('${dateId}')"><i class="bi bi-calendar-date-fill me-2"></i>${dateStr}</h5>`;
            htmlBuilder += `<div id='${dateId}' class='ms-3' style="display:none;">`;

            sessions.forEach((sessionEntry) => {
                const [sessionName, models] = sessionEntry;
                const sessionId = `${dateId}-sd-${sessionName.replace(/[\W_]+/g, '-')}`;
                htmlBuilder += `<h6 class='text-light mt-1 hand-cursor' onclick="toggleVisibility('${sessionId}')"><i class="bi bi-camera-reels-fill me-2"></i>${sessionName}</h6>`;
                htmlBuilder += `<div id='${sessionId}' class='ms-3' style="display:none;">`;

                models.forEach((modelEntry) => {
                    const [modelName, behaviors] = modelEntry;
                    const modelId = `${sessionId}-md-${modelName.replace(/[\W_]+/g, '-')}`;
                    htmlBuilder += `<div class='text-info mt-1 hand-cursor' onclick="toggleVisibility('${modelId}')"><i class="bi bi-cpu-fill me-2"></i>${modelName}</div>`;
                    htmlBuilder += `<div id='${modelId}' class='ms-3' style="display:none;">`;
                    
                    behaviors.forEach((behaviorName) => {
                        const behaviorId = `${modelId}-beh-${behaviorName.replace(/[\W_]+/g, '-')}`;
                        htmlBuilder += `
                            <div class="form-check my-1">
                                <input class="form-check-input behavior-checkbox" type="checkbox" id="${behaviorId}" 
                                       data-root="${dateStr}" data-session="${sessionName}" data-model="${modelName}" data-behavior="${behaviorName}"
                                       onclick="handleBehaviorSelection(this)">
                                <label class="form-check-label text-light small" for="${behaviorId}">${behaviorName}</label>
                            </div>`;
                    });
                    htmlBuilder += `</div>`;
                });
                htmlBuilder += `</div>`;
            });
            htmlBuilder += `</div>`;
        });
        container.innerHTML = htmlBuilder;

    } catch (error) {
        console.error("Error initializing page:", error);
        container.innerHTML = "<p class='text-danger text-center'>Error loading data.</p>";
        showErrorOnVisualizePage(`Error loading data: ${error.message || error}`);
    }
}

async function exportActogramData() {
    const checkedBehaviors = Array.from(document.querySelectorAll('.behavior-checkbox:checked'));
    if (checkedBehaviors.length === 0) {
        showErrorOnVisualizePage("Please select at least one behavior to export.");
        return;
    }

    if (!window.electronAPI) {
        showErrorOnVisualizePage("Export function is not available in this environment.");
        return;
    }

    try {
        const folderPath = await window.electronAPI.invoke('show-folder-dialog');
        if (folderPath) {
            console.log("Folder path chosen, now calling Python to generate and save data.");
            
            const firstCheckbox = checkedBehaviors[0];
            const rootDir = firstCheckbox.dataset.root;
            const sessionDir = firstCheckbox.dataset.session;
            const modelName = firstCheckbox.dataset.model;
            const behaviorNames = checkedBehaviors.map(cb => cb.dataset.behavior);
    
            const framerate = document.getElementById('vs-framerate').value;
            const binsize = document.getElementById('vs-binsize').value;
            const start = document.getElementById('vs-start').value;
            const threshold = document.getElementById('vs-threshold').value;

            eel.generate_and_save_data(folderPath, rootDir, sessionDir, modelName, behaviorNames, framerate, binsize, start, threshold)();
        } else {
            console.log("User cancelled the folder selection dialog.");
        }
    } catch (err) {
        console.error("Folder selection error:", err);
        showErrorOnVisualizePage("Could not open the folder selection dialog.");
    }
}

async function initializeEthogramUI() {
    const container = document.getElementById('directories');
    if (!container) return;

    container.innerHTML = '<div class="text-center text-muted p-3">Loading videos...</div>';
    
    try {
        const videoTree = await eel.get_classified_video_tree()();
        if (!videoTree || videoTree.length === 0) {
            container.innerHTML = "<p class='text-light p-3'>No classified videos available.</p>";
            return;
        }

        let htmlBuilder = '';
        videoTree.forEach(sessionEntry => {
            const [sessionName, subjects] = sessionEntry;
            const sessionId = `etho-sess-${sessionName.replace(/[\W_]+/g, '-')}`;
            htmlBuilder += `<h5 class='text-light mt-2 hand-cursor' onclick="toggleVisibility('${sessionId}')"><i class="bi bi-camera-reels-fill me-2"></i>${sessionName}</h5>`;
            htmlBuilder += `<div id='${sessionId}' class='ms-3' style="display:none;">`;

            subjects.forEach(subjectEntry => {
                const [subjectName, videos] = subjectEntry;
                const subjectId = `${sessionId}-subj-${subjectName.replace(/[\W_]+/g, '-')}`;
                htmlBuilder += `<h6 class='text-info mt-1 hand-cursor' onclick="toggleVisibility('${subjectId}')"><i class="bi bi-person-fill me-2"></i>${subjectName}</h6>`;
                htmlBuilder += `<div id='${subjectId}' class='ms-3' style="display:none;">`;

                videos.forEach(video => {
                    htmlBuilder += `<div class="small hand-cursor text-light py-1" onclick="generateEthogram('${video.path}')"><i class="bi bi-film me-2"></i>${video.name}</div>`;
                });
                htmlBuilder += `</div>`;
            });
            htmlBuilder += `</div>`;
        });
        container.innerHTML = htmlBuilder;
    } catch (error) {
        console.error("Error initializing ethogram UI:", error);
        container.innerHTML = "<p class='text-danger text-center p-3'>Error loading video data.</p>";
    }
}

async function generateEthogram(videoPath) {
    const ethogramContainer = document.getElementById('ethogram-container');
    const spinner = document.getElementById('loading-spinner-ethogram');
    if (!ethogramContainer || !spinner) return;

    spinner.style.display = 'block';
    ethogramContainer.innerHTML = '';

    try {
        const result = await eel.generate_ethogram(videoPath)();
        if (result && result.blob) {
            ethogramContainer.innerHTML = `<img src="data:image/png;base64,${result.blob}" class="img-fluid" alt="Ethogram for ${result.name}">`;
        } else {
            ethogramContainer.innerHTML = '<p class="text-warning">Could not generate ethogram. The classification file might be empty or invalid.</p>';
        }
    } catch (error) {
        console.error("Error generating ethogram:", error);
        ethogramContainer.innerHTML = '<p class="text-danger">An error occurred while generating the plot.</p>';
    } finally {
        spinner.style.display = 'none';
    }
}

// =================================================================
// PAGE INITIALIZATION & EVENT LISTENERS
// =================================================================

document.addEventListener('DOMContentLoaded', async () => {
    // Call the ACTOGRAM initializer by default when the page loads
    initializeActogramUI();

    // Attach event listeners for the actogram controls
    const adjustmentControlsIds = ['vs-framerate', 'vs-binsize', 'vs-start', 'vs-threshold', 'vs-lcycle', 'vs-acrophase'];
    adjustmentControlsIds.forEach(controlId => {
        const elem = document.getElementById(controlId);
        if (elem) {
            const eventType = (elem.type === 'checkbox' || elem.tagName.toLowerCase() === 'select') ? 'change' : 'input';
            
            elem.addEventListener(eventType, () => {
                clearTimeout(actogramDebounceTimer);
                actogramDebounceTimer = setTimeout(generateAndDisplayActograms, 200);
            });
        }
    });
});