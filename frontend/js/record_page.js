/**
 * @file Manages the Record page UI, including the new in-app live preview feature.
 * This version uses a robust request-response model for loading initial data.
 */

// =================================================================
// EEL-EXPOSED FUNCTIONS (API for Python to call)
// =================================================================

eel.expose(update_log_panel);
eel.expose(update_live_frame);
eel.expose(end_live_preview);

// =================================================================
// LOG PANEL MANAGEMENT
// =================================================================

function update_log_panel(message) {
    const logContainer = document.getElementById('log-panel-content');
    if (!logContainer) return;

    let logHistory = JSON.parse(sessionStorage.getItem('logHistory') || '[]');
    logHistory.push(message);
    while (logHistory.length > 500) logHistory.shift();
    sessionStorage.setItem('logHistory', JSON.stringify(logHistory));

    renderLogMessage(message, logContainer);
    logContainer.scrollTop = logContainer.scrollHeight;
}

function renderLogMessage(message, container) {
    const logEntry = document.createElement('div');
    logEntry.className = 'log-message';

    if (message.includes('[ERROR]')) logEntry.classList.add('log-level-ERROR');
    else if (message.includes('[WARN]')) logEntry.classList.add('log-level-WARN');
    else logEntry.classList.add('log-level-INFO');
    
    logEntry.textContent = message;
    container.appendChild(logEntry);
}


// =================================================================
// LIVE PREVIEW LOGIC
// =================================================================

let activePreviewCamera = null; // Holds the name of the camera currently in live preview mode

function update_live_frame(cameraName, base64Val) {
    if (cameraName !== activePreviewCamera) return; // Ignore frames for non-active previews

    const canvas = document.getElementById(`camera-${cameraName}`);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const img = new Image();

    img.onload = () => {
        const camData = allCameraData.find(c => c.name === cameraName);
        if (camData) {
            const crop = {
                x: camData.crop_left_x * img.naturalWidth, y: camData.crop_top_y * img.naturalHeight,
                w: camData.crop_width * img.naturalWidth, h: camData.crop_height * img.naturalHeight
            };
            drawImageOnCanvas(img, ctx, crop.x, crop.y, crop.w, crop.h);
        } else {
             ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        }
    };
    img.src = `data:image/jpeg;base64,${base64Val}`;
}

function end_live_preview(cameraName) {
    if (activePreviewCamera && cameraName !== activePreviewCamera) {
         // A new preview started, so we need to reset the button of the *old* one.
        resetPreviewButton(activePreviewCamera);
    }
    
    console.log(`Live preview ended for ${cameraName}.`);
    if (cameraName === activePreviewCamera) {
        activePreviewCamera = null;
    }

    resetPreviewButton(cameraName);
    
    // Fetch a single static thumbnail to replace the live view
    refreshSingleThumbnail(cameraName);
}

async function toggleLivePreview(cameraName) {
    // If a different camera is already live, stop it first.
    if (activePreviewCamera && activePreviewCamera !== cameraName) {
        await eel.stop_live_preview()();
    }

    if (activePreviewCamera === cameraName) {
        // If we are clicking the button for the camera that is already live, stop it.
        console.log(`User stopping live preview for ${cameraName}`);
        await eel.stop_live_preview()();
    } else {
        // Start the new preview.
        console.log(`User starting live preview for ${cameraName}`);
        activePreviewCamera = cameraName;

        // Change button appearance immediately for responsiveness
        const liveViewBtn = document.getElementById(`live-view-btn-${cameraName}`);
        if(liveViewBtn){
            liveViewBtn.classList.remove('btn-outline-light');
            liveViewBtn.classList.add('btn-warning');
            liveViewBtn.innerHTML = '<i class="bi bi-stop-circle-fill"></i>';
            const tooltip = bootstrap.Tooltip.getInstance(liveViewBtn);
            if(tooltip) tooltip.setContent({ '.tooltip-inner': 'Stop Preview' });
        }

        await eel.start_live_preview(cameraName)();
    }
}

async function refreshSingleThumbnail(cameraName){
    const camData = allCameraData.find(c => c.name === cameraName);
    if(!camData) return;

    try {
        // Call the NEW, DEDICATED backend function that takes one argument.
        const thumbnailBlob = await eel.get_single_camera_thumbnail(cameraName)();
        
        if (thumbnailBlob) {
            const canvas = document.getElementById(`camera-${cameraName}`);
            const ctx = canvas.getContext('2d');
            const img = new Image();
            img.onload = () => {
                 const crop = { x: camData.crop_left_x * img.naturalWidth, y: camData.crop_top_y * img.naturalHeight, w: camData.crop_width * img.naturalWidth, h: camData.crop_height * img.naturalHeight };
                 drawImageOnCanvas(img, ctx, crop.x, crop.y, crop.w, crop.h);
            };
            img.src = `data:image/jpeg;base64,${thumbnailBlob}`;
        } else {
             // Fallback if the refresh fails, draw the "no connection" image
             const canvas = document.getElementById(`camera-${cameraName}`);
             const ctx = canvas.getContext('2d');
             const placeholder = new Image();
             placeholder.onload = () => ctx.drawImage(placeholder, 0, 0, canvas.width, canvas.height);
             placeholder.src = "assets/noConnection.png";
        }
    } catch(e) {
        console.error(`Failed to refresh single thumbnail for ${cameraName}:`, e);
    }
}

function resetPreviewButton(cameraName) {
    const liveViewBtn = document.getElementById(`live-view-btn-${cameraName}`);
    if (liveViewBtn) {
        liveViewBtn.classList.remove('btn-warning');
        liveViewBtn.classList.add('btn-outline-light');
        liveViewBtn.innerHTML = '<i class="bi bi-eye-fill"></i>';
        const tooltip = bootstrap.Tooltip.getInstance(liveViewBtn);
        if(tooltip) tooltip.setContent({ '.tooltip-inner': 'Live Preview' });
    }
    const liveViewBtnRec = document.getElementById(`live-view-btn-recording-${cameraName}`);
     if (liveViewBtnRec) {
        liveViewBtnRec.classList.remove('btn-warning');
        liveViewBtnRec.classList.add('btn-outline-light');
        liveViewBtnRec.innerHTML = '<i class="bi bi-eye-fill"></i>';
        const tooltip = bootstrap.Tooltip.getInstance(liveViewBtnRec);
        if(tooltip) tooltip.setContent({ '.tooltip-inner': 'Live Preview' });
    }
}


// =================================================================
// GLOBAL STATE & VARIABLES
// =================================================================

let routingInProgress = false;
let originalCameraNameForSettings = "";
let modalPreviewImage = new Image();
let allCameraData = []; // Cache for camera data to avoid re-fetching

let addCameraBsModal, statusBsModal, cameraSettingsBsModal, generalErrorBsModal;

// =================================================================
// UI INTERACTION & EVENT HANDLERS
// =================================================================

function routeToRecordPage() { routingInProgress = true; window.location.href = './record.html'; }
function routeToLabelTrainPage() { routingInProgress = true; window.location.href = './label-train.html'; }
function routeToVisualizePage() { routingInProgress = true; window.location.href = './visualize.html'; }

function showErrorOnRecordPage(message) {
    const el = document.getElementById("error-message");
    if (el && generalErrorBsModal) {
        el.innerText = message;
        generalErrorBsModal.show();
    } else {
        alert(message);
    }
}

function showAddCameraModal() { addCameraBsModal?.show(); }

async function showStatusModal() {
    try {
        const status = await eel.get_cbas_status()();
        document.getElementById("status-streams").innerText = status.streams ? "Recording: " + status.streams.join(", ") : "No cameras recording.";
        document.getElementById("status-encode-count").innerText = "Files to encode: " + status.encode_file_count;
        statusBsModal?.show();
    } catch (e) { console.error("Get status error:", e); }
}

async function addCameraSubmit() {
    const name = document.getElementById('camera-name-modal-input').value;
    const rtsp = document.getElementById('rtsp-url-modal-input').value;
    if (!name.trim() || !rtsp.trim()) { showErrorOnRecordPage('Name and RTSP URL are required.'); return; }
    
    const success = await eel.create_camera(name, rtsp)();
    if (success) {
        addCameraBsModal?.hide();
        document.getElementById('camera-name-modal-input').value = "";
        document.getElementById('rtsp-url-modal-input').value = "";
        await loadCameras();
    } else {
        showErrorOnRecordPage(`Failed to create camera '${name}'. It may already exist.`);
    }
}

async function startCamera(cameraName) {
    const sessionName = document.getElementById('session-name-input').value;
    if (!sessionName.trim()) {
        showErrorOnRecordPage('Please enter a Session Name before starting a recording.');
        return;
    }
    await eel.start_camera_stream(cameraName, sessionName, 600)();
    await updateRecordingStatus();
}

async function stopCamera(cameraName) {
    await eel.stop_camera_stream(cameraName)();
    await updateRecordingStatus();
}

async function startAllCameras() {
    if (allCameraData.length > 0) {
        for (const cam of allCameraData) await startCamera(cam.name);
    } else {
        alert("No cameras are configured to start.");
    }
}

async function stopAllCameras() {
    const activeStreams = await eel.get_active_streams()() || [];
    if (activeStreams.length > 0) {
        for (const name of activeStreams) await stopCamera(name);
    }
}

async function saveCameraSettings() {
    const newName = document.getElementById('cs-name').value;
    if (!newName.trim()) { showErrorOnRecordPage("Camera name cannot be empty."); return; }
    
    const settings = {
        "name": newName,
        "rtsp_url": document.getElementById('cs-url').value,
        "framerate": parseInt(document.getElementById('cs-framerate').value) || 10,
        "resolution": parseInt(document.getElementById('cs-resolution').value) || 256,
        'crop_left_x': parseFloat(document.getElementById('cs-cropx').value) || 0,
        'crop_top_y': parseFloat(document.getElementById('cs-cropy').value) || 0,
        'crop_width': parseFloat(document.getElementById('cs-crop-width').value) || 1,
        'crop_height': parseFloat(document.getElementById('cs-crop-height').value) || 1,
    };

    if (newName !== originalCameraNameForSettings) {
        const renameSuccess = await eel.rename_camera(originalCameraNameForSettings, newName)();
        if (!renameSuccess) { showErrorOnRecordPage(`Failed to rename. '${newName}' may exist.`); return; }
    }
    
    await eel.save_camera_settings(newName, settings)();
    cameraSettingsBsModal?.hide();
    await loadCameras();
}

// =================================================================
// CORE APPLICATION LOGIC
// =================================================================

async function loadCameras() {
    const container = document.getElementById('camera-container');
    const spinner = document.getElementById('cover-spin');
    if (!container || !spinner) return;

    spinner.style.visibility = 'visible';
    container.innerHTML = "";

    try {
        allCameraData = await eel.get_cameras_with_thumbnails()();
        
        if (!allCameraData || allCameraData.length === 0) {
            container.innerHTML = "<div class='col'><p class='text-light text-center mt-3'>No cameras configured. Click the '+' button to add one.</p></div>";
            spinner.style.visibility = 'hidden';
            return;
        }

        await loadCameraHTMLCards();

        for (const cam of allCameraData) {
            const canvas = document.getElementById(`camera-${cam.name}`);
            const ctx = canvas.getContext('2d');
            const img = new Image();

            img.onload = () => {
                canvas.setAttribute("cbas_image_source", img.src);
                const crop = { x: cam.crop_left_x * img.naturalWidth, y: cam.crop_top_y * img.naturalHeight, w: cam.crop_width * img.naturalWidth, h: cam.crop_height * img.naturalHeight };
                drawImageOnCanvas(img, ctx, crop.x, crop.y, crop.w, crop.h);
            };

            if (cam.thumbnail_blob) {
                img.src = `data:image/jpeg;base64,${cam.thumbnail_blob}`;
            } else {
                img.src = "assets/noConnection.png";
            }
        }
        
        await updateRecordingStatus();
        
        // Re-initialize tooltips after new HTML is added
        var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'))
        var tooltipList = tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl)
        })

    } catch (error) {
        console.error("Failed to load cameras:", error);
        showErrorOnRecordPage("An error occurred while fetching camera data.");
    } finally {
        spinner.style.visibility = 'hidden';
    }
}

async function loadCameraHTMLCards() {
    const container = document.getElementById('camera-container');
    if (!container) return;
    container.innerHTML = "";

    let htmlContent = "";
    for (const cam of allCameraData) {
        const isCropped = cam.crop_left_x !== 0 || cam.crop_top_y !== 0 || cam.crop_width !== 1 || cam.crop_height !== 1;
        const displayName = isCropped ? cam.name : `${cam.name} <small class='text-muted'>(uncropped)</small>`;
        
        htmlContent += `
            <div class="col-auto mb-3">
                <div class="card shadow text-white bg-dark" style="width: 320px;">
                    <div class="card-header py-2"><h5 class="card-title mb-0">${displayName}</h5></div>
                    <canvas id="camera-${cam.name}" width="300" height="225" style="background-color: #343a40; display: block; margin: auto; margin-top:10px;"></canvas>
                    <div class="card-footer d-flex justify-content-center p-2">
                        <div id="before-recording-${cam.name}" style="display: flex;">
                            <button class="btn btn-sm btn-outline-light me-1" onclick="loadCameraSettings('${cam.name}')" data-bs-toggle="tooltip" data-bs-placement="bottom" title="Settings/Crop"><i class="bi bi-crop"></i></button>
                            <button id="live-view-btn-${cam.name}" class="btn btn-sm btn-outline-light me-1" onclick="toggleLivePreview('${cam.name}')" data-bs-toggle="tooltip" data-bs-placement="bottom" title="Live Preview"><i class="bi bi-eye-fill"></i></button>
                            <button class="btn btn-sm btn-success" onclick="startCamera('${cam.name}')" data-bs-toggle="tooltip" data-bs-placement="bottom" title="Start Recording"><i class="bi bi-camera-video-fill"></i> Start</button>
                        </div>
                        <div id="during-recording-${cam.name}" style="display: none;">
                             <button id="live-view-btn-recording-${cam.name}" class="btn btn-sm btn-outline-light me-1" onclick="toggleLivePreview('${cam.name}')" data-bs-toggle="tooltip" data-bs-placement="bottom" title="Live Preview"><i class="bi bi-eye-fill"></i></button>
                            <button class="btn btn-sm btn-danger" onclick="stopCamera('${cam.name}')" data-bs-toggle="tooltip" data-bs-placement="bottom" title="Stop Recording"><i class="bi bi-square-fill"></i> Stop</button>
                        </div>
                    </div>
                </div>
            </div>`;
    }
    container.innerHTML = htmlContent;
}

async function updateRecordingStatus() {
    try {
        const activeStreams = await eel.get_active_streams()() || [];
        setRecordAllIcon(activeStreams.length > 0);

        for (const cam of allCameraData) {
            const beforeRec = document.getElementById(`before-recording-${cam.name}`);
            const duringRec = document.getElementById(`during-recording-${cam.name}`);
            if (beforeRec && duringRec) {
                const isActive = activeStreams.includes(cam.name);
                beforeRec.style.display = isActive ? 'none' : 'flex';
                duringRec.style.display = isActive ? 'flex' : 'none';
            }
        }
    } catch(e) {
        console.error("Could not update recording status:", e);
    }
}

async function updateStatusIcon() {
    const icon = document.getElementById("status-camera-icon");
    if (!icon) return;
    try {
        const status = await eel.get_cbas_status()();
        const isStreaming = status && status.streams && status.streams.length > 0;
        icon.style.color = isStreaming ? "red" : "white";
        isStreaming ? icon.classList.add("blinking") : icon.classList.remove("blinking");
    } catch(e) {/* fail silently */}
}

function setRecordAllIcon(isAnyRecording) {
    const fab = document.querySelector('.fab-container-right .fab[onclick*="All"]');
    if (!fab) return;
    const icon = fab.querySelector('i');
    const tooltip = bootstrap.Tooltip.getInstance(fab);

    if (isAnyRecording) {
        icon.className = 'bi bi-square-fill';
        fab.setAttribute('onclick', 'stopAllCameras()');
        if (tooltip) tooltip.setContent({ '.tooltip-inner': 'Stop All Recordings' });
    } else {
        icon.className = 'bi bi-camera-video-fill';
        fab.setAttribute('onclick', 'startAllCameras()');
        if (tooltip) tooltip.setContent({ '.tooltip-inner': 'Start All Recordings' });
    }
}

function drawImageOnCanvas(img, ctx, sx, sy, sw, sh) {
    const canvas = ctx.canvas;
    if (sw <= 0 || sh <= 0) { ctx.clearRect(0, 0, canvas.width, canvas.height); return; }
    const ratio = Math.min(canvas.width / sw, canvas.height / sh);
    const drawW = sw * ratio, drawH = sh * ratio;
    const destX = (canvas.width - drawW) / 2;
    const destY = (canvas.height - drawH) / 2;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, sx, sy, sw, sh, destX, destY, drawW, drawH);
}

async function loadCameraSettings(cameraName) {
    const modalCanvas = document.getElementById("camera-image");
    if (!modalCanvas) return;

    const settings = allCameraData.find(c => c.name === cameraName);
    if (settings) {
        originalCameraNameForSettings = cameraName;
        document.getElementById('cs-name').value = settings.name;
        document.getElementById('cs-url').value = settings.rtsp_url;
        document.getElementById('cs-framerate').value = settings.framerate;
        document.getElementById('cs-resolution').value = settings.resolution;
        document.getElementById('cs-cropx').value = settings.crop_left_x;
        document.getElementById('cs-cropy').value = settings.crop_top_y;
        document.getElementById('cs-crop-width').value = settings.crop_width;
        document.getElementById('cs-crop-height').value = settings.crop_height;

        const mainCanvasImageSrc = document.getElementById(`camera-${cameraName}`)?.getAttribute("cbas_image_source");
        
        modalPreviewImage.onload = () => {
            const aspectRatio = modalPreviewImage.naturalWidth / modalPreviewImage.naturalHeight;
            modalCanvas.width = 600;
            modalCanvas.height = modalCanvas.width / aspectRatio;
            drawBoundsOnModalCanvas(modalPreviewImage);
        };
        modalPreviewImage.src = mainCanvasImageSrc || "assets/noConnection.png";

        cameraSettingsBsModal?.show();
    } else {
        showErrorOnRecordPage(`Could not find settings for camera: ${cameraName}`);
    }
}

function drawBoundsOnModalCanvas(imageToDraw) {
    const modalCanvas = document.getElementById("camera-image");
    if (!modalCanvas || !imageToDraw.complete || imageToDraw.naturalWidth === 0) return;
    const modalCtx = modalCanvas.getContext("2d");

    let cx = parseFloat(document.getElementById('cs-cropx').value) || 0;
    let cy = parseFloat(document.getElementById('cs-cropy').value) || 0;
    let cw = parseFloat(document.getElementById('cs-crop-width').value) || 1;
    let ch = parseFloat(document.getElementById('cs-crop-height').value) || 1;

    modalCtx.clearRect(0, 0, modalCanvas.width, modalCanvas.height);
    modalCtx.drawImage(imageToDraw, 0, 0, modalCanvas.width, modalCanvas.height);
    
    const sx = cx * modalCanvas.width, sy = cy * modalCanvas.height;
    const sw = (cw * modalCanvas.width), sh = (ch * modalCanvas.height);

    modalCtx.strokeStyle = 'rgba(255, 0, 0, 0.9)';
    modalCtx.lineWidth = 2;
    modalCtx.strokeRect(sx, sy, sw, sh);
}

// =================================================================
// PAGE INITIALIZATION
// =================================================================

document.addEventListener('DOMContentLoaded', () => {
    // Initialize Bootstrap Components
    const addCameraModalElement = document.getElementById('addCamera');
    const statusModalElement = document.getElementById('statusModal');
    const cameraSettingsModalElement = document.getElementById('cameraSettings');
    const errorModalElement = document.getElementById('errorModal');
    
    if (addCameraModalElement) addCameraBsModal = new bootstrap.Modal(addCameraModalElement);
    if (statusModalElement) statusBsModal = new bootstrap.Modal(statusModalElement);
    if (cameraSettingsModalElement) cameraSettingsBsModal = new bootstrap.Modal(cameraSettingsModalElement);
    if (errorModalElement) generalErrorBsModal = new bootstrap.Modal(errorModalElement);
    
    loadCameras(); // Initial load
    
    setInterval(updateStatusIcon, 3000);
    
    // Attach event listeners
    document.getElementById('addCameraButton')?.addEventListener('click', addCameraSubmit);
    // Bind the refresh button
    document.querySelector('.fab-container-left .fab')?.addEventListener('click', loadCameras);

    const settingIds = ['cs-cropx', 'cs-cropy', 'cs-crop-width', 'cs-crop-height'];
    settingIds.forEach(id => {
        document.getElementById(id)?.addEventListener('input', () => drawBoundsOnModalCanvas(modalPreviewImage));
    });

    // Log panel logic
    const logContainer = document.getElementById('log-panel-content');
    if (logContainer) {
        const logHistory = JSON.parse(sessionStorage.getItem('logHistory') || '[]');
        logHistory.forEach(msg => renderLogMessage(msg, logContainer));
        logContainer.scrollTop = logContainer.scrollHeight;
        
        document.getElementById('clear-log-btn')?.addEventListener('click', () => {
            logContainer.innerHTML = '';
            sessionStorage.setItem('logHistory', '[]'); 
            update_log_panel('Log cleared.');
        });
        
        const logCollapseElement = document.getElementById('log-panel-collapse');
        const fabLeft = document.querySelector('.fab-container-left');
        const fabRight = document.querySelector('.fab-container-right');

        if(logCollapseElement && fabLeft && fabRight){
            const fabUpPosition = `${200 + 45 + 5}px`; 
            const fabDownPosition = '65px';
            logCollapseElement.addEventListener('show.bs.collapse', () => {
                fabLeft.style.bottom = fabUpPosition;
                fabRight.style.bottom = fabUpPosition;
            });
            logCollapseElement.addEventListener('hide.bs.collapse', () => {
                fabLeft.style.bottom = fabDownPosition;
                fabRight.style.bottom = fabDownPosition;
            });
        }
    }
});

window.addEventListener("beforeunload", () => {
    if (!routingInProgress) eel.kill_streams()?.catch(console.error);
});