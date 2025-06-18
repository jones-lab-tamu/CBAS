// In frontend/js/record_page.js

/**
 * @file Manages the Record page UI, including live preview and the new interactive cropping tool.
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
        resetPreviewButton(activePreviewCamera);
    }
    
    console.log(`Live preview ended for ${cameraName}.`);
    if (cameraName === activePreviewCamera) {
        activePreviewCamera = null;
    }

    resetPreviewButton(cameraName);
    
    refreshSingleThumbnail(cameraName);
}

async function toggleLivePreview(cameraName) {
    if (activePreviewCamera && activePreviewCamera !== cameraName) {
        await eel.stop_live_preview()();
    }

    if (activePreviewCamera === cameraName) {
        console.log(`User stopping live preview for ${cameraName}`);
        await eel.stop_live_preview()();
    } else {
        console.log(`User starting live preview for ${cameraName}`);
        activePreviewCamera = cameraName;

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
        const thumbnailBlob = await eel.get_single_camera_thumbnail(cameraName)();
        if(thumbnailBlob){
            const canvas = document.getElementById(`camera-${cameraName}`);
            const ctx = canvas.getContext('2d');
            const img = new Image();
            img.onload = () => {
                 const crop = { x: camData.crop_left_x * img.naturalWidth, y: camData.crop_top_y * img.naturalHeight, w: camData.crop_width * img.naturalWidth, h: camData.crop_height * img.naturalHeight };
                 drawImageOnCanvas(img, ctx, crop.x, crop.y, crop.w, crop.h);
            };
            img.src = `data:image/jpeg;base64,${thumbnailBlob}`;
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
let allCameraData = [];

let addCameraBsModal, statusBsModal, cameraSettingsBsModal, generalErrorBsModal;

// --- STATE FOR INTERACTIVE CROPPING ---
let cropCanvas, cropCtx, imageCanvas, imageCtx;
let cropRect = { x: 0, y: 0, w: 0, h: 0 };
let isDragging = false;
let isResizing = false;
let resizeHandle = null;
let startCoords = { x: 0, y: 0 };
const handleSize = 8;

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
        const isCropped = cam.crop_left_x != 0 || cam.crop_top_y != 0 || cam.crop_width != 1 || cam.crop_height != 1;
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

// =================================================================
// INTERACTIVE CROPPING LOGIC
// =================================================================

async function loadCameraSettings(cameraName) {
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
            setupCropCanvas();
            drawImageOnCropCanvas(modalPreviewImage);
            updateCropRectFromInputs();
            drawCropOverlay();
        };
        modalPreviewImage.src = mainCanvasImageSrc || "assets/noConnection.png";

        cameraSettingsBsModal?.show();
    } else {
        showErrorOnRecordPage(`Could not find settings for camera: ${cameraName}`);
    }
}

function setupCropCanvas() {
    imageCanvas = document.getElementById("camera-image");
    imageCtx = imageCanvas.getContext("2d");
    cropCanvas = document.getElementById("crop-overlay");
    cropCtx = cropCanvas.getContext("2d");

    const container = document.getElementById('canvas-container-modal');
    imageCanvas.width = container.clientWidth;
    imageCanvas.height = container.clientHeight;
    cropCanvas.width = container.clientWidth;
    cropCanvas.height = container.clientHeight;
}

function drawImageOnCropCanvas(img) {
    if (!imageCanvas || !img.complete || img.naturalWidth === 0) return;
    const aspectRatio = img.naturalWidth / img.naturalHeight;
    const container = document.getElementById('canvas-container-modal');
    container.style.height = `${container.clientWidth / aspectRatio}px`;
    setupCropCanvas();
    imageCtx.drawImage(img, 0, 0, imageCanvas.width, imageCanvas.height);
}

function drawCropOverlay() {
    if (!cropCtx) return;
    cropCtx.clearRect(0, 0, cropCanvas.width, cropCanvas.height);

    cropCtx.fillStyle = "rgba(0, 0, 0, 0.5)";
    cropCtx.fillRect(0, 0, cropCanvas.width, cropCanvas.height);
    cropCtx.clearRect(cropRect.x, cropRect.y, cropRect.w, cropRect.h);

    cropCtx.strokeStyle = "rgba(255, 0, 0, 0.9)";
    cropCtx.lineWidth = 2;
    cropCtx.strokeRect(cropRect.x, cropRect.y, cropRect.w, cropRect.h);

    drawHandles();
}

function drawHandles() {
    const { x, y, w, h } = cropRect;
    const handles = {
        tl: { x: x, y: y }, tr: { x: x + w, y: y },
        bl: { x: x, y: y + h }, br: { x: x + w, y: y + h },
        t: { x: x + w / 2, y: y }, b: { x: x + w / 2, y: y + h },
        l: { x: x, y: y + h / 2 }, r: { x: x + w, y: y + h / 2 }
    };
    cropCtx.fillStyle = "rgba(255, 0, 0, 0.9)";
    for (const handle in handles) {
        cropCtx.fillRect(handles[handle].x - handleSize / 2, handles[handle].y - handleSize / 2, handleSize, handleSize);
    }
}

function getHandleAt(mouseX, mouseY) {
    const { x, y, w, h } = cropRect;
    const handles = {
        tl: { x: x, y: y }, tr: { x: x + w, y: y },
        bl: { x: x, y: y + h }, br: { x: x + w, y: y + h },
        t: { x: x + w / 2, y: y }, b: { x: x + w / 2, y: y + h },
        l: { x: x, y: y + h / 2 }, r: { x: x + w, y: y + h / 2 }
    };
    for (const handle in handles) {
        if (Math.abs(mouseX - handles[handle].x) < handleSize && Math.abs(mouseY - handles[handle].y) < handleSize) {
            return handle;
        }
    }
    return null;
}

function updateInputsFromCropRect() {
    if (!cropCanvas) return;
    document.getElementById('cs-cropx').value = (cropRect.x / cropCanvas.width).toFixed(4);
    document.getElementById('cs-cropy').value = (cropRect.y / cropCanvas.height).toFixed(4);
    document.getElementById('cs-crop-width').value = (cropRect.w / cropCanvas.width).toFixed(4);
    document.getElementById('cs-crop-height').value = (cropRect.h / cropCanvas.height).toFixed(4);
}

function updateCropRectFromInputs() {
    if (!cropCanvas) return;
    cropRect.x = parseFloat(document.getElementById('cs-cropx').value) * cropCanvas.width;
    cropRect.y = parseFloat(document.getElementById('cs-cropy').value) * cropCanvas.height;
    cropRect.w = parseFloat(document.getElementById('cs-crop-width').value) * cropCanvas.width;
    cropRect.h = parseFloat(document.getElementById('cs-crop-height').value) * cropCanvas.height;
    drawCropOverlay();
}

function onMouseDown(e) {
    const { offsetX, offsetY } = e;
    startCoords = { x: offsetX, y: offsetY };
    resizeHandle = getHandleAt(offsetX, offsetY);
    if (resizeHandle) {
        isResizing = true;
    } else if (offsetX > cropRect.x && offsetX < cropRect.x + cropRect.w && offsetY > cropRect.y && offsetY < cropRect.y + cropRect.h) {
        isDragging = true;
    }
}

function onMouseMove(e) {
    const { offsetX, offsetY, movementX, movementY } = e;
    if (isDragging) {
        cropRect.x += movementX;
        cropRect.y += movementY;
    } else if (isResizing) {
        if (resizeHandle.includes('l')) { cropRect.x += movementX; cropRect.w -= movementX; }
        if (resizeHandle.includes('r')) { cropRect.w += movementX; }
        if (resizeHandle.includes('t')) { cropRect.y += movementY; cropRect.h -= movementY; }
        if (resizeHandle.includes('b')) { cropRect.h += movementY; }
    } else {
        const handle = getHandleAt(offsetX, offsetY);
        if (handle) {
            if (handle === 'tl' || handle === 'br') cropCanvas.style.cursor = 'nwse-resize';
            else if (handle === 'tr' || handle === 'bl') cropCanvas.style.cursor = 'nesw-resize';
            else if (handle === 't' || handle === 'b') cropCanvas.style.cursor = 'ns-resize';
            else if (handle === 'l' || handle === 'r') cropCanvas.style.cursor = 'ew-resize';
        } else if (offsetX > cropRect.x && offsetX < cropRect.x + cropRect.w && offsetY > cropRect.y && offsetY < cropRect.y + cropRect.h) {
            cropCanvas.style.cursor = 'move';
        } else {
            cropCanvas.style.cursor = 'crosshair';
        }
    }
    if (isDragging || isResizing) {
        drawCropOverlay();
        updateInputsFromCropRect();
    }
}

function onMouseUp(e) {
    isDragging = false;
    isResizing = false;
    resizeHandle = null;
}

// =================================================================
// PAGE INITIALIZATION
// =================================================================

document.addEventListener('DOMContentLoaded', () => {
    const addCameraModalElement = document.getElementById('addCamera');
    const statusModalElement = document.getElementById('statusModal');
    const cameraSettingsModalElement = document.getElementById('cameraSettings');
    const errorModalElement = document.getElementById('errorModal');
    
    if (addCameraModalElement) addCameraBsModal = new bootstrap.Modal(addCameraModalElement);
    if (statusModalElement) statusBsModal = new bootstrap.Modal(statusModalElement);
    if (cameraSettingsModalElement) cameraSettingsBsModal = new bootstrap.Modal(cameraSettingsModalElement);
    if (errorModalElement) generalErrorBsModal = new bootstrap.Modal(errorModalElement);
    
    loadCameras();
    setInterval(updateStatusIcon, 3000);
    
    document.getElementById('addCameraButton')?.addEventListener('click', addCameraSubmit);
    document.querySelector('.fab-container-left .fab')?.addEventListener('click', loadCameras);
    
    cropCanvas = document.getElementById("crop-overlay");
    if(cropCanvas) {
        cropCanvas.addEventListener('mousedown', onMouseDown);
        cropCanvas.addEventListener('mousemove', onMouseMove);
        cropCanvas.addEventListener('mouseup', onMouseUp);
        cropCanvas.addEventListener('mouseleave', () => { isDragging = false; isResizing = false; });
    }

    const cropInputs = ['cs-cropx', 'cs-cropy', 'cs-crop-width', 'cs-crop-height'];
    cropInputs.forEach(id => {
        document.getElementById(id)?.addEventListener('input', updateCropRectFromInputs);
    });

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