/**
 * @file Manages the Record page UI, including live preview and interactive cropping.
 */

// =================================================================
// 1. EEL-EXPOSED FUNCTIONS (API for Python to call)
// =================================================================

eel.expose(update_live_frame);
eel.expose(end_live_preview);
eel.expose(updateImageSrc);

// =================================================================
// 2. GLOBAL STATE & VARIABLES
// =================================================================

let routingInProgress = false;
let originalCameraNameForSettings = "";
let modalPreviewImage = new Image();
let allCameraData = [];

let addCameraBsModal, statusBsModal, cameraSettingsBsModal, generalErrorBsModal;

let cropCanvas, cropCtx, imageCanvas, imageCtx;
let cropRect = { x: 0, y: 0, w: 0, h: 0 };
let isDragging = false;
let isResizing = false;
let resizeHandle = null;
const handleSize = 8;
let activePreviewCamera = null;

let camerasToFetchCount = 0;
let camerasFetchedCount = 0;
let activeStreamsInfo = {}; // Will store { camName: startTime, ... }
let recordingTimerInterval = null;

// =================================================================
// 3. FUNCTION DEFINITIONS
// =================================================================

function waitForEelConnection() {
    return new Promise(resolve => {
        if (eel._websocket && eel._websocket.readyState === 1) {
            resolve();
            return;
        }
        const interval = setInterval(() => {
            if (eel._websocket && eel._websocket.readyState === 1) {
                clearInterval(interval);
                resolve();
            }
        }, 100);
    });
}

function updateRecordingTimers() {
    // This function will only run if there are active streams
    if (Object.keys(activeStreamsInfo).length === 0) {
        clearInterval(recordingTimerInterval);
        recordingTimerInterval = null;
        return;
    }

    const now = Date.now() / 1000; // Get current time in seconds

    for (const cameraName in activeStreamsInfo) {
        const timerElement = document.getElementById(`timer-${cameraName}`);
        if (timerElement) {
            const startTime = activeStreamsInfo[cameraName];
            const elapsedTime = now - startTime;

            if (elapsedTime < 0) continue;

            const hours = Math.floor(elapsedTime / 3600);
            const minutes = Math.floor((elapsedTime % 3600) / 60);
            const seconds = Math.floor(elapsedTime % 60);

            let displayTime;
            let tooltipTime = `${hours}h ${minutes}m ${seconds}s`; // Full, precise time for tooltip

            // Change format for very long recordings
            if (hours < 100) {
                // For recordings under 100 hours, use HH:MM:SS
                displayTime = 
                    String(hours).padStart(2, '0') + ':' +
                    String(minutes).padStart(2, '0') + ':' +
                    String(seconds).padStart(2, '0');
            } else {
                // For recordings over 100 hours, switch to a more compact Day/Hour format
                const days = Math.floor(hours / 24);
                const remainingHours = hours % 24;
                displayTime = `${days}d ${String(remainingHours).padStart(2, '0')}h`;
            }
            
            // Update both the visible text and the hover-over tooltip
            timerElement.textContent = displayTime;

            // Update the tooltip to always show the precise time on hover
            const tooltip = bootstrap.Tooltip.getInstance(timerElement);
            if (tooltip) {
                tooltip.setContent({ '.tooltip-inner': tooltipTime });
            } else {
                // If tooltip not initialized, set the title attribute and initialize it
                timerElement.setAttribute('data-bs-toggle', 'tooltip');
                timerElement.setAttribute('data-bs-placement', 'top');
                timerElement.setAttribute('title', tooltipTime);
                new bootstrap.Tooltip(timerElement);
            }
        }
    }
}

function revealRecordingFolder(sessionName, cameraName) {
    if (!sessionName || !cameraName) {
        showErrorOnRecordPage("Cannot open folder without a session and camera name.");
        return;
    }
    eel.reveal_recording_folder(sessionName, cameraName)();
}

function updateImageSrc(cameraName, base64Val) {
    const spinner = document.getElementById(`spinner-${cameraName}`);
    const canvas = document.getElementById(`camera-${cameraName}`);
    if (!canvas || !spinner) return;

    const wasCached = sessionStorage.getItem(`thumbnail_${cameraName}`) !== null;
    if (!wasCached) {
        camerasFetchedCount++;
        if (camerasToFetchCount > 0) {
            const percent = (camerasFetchedCount / camerasToFetchCount) * 100;
            update_thumbnail_progress(percent, `Connecting... (${camerasFetchedCount}/${camerasToFetchCount})`);
        }
    }

    spinner.style.display = 'none';
    canvas.style.display = 'block';
    
    const ctx = canvas.getContext('2d');
    const img = new Image();
    
    img.onload = () => {
        canvas.setAttribute("cbas_image_source", img.src); 
        const camSettings = allCameraData.find(c => c.name === cameraName);
        if (camSettings) {
            const cropX = camSettings.crop_left_x * img.width;
            const cropY = camSettings.crop_top_y * img.height;
            const cropW = camSettings.crop_width * img.width;
            const cropH = camSettings.crop_height * img.height;
            drawImageScaled(img, ctx, cropX, cropY, cropW, cropH);
        } else {
            drawImageScaled(img, ctx, 0, 0, img.width, img.height);
        }
        updateRecordingStatus();
    };

    if (base64Val) {
        img.src = `data:image/jpeg;base64,${base64Val}`;
        try {
            sessionStorage.setItem(`thumbnail_${cameraName}`, img.src);
        } catch (e) {
            console.warn("Could not write to sessionStorage. Cache might be full.", e);
        }
    } else {
        img.src = "assets/noConnection.png"; 
    }
}

function clearThumbnailCache() {
    console.log("Clearing thumbnail cache...");
    Object.keys(sessionStorage).forEach(key => {
        if (key.startsWith('thumbnail_')) {
            sessionStorage.removeItem(key);
        }
    });
}

function update_thumbnail_progress(percent, message) {
    const container = document.getElementById('thumbnail-progress-container');
    const bar = document.getElementById('thumbnail-progress-bar');
    const label = document.getElementById('thumbnail-progress-label');
    if (!container || !bar || !label) return;
    
    if (camerasToFetchCount > 0) {
        container.style.display = 'block';
        label.textContent = message;
        const displayPercent = Math.round(percent);
        bar.style.width = `${displayPercent}%`;
        bar.textContent = `${displayPercent}%`;
        bar.setAttribute('aria-valuenow', displayPercent);
        
        if (percent >= 100) {
            setTimeout(() => { container.style.display = 'none'; }, 2000);
        }
    } else {
        container.style.display = 'none';
    }
}

function drawImageScaled(img, ctx, sx = 0, sy = 0, sw = img.width, sh = img.height) {
    var canvas = ctx.canvas;
    var hRatio = canvas.width / sw;
    var vRatio = canvas.height / sh;
    var ratio = Math.min(hRatio, vRatio);
    var centerShift_x = (canvas.width - sw * ratio) / 2;
    var centerShift_y = (canvas.height - sh * ratio) / 2;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(img, sx, sy, sw, sh, centerShift_x, centerShift_y, sw * ratio, sh * ratio);
}

function update_live_frame(cameraName, base64Val) {
    if (cameraName !== activePreviewCamera) return;
    const canvas = document.getElementById(`camera-${cameraName}`);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.onload = () => {
        drawImageScaled(img, ctx, 0, 0, img.width, img.height);
    };
    img.src = `data:image/jpeg;base64,${base64Val}`;
}

function end_live_preview(cameraName) {
    if (cameraName === activePreviewCamera) {
        activePreviewCamera = null;
    }
    resetPreviewButton(cameraName);
    const spinner = document.getElementById(`spinner-${cameraName}`);
    if (spinner) spinner.style.display = 'block';
    eel.get_single_camera_thumbnail(cameraName)();
}

async function toggleLivePreview(cameraName) {
    const previouslyActive = activePreviewCamera;
    const clickedIsActive = (previouslyActive === cameraName);

    if (previouslyActive) {
        await eel.stop_live_preview()(); 
    }

    if (!clickedIsActive) {
        activePreviewCamera = cameraName;
        setPreviewButtonActive(cameraName);
        await eel.start_live_preview(cameraName)();
    } else {
        activePreviewCamera = null;
    }
}

function setPreviewButtonActive(cameraName) {
    const liveViewBtn = document.getElementById(`live-view-btn-${cameraName}`);
    if(liveViewBtn){
        liveViewBtn.classList.remove('btn-outline-light');
        liveViewBtn.classList.add('btn-warning');
        liveViewBtn.innerHTML = '<i class="bi bi-stop-circle-fill"></i>';
        bootstrap.Tooltip.getInstance(liveViewBtn)?.setContent({ '.tooltip-inner': 'Stop Preview' });
    }
}

function resetPreviewButton(cameraName) {
    const liveViewBtn = document.getElementById(`live-view-btn-${cameraName}`);
    if (liveViewBtn) {
        liveViewBtn.classList.remove('btn-warning');
        liveViewBtn.classList.add('btn-outline-light');
        liveViewBtn.innerHTML = '<i class="bi bi-eye-fill"></i>';
        bootstrap.Tooltip.getInstance(liveViewBtn)?.setContent({ '.tooltip-inner': 'Live Preview' });
    }
    const liveViewBtnRec = document.getElementById(`live-view-btn-recording-${cameraName}`);
     if (liveViewBtnRec) {
        liveViewBtnRec.classList.remove('btn-warning');
        liveViewBtnRec.classList.add('btn-outline-light');
        liveViewBtnRec.innerHTML = '<i class="bi bi-eye-fill"></i>';
        bootstrap.Tooltip.getInstance(liveViewBtnRec)?.setContent({ '.tooltip-inner': 'Live Preview' });
    }
}


// The main toggle function is now robust against race conditions.
async function toggleLivePreview(cameraName) {
    const previouslyActive = activePreviewCamera;
    const clickedActive = (previouslyActive === cameraName);

    // If any preview is running, tell it to stop.
    if (previouslyActive) {
        await eel.stop_live_preview()(); 
        // We let the end_live_preview callback handle resetting the UI for the old camera.
    }

    // If the user clicked a new camera (not the one that was already active), start it.
    if (!clickedActive) {
        activePreviewCamera = cameraName;
        setPreviewButtonActive(cameraName);
        await eel.start_live_preview(cameraName)();
    } else {
        // If the user clicked the camera that was already active, we just want to stop it.
        activePreviewCamera = null;
    }
}

function resetPreviewButton(cameraName) {
    const liveViewBtn = document.getElementById(`live-view-btn-${cameraName}`);
    if (liveViewBtn) {
        liveViewBtn.classList.remove('btn-warning');
        liveViewBtn.classList.add('btn-outline-light');
        liveViewBtn.innerHTML = '<i class="bi bi-eye-fill"></i>';
        bootstrap.Tooltip.getInstance(liveViewBtn)?.setContent({ '.tooltip-inner': 'Live Preview' });
    }
    const liveViewBtnRec = document.getElementById(`live-view-btn-recording-${cameraName}`);
     if (liveViewBtnRec) {
        liveViewBtnRec.classList.remove('btn-warning');
        liveViewBtnRec.classList.add('btn-outline-light');
        liveViewBtnRec.innerHTML = '<i class="bi bi-eye-fill"></i>';
        bootstrap.Tooltip.getInstance(liveViewBtnRec)?.setContent({ '.tooltip-inner': 'Live Preview' });
    }
}

function routeToRecordPage() { routingInProgress = true; window.location.href = './record.html'; }
function routeToLabelTrainPage() { routingInProgress = true; window.location.href = './label-train.html'; }
function routeToVisualizePage() { routingInProgress = true; window.location.href = './visualize.html'; }

function showErrorOnRecordPage(message) {
    const el = document.getElementById("error-message");
    if (el && generalErrorBsModal) { el.innerText = message; generalErrorBsModal.show(); }
    else { alert(message); }
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
        await loadCameras(true);
    } else { showErrorOnRecordPage(`Failed to create camera '${name}'. It may already exist.`); }
}

async function startCamera(cameraName) {
    const sessionName = document.getElementById('session-name-input').value;
    if (!sessionName.trim()) { showErrorOnRecordPage('Please enter a Session Name before starting a recording.'); return; }
    await eel.start_camera_stream(cameraName, sessionName)();
    await updateRecordingStatus();
}

async function stopCamera(cameraName) {
    await eel.stop_camera_stream(cameraName)();
    await updateRecordingStatus();
}

async function startAllCameras() {
    if (allCameraData.length === 0) { alert("No cameras are configured to start."); return; }
    for (const cam of allCameraData) {
        startCamera(cam.name);
    }
    setTimeout(updateRecordingStatus, 1500);
}

async function stopAllCameras() {
    await eel.stop_all_camera_streams()(); 
    await updateRecordingStatus();
}

async function saveCameraSettings() {
    const newName = document.getElementById('cs-name').value;
    if (!newName.trim()) { showErrorOnRecordPage("Camera name cannot be empty."); return; }
    
    const settings = {
        "name": newName, "rtsp_url": document.getElementById('cs-url').value,
        "framerate": parseInt(document.getElementById('cs-framerate').value) || 10,
        "resolution": parseInt(document.getElementById('cs-resolution').value) || 256,
        "segment_seconds": parseInt(document.getElementById('cs-segment-duration').value) || 600,
        'crop_left_x': parseFloat(document.getElementById('cs-cropx').value) || 0,
        'crop_top_y': parseFloat(document.getElementById('cs-cropy').value) || 0,
        'crop_width': parseFloat(document.getElementById('cs-crop-width').value) || 1,
        'crop_height': parseFloat(document.getElementById('cs-crop-height').value) || 1,
    };

    cameraSettingsBsModal?.hide();
    document.getElementById('cover-spin').style.visibility = 'visible';

    if (newName !== originalCameraNameForSettings) {
        const renameSuccess = await eel.rename_camera(originalCameraNameForSettings, newName)();
        if (!renameSuccess) { 
            showErrorOnRecordPage(`Failed to rename. '${newName}' may exist.`); 
            document.getElementById('cover-spin').style.visibility = 'hidden';
            return; 
        }
    }

    await eel.save_camera_settings(newName, settings)();

    // The only action required is to clear the cache for this specific camera.
    sessionStorage.removeItem(`thumbnail_${originalCameraNameForSettings}`);
    sessionStorage.removeItem(`thumbnail_${newName}`);
    
    // Now, call the main load function. It will be fast because all other
    // cameras are still cached. It will only fetch the one we just edited.
    await loadCameras();
    
    document.getElementById('cover-spin').style.visibility = 'hidden';
}

async function loadCameras(forceRefresh = false) {
    if (forceRefresh) {
        clearThumbnailCache();
    }

    const container = document.getElementById('camera-container');
    if (!container) return;
    
    container.innerHTML = "";
    
    try {
        camerasFetchedCount = 0;
        
        allCameraData = await eel.get_camera_list()(); 
        
        if (!allCameraData || allCameraData.length === 0) {
            container.innerHTML = "<div class='col'><p class='text-light text-center mt-3'>No cameras configured.</p></div>";
            camerasToFetchCount = 0;
            update_thumbnail_progress(100, "No cameras configured.");
            return;
        }

        await loadCameraHTMLCards();
        
        const camerasToFetch = [];
        allCameraData.forEach(cam => {
            const cachedSrc = sessionStorage.getItem(`thumbnail_${cam.name}`);
            if (cachedSrc) {
                const base64Val = cachedSrc.split(',')[1];
                updateImageSrc(cam.name, base64Val);
            } else {
                camerasToFetch.push(cam.name);
            }
        });
        
        camerasToFetchCount = camerasToFetch.length;
        
        if (camerasToFetchCount > 0) {
            update_thumbnail_progress(0, `Connecting... (0/${camerasToFetchCount})`);
            eel.fetch_specific_thumbnails(camerasToFetch)();
        } else {
            update_thumbnail_progress(100, "All thumbnails loaded from cache.");
        }

        await updateRecordingStatus();
        
        var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) { return new bootstrap.Tooltip(tooltipTriggerEl) });

    } catch (error) {
        console.error("Failed to load cameras:", error);
        showErrorOnRecordPage("An error occurred while fetching camera data.");
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
                    <div class="card-header py-2 d-flex justify-content-between align-items-center">
                        <h5 class="card-title mb-0">${displayName}</h5>
                        <span id="status-dot-${cam.name}" class="badge rounded-pill bg-secondary" title="Status Unknown"> </span>
                    </div>

                    <div class="d-flex justify-content-center align-items-center" style="height: 225px; background-color: #343a40; margin: 10px auto; width: 300px;">
                        <div class="spinner-border text-light" role="status" id="spinner-${cam.name}">
                            <span class="visually-hidden">Loading...</span>
                        </div>
                        <canvas id="camera-${cam.name}" width="300" height="225" style="display: none;"></canvas>
                    </div>

                    <div class="card-footer p-2">
                        <!-- STATE 1: IDLE -->
                        <div id="before-recording-${cam.name}">
                            <div class="d-flex justify-content-center align-items-center gap-3 flex-wrap">
                                <div class="btn-group btn-group-sm" role="group">
                                    <button class="btn btn-outline-light" onclick="loadCameraSettings('${cam.name}')" data-bs-toggle="tooltip" data-bs-placement="bottom" title="Settings/Crop">
                                        <i class="bi bi-crop"></i>
                                    </button>
                                    <button class="btn btn-outline-danger" onclick="deleteCamera('${cam.name}')" data-bs-toggle="tooltip" data-bs-placement="bottom" title="Delete Camera">
                                        <i class="bi bi-trash-fill"></i>
                                    </button>
                                    <button id="live-view-btn-${cam.name}" class="btn btn-outline-light" onclick="toggleLivePreview('${cam.name}')" data-bs-toggle="tooltip" data-bs-placement="bottom" title="Live Preview">
                                        <i class="bi bi-eye-fill"></i>
                                    </button>
                                </div>
                                <div class="btn-group btn-group-sm" role="group">
                                    <button class="btn btn-success" onclick="startCamera('${cam.name}')" data-bs-toggle="tooltip" data-bs-placement="bottom" title="Start Recording" style="width: 90px;">
                                        <i class="bi bi-camera-video-fill"></i> Start
                                    </button>
                                </div>
                            </div>
                        </div>

                        <!-- STATE 2: RECORDING -->
                        <div id="during-recording-${cam.name}" style="display: none;">
                            <div class="d-flex justify-content-center align-items-center gap-2">
                                <span id="timer-${cam.name}" class="badge bg-primary me-3" style="font-size: 0.8rem;">00:00:00</span>
                                <div class="btn-group btn-group-sm" role="group">
                                    <button class="btn btn-outline-light" data-bs-toggle="tooltip" data-bs-placement="bottom" title="Settings disabled during recording" disabled>
                                        <i class="bi bi-crop"></i>
                                    </button>
                                    <button class="btn btn-outline-light" data-bs-toggle="tooltip" data-bs-placement="bottom" title="Live Preview disabled during recording" disabled>
                                        <i class="bi bi-eye-fill"></i>
                                    </button>
                                    <button class="btn btn-outline-info" onclick="revealRecordingFolder(document.getElementById('session-name-input').value, '${cam.name}')" data-bs-toggle="tooltip" data-bs-placement="bottom" title="Show Recording Folder">
                                        <i class="bi bi-folder2-open"></i>
                                    </button>
                                </div>
                                <div class="btn-group btn-group-sm ms-3" role="group">
                                    <button class="btn btn-danger" onclick="stopCamera('${cam.name}')" data-bs-toggle="tooltip" data-bs-placement="bottom" title="Stop Recording" style="width: 90px;">
                                        <i class="bi bi-square-fill"></i> Stop
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>`;
    }

    container.innerHTML = htmlContent;
}

async function syncAllCameraSettings() {
    const framerate = parseInt(document.getElementById('cs-framerate').value) || 10;
    const resolution = parseInt(document.getElementById('cs-resolution').value) || 256;
    const segment_seconds = parseInt(document.getElementById('cs-segment-duration').value) || 600;

    const message = `Are you sure you want to apply these settings to ALL cameras?\n\n- Framerate: ${framerate}\n- Resolution: ${resolution}\n- Segment Duration: ${segment_seconds}s\n\nThis will overwrite the current settings on every camera.`;

    if (confirm(message)) {
        cameraSettingsBsModal?.hide();
        document.getElementById('cover-spin').style.visibility = 'visible';
        
        try {
            const success = await eel.save_all_camera_settings({
                framerate,
                resolution,
                segment_seconds
            })();
            
            if (!success) {
                showErrorOnRecordPage("Failed to sync settings across all cameras.");
            } else {
                // Important: Clear the entire thumbnail cache as all cameras have been updated.
                clearThumbnailCache();
            }
        } catch (e) {
            console.error("Error syncing all camera settings:", e);
            showErrorOnRecordPage("An error occurred while syncing settings.");
        } finally {
            // Reload all cameras to reflect the changes.
            await loadCameras();
            document.getElementById('cover-spin').style.visibility = 'hidden';
        }
    }
}

async function deleteCamera(cameraName) {
    // Confirmation dialog is crucial for a destructive action
    if (confirm(`Are you sure you want to permanently delete the camera '${cameraName}'?\nThis action cannot be undone.`)) {
        try {
            const success = await eel.delete_camera(cameraName)();
            if (success) {
                // Refresh the camera list to show the change
                await loadCameras();
            } else {
                showErrorOnRecordPage(`Failed to delete camera '${cameraName}'. It might be currently recording.`);
            }
        } catch (e) {
            console.error("Error calling delete_camera:", e);
            showErrorOnRecordPage("An error occurred while trying to delete the camera.");
        }
    }
}

async function updateRecordingStatus() {
    try {
        // The backend returns a dictionary: {'cam01': 162534.123, ...} or false
        const activeStreams = await eel.get_active_streams()() || {};
        activeStreamsInfo = activeStreams; // Update our global state

        setRecordAllIcon(Object.keys(activeStreams).length > 0);

        // Start the timer if it's not running and there are active streams
        if (Object.keys(activeStreams).length > 0 && !recordingTimerInterval) {
            recordingTimerInterval = setInterval(updateRecordingTimers, 1000);
        } 
        // Stop the timer if it is running and there are no active streams
        else if (Object.keys(activeStreams).length === 0 && recordingTimerInterval) {
            clearInterval(recordingTimerInterval);
            recordingTimerInterval = null;
        }
        
        for (const cam of allCameraData) {
            const isActive = cam.name in activeStreams;
            const statusDot = document.getElementById(`status-dot-${cam.name}`);
            const canvas = document.getElementById(`camera-${cam.name}`);
            
            if (statusDot && canvas) {
                let statusClass = '';
                let statusTitle = '';

                if (isActive) {
                    statusClass = 'bg-primary blinking';
                    statusTitle = 'Camera is Recording';
                } else {
                    const currentSrc = canvas.getAttribute("cbas_image_source");
                    if (currentSrc && currentSrc.includes("assets/noConnection.png")) {
                        statusClass = 'bg-secondary';
                        statusTitle = 'Camera Offline';
                    } else {
                        statusClass = 'bg-success';
                        statusTitle = 'Camera Online and Idle';
                    }
                }
                
                statusDot.className = 'badge rounded-pill ' + statusClass;
                const tooltip = bootstrap.Tooltip.getInstance(statusDot);
                if (tooltip) {
                    tooltip.setContent({ '.tooltip-inner': statusTitle });
                } else {
                    statusDot.setAttribute('title', statusTitle);
                }
            }
            
            const beforeRec = document.getElementById(`before-recording-${cam.name}`);
            const duringRec = document.getElementById(`during-recording-${cam.name}`);
            if (beforeRec && duringRec) {
                beforeRec.style.display = isActive ? 'none' : 'flex';
                duringRec.style.display = isActive ? 'flex' : 'none';
            }
        }
    } catch(e) { console.error("Could not update recording status:", e); }
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

async function loadCameraSettings(cameraName) {
    const settings = allCameraData.find(c => c.name === cameraName);
    if (settings) {
        originalCameraNameForSettings = cameraName;
        document.getElementById('cs-name').value = settings.name;
        document.getElementById('cs-url').value = settings.rtsp_url;
        document.getElementById('cs-framerate').value = settings.framerate;
        document.getElementById('cs-resolution').value = settings.resolution;
        document.getElementById('cs-segment-duration').value = settings.segment_seconds || 600;
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

function onMouseDown(e) {
    const { offsetX, offsetY } = e;
    resizeHandle = getHandleAt(offsetX, offsetY);
    if (resizeHandle) { isResizing = true; } 
    else if (offsetX > cropRect.x && offsetX < cropRect.x + cropRect.w && offsetY > cropRect.y && offsetY < cropRect.y + cropRect.h) { isDragging = true; }
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
            if (handle.includes('n') || handle.includes('s')) cropCanvas.style.cursor = 'ns-resize';
            else if (handle.includes('e') || handle.includes('w')) cropCanvas.style.cursor = 'ew-resize';
        } else if (offsetX > cropRect.x && offsetX < cropRect.x + cropRect.w && offsetY > cropRect.y && offsetY < cropRect.y + cropRect.h) {
            cropCanvas.style.cursor = 'move';
        } else { cropCanvas.style.cursor = 'crosshair'; }
    }
    if (isDragging || isResizing) {
        drawCropOverlay();
        updateInputsFromCropRect();
    }
}

function onMouseUp(e) { isDragging = false; isResizing = false; resizeHandle = null; }

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
    if (!cropCtx) return;
    const { x, y, w, h } = cropRect;
    const handles = {
        tl: { x: x, y: y }, tr: { x: x + w, y: y },
        bl: { x: x, y: y + h }, br: { x: x + w, y: y + h },
    };
    cropCtx.fillStyle = "rgba(255, 0, 0, 0.9)";
    for (const handle in handles) {
        cropCtx.fillRect(handles[handle].x - handleSize / 2, handles[handle].y - handleSize / 2, handleSize, handleSize);
    }
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


// =================================================================
// 4. PAGE INITIALIZATION
// =================================================================

document.addEventListener('DOMContentLoaded', async () => {
    await waitForEelConnection();
    
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
    
    const refreshButton = document.querySelector('.fab-container-left .fab');
    if (refreshButton) {
        refreshButton.addEventListener('click', () => {
            loadCameras(true);
        });
    }
    
    cropCanvas = document.getElementById("crop-overlay");
    if(cropCanvas) {
        cropCanvas.addEventListener('mousedown', onMouseDown);
        cropCanvas.addEventListener('mousemove', onMouseMove);
        cropCanvas.addEventListener('mouseup', onMouseUp);
        cropCanvas.addEventListener('mouseleave', () => { isDragging = false; isResizing = false; });
    }

    const cropInputs = ['cs-cropx', 'cs-cropy', 'cs-crop-width', 'cs-crop-height', 'cs-segment-duration'];
    cropInputs.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.addEventListener('input', updateCropRectFromInputs);
    });

    const sessionInput = document.getElementById('session-name-input');
    const editBtn = document.getElementById('edit-session-name-btn');
    if (sessionInput && editBtn) {
        const savedSession = localStorage.getItem('lastSessionName');
        if (savedSession) {
            sessionInput.value = savedSession;
            sessionInput.readOnly = true;
            editBtn.style.display = 'block';
        }
        sessionInput.addEventListener('blur', () => {
            const sessionName = sessionInput.value.trim();
            if (sessionName !== '') {
                sessionInput.readOnly = true;
                editBtn.style.display = 'block';
                localStorage.setItem('lastSessionName', sessionName);
            }
        });
        sessionInput.addEventListener('keydown', (event) => {
            if (event.key === 'Enter') {
                sessionInput.blur();
            }
        });
        editBtn.addEventListener('click', () => {
            sessionInput.readOnly = false;
            editBtn.style.display = 'none';
            sessionInput.focus();
        });
    }

});