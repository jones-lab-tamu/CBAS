/**
 * @file global.js
 * @description Contains JavaScript functions and Eel exposures that are shared across all pages of the CBAS application.
 */

// Make sure Eel is ready before trying to expose functions
document.addEventListener('DOMContentLoaded', () => {
    if (typeof eel !== 'undefined') {
        eel.expose(update_log_panel);
        eel.expose(update_global_encoding_progress);
    } else {
        console.error("Eel is not defined. Cannot expose functions.");
    }
});


/**
 * Updates the global two-tier encoding progress bar.
 * @param {object} status - An object with progress details.
 * e.g., { overall_processed: 1, overall_total: 10, current_percent: 50, current_file: "video.mp4" }
 */
function update_global_encoding_progress(status) {
    const overlay = document.getElementById('encoding-progress-overlay');
    if (!overlay) return;

    const wasHidden = overlay.style.display === 'none';

    // Get all the elements
    const overallLabel = document.getElementById('encoding-progress-label-overall');
    const overallBar = document.getElementById('encoding-progress-bar-overall');
    const currentLabel = document.getElementById('encoding-progress-label-current');
    const currentBar = document.getElementById('encoding-progress-bar-current');

    // Hide the overlay if the job is done (total is 0)
    if (!status || status.overall_total === 0) {
        overlay.style.display = 'none';
        return;
    }

    // --- Update UI ---
    overlay.style.display = 'block';

    // Update Overall Bar
    const overallPercent = status.overall_total > 0 ? (status.overall_processed / status.overall_total) * 100 : 0;
    const overallDisplayPercent = Math.floor(overallPercent); // Use floor to not show 100% until truly done
    overallLabel.textContent = `Overall Queue: (${status.overall_processed} / ${status.overall_total})`;
    overallBar.style.width = `${overallDisplayPercent}%`;
    overallBar.textContent = `${overallDisplayPercent}%`;
    overallBar.setAttribute('aria-valuenow', overallDisplayPercent);

    if (!status || status.overall_total === 0) {
        overlay.style.display = 'none';
        if (!wasHidden) updateFooterLayout(); // Update layout if it just became hidden
        return;
    }
    
    overlay.style.display = 'block';
    if (wasHidden) updateFooterLayout(); // Update layout if it just became visible

    // Update Current File Bar
    const currentDisplayPercent = Math.round(status.current_percent);
    currentLabel.textContent = `Encoding: ${status.current_file}`;
    currentBar.style.width = `${currentDisplayPercent}%`;
    currentBar.setAttribute('aria-valuenow', currentDisplayPercent);
}

/**
 * Appends a new message to the log panel in the footer.
 * Also saves the message to sessionStorage to persist across page loads.
 * @param {string} message - The log message from the backend.
 */
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

/**
 * Renders a single log message to the DOM with appropriate styling.
 * @param {string} message - The log message text.
 * @param {HTMLElement} container - The container to append the message to.
 */
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

/**
 * Adjusts the main content padding and FAB positions to prevent overlap with the footer.
 * This should be called any time the footer's height might change.
 */
function updateFooterLayout() {
    const footerContainer = document.getElementById('page-footer-container');
    const fabLeft = document.querySelector('.fab-container-left');
    const fabRight = document.querySelector('.fab-container-right');

    if (!footerContainer) return;

    const footerHeight = footerContainer.offsetHeight;

    // 1. Add padding to the bottom of the body so content can scroll up past the footer.
    document.body.style.paddingBottom = `${footerHeight + 20}px`;

    // 2. Adjust the position of the Floating Action Buttons.
    const fabBottomPosition = `${footerHeight + 20}px`;
    if (fabLeft) fabLeft.style.bottom = fabBottomPosition;
    if (fabRight) fabRight.style.bottom = fabBottomPosition;
}


/**
 * Initializes the log panel on page load by rendering messages from sessionStorage
 * and attaching all global event listeners.
 */
function initializeGlobalUI() {
    // --- Initialize Log Panel ---
    const logContainer = document.getElementById('log-panel-content');
    if (logContainer) {
        const logHistory = JSON.parse(sessionStorage.getItem('logHistory') || '[]');
        logHistory.forEach(msg => renderLogMessage(msg, logContainer));
        if(logContainer.innerHTML) {
            logContainer.scrollTop = logContainer.scrollHeight;
        }

        document.getElementById('clear-log-btn')?.addEventListener('click', () => {
            logContainer.innerHTML = '';
            sessionStorage.setItem('logHistory', '[]');
            update_log_panel('Log cleared.');
        });
    }

    // --- Initialize Log Panel Animation & FABs ---
    const logCollapseElement = document.getElementById('log-panel-collapse');
    if (logCollapseElement) {
        // Just call our central layout function on show and hide events.
        logCollapseElement.addEventListener('shown.bs.collapse', updateFooterLayout);
        logCollapseElement.addEventListener('hidden.bs.collapse', updateFooterLayout);
    }
}

// Initialize all global UI components when the DOM is ready
document.addEventListener('DOMContentLoaded', async () => { // Make it async
    initializeGlobalUI();
    updateFooterLayout(); // Set initial state

    // After initializing, synchronize the state with the backend
    if (typeof eel !== 'undefined') {
        try {
            // Ask the backend for the current encoding status
            const status = await eel.get_encoding_queue_status()();
            if (status && status.total > 0) {
                // If a job is running, update the progress bar immediately
                update_global_encoding_progress(status.processed, status.total);
            }
        } catch (error) {
            console.error("Could not synchronize encoding status on page load:", error);
        }
    }
});