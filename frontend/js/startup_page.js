/**
 * @file Manages the logic for the project startup page.
 * @description This script handles creating new projects and opening existing ones.
 * It uses Electron's IPC for native file dialogs and Eel to communicate with the Python backend.
 */

// =================================================================
// GLOBAL STATE & VARIABLES
// =================================================================

/** Flag to prevent `kill_streams` from being called during intentional page navigation. */
let routingInProgress = false;

/**
 * Flag to distinguish file picker mode for the IPC dialog.
 * 0 = selecting a parent directory to create a new project.
 * 1 = selecting an existing project directory to open.
 * @type {number}
 */
let filePickerMode = 0;


// =================================================================
// UI INITIALIZATION & UTILITY FUNCTIONS
// =================================================================

/** Displays a random "fun fact" blurb on the startup card. */
function initializeFunBlurb() {
    const blurbs = [
        "eel included", "electronic!", "thanks Yann", "just lin alg",
        "add <i>Mus musculus</i>", "around the clock",
    ];
    const links = [
        "https://github.com/python-eel/Eel", "https://www.electronjs.org/",
        "https://ai.meta.com/blog/self-supervised-learning-the-dark-matter-of-intelligence/",
        "https://en.wikipedia.org/wiki/Linear_algebra", "https://en.wikipedia.org/wiki/Laboratory_mouse",
        "https://en.wikipedia.org/wiki/Circadian_rhythm",
    ];

    const funBlurbElement = document.getElementById("fun-blurb-text");
    if (funBlurbElement) {
        const randomIndex = Math.floor(Math.random() * blurbs.length);
        funBlurbElement.innerHTML = blurbs[randomIndex];
        funBlurbElement.href = links[randomIndex];
    }
}

/** Shows the error modal with a custom message. */
function showErrorOnStartup(message) {
    const errorModalElement = document.getElementById("errorModal");
    const errorMessageElement = document.getElementById("error-message");
    if (errorMessageElement && errorModalElement) {
        errorMessageElement.innerText = message;
        new bootstrap.Modal(errorModalElement).show();
    } else {
        alert(message); // Fallback
    }
}

/** Navigates to the main record page after a project is loaded/created. */
function routeToRecordPage() {
    routingInProgress = true;
    window.location.href = "./record.html";
}


// =================================================================
// CORE APPLICATION LOGIC (Project Creation & Loading)
// =================================================================

/**
 * Handles the final submission for creating a new project after a directory and name are provided.
 */
async function handleCreateProjectSubmit() {
    const parentDir = document.getElementById("parent-directory")?.innerText;
    const projectName = document.getElementById("project-name")?.value;

    if (!parentDir || !projectName?.trim()) {
        showErrorOnStartup("Project name cannot be empty.");
        return;
    }

    try {
        const [isCreated, projectDetails] = await eel.create_project(parentDir, projectName)();
        if (isCreated && projectDetails) {
            window.localStorage.setItem("project", JSON.stringify(projectDetails));
            routeToRecordPage();
        } else {
            showErrorOnStartup("Project could not be created. It might already exist or the name is invalid.");
        }
    } catch (error) {
        console.error("Error calling eel.create_project:", error);
        showErrorOnStartup(`An error occurred: ${error.message || "Unknown error"}`);
    }
}

/**
 * Handles the response from the Electron main process after a directory has been selected.
 * @param {object} _event - The IPC event object (unused).
 * @param {string | null} selectedPath - The path chosen by the user, or null if cancelled.
 */
async function onDirectorySelected(_event, selectedPath) {
    if (!selectedPath) {
        console.log("Directory selection was cancelled.");
        return;
    }

    if (filePickerMode === 0) { // CREATE mode
        const createModalElem = document.getElementById("createModal");
        const parentDirElem = document.getElementById("parent-directory");
        if (createModalElem && parentDirElem) {
            parentDirElem.innerText = selectedPath;
            document.getElementById("project-name").value = ""; // Clear previous entry
            new bootstrap.Modal(createModalElem).show();
        }
    } else { // OPEN mode
        try {
            const [isLoaded, projectDetails] = await eel.load_project(selectedPath)();
            if (isLoaded && projectDetails) {
                window.localStorage.setItem("project", JSON.stringify(projectDetails));
                routeToRecordPage();
            } else {
                showErrorOnStartup("Selected directory is not a valid CBAS project or could not be loaded.");
            }
        } catch (error) {
            console.error("Error calling eel.load_project:", error);
            showErrorOnStartup(`An error occurred while loading the project: ${error.message || "Unknown error"}`);
        }
    }
}


// =================================================================
// EVENT LISTENERS
// =================================================================

// --- IPC Listener (from Electron Main Process) ---
// This requires `contextIsolation: false` and `nodeIntegration: true` in the Electron window setup.
// If `window.ipcRenderer` is not available, these event listeners will not attach.
try {
    const ipc = window.ipcRenderer;
    if (ipc) {
        ipc.on("selected-directory", onDirectorySelected);
    } else {
        console.warn("`ipcRenderer` not found. File dialogs will not work outside of a full Electron context.");
    }
} catch (error) {
    console.warn("Could not attach IPC listeners. Running in a standard browser?");
}


// --- DOM Content Loaded (Initial Setup) ---
document.addEventListener('DOMContentLoaded', () => {
    initializeFunBlurb();

    // Attach listeners to the main project selection buttons
    document.getElementById("create")?.addEventListener("click", () => {
        filePickerMode = 0;
        window.ipcRenderer?.send("open-file-dialog");
    });
    document.getElementById("open")?.addEventListener("click", () => {
        filePickerMode = 1;
        window.ipcRenderer?.send("open-file-dialog");
    });

    // Attach listener to the modal's "Create" button
    document.querySelector('#createModal .btn-primary')?.addEventListener('click', handleCreateProjectSubmit);
});


// --- Page Unload Listeners ---
// These attempt to call a Python function to clean up FFMPEG streams if the
// user closes the window unexpectedly.
window.addEventListener("unload", () => {
    if (!routingInProgress) {
        eel.kill_streams()?.().catch(err => console.error("Error on unload:", err));
    }
});
window.onbeforeunload = () => {
    if (!routingInProgress) {
        eel.kill_streams()?.().catch(err => console.error("Error on beforeunload:", err));
    }
};