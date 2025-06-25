const { app, BrowserWindow, dialog, ipcMain } = require('electron');
const path = require('path');
const child_process = require('child_process');
const fs = require('fs');

let pythonProcess = null;
let splashWindow = null;
let appWindow = null;

const gotTheLock = app.requestSingleInstanceLock();
if (!gotTheLock) {
  app.quit();
} else {
  app.on('second-instance', () => {
    // If someone tries to open a second instance, focus our main app window
    if (appWindow) {
      if (appWindow.isMinimized()) appWindow.restore();
      appWindow.focus();
    }
  });

  function createPythonProcess() {
    const appRoot = app.getAppPath();
    const venvPython = path.join(appRoot, 'venv', 'Scripts', 'python.exe');
    const scriptPath = path.join(appRoot, 'backend', 'app.py');
    const pythonArgs = ['-u', scriptPath];
    console.log(`Spawning Python: "${venvPython}" ${pythonArgs.join(' ')}`);

    if (!fs.existsSync(venvPython)) {
      // Handle error - e.g., show a dialog
      dialog.showErrorBox("Python Error", `Virtual environment not found at ${venvPython}. Please run the installation steps.`);
      app.quit();
      return;
    }
    pythonProcess = child_process.spawn(venvPython, pythonArgs);
    pythonProcess.stdout.on('data', (data) => console.log(`[Python]: ${data.toString().trim()}`));
    pythonProcess.stderr.on('data', (data) => console.error(`[Python ERR]: ${data.toString().trim()}`));
    pythonProcess.on('close', (code) => {
        if (!app.isQuitting) {
            dialog.showErrorBox("Backend Error", `The Python backend process has crashed (code: ${code}). Please restart the application.`);
        }
    });
  }

  function createWindow() {
    // 1. Create the splash screen window FIRST.
    splashWindow = new BrowserWindow({
      width: 500,
      height: 300,
      transparent: true,
      frame: false,
      alwaysOnTop: true,
      webPreferences: {
          nodeIntegration: true,
          contextIsolation: false,
      }
    });
    splashWindow.loadFile(path.join(__dirname, 'frontend/loading.html'));

    // 2. Create the main application window, but keep it hidden.
    appWindow = new BrowserWindow({
      width: 1400,
      height: 900,
      show: false, // <-- Keep it hidden initially
      webPreferences: {
        preload: path.join(__dirname, 'preload.js'),
        contextIsolation: true,
        nodeIntegration: false,
      }
    });

    let webAppUrl = null;
    pythonProcess.stdout.on('data', (data) => {
      const output = data.toString();
      const match = output.match(/Eel server starting on (http:\/\/localhost:\d+)/);
      if (match && !webAppUrl) {
        webAppUrl = match[1];
        console.log(`Eel server detected. Loading URL: ${webAppUrl}`);
        
        // 3. Load the main app into the HIDDEN window.
        appWindow.loadURL(webAppUrl);
      }
    });

    // 4. When the hidden window is fully loaded, show it and close the splash screen.
    appWindow.webContents.on('did-finish-load', () => {
	  // Only perform this logic if the splash screen actually exists.
	  if (splashWindow) {
		appWindow.show();
		splashWindow.close();
		splashWindow = null;
	  }
	});
    
    // IPC for file dialogs
    ipcMain.on('open-file-dialog', (event, options) => {
      let startPath;
      if (options?.defaultPath) {
        // If a path is provided, we want to open its PARENT directory.
        // This is more user-friendly. e.g., if project is in C:\Users\Me\Projects,
        // we open C:\Users\Me\ so they can see all their projects.
        startPath = path.dirname(options.defaultPath);
      } else {
        // Fallback to the user's documents folder if no path is known.
        startPath = app.getPath('documents');
      }

      const dialogOptions = { 
        properties: ['openDirectory'],
        defaultPath: startPath // Use our calculated start path
      };

      dialog.showOpenDialog(appWindow, dialogOptions)
        .then(result => {
          if (!result.canceled) event.sender.send('selected-directory', result.filePaths[0]);
        });
    });
	
	ipcMain.on('save-file-to-disk', (event, filePath, data) => {
		try {
			fs.writeFileSync(filePath, data);
		} catch (err) {
			console.error("Failed to save file:", err);
			dialog.showErrorBox("Save Error", "Could not save the file to the selected location.");
		}
	});
	
	// This handler is specifically for choosing the EXPORT folder
	ipcMain.handle('show-folder-dialog', async (event, options) => {
	  const { filePaths } = await dialog.showOpenDialog(appWindow, {
		properties: ['openDirectory'],
		title: 'Select Folder to Save Exported CSVs'
	  });
	  // Return the first selected path, or null if cancelled
	  return filePaths && filePaths.length > 0 ? filePaths[0] : null;
	});

    ipcMain.handle('show-open-video-dialog', async () => {
        const { filePaths } = await dialog.showOpenDialog(appWindow, {
            title: 'Select Video Files to Import',
            properties: ['openFile', 'multiSelections'], // Allow multiple files
            filters: [
                { name: 'Videos', extensions: ['mp4'] }
            ]
        });
        return filePaths; // Return the array of selected paths
    });


    appWindow.on('closed', () => { appWindow = null; });
  }

  // --- Standard Electron lifecycle events ---
  app.on('ready', () => { createPythonProcess(); createWindow(); });
  app.on('window-all-closed', () => { if (process.platform !== 'darwin') app.quit(); });
  app.on('before-quit', () => {
  app.isQuitting = true;

  if (pythonProcess && !pythonProcess.killed) {
    console.log("Main app is quitting. Terminating Python backend and its entire process tree...");

    // On Windows, the /T flag terminates the specified process and any child processes.
    // The /F flag forcefully terminates the process.
    // This is the most reliable way to ensure all ffmpeg.exe instances are killed.
    child_process.exec(`taskkill /pid ${pythonProcess.pid} /T /F`, (error, stdout, stderr) => {
        if (error) {
            console.error(`taskkill error: ${error.message}`);
            // Fallback to the original method if taskkill fails for some reason
            pythonProcess.kill();
            return;
        }
        if (stderr) {
            console.error(`taskkill stderr: ${stderr}`);
            return;
        }
        console.log(`taskkill stdout: ${stdout}`);
        console.log("Python process tree terminated successfully.");
    });
    
    // Clear the reference
    pythonProcess = null;
  }
});
  app.on('activate', () => { if (appWindow === null && !splashWindow) createWindow(); });
}