# 🚀 Complete Project Guide: Silent Remote Execution on NVIDIA GPU

This guide covers everything required to export the project, securely connect to a remote Windows laptop with a high-end NVIDIA GPU (like the RTX 5080) via a persistent background service, and run the PyTorch training pipeline.

## Step 1: Securely Pack the Data
The current `rpr` project contains hundreds of megabytes of raw ECG data that we no longer need since it has already been processed. To make sharing trivial, follow these rules:

1. **Duplicate the Project:** Copy the entire `E:\rpr` folder and paste it onto your Desktop so you have a secondary copy.
2. **Delete Heavy Raw Data:** Inside the copied folder (`Desktop/rpr/data`), permanently delete the raw data folders (e.g., `mitbih_raw/`, `ptbxl_raw/`) to save extreme amounts of space. **Keep ONLY the `mitbih_processed.csv` and `ptbxl_processed.csv` files.**
3. **Zip the Shrink-wrapped Folder:** Right-click the newly cleaned `rpr` folder on your desktop and select **Compress to ZIP file**.
4. **Share:** Upload the `rpr.zip` file to Google Drive and send the link to your friend.

## Step 2: Establish the Permanent & Invisible Remote Tunnel
Because you want real-time view into the terminal without lagging screen-share video, AND you want it running silently in the background 24/7 (even if he restarts his laptop), you must install it as a **Windows Background Service**. 

**Have your friend run these commands in an Administrator PowerShell inside the extracted folder:**

1. **Download the VS Code CLI:**
```powershell
winget install "Visual Studio Code CLI"
```

2. **Login and Create the Tunnel:**
```powershell
code tunnel --name rtx5080-gpu
```
**CRITICAL AUTHORIZATION STEP:** The terminal will spit out an 8-digit device code. **Have him text you this code.** You will go to `github.com/login/device` on your phone/laptop, log into YOUR GitHub account, and type in the code. This permanently links his hidden tunnel directly to your identity securely.

3. **Install it as an Invisible Windows Service:**
```powershell
code tunnel service install
```

**What happens next?**
You now have permanent access. You simply go to `vscode.dev` on your own laptop, log into your GitHub, and you'll see a button to instantly connect to `rtx5080-gpu`. You can code, drag-and-drop files, or run scripts completely invisibly. The friend will never see a window pop up.

## Step 3: Install NVIDIA PyTorch Dependencies
To unlock the massive power of the RTX 5080, PyTorch needs to communicate with NVIDIA CUDA.

When you access his terminal via VS Code on your laptop, paste these commands:

```powershell
# 1. Install standard requirements
pip install -r requirements.txt

# 2. Install PyTorch initialized for NVIDIA CUDA 11.8 (or 12.1+, but 11.8 is highly stable)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Step 4: Unleash the Pipeline
Because the python code (`src/train.py`) is already written to automatically prioritize NVIDIA GPUs if it detects them, you do **not** need to change any python code. 

Once the dependencies are installed, simply run the master automation script from his terminal remotely:

```powershell
.\experiments\run_ablations.ps1
```

**Verification:** You should see `Using device: cuda:0` instead of `cpu` in the logs. A training step that took 3 hours on your CPU will now complete in roughly 2-4 minutes!
