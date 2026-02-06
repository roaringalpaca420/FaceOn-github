# FaceOn

Web app that drives a 3D avatar with your face using MediaPipe face tracking and 52 ARKit blend shapes.

## Live Demo

Deploy to GitHub Pages and open the URL. Requires a browser with webcam support.

## Setup

1. **Add your model**: Place `WatchDog_52blendshapes.glb` (or your GLB with 52 ARKit blend shapes) in the `models/` folder.

2. **Run locally** (optional):
   - Open `index.html` in a browser, or
   - Use a local server: `npx serve .` or `python -m http.server 8000`

3. **Deploy to GitHub Pages**:
   - Create a new repo on GitHub
   - Copy all files from `FaceOn-github` into the repo
   - Go to Settings → Pages → Source: Deploy from branch
   - Branch: `main`, folder: `/ (root)`
   - Save. Your site will be at `https://YOUR_USERNAME.github.io/REPO_NAME/`

## Usage

- Allow webcam access when prompted
- Your face drives the 3D avatar in real time
- Use the dropdown to switch between WatchDog and Raccoon (demo)
- Or upload a custom GLB file (must have 52 ARKit blend shapes)

## Models

GLB models need 52 ARKit-compatible blend shape names (eyeBlinkLeft, mouthSmileLeft, etc.). Use the `add_arkit_blendshapes.py` script in the parent FaceOn folder to add them to Meshy models in Blender.
