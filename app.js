/**
 * FaceOn — Face tracking avatar with MediaPipe + Three.js
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import {
  FilesetResolver,
  FaceLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const MODELS = {
  watchdog: "WatchDog_52blendshapes.glb",
  raccoon: "https://assets.codepen.io/9177687/raccoon_head.glb",
};

function log(msg, detail) {
  if (window.faceOnLog) window.faceOnLog(msg, detail);
  else console.log(detail ? msg + " | " + detail : msg);
}

function getViewportSizeAtDepth(camera, depth) {
  const h = 2 * depth * Math.tan(THREE.MathUtils.degToRad(0.5 * camera.fov));
  return new THREE.Vector2(h * camera.aspect, h);
}

function createCameraPlaneMesh(camera, depth, material) {
  const size = getViewportSizeAtDepth(camera, depth);
  const geom = new THREE.PlaneGeometry(size.width, size.height);
  geom.translate(0, 0, -depth);
  return new THREE.Mesh(geom, material);
}

class BasicScene {
  constructor() {
    this.height = window.innerHeight;
    this.width = (this.height * 1280) / 720;
    this.lastTime = 0;
    this.callbacks = [];
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(60, this.width / this.height, 0.01, 5000);
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setSize(this.width, this.height);
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    document.getElementById("canvasContainer").appendChild(this.renderer.domElement);

    this.scene.add(new THREE.AmbientLight(0xffffff, 0.5));
    const d = new THREE.DirectionalLight(0xffffff, 0.5);
    d.position.set(0, 1, 0);
    this.scene.add(d);

    this.camera.position.z = 0;
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.copy(this.camera.position).z -= 5;
    this.controls.update();

    this.scene.background = new THREE.Color(0x000000);

    this.render();
    window.addEventListener("resize", () => this.resize());
  }

  resize() {
    this.width = window.innerWidth;
    this.height = window.innerHeight;
    this.camera.aspect = this.width / this.height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(this.width, this.height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  }

  render(time = this.lastTime) {
    const delta = (time - this.lastTime) / 1000;
    this.lastTime = time;
    for (const cb of this.callbacks) cb(delta);
    this.renderer.render(this.scene, this.camera);
    requestAnimationFrame((t) => this.render(t));
  }
}

class Avatar {
  constructor(url, scene) {
    this.url = url;
    this.scene = scene;
    this.loader = new GLTFLoader();
    this.gltf = null;
    this.root = null;
    this.morphTargetMeshes = [];
    this.loadModel(url);
  }

  loadModel(url) {
    this.url = url;
    this._lastLoggedPct = -1;
    log("loadModel", "URL: " + url);
    this.loader.load(
      url,
      (gltf) => {
        if (this.gltf) {
          this.gltf.scene.remove();
          this.morphTargetMeshes = [];
        }
        this.gltf = gltf;
        gltf.scene.position.set(0, 0, -200);
        gltf.scene.scale.setScalar(40);
        this.scene.add(gltf.scene);
        this.init(gltf);
        log("loadModel OK", "morph meshes: " + this.morphTargetMeshes.length + (this.morphTargetMeshes.length === 0 ? " (no blend shapes - use Raccoon or add in Blender)" : ""));
      },
      (xhr) => {
        if (xhr.lengthComputable && xhr.total > 0) {
          const pct = Math.floor(100 * xhr.loaded / xhr.total);
          for (const m of [25, 50, 75, 100]) {
            if (pct >= m && this._lastLoggedPct < m) {
              this._lastLoggedPct = m;
              log("loadModel progress", m + "%");
              break;
            }
          }
        }
      },
      (err) => {
        log("loadModel ERROR", err && err.message ? err.message : String(err));
      }
    );
  }

  init(gltf) {
    this.root = null;
    gltf.scene.traverse((obj) => {
      if (obj.isBone && !this.root) this.root = obj;
      if (!obj.isMesh) return;
      obj.frustumCulled = false;
      obj.renderOrder = 1;
      if (obj.morphTargetDictionary && obj.morphTargetInfluences)
        this.morphTargetMeshes.push(obj);
    });
  }

  updateBlendshapes(blendshapes) {
    for (const mesh of this.morphTargetMeshes) {
      if (!mesh.morphTargetDictionary || !mesh.morphTargetInfluences) continue;
      for (const [name, value] of blendshapes) {
        if (!(name in mesh.morphTargetDictionary)) continue;
        mesh.morphTargetInfluences[mesh.morphTargetDictionary[name]] = value;
      }
    }
  }

  applyMatrix(matrix, opts = {}) {
    const scale = opts.scale ?? 40;
    if (!this.gltf) return;
    const m = matrix.clone();
    m.scale(new THREE.Vector3(scale, scale, scale));
    this.gltf.scene.matrixAutoUpdate = false;
    this.gltf.scene.matrix.copy(m);
  }

  remove() {
    if (this.gltf && this.gltf.scene && this.gltf.scene.parent) {
      this.gltf.scene.parent.remove(this.gltf.scene);
    }
    this.morphTargetMeshes = [];
  }
}

let faceLandmarker = null;
let video = null;
let avatar = null;
let trackingActive = false;
let faceDetectedCount = 0;
let scene = null;
let statusEl = null;
let startBtn = null;

function setStatus(msg) {
  if (statusEl) statusEl.textContent = msg;
}

function detectFaceLandmarks(time) {
  if (!faceLandmarker || !video || !trackingActive) return;
  try {
    const landmarks = faceLandmarker.detectForVideo(video, time);
    const matrices = landmarks.facialTransformationMatrixes;
    const blendshapes = landmarks.faceBlendshapes;
    if (matrices && matrices.length > 0 && avatar) {
      const m = new THREE.Matrix4().fromArray(matrices[0].data);
      avatar.applyMatrix(m, { scale: 40 });
    }
    if (blendshapes && blendshapes.length > 0 && avatar) {
      faceDetectedCount++;
      const map = new Map();
      for (const c of blendshapes[0].categories) {
        let s = c.score;
        if (["browOuterUpLeft", "browOuterUpRight", "eyeBlinkLeft", "eyeBlinkRight"].includes(c.categoryName))
          s *= 1.2;
        map.set(c.categoryName, s);
      }
      avatar.updateBlendshapes(map);
    }
  } catch (e) {
    log("detectFaceLandmarks ERROR", e && e.message ? e.message : String(e));
  }
}

function onVideoFrame(time) {
  detectFaceLandmarks(time);
  if (video) video.requestVideoFrameCallback(onVideoFrame);
}

async function streamWebcam() {
  const videoEl = document.getElementById("video");
  if (!videoEl) {
    log("streamWebcam", "video element not found");
    setStatus("Error: video element missing");
    return;
  }
  video = videoEl;
  setStatus("Requesting camera...");
  log("streamWebcam", "Calling getUserMedia (facingMode: user)");
  startBtn.disabled = true;
  try {
    const constraints = {
      audio: false,
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
    };
    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    log("streamWebcam", "getUserMedia succeeded, tracks: " + stream.getVideoTracks().length);
    video.srcObject = stream;
    const track = stream.getVideoTracks()[0];
    log("streamWebcam", "Video track: " + (track ? track.label : "none") + ", readyState: " + (track ? track.readyState : "?"));
    await new Promise((resolve, reject) => {
      video.onloadedmetadata = () => {
        log("streamWebcam", "video.onloadedmetadata fired, dimensions: " + video.videoWidth + "x" + video.videoHeight);
        resolve();
      };
      video.onerror = (e) => reject(new Error("Video error: " + (video.error ? video.error.message : "unknown")));
      if (video.readyState >= 1) resolve();
    });
    log("streamWebcam", "Starting video.play()");
    await video.play();
    log("streamWebcam", "Video playing, readyState: " + video.readyState);
    trackingActive = true;
    video.requestVideoFrameCallback(onVideoFrame);
    log("streamWebcam", "requestVideoFrameCallback registered");
    setStatus(faceLandmarker ? "Tracking. Your face drives the avatar." : "Camera on. Loading face tracking...");
  } catch (e) {
    const errMsg = e && e.message ? e.message : (e && e.name ? e.name : String(e));
    log("streamWebcam ERROR", errMsg);
    if (e && e.stack) log("stack", e.stack.slice(0, 400));
    setStatus("Camera error: " + errMsg);
    startBtn.disabled = false;
  }
}

function loadAvatar(url) {
  if (!scene) return;
  if (avatar) {
    avatar.remove();
    avatar = null;
  }
  avatar = new Avatar(url, scene.scene);
}

function initModelPicker() {
  const sel = document.getElementById("modelSelect");
  const fileIn = document.getElementById("fileInput");
  sel.addEventListener("change", () => {
    loadAvatar(sel.value === "raccoon" ? MODELS.raccoon : MODELS.watchdog);
  });
  fileIn.addEventListener("change", (e) => {
    const f = e.target.files?.[0];
    if (f) loadAvatar(URL.createObjectURL(f));
  });
}

function initSettings() {
  // Logs already provided by inline script
}

async function initMediaPipe() {
  try {
    log("initMediaPipe", "Loading FilesetResolver...");
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );
    log("initMediaPipe", "FilesetResolver ready, loading FaceLandmarker model...");
    faceLandmarker = await FaceLandmarker.createFromModelPath(
      vision,
      "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    );
    faceLandmarker.setOptions({
      baseOptions: { delegate: "GPU" },
      runningMode: "VIDEO",
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
    });
    log("initMediaPipe OK", "FaceLandmarker ready");
    if (trackingActive) setStatus("Tracking. Your face drives the avatar.");
  } catch (e) {
    const errMsg = e && e.message ? e.message : String(e);
    log("initMediaPipe ERROR", errMsg);
    if (e && e.stack) log("stack", e.stack.slice(0, 400));
    setStatus("Face tracking failed. Check logs.");
  }
}

export async function startApp() {
  log("startApp", "Entry");
  statusEl = document.getElementById("status");
  startBtn = document.getElementById("startBtn");
  if (!statusEl || !startBtn) {
    log("startApp ERROR", "status or startBtn element not found");
    return;
  }
  setStatus("Loading scene...");
  try {
    log("startApp", "Creating BasicScene");
    scene = new BasicScene();
    log("startApp", "BasicScene created");
  } catch (e) {
    const errMsg = e && e.message ? e.message : String(e);
    log("startApp ERROR", "BasicScene: " + errMsg);
    if (e && e.stack) log("stack", e.stack.slice(0, 400));
    setStatus("Scene failed. Check logs.");
    return;
  }
  initModelPicker();
  initSettings();
  loadAvatar(MODELS.raccoon);
  setStatus("Loading face tracking...");
  try {
    await initMediaPipe();
  } catch (e) {
    log("startApp", "initMediaPipe threw: " + (e && e.message ? e.message : e));
  }
  startBtn.addEventListener("click", streamWebcam);
  setStatus("Starting camera...");
  await streamWebcam();
}
