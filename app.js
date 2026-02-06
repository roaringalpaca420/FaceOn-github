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
  watchdog: "models/WatchDog_52blendshapes.glb",
  raccoon: "https://assets.codepen.io/9177687/raccoon_head.glb",
};

// Logging
const logs = [];
const MAX_LOGS = 100;

function log(msg, type = "info") {
  const entry = { t: new Date().toISOString().slice(11, 23), msg, type };
  logs.push(entry);
  if (logs.length > MAX_LOGS) logs.shift();
  console.log(`[${entry.t}] ${msg}`);
}

function getLogText() {
  return logs.map((e) => `[${e.t}] ${e.msg}`).join("\n");
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

    const video = document.getElementById("video");
    const tex = new THREE.VideoTexture(video);
    tex.colorSpace = THREE.SRGBColorSpace;
    this.scene.add(createCameraPlaneMesh(this.camera, 500, new THREE.MeshBasicMaterial({ map: tex })));

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
    log("Loading model: " + url);
    this.loader.load(
      url,
      (gltf) => {
        if (this.gltf) {
          this.gltf.scene.remove();
          this.morphTargetMeshes = [];
        }
        this.gltf = gltf;
        this.scene.add(gltf.scene);
        this.init(gltf);
        log("Model loaded, morph meshes: " + this.morphTargetMeshes.length);
      },
      undefined,
      (err) => {
        log("Model load error: " + (err.message || err), "error");
      }
    );
  }

  init(gltf) {
    this.root = null;
    gltf.scene.traverse((obj) => {
      if (obj.isBone && !this.root) this.root = obj;
      if (!obj.isMesh) return;
      obj.frustumCulled = false;
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
    matrix.scale(new THREE.Vector3(scale, scale, scale));
    this.gltf.scene.matrixAutoUpdate = false;
    this.gltf.scene.matrix.copy(matrix);
  }
}

let faceLandmarker = null;
let video = null;
let avatar = null;
let trackingActive = false;
let faceDetectedCount = 0;

const scene = new BasicScene();
const statusEl = document.getElementById("status");
const startBtn = document.getElementById("startBtn");

function setStatus(msg) {
  statusEl.textContent = msg;
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
    log("Detect error: " + e.message, "error");
  }
}

function onVideoFrame(time) {
  detectFaceLandmarks(time);
  if (video) video.requestVideoFrameCallback(onVideoFrame);
}

async function streamWebcam() {
  video = document.getElementById("video");
  setStatus("Requesting camera...");
  startBtn.disabled = true;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: "user", width: 1280, height: 720 },
    });
    video.srcObject = stream;
    video.onloadedmetadata = () => video.play();
    log("Camera started");
    setStatus("Loading face tracking...");
    trackingActive = true;
    video.requestVideoFrameCallback(onVideoFrame);
    if (faceLandmarker) {
      setStatus("Tracking. Your face drives the avatar.");
    } else {
      setStatus("Loading MediaPipe...");
    }
  } catch (e) {
    log("Camera error: " + (e.message || e.name || "Permission denied"), "error");
    setStatus("Camera error: " + (e.message || "Allow access when prompted."));
    startBtn.disabled = false;
  }
}

function loadAvatar(url) {
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
  window.faceOnGetLogs = getLogText;
}

async function initMediaPipe() {
  try {
    log("Loading MediaPipe...");
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );
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
    log("MediaPipe ready");
    if (trackingActive) setStatus("Tracking. Your face drives the avatar.");
  } catch (e) {
    log("MediaPipe error: " + (e.message || e), "error");
    setStatus("Face tracking failed. Check console.");
  }
}

async function run() {
  log("FaceOn starting");
  setStatus("Loading...");
  window.faceOnStartCamera = streamWebcam;
  initModelPicker();
  initSettings();
  loadAvatar(MODELS.watchdog);
  await initMediaPipe();
  window.faceOnReady = true;
  if (!trackingActive) {
    setStatus("Click Start Camera to begin.");
  }
}

run().catch((e) => {
  log("Startup error: " + (e.message || e), "error");
  setStatus("Error: " + (e.message || "Check console."));
});
