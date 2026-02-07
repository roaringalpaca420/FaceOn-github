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
  watchdog: "Watchdog Shape Keys Compressed .glb",
  raccoon: "https://assets.codepen.io/9177687/raccoon_head.glb",
};

const BLENDSHAPE_ALIASES = {
  jawOpen: ["mouthOpen", "Mouth_Open", "mouth_open", "jaw_open"],
  mouthClose: ["mouth_close", "Mouth_Close"],
  mouthSmileLeft: ["mouthSmile_L", "Mouth_Smile_L", "mouth_smile_L"],
  mouthSmileRight: ["mouthSmile_R", "Mouth_Smile_R", "mouth_smile_R"],
  mouthFrownLeft: ["mouthFrown_L", "Mouth_Frown_L"],
  mouthFrownRight: ["mouthFrown_R", "Mouth_Frown_R"],
  mouthPucker: ["mouth_pucker", "Mouth_Pucker"],
  mouthFunnel: ["mouth_funnel", "Mouth_Funnel"],
  eyeBlinkLeft: ["eyeBlink_L", "Eye_Blink_L", "eye_blink_L"],
  eyeBlinkRight: ["eyeBlink_R", "Eye_Blink_R", "eye_blink_R"],
  browInnerUp: ["brow_inner_up", "Brow_Inner_Up"],
  browOuterUpLeft: ["browOuterUp_L", "Brow_Outer_Up_L"],
  browOuterUpRight: ["browOuterUp_R", "Brow_Outer_Up_R"],
  browDownLeft: ["browDown_L", "Brow_Down_L"],
  browDownRight: ["browDown_R", "Brow_Down_R"],
  cheekPuff: ["cheek_puff", "Cheek_Puff"],
  cheekSquintLeft: ["cheekSquint_L", "Cheek_Squint_L"],
  cheekSquintRight: ["cheekSquint_R", "Cheek_Squint_R"],
};

function log(msg, detail, opts) {
  if (window.faceOnLog) window.faceOnLog(msg, detail, opts || {});
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

    this.scene.add(new THREE.AmbientLight(0xffffff, 1.5));
    const d = new THREE.DirectionalLight(0xffffff, 1.5);
    d.position.set(0, 1, 0);
    this.scene.add(d);
    const fill = new THREE.DirectionalLight(0xffffff, 0.5);
    fill.position.set(-0.5, 0.5, 1);
    this.scene.add(fill);

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
    const fullUrl = url.startsWith("http") ? url : (window.location.origin + window.location.pathname.replace(/\/[^/]*$/, "/") + url);
    log("loadModel START", "url=" + url + " resolved=" + fullUrl);
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
        const morphCount = this.morphTargetMeshes.length;
        log("loadModel OK", "morph meshes=" + morphCount + (morphCount === 0 ? " (no blend shapes)" : ""));
      },
      (xhr) => {
        if (xhr.lengthComputable && xhr.total > 0) {
          const pct = Math.floor(100 * xhr.loaded / xhr.total);
          for (const m of [10, 25, 50, 75, 90, 100]) {
            if (pct >= m && this._lastLoggedPct < m) {
              this._lastLoggedPct = m;
              log("loadModel progress", m + "% loaded=" + xhr.loaded + " total=" + xhr.total);
              break;
            }
          }
        } else {
          log("loadModel progress", "loaded=" + xhr.loaded + " total=" + (xhr.total || "unknown"));
        }
      },
      (err) => {
        const errStr = err ? (err.message || err.toString()) : "unknown";
        log("loadModel ERROR", "url=" + this.url + " error=" + errStr, err && err.stack ? { stack: err.stack } : {});
        setStatus("Model failed");
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
      if (obj.morphTargetDictionary && obj.morphTargetInfluences) {
        const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
        mats.forEach((m) => { if (m) m.morphTargets = true; });
        this.morphTargetMeshes.push(obj);
        const names = Object.keys(obj.morphTargetDictionary);
        log("model blendshapes", "mesh=" + obj.name + " count=" + names.length + " names=" + (names.length > 5 ? names.slice(0, 5).join(",") + "..." : names.join(",")));
      }
    });
  }

  updateBlendshapes(blendshapes, categories) {
    for (const mesh of this.morphTargetMeshes) {
      if (!mesh.morphTargetInfluences) continue;
      const dict = mesh.morphTargetDictionary || {};
      let appliedByName = 0;
      for (const [name, value] of blendshapes) {
        let key = name;
        if (!(key in dict)) {
          const aliases = BLENDSHAPE_ALIASES[name];
          if (aliases) {
            for (const alt of aliases) {
              if (alt in dict) { key = alt; break; }
            }
          }
        }
        if (key in dict) {
          mesh.morphTargetInfluences[dict[key]] = value;
          appliedByName++;
        }
      }
      if (appliedByName === 0 && categories && categories.length > 0) {
        for (let i = 0; i < categories.length && i < mesh.morphTargetInfluences.length; i++) {
          mesh.morphTargetInfluences[i] = categories[i].score;
        }
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

let faceDetectedLogged = false;
let detectFrameCount = 0;

function detectFaceLandmarks(time) {
  if (!faceLandmarker || !video || !trackingActive) return;
  detectFrameCount++;
  try {
    const landmarks = faceLandmarker.detectForVideo(video, time);
    const matrices = landmarks.facialTransformationMatrixes;
    const blendshapes = landmarks.faceBlendshapes;
    const matricesLen = matrices ? matrices.length : 0;
    const blendshapesLen = blendshapes ? blendshapes.length : 0;
    const hasFace = matricesLen > 0 || blendshapesLen > 0;
    if (hasFace && !faceDetectedLogged) {
      faceDetectedLogged = true;
      const mpNames = blendshapes && blendshapes[0] ? blendshapes[0].categories.map((c) => c.categoryName).join(",") : "";
      log("faceDetected", "first frame matrices=" + matricesLen + " blendshapes=" + blendshapesLen);
      log("mediapipe blendshapes", "categories=" + mpNames);
    }
    const logInterval = 30;
    if (detectFrameCount % logInterval === 0) {
      if (hasFace && blendshapes && blendshapes[0]) {
        const cat = (n) => { const c = blendshapes[0].categories.find((x) => x.categoryName === n); return c ? c.score.toFixed(2) : "-"; };
        const head = matricesLen > 0 ? "ok" : "no";
        const mouth = "jawOpen=" + cat("jawOpen") + " mouthSmileL=" + cat("mouthSmileLeft") + " mouthSmileR=" + cat("mouthSmileRight") + " mouthClose=" + cat("mouthClose");
        const eyes = "blinkL=" + cat("eyeBlinkLeft") + " blinkR=" + cat("eyeBlinkRight") + " wideL=" + cat("eyeWideLeft") + " wideR=" + cat("eyeWideRight");
        log("tracking", "frame=" + detectFrameCount + " head=" + head + " face=ok | mouth: " + mouth + " | eyes: " + eyes);
      } else {
        log("tracking", "frame=" + detectFrameCount + " head=no face=no | mouth: - | eyes: -");
      }
    } else if (detectFrameCount <= 3) {
      log("detectFaceLandmarks", "frame=" + detectFrameCount + " ts=" + time + " matrices=" + matricesLen + " blendshapes=" + blendshapesLen + " avatar=" + (avatar ? "yes" : "no"));
    }
    if (!hasFace && detectFrameCount <= 5) {
      log("detectFaceLandmarks EMPTY", "frame=" + detectFrameCount + " ts=" + time + " matrices=" + matricesLen + " blendshapes=" + blendshapesLen);
    }
    if (matrices && matrices.length > 0 && avatar) {
      const m = new THREE.Matrix4().fromArray(matrices[0].data);
      avatar.applyMatrix(m, { scale: 40 });
    }
    if (blendshapes && blendshapes.length > 0 && avatar) {
      faceDetectedCount++;
      const cats = blendshapes[0].categories;
      const map = new Map();
      for (const c of cats) {
        let s = c.score;
        if (["browOuterUpLeft", "browOuterUpRight", "eyeBlinkLeft", "eyeBlinkRight"].includes(c.categoryName))
          s *= 1.2;
        map.set(c.categoryName, s);
      }
      avatar.updateBlendshapes(map, cats);
    }
  } catch (e) {
    log("detectFaceLandmarks ERROR", "message=" + (e && e.message ? e.message : String(e)), e && e.stack ? { stack: e.stack } : {});
  }
}

function onVideoFrame(now, metadata) {
  const videoTimeMs = typeof now === "number" ? now : performance.now();
  detectFaceLandmarks(videoTimeMs);
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
    video.classList.add("camera-live");
    trackingActive = true;
    video.requestVideoFrameCallback(onVideoFrame);
    log("streamWebcam", "requestVideoFrameCallback registered");
    setStatus(faceLandmarker ? "Tracking. Your face drives the avatar." : "Camera on. Loading face tracking...");
  } catch (e) {
    const errMsg = e && e.message ? e.message : (e && e.name ? e.name : String(e));
    log("streamWebcam ERROR", errMsg, e && e.stack ? { stack: e.stack } : {});
    setStatus("Camera error: " + errMsg);
    startBtn.disabled = false;
  }
}

function loadAvatar(url, source) {
  if (!scene) return;
  log("loadAvatar", "url=" + url + (source ? " source=" + source : ""));
  if (avatar) {
    avatar.remove();
    avatar = null;
  }
  avatar = new Avatar(url, scene.scene);
}

function initModelPicker() {
  const sel = document.getElementById("modelSelect");
  const fileIn = document.getElementById("fileInput");
  sel.value = "watchdog";
  sel.addEventListener("change", () => {
    const choice = sel.value;
    log("modelSelect change", "choice=" + choice);
    if (choice === "raccoon") {
      loadAvatar(MODELS.raccoon, "user-selected-raccoon");
    } else {
      loadAvatar(MODELS.watchdog, "user-selected-watchdog");
    }
  });
  fileIn.addEventListener("change", (e) => {
    const f = e.target.files?.[0];
    if (f) {
      log("fileInput", "file=" + (f.name || "unknown"));
      loadAvatar(URL.createObjectURL(f), "user-upload");
    }
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
    log("initMediaPipe ERROR", errMsg, e && e.stack ? { stack: e.stack } : {});
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
    log("startApp ERROR", "BasicScene: " + errMsg, e && e.stack ? { stack: e.stack } : {});
    setStatus("Scene failed. Check logs.");
    return;
  }
  initModelPicker();
  initSettings();
  log("startApp", "Loading primary model: Watchdog Shape Keys Compressed .glb");
  loadAvatar(MODELS.watchdog, "startup-default");
  setStatus("Loading Watchdog model...");
  try {
    await initMediaPipe();
  } catch (e) {
    log("startApp", "initMediaPipe threw: " + (e && e.message ? e.message : e));
  }
  startBtn.addEventListener("click", streamWebcam);
  setStatus("Starting camera...");
  await streamWebcam();
}
