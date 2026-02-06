/**
 * FaceOn — Face tracking avatar powered by MediaPipe + Three.js
 * Uses 52 ARKit blend shapes for facial expressions.
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import {
  FilesetResolver,
  FaceLandmarker,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// Model URLs
const MODELS = {
  watchdog: "models/WatchDog_52blendshapes.glb",
  raccoon: "https://assets.codepen.io/9177687/raccoon_head.glb",
};

function getViewportSizeAtDepth(camera, depth) {
  const viewportHeightAtDepth =
    2 * depth * Math.tan(THREE.MathUtils.degToRad(0.5 * camera.fov));
  const viewportWidthAtDepth = viewportHeightAtDepth * camera.aspect;
  return new THREE.Vector2(viewportWidthAtDepth, viewportHeightAtDepth);
}

function createCameraPlaneMesh(camera, depth, material) {
  const viewportSize = getViewportSizeAtDepth(camera, depth);
  const geometry = new THREE.PlaneGeometry(
    viewportSize.width,
    viewportSize.height
  );
  geometry.translate(0, 0, -depth);
  return new THREE.Mesh(geometry, material);
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

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
    this.scene.add(ambientLight);
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.5);
    dirLight.position.set(0, 1, 0);
    this.scene.add(dirLight);

    this.camera.position.z = 0;
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    const orbitTarget = this.camera.position.clone();
    orbitTarget.z -= 5;
    this.controls.target = orbitTarget;
    this.controls.update();

    const video = document.getElementById("video");
    const videoTexture = new THREE.VideoTexture(video);
    videoTexture.colorSpace = THREE.SRGBColorSpace;
    const plane = createCameraPlaneMesh(
      this.camera,
      500,
      new THREE.MeshBasicMaterial({ map: videoTexture })
    );
    this.scene.add(plane);

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
        console.log("Model loaded:", url);
      },
      undefined,
      (err) => console.error("Failed to load model:", err)
    );
  }

  init(gltf) {
    this.root = null;
    gltf.scene.traverse((object) => {
      if (object.isBone && !this.root) this.root = object;
      if (!object.isMesh) return;
      const mesh = object;
      mesh.frustumCulled = false;
      if (mesh.morphTargetDictionary && mesh.morphTargetInfluences) {
        this.morphTargetMeshes.push(mesh);
      }
    });
  }

  updateBlendshapes(blendshapes) {
    for (const mesh of this.morphTargetMeshes) {
      if (!mesh.morphTargetDictionary || !mesh.morphTargetInfluences) continue;
      for (const [name, value] of blendshapes) {
        if (!(name in mesh.morphTargetDictionary)) continue;
        const idx = mesh.morphTargetDictionary[name];
        mesh.morphTargetInfluences[idx] = value;
      }
    }
  }

  applyMatrix(matrix, opts = {}) {
    const { scale = 40 } = opts;
    if (!this.gltf) return;
    matrix.scale(new THREE.Vector3(scale, scale, scale));
    this.gltf.scene.matrixAutoUpdate = false;
    this.gltf.scene.matrix.copy(matrix);
  }
}

let faceLandmarker = null;
let video = null;
let avatar = null;

const scene = new BasicScene();

function detectFaceLandmarks(time) {
  if (!faceLandmarker || !video) return;
  const landmarks = faceLandmarker.detectForVideo(video, time);

  const matrices = landmarks.facialTransformationMatrixes;
  if (matrices && matrices.length > 0 && avatar) {
    const matrix = new THREE.Matrix4().fromArray(matrices[0].data);
    avatar.applyMatrix(matrix, { scale: 40 });
  }

  const blendshapes = landmarks.faceBlendshapes;
  if (blendshapes && blendshapes.length > 0 && avatar) {
    const coefsMap = new Map();
    for (const cat of blendshapes[0].categories) {
      let score = cat.score;
      if (["browOuterUpLeft", "browOuterUpRight", "eyeBlinkLeft", "eyeBlinkRight"].includes(cat.categoryName)) {
        score *= 1.2;
      }
      coefsMap.set(cat.categoryName, score);
    }
    avatar.updateBlendshapes(coefsMap);
  }
}

function onVideoFrame(time) {
  detectFaceLandmarks(time);
  video.requestVideoFrameCallback(onVideoFrame);
}

async function streamWebcam() {
  video = document.getElementById("video");
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: "user", width: 1280, height: 720 },
  });
  video.srcObject = stream;
  video.onloadedmetadata = () => video.play();
  video.requestVideoFrameCallback(onVideoFrame);
}

function loadAvatar(url) {
  avatar = new Avatar(url, scene.scene);
}

function initModelPicker() {
  const select = document.getElementById("modelSelect");
  const fileInput = document.getElementById("fileInput");

  select.addEventListener("change", () => {
    const val = select.value;
    if (val === "raccoon") loadAvatar(MODELS.raccoon);
    else loadAvatar(MODELS.watchdog);
  });

  fileInput.addEventListener("change", (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    loadAvatar(url);
  });
}

async function run() {
  initModelPicker();
  loadAvatar(MODELS.watchdog);

  await streamWebcam();

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

  console.log("FaceOn ready.");
}

run().catch((e) => console.error(e));
