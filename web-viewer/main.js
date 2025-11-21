// main.js
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";

// 基本场景
const container = document.getElementById("viewer-container");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x050608);

const camera = new THREE.PerspectiveCamera(
  45,
  window.innerWidth / window.innerHeight,
  0.001,
  10.0
);
camera.position.set(0.2, 0.2, 0.2);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setSize(window.innerWidth, window.innerHeight);
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 0, 0);

const ambient = new THREE.AmbientLight(0xffffff, 0.4);
scene.add(ambient);
const dirLight = new THREE.DirectionalLight(0xffffff, 0.6);
dirLight.position.set(1, 1, 1);
scene.add(dirLight);

const loader = new PLYLoader();

// 当前加载的模型
let currentPoints = null;
let bboxGlobal = new THREE.Box3();
let bboxInitialized = false;

function clearCurrentPoints() {
  if (currentPoints) {
    scene.remove(currentPoints);
    currentPoints.geometry.dispose();
    currentPoints.material.dispose();
    currentPoints = null;
  }
  bboxInitialized = false;
  bboxGlobal.makeEmpty();
}

function updateCameraToFitBox() {
  if (!bboxInitialized) return;

  const size = new THREE.Vector3();
  const center = new THREE.Vector3();
  bboxGlobal.getSize(size);
  bboxGlobal.getCenter(center);

  const radius = size.length() * 0.5 || 0.1;
  const dist = radius * 3.0;

  camera.position.copy(center.clone().add(new THREE.Vector3(dist, dist, dist)));
  camera.near = radius / 100.0;
  camera.far = radius * 20.0;
  camera.updateProjectionMatrix();

  controls.target.copy(center);
  controls.update();
}

function createPointsFromGeometry(geometry) {
  geometry.computeBoundingBox();

  const material = new THREE.PointsMaterial({
    size: 0.0015,
    color: new THREE.Color(0xffb020),
    transparent: true,
    opacity: 0.9,
  });

  const points = new THREE.Points(geometry, material);

  const attr = geometry.getAttribute("position");
  const box = new THREE.Box3().setFromBufferAttribute(attr);
  if (!bboxInitialized) {
    bboxGlobal.copy(box);
    bboxInitialized = true;
  } else {
    bboxGlobal.union(box);
  }

  return points;
}

function loadModel(filename) {
  if (!filename) return;
  clearCurrentPoints();

  const fullPath = `./models/${filename}`;
  console.log("Loading PLY:", fullPath);

  loader.load(
    fullPath,
    (geometry) => {
      geometry.center(); // 让模型在原点附近
      currentPoints = createPointsFromGeometry(geometry);
      scene.add(currentPoints);
      updateCameraToFitBox();
      console.log("Loaded:", filename);
    },
    undefined,
    (err) => {
      console.error("Error loading PLY:", err);
    }
  );
}

// 从 index.json 读取文件列表，填充下拉框
async function initModelList() {
  console.log("Initializing model list...");
  const select = document.getElementById("model-select");
  try {
    const resp = await fetch("./models/index.json");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    
    const data = await resp.json();
    console.log("Loaded models/index.json:", data);
    const models = data.models;

    select.innerHTML = "";
    if (!models || !models.length) {
      const opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "-- no PLY files found --";
      select.appendChild(opt);
      return;
    }

    // 默认插入一个“请选择”
    const opt0 = document.createElement("option");
    opt0.value = "";
    opt0.textContent = "-- select a PLY model --";
    select.appendChild(opt0);

    for (const m of models) {
      const opt = document.createElement("option");
      opt.value = m.filename;
      opt.textContent = m.label || m.filename;
      select.appendChild(opt);
    }

    // 当选择变化时加载模型
    select.addEventListener("change", () => {
      const fname = select.value;
      if (fname) {
        loadModel(fname);
      }
    });

    // 也可以默认选择第一个模型：
    // select.value = files[0];
    // loadModel(files[0]);
  } catch (e) {
    console.error("Failed to load models/index.json:", e);
    select.innerHTML = "";
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "-- failed to load models/index.json --";
    select.appendChild(opt);
  }
}

initModelList();

// 自适应窗口
window.addEventListener("resize", () => {
  const w = window.innerWidth;
  const h = window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
});

// 渲染循环
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();
