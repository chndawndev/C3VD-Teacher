// main.js
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";

// Basic scene
const container = document.getElementById("viewer-container");
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1e2330); // CAD-like dark blue

const gridHelper = new THREE.GridHelper(10, 20, 0x506070, 0x2a3040);
scene.add(gridHelper);

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

// Currently loaded models (up to 2)
const loadedModels = [null, null];
const modelColors = [0x99aabb, 0xffaa88]; // Blue-ish for #1, Red-ish for #2

let bboxGlobal = new THREE.Box3();

function updateGlobalBBox() {
  bboxGlobal.makeEmpty();
  let empty = true;
  
  loadedModels.forEach(pts => {
    if (pts) {
      if (!pts.geometry.boundingBox) pts.geometry.computeBoundingBox();
      bboxGlobal.union(pts.geometry.boundingBox);
      empty = false;
    }
  });
  
  return !empty;
}

function clearModel(index) {
  if (loadedModels[index]) {
    scene.remove(loadedModels[index]);
    loadedModels[index].geometry.dispose();
    loadedModels[index].material.dispose();
    loadedModels[index] = null;
  }
  updateGlobalBBox();
}

function updateCameraToFitBox() {
  if (!updateGlobalBBox()) return;

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

function createPointsFromGeometry(geometry, color) {
  geometry.computeBoundingBox();

  const material = new THREE.PointsMaterial({
    size: 0.0015,
    color: new THREE.Color(color),
    transparent: true,
    opacity: 0.9,
  });

  const points = new THREE.Points(geometry, material);

  // Add axes helper to the model
  const axesHelper = new THREE.AxesHelper(0.05); // Size 0.05m
  points.add(axesHelper);

  return points;
}

function loadModel(filename, index) {
  clearModel(index);
  if (!filename) return;

  const fullPath = `./models/${filename}`;
  console.log(`Loading PLY [${index}]:`, fullPath);

  loader.load(
    fullPath,
    (geometry) => {
      // geometry.center(); // Removed to allow alignment comparison
      const points = createPointsFromGeometry(geometry, modelColors[index]);
      loadedModels[index] = points;
      scene.add(points);
      
      updateGlobalBBox();
      updateCameraToFitBox();
      console.log(`Loaded [${index}]:`, filename);
    },
    undefined,
    (err) => {
      console.error("Error loading PLY:", err);
    }
  );
}

// Read file list from index.json and populate the dropdown
async function initModelList() {
  console.log("Initializing model list...");
  const select1 = document.getElementById("model-select-1");
  const select2 = document.getElementById("model-select-2");
  
  try {
    const resp = await fetch("./models/index.json");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    
    const data = await resp.json();
    console.log("Loaded models/index.json:", data);
    const models = data.models;

    const populate = (sel, idx) => {
      sel.innerHTML = "";
      const def = document.createElement("option");
      def.value = "";
      def.textContent = idx === 0 ? "-- select model 1 --" : "-- select model 2 (optional) --";
      sel.appendChild(def);

      if (models) {
        for (const m of models) {
          const opt = document.createElement("option");
          opt.value = m.filename;
          opt.textContent = m.label || m.filename;
          sel.appendChild(opt);
        }
      }
      
      sel.addEventListener("change", () => {
        loadModel(sel.value, idx);
      });
    };

    populate(select1, 0);
    populate(select2, 1);

  } catch (e) {
    console.error("Failed to load models/index.json:", e);
  }
}

initModelList();

// Handle window resize
window.addEventListener("resize", () => {
  const w = window.innerWidth;
  const h = window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
});

// Render loop
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();
