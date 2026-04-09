import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";

const MODELS = {
  dino: {
    title: "Dino Ring",
    path: "models/dino.glb",
    dataset: {
      dir: "datasets/dino",
      label: "Middlebury Dino dataset",
    },
  },
  temple: {
    title: "Temple",
    path: "models/temple.glb",
    dataset: {
      dir: "datasets/temple",
      label: "Middlebury Temple dataset",
    },
  },
  templeSparseRing: {
    title: "Temple Sparse Ring",
    path: "models/templeSparseRing.glb",
    dataset: {
      dir: "datasets/templeSparseRing",
      label: "Middlebury Temple Sparse Ring dataset",
    },
  },
};

const params = new URLSearchParams(window.location.search);
const modelKey = params.get("model") || "dino";
const modelInfo = MODELS[modelKey];

const titleEl = document.getElementById("model-title");
const loadingEl = document.getElementById("loading-overlay");
const loadingText = document.getElementById("loading-text");
const container = document.getElementById("viewer-container");

if (!modelInfo) {
  titleEl.textContent = "Model not found";
  loadingText.textContent = `Unknown model: ${modelKey}`;
  throw new Error(`Unknown model: ${modelKey}`);
}

titleEl.textContent = modelInfo.title;

// --- Scene setup ---
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

const camera = new THREE.PerspectiveCamera(
  50,
  container.clientWidth / container.clientHeight,
  0.01,
  1000
);
camera.position.set(0, 0, 3);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
container.appendChild(renderer.domElement);

// --- Lighting ---
// Ambient light so the mesh is never pitch black on the shadowed side
scene.add(new THREE.AmbientLight(0xffffff, 0.6));

// Key light from the front-top
const keyLight = new THREE.DirectionalLight(0xffffff, 0.9);
keyLight.position.set(5, 5, 5);
scene.add(keyLight);

// Fill light from the opposite side for softer shading
const fillLight = new THREE.DirectionalLight(0xffffff, 0.4);
fillLight.position.set(-5, -2, -5);
scene.add(fillLight);

// --- Controls ---
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.screenSpacePanning = true;

// --- Load GLB (Draco-compressed) ---
const dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath("https://unpkg.com/three@0.160.0/examples/jsm/libs/draco/");
dracoLoader.setDecoderConfig({ type: "js" });

const loader = new GLTFLoader();
loader.setDRACOLoader(dracoLoader);

loader.load(
  modelInfo.path,
  (gltf) => {
    // Find the first mesh in the glTF scene graph
    let sourceMesh = null;
    gltf.scene.traverse((obj) => {
      if (!sourceMesh && obj.isMesh) sourceMesh = obj;
    });

    if (!sourceMesh) {
      console.error("[viewer] No mesh found in GLB");
      loadingText.textContent = "No mesh found in model.";
      return;
    }

    const geometry = sourceMesh.geometry;

    // GLB files from the converter may not include normals — recompute so
    // lighting always works. Keeping existing normals if present.
    if (!geometry.hasAttribute("normal")) {
      geometry.computeVertexNormals();
    }

    const hasColors = geometry.hasAttribute("color");

    console.log("[viewer] GLB loaded", {
      hasColors,
      vertexCount: geometry.attributes.position.count,
      hasIndex: !!geometry.index,
      indexCount: geometry.index ? geometry.index.count : 0,
      attributes: Object.keys(geometry.attributes),
    });

    // Replace the glTF-assigned material with our own museum-styled one so
    // the look matches the old PLY viewer exactly. Draco-decoded vertex
    // colors are stored as COLOR_0 and picked up by vertexColors: true.
    const material = new THREE.MeshStandardMaterial({
      color: hasColors ? 0xffffff : 0xcccccc,
      vertexColors: hasColors,
      metalness: 0.0,
      roughness: 0.85,
      side: THREE.DoubleSide,
      flatShading: false,
    });

    const mesh = new THREE.Mesh(geometry, material);

    // Center + size the mesh robustly. MVS meshes often have stray background
    // geometry that would drag the mean/bbox far from the actual object, so
    // we use medians on each axis.
    const position = geometry.attributes.position;
    const n = position.count;

    const xs = new Float32Array(n);
    const ys = new Float32Array(n);
    const zs = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      xs[i] = position.getX(i);
      ys[i] = position.getY(i);
      zs[i] = position.getZ(i);
    }
    xs.sort();
    ys.sort();
    zs.sort();

    // Use 10th-90th percentile range per axis instead of median, to tolerate
    // asymmetric outlier tails (e.g. a long background surface on one side).
    const p10 = Math.floor(n * 0.1);
    const p50 = Math.floor(n * 0.5);
    const p90 = Math.floor(n * 0.9);

    const centerX = (xs[p10] + xs[p90]) / 2;
    const centerY = (ys[p10] + ys[p90]) / 2;
    const centerZ = (zs[p10] + zs[p90]) / 2;

    // Bake the centering directly into the geometry so we don't have to juggle
    // mesh.position vs. mesh.scale ordering.
    geometry.translate(-centerX, -centerY, -centerZ);

    // Compute robust radius from distances in the newly-centered geometry
    const distances = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const dx = position.getX(i);
      const dy = position.getY(i);
      const dz = position.getZ(i);
      distances[i] = Math.sqrt(dx * dx + dy * dy + dz * dz);
    }
    distances.sort();
    const robustRadius = distances[Math.floor(n * 0.9)] || 1.0;

    // Normalize scale so the robust radius maps to a fixed target
    const targetRadius = 1.0;
    const scale = targetRadius / robustRadius;
    geometry.scale(scale, scale, scale);

    // After baking transforms into the geometry, recompute its bounding box
    // and shift the mesh so its TRUE center sits at the origin. This is the
    // reliable way to guarantee the camera is looking at the object.
    geometry.computeBoundingBox();
    const bbox = geometry.boundingBox;
    const bboxCenter = new THREE.Vector3();
    bbox.getCenter(bboxCenter);
    mesh.position.set(-bboxCenter.x, -bboxCenter.y, -bboxCenter.z);

    const bboxSize = new THREE.Vector3();
    bbox.getSize(bboxSize);
    const actualRadius = Math.max(bboxSize.x, bboxSize.y, bboxSize.z) / 2;

    scene.add(mesh);

    // Frame the camera to the actual post-transform bounding sphere.
    const fov = THREE.MathUtils.degToRad(camera.fov);
    const fitDistance = (actualRadius / Math.sin(fov / 2)) * 1.2;
    camera.position.set(fitDistance * 0.7, fitDistance * 0.35, fitDistance * 0.7);
    camera.near = Math.max(0.001, fitDistance / 1000);
    camera.far = fitDistance * 1000;
    camera.lookAt(0, 0, 0);
    camera.updateProjectionMatrix();

    controls.target.set(0, 0, 0);
    controls.minDistance = fitDistance * 0.02;
    controls.maxDistance = fitDistance * 50;
    controls.update();

    console.log("[viewer] framing", {
      preScaleCenter: [centerX, centerY, centerZ],
      robustRadius,
      scale,
      actualRadius,
      bboxCenterAfterBake: bboxCenter.toArray(),
      fitDistance,
      cameraPos: camera.position.toArray(),
      meshPos: mesh.position.toArray(),
    });

    // Hide loading overlay
    loadingEl.classList.add("hidden");
  },
  (event) => {
    if (event.lengthComputable) {
      const pct = Math.round((event.loaded / event.total) * 100);
      loadingText.textContent = `Loading model... ${pct}%`;
    } else {
      const mb = (event.loaded / (1024 * 1024)).toFixed(1);
      loadingText.textContent = `Loading model... ${mb} MB`;
    }
  },
  (error) => {
    console.error("Failed to load GLB:", error);
    loadingText.textContent = "Failed to load model.";
  }
);

// --- Dataset modal ---
const datasetButton = document.getElementById("dataset-button");
const datasetModal = document.getElementById("dataset-modal");
const datasetGrid = document.getElementById("dataset-modal-grid");
const datasetModalTitle = document.getElementById("dataset-modal-title");
const datasetModalSubtitle = document.getElementById("dataset-modal-subtitle");
const lightbox = document.getElementById("lightbox");
const lightboxImg = document.getElementById("lightbox-img");
const lightboxCaption = document.getElementById("lightbox-caption");

let datasetLoaded = false;

function openModal(el) {
  el.classList.remove("hidden");
  el.setAttribute("aria-hidden", "false");
}

function closeModal(el) {
  el.classList.add("hidden");
  el.setAttribute("aria-hidden", "true");
}

async function loadDataset() {
  if (datasetLoaded) return;
  datasetLoaded = true;

  datasetModalTitle.textContent = `${modelInfo.title} — Dataset`;
  datasetGrid.innerHTML =
    '<p style="color: var(--fg-dim); font-size: 0.75rem; grid-column: 1 / -1; text-align: center; padding: 20px;">Loading images…</p>';

  try {
    const manifestUrl = `${modelInfo.dataset.dir}/manifest.json`;
    const response = await fetch(manifestUrl);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const files = await response.json();

    datasetModalSubtitle.textContent = `${files.length} images · ${modelInfo.dataset.label}`;
    datasetGrid.innerHTML = "";

    for (const file of files) {
      const thumb = document.createElement("div");
      thumb.className = "dataset-thumb";
      thumb.setAttribute("role", "button");
      thumb.setAttribute("tabindex", "0");

      const img = document.createElement("img");
      img.loading = "lazy";
      img.decoding = "async";
      img.src = `${modelInfo.dataset.dir}/${file}`;
      img.alt = file;

      const label = document.createElement("span");
      label.className = "dataset-thumb-label";
      label.textContent = file;

      thumb.appendChild(img);
      thumb.appendChild(label);

      const openFile = () => {
        lightboxImg.src = img.src;
        lightboxImg.alt = file;
        lightboxCaption.textContent = file;
        openModal(lightbox);
      };

      thumb.addEventListener("click", openFile);
      thumb.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          openFile();
        }
      });

      datasetGrid.appendChild(thumb);
    }
  } catch (err) {
    console.error("[viewer] Failed to load dataset manifest", err);
    datasetGrid.innerHTML =
      '<p style="color: var(--fg-dim); font-size: 0.75rem; grid-column: 1 / -1; text-align: center; padding: 20px;">Failed to load dataset.</p>';
  }
}

datasetButton.addEventListener("click", async () => {
  openModal(datasetModal);
  await loadDataset();
});

document.querySelectorAll("[data-close]").forEach((el) => {
  el.addEventListener("click", () => {
    const modal = el.closest(".dataset-modal, .lightbox");
    if (modal) closeModal(modal);
  });
});

document.addEventListener("keydown", (e) => {
  if (e.key !== "Escape") return;
  if (!lightbox.classList.contains("hidden")) {
    closeModal(lightbox);
  } else if (!datasetModal.classList.contains("hidden")) {
    closeModal(datasetModal);
  }
});

// --- Resize handling ---
window.addEventListener("resize", () => {
  const w = container.clientWidth;
  const h = container.clientHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
});

// Expose for debugging in the browser console
window.__viewer = { scene, camera, controls, renderer, THREE };

// --- Render loop ---
function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();