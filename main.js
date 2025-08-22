import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// --- Constants and Globals ---
const MODEL_URL = 'https://timmh.github.io/monocular_depth_estimation_demo/midas_u8/model.json';
let depthModel = null;
let scene, camera, renderer, controls;
let currentMesh = null;

// --- DOM Elements ---
const uploadInput = document.getElementById('image-upload');
const generateBtn = document.getElementById('generate-btn');
const loadingIndicator = document.getElementById('loading-indicator');
const canvas = document.getElementById('renderer-canvas');
const imagePreview = document.createElement('img');
imagePreview.style.display = 'none';
document.body.appendChild(imagePreview);

// --- Three.js Setup ---
function initThree() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0xeeeeee);

    const aspectRatio = canvas.clientWidth / canvas.clientHeight || 1;
    camera = new THREE.PerspectiveCamera(75, aspectRatio, 0.1, 1000);
    camera.position.z = 500; // Start further back

    renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    renderer.setSize(canvas.parentElement.clientWidth, 500);
    renderer.setPixelRatio(window.devicePixelRatio);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);
    const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
    directionalLight.position.set(5, 10, 7.5);
    scene.add(directionalLight);

    animate();
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

// --- AI & Core Logic ---
async function loadModel() {
    console.log('Loading model...');
    loadingIndicator.innerText = 'Loading AI Model...';
    loadingIndicator.style.display = 'block';
    try {
        depthModel = await tf.loadGraphModel(MODEL_URL);
        console.log('Model loaded successfully.');
        loadingIndicator.innerText = 'Model loaded. Ready to generate.';
    } catch (e) {
        console.error('Failed to load model:', e);
        alert('Failed to load the AI model. Please check the console for details.');
    }
}

async function estimateDepth(imgElement) {
    if (!depthModel) {
        alert('Model is not loaded yet.');
        return null;
    }
    console.log('Estimating depth...');
    loadingIndicator.innerText = 'Estimating depth...';
    loadingIndicator.style.display = 'block';

    const tensor = tf.tidy(() => {
        let input = tf.browser.fromPixels(imgElement);
        input = tf.image.resizeBilinear(input, [256, 256]);
        input = tf.div(input, 255);
        input = tf.transpose(input, [2, 0, 1]);
        input = tf.expandDims(input);
        let output = depthModel.execute(input);
        output = tf.squeeze(output);
        output = tf.div(tf.sub(output, tf.min(output)), tf.sub(tf.max(output), tf.min(output)));
        return output;
    });

    console.log('Depth estimation complete.');
    loadingIndicator.style.display = 'none';
    return tensor;
}

/**
 * Creates a 3D mesh from a depth map tensor and a texture image.
 * @param {tf.Tensor} depthMapTensor The depth map.
 * @param {HTMLImageElement} textureImage The image to use as a texture.
 * @returns {THREE.Mesh} The generated 3D mesh.
 */
async function createMeshFromDepthMap(depthMapTensor, textureImage) {
    const depthMap = await depthMapTensor.array();
    const [height, width] = depthMapTensor.shape;
    const extrusionScale = 100.0;

    const geometry = new THREE.PlaneGeometry(width, height, width - 1, height - 1);
    const positionAttribute = geometry.getAttribute('position');

    for (let i = 0; i < positionAttribute.count; i++) {
        const x = Math.round(i % width);
        const y = Math.floor(i / width);
        const depth = depthMap[y][x];
        positionAttribute.setZ(i, depth * extrusionScale);
    }
    geometry.computeVertexNormals();

    const texture = new THREE.Texture(textureImage);
    texture.needsUpdate = true;

    const material = new THREE.MeshStandardMaterial({
        map: texture,
        side: THREE.DoubleSide
    });

    const mesh = new THREE.Mesh(geometry, material);
    // Orient the mesh so it faces the camera
    mesh.rotation.x = -Math.PI / 2;

    return mesh;
}

// --- Event Listeners ---
document.addEventListener('DOMContentLoaded', () => {
    initThree();
    loadModel();

    generateBtn.addEventListener('click', () => {
        const file = uploadInput.files[0];
        if (!file) {
            alert('Please select an image file first.');
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.onload = async () => {
                const depthMapTensor = await estimateDepth(imagePreview);
                if (depthMapTensor) {
                    if (currentMesh) {
                        scene.remove(currentMesh);
                        currentMesh.geometry.dispose();
                        currentMesh.material.dispose();
                    }

                    currentMesh = await createMeshFromDepthMap(depthMapTensor, imagePreview);
                    scene.add(currentMesh);
                    canvas.style.display = 'block';

                    depthMapTensor.dispose();
                }
            };
        };
        reader.readAsDataURL(file);
    });
});
