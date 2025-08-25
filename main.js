import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// --- Constants and Globals ---
const MODEL_URL = 'https://github.com/isl-org/MiDaS/releases/download/v2_1/model_opt.tflite';
let tfliteModel = null;
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
    camera.position.z = 500;
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

    window.addEventListener('resize', onWindowResize, false);

    animate();
}

function onWindowResize() {
    camera.aspect = canvas.parentElement.clientWidth / 500;
    camera.updateProjectionMatrix();
    renderer.setSize(canvas.parentElement.clientWidth, 500);
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

// --- AI & Core Logic ---
async function loadModel() {
    console.log('Loading TFLite model...');
    loadingIndicator.innerText = 'Loading AI Model...';
    loadingIndicator.style.display = 'block';
    try {
        tfliteModel = await tflite.loadTFLiteModel(MODEL_URL);
        console.log('Model loaded successfully.');
        loadingIndicator.innerText = 'Model loaded. Ready to generate.';
    } catch (e) {
        console.error('Failed to load model:', e);
        alert('Failed to load the AI model. Please check the console for details.');
        loadingIndicator.innerText = 'Model failed to load.';
    }
}

async function estimateDepth(imgElement) {
    if (!tfliteModel) {
        alert('Model is not loaded yet.');
        return null;
    }
    console.log('Estimating depth with TFLite model...');
    loadingIndicator.innerText = 'Estimating depth...';
    loadingIndicator.style.display = 'block';

    const tensor = tf.tidy(() => {
        let input = tf.browser.fromPixels(imgElement);
        // TFLite model expects input of size 256x256
        const resized = tf.image.resizeBilinear(input, [256, 256]);
        // Normalize to [0,1]
        const normalized = resized.div(255.0);
        // Add batch dimension
        const batched = normalized.expandDims(0);

        // Run inference
        let output = tfliteModel.predict(batched);

        // Post-process the output
        output = tf.squeeze(output);
        output = tf.div(
            tf.sub(output, tf.min(output)),
            tf.sub(tf.max(output), tf.min(output))
        );
        return output;
    });

    console.log('Depth estimation complete.');
    loadingIndicator.style.display = 'none';
    return tensor;
}

async function createMeshFromDepthMap(depthMapTensor, textureImage) {
    // --- DEBUGGING: Use a synthetic depth map (a ramp) instead of the AI output ---
    const height = 256;
    const width = 256;
    const depthMap = [];
    for (let y = 0; y < height; y++) {
        const row = [];
        for (let x = 0; x < width; x++) {
            row.push(y / height); // Create a simple ramp from 0 to 1
        }
        depthMap.push(row);
    }
    // --- END DEBUGGING ---

    const extrusionScale = 100.0;

    const geometry = new THREE.PlaneGeometry(width, height, width - 1, height - 1);
    const positionAttribute = geometry.getAttribute('position');

    for (let i = 0; i < positionAttribute.count; i++) {
        const x = Math.round(i % width);
        const y = Math.floor(i / width);
        const depth = depthMap[y][x];
        positionAttribute.setZ(i, (1.0 - depth) * extrusionScale); // Invert depth
    }
    geometry.computeVertexNormals();

    const texture = new THREE.Texture(textureImage);
    texture.needsUpdate = true;

    const material = new THREE.MeshStandardMaterial({
        map: texture,
        side: THREE.DoubleSide
    });

    const mesh = new THREE.Mesh(geometry, material);
    mesh.rotation.x = -Math.PI / 2;
    return mesh;
}

// --- Event Listeners ---
document.addEventListener('DOMContentLoaded', () => {
    initThree();

    // Poll until the tflite library is ready
    function waitForTFLite() {
        if (typeof tflite !== 'undefined') {
            loadModel();
        } else {
            console.log('TFLite library not ready, waiting...');
            setTimeout(waitForTFLite, 100);
        }
    }
    waitForTFLite();

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

                    // The geometry is already created with the correct aspect ratio (256x256).
                    // We don't need to scale it further.
                    // Let's center the camera and set a reasonable distance.
                    const boundingBox = new THREE.Box3().setFromObject(currentMesh);
                    const center = boundingBox.getCenter(new THREE.Vector3());
                    const size = boundingBox.getSize(new THREE.Vector3());

                    controls.target.copy(center);
                    camera.position.z = Math.max(size.x, size.y, size.z) * 1.5;
                    camera.lookAt(center);

                    scene.add(currentMesh);
                    canvas.style.display = 'block';

                    depthMapTensor.dispose();
                }
            };
        };
        reader.readAsDataURL(file);
    });
});
