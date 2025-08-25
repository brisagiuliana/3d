import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';

// --- Constants and Globals ---
const MODELS = {
    default: './assets/models/model_opt.tflite',
    fallback: 'https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/1/selfie_segmenter.tflite'
};
let MODEL_URL = MODELS.default;
let tfliteModel = null;
let scene, camera, renderer, controls;

// Función para verificar si un archivo existe
async function checkFileExists(url) {
    try {
        const response = await fetch(url, { method: 'HEAD' });
        return response.ok;
    } catch (e) {
        console.warn('Error checking file:', e);
        return false;
    }
}
let currentMesh = null;

// --- DOM Elements ---
const uploadInput = document.getElementById('image-upload');
const generateBtn = document.getElementById('generate-btn');
const downloadBtn = document.getElementById('download-btn');
const loadingIndicator = document.getElementById('loading-indicator');
const canvas = document.getElementById('renderer-canvas');
const imagePreview = document.createElement('img');
imagePreview.style.display = 'none';
document.body.appendChild(imagePreview);

// --- Three.js Setup ---
function initThree() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x132F4C);
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
    console.log('Iniciando verificación de modelos disponibles...');
    loadingIndicator.innerText = 'Inicializando sistema de IA...';
    loadingIndicator.style.display = 'block';
    
    try {
        // Verificar si el modelo principal está disponible
        const mainModelExists = await checkFileExists(MODELS.default);
        console.log('Modelo principal disponible:', mainModelExists);
        
        if (!mainModelExists) {
            console.log('Cambiando a modelo alternativo...');
            MODEL_URL = MODELS.fallback;
        }
        
        // Intentar cargar el modelo seleccionado
        console.log('Iniciando carga del modelo desde:', MODEL_URL);
        
        // Intentar la carga con timeout
        const modelLoadPromise = tflite.loadTFLiteModel(MODEL_URL);
        const timeoutPromise = new Promise((_, reject) => 
            setTimeout(() => reject(new Error('Timeout loading model')), 30000));
        
        tfliteModel = await Promise.race([modelLoadPromise, timeoutPromise]);
        
        if (tfliteModel) {
            console.log('Modelo cargado exitosamente:', {
                modelUrl: MODEL_URL,
                modelType: MODEL_URL === MODELS.default ? 'Principal' : 'Alternativo'
            });
            loadingIndicator.innerText = 'Modelo listo. Puede comenzar a generar.';
            return true;
        }
    } catch (e) {
        console.error('Error detallado al cargar el modelo:', {
            message: e.message,
            stack: e.stack,
            modelUrl: MODEL_URL,
            browserInfo: navigator.userAgent
        });
        
        // Si falló con el modelo principal, intentar con el alternativo
        if (MODEL_URL === MODELS.default) {
            console.log('Intentando con modelo alternativo...');
            MODEL_URL = MODELS.fallback;
            return loadModel(); // Recursión para intentar con el modelo alternativo
        }
        
        alert(`Error al cargar el modelo: ${e.message}. Por favor, verifique su conexión a internet.`);
        loadingIndicator.innerText = 'Error al cargar el modelo. Intente recargar la página.';
        return false;
    }
}

async function estimateDepth(imgElement) {
    if (!tfliteModel) {
        console.error('Intento de usar el modelo antes de que esté cargado');
        alert('El modelo aún no está cargado. Por favor, espere a que se complete la carga.');
        return null;
    }
    console.log('Iniciando estimación de profundidad...', {
        modelStatus: tfliteModel ? 'Cargado' : 'No cargado',
        imageSize: `${imgElement.width}x${imgElement.height}`
    });
    loadingIndicator.innerText = 'Calculando profundidad de la imagen...';
    loadingIndicator.style.display = 'block';

    const tensor = tf.tidy(() => {
        let input = tf.browser.fromPixels(imgElement);
        const resized = tf.image.resizeBilinear(input, [256, 256]);
        const normalized = resized.div(255.0);
        const batched = normalized.expandDims(0);
        let output = tfliteModel.predict(batched);
        output = tf.squeeze(output);
        output = tf.div(tf.sub(output, tf.min(output)), tf.sub(tf.max(output), tf.min(output)));
        return output;
    });

    console.log('Depth estimation complete.');
    loadingIndicator.style.display = 'none';
    return tensor;
}

async function createMeshFromDepthMap(depthMapTensor, textureImage) {
    const depthMap = await depthMapTensor.array();
    const [height, width] = depthMapTensor.shape;
    const extrusionScale = 100.0;
    const geometry = new THREE.PlaneGeometry(width, height, width - 1, height - 1);
    const positionAttribute = geometry.getAttribute('position');
    for (let i = 0; i < positionAttribute.count; i++) {
        const y = Math.floor(i / width);
        const depth = depthMap[y][i % width];
        positionAttribute.setZ(i, (1.0 - depth) * extrusionScale);
    }
    geometry.computeVertexNormals();
    const texture = new THREE.Texture(textureImage);
    texture.needsUpdate = true;
    const material = new THREE.MeshStandardMaterial({ map: texture, side: THREE.DoubleSide });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.rotation.x = -Math.PI / 2;
    return mesh;
}

// --- Exporter ---
function downloadGLB() {
    if (!currentMesh) {
        alert("No model to download. Please generate a model first.");
        return;
    }
    const exporter = new GLTFExporter();
    exporter.parse(
        currentMesh,
        (result) => {
            const blob = new Blob([result], { type: 'application/octet-stream' });
            const link = document.createElement('a');
            link.href = URL.createObjectURL(blob);
            link.download = 'model.glb';
            link.click();
        },
        (error) => {
            console.error('An error happened during GLB export.', error);
            alert('Failed to export model.');
        },
        { binary: true }
    );
}

// --- Event Listeners ---
document.addEventListener('DOMContentLoaded', () => {
    // Verificar requisitos del navegador
    if (!window.WebGLRenderingContext) {
        alert('Tu navegador no soporta WebGL, necesario para la visualización 3D.');
        return;
    }

    if (typeof tf === 'undefined') {
        console.error('TensorFlow.js no se ha cargado correctamente');
        alert('Error: No se pudo cargar la biblioteca de IA. Por favor, recarga la página.');
        return;
    }

    // Inicializar Three.js
    initThree();

    // Verificar y cargar TFLite
    let attempts = 0;
    const maxAttempts = 50; // 5 segundos máximo

    function waitForTFLite() {
        if (typeof tflite !== 'undefined') {
            console.log('TFLite detectado, iniciando carga del modelo...');
            loadModel().then(success => {
                if (!success) {
                    console.error('No se pudo cargar ningún modelo');
                }
            });
        } else {
            attempts++;
            if (attempts >= maxAttempts) {
                console.error('TFLite no se pudo cargar después de varios intentos');
                alert('Error: No se pudo inicializar el sistema de IA. Por favor, recargue la página o intente con otro navegador.');
                return;
            }
            console.log(`Esperando que TFLite esté disponible... (intento ${attempts}/${maxAttempts})`);
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
                    const boundingBox = new THREE.Box3().setFromObject(currentMesh);
                    const center = boundingBox.getCenter(new THREE.Vector3());
                    const size = boundingBox.getSize(new THREE.Vector3());
                    controls.target.copy(center);
                    camera.position.z = Math.max(size.x, size.y, size.z) * 1.5;
                    camera.lookAt(center);
                    scene.add(currentMesh);
                    canvas.style.display = 'block';
                    downloadBtn.style.display = 'inline-block';
                    depthMapTensor.dispose();
                }
            };
        };
        reader.readAsDataURL(file);
    });

    downloadBtn.addEventListener('click', downloadGLB);
});
