import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';

// --- Constants and Globals ---
const MODELS = {
    default: './assets/models/model_opt.tflite',
    fallback: 'https://raw.githubusercontent.com/brisagiuliana/3d/main/assets/models/model_opt.tflite'
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
async function loadTFLiteModel(modelUrl) {
    let attempts = 0;
    const maxAttempts = 3;

    // Función auxiliar para verificar si el archivo existe y es accesible
    async function checkModelFile(url) {
        try {
            const response = await fetch(url);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            // Verificar que el archivo tiene contenido
            const buffer = await response.arrayBuffer();
            if (buffer.byteLength === 0) {
                throw new Error('El archivo del modelo está vacío');
            }
            return buffer;
        } catch (error) {
            console.error('Error al verificar el modelo:', error);
            return null;
        }
    }

    while (attempts < maxAttempts) {
        try {
            console.log(`Intento ${attempts + 1} de cargar el modelo desde: ${modelUrl}`);
            
            // Primero verificar si podemos acceder al archivo
            const modelBuffer = await checkModelFile(modelUrl);
            if (!modelBuffer) {
                throw new Error('No se pudo acceder al archivo del modelo');
            }

            // Intentar cargar el modelo desde el buffer
            const tfliteModel = await tflite.loadTFLiteModel(modelBuffer);
            console.log('Modelo TFLite cargado correctamente');
            return tfliteModel;
        } catch (error) {
            attempts++;
            console.error(`Error en intento ${attempts}:`, error);
            
            if (attempts === maxAttempts && modelUrl === MODELS.default) {
                console.log('Intentando cargar modelo alternativo...');
                return loadTFLiteModel(MODELS.fallback);
            } else if (attempts === maxAttempts) {
                throw new Error('No se pudo cargar el modelo después de múltiples intentos. Por favor, verifique que el modelo existe y es accesible.');
            }
            
            // Esperar antes del siguiente intento
            await new Promise(resolve => setTimeout(resolve, 2000));
        }
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
    
    // Obtener valores de los controles
    const depthScale = document.getElementById('depth-scale').value / 100;
    const baseHeight = document.getElementById('base-height').value / 100;
    const extrusionScale = 100.0 * depthScale;
    
    const geometry = new THREE.PlaneGeometry(width, height, width - 1, height - 1);
    const positionAttribute = geometry.getAttribute('position');
    
    // Encontrar el rango de profundidades para normalización
    let minDepth = 1.0;
    let maxDepth = 0.0;
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const depth = depthMap[y][x];
            minDepth = Math.min(minDepth, depth);
            maxDepth = Math.max(maxDepth, depth);
        }
    }
    
    // Aplicar la profundidad con base ajustable
    for (let i = 0; i < positionAttribute.count; i++) {
        const y = Math.floor(i / width);
        const depth = depthMap[y][i % width];
        
        // Normalizar la profundidad y aplicar el recorte de base
        const normalizedDepth = (depth - minDepth) / (maxDepth - minDepth);
        const adjustedDepth = Math.max(normalizedDepth, baseHeight);
        
        positionAttribute.setZ(i, (1.0 - adjustedDepth) * extrusionScale);
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

// --- Model Loading ---
let modelLoadAttempts = 0;
const maxModelLoadAttempts = 50; // 5 segundos máximo

async function loadModel() {
    try {
        tfliteModel = await loadTFLiteModel(MODEL_URL);
        console.log('Modelo cargado exitosamente');
        return true;
    } catch (error) {
        console.error('Error al cargar el modelo:', error);
        alert('Error al cargar el modelo. Por favor, recarga la página o verifica tu conexión a internet.');
        return false;
    }
}

function waitForTFLite() {
    if (typeof tflite !== 'undefined') {
        console.log('TFLite detectado, iniciando carga del modelo...');
        loadModel().then(success => {
            if (!success) {
                console.error('No se pudo cargar ningún modelo');
            }
        });
    } else {
        modelLoadAttempts++;
        if (modelLoadAttempts >= maxModelLoadAttempts) {
            console.error('TFLite no se pudo cargar después de varios intentos');
            alert('Error: No se pudo inicializar el sistema de IA. Por favor, recargue la página o intente con otro navegador.');
            return;
        }
        console.log(`Esperando que TFLite esté disponible... (intento ${modelLoadAttempts}/${maxModelLoadAttempts})`);
        setTimeout(waitForTFLite, 100);
    }
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

    // Iniciar carga del modelo
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

    // Event listeners para los controles de ajuste
    const depthSlider = document.getElementById('depth-scale');
    const baseSlider = document.getElementById('base-height');
    const depthValue = document.getElementById('depth-value');
    const baseValue = document.getElementById('base-value');

    function updateDepthValue() {
        depthValue.textContent = `${depthSlider.value}%`;
    }

    function updateBaseValue() {
        baseValue.textContent = `${baseSlider.value}%`;
    }

    function regenerateModel() {
        if (currentMesh && imagePreview.src) {
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
                        
                        if (document.getElementById('auto-center').checked) {
                            const boundingBox = new THREE.Box3().setFromObject(currentMesh);
                            const center = boundingBox.getCenter(new THREE.Vector3());
                            const size = boundingBox.getSize(new THREE.Vector3());
                            controls.target.copy(center);
                            camera.position.z = Math.max(size.x, size.y, size.z) * 1.5;
                            camera.lookAt(center);
                        }
                        
                        scene.add(currentMesh);
                        depthMapTensor.dispose();
                    }
                };
            };
            reader.readAsDataURL(uploadInput.files[0]);
        }
    }

    depthSlider.addEventListener('input', () => {
        updateDepthValue();
        regenerateModel();
    });

    baseSlider.addEventListener('input', () => {
        updateBaseValue();
        regenerateModel();
    });

    // Inicializar valores
    updateDepthValue();
    updateBaseValue();
});
