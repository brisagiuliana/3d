import os
import uuid
from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
import trimesh
from PIL import Image
from transformers import pipeline
import cv2

# --- Application Setup ---
app = Flask(__name__)
# Create a directory for static files if it doesn't exist
STATIC_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
os.makedirs(STATIC_FOLDER, exist_ok=True)
app.config['STATIC_FOLDER'] = STATIC_FOLDER

# --- AI Model Initialization ---
# Load the model once at startup for efficiency
print("Initializing depth estimation model...")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")
try:
    DEPTH_ESTIMATOR = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-base-hf", device=DEVICE)
    print("Depth estimation model loaded successfully.")
except Exception as e:
    print(f"Error loading AI model: {e}")
    DEPTH_ESTIMATOR = None

def generate_normal_map(depth_map, strength=2.0):
    """
    Generates a normal map from a depth map (heightmap) using Sobel filters.
    Args:
        depth_map (np.array): The input depth map.
        strength (float): How much to exaggerate the normal map details.
    Returns:
        Image: A PIL Image of the generated normal map.
    """
    # Use Sobel filters to find the gradients in x and y directions
    sobel_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=5)

    # The z component is constant before normalization. We scale x and y.
    normal_map_vectors = np.stack([
        -sobel_x * strength,
        -sobel_y * strength,
        np.ones_like(depth_map)
    ], axis=-1)

    # Normalize each vector to have a length of 1
    norms = np.linalg.norm(normal_map_vectors, axis=2, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    normal_map_vectors /= norms

    # Map the vectors from the [-1, 1] range to the [0, 255] range for an image
    normal_map_image_data = (normal_map_vectors * 0.5 + 0.5) * 255

    # Convert to an 8-bit integer image and then to a PIL Image
    normal_map_bgr = normal_map_image_data.astype(np.uint8)
    normal_map_rgb = cv2.cvtColor(normal_map_bgr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(normal_map_rgb, 'RGB')

# --- Core 3D Conversion Logic ---
def create_3d_model_from_image(
    image_stream,
    output_path,
    extrusion_scale=50.0,
    generate_normal=False
):
    if DEPTH_ESTIMATOR is None:
        raise RuntimeError("AI model is not available.")

    print("Loading image from stream...")
    image = Image.open(image_stream).convert("RGB")
    width, height = image.size

    print("Estimating depth from image...")
    result = DEPTH_ESTIMATOR(image)
    depth_tensor = result['predicted_depth']
    depth_numpy = depth_tensor.squeeze().cpu().numpy()

    depth_height, depth_width = depth_numpy.shape
    image_resized = image.resize((depth_width, depth_height))

    # Generate Normal Map if requested
    normal_map_image = None
    if generate_normal:
        print("Generating normal map...")
        normal_map_image = generate_normal_map(depth_numpy)

    print("Creating 3D mesh...")
    depth_min, depth_max = depth_numpy.min(), depth_numpy.max()
    if depth_max > depth_min:
        depth_normalized = 1.0 - ((depth_numpy - depth_min) / (depth_max - depth_min))
    else:
        depth_normalized = np.zeros_like(depth_numpy)

    z_coords = depth_normalized * extrusion_scale

    x = np.linspace(0, width, depth_width)
    y = np.linspace(0, height, depth_height)
    xv, yv = np.meshgrid(x, y)

    vertices = np.stack([
        xv.flatten() - (width / 2),
        -(yv.flatten() - (height / 2)),
        z_coords.flatten()
    ], axis=-1)

    faces = []
    for i in range(depth_height - 1):
        for j in range(depth_width - 1):
            v1, v2 = i * depth_width + j, i * depth_width + (j + 1)
            v3, v4 = (i + 1) * depth_width + j, (i + 1) * depth_width + (j + 1)
            faces.extend([[v1, v2, v4], [v1, v4, v3]])
    faces = np.array(faces)

    print("Applying texture...")
    u = xv.flatten() / width
    v = 1.0 - (yv.flatten() / height)
    uv = np.stack([u, v], axis=-1)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    material_kwargs = {
        'baseColorTexture': image_resized
    }
    if normal_map_image:
        print("Attaching normal map to material...")
        material_kwargs['normalTexture'] = normal_map_image

    pbr_material = trimesh.visual.material.PBRMaterial(**material_kwargs)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=pbr_material)

    print(f"Exporting model to {output_path}...")
    mesh.export(file_obj=output_path, file_type='glb')
    print("Model exported successfully.")

# --- Flask Routes ---
@app.route('/')
def index():
    """Serves the main HTML page."""
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_model():
    """Handles image upload and 3D model generation."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No selected file'})

    if file and DEPTH_ESTIMATOR:
        try:
            filename = f"{uuid.uuid4()}.glb"
            output_filepath = os.path.join(app.config['STATIC_FOLDER'], filename)

            # Get generation parameters from the form
            scale = float(request.form.get('scale', 50.0))
            use_normal_map = request.form.get('use_normal_map') == 'true'

            create_3d_model_from_image(
                file.stream,
                output_filepath,
                extrusion_scale=scale,
                generate_normal=use_normal_map
            )

            model_url = f'/static/{filename}'
            return jsonify({'success': True, 'model_url': model_url})
        except Exception as e:
            print(f"An error occurred during model generation: {e}")
            return jsonify({'success': False, 'error': str(e)})

    return jsonify({'success': False, 'error': 'Server or model not ready'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
