import os
import uuid
from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
import trimesh
from PIL import Image
from transformers import pipeline

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

# --- Core 3D Conversion Logic ---
def create_3d_model_from_image(image_stream, output_path, extrusion_scale=50.0):
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
    material = trimesh.visual.texture.SimpleMaterial(image=image_resized)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)

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

            # Get extrusion scale from form data, with a default value
            scale = float(request.form.get('scale', 50.0))

            create_3d_model_from_image(file.stream, output_filepath, extrusion_scale=scale)

            model_url = f'/static/{filename}'
            return jsonify({'success': True, 'model_url': model_url})
        except Exception as e:
            print(f"An error occurred during model generation: {e}")
            return jsonify({'success': False, 'error': str(e)})

    return jsonify({'success': False, 'error': 'Server or model not ready'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
