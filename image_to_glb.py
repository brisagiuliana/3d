import argparse
import numpy as np
import torch
import trimesh
from PIL import Image
from transformers import pipeline

def create_3d_model_from_image(image_path, output_path, extrusion_scale=50.0):
    """
    Creates a 3D model from a single 2D image using a depth estimation AI model.

    Args:
        image_path (str): Path to the input JPG image.
        output_path (str): Path to save the output GLB file.
        extrusion_scale (float): Factor to scale the depth map and control the 3D effect.
    """
    print("Loading image...")
    try:
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Input image not found at {image_path}")
        return

    width, height = image.size

    print("Initializing depth estimation model... (This may take a moment on first run)")
    # Use CUDA if available, otherwise CPU
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    depth_estimator = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-base-hf", device=device)

    print("Estimating depth from image...")
    # The pipeline returns a dictionary, we need the 'predicted_depth' tensor
    result = depth_estimator(image)
    depth_tensor = result['predicted_depth']

    # Move tensor to CPU and convert to NumPy array
    depth_numpy = depth_tensor.squeeze().cpu().numpy()

    # The output depth map might have a different resolution.
    # We resize the original image to match the depth map for texturing.
    depth_height, depth_width = depth_numpy.shape
    image_resized = image.resize((depth_width, depth_height))

    print("Creating 3D mesh...")

    # Normalize depth map for scaling
    depth_min = depth_numpy.min()
    depth_max = depth_numpy.max()
    if depth_max > depth_min:
        # Invert the depth map so that brighter (closer) parts pop out.
        # Models often output depth where smaller values are closer.
        # We subtract from 1 to make closer objects have higher values.
        depth_normalized = 1.0 - ((depth_numpy - depth_min) / (depth_max - depth_min))
    else:
        depth_normalized = np.zeros_like(depth_numpy)

    z_coords = depth_normalized * extrusion_scale

    # Create a grid of vertices
    x = np.linspace(0, width, depth_width)
    y = np.linspace(0, height, depth_height)
    xv, yv = np.meshgrid(x, y)

    # Vertices are (x, y, z)
    # We flip yv to match image coordinates (origin top-left) and center the mesh
    vertices = np.stack([
        xv.flatten() - (width / 2),
        -(yv.flatten() - (height / 2)),
        z_coords.flatten()
    ], axis=-1)

    # Create faces for the grid
    faces = []
    for i in range(depth_height - 1):
        for j in range(depth_width - 1):
            v1 = i * depth_width + j
            v2 = i * depth_width + (j + 1)
            v3 = (i + 1) * depth_width + j
            v4 = (i + 1) * depth_width + (j + 1)
            # Create two triangles for each quad
            faces.append([v1, v2, v4])
            faces.append([v1, v4, v3])
    faces = np.array(faces)

    print("Applying texture...")

    # Create UV coordinates for texturing
    u = xv.flatten() / (width if width > 0 else 1)
    v = 1.0 - (yv.flatten() / (height if height > 0 else 1))
    uv = np.stack([u, v], axis=-1)

    # Create the Trimesh object
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Apply the texture using TextureVisuals
    material = trimesh.visual.texture.SimpleMaterial(image=image_resized)
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)

    print(f"Exporting model to {output_path}...")
    try:
        mesh.export(file_obj=output_path, file_type='glb')
        print(f"Model exported successfully to {output_path}")
    except Exception as e:
        print(f"Error exporting file: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a 2D image to a 3D GLB model.")
    parser.add_argument("input_image", type=str, help="Path to the input JPG or PNG image.")
    parser.add_argument("output_glb", type=str, help="Path to save the output GLB file.")
    parser.add_argument("--scale", type=float, default=50.0, help="Extrusion scale for the 3D effect. Adjust based on the image content.")

    args = parser.parse_args()

    create_3d_model_from_image(args.input_image, args.output_glb, args.scale)
