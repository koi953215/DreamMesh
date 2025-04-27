import numpy as np
import json
from PIL import Image
import depth_pro

image_path = "background_mesh_generation/background.webp"

# Load model and preprocessing transform
model, transform = depth_pro.create_model_and_transforms()
model.eval()

# Load and preprocess an image.
image, _, f_px = depth_pro.load_rgb(image_path)
image = transform(image)

# Run inference.
prediction = model.infer(image, f_px=f_px)
depth = prediction["depth"]  # Depth in [m].
focallength_px = prediction["focallength_px"]  # Focal length in pixels.

depth = depth.cpu().numpy()  # Convert to numpy array

depth_min = np.min(depth)
depth_max = np.max(depth)
depth_normalized = (depth - depth_min) / (depth_max - depth_min + 1e-8)  # avoid division by 0
depth_uint8 = (depth_normalized * 255).astype(np.uint8)

depth_image = Image.fromarray(depth_uint8)
depth_image.save("background_mesh_generation/depth_map.jpg")

# --- Save depth map as NPY (raw depth values) ---
np.save("background_mesh_generation/depth_map.npy", depth)

# --- Save focal length to JSON ---
focal_data = {"focal_length_px": float(focallength_px)}
with open("background_mesh_generation/focal_length.json", "w") as f:
    json.dump(focal_data, f, indent=4)