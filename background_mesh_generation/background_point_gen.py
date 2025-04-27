
import yaml
from pathlib import Path
import os
import json

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
import torch
from tqdm import tqdm


image_path = "background_mesh_generation/background.webp"
save_dir = "background_mesh_generation"

# Load the image
# Note: here we assume that both the image and the background image are the same height and width
color_image = Image.open(image_path).convert('RGB')
width, height = color_image.size

depth = np.load("background_mesh_generation/depth_map.npy")  # Get the depth map for the current image
resized_depth = Image.fromarray(depth).resize((width, height), Image.NEAREST)
focal_length = json.load(open("background_mesh_generation/focal_length.json"))["focal_length_px"]

# Generate mesh grid and calculate point cloud coordinates
x, y = np.meshgrid(np.arange(width), np.arange(height))
x = (x - width / 2) / focal_length  # Normalize x coordinates
# opencv image Y-down, while open3d Y-up so we need to flip the y axis
y = (y - height / 2) / focal_length # Normalize y coordinates
z = np.array(resized_depth)  # Depth in meters
pointmap = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1)

points = pointmap.reshape(-1, 3)
colors = np.array(color_image).reshape(-1, 3) / 255.0

# Create the point cloud and save it to the output directory
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud(os.path.join(save_dir, "background_points.ply"), pcd)
print(f'Point cloud saved to {os.path.join(save_dir, "background_points.ply")}')
pointmap = np.concatenate([pointmap, np.array(color_image) / 255.0], axis=-1) # [H, W, 6]
point_map_path = os.path.join(save_dir, "background_pointmap.npy")
np.save(point_map_path, pointmap)
print(f'Point map saved to {point_map_path}')

camera_save_path = os.path.join(save_dir, "camera.json")
cam_dict = {
    "width": float(width),
    "height": float(height),
    "focal_length": float(focal_length),
}
with open(camera_save_path, "w") as f:
    json.dump(cam_dict, f, indent=4)
print(f'Camera infos saved to {camera_save_path}')