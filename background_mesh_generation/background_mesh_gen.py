import yaml
from pathlib import Path
import open3d as o3d
import numpy as np
import trimesh
import cv2
from pymeshlab import MeshSet
from PIL import Image
from sklearn.neighbors import NearestNeighbors
import json
from tqdm import tqdm

def create_heightmap_mesh_from_depth():
    """
    from single depth image + color image, create a regular mesh (heightmap) and apply the original image as texture.
    return: trimesh.Trimesh  (with texture)
    """
    out_dir = Path("background_mesh_generation")

    depth_path = out_dir  /  Path("depth_map.jpg")
    point_path = out_dir  / Path("background_pointmap.npy")
    color_path = out_dir  / Path("background.webp")
    save_path = out_dir  / Path("background_textured_mesh.glb")
    cam_path = out_dir  / Path("camera.json")
    cam_dict = json.load(open(cam_path))
    fx, fy = cam_dict['focal_length'], cam_dict['focal_length']

    depth_img = Image.open(depth_path)
    depth_arr = np.array(depth_img, dtype=np.float32)

    H, W = depth_arr.shape[:2]
    print(f"Depth image size = {W} x {H}")
    cx, cy = W / 2, H / 2

    pointmap = np.load(point_path)
    H, W = pointmap.shape[:2]

    color_img = Image.open(color_path).convert("RGB")
    if color_img.size != (W, H):
        print("Warning: color image size != depth size. Will resize color image.")
        color_img = color_img.resize((W, H), Image.LANCZOS)

    # # construct vertices (X, Y, Z)
    # # every pixel (i,j) → vertex index: idx = i*W + j
    vertices = np.zeros((H * W, 3), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            # d = depth_arr[i, j]
            d = pointmap[i, j, 2]  # depth is the third channel
            if d <= 0:
                d = 1e-6  # avoid zero depth

            X = (j - cx) / fx * d
            Y = (i - cy) / fy * d
            # X = pointmap[i, j, 0]
            # Y = pointmap[i, j, 1]
            Z = d
            idx = i * W + j
            vertices[idx, 0] = X
            vertices[idx, 1] = Y
            vertices[idx, 2] = Z


    # vertices = pointmap[..., :3]

    # construct faces (triangles)
    #    t1 = (i,j), (i,j+1), (i+1,j)
    #    t2 = (i,j+1), (i+1,j+1), (i+1,j)
    faces = []
    for i in range(H - 1):
        for j in range(W - 1):
            idx0 = i * W + j
            idx1 = i * W + (j + 1)
            idx2 = (i + 1) * W + j
            idx3 = (i + 1) * W + (j + 1)

            # triangle1
            faces.append([idx0, idx1, idx2])
            # triangle2
            faces.append([idx1, idx3, idx2])

    faces = np.array(faces, dtype=np.int32)

    # compute UV coords: (u, v) in [0,1]
    #    u = j / (W - 1), v = 1 - (i / (H - 1))
    uv_coords = np.zeros((H * W, 2), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            idx = i * W + j
            u = j / float(W - 1) if W > 1 else 0
            v = 1.0 - (i / float(H - 1) if H > 1 else 0)
            uv_coords[idx, 0] = u
            uv_coords[idx, 1] = v

    # use trimesh to build the mesh with predefined vertices, faces, and UV coords
    # process=False to avoid trimesh's internal processing (e.g., merging vertices)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # apply texture in trimesh
    #    - trimesh.visual.TextureVisuals requires: uv.shape == (num_vertices, 2)
    #    - image is PIL Image object
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv_coords, image=color_img)

    # thicken the mesh
    # thick_mesh = add_thickness_below_mesh_preserve_texture(mesh, thickness=0.1)
    thick_mesh = mesh

    # save to GLB format
    thick_mesh.export(str(save_path))
    print(f"Saved mesh with texture to: {save_path}")

    return thick_mesh

if __name__ == "__main__":
    create_heightmap_mesh_from_depth()