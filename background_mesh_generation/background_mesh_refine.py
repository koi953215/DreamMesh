from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
import transforms3d
import json

def visualize_plane_normal(plane_pts, normal, filename, num_arrow_pts=200, normal_length=0.5):
    """
    Save plane point cloud and its normal vector as a combined .glb file.

    Args:
        plane_pts (np.ndarray): Nx3 array of point cloud.
        normal (np.ndarray): 3D unit normal vector.
        filename (str): Output file name.
        point_radius (float): Radius for visualizing points as small spheres.
        normal_length (float): Length of the normal vector arrow.
    """

    plane_colors = np.tile(np.array([[0.6, 0.6, 0.6]]), (plane_pts.shape[0], 1))  # gray

    origin = np.mean(plane_pts, axis=0)
    arrow_points = np.linspace(0, normal_length, num_arrow_pts).reshape(-1, 1) * normal + origin
    arrow_colors = np.tile(np.array([[1.0, 0.0, 0.0]]), (num_arrow_pts, 1))  # red

    all_pts = np.vstack([plane_pts, arrow_points])
    all_colors = np.vstack([plane_colors, arrow_colors])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts)
    pcd.colors = o3d.utility.Vector3dVector(all_colors)
    o3d.io.write_point_cloud(filename, pcd)

def recover_camera_pose(plane_pts, out_dir):
    """
    recover camera pose from the plane points
    plane_pts: [N, 3] numpy array
    """

    # step-1: compute the normal vector of the plane using ransac
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(plane_pts)
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                             ransac_n=3,
                                             num_iterations=1000)

    # plane_model = [a, b, c, d], where ax + by + cz + d = 0
    a, b, c, d = plane_model
    normal = np.array([a, b, c], dtype=float)
    normal = normal / np.linalg.norm(normal)
    # flip the normal vector if possible

    # obj_center = np.mean(obj_pts, axis=0)
    # sign = a*obj_center[0] + b*obj_center[1] + c*obj_center[2] + d
    # if sign < 0:
    normal = -normal

    # visualize the plane normal
    save_plane_normal_path = str(out_dir / Path("ground_normal.ply"))
    visualize_plane_normal(plane_pts, normal, save_plane_normal_path, num_arrow_pts=200, normal_length=0.5)

    # step-2: compute the rotation R applied to the plane pts, so that the plane normal is aligned with the +z-axis in the world frame
    z_axis = np.array([0,0,1], dtype=np.float64)
    dot_val = np.dot(normal, z_axis)              # = cos(θ)
    angle = np.arccos(np.clip(dot_val, -1.0, 1.0))
    rot_axis = np.cross(normal, z_axis)
    axis_len = np.linalg.norm(rot_axis)

    if axis_len < 1e-8:
        # if the normal is already aligned with the z-axis
        R = np.eye(3, dtype=np.float64)
    else:
        rot_axis = rot_axis / axis_len
        R = transforms3d.axangles.axangle2mat(rot_axis, angle)

    # step-3: compute the new camera orientation using inv(R) (previously camera is facing +z-axis in the world frame)
    # R_inv = R.T
    # cam_euler = transforms3d.euler.mat2euler(R_inv, axes='sxyz')
    cam_quat = transforms3d.quaternions.mat2quat(R)
    # step-4: return: T 4x4 matrix -> point cloud transform, O 1x3 camera orientation in euler angles
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R  # without translation

    return T, cam_quat.tolist()

def mesh_refinement():
    out_dir = Path("background_mesh_generation")

    pointmap = np.load(out_dir / Path("background_pointmap.npy"))
    H, W = pointmap.shape[:2]
    pointmap = pointmap.reshape(H*W, 6)  # [H*W, 6]
    pointmap = pointmap[:, :3]  # [H*W, 3]

    T_gravity, cam_pose = recover_camera_pose(pointmap, out_dir)
    print(f"Gravity transform: {T_gravity}")
    cam_path = out_dir / Path("camera.json")
    cam_dict = json.load(open(cam_path))
    cam_dict['cam_orientation'] = cam_pose

    background_mesh_path = str(out_dir / Path("background_textured_mesh.glb"))
    background_mesh_aligned_path = str(out_dir / Path("background_textured_mesh_aligned.glb"))
    bg_mesh = trimesh.load(background_mesh_path)    
    bg_mesh.apply_transform(T_gravity)

    # Load object positions
    object_json_path = out_dir / Path("scene_generated.json")
    with open(object_json_path, "r") as f:
        object_data = json.load(f)
    
    object_positions = np.array([
        [obj["position"]["x"], obj["position"]["y"], obj["position"]["z"]] 
        for obj in object_data["objects"]
    ])  # Shape: [num_objects, 3]

    # Compute bounding box of object positions
    xyz_min = object_positions.min(axis=0)  # [3]
    xyz_max = object_positions.max(axis=0)  # [3]

    center_xy = (xyz_min[:2] + xyz_max[:2]) / 2.0  # [x_center, y_center]
    z_min = xyz_min[2]

    # Background mesh current bounds
    mesh_bounds = bg_mesh.bounds  # (2, 3)
    mesh_center = (mesh_bounds[0] + mesh_bounds[1]) / 2.0

    # Mesh extents in XY
    mesh_extent = mesh_bounds[1] - mesh_bounds[0]  # [x_size, y_size, z_size]

    # Object extents in XY
    object_extent = xyz_max - xyz_min  # [x_size, y_size, z_size]

    # Compute scaling factors
    scale_x = object_extent[0] / mesh_extent[0]
    scale_y = object_extent[1] / mesh_extent[1]
    scale_factor = max(scale_x, scale_y) * 1.1  # Add 10% margin

    if scale_factor > 1.0:
        print(f"Scaling background mesh by factor {scale_factor:.3f} to cover all objects.")
        bg_mesh.apply_scale(scale_factor)
    else:
        print("Background mesh is large enough, no scaling needed.")

    # After scaling, re-compute mesh center
    mesh_bounds = bg_mesh.bounds
    mesh_center = (mesh_bounds[0] + mesh_bounds[1]) / 2.0
    # Translation to recenter XY and align bottom Z to z_min
    translation = np.array([
        -mesh_center[0] + center_xy[0],
        -mesh_center[1] + center_xy[1],
        -mesh_bounds[0,2] + z_min -0.7,  # bottom Z to z_min
    ])
    bg_mesh.apply_translation(translation)

    T_trimesh_to_blender = np.array([
        [1, 0, 0, 0],    # X -> X
        [0, 0, 1, 0],    # Z -> Y
        [0, -1, 0, 0],   # Y -> -Z
        [0, 0, 0, 1]
    ], dtype=np.float64)
    bg_mesh.apply_transform(T_trimesh_to_blender)

    # Save the aligned and resized mesh
    bg_mesh.export(background_mesh_aligned_path)

    # Save updated camera pose
    with open(cam_path, "w") as f:
        json.dump(cam_dict, f, indent=4)

    print(f"Saved aligned and resized mesh to {background_mesh_aligned_path}")
    print(f"Updated camera pose saved to {cam_path}")


if __name__ == "__main__":
    mesh_refinement()