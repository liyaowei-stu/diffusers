import sys
import os
import cv2
import numpy as np
import torch


from vggt.utils.geometry import unproject_depth_map_to_point_map
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    AlphaCompositor,
)
from torchvision.utils import save_image

from .common import visualize_i_view_with_keypoints

# -------------------------------------------------------------------------
# 1) Geometry functions (extrinsic, intrinsic, etc.)
# -------------------------------------------------------------------------

def convert_se3_to_homogeneous(se3):
    """
    Args:
        se3: (b, 3, 4) or (3, 4) or (b, s, 3, 4) or (s, 3, 4)

    convert se3 to homogeneous matrix

    Returns:
        homogeneous: similar to se3
    """
    input_dim = len(se3.shape)
    if len(se3.shape) == 2:
        se3 = se3[None][None]
    if len(se3.shape) == 3:
        se3 = se3[None]
    
    homogeneous = torch.eye(4).to(se3.device)

    homogeneous = homogeneous[None, None].repeat(se3.shape[0], se3.shape[1], 1, 1)
    homogeneous[:, :, :3, :3] = se3[:, :, :3, :3]
    homogeneous[:, :, :3, 3] = se3[:, :, :3, 3]

    while len(homogeneous.shape) != input_dim:
        homogeneous = homogeneous.squeeze(0)
    
    return homogeneous


def convert_homogeneous_to_se3(homogeneous):
    """
    Args:
        homogeneous: (b, 4, 4) or (4, 4) or (b, s, 4, 4) or (s, 4, 4)
    """
    if len(homogeneous.shape) == 3:
        homogeneous = homogeneous[:, :3, :]
    elif len(homogeneous.shape) == 4:
        homogeneous = homogeneous[:, :, :3, :]
    elif len(homogeneous.shape) == 2:
        homogeneous = homogeneous[:3, :]
    else:
        raise ValueError(f"relative_w2c has invalid shape: {homogeneous.shape}")
    return homogeneous


def convert_camera_pose_to_relative(extrinsic, anchor_extrinsic):
    """
    Args:
        extrinsic: (b, 3, 4) or (3, 4) or (b, s, 3, 4) or (s, 3, 4)
        anchor_extrinsic: (b, 3, 4) or (3, 4) or (b, s, 3, 4) or (s, 3, 4)

    Returns:
        relative_w2c: (b, 3, 4) or (3, 4) or (b, s, 3, 4) or (s, 3, 4)
    """
    extrinsic = convert_se3_to_homogeneous(extrinsic).to(torch.float64)
    anchor_extrinsic = convert_se3_to_homogeneous(anchor_extrinsic).to(torch.float64)

    # assert len(extrinsic.shape) == len(anchor_extrinsic.shape), "extrinsic and anchor_extrinsic must have the same number of dimensions"

    c2w = torch.linalg.inv(extrinsic)
    
    relative_c2w = anchor_extrinsic @ c2w

    relative_w2c = torch.linalg.inv(relative_c2w)

    relative_w2c = convert_homogeneous_to_se3(relative_w2c)

    relative_w2c = relative_w2c.to(torch.float32)

    return relative_w2c


def convert_opencv_to_pytorch3d_c2w(c2w):
    """
    c2w: (N, 3, 4), column major
    return: (N, 3, 4)
    """
    # rotate axis
    opencv_R, T = c2w[:, :3, :3], c2w[:, :3, 3]

    is_numpy = isinstance(opencv_R, np.ndarray)

    if is_numpy:
        pytorch_three_d_R = np.stack([-opencv_R[:, :, 0], -opencv_R[:, :, 1], opencv_R[:, :, 2]], 2)
    else:
        pytorch_three_d_R = torch.stack([-opencv_R[:, :, 0], -opencv_R[:, :, 1], opencv_R[:, :, 2]], 2)

    # pytorch_three_d_R = np.stack([-opencv_R[:, 0, :], -opencv_R[:, 1, :], opencv_R[:, 2, :]], 1)
    
    # convert to w2c
    if is_numpy:
        new_c2w = np.concatenate([pytorch_three_d_R, T[:, :, None]], axis=2) # 3*4
    else:
        new_c2w = torch.cat([pytorch_three_d_R, T[:, :, None]], dim=2) # 3*4

    return new_c2w


def calculate_extrinsic_correction(extrinsics_1, extrinsics_2):
    """
    Calculate the correction matrix between extrinsics_1 and extrinsics_2
    The correction matrix transforms extrinsics_1 to match extrinsics_2

    Args:
        extrinsics_1: (b, s, 3, 4)
        extrinsics_2: (b, s, 3, 4)

    Returns:
        correction_matrix: (b, s, 3, 4)
    """
    extrinsics_1 = convert_se3_to_homogeneous(extrinsics_1).to(torch.float64)
    extrinsics_2 = convert_se3_to_homogeneous(extrinsics_2).to(torch.float64)

    correction_matrix = torch.matmul(extrinsics_2, torch.linalg.inv(extrinsics_1))

    correction_matrix = correction_matrix[:, :, :3, :].to(torch.float32)

    correction_matrix = correction_matrix.mean(dim=1)

    correction_matrix = correction_matrix.unsqueeze(1)

    return correction_matrix


def excute_extrinsic_correction(extrinsics, correction_matrix):
    """
    Execute the correction matrix on the extrinsics

    Args:
        extrinsics: (b, s, 3, 4)
        correction_matrix: (b, s, 3, 4)

    Returns:
        corrected_extrinsics: (b, s, 3, 4)
    """

    extrinsics = convert_se3_to_homogeneous(extrinsics).to(torch.float64)
    correction_matrix = convert_se3_to_homogeneous(correction_matrix).to(torch.float64)

    corrected_extrinsics = torch.matmul(correction_matrix, extrinsics)

    corrected_extrinsics = convert_homogeneous_to_se3(corrected_extrinsics)

    corrected_extrinsics = corrected_extrinsics.to(torch.float32)

    return corrected_extrinsics


def calculate_intrinsic_correction(intrinsic_1, intrinsic_2):
    """
    Calculate the scale between two intrinsic matrices.
    Mainly focuses on the focal length scale change.
    The scale transforms intrinsic_1 to match intrinsic_2
    
    Args:
        intrinsic_1: (B, 3, 3) or (B, S, 3, 3)
        intrinsic_2: (B, 3, 3) or (B, S, 3, 3)
        
    Returns:
        correction_scale: (B,) or (B, S) - scale factors for focal length
    """
    # Extract focal lengths (fx, fy)
    if len(intrinsic_1.shape) == 3:  # (B, 3, 3)
        fx1, fy1 = intrinsic_1[:, 0, 0], intrinsic_1[:, 1, 1]
        fx2, fy2 = intrinsic_2[:, 0, 0], intrinsic_2[:, 1, 1]
    else:  # (B, S, 3, 3)
        fx1, fy1 = intrinsic_1[:, :, 0, 0], intrinsic_1[:, :, 1, 1]
        fx2, fy2 = intrinsic_2[:, :, 0, 0], intrinsic_2[:, :, 1, 1]
    
    # Calculate scale factors
    scale_x = fx2 / fx1
    scale_y = fy2 / fy1
    
    # Average the scale factors (could also use just one if they should be identical)
    correction_scale = (scale_x + scale_y) / 2.0

    scale_x = scale_x.mean()
    scale_y = scale_y.mean()
    correction_scale = correction_scale.mean()
    
    return scale_x, scale_y, correction_scale


def excute_intrinsic_correction(intrinsic, scale_x, scale_y, intrinsic_scale=None):
    """
    Execute the scale on the intrinsic.
    Args:
        intrinsic: (B, 3, 3) or (B, S, 3, 3)
        scale_x: (B,) or (B, S)
        scale_y: (B,) or (B, S)

    Returns:
        corrected_intrinsic: (B, 3, 3) or (B, S, 3, 3)
    """
    if intrinsic_scale is not None:
        scale_x = intrinsic_scale
        scale_y = intrinsic_scale

    intrinsic = intrinsic.clone()
    if len(intrinsic.shape) == 3:
        intrinsic[:, 0, 0] *= scale_x
        intrinsic[:, 1, 1] *= scale_y
    else:
        intrinsic[:, :, 0, 0] *= scale_x
        intrinsic[:, :, 1, 1] *= scale_y

    return intrinsic


def unproject_pixels_to_points(pixels, depth, intrinsic):
    """
    Unproject 2D pixels to 3D points using depth and camera intrinsic matrix.
    
    Args:
        pixels: (N, 2) tensor of pixel coordinates (x, y)
        depth: (N,) tensor of depth values for each pixel
        intrinsic: (3, 3) camera intrinsic matrix
        
    Returns:
        points_3d: (N, 3) tensor of 3D points in camera coordinate system
    """
    # Make sure inputs are tensors
    if not isinstance(pixels, torch.Tensor):
        pixels = torch.tensor(pixels, dtype=torch.float32)
    if not isinstance(depth, torch.Tensor):
        depth = torch.tensor(depth, dtype=torch.float32)
    if not isinstance(intrinsic, torch.Tensor):
        intrinsic = torch.tensor(intrinsic, dtype=torch.float32)
    
    # Extract camera parameters
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    
    # Unproject pixels to normalized image coordinates
    x = (pixels[:, 0] - cx) / fx
    y = (pixels[:, 1] - cy) / fy
    
    # Scale by depth to get 3D points in camera coordinates
    points_3d = torch.stack([
        x * depth,  # X = (u - cx) * Z / fx
        y * depth,  # Y = (v - cy) * Z / fy
        depth       # Z = depth
    ], dim=1)
    
    return points_3d


def transform_points_between_coordinate_system(points_3d, extrinsic_src, extrinsic_dst):
    """
    Transform 3D points from one camera coordinate system to another.
    
    Args:
        points_3d: (N, 3) tensor of 3D points in source camera coordinate system
        extrinsic_src: (3, 4) or (4, 4) extrinsic matrix of source camera
        extrinsic_dst: (3, 4) or (4, 4) extrinsic matrix of destination camera
        
    Returns:
        points_3d_dst: (N, 3) tensor of 3D points in destination camera coordinate system
    """
    # Make sure inputs are tensors
    if not isinstance(points_3d, torch.Tensor):
        points_3d = torch.tensor(points_3d, dtype=torch.float32)
    if not isinstance(extrinsic_src, torch.Tensor):
        extrinsic_src = torch.tensor(extrinsic_src, dtype=torch.float32)
    if not isinstance(extrinsic_dst, torch.Tensor):
        extrinsic_dst = torch.tensor(extrinsic_dst, dtype=torch.float32)
    
    # Convert extrinsics to homogeneous if needed
    extrinsic_src_homo = convert_se3_to_homogeneous(extrinsic_src)
    extrinsic_dst_homo = convert_se3_to_homogeneous(extrinsic_dst)
    
    # Convert points to homogeneous coordinates
    N = points_3d.shape[0]
    points_homo = torch.ones((N, 4), dtype=points_3d.dtype, device=points_3d.device)
    points_homo[:, :3] = points_3d
    
    # Transform points from camera to world coordinates
    c2w_src = torch.linalg.inv(extrinsic_src_homo)
    points_world = torch.matmul(c2w_src, points_homo.t()).t()  # (N, 4)
    
    # Transform points from world to destination camera coordinates
    points_dst = torch.matmul(extrinsic_dst_homo, points_world.t()).t()  # (N, 4)
    
    # Return 3D points in destination camera coordinates
    return points_dst[:, :3]


def project_points_to_pixels(points_3d, intrinsic):
    """
    Project 3D points to 2D pixels using camera intrinsic matrix.
    
    Args:
        points_3d: (N, 3) tensor of 3D points in camera coordinate system
        intrinsic: (3, 3) camera intrinsic matrix
        
    Returns:
        pixels: (N, 2) tensor of pixel coordinates (x, y)
    """
    # Make sure inputs are tensors
    if not isinstance(points_3d, torch.Tensor):
        points_3d = torch.tensor(points_3d, dtype=torch.float32)
    if not isinstance(intrinsic, torch.Tensor):
        intrinsic = torch.tensor(intrinsic, dtype=torch.float32)
    
    # Extract camera parameters
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    
    # Get X, Y, Z coordinates
    X = points_3d[:, 0]
    Y = points_3d[:, 1]
    Z = points_3d[:, 2]
    
    # Handle points behind the camera (negative Z)
    valid_mask = Z > 0
    
    # Project 3D points to 2D pixels
    x = torch.zeros_like(X)
    y = torch.zeros_like(Y)
    
    # Only project valid points
    x[valid_mask] = fx * (X[valid_mask] / Z[valid_mask]) + cx
    y[valid_mask] = fy * (Y[valid_mask] / Z[valid_mask]) + cy
    
    pixels = torch.stack([x, y], dim=1)
    
    return pixels


# -------------------------------------------------------------------------
# 2) Rasterization and Rendering functions
# -------------------------------------------------------------------------

def define_rasterizer_renderer(cameras, image_size=(392, 518), radius=0.003, points_per_pixel=10, bin_size=None):
    """
    Define the rasterizer and renderer for the point cloud
    """
    raster_settings = PointsRasterizationSettings(
        image_size=image_size, 
        radius = radius,
        points_per_pixel = points_per_pixel,
        bin_size=bin_size,
    )
    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor(background_color=(255, 255, 255))
    )
    
    return rasterizer, renderer


def verify_point_coordinate_mapping(points_flat_ori, point, coord, h, w):
    """
    Verify that the point at the given coordinate in the original flat array matches the given point.
    """
    # coord: (3,) -> (b, y, x)
    # points_flat_ori: (N, 3)
    # h, w: image height and width
    # The index in the flat array is: idx = coord[1] * w + coord[2]
    idx = coord[1] * w + coord[2]
    assert np.allclose(points_flat_ori[idx], point), "points_ori_flat and points_flat are not the same at the given coord"


def get_points_from_mask(points_flat, colors_flat, conf_flat, coords_flat, masks):
    """
    Efficiently filter points, colors, conf, and coords by mask.

    Args:
        points_flat: (N, 3), numpy array
        colors_flat: (N, 3), numpy array
        conf_flat: (N,), numpy array
        coords_flat: (N, 3), numpy array
        masks: (B, S, H, W), numpy array

    Returns:
        points_flat, colors_flat, conf_flat, coords_flat: filtered arrays
    """
    h, w = masks.shape[-2:]
    masks_flat_bool = masks.reshape(-1).astype(bool)
    # Use boolean indexing directly, no need to copy points_flat
    points_flat_masked = points_flat[masks_flat_bool]
    colors_flat_masked = colors_flat[masks_flat_bool]
    conf_flat_masked = conf_flat[masks_flat_bool]
    coords_flat_masked = coords_flat[masks_flat_bool]

    # Optionally check the first point for debugging, but skip in production for speed
    # verify_point_coordinate_mapping(points_flat, points_flat_masked[0], coords_flat_masked[0], h, w)

    return points_flat_masked, colors_flat_masked, conf_flat_masked, coords_flat_masked


def get_points_from_predictions(predictions, use_point_map: bool = False, max_points: int = 1000000, conf_threshold: float = 20.0, conf_threshold_value: float = 2.0, apply_mask: bool = False, recon_intrinsic: torch.Tensor = None, recon_extrinsic: torch.Tensor = None):
    """
    Args:
    predictions: dict
    use_point_map: bool, if True, use point map, otherwise use depth map
    max_points: int, max points to sample
    conf_threshold: float, confidence threshold
    apply_mask: bool, if True, apply mask to the points
    device: str, device to use

    Returns:
        points_flat: (N, 3), numpy array
        colors_flat: (N, 3), numpy array
        conf_flat: (N,), numpy array
    """
        
    images = predictions["images"]  # (B, S, 3, H, W)
    world_points_map = predictions["world_points"]  # (B, S, H, W, 3)
    conf_map = predictions["world_points_conf"]  # (B, S, H, W)
    depth_map = predictions["depth"]  # (B, S, H, W, 1) 
    depth_conf = predictions["depth_conf"]  # (B, S, H, W)
    extrinsics_cam = predictions["extrinsic"]  # (B, S, 3, 4)
    intrinsics_cam = predictions["intrinsic"]  # (B, S, 3, 3)
    masks = predictions["masks"]  # (B, S, H, W)

    if recon_intrinsic is not None and recon_extrinsic is not None:
        extrinsics_cam = recon_extrinsic.cpu().numpy()
        intrinsics_cam = recon_intrinsic.cpu().numpy()

    bsz, s, h, w, _ = world_points_map.shape

    points_flat_list = []
    colors_flat_list = []
    conf_flat_list = []
    original_coords_list = []

    
    for i in range(bsz):
        if not use_point_map:
            world_points = unproject_depth_map_to_point_map(depth_map[i], extrinsics_cam[i], intrinsics_cam[i])
            conf = depth_conf[i]
        else:
            world_points = world_points_map[i]
            conf = conf_map[i]

        # Create coordinates with correct ordering: (b, x, y) where b is batch, y is vertical and x is horizontal
        b_indices = np.arange(world_points.shape[0])
        y_indices = np.arange(world_points.shape[1])
        x_indices = np.arange(world_points.shape[2])
        b_grid, y_grid, x_grid = np.meshgrid(b_indices, y_indices, x_indices, indexing='ij')
        coords = np.stack([b_grid, x_grid, y_grid], axis=-1) 

        points_flat = world_points.reshape(-1, 3)
        colors_flat = (images[i].transpose(0, 2, 3, 1).reshape(-1, 3) * 255).astype(np.uint8)
        conf_flat = conf.reshape(-1)
        coords_flat = coords.reshape(-1, 3)

        mask = masks[i]
        
        if apply_mask:
            points_flat, colors_flat, conf_flat, coords_flat = get_points_from_mask(points_flat, colors_flat, conf_flat, coords_flat, mask)

        if points_flat.shape[0] > max_points:
            # get the top max_points points
            sorted_indices = np.argsort(conf_flat)[::-1]
            sorted_indices = sorted_indices[:max_points]
            points_flat = points_flat[sorted_indices]
            colors_flat = colors_flat[sorted_indices]
            conf_flat = conf_flat[sorted_indices]
            coords_flat = coords_flat[sorted_indices]

        # Filter points based on confidence threshold
        # Check if conf_flat is empty before calculating percentile
        if conf_flat.size > 0:
            threshold_val = np.percentile(conf_flat, conf_threshold)
            threshold_val = max(threshold_val, conf_threshold_value)
            conf_mask = conf_flat > threshold_val
            points_flat = points_flat[conf_mask]
            colors_flat = colors_flat[conf_mask]
            conf_flat = conf_flat[conf_mask]
            coords_flat = coords_flat[conf_mask]

        # Optionally check the first point for debugging
        # visualize_i_view_with_keypoints(coords_flat, images[i], mask, 0)

        original_coords_list.append(coords_flat)
        points_flat_list.append(points_flat)
        colors_flat_list.append(colors_flat)
        conf_flat_list.append(conf_flat)

    return points_flat_list, colors_flat_list, conf_flat_list, original_coords_list


def get_depth_from_mask(depth_map, mask):
    """
    Args:
    depth_map: (B, H, W, 1), numpy array
    mask: (B, H, W), numpy array

    Returns:
        depth_fill: (B, H, W, 1), numpy array
    """

    depth_fill = np.zeros(depth_map.shape)
    depth_fill_flat = depth_fill.reshape(-1, 1)

    mask_flat_bool = mask.reshape(-1).astype(bool)

    depth_flat = depth_map.reshape(-1, 1)[mask_flat_bool]
    
    if len(depth_flat) == 0 or (depth_flat.max() - depth_flat.min()) < 1e-4:
        return depth_map

    depth_flat_norm = (depth_flat - depth_flat.min()) / (depth_flat.max() - depth_flat.min())

    depth_fill_flat[mask_flat_bool] = depth_flat_norm

    depth_fill = depth_fill_flat.reshape(depth_map.shape)

    return depth_fill


def get_depth_fill_from_predictions(predictions,  max_points: int = 1000000, conf_threshold: float = 20.0, conf_threshold_value: float = 2.0, apply_mask: bool = False):
    """"
    This function is used to get the depth fill from the predictions
    Due to the different methods of point selection, it is handled separately from get_points_from_predictions.

    apply_mask will be forbidden in this function

    Args:
    predictions: dict
    max_points: int, max points to sample
    conf_threshold: float, confidence threshold
    apply_mask: bool, if True, apply mask to the points

    Returns:
        depth_fill: (B, S, H, W, 1), numpy array
    """
    depths_map = predictions["depth"]  # (B, S, H, W, 1) 
    depths_conf = predictions["depth_conf"]  # (B, S, H, W)
    masks = predictions["masks"]  # (B, S, H, W)

    bsz, s, h, w, _ = depths_map.shape

    depth_fill_list = []
    for i in range(bsz):
        mask = masks[i]
        depth_map = depths_map[i]
        depth_conf = depths_conf[i]


        if apply_mask:
            depth_map = get_depth_from_mask(depth_map, mask)
            depth_conf = get_depth_from_mask(depth_conf, mask)
        
        depth_conf_flat = depth_conf.reshape(-1)
        depth_fill = np.zeros(depth_map.shape)
        depth_fill_flat = depth_fill.reshape(-1, 1)
        depth_flat = depth_map.reshape(-1, 1)

        if depth_flat.shape[0] > max_points:
            # get the top max_points points
            sorted_indices = np.argsort(depth_conf_flat)[::-1]
            sorted_indices = sorted_indices[:max_points]
            depth_flat = depth_flat[sorted_indices]
            depth_conf_flat = depth_conf_flat[sorted_indices]

        # Filter points based on confidence threshold
        # Check if depth_conf_flat is empty before calculating percentile
        if depth_conf_flat.size > 0:
            threshold_val = np.percentile(depth_conf_flat, conf_threshold)
            threshold_val = max(threshold_val, conf_threshold_value)
            conf_mask = depth_conf_flat > threshold_val
            depth_flat = depth_flat[conf_mask]
        
        # Handle the case where depth_flat is empty (zero-size array)
        if depth_flat.size > 0:
            depth_flat_norm = (depth_flat - depth_flat.min()) / (depth_flat.max() - depth_flat.min())
            conf_mask_sorted_indices = sorted_indices[conf_mask]
            depth_fill_flat[conf_mask_sorted_indices] = depth_flat_norm
            depth_fill = depth_fill_flat.reshape(depth_map.shape)
        else:
            depth_fill = depth_map

        depth_fill_list.append(depth_fill)

    depth_fill =  np.stack(depth_fill_list, axis=0)

    return depth_fill


def np_depth_to_colormap(depth, min_conf=-0.9):
    """ 
    Args:
    depth: [B, N, H, W, 1] or [B, N, H, W]  or [B, H, W]
    
    Returns:
    depth_color: [B, N, H, W, 3] or [B, H, W, 3]
    depth_normalized: [B, N, H, W, 1] or [B, H, W, 1]
    """

    if depth.ndim == 5 and depth.shape[-1] == 1:
        depth = depth.squeeze(-1)

    if depth.ndim == 4:
        dpt_ndim = 4
        b, n, h, w = depth.shape
        depth = depth.reshape(-1, h, w)
    elif depth.ndim == 3:
        dpt_ndim = 3
        b, h, w = depth.shape
        depth = depth.reshape(-1, h, w)
    else:
        raise ValueError(f"Depth dimension is not supported: {depth.ndim}")

    depth_colors = []
    depth_normalized = []

    for dpt in depth:
        dpt_normalized = np.zeros(dpt.shape)
        valid_mask_dp = dpt > min_conf # valid

        if valid_mask_dp.sum() > 0:
            d_valid = dpt[valid_mask_dp]
            min_val = d_valid.min()
            max_val = d_valid.max()
            if max_val > min_val:  # Avoid division by zero
                dpt_normalized[valid_mask_dp] = (d_valid - min_val) / (max_val - min_val)
            else:
                # If all values are the same, set normalized value to 0.5
                dpt_normalized[valid_mask_dp] = 0.5

            dpt_np = (dpt_normalized * 255).astype(np.uint8)
            dpt_color = cv2.applyColorMap(dpt_np, cv2.COLORMAP_JET)
            dpt_color = cv2.cvtColor(dpt_color, cv2.COLOR_BGR2RGB)
            depth_colors.append(dpt_color)
            depth_normalized.append(dpt_normalized)
        else:
            print('!!!! No depth projected !!!')
            dpt_color = np.zeros((dpt.shape[0], dpt.shape[1], 3), dtype=np.uint8)
            dpt_normalized = np.zeros(dpt.shape, dtype=np.float32)
            depth_colors.append(dpt_color)
            depth_normalized.append(dpt_normalized)

    depth_colors = np.stack(depth_colors, axis=0)
    depth_normalized = np.stack(depth_normalized, axis=0)

    if dpt_ndim == 4:
        depth_colors = depth_colors.reshape(b, n, h, w, 3)
        depth_normalized = depth_normalized.reshape(b, n, h, w, 1)
    elif dpt_ndim == 3:
        depth_colors = depth_colors.reshape(b, h, w, 3)
        depth_normalized = depth_normalized.reshape(b, h, w, 1)

    return depth_colors, depth_normalized


def get_mask_from_fragments(fragments, fill_holes=True, use_gaussian_blur=False, use_binary_mask=False):
    """
    Generate a mask from point cloud rendering fragments.
    
    This function creates a mask by analyzing the distance of rendered points to each pixel.
    The main idea is to:
    1. Use the squared distances from fragments to identify pixels with nearby points
    2. Create a smooth mask where closer points have higher values
    3. Optionally fill holes to create a connected region
    4. Optionally apply Gaussian blur for smoother edges
    
    Args:
        fragments: idx, zbuf, dists, see pytorch3d.renderer.points.rasterizer.PointFragments
        fill_holes: Whether to fill holes in the mask using contour detection
        use_gaussian_blur: Whether to apply Gaussian blur to the mask
        use_binary_mask: Whether to return a binary mask instead of a smooth one

    Returns:
        output_mask: (B, H, W) mask where values range from 0 to 1
    """
    # use dists2 (distance squared) of fragments to generate smooth mask
    render_idx = fragments.idx.cpu().numpy()
    dists2 = fragments.dists.cpu().numpy()  # (B, H, W, K)

    # calculate the minimum distance for each pixel (ignore invalid points)
    valid_mask = (render_idx != -1)
    dists2_valid = np.where(valid_mask, dists2, np.inf)
    min_dists2 = np.min(dists2_valid, axis=-1)  # (B, H, W)

    # normalize distance to [0, 1], the larger the distance, the smaller the mask
    # only normalize valid pixels
    min_dists2_valid = min_dists2[valid_mask[..., 0]]
    if min_dists2_valid.size > 0:
        d_min, d_max = min_dists2_valid.min(), min_dists2_valid.max()
        # avoid division by zero
        d_range = d_max - d_min if d_max > d_min else 1.0
        min_dists2_norm = (min_dists2 - d_min) / d_range
        min_dists2_norm = 1.0 - min_dists2_norm  # the smaller the mask is
        min_dists2_norm = np.clip(min_dists2_norm, 0, 1)
    else:
        min_dists2_norm = np.zeros_like(min_dists2)

    # only keep valid pixels
    smooth_mask = np.where(valid_mask[..., 0], min_dists2_norm, 0)

    output_binary_mask = smooth_mask.copy()

    # Optionally fill holes to ensure the mask is closed and connected
    if fill_holes:
        for i in range(smooth_mask.shape[0]):
            # Binarize the mask with a small threshold to get the main region
            binary_mask = (smooth_mask[i] > 0.05).astype(np.uint8)
            # Find contours
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # Create a new mask and fill the largest contour
            filled = np.zeros_like(binary_mask)
            if contours:
                # Find the largest contour (by area)
                largest_contour = max(contours, key=cv2.contourArea)
                cv2.drawContours(filled, [largest_contour], -1, 1, thickness=cv2.FILLED)
            # Optionally, fill all contours (uncomment if you want all regions)
            # for cnt in contours:
            #     cv2.drawContours(filled, [cnt], -1, 1, thickness=cv2.FILLED)

            # Multiply the original smooth mask by the filled mask to keep smoothness inside the closed region
            smooth_mask[i] = smooth_mask[i] * filled
            output_binary_mask[i] = filled

    if use_gaussian_blur:
        for i in range(smooth_mask.shape[0]):
            smooth_mask[i] = cv2.GaussianBlur(smooth_mask[i], (7, 7), 0)

    # optional: output binary mask
    if use_binary_mask:
        return output_binary_mask

    return smooth_mask

