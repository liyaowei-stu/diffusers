import torch
import numpy as np

from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.structures import Pointclouds

from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import closed_form_inverse_se3

from mv_kontext.utils import get_points_from_predictions, get_depth_fill_from_predictions, np_depth_to_colormap, convert_opencv_to_pytorch3d_c2w, define_rasterizer_renderer

def get_recon_points_and_depth(sample_recon_images, sample_recon_extrinsics, sample_recon_intrinsics, sample_recon_masks, sample_recon_depths, sample_recon_depth_confs, sample_recon_world_points, sample_recon_world_points_confs, use_point_map, max_points, conf_threshold, conf_threshold_value, apply_mask):
    pred_dict = {}
    pred_dict["images"] = sample_recon_images.cpu().numpy()
    pred_dict["extrinsic"] = sample_recon_extrinsics.cpu().numpy()
    pred_dict["intrinsic"] = sample_recon_intrinsics.cpu().numpy()
    pred_dict["masks"] = sample_recon_masks.cpu().numpy()
    pred_dict["depth"] = sample_recon_depths.cpu().numpy()
    pred_dict["depth_conf"] = sample_recon_depth_confs.cpu().numpy()
    pred_dict["world_points"] = sample_recon_world_points.cpu().numpy()
    pred_dict["world_points_conf"] = sample_recon_world_points_confs.cpu().numpy()

    points_flat, colors_flat, conf_flat, original_coords = get_points_from_predictions(pred_dict, use_point_map, max_points, conf_threshold, conf_threshold_value, apply_mask)

    depth_fill = get_depth_fill_from_predictions(pred_dict, max_points, conf_threshold, conf_threshold_value, apply_mask)

    return points_flat, colors_flat, conf_flat, depth_fill, original_coords


def recenter_points(points_flat):
    """
    Args:
    points_flat: [B, N, 3], numpy array

    Returns:
    points_flat_centered: [B, N, 3], numpy array
    scene_centers: [B, 3], numpy array
    """
    scene_centers = []
    points_flat_centered = []
    for points_flat in points_flat:
        scene_center = np.mean(points_flat, axis=0)
        points_flat_centered.append(points_flat - scene_center)
        scene_centers.append(scene_center)

    scene_centers = np.stack(scene_centers, axis=0)
    return points_flat_centered, scene_centers




def render_pipe(
    sample_recon_images,
    sample_recon_extrinsics, 
    sample_recon_intrinsics, 
    sample_recon_masks, 
    sample_recon_depths, 
    sample_recon_depth_confs, 
    sample_recon_world_points, 
    sample_recon_world_points_confs, 
    sample_render_extrinsic,
    sample_render_intrinsic,
    sample_render_ori_gt_images,
    use_point_map, 
    max_points, 
    conf_threshold,
    conf_threshold_value, 
    apply_mask, 
    radius, 
    points_per_pixel, 
    bin_size, 
    device,
    viz_depth=False,
    ):
    """
    Args:
    sample_recon_images: [B, S, 3, H, W], tensor
    sample_recon_extrinsics: [B, S, 3, 4], tensor
    sample_recon_intrinsics: [B, S, 3, 3], tensor
    sample_recon_masks: [B, S, 1, H, W], tensor
    sample_recon_depths: [B, S, 1, H, W], tensor
    sample_recon_depth_confs: [B, S, 1, H, W], tensor
    sample_recon_world_points: [B, S, N, 3], tensor

    Returns:
    novel_images: [B, S, 3, H, W], tensor
    novel_depths: [B, S, 1, H, W], tensor
    novel_depth_color: [B, S, 3, H, W], tensor
    novel_depth_normalized: [B, S, 1, H, W], tensor
    recon_depths: [B, S, 1, H, W], tensor
    recon_depth_color: [B, S, 3, H, W], tensor
    recon_depth_normalized: [B, S, 1, H, W], tensor
    """
    ## 1. unproject depth to point cloud and filter points
    points_flat, colors_flat, conf_flat, recon_depths = get_recon_points_and_depth(sample_recon_images, sample_recon_extrinsics, sample_recon_intrinsics, sample_recon_masks, sample_recon_depths, sample_recon_depth_confs, sample_recon_world_points, sample_recon_world_points_confs, use_point_map, max_points, conf_threshold, conf_threshold_value, apply_mask)

    
    ## 2. move the scene and camera to the center
    bsz, s, _, h, w = sample_recon_images.shape
    points_flat_centered, scene_centers = recenter_points(points_flat)
    
    sample_render_c2w = closed_form_inverse_se3(sample_render_extrinsic)
    sample_render_c2w = sample_render_c2w[:, :3, :].cpu().numpy()
    sample_render_c2w[..., -1] -= scene_centers

    ## 3. convert the opencv to pytorch3d
    sample_render_c2w = convert_opencv_to_pytorch3d_c2w(sample_render_c2w)
    sample_render_w2c = closed_form_inverse_se3(sample_render_c2w)
    sample_render_w2c_R, sample_render_w2c_T = sample_render_w2c[:, :3, :3], sample_render_w2c[:, :3, 3]

    # pytorch3d is row major, so we need to transpose the R
    sample_render_w2c_R = torch.from_numpy(sample_render_w2c_R.transpose(0, 2, 1)).float()
    sample_render_w2c_T = torch.from_numpy(sample_render_w2c_T).float()

    fx, fy = sample_render_intrinsic[:, 0, 0], sample_render_intrinsic[:, 1, 1]
    ux, uy = sample_render_intrinsic[:, 0, 2], sample_render_intrinsic[:, 1, 2]


    ## 4. define the camera, rasterizer and render
    image_size = [ [sample_render_ori_gt_images.shape[-2], sample_render_ori_gt_images.shape[-1]] for _ in range(bsz)]

    fcl_screen = torch.stack((fx, fy), dim=-1)
    prp_screen = torch.stack((ux, uy), dim=-1)

    resize_ratio = max(image_size[0]) / 518
    fcl_screen = fcl_screen * resize_ratio
    real_pp = [[image_size[0][1] / 2, image_size[0][0] / 2] for _ in range(len(image_size))] # wh
    prp_screen = torch.Tensor(real_pp).to(prp_screen)

    # ## squeeze the num_views dimension
    cameras = PerspectiveCameras(device=device, R=sample_render_w2c_R, T=sample_render_w2c_T, focal_length=fcl_screen, principal_point=prp_screen, image_size=image_size, in_ndc=False)

    rasterizer, renderer = define_rasterizer_renderer(cameras, image_size=image_size[0], radius=radius, points_per_pixel=points_per_pixel, bin_size=bin_size)

    ## 5. render the novel views
    points_flat_centered = [torch.from_numpy(points_flat_centered[i]).float().to(device) for i in range(bsz)]
    colors_flat = [torch.from_numpy(colors_flat[i]).float().to(device) for i in range(bsz)]
    point_cloud = Pointclouds(points=points_flat_centered, features=colors_flat)
    novel_images, fragments = renderer(point_cloud)
    novel_depths = fragments.zbuf[:, :, :, 0].cpu().numpy()


    if viz_depth:
        recon_depth_color, recon_depth_normalized = np_depth_to_colormap(recon_depths)
        novel_depth_color, novel_depth_normalized = np_depth_to_colormap(novel_depths)
        return novel_images, novel_depths, recon_depths, novel_depth_color, novel_depth_normalized, recon_depth_color, recon_depth_normalized
    else:
        return novel_images, novel_depths, recon_depths
