import os, sys
import glob
import json
import argparse
import datetime
import random
import shutil

import numpy as np
from einops import rearrange
from PIL import Image
import trimesh
import cv2
from tqdm import tqdm

import torch
from torchvision.utils import save_image

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import closed_form_inverse_se3


from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PerspectiveCameras
)


sys.path.append(os.getcwd())
from mv_kontext.mv_datasets.navi_datasets import loader as navi_loader
from mv_kontext.utils import define_rasterizer_renderer, get_points_from_predictions, get_depth_fill_from_predictions, np_depth_to_colormap, convert_opencv_to_pytorch3d_c2w, img_tensor_to_pil
from mv_kontext.pipeline.render_func import get_recon_points_and_depth, recenter_points



def run_vggt_on_images(images, model):
    """
    images: [B, S, 3, H, W], tensor
    """
    with torch.cuda.amp.autocast(dtype=dtype):

        predictions = model(images)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy()

    return predictions

        

@torch.no_grad()
def main(
    args, 
    ):

    import ipdb; ipdb.set_trace()

    data_loader = navi_loader(train_batch_size=args.batch_size, num_workers=0, meta_path=args.meta_path, data_dir=args.data_dir, shuffle=True, min_recon_num=8, max_recon_num=8)

    for i, data in enumerate(tqdm(data_loader)):

        sample_render_extrinsic = data["sample_render_extrinsics"]
        sample_render_intrinsic = data["sample_render_intrinsics"]
        sample_render_gt_images = data["sample_render_gt_images"]
        sample_render_gt_masks = data["sample_render_gt_masks"]
        sample_render_idxs = data["sample_render_idxs"]
        sample_render_ori_gt_images = data["sample_render_ori_gt_images"]
        sample_render_instructions = data["sample_render_instructions"]

        sample_recon_intrinsics = data["sample_recon_intrinsics"]
        sample_recon_extrinsics = data["sample_recon_extrinsics"]
        sample_recon_images = data["sample_recon_images"]
        sample_recon_masks = data["sample_recon_masks"]
        sample_recon_idxs = data["sample_recon_idxs"]
        sample_recon_depths = data["sample_recon_depths"]
        sample_recon_depth_confs = data["sample_recon_depth_confs"]
        sample_recon_world_points = data["sample_recon_world_points"]
        sample_recon_world_points_confs = data["sample_recon_world_points_confs"]
        sample_multi_view_paths = data["sample_multi_view_paths"]


        bsz, s, _, h, w = sample_recon_images.shape


        ## recon and filter points
        points_flat, colors_flat, conf_flat, depth_fill, _ = get_recon_points_and_depth(sample_recon_images, sample_recon_extrinsics, sample_recon_intrinsics, sample_recon_masks, sample_recon_depths, sample_recon_depth_confs, sample_recon_world_points, sample_recon_world_points_confs, args.use_point_map, args.max_points, args.conf_threshold, args.conf_threshold_value, args.apply_mask)
        

        depth_color, depth_normalized = np_depth_to_colormap(depth_fill)
     
        # squeeze the num_views dimension
        if len(sample_render_extrinsic.shape) == 4:
            sample_render_extrinsic = sample_render_extrinsic.squeeze(1)
        if len(sample_render_intrinsic.shape) == 4:
            sample_render_intrinsic = sample_render_intrinsic.squeeze(1)


        # Save point cloud for fast visualization
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # trimesh.PointCloud(points_flat[0], colors=colors_flat[0]).export(f"{timestamp}_points_max_points_{args.max_points}_conf_{args.conf_threshold}_use_point_map_{args.use_point_map}_apply_mask_{args.apply_mask}.ply")

        # move the scene to the center
        points_flat_centered, scene_centers = recenter_points(points_flat)

        
        sample_render_c2w = closed_form_inverse_se3(sample_render_extrinsic)
        sample_render_c2w = sample_render_c2w[:, :3, :]
        # sample_render_c2w = sample_render_c2w[:, :3, :].cpu().numpy()
        scene_centers = torch.from_numpy(scene_centers).float()

        sample_render_c2w[..., -1] -= scene_centers
    
        ## coordinate system conversion
        sample_render_c2w = convert_opencv_to_pytorch3d_c2w(sample_render_c2w)
        sample_render_w2c = closed_form_inverse_se3(sample_render_c2w)
        sample_render_w2c_R, sample_render_w2c_T = sample_render_w2c[:, :3, :3], sample_render_w2c[:, :3, 3]

        # pytorch3d is row major, so we need to transpose the R
        # sample_render_w2c_R = torch.from_numpy(sample_render_w2c_R.transpose(0, 2, 1)).float()
        # sample_render_w2c_T = torch.from_numpy(sample_render_w2c_T).float()
        sample_render_w2c_R = sample_render_w2c_R.permute(0, 2, 1).float()
        sample_render_w2c_T = sample_render_w2c_T.float()


        fx, fy = sample_render_intrinsic[:, 0, 0], sample_render_intrinsic[:, 1, 1]
        ux, uy = sample_render_intrinsic[:, 0, 2], sample_render_intrinsic[:, 1, 2]

       
        # image_size = [ [sample_recon_images.shape[-2], sample_recon_images.shape[-1]] for _ in range(bsz)]
        image_size = [ [sample_render_ori_gt_images.shape[-2], sample_render_ori_gt_images.shape[-1]] for _ in range(bsz)]

        fcl_screen = torch.stack((fx, fy), dim=-1)
        prp_screen = torch.stack((ux, uy), dim=-1)

        resize_ratio = max(image_size[0]) / 518
        fcl_screen = fcl_screen * resize_ratio
        real_pp = [[image_size[0][1] / 2, image_size[0][0] / 2] for _ in range(len(image_size))] # wh
        prp_screen = torch.Tensor(real_pp).to(prp_screen)


        # ## squeeze the num_views dimension
        cameras = PerspectiveCameras(device=device, R=sample_render_w2c_R, T=sample_render_w2c_T, focal_length=fcl_screen, principal_point=prp_screen, image_size=image_size, in_ndc=False)

        rasterizer, renderer = define_rasterizer_renderer(cameras, image_size=image_size[0], radius=args.radius, points_per_pixel=args.points_per_pixel, bin_size=args.bin_size)

        points_flat_centered = [torch.from_numpy(points_flat_centered[i]).float().to(device) for i in range(bsz)]
        colors_flat = [torch.from_numpy(colors_flat[i]).float().to(device) for i in range(bsz)]
        point_cloud = Pointclouds(points=points_flat_centered, features=colors_flat)

        new_images, fragments = renderer(point_cloud)

        new_depths = fragments.zbuf[:, :, :, 0].cpu().numpy()
        new_depth_color, new_depth_normalized = np_depth_to_colormap(new_depths)

        ## save info
        for i, new_image in enumerate(new_images):
            
            save_dir = f"{args.output_dir}/conf_{args.conf_threshold}_{len(sample_recon_images[0])}_recon_num/{sample_multi_view_paths[i]}"

            if os.path.exists(f"{save_dir}/renders"):
                shutil.rmtree(f"{save_dir}/renders")
            os.makedirs(f"{save_dir}/renders", exist_ok=True)

            ## save renders
            new_image = new_image.cpu().numpy()
            new_image = new_image.astype(np.uint8)
            new_image = Image.fromarray(new_image)
            save_path = f"{save_dir}/renders/rasterized/sample_idx_{sample_render_idxs[i].item()}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            new_image.save(save_path)

            save_path = f"{save_dir}/renders/gt/sample_idx_{sample_render_idxs[i].item()}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            save_image(sample_render_ori_gt_images[i], save_path)

            save_path = f"{save_dir}/renders/rasterized_depth_color/sample_idx_{sample_render_idxs[i].item()}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, new_depth_color[i])
            save_path = f"{save_dir}/renders/rasterized_depth/sample_idx_{sample_render_idxs[i].item()}.png"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, (new_depth_normalized[i] * 255).astype(np.uint8))

            ## save recon
            sample_recon_idx = sample_recon_idxs[i]
            sample_recon_image = sample_recon_images[i]
            dpt_color = depth_color[i]
            dpt_normalized = depth_normalized[i]

            ## remove the recon folder if it exists
            if os.path.exists(f"{save_dir}/recon"):
                shutil.rmtree(f"{save_dir}/recon")

            for j in range(len(sample_recon_image)):
                save_path = f"{save_dir}/recon/images/sample_idx_{sample_recon_idx[j].item()}.png"
                
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                save_image(sample_recon_image[j], save_path)

                save_path = f"{save_dir}/recon/depth_color/sample_idx_{sample_recon_idx[j].item()}.png"
                
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, dpt_color[j])

                save_path = f"{save_dir}/recon/depth/sample_idx_{sample_recon_idx[j].item()}.png"
    
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                cv2.imwrite(save_path, (dpt_normalized[j] * 255).astype(np.uint8))

            save_path = f"{save_dir}/recon/points/points.ply"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            trimesh.PointCloud(points_flat_centered[i].cpu().numpy(), colors=(colors_flat[i].cpu().numpy()).astype(np.uint8)).export(save_path)

            # print("="*100)

    print("Process done!")



def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--use_point_map", type=bool, default=False)
    parser.add_argument("--max_points", type=int, default=1000000)
    parser.add_argument("--conf_threshold", type=float, default=20.0)
    parser.add_argument("--conf_threshold_value", type=float, default=2.0)
    parser.add_argument("--radius", type=float, default=0.003)
    parser.add_argument("--points_per_pixel", type=int, default=20)
    parser.add_argument("--bin_size", type=int, default=None)
    parser.add_argument("--apply_mask", action="store_true")
    parser.add_argument("--batch_size", type=int, default=1)
    return parser.parse_args()



if __name__ == "__main__":
    args = parser_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    args.device = device
    args.dtype = dtype

    main(args)
