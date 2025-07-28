import os,sys
import pandas as pd
import numpy as np
import cv2
import json
import random
from datetime import datetime

from PIL import Image
from einops import rearrange, repeat

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

sys.path.append(os.getcwd())
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map, closed_form_inverse_se3

from mv_kontext.utils import convert_camera_pose_to_relative, find_nearest_bucket, paired_transform, img_tensor_to_pil
from mv_kontext.utils.data_handlers import convert_tensor_to_img_grid, img_tensor_to_pil


class MVImageDataset(Dataset):
        
    def __init__(
        self, 
        meta_path, 
        data_dir
    ):
        """
        For efficiency in forward VGGT predictions, currently we only support the output has the same number and resolution of recon images in the same batch.

        min_recon_num: the minimum number of recon images in a batch
        max_recon_num: the maximum number of recon images in a batch
        """
        self.meta_path = meta_path
        self.data_dir = data_dir

        with open(meta_path, 'r') as f:
            self.multi_view_data = json.load(f)

        
    def __len__(self):
        return len(self.multi_view_data)
    

    def __getitem__(self, idx):
        while True:
            try:
                obj_id = self.multi_view_data[idx]["obj_id"]
                camera_id = self.multi_view_data[idx]["camera_id"]
                

                sample_render_extrinsic = np.array(self.multi_view_data[idx]["sample_render_extrinsic"])
                sample_render_intrinsic = np.array(self.multi_view_data[idx]["sample_render_intrinsic"])
                sample_render_idxs = self.multi_view_data[idx]["sample_render_idxs"]
                sample_render_instructions = self.multi_view_data[idx]["sample_render_instructions"]

                sample_recon_intrinsics = np.array(self.multi_view_data[idx]["sample_recon_intrinsics"])
                sample_recon_extrinsics = np.array(self.multi_view_data[idx]["sample_recon_extrinsics"])
                sample_recon_idxs = self.multi_view_data[idx]["sample_recon_idxs"]

                sample_multi_view_paths = self.multi_view_data[idx]["sample_multi_view_paths"]

                data_path = os.path.join(self.data_dir, f"test_{obj_id}_{camera_id}.npy")
                data = np.load(data_path, allow_pickle=True).item()
                
                sample_render_gt_images = data["sample_render_gt_images"]
                sample_render_gt_masks = data["sample_render_gt_masks"]
                sample_render_ori_gt_images = data["sample_render_ori_gt_images"]
                
                sample_recon_images = data["sample_recon_images"]
                sample_recon_masks = data["sample_recon_masks"]
                sample_recon_depths = data["sample_recon_depths"]
                sample_recon_depth_confs = data["sample_recon_depth_confs"]
                sample_recon_world_points = data["sample_recon_world_points"]
                sample_recon_world_points_confs = data["sample_recon_world_points_confs"]
                
                return_dict = {
                        "sample_render_intrinsics": torch.from_numpy(sample_render_intrinsic).float().squeeze(0),
                        "sample_render_extrinsics": torch.from_numpy(sample_render_extrinsic).float().squeeze(0),
                        "sample_render_gt_images": torch.from_numpy(sample_render_gt_images).float().squeeze(0),
                        "sample_render_gt_masks": torch.from_numpy(sample_render_gt_masks).float().squeeze(0),
                        "sample_render_instructions": sample_render_instructions[0],
                        "sample_render_ori_gt_images": torch.from_numpy(sample_render_ori_gt_images).float().squeeze(0),
                        "sample_render_idxs": torch.tensor(sample_render_idxs).float().squeeze(0),
                        "sample_recon_intrinsics": torch.from_numpy(sample_recon_intrinsics).float().squeeze(0),
                        "sample_recon_extrinsics": torch.from_numpy(sample_recon_extrinsics).float().squeeze(0),
                        "sample_recon_images": torch.from_numpy(sample_recon_images).float().squeeze(0),
                        "sample_recon_masks": torch.from_numpy(sample_recon_masks).float().squeeze(0),
                        "sample_recon_idxs": torch.tensor(sample_recon_idxs).float().squeeze(0),
                        "sample_recon_depths": torch.from_numpy(sample_recon_depths).float().squeeze(0),
                        "sample_recon_depth_confs": torch.from_numpy(sample_recon_depth_confs).float().squeeze(0),
                        "sample_recon_world_points": torch.from_numpy(sample_recon_world_points).float().squeeze(0),
                        "sample_recon_world_points_confs": torch.from_numpy(sample_recon_world_points_confs).float().squeeze(0),
                        "sample_multi_view_paths": sample_multi_view_paths[0]
                    }

                return return_dict

                    
            except Exception as e:
                import ipdb; ipdb.set_trace()
                print("data load exception", e)
                print(e)


def loader(train_batch_size, num_workers, shuffle=False, **args):
    dataset = MVImageDataset(**args)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=shuffle)

if __name__ == '__main__':

    import ipdb; ipdb.set_trace()

    train_batch_size = 1
    data_dir = 'data/navi/navi_test_data'
    meta_path = 'data/navi/navi_test_data/test_metainfo_20250714_223602.json'
    dataloader = loader(train_batch_size=train_batch_size, num_workers=0, meta_path=meta_path, data_dir=data_dir)
    
    for i, data in enumerate(dataloader):
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

        obj_id = sample_multi_view_paths[0].split("/")[-2]
        camera_id = sample_multi_view_paths[0].split("/")[-1]

        recon_img_grid = convert_tensor_to_img_grid(sample_recon_images[0], rescale=False, num_rows=2)
        recon_mask_grid = convert_tensor_to_img_grid(sample_recon_masks[0], rescale=False, num_rows=2)

        render_img_gt_grid = convert_tensor_to_img_grid(sample_render_ori_gt_images[0], rescale=True, num_rows=2)

        save_path = os.path.join(data_dir, "images", f"{obj_id}_{camera_id}", f"recon_img_grid.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        recon_img_grid.save(save_path)
        save_path = os.path.join(data_dir, "images", f"{obj_id}_{camera_id}", f"recon_mask_grid.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        recon_mask_grid.save(save_path)
        save_path = os.path.join(data_dir, "images", f"{obj_id}_{camera_id}", f"render_img_gt_grid.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        render_img_gt_grid.save(save_path)    

        print("sample_render_extrinsic.shape", sample_render_extrinsic.shape)
        print("sample_render_intrinsic.shape", sample_render_intrinsic.shape)
        print("sample_render_gt_images.shape", sample_render_gt_images.shape)
        print("sample_render_gt_masks.shape", sample_render_gt_masks.shape)
        print("sample_render_ori_gt_images.shape", sample_render_ori_gt_images.shape)
        print("sample_render_instructions.shape", len(sample_render_instructions))
        print("sample_render_idxs.shape", sample_render_idxs.shape)

        print("sample_recon_intrinsics.shape", sample_recon_intrinsics.shape)
        print("sample_recon_extrinsics.shape", sample_recon_extrinsics.shape)
        print("sample_recon_images.shape", sample_recon_images.shape)
        print("sample_recon_masks.shape", sample_recon_masks.shape)
        print("sample_recon_idxs.shape", sample_recon_idxs.shape)
        print("sample_recon_depths.shape", sample_recon_depths.shape)
        print("sample_recon_depth_confs.shape", sample_recon_depth_confs.shape)
        print("sample_recon_world_points.shape", sample_recon_world_points.shape)
        print("sample_recon_world_points_confs.shape", sample_recon_world_points_confs.shape)   
        print("="*100)

