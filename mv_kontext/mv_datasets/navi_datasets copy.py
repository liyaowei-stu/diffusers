import os,sys
import pandas as pd
import numpy as np
import cv2
import json
import random

from PIL import Image
from einops import rearrange, repeat

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

sys.path.append(os.getcwd())
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map, closed_form_inverse_se3

from mv_kontext.utils import convert_camera_pose_to_relative


class MVImageDataset(Dataset):
        
    def __init__(
        self, 
        meta_path, 
        data_dir, 
        min_recon_num=1, 
        max_recon_num=8,
        buckets=None,
    ):
        """
        For efficiency in forward VGGT predictions, currently we only support the output has the same number and resolution of recon images in the same batch.

        min_recon_num: the minimum number of recon images in a batch
        max_recon_num: the maximum number of recon images in a batch
        """
        self.meta_path = meta_path
        self.data_dir = data_dir
        self.min_recon_num = min_recon_num
        self.max_recon_num = max_recon_num
        self.buckets = buckets

        with open(meta_path, 'r') as f:
            self.data = json.load(f)

        self.reorganize_data()
        self.current_recon_num = None
        
    def reorganize_data(self):
        multi_view_img_paths = []
        video_paths = []
        wild_set_paths = []
        for item in self.data:
            multiview_img_path = item["multiview_img_path"]
            video_path = item["video_path"]
            wild_set_path = item["wild_set_path"]
            multi_view_img_paths.extend(multiview_img_path)
            video_paths.extend(video_path)
            wild_set_paths.extend(wild_set_path)

        self.multi_view_img_paths = list(set(multi_view_img_paths))
        self.video_paths = list(set(video_paths))
        self.multi_view_paths = self.multi_view_img_paths + self.video_paths
        self.wild_set_paths = list(set(wild_set_paths))
        
    def __len__(self):
        return len(self.multi_view_paths)
    
    def set_recon_num(self, recon_num=None):
        """Set a fixed recon_num for the entire batch"""
        if recon_num is None:
            self.current_recon_num = random.randint(self.min_recon_num, self.max_recon_num)
        else:
            self.current_recon_num = recon_num
        return self.current_recon_num

    def __getitem__(self, idx):
        while True:
            try:
                multi_view_path = self.multi_view_paths[idx]

                vggt_preds_path = os.path.join(self.data_dir, multi_view_path, "vggt_predictions.npz")
                if not os.path.exists(vggt_preds_path):
                    print(f"!!!! {vggt_preds_path} not found !!!!")
                    continue

                predictions = np.load(vggt_preds_path)

                ## relative pose: render related to recon

                ## recon info 
                gt_images, gt_masks = predictions["images"], predictions["masks"]
                gt_depth, gt_depth_conf = predictions["depth"], predictions["depth_conf"]
                gt_world_points, gt_world_points_conf = predictions["world_points"], predictions["world_points_conf"]

                # Use the batch-level recon_num if set, otherwise use random
                if self.current_recon_num is not None:
                    recon_num = self.current_recon_num
                else:
                    recon_num = random.randint(self.min_recon_num, self.max_recon_num)
                
                # Handle case where we don't have enough images
                if len(gt_images) < recon_num:
                    # Get all available indices
                    available_indices = list(range(len(gt_images)))
                    # Repeat indices as needed to reach recon_num
                    sample_recon_idx = available_indices + [random.choice(available_indices) for _ in range(recon_num - len(available_indices))]
                else:
                    # Normal case: randomly sample recon_num indices
                    sample_recon_idx = random.sample(list(range(len(gt_images))), recon_num)
                
                sample_recon_images = torch.from_numpy(gt_images[sample_recon_idx])
                sample_recon_masks = torch.from_numpy(gt_masks[sample_recon_idx])
                sample_recon_depth = torch.from_numpy(gt_depth[sample_recon_idx])
                sample_recon_depth_conf = torch.from_numpy(gt_depth_conf[sample_recon_idx])
                sample_recon_world_points = torch.from_numpy(gt_world_points[sample_recon_idx])
                sample_recon_world_points_conf = torch.from_numpy(gt_world_points_conf[sample_recon_idx])


                ## render info 
                sample_render_idx = random.randint(0, len(predictions["images"]) - 1)
                # sample_render_idx = sample_recon_idx[1]

                sample_render_image = torch.from_numpy(gt_images[sample_render_idx])
                sample_render_mask = torch.from_numpy(gt_masks[sample_render_idx])


                ## scale pose enc to b,s,9
                sample_render_pose_enc = predictions["pose_enc"][sample_render_idx][None][None]
                sample_render_extrinsic, sample_render_intrinsic = pose_encoding_to_extri_intri(torch.from_numpy(sample_render_pose_enc), predictions["images"].shape[-2:])
                sample_render_extrinsic, sample_render_intrinsic = sample_render_extrinsic.squeeze(0).squeeze(0), sample_render_intrinsic.squeeze(0).squeeze(0)
                

                ## convert to relative pose
                sample_recon_pose_enc = predictions["pose_enc"][sample_recon_idx][None]
                sample_recon_extrinsic, sample_recon_intrinsic = pose_encoding_to_extri_intri(torch.from_numpy(sample_recon_pose_enc), predictions["images"].shape[-2:])
                sample_recon_extrinsic = sample_recon_extrinsic.squeeze(0)
                sample_recon_intrinsic = sample_recon_intrinsic.squeeze(0)


                anchor_extrinsic = sample_recon_extrinsic[0].squeeze(0)

                sample_recon_extrinsic = convert_camera_pose_to_relative(sample_recon_extrinsic, anchor_extrinsic)

                sample_render_extrinsic = convert_camera_pose_to_relative(sample_render_extrinsic, anchor_extrinsic)

                sample_recon_idx = torch.from_numpy(np.array(sample_recon_idx))
                sample_render_idx = torch.from_numpy(np.array(sample_render_idx))

                
                return sample_render_intrinsic, sample_render_extrinsic, sample_render_image, sample_render_mask, sample_render_idx, sample_recon_intrinsic, sample_recon_extrinsic, sample_recon_images, sample_recon_masks, sample_recon_idx, sample_recon_depth, sample_recon_depth_conf, sample_recon_world_points, sample_recon_world_points_conf, multi_view_path
                    
            except Exception as e:
                import ipdb; ipdb.set_trace()
                print("data load exception", e)
                print(e)


def collate_fn(batch):
    sample_render_intrinsics, sample_render_extrinsics, sample_render_images, sample_render_masks, sample_render_idxs, sample_recon_intrinsics, sample_recon_extrinsics, sample_recon_images, sample_recon_masks, sample_recon_idxs, sample_recon_depths, sample_recon_depth_confs, sample_recon_world_points, sample_recon_world_points_confs, multi_view_paths = zip(*batch)

    sample_render_intrinsics = torch.stack(sample_render_intrinsics, dim=0)
    sample_render_extrinsics = torch.stack(sample_render_extrinsics, dim=0)
    sample_render_images = torch.stack(sample_render_images, dim=0)
    sample_render_masks = torch.stack(sample_render_masks, dim=0)
    sample_render_idxs = torch.stack(sample_render_idxs, dim=0)

    sample_recon_intrinsics = torch.stack(sample_recon_intrinsics, dim=0)
    sample_recon_extrinsics = torch.stack(sample_recon_extrinsics, dim=0)
    sample_recon_images = torch.stack(sample_recon_images, dim=0)
    sample_recon_masks = torch.stack(sample_recon_masks, dim=0)
    sample_recon_idxs = torch.stack(sample_recon_idxs, dim=0)
    sample_recon_depths = torch.stack(sample_recon_depths, dim=0)
    sample_recon_depth_confs = torch.stack(sample_recon_depth_confs, dim=0)
    sample_recon_world_points = torch.stack(sample_recon_world_points, dim=0)
    sample_recon_world_points_confs = torch.stack(sample_recon_world_points_confs, dim=0)

    return_dict = {
        "sample_render_intrinsics": sample_render_intrinsics,
        "sample_render_extrinsics": sample_render_extrinsics,
        "sample_render_images": sample_render_images,
        "sample_render_masks": sample_render_masks,
        "sample_render_idxs": sample_render_idxs,
        "sample_recon_intrinsics": sample_recon_intrinsics,
        "sample_recon_extrinsics": sample_recon_extrinsics,
        "sample_recon_images": sample_recon_images,
        "sample_recon_masks": sample_recon_masks,
        "sample_recon_idxs": sample_recon_idxs,
        "sample_recon_depths": sample_recon_depths,
        "sample_recon_depth_confs": sample_recon_depth_confs,
        "sample_recon_world_points": sample_recon_world_points,
        "sample_recon_world_points_confs": sample_recon_world_points_confs,
        "multi_view_paths": multi_view_paths
    }

    return return_dict


class BatchSampler(torch.utils.data.Sampler):
    """Sampler that ensures each batch uses the same recon_num"""
    def __init__(self, dataset, batch_size, shuffle):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = len(dataset)
        
    def __iter__(self):
        indices = list(range(self.num_samples))
        if self.shuffle:
            random.shuffle(indices)
        
        # Set a fixed recon_num for the entire epoch
        fixed_recon_num = random.randint(self.dataset.min_recon_num, self.dataset.max_recon_num)
        self.dataset.set_recon_num(fixed_recon_num)
        
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            if len(batch_indices) == self.batch_size:  # Only yield complete batches
                yield batch_indices
                
    def __len__(self):
        return self.num_samples // self.batch_size


def loader(train_batch_size, num_workers, shuffle=False, **args):
    dataset = MVImageDataset(**args)
    batch_sampler = BatchSampler(dataset, train_batch_size, shuffle=shuffle)
    return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=collate_fn)

if __name__ == '__main__':

    import ipdb; ipdb.set_trace()

    train_batch_size = 2
    dataloader = loader(train_batch_size=train_batch_size, num_workers=0, meta_path='data/navi/metainfo/navi_v1.5_metainfo.json', data_dir='data/navi/navi_v1.5_vggt')
    print("num samples", len(dataloader)*train_batch_size)

    for i, data in enumerate(dataloader):
        sample_render_intrinsic, sample_render_extrinsic, sample_render_images, sample_render_masks, sample_render_idxs, sample_recon_intrinsics, sample_recon_extrinsics, sample_recon_images, sample_recon_masks, sample_recon_idxs, sample_recon_depths, sample_recon_depth_confs, sample_recon_world_points, sample_recon_world_points_confs, multi_view_paths = data
        print("sample_render_extrinsic.shape", sample_render_extrinsic.shape)
        print("sample_render_intrinsic.shape", sample_render_intrinsic.shape)
        print("sample_render_images.shape", sample_render_images.shape)
        print("sample_render_masks.shape", sample_render_masks.shape)
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
        
        if i == 6:
            break 
