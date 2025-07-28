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


# wh
PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


def check_and_load_npz(path):
    if not os.path.exists(path):
        print(f"!!!! {path} not found !!!!")
        return None
    return np.load(path)


def check_and_load_image(path):
    if not os.path.exists(path):
        print(f"!!!! {path} not found !!!!")
        return None
    return Image.open(path).convert("RGB")


def get_sample_gt_idx(multi_view_data, sample_idx):
    """
    Args:
        multi_view_data: dict
        sample_idx: int
    """
     ## some multiviews have been sampled to ensure the total num < 200, so we need to get true sample_render_idx
    recon_sample_idx = multi_view_data["recon_sample_idx"]
    if recon_sample_idx == []:
        sample_gt_idx = sample_idx
    else:
        sample_gt_idx = recon_sample_idx[sample_idx]
    return sample_gt_idx


def resize_image_to_bucket(image, width, height, buckets):
    bucket_idx = find_nearest_bucket(height, width, buckets)
    target_height, target_width = buckets[bucket_idx]
    return paired_transform(image, size=(target_height, target_width))[0]


def sample_recon_indices(num_images, recon_num):
    if num_images < recon_num:
        available_indices = list(range(num_images))
        # Repeat indices as needed to reach recon_num
        sample_recon_idx = available_indices + [random.choice(available_indices) for _ in range(recon_num - num_images)]
    else:
        # Normal case: randomly sample recon_num indices
        # random.sample ensures unique indices without replacement
        sample_recon_idx = random.sample(list(range(num_images)), recon_num)
    return sample_recon_idx


def get_pose_encodings(predictions, indices, image_shape):
    pose_enc = predictions["pose_enc"][indices]
    if pose_enc.ndim == 1:
        pose_enc = pose_enc[None][None]
    elif pose_enc.ndim == 2:
        pose_enc = pose_enc[None]
    else:
        pass
    extrinsic, intrinsic = pose_encoding_to_extri_intri(torch.from_numpy(pose_enc), image_shape)
    return extrinsic.squeeze(0), intrinsic.squeeze(0)


def collect_gt_images(multi_view_data, sample_multi_view_path, gt_idxs, buckets):
    """
    Collect ground truth images given indices (single int or list of ints).
    Returns a single image if gt_idxs is int, or a stacked tensor if list.
    """
    if isinstance(gt_idxs, int):
        gt_idxs = [gt_idxs]
        single = True
    else:
        single = False

    images = []
    for idx in gt_idxs:
        item = multi_view_data["imgs"][idx]
        path = os.path.join(sample_multi_view_path, "images", item["img_name"])
        img = check_and_load_image(path)
        if img is None:
            raise ValueError(f"Image {path} not found")
        img = resize_image_to_bucket(img, *item["wh"], buckets)
        images.append(img)
    if single:
        return images[0]
    else:
        return torch.stack(images, dim=0)


def get_instruction(multi_view_data, idx):
    return multi_view_data["imgs"][idx]["gemini_flash_captions"]


def get_image_path(base_path, img_name):
    return os.path.join(base_path, "images", img_name)

def get_vggt_preds_path(base_path):
    return os.path.join(base_path, "vggt_predictions.npz")

def safe_load_or_retry(load_func, *args, **kwargs):
    result = load_func(*args, **kwargs)
    if result is None:
        raise RuntimeError(f"Failed to load with {load_func.__name__} and args {args}")
    return result

def torch_from_numpy_indices(array, indices):
    return torch.from_numpy(array[indices])

def get_pose_and_convert(predictions, indices, image_shape, anchor_extrinsic=None):
    extrinsic, intrinsic = get_pose_encodings(predictions, indices, image_shape)
    if anchor_extrinsic is not None:
        extrinsic = convert_camera_pose_to_relative(extrinsic, anchor_extrinsic)
    return extrinsic, intrinsic


class MVImageDataset(Dataset):
        
    def __init__(
        self, 
        meta_path, 
        data_dir, 
        min_recon_num=1, 
        max_recon_num=8,
        buckets=PREFERRED_KONTEXT_RESOLUTIONS,
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

        # w,h  -> h,w
        self.buckets = [(h, w) for w, h in buckets]

        with open(meta_path, 'r') as f:
            self.multi_view_data = json.load(f)

        self.current_recon_num = None
        
        # Pre-calculate bucket indices for each item
        self.bucket_indices = {}
        for idx in range(len(self.multi_view_data)):
            try:
                multi_view_data = self.multi_view_data[idx]
                sample_render_gt_item = multi_view_data["imgs"][0]  # Use the first image as reference
                width, height = sample_render_gt_item["wh"]
                bucket_idx = find_nearest_bucket(height, width, self.buckets)
                self.bucket_indices[idx] = bucket_idx
            except Exception as e:
                # If there's an error, assign a default bucket
                self.bucket_indices[idx] = 0
                print(f"Error calculating bucket for index {idx}: {e}")
        
    def __len__(self):
        return len(self.multi_view_data)
    
    def set_recon_num(self, recon_num=None):
        """Set a fixed recon_num for the entire batch"""
        if recon_num is None:
            self.current_recon_num = random.randint(self.min_recon_num, self.max_recon_num)
        else:
            self.current_recon_num = recon_num
        return self.current_recon_num
    
    def get_bucket_idx(self, idx):
        """Get the bucket index for a given dataset index"""
        return self.bucket_indices.get(idx, 0)

    def __getitem__(self, idx):
        while True:
            try:
                # 1. 路径拼接
                multi_view_data = self.multi_view_data[idx]
                obj_id = multi_view_data["obj_id"]
                camera_id = multi_view_data["camera_id"]

                sample_multi_view_path = os.path.join(self.data_dir, obj_id, camera_id)

                if not os.path.exists(sample_multi_view_path):
                    print(f"!!!! {sample_multi_view_path} not found !!!!")
                    continue


                vggt_preds_path = get_vggt_preds_path(sample_multi_view_path)
                predictions = safe_load_or_retry(check_and_load_npz, vggt_preds_path)

                ## recon info 
                gt_images, gt_masks = predictions["images"], predictions["masks"]
                gt_depth, gt_depth_conf = predictions["depth"], predictions["depth_conf"]
                gt_world_points, gt_world_points_conf = predictions["world_points"], predictions["world_points_conf"]

                # Use the batch-level recon_num if set, otherwise use random
                if self.current_recon_num is not None:
                    recon_num = self.current_recon_num
                else:
                    # 如果没有设置batch级别的recon_num，使用固定值以避免分布式训练问题
                    # 这种情况通常不应该发生，因为BucketBatchSampler会设置current_recon_num
                    recon_num = (self.min_recon_num + self.max_recon_num) // 2
                
                # Handle case where we don't have enough images
                sample_recon_idx = sample_recon_indices(len(gt_images), recon_num)
                
                # 3. torch.from_numpy 封装
                sample_recon_images = torch_from_numpy_indices(gt_images, sample_recon_idx)
                sample_recon_masks = torch_from_numpy_indices(gt_masks, sample_recon_idx)
                sample_recon_depth = torch_from_numpy_indices(gt_depth, sample_recon_idx)
                sample_recon_depth_conf = torch_from_numpy_indices(gt_depth_conf, sample_recon_idx)
                sample_recon_world_points = torch_from_numpy_indices(gt_world_points, sample_recon_idx)
                sample_recon_world_points_conf = torch_from_numpy_indices(gt_world_points_conf, sample_recon_idx)

                ## get recon gt image
                sample_recon_gt_idxs = get_sample_gt_idx(multi_view_data, sample_recon_idx)
                sample_recon_ori_gt_images = collect_gt_images(multi_view_data, sample_multi_view_path, sample_recon_gt_idxs, self.buckets)


                ## render info 
                sample_render_idx = random.randint(0, len(predictions["images"]) - 1)
                # sample_render_idx = sample_recon_idx[1]

                sample_render_gt_image = torch.from_numpy(gt_images[sample_render_idx])
                sample_render_gt_mask = torch.from_numpy(gt_masks[sample_render_idx])

                ## get render gt image
                sample_render_gt_idx = get_sample_gt_idx(multi_view_data, sample_render_idx)
                sample_render_ori_gt_image = collect_gt_images(multi_view_data, sample_multi_view_path, sample_render_gt_idx, self.buckets)

                ## get render instruction
                sample_render_instruction = get_instruction(multi_view_data, sample_render_gt_idx)

                ## get render pose and recon pose, and convert to relative pose
                ## scale pose enc to b,s,9
                sample_render_extrinsic, sample_render_intrinsic = get_pose_and_convert(predictions, sample_render_idx, predictions["images"].shape[-2:])

                ## convert to relative pose
                sample_recon_extrinsic, sample_recon_intrinsic = get_pose_and_convert(predictions, sample_recon_idx, predictions["images"].shape[-2:])

                anchor_extrinsic = sample_recon_extrinsic[0].squeeze(0)

                sample_recon_extrinsic = convert_camera_pose_to_relative(sample_recon_extrinsic, anchor_extrinsic)

                sample_render_extrinsic = convert_camera_pose_to_relative(sample_render_extrinsic, anchor_extrinsic)

                ## sample idx
                sample_recon_idx = torch.from_numpy(np.array(sample_recon_idx))
                sample_render_idx = torch.from_numpy(np.array(sample_render_idx))

                return sample_render_intrinsic, sample_render_extrinsic, sample_render_gt_image, sample_render_gt_mask, sample_render_instruction, sample_render_ori_gt_image, sample_render_idx, sample_recon_intrinsic, sample_recon_extrinsic, sample_recon_images, sample_recon_masks, sample_recon_ori_gt_images, sample_recon_idx, sample_recon_depth, sample_recon_depth_conf, sample_recon_world_points, sample_recon_world_points_conf, sample_multi_view_path
                    
            except Exception as e:
                print("data load exception", e)
                continue


def collate_fn(batch):
    sample_render_intrinsics, sample_render_extrinsics, sample_render_gt_images, sample_render_gt_masks, sample_render_instructions, sample_render_ori_gt_images, sample_render_idxs, sample_recon_intrinsics, sample_recon_extrinsics, sample_recon_images, sample_recon_masks, sample_recon_ori_gt_images, sample_recon_idxs, sample_recon_depths, sample_recon_depth_confs, sample_recon_world_points, sample_recon_world_points_confs, sample_multi_view_paths = zip(*batch)

    sample_render_intrinsics = torch.stack(sample_render_intrinsics, dim=0)
    sample_render_extrinsics = torch.stack(sample_render_extrinsics, dim=0)
    sample_render_gt_images = torch.stack(sample_render_gt_images, dim=0)
    sample_render_gt_masks = torch.stack(sample_render_gt_masks, dim=0)
    sample_render_idxs = torch.stack(sample_render_idxs, dim=0)
    sample_render_ori_gt_images = torch.stack(sample_render_ori_gt_images, dim=0)
    sample_render_instructions = [sample_render_instruction for sample_render_instruction in sample_render_instructions]

    sample_recon_intrinsics = torch.stack(sample_recon_intrinsics, dim=0)
    sample_recon_extrinsics = torch.stack(sample_recon_extrinsics, dim=0)
    sample_recon_images = torch.stack(sample_recon_images, dim=0)
    sample_recon_masks = torch.stack(sample_recon_masks, dim=0)
    sample_recon_idxs = torch.stack(sample_recon_idxs, dim=0)
    sample_recon_depths = torch.stack(sample_recon_depths, dim=0)
    sample_recon_depth_confs = torch.stack(sample_recon_depth_confs, dim=0)
    sample_recon_world_points = torch.stack(sample_recon_world_points, dim=0)
    sample_recon_world_points_confs = torch.stack(sample_recon_world_points_confs, dim=0)
    sample_recon_ori_gt_images = torch.stack(sample_recon_ori_gt_images, dim=0)



    return_dict = {
        "sample_render_intrinsics": sample_render_intrinsics,
        "sample_render_extrinsics": sample_render_extrinsics,
        "sample_render_gt_images": sample_render_gt_images,
        "sample_render_gt_masks": sample_render_gt_masks,
        "sample_render_instructions": sample_render_instructions,
        "sample_render_ori_gt_images": sample_render_ori_gt_images,
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
        "sample_multi_view_paths": sample_multi_view_paths,
        "sample_recon_ori_gt_images": sample_recon_ori_gt_images
    }

    return return_dict


class BucketBatchSampler(torch.utils.data.Sampler):
    """Sampler that ensures each batch uses the same resolution bucket and recon_num"""
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False, rank=0, world_size=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.num_samples = len(dataset)
        
        # 分布式训练参数
        self.rank = rank
        self.world_size = world_size
        
        # 设置固定的随机种子，确保所有进程生成相同的随机序列
        self.base_seed = 1234
        
        # Group indices by bucket
        self.bucket_indices = [[] for _ in range(len(dataset.buckets))]
        for idx in range(len(dataset)):
            bucket_idx = dataset.get_bucket_idx(idx)
            self.bucket_indices[bucket_idx].append(idx)
        
    def __iter__(self):
        # 创建一个确定性的随机数生成器
        g = torch.Generator()
        g.manual_seed(self.base_seed)
        
        # Create batches for each bucket
        all_batches = []
        for bucket_idx, indices_in_bucket in enumerate(self.bucket_indices):
            # Create batches from this bucket
            for i in range(0, len(indices_in_bucket), self.batch_size):
                batch = indices_in_bucket[i:i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue  # Skip incomplete batches if drop_last is True
                if len(batch) == self.batch_size:  # Only yield complete batches
                    all_batches.append(batch)
        
        # Shuffle the order of batches
        if self.shuffle:
            # 使用确定性洗牌，确保所有进程得到相同的批次顺序
            batch_indices = torch.randperm(len(all_batches), generator=g).tolist()
            all_batches = [all_batches[i] for i in batch_indices]
        
        # 为每个批次预先生成随机的recon_num值
        # 使用相同的生成器，确保所有进程生成相同的recon_num序列
        batch_recon_nums = []
        for _ in range(len(all_batches)):
            # 生成一个min_recon_num到max_recon_num之间的随机整数
            recon_num = torch.randint(
                self.dataset.min_recon_num, 
                self.dataset.max_recon_num + 1,  # +1是因为randint的上界是开区间
                (1,), 
                generator=g
            ).item()
            batch_recon_nums.append(recon_num)
            
        # Yield batches with a random recon_num for each batch
        for batch_idx, batch in enumerate(all_batches):
            # 为当前批次设置预先生成的recon_num
            self.dataset.set_recon_num(batch_recon_nums[batch_idx])
            yield batch
                
    def __len__(self):
        if self.drop_last:
            return sum(len(indices) // self.batch_size for indices in self.bucket_indices)
        else:
            return sum((len(indices) + self.batch_size - 1) // self.batch_size for indices in self.bucket_indices)


def loader(train_batch_size, num_workers, shuffle=False, rank=0, world_size=1, **args):
    dataset = MVImageDataset(**args)
    batch_sampler = BucketBatchSampler(dataset, train_batch_size, shuffle=shuffle, rank=rank, world_size=world_size)
    return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=collate_fn)

def print_tensor_shapes(**kwargs):
    for k, v in kwargs.items():
        print(f"{k}.shape", v.shape if hasattr(v, 'shape') else type(v))

if __name__ == '__main__':

    import ipdb; ipdb.set_trace()

    train_batch_size = 1
    dataloader = loader(train_batch_size=train_batch_size, num_workers=0, meta_path='mv_kontext/data/navi/metainfo/navi_v1.5_metainfo_reorg.json', data_dir='mv_kontext/data/navi/navi_v1.5', shuffle=True)
    print("num samples", len(dataloader)*train_batch_size)


    metadata_list = []
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
        sample_recon_ori_gt_images = data["sample_recon_ori_gt_images"]
        
        # save_dir = "data/navi/navi_test_data"
        # os.makedirs(save_dir, exist_ok=True)
        
        sample_multi_view_path = sample_multi_view_paths[0]
        obj_id = sample_multi_view_path.split("/")[-2]
        camera_id = sample_multi_view_path.split("/")[-1]


        # Save image data as NPY
        image_data = {
            "sample_render_gt_images": sample_render_gt_images.numpy(),
            "sample_render_gt_masks": sample_render_gt_masks.numpy(),
            "sample_render_ori_gt_images": sample_render_ori_gt_images.numpy(),
            "sample_recon_images": sample_recon_images.numpy(),
            "sample_recon_masks": sample_recon_masks.numpy(),
            "sample_recon_depths": sample_recon_depths.numpy(),
            "sample_recon_depth_confs": sample_recon_depth_confs.numpy(),
            "sample_recon_world_points": sample_recon_world_points.numpy(),
            "sample_recon_world_points_confs": sample_recon_world_points_confs.numpy(),
            "sample_recon_ori_gt_images": sample_recon_ori_gt_images.numpy()
        }
        
        # npy_path = os.path.join(save_dir, f"test_{obj_id}_{camera_id}.npy")
        # np.save(npy_path, image_data)
        
        print(f"Saved image data to {npy_path}")

        # Save metadata as JSON
        metadata = {
            "obj_id": obj_id,
            "camera_id": camera_id,
            "sample_render_extrinsic": sample_render_extrinsic.cpu().numpy().tolist(),
            "sample_render_intrinsic": sample_render_intrinsic.cpu().numpy().tolist(),
            "sample_render_idxs": sample_render_idxs.cpu().numpy().tolist(),
            "sample_render_instructions": [str(instr) for instr in sample_render_instructions],
            "sample_recon_intrinsics": sample_recon_intrinsics.cpu().numpy().tolist(),
            "sample_recon_extrinsics": sample_recon_extrinsics.cpu().numpy().tolist(),
            "sample_recon_idxs": sample_recon_idxs.cpu().numpy().tolist(),
            "sample_multi_view_paths": [str(path) for path in sample_multi_view_paths],
        }
        metadata_list.append(metadata)


        print_tensor_shapes(
            sample_render_extrinsic=sample_render_extrinsic,
            sample_render_intrinsic=sample_render_intrinsic,
            sample_render_gt_images=sample_render_gt_images,
            sample_render_gt_masks=sample_render_gt_masks,
            sample_render_ori_gt_images=sample_render_ori_gt_images,
            sample_render_instructions=sample_render_instructions,
            sample_render_idxs=sample_render_idxs,
            sample_recon_intrinsics=sample_recon_intrinsics,
            sample_recon_extrinsics=sample_recon_extrinsics,
            sample_recon_images=sample_recon_images,
            sample_recon_masks=sample_recon_masks,
            sample_recon_idxs=sample_recon_idxs,
            sample_recon_depths=sample_recon_depths,
            sample_recon_depth_confs=sample_recon_depth_confs,
            sample_recon_world_points=sample_recon_world_points,
            sample_recon_world_points_confs=sample_recon_world_points_confs,
            sample_recon_ori_gt_images=sample_recon_ori_gt_images
        )

        print("="*100)
        
        if i == 3:
            break 
    

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # json_path = os.path.join(save_dir, f"test_metainfo_{timestamp}.json")
    # with open(json_path, 'w') as f:
    #     json.dump(metadata_list, f, indent=2)

    # print(f"Saved metadata to {json_path}")