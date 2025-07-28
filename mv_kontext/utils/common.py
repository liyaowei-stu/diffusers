import numpy as np
import torch
from torchvision.utils import save_image

def get_batch_split_indices(coords, batch_idx=None):
    """
    Args:
        coords: (N, 3) or (N,), the 0th dim is batch index (e.g., (s, x, y), and s indices are assumed to be contiguous integers from 0 to S-1.
        batch_idx: (N,), optional, the batch index of each point. If not provided, will use coords[..., 0].
    Returns:
        split_indices: (B+1,), the start/end indices for each batch in the flattened array.
    """
    # 支持传入batch_idx或者直接用coords[...,0]
    
    if batch_idx is None:
        if coords.ndim == 1:
            batch_indices = coords
        else:
            batch_indices = coords[..., 0]
    else:
        batch_indices = batch_idx
    # 转为numpy并保证为int64
    if hasattr(batch_indices, "cpu"):
        batch_indices = batch_indices.cpu().numpy()
    else:
        batch_indices = np.asarray(batch_indices)
    batch_indices = batch_indices.astype(np.int64)
    # 统计每个batch的点数
    batch_counts = np.bincount(batch_indices)
    # 计算每个batch的起止索引
    split_indices = np.zeros(len(batch_counts) + 1, dtype=np.int64)
    split_indices[1:] = np.cumsum(batch_counts)
    return split_indices


def visualize_i_view_with_keypoints(coords_flat, images, mask=None, i=0, output_prefix="view_wt", output_dir="."):
    """
    Visualize the i-th view with keypoints and save the images.
    
    Args:
        coords_flat: Flattened coordinates, (N, 3)
        images: Images tensor, (N, 3, H, W)
        mask: Mask tensor, (N, H, W)
        i: View index to visualize, 0~N-1
        output_prefix: Prefix for output filenames
    """
    from . import viz2d
    
    view_split_idx = get_batch_split_indices(coords_flat)
    assert i < len(view_split_idx) - 1, f"View index {i} is out of range, should be 0~{len(view_split_idx)-2}"
    i_view_coords_flat = coords_flat[view_split_idx[i]:view_split_idx[i+1]][:, 1:]

    if isinstance(images, torch.Tensor):
        i_view = images[i].permute(1, 2, 0).cpu().numpy()
    else:
        i_view = images[i].transpose(1, 2, 0)

    
    viz2d.plot_images([i_view])
    viz2d.plot_keypoints([i_view_coords_flat])
    viz2d.save_plot(f"{output_dir}/{output_prefix}_{i}_kpts.png")
    save_image(images[i], f"{output_dir}/{output_prefix}_{i}_img.png")

    if mask is not None:
        save_image(mask[i], f"{output_dir}/{output_prefix}_{i}_mask.png")



