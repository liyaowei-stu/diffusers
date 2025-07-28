import sys,os
import numpy as np
import matplotlib.pyplot as plt
import torch


from . import viz2d
from .common import get_batch_split_indices
from .check_funcs import check_correspondence_quality

def visualize_paired_correspondence(recon_coords, render_coords, recon_images, render_images, output_prefix="correspondence_visualization", output_dir=".", max_points=None, denormalize_coords=True, padded_h=518, padded_w=518, lw=0.2, color="lime", ps=4, a=1.0, conf_packed=None, sample_recon_extrinsics=None,  sample_render_extrinsics=None, match_ratio_threshold=0.15, rotation_threshold=90, translation_threshold=1.0):
    """
    Visualize correspondences between reconstruction and rendering images.
    
    Args:
        recon_coords: (N, 3), the 3rd dim is (i,j,k) --> (b,w,h)
        render_coords: (N, 3), the 3rd dim is (i,j,k) --> (b,w,h)
        recon_images: (S, 3, H, W), the value is in [-1, 1], torch.Tensor or np.ndarray, (S, 3, H, W)
        render_images: (3, H, W) or (1, 3, H, W), the value is in [-1, 1], torch.Tensor or np.ndarray, (1, 3, H, W)
        denormalize_coords: bool, if True, denormalize the coordinates to pixel coordinates
        max_points: int, the maximum number of points to visualize
        output_prefix: str, the prefix of the output file
        output_dir: str, the directory of the output file
        padded_h: int, the height of the padded image (vggt)
        padded_w: int, the width of the padded image (vggt)
        lw: float, the width of the lines
        color: str, the color of the lines
        ps: int, the size of the points
        a: float, the alpha of the points
        conf_packed: (optional) (N, 1) array, corresponding one-to-one with recon_coords, the confidence of the points
        sample_recon_extrinsics: (optional) (S, 3, 4) array, the extrinsic matrices of the reconstruction images
        sample_render_extrinsics: (optional) (S, 3, 4) array, the extrinsic matrices of the rendering images
        
    Returns:
        None
    """

    if isinstance(recon_images, np.ndarray):
        recon_images = torch.from_numpy(recon_images)
    if isinstance(render_images, np.ndarray):
        render_images = torch.from_numpy(render_images)

    if render_images.ndim == 3:
        render_images = render_images[None, :, :, :]

    if sample_render_extrinsics is not None:
        if sample_render_extrinsics.ndim == 2:
            sample_render_extrinsics = sample_render_extrinsics[None, :, :]

    if sample_recon_extrinsics is not None:
        if sample_recon_extrinsics.ndim == 2:
            sample_recon_extrinsics = sample_recon_extrinsics[None, :, :]

    # Normalize images to [0, 1] range
    render_images = (render_images + 1) / 2
    recon_images = (recon_images + 1) / 2


    if denormalize_coords:
        # Scale normalized coordinates to pixel coordinates
        render_coords = denormalize_image_coordinates(render_coords, height=render_images.shape[-2], width=render_images.shape[-1], align_corners=True)[0]

        recon_coords = transfer_coords_list_from_padded_to_original(recon_coords, padded_h=padded_h, padded_w=padded_w, original_h=render_images.shape[-2], original_w=render_images.shape[-1], is_normalized=True)    



    # Get batch split indices
    recon_coords_to_packed_first_idx = get_batch_split_indices(recon_coords)

    # Get x,y coordinates
    recon_coords_xy = recon_coords[..., 1:]
    render_coords_xy = render_coords[..., 1:]

    # Process each batch
    for idx in range(len(recon_coords_to_packed_first_idx) - 1):
        start = recon_coords_to_packed_first_idx[idx]
        end = recon_coords_to_packed_first_idx[idx + 1]
        recon_coords_xy_batch = recon_coords_xy[start:end]
        render_coords_xy_batch = render_coords_xy[start:end]
        if conf_packed is not None:
            conf_packed_batch = conf_packed[start:end]

        # Limit maximum number of visualization points, currently use points_conf to filtering (maybe not the best way)
        num_points = recon_coords_xy_batch.shape[0]
        if max_points is not None and num_points > max_points:
            # get the top max_points points
            top_indices = np.argsort(conf_packed_batch)[-max_points:]
            # select these points
            recon_coords_xy_batch = recon_coords_xy_batch[top_indices]
            render_coords_xy_batch = render_coords_xy_batch[top_indices]
            # update the conf_packed_batch
            conf_packed_batch = conf_packed_batch[top_indices]


        # check the camera view difference
        if sample_recon_extrinsics is not None and sample_render_extrinsics is not None:
            extrinsic1 = sample_recon_extrinsics[idx]
            extrinsic2 = sample_render_extrinsics[0] if sample_render_extrinsics.shape[0] == 1 else sample_render_extrinsics[idx]
            match_num, total_match_num = end - start, recon_coords_to_packed_first_idx[-1]
            match_ratio = match_num/total_match_num
            too_different, rotation_angle, translation_magnitude = check_correspondence_quality(extrinsic1, extrinsic2, match_ratio, match_ratio_threshold=match_ratio_threshold, rotation_threshold=rotation_threshold, translation_threshold=translation_threshold)
            if too_different:
                print(f"View {idx} is too different, match_ratio: {match_ratio}, rotation_angle: {rotation_angle}, translation_magnitude: {translation_magnitude}")
                print("-"*100)
            else:
                print(f"View {idx} is good, match_ratio: {match_ratio}, rotation_angle: {rotation_angle}, translation_magnitude: {translation_magnitude}")
                print("="*100)

        # Create visualization
        render_image = render_images[0] if render_images.shape[0] == 1 else render_images[idx]
        axes = viz2d.plot_images([render_image, recon_images[idx]])
        if max_points is not None:
            # view matches, too slow when num_points is large
            viz2d.plot_matches(render_coords_xy_batch, recon_coords_xy_batch, color=color, lw=lw, ps=ps, a=a)
        else:
            # view keypoints
            viz2d.plot_keypoints([render_coords_xy_batch, recon_coords_xy_batch], colors=color, ps=ps, a=a)
        viz2d.add_text(0, f'Render', fs=40)
        viz2d.save_plot(f"{output_dir}/{output_prefix}_{idx}.png")


def normalize_image_coordinates(coords, height, width, align_corners=True):
    """
    Normalize image coordinates to range [0, 1].
    
    Args:
        coords: list of (N, 3);  (bsz,x,y)
        height: int
        width: int
        align_corners: bool, if True, extreme values [0, 1] are considered as referring to the center points of the edge pixels

    Returns:
        normalized_coords: list of (N, 3)
    """

    if isinstance(coords, list):
        coords = [coord.reshape(-1, 3) for coord in coords]
    else:
        coords = [coords.reshape(-1, 3)]

    normalized_coords = []
    for coord in coords:
        coord = coord.astype(np.float32, copy=True)
        if align_corners:
            coord[:, 1] = coord[:, 1] / float(width - 1)
            coord[:, 2] = coord[:, 2] / float(height - 1)
        else:
            coord[:, 1] = coord[:, 1] / float(width)
            coord[:, 2] = coord[:, 2] / float(height)
        normalized_coords.append(coord)
    return normalized_coords


def denormalize_image_coordinates(normalized_coords, height, width, align_corners=True):
    """
    Denormalize image coordinates from range [0, 1] to original image coordinates.
    
    Args:
        normalized_coords: list of (N, 3);  (bsz,x,y) with x,y in [0,1]
        height: int
        width: int
        align_corners: bool, if True, extreme values [0, 1] are considered as referring to the center points of the edge pixels

    Returns:
        coords: list of (N, 3) with x,y in pixel coordinates
    """
    if isinstance(normalized_coords, list):
        normalized_coords = [coord.reshape(-1, 3) for coord in normalized_coords]
    else:
        normalized_coords = [normalized_coords.reshape(-1, 3)]
    
    denormalized_coords = []
    for coord in normalized_coords:
        coord = coord.astype(np.float32, copy=True)
        if align_corners:
            coord[:, 1] = coord[:, 1] * float(width - 1)
            coord[:, 2] = coord[:, 2] * float(height - 1)
        else:
            coord[:, 1] = coord[:, 1] * float(width)
            coord[:, 2] = coord[:, 2] * float(height)
        denormalized_coords.append(coord)
    return denormalized_coords


def transfer_coords_from_padded_to_original(
    coords,
    padded_h=518,
    padded_w=518,
    original_h=752,
    original_w=1392,
    is_normalized=False
):
    """
    Transfer coordinates from a padded square image back to the original image while preserving aspect ratio.
    This function is designed to work with VGGT, which resizes the long edge of input images to 518 pixels,
    resizes the short edge while maintaining aspect ratio, and pads the short edge with zeros to 518 pixels.
    
    Args:
        coords: Coordinates on the padded image, shape (N,3), where first column is batch index, followed by x,y coordinates, the x,y coordinates are normalized to 0-1 range when is_normalized is True, otherwise, the x,y coordinates are in padded image pixel coordinates.
        padded_h, padded_w: Dimensions of the padded image.
        original_h, original_w: Height and width of the original image.
        is_normalized: Whether input coordinates are already normalized to 0-1 range. If True, will first convert to pixel coordinates.
    
    Returns:
        original_coords: Coordinates on the original image, shape (N,3), where first column is batch index, followed by x,y coordinates. The x,y coordinates are in original image pixel coordinates.
    """
    # Ensure numpy array
    coords = np.array(coords)
    original_coords = coords.copy()
    
    # If input is normalized coordinates, convert to pixel coordinates first
    if is_normalized:
        original_coords[:, 1] = original_coords[:, 1] * padded_w
        original_coords[:, 2] = original_coords[:, 2] * padded_h
    
    # Calculate aspect ratio of original image
    original_ratio = original_w / original_h
    
    # Determine scaling factor and padding
    if original_ratio > 1:  # Landscape image (width > height)
        # Width is the limiting factor, height will have padding
        scale_factor = padded_w / original_w
        effective_h = original_h * scale_factor
        pad_y = (padded_h - effective_h) / 2
        pad_x = 0
    else:  # Portrait image (height > width)
        # Height is the limiting factor, width will have padding
        scale_factor = padded_h / original_h
        effective_w = original_w * scale_factor
        pad_x = (padded_w - effective_w) / 2
        pad_y = 0
    
    # Subtract padding and apply inverse scaling
    original_coords[:, 1] = (original_coords[:, 1] - pad_x) / scale_factor
    original_coords[:, 2] = (original_coords[:, 2] - pad_y) / scale_factor
    
    # Ensure coordinates are within valid range
    original_coords[:, 1] = np.clip(original_coords[:, 1], 0, original_w)
    original_coords[:, 2] = np.clip(original_coords[:, 2], 0, original_h)
    
    return original_coords


def transfer_coords_list_from_padded_to_original(
    coords_list,
    original_h,
    original_w,
    padded_h=518,
    padded_w=518,
    is_normalized=False
):
    """
    Transfer coordinates from padded image to original image while preserving the aspect ratio
    """
    if isinstance(coords_list, list):
        coords_list = [transfer_coords_from_padded_to_original(coords, padded_h, padded_w, original_h, original_w, is_normalized) for coords in coords_list]
    else:
        coords_list = transfer_coords_from_padded_to_original(coords_list, padded_h, padded_w, original_h, original_w, is_normalized)
    return coords_list


def group_by_first_column(recon_coords, render_coords):
    """
    Group by the first column (view index) of recon_coords
    
    Args:
        recon_coords: (N, 3) array, first column is view index (0~7)
        render_coords: (N, 3) array, corresponding one-to-one with recon_coords
        
    Returns:
        grouped_recon_coords: list of recon_coords grouped by view index
        grouped_render_coords: list of render_coords grouped by view index
    """
    # First sort by the first column
    sort_indices = np.argsort(recon_coords[:, 0])
    sorted_recon_coords = recon_coords[sort_indices]
    sorted_render_coords = render_coords[sort_indices]
    
    # Get unique view indices
    unique_view_indices = np.unique(sorted_recon_coords[:, 0])
    
    # Group by view index
    grouped_recon_coords = []
    grouped_render_coords = []
    for view_idx in unique_view_indices:
        mask = (sorted_recon_coords[:, 0] == view_idx)
        grouped_recon_coords.append(sorted_recon_coords[mask])
        grouped_render_coords.append(sorted_render_coords[mask])
    
    return grouped_recon_coords, grouped_render_coords


def sort_by_first_column(recon_coords, render_coords, conf_packed=None):
    """
    Sort by the first column (view index) of recon_coords in ascending order
    
    Args:
        recon_coords: (N, 3) array, first column is view index (e.g., [0, 1, 2, 3, 4, 5, 6, 7])
        render_coords: (N, 3) array, corresponding one-to-one with recon_coords
        conf_packed: (optional) (N, 1) array, corresponding one-to-one with recon_coords, the confidence of the points
    Returns:
        sorted_recon_coords: sorted recon_coords
        sorted_render_coords: sorted render_coords
        sorted_conf_packed: sorted conf_packed
    """
    # Get sorting indices
    sort_indices = np.argsort(recon_coords[:, 0])
    
    # Apply sorting indices
    sorted_recon_coords = recon_coords[sort_indices]
    sorted_render_coords = render_coords[sort_indices]
    if conf_packed is not None:
        sorted_conf_packed = conf_packed[sort_indices]

    return sorted_recon_coords, sorted_render_coords, sorted_conf_packed
    

def extract_correspondence_from_fragments(fragments, recon_coords_packed_list_normalized, topk=1, align_corners=True, recon_images=None, conf_flat_list=None, render_mask=None):
    """
    Args:
        fragments: idx, zbuf, dists, see pytorch3d.renderer.points. fragments.idx: (B, H, W, num_points_per_pixel)
        topk: topk of num_points_per_pixel correspondence
        recon_coords_packed_list_normalized: list of (K_i, 3)  # (b, x, y)
        align_corners: bool, if True, align the corners of the image
        recon_images: (B, 3, H, W) array, the reconstruction images, for debug visualization
        conf_flat_list: list of (K_i, 1) array, the confidence of the points
        render_mask: (B, H, W) array, the mask of the rendering image

    Returns:
        recon_coords_packed_list_normalized_valid: list of (M_i, 3)  # (b, x, y), M_i < K_i
        render_coords_packed_list_normalized_valid: list of (M_i, 3)   # original_coords
    """

    # # check the input recon_coords_packed_list_normalized, for debug
    # recon_coords_packed_list_denormalized = denormalize_image_coordinates(recon_coords_packed_list_normalized, height=recon_images.shape[-2], width=recon_images.shape[-1], align_corners=align_corners)
    # visualize_i_view_with_keypoints(recon_coords_packed_list_denormalized[0], recon_images, i=0, output_prefix="recon_coords_denormalized")


    idx = fragments.idx[..., :topk]
    if hasattr(idx, "cpu"):
        idx_np = idx.cpu().numpy()
    else:
        idx_np = np.asarray(idx)

    b, h, w, _ = idx_np.shape

    valid_mask = (idx_np != -1)
    # Visualize valid_mask for debugging
    if recon_images is not None:
        for b_idx in range(b):
            # Create a visualization of the valid mask
            mask_vis = valid_mask[b_idx, :, :, 0].astype(np.uint8) * 255
            
            # Save the visualization
            save_dir = os.path.join(".", "debug_visualizations")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"valid_mask_batch_{b_idx}.png")
            
            # Convert to PIL image and save
            from PIL import Image
            Image.fromarray(mask_vis).save(save_path)
            
            # Optional: overlay the mask on the render image for better visualization
            if render_mask is not None:
                combined_mask = (valid_mask[b_idx, :, :, 0] & render_mask[b_idx]).astype(np.uint8) * 255
                save_path = os.path.join(save_dir, f"combined_mask_batch_{b_idx}.png")
                Image.fromarray(combined_mask).save(save_path)

    if render_mask is not None:
        valid_mask = valid_mask & render_mask[..., None].astype(bool)

    # (b, y, x, k) order from np.where
    b_idx, y_idx, x_idx, k_idx = np.where(valid_mask)

    packed_indices = idx_np[b_idx, y_idx, x_idx, k_idx]
    
    # Concatenate all coordinates with batch indices
    recon_coords_packed_normalized = np.concatenate(recon_coords_packed_list_normalized, axis=0)
    recon_coords_packed_normalized_valid = recon_coords_packed_normalized[packed_indices]

    if conf_flat_list is not None:
        conf_packed = np.concatenate(conf_flat_list, axis=0)
        conf_packed_valid = conf_packed[packed_indices]
    else:
        conf_packed_valid = None

    # render_coords: (N, 3)  # (b, x, y)
    render_coords_packed = np.stack([b_idx, x_idx, y_idx], axis=1)
    render_coords_packed_normalized_valid = normalize_image_coordinates(render_coords_packed, height=h, width=w, align_corners=align_corners)[0]


    # Ensure all outputs are numpy arrays
    recon_coords_packed_normalized_valid = np.asarray(recon_coords_packed_normalized_valid)
    render_coords_packed_normalized_valid = np.asarray(render_coords_packed_normalized_valid)

    # split the valid points into batches
    batch_split_indices = get_batch_split_indices(b_idx)
    recon_coords_packed_list_normalized_valid = []
    render_coords_packed_list_normalized_valid = []
    conf_packed_list_valid = []
    for i in range(len(batch_split_indices) - 1):
        start = batch_split_indices[i]
        end = batch_split_indices[i + 1]
        recon_coords_packed_normalized_valid_slice = recon_coords_packed_normalized_valid[start:end]
        render_coords_packed_normalized_valid_slice = render_coords_packed_normalized_valid[start:end]

        if conf_packed_valid is not None:
            conf_packed_valid_slice = conf_packed_valid[start:end]

        recon_coords_packed_normalized_valid_slice, render_coords_packed_normalized_valid_slice, conf_packed_valid_slice = sort_by_first_column(recon_coords_packed_normalized_valid_slice, render_coords_packed_normalized_valid_slice, conf_packed_valid_slice)

        recon_coords_packed_list_normalized_valid.append(recon_coords_packed_normalized_valid_slice)
        render_coords_packed_list_normalized_valid.append(render_coords_packed_normalized_valid_slice)
        conf_packed_list_valid.append(conf_packed_valid_slice)
        
    # check the output recon_coords_packed_list_normalized_valid, for debug
    # recon_coords_packed_list_denormalized_valid = denormalize_image_coordinates(recon_coords_packed_list_normalized_valid, height=recon_images.shape[-2], width=recon_images.shape[-1], align_corners=align_corners)
    # visualize_i_view_with_keypoints(recon_coords_packed_list_denormalized_valid[0], recon_images, i=0, output_prefix="recon_coords_denormalized_valid")

    return recon_coords_packed_list_normalized_valid, render_coords_packed_list_normalized_valid, conf_packed_list_valid


