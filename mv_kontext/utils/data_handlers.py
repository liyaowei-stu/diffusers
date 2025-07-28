import random
import math

import torch
import numpy as np

from PIL import Image

from torchvision import transforms
from torchvision.transforms import functional as TF
from torchvision.utils import make_grid



def find_nearest_bucket(h, w, bucket_options):
    """Finds the closes bucket to the given height and width."""
    min_metric = float("inf")
    best_bucket_idx = None
    for bucket_idx, (bucket_h, bucket_w) in enumerate(bucket_options):
        metric = abs(h * bucket_w - w * bucket_h)
        if metric <= min_metric:
            min_metric = metric
            best_bucket_idx = bucket_idx
    return best_bucket_idx


def paired_transform(image, dest_image=None, size=(224, 224), random_flip=False):
    # 1. Resize (deterministic)
    resize = transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)
    image = resize(image)
    if dest_image is not None:
        dest_image = resize(dest_image)

    # 2. Random horizontal flip with the SAME coin flip
    if random_flip:
        do_flip = random.random() < 0.5
        if do_flip:
            image = TF.hflip(image)
            if dest_image is not None:
                dest_image = TF.hflip(dest_image)

    # 3. ToTensor + Normalize (deterministic)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize([0.5], [0.5])
    image = normalize(to_tensor(image))
    if dest_image is not None:
        dest_image = normalize(to_tensor(dest_image))

    return (image, dest_image) if dest_image is not None else (image, None)


def img_tensor_to_pil(image, rescale=True):
    """
    Convert a tensor image to a PIL image.
    
    Args:
        image: Tensor of shape [channels, height, width] with values in range 0-1
        rescale: Whether to rescale the image from -1~1 to 0~1

    Returns:
        PIL Image
    """
    if rescale:
        image = image * 0.5 + 0.5
    image = image.cpu().numpy()
    image = image.transpose(1, 2, 0)
    image = (image * 255).astype(np.uint8)
    image = Image.fromarray(image)
    return image


def convert_tensor_to_img_grid(images, rescale=True, num_rows=2):
    """
    Convert a batch of tensor images into a single grid image.
    
    Args:
        images: Tensor of shape [batch_size, channels, height, width] with values in range 0~1 or -1~1
        rescale: Whether to rescale the image from -1~1 to 0~1
        num_rows: Number of rows in the grid
    
    Returns:
        PIL Image containing the grid of images
    """
    if rescale:
        images = images * 0.5 + 0.5
    # Calculate number of columns based on number of images and rows
    batch_size = images.shape[0]
    num_cols = math.ceil(batch_size / num_rows)
    
    if images.ndim == 3:
        if images.shape[0] == 3 or images.shape[0] == 1:
            images = images.unsqueeze(0)
        else:
            images = images.unsqueeze(1)

    if images.shape[1] == 1:
        images = images.repeat(1, 3, 1, 1)

    # Create grid
    grid = make_grid(images, nrow=num_cols, padding=2, normalize=False)
    
    # Convert to PIL image
    grid_np = grid.cpu().numpy().transpose(1, 2, 0)
    grid_np = (grid_np * 255).astype(np.uint8)
    grid_img = Image.fromarray(grid_np).convert("RGB")
    
    return grid_img
    