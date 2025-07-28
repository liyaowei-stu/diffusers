import torch
import numpy as np
import cv2

from .geometry import convert_se3_to_homogeneous, convert_camera_pose_to_relative, transform_points_between_coordinate_system, project_points_to_pixels, unproject_pixels_to_points


# -------------------------------------------------------------------------
# 1) Check geometry and correspondence functions (extrinsic, intrinsic, etc.)
# -------------------------------------------------------------------------
def check_camera_view_difference(extrinsic1, extrinsic2, 
                                rotation_threshold=30, translation_threshold=1.0):
    """
    Check if two camera views are too different based on their extrinsic and intrinsic parameters.
    
    Args:
        extrinsic1: (3, 4) Camera 1 extrinsic matrix
        extrinsic2: (3, 4) Camera 2 extrinsic matrix
        rotation_threshold: Maximum allowed rotation difference in degrees
        translation_threshold: Maximum allowed translation difference in units
        
    Returns:
        bool: True if views are too different, False otherwise
        angle: axis-angle representation, the minimum angle along the certain axis to rotate the camera1 to the camera2, in degrees
        translation_magnitude: the translation magnitude between the two cameras, in units
    """
    # Convert to homogeneous if needed
    extrinsic1 = convert_se3_to_homogeneous(extrinsic1)
    extrinsic2 = convert_se3_to_homogeneous(extrinsic2)
    
    # Calculate relative pose
    relative_pose = convert_camera_pose_to_relative(extrinsic2, extrinsic1)
    
    # Extract rotation matrix and translation vector
    R = relative_pose[:3, :3]
    t = relative_pose[:3, 3]
    
    # Calculate rotation angle in degrees
    # (Trace of R = 1 + 2*cos(theta))
    cos_theta = (torch.trace(R) - 1) / 2
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)  # Ensure value is in valid range
    rotation_angle = torch.acos(cos_theta) * 180 / torch.pi
    
    # Calculate translation magnitude
    translation_magnitude = torch.norm(t)
    
    # Check if differences exceed thresholds
    too_different = (rotation_angle > rotation_threshold or 
                     translation_magnitude > translation_threshold)
    
    return too_different, rotation_angle.item(), translation_magnitude.item()


def check_intrinsic_difference(intrinsic1, intrinsic2, focal_length_ratio_threshold=1.5):
    """
    Check if intrinsic parameters are too different.
    
    Args:
        intrinsic1: (3, 3) Camera 1 intrinsic matrix
        intrinsic2: (3, 3) Camera 2 intrinsic matrix
        focal_length_ratio_threshold: Maximum allowed ratio between focal lengths
        
    Returns:
        bool: True if intrinsics are too different, False otherwise
    """
    # Extract focal lengths
    fx1, fy1 = intrinsic1[0, 0], intrinsic1[1, 1]
    fx2, fy2 = intrinsic2[0, 0], intrinsic2[1, 1]
    
    # Calculate ratios
    fx_ratio = max(fx1, fx2) / min(fx1, fx2)
    fy_ratio = max(fy1, fy2) / min(fy1, fy2)
    
    # Check if differences exceed thresholds
    too_different = (fx_ratio > focal_length_ratio_threshold or 
                     fy_ratio > focal_length_ratio_threshold)
    
    return too_different, fx_ratio, fy_ratio


def estimate_view_overlap(extrinsic1, extrinsic2, intrinsic1, intrinsic2, 
                          depth_map1, image_width, image_height, 
                          min_overlap_ratio=0.3):
    """
    Estimate the overlap between two camera views.
    
    Args:
        extrinsic1, extrinsic2: Camera extrinsic matrices
        intrinsic1, intrinsic2: Camera intrinsic matrices
        depth_map1: Depth map from camera 1
        image_width, image_height: Image dimensions
        min_overlap_ratio: Minimum required overlap ratio
        
    Returns:
        bool: True if overlap is sufficient, False otherwise
    """
    # Create a grid of points in camera 1's image plane
    y, x = torch.meshgrid(torch.arange(image_height), torch.arange(image_width))
    pixels = torch.stack([x.flatten(), y.flatten()], dim=1)
    
    # Unproject to 3D points using depth
    points_3d = unproject_pixels_to_points(pixels, depth_map1.flatten(), intrinsic1)
    
    # Transform points to camera 2's coordinate system
    points_3d_cam2 = transform_points_between_coordinate_system(points_3d, extrinsic1, extrinsic2)
    
    # Project points to camera 2's image plane
    pixels_cam2 = project_points_to_pixels(points_3d_cam2, intrinsic2)
    
    # Count points that fall within camera 2's image bounds
    valid_x = (pixels_cam2[:, 0] >= 0) & (pixels_cam2[:, 0] < image_width)
    valid_y = (pixels_cam2[:, 1] >= 0) & (pixels_cam2[:, 1] < image_height)
    valid_points = valid_x & valid_y
    
    # Calculate overlap ratio
    overlap_ratio = valid_points.sum().float() / pixels.shape[0]
    
    return overlap_ratio >= min_overlap_ratio, overlap_ratio.item()


def check_feature_correspondence(image1, image2, min_matches=50):
    """
    Check if two images have sufficient feature correspondences.
    
    Args:
        image1, image2: Input images
        min_matches: Minimum number of required matches
        
    Returns:
        bool: True if sufficient correspondences exist, False otherwise
    """
    if isinstance(image1, torch.Tensor):
        image1 = (image1.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    if isinstance(image2, torch.Tensor):
        image2 = (image2.cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    
    # Convert images to grayscale if needed
    if len(image1.shape) == 3:
        image1_gray = cv2.cvtColor(image1, cv2.COLOR_RGB2GRAY)
        image2_gray = cv2.cvtColor(image2, cv2.COLOR_RGB2GRAY)
    else:
        image1_gray = image1
        image2_gray = image2
    
    # Detect features
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(image1_gray, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2_gray, None)
    
    # Match features
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    # Check if enough good matches are found
    has_sufficient_matches = len(good_matches) >= min_matches
    
    return has_sufficient_matches, len(good_matches)


def are_views_compatible(extrinsic1, extrinsic2, intrinsic1, intrinsic2, 
                         image1, image2, depth_map1=None,
                         rotation_threshold=30, translation_threshold=1.0,
                         focal_length_ratio_threshold=1.5, min_matches=50,
                         min_overlap_ratio=0.3):
    """
    Comprehensive check if two views are compatible for correspondence.
    
    Returns:
        bool: True if views are compatible, False otherwise
        dict: Detailed metrics
    """
    metrics = {}
    
    # Check extrinsic parameters
    too_different_extrinsic, rotation_angle, translation_magnitude = check_camera_view_difference(
        extrinsic1, extrinsic2,
        rotation_threshold, translation_threshold
    )
    metrics['rotation_angle'] = rotation_angle
    metrics['translation_magnitude'] = translation_magnitude
    
    # Check intrinsic parameters
    too_different_intrinsic, fx_ratio, fy_ratio = check_intrinsic_difference(
        intrinsic1, intrinsic2, focal_length_ratio_threshold
    )
    metrics['fx_ratio'] = fx_ratio
    metrics['fy_ratio'] = fy_ratio
    
    # Check feature correspondences
    has_sufficient_matches, num_matches = check_feature_correspondence(
        image1, image2, min_matches
    )
    metrics['num_matches'] = num_matches
    
    # Check view overlap if depth map is available
    if depth_map1 is not None:
        sufficient_overlap, overlap_ratio = estimate_view_overlap(
            extrinsic1, extrinsic2, intrinsic1, intrinsic2, 
            depth_map1, image1.shape[1], image1.shape[0], 
            min_overlap_ratio
        )
        metrics['overlap_ratio'] = overlap_ratio
    else:
        sufficient_overlap = True  # Skip this check if depth is not available
    
    # Views are compatible if they pass all checks
    compatible = (not too_different_extrinsic and 
                  not too_different_intrinsic and 
                  has_sufficient_matches and 
                  sufficient_overlap)
    
    return compatible, metrics



def check_match_ratio(match_ratio, match_ratio_threshold=0.2):
    """
    Check the quality of the correspondence
    """
    return match_ratio < match_ratio_threshold


def check_correspondence_quality(extrinsic1, extrinsic2, match_ratio, match_ratio_threshold=0.2, rotation_threshold=90, translation_threshold=1.0):
    """
    Check the quality of the correspondence

    Args:
        extrinsic1, extrinsic2: Camera extrinsic matrices
        match_ratio: The ratio of the matched points
        match_ratio_threshold: The threshold of the match ratio
        rotation_threshold: The threshold of the rotation angle
        translation_threshold: The threshold of the translation magnitude
    
    Returns:
        bool: True if the correspondence is good, False otherwise
    """
    view_too_different, rotation_angle, translation_magnitude = check_camera_view_difference(extrinsic1, extrinsic2, rotation_threshold=rotation_threshold, translation_threshold=translation_threshold)

    match_too_low = check_match_ratio(match_ratio, match_ratio_threshold)

    too_different = view_too_different and match_too_low

    return too_different, rotation_angle, translation_magnitude