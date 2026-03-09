"""
Point Cloud Colorization Module

Assigns RGB colors to 3D points by projecting them to source images
and handling multi-view color consistency through weighted averaging.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from ..geometry.two_view import CameraPose
from ..utils.logger import logger


class PointCloudColorizer:
    """
    Colorizes 3D point clouds using multi-view images

    Projects 3D points to all visible camera views and computes
    weighted average of colors based on viewing angle and distance.
    """

    def __init__(
        self,
        angle_threshold: float = 85.0,
        distance_weight_sigma: float = 1.0,
        angle_weight_power: float = 2.0
    ):
        """
        Initialize point cloud colorizer

        Args:
            angle_threshold: Maximum viewing angle (degrees) for valid observation
            distance_weight_sigma: Sigma for distance-based Gaussian weighting
            angle_weight_sigma: Power for angle-based cosine weighting
        """
        self.angle_threshold = np.deg2rad(angle_threshold)
        self.distance_weight_sigma = distance_weight_sigma
        self.angle_weight_power = angle_weight_power

        logger.info(f"Initialized PointCloudColorizer: "
                   f"angle_threshold={angle_threshold}°, "
                   f"distance_sigma={distance_weight_sigma}")

    def colorize(
        self,
        points_3d: np.ndarray,
        images: List[np.ndarray],
        camera_poses: Dict[int, CameraPose],
        K: np.ndarray,
        existing_colors: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Colorize 3D points using multi-view images

        Args:
            points_3d: Nx3 array of 3D points
            images: List of images
            camera_poses: Dict mapping img_idx -> CameraPose
            K: 3x3 camera intrinsic matrix
            existing_colors: Optional Nx3 existing colors (will be blended)

        Returns:
            Nx3 array of RGB colors (0-255)
        """
        logger.info(f"Colorizing {len(points_3d)} points from {len(camera_poses)} views...")

        n_points = len(points_3d)
        colors = np.zeros((n_points, 3), dtype=np.float32)
        total_weights = np.zeros(n_points, dtype=np.float32)

        camera_indices = sorted(camera_poses.keys())

        # Accumulate weighted colors from all views
        for img_idx in camera_indices:
            image = images[img_idx]
            pose = camera_poses[img_idx]

            # Project points to image
            visible_mask, pixels, weights = self._compute_visibility_and_weights(
                points_3d, pose, K, image.shape[:2]
            )

            if not np.any(visible_mask):
                continue

            # Sample colors from image
            sampled_colors = self._bilinear_sample(
                image, pixels[visible_mask]
            )

            # Accumulate weighted colors
            colors[visible_mask] += sampled_colors * weights[visible_mask, np.newaxis]
            total_weights[visible_mask] += weights[visible_mask]

        # Normalize by total weights
        valid_mask = total_weights > 0
        colors[valid_mask] /= total_weights[valid_mask, np.newaxis]

        # Blend with existing colors if provided
        if existing_colors is not None:
            blend_weight = 0.7  # Favor new colors
            colors[valid_mask] = (
                blend_weight * colors[valid_mask] +
                (1 - blend_weight) * existing_colors[valid_mask]
            )

        # For points with no valid observations, use gray or existing color
        if not np.all(valid_mask):
            if existing_colors is not None:
                colors[~valid_mask] = existing_colors[~valid_mask]
            else:
                colors[~valid_mask] = 128.0  # Gray

        n_colored = np.sum(valid_mask)
        logger.info(f"Successfully colored {n_colored}/{n_points} points "
                   f"({n_colored/n_points*100:.1f}%)")

        return colors.astype(np.uint8)

    def _compute_visibility_and_weights(
        self,
        points_3d: np.ndarray,
        pose: CameraPose,
        K: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute which points are visible and their color weights

        Args:
            points_3d: Nx3 3D points
            pose: Camera pose
            K: Intrinsic matrix
            image_shape: (height, width)

        Returns:
            Tuple of (visible_mask, pixels, weights)
        """
        h, w = image_shape
        n_points = len(points_3d)

        # Transform to camera coordinates
        points_cam = (pose.R @ points_3d.T).T + pose.t

        # Check depth (in front of camera)
        depth_mask = points_cam[:, 2] > 0.01

        # Project to image
        points_2d_hom = (K @ points_cam.T).T
        pixels = points_2d_hom[:, :2] / (points_2d_hom[:, 2:3] + 1e-8)

        # Check if inside image bounds (with small margin)
        margin = 1
        bounds_mask = (
            (pixels[:, 0] >= margin) & (pixels[:, 0] < w - margin) &
            (pixels[:, 1] >= margin) & (pixels[:, 1] < h - margin)
        )

        # Compute viewing angles
        C = -pose.R.T @ pose.t  # Camera center
        view_directions = points_3d - C
        view_directions = view_directions / (np.linalg.norm(view_directions, axis=1, keepdims=True) + 1e-8)

        # Camera viewing direction (negative z-axis in camera frame)
        camera_direction = -pose.R.T[:, 2]

        # Cosine of viewing angle
        cos_angles = np.abs(np.dot(view_directions, camera_direction))
        angle_mask = np.arccos(np.clip(cos_angles, -1, 1)) < self.angle_threshold

        # Combined visibility mask
        visible_mask = depth_mask & bounds_mask & angle_mask

        # Compute weights based on viewing angle and distance
        distances = np.linalg.norm(points_cam, axis=1)

        # Angle weight: favor perpendicular views
        angle_weights = np.power(cos_angles, self.angle_weight_power)

        # Distance weight: Gaussian based on distance
        median_dist = np.median(distances[visible_mask]) if np.any(visible_mask) else 1.0
        distance_weights = np.exp(-np.square(distances - median_dist) / (2 * self.distance_weight_sigma ** 2))

        # Combined weight
        weights = angle_weights * distance_weights

        return visible_mask, pixels, weights

    def _bilinear_sample(
        self,
        image: np.ndarray,
        pixels: np.ndarray
    ) -> np.ndarray:
        """
        Bilinear sampling of image at non-integer pixel locations

        Args:
            image: HxWx3 image
            pixels: Nx2 pixel coordinates

        Returns:
            Nx3 sampled colors
        """
        h, w = image.shape[:2]
        x = pixels[:, 0]
        y = pixels[:, 1]

        # Get integer coordinates
        x0 = np.floor(x).astype(np.int32)
        x1 = x0 + 1
        y0 = np.floor(y).astype(np.int32)
        y1 = y0 + 1

        # Clip to image bounds
        x0 = np.clip(x0, 0, w - 1)
        x1 = np.clip(x1, 0, w - 1)
        y0 = np.clip(y0, 0, h - 1)
        y1 = np.clip(y1, 0, h - 1)

        # Get interpolation weights
        wa = ((x1 - x) * (y1 - y))[:, np.newaxis]
        wb = ((x1 - x) * (y - y0))[:, np.newaxis]
        wc = ((x - x0) * (y1 - y))[:, np.newaxis]
        wd = ((x - x0) * (y - y0))[:, np.newaxis]

        # Sample and interpolate
        Ia = image[y0, x0].astype(np.float32)
        Ib = image[y1, x0].astype(np.float32)
        Ic = image[y0, x1].astype(np.float32)
        Id = image[y1, x1].astype(np.float32)

        sampled = wa * Ia + wb * Ib + wc * Ic + wd * Id

        return sampled


def colorize_sparse_points(
    points_3d: np.ndarray,
    point_observations: List[Dict],
    images: List[np.ndarray],
    camera_poses: Dict[int, CameraPose]
) -> np.ndarray:
    """
    Colorize sparse 3D points using their original observations

    This is simpler than full colorization as we know which images
    see each point.

    Args:
        points_3d: Nx3 3D points
        point_observations: List of N dicts {img_idx: (x, y)}
        images: List of images
        camera_poses: Dict of camera poses

    Returns:
        Nx3 RGB colors (0-255)
    """
    n_points = len(points_3d)
    colors = np.zeros((n_points, 3), dtype=np.float32)
    counts = np.zeros(n_points, dtype=np.int32)

    for pt_idx, obs_dict in enumerate(point_observations):
        for img_idx, pt_2d in obs_dict.items():
            if img_idx >= len(images):
                continue

            image = images[img_idx]
            x, y = pt_2d

            # Convert to integer coordinates
            x_int = int(np.round(x))
            y_int = int(np.round(y))

            # Check bounds
            if 0 <= x_int < image.shape[1] and 0 <= y_int < image.shape[0]:
                colors[pt_idx] += image[y_int, x_int].astype(np.float32)
                counts[pt_idx] += 1

    # Average colors
    valid_mask = counts > 0
    colors[valid_mask] /= counts[valid_mask, np.newaxis]

    # For points with no valid observations, use gray
    colors[~valid_mask] = 128.0

    logger.info(f"Colored {np.sum(valid_mask)}/{n_points} sparse points")

    return colors.astype(np.uint8)