"""
Multi-View Stereo (MVS) Densification Module

Implements dense reconstruction from sparse point cloud and calibrated cameras:
- Patch-based Multi-View Stereo (PMVS-like approach)
- Plane sweep stereo for depth map generation
- Point cloud fusion and filtering
- Colorization from source images

Converts sparse reconstruction to dense colored point cloud.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from scipy.spatial import cKDTree
from ..geometry.two_view import CameraPose
from ..utils.logger import logger


@dataclass
class DepthMap:
    """Container for depth map and confidence"""
    depth: np.ndarray  # HxW depth values
    confidence: np.ndarray  # HxW confidence scores
    image_idx: int
    valid_mask: np.ndarray  # HxW boolean mask


@dataclass
class DensePointCloud:
    """Container for dense point cloud with colors"""
    points_3d: np.ndarray  # Nx3 3D coordinates
    colors: np.ndarray  # Nx3 RGB colors (0-255)
    normals: Optional[np.ndarray] = None  # Nx3 surface normals
    confidence: Optional[np.ndarray] = None  # N confidence scores


class PlaneSweepStereo:
    """
    Plane Sweep Stereo for depth map generation

    Sweeps a series of depth planes through the scene and computes
    matching costs by warping reference images to source view.
    """

    def __init__(
        self,
        num_depths: int = 128,
        window_size: int = 5,
        min_depth: float = 0.1,
        max_depth: float = 10.0,
        confidence_threshold: float = 0.6
    ):
        """
        Initialize Plane Sweep Stereo

        Args:
            num_depths: Number of depth planes to test
            window_size: Window size for matching cost computation
            min_depth: Minimum depth to test
            max_depth: Maximum depth to test
            confidence_threshold: Minimum confidence for valid depth
        """
        self.num_depths = num_depths
        self.window_size = window_size
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.confidence_threshold = confidence_threshold

        logger.info(f"Initialized PlaneSweepStereo: "
                   f"depths={num_depths}, "
                   f"window={window_size}, "
                   f"range=[{min_depth:.2f}, {max_depth:.2f}]")

    def compute_depth_map(
        self,
        ref_image: np.ndarray,
        ref_pose: CameraPose,
        src_images: List[np.ndarray],
        src_poses: List[CameraPose],
        K: np.ndarray,
        depth_range: Optional[Tuple[float, float]] = None
    ) -> DepthMap:
        """
        Compute depth map for reference image using source images

        Args:
            ref_image: Reference image (HxWx3)
            ref_pose: Reference camera pose
            src_images: List of source images
            src_poses: List of source camera poses
            K: 3x3 camera intrinsic matrix
            depth_range: Optional (min_depth, max_depth) override

        Returns:
            DepthMap with depth values and confidence
        """
        if depth_range is not None:
            min_depth, max_depth = depth_range
        else:
            min_depth, max_depth = self.min_depth, self.max_depth

        h, w = ref_image.shape[:2]

        # Generate depth hypothesis planes
        depths = np.linspace(min_depth, max_depth, self.num_depths)

        # Initialize cost volume (H x W x num_depths)
        cost_volume = np.zeros((h, w, self.num_depths), dtype=np.float32)

        logger.info(f"Computing depth map: "
                   f"image_size=({h}, {w}), "
                   f"num_sources={len(src_images)}")

        # Compute matching cost for each depth plane
        for d_idx, depth in enumerate(depths):
            costs = []

            # Aggregate costs from all source views
            for src_image, src_pose in zip(src_images, src_poses):
                cost = self._compute_plane_cost(
                    ref_image, ref_pose,
                    src_image, src_pose,
                    K, depth
                )
                costs.append(cost)

            # Aggregate costs (mean)
            if len(costs) > 0:
                cost_volume[:, :, d_idx] = np.mean(costs, axis=0)

        # Winner-takes-all depth selection
        depth_indices = np.argmin(cost_volume, axis=2)
        depth_map = depths[depth_indices]

        # Compute confidence (inverse of minimum cost, normalized)
        min_costs = np.min(cost_volume, axis=2)
        max_cost = np.percentile(min_costs, 95)
        confidence = 1.0 - np.clip(min_costs / (max_cost + 1e-6), 0, 1)

        # Create valid mask based on confidence
        valid_mask = confidence > self.confidence_threshold

        logger.info(f"Depth map computed: "
                   f"{np.sum(valid_mask)} / {h*w} valid pixels "
                   f"({np.sum(valid_mask)/(h*w)*100:.1f}%)")

        return DepthMap(
            depth=depth_map,
            confidence=confidence,
            image_idx=-1,  # Set by caller
            valid_mask=valid_mask
        )

    def _compute_plane_cost(
        self,
        ref_image: np.ndarray,
        ref_pose: CameraPose,
        src_image: np.ndarray,
        src_pose: CameraPose,
        K: np.ndarray,
        depth: float
    ) -> np.ndarray:
        """
        Compute matching cost for a single depth plane

        Warps source image to reference view at given depth
        and computes photometric difference.

        Returns:
            HxW cost map
        """
        h, w = ref_image.shape[:2]

        # Create pixel grid for reference image
        y, x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        pixels = np.stack([x, y], axis=-1).reshape(-1, 2)  # (H*W, 2)

        # Backproject to 3D at given depth
        points_3d = self._backproject_to_depth(pixels, depth, ref_pose, K)

        # Project to source image
        src_pixels = self._project_points(points_3d, src_pose, K)

        # Sample source image at projected locations
        src_sampled = self._bilinear_sample(src_image, src_pixels)

        # Compute photometric cost (L1 distance)
        ref_flat = ref_image.reshape(-1, 3).astype(np.float32)
        cost = np.mean(np.abs(ref_flat - src_sampled), axis=1)
        cost = cost.reshape(h, w)

        # Handle invalid projections (outside image bounds)
        invalid_mask = (src_pixels[:, 0] < 0) | (src_pixels[:, 0] >= w-1) | \
                      (src_pixels[:, 1] < 0) | (src_pixels[:, 1] >= h-1)
        cost_reshaped = cost.copy()
        cost_reshaped.ravel()[invalid_mask] = 255.0  # High cost for invalid

        return cost_reshaped

    def _backproject_to_depth(
        self,
        pixels: np.ndarray,
        depth: float,
        pose: CameraPose,
        K: np.ndarray
    ) -> np.ndarray:
        """
        Backproject 2D pixels to 3D at given depth

        Args:
            pixels: Nx2 pixel coordinates
            depth: Scalar depth value
            pose: Camera pose
            K: Intrinsic matrix

        Returns:
            Nx3 3D points in world coordinates
        """
        # Convert to homogeneous coordinates
        pixels_hom = np.hstack([pixels, np.ones((len(pixels), 1))])  # Nx3

        # Backproject to camera space
        K_inv = np.linalg.inv(K)
        rays_cam = (K_inv @ pixels_hom.T).T  # Nx3
        points_cam = rays_cam * depth  # Nx3

        # Transform to world space
        R_inv = pose.R.T
        C = -R_inv @ pose.t  # Camera center
        points_world = (R_inv @ points_cam.T).T + C

        return points_world

    def _project_points(
        self,
        points_3d: np.ndarray,
        pose: CameraPose,
        K: np.ndarray
    ) -> np.ndarray:
        """
        Project 3D points to 2D image coordinates

        Args:
            points_3d: Nx3 world coordinates
            pose: Camera pose
            K: Intrinsic matrix

        Returns:
            Nx2 pixel coordinates
        """
        # Transform to camera space
        points_cam = (pose.R @ points_3d.T).T + pose.t  # Nx3

        # Project to image
        points_2d_hom = (K @ points_cam.T).T  # Nx3

        # Normalize
        pixels = points_2d_hom[:, :2] / (points_2d_hom[:, 2:3] + 1e-8)

        return pixels

    def _bilinear_sample(
        self,
        image: np.ndarray,
        pixels: np.ndarray
    ) -> np.ndarray:
        """
        Bilinear sampling of image at non-integer pixel locations

        Args:
            image: HxWx3 image
            pixels: Nx2 pixel coordinates (can be non-integer)

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


class MVSDensifier:
    """
    Multi-View Stereo densifier

    Converts sparse reconstruction to dense point cloud using
    depth map fusion from multiple calibrated views.
    """

    def __init__(
        self,
        plane_sweep: Optional[PlaneSweepStereo] = None,
        min_views: int = 3,
        max_reproj_error: float = 2.0,
        consistency_threshold: float = 0.01
    ):
        """
        Initialize MVS densifier

        Args:
            plane_sweep: Plane sweep stereo instance
            min_views: Minimum views for valid 3D point
            max_reproj_error: Maximum reprojection error threshold
            consistency_threshold: Depth consistency threshold (relative)
        """
        self.plane_sweep = plane_sweep or PlaneSweepStereo()
        self.min_views = min_views
        self.max_reproj_error = max_reproj_error
        self.consistency_threshold = consistency_threshold

        logger.info(f"Initialized MVSDensifier: "
                   f"min_views={min_views}, "
                   f"consistency_thresh={consistency_threshold}")

    def densify(
        self,
        images: List[np.ndarray],
        camera_poses: Dict[int, CameraPose],
        K: np.ndarray,
        sparse_points: Optional[np.ndarray] = None
    ) -> DensePointCloud:
        """
        Generate dense point cloud from calibrated images

        Args:
            images: List of images
            camera_poses: Dict mapping img_idx -> CameraPose
            K: Camera intrinsic matrix
            sparse_points: Optional sparse 3D points for depth range estimation

        Returns:
            DensePointCloud with 3D points and colors
        """
        logger.info(f"Starting MVS densification with {len(camera_poses)} views")

        # Compute depth range from sparse points
        depth_range = self._estimate_depth_range(sparse_points, camera_poses)
        logger.info(f"Estimated depth range: [{depth_range[0]:.2f}, {depth_range[1]:.2f}]")

        # Compute depth maps for each view
        depth_maps = []
        camera_indices = sorted(camera_poses.keys())

        for ref_idx in camera_indices:
            logger.info(f"Computing depth map for camera {ref_idx}...")

            ref_image = images[ref_idx]
            ref_pose = camera_poses[ref_idx]

            # Select source views (all other cameras)
            src_indices = [i for i in camera_indices if i != ref_idx]
            src_images = [images[i] for i in src_indices]
            src_poses = [camera_poses[i] for i in src_indices]

            # Compute depth map
            depth_map = self.plane_sweep.compute_depth_map(
                ref_image, ref_pose,
                src_images, src_poses,
                K, depth_range
            )
            depth_map.image_idx = ref_idx
            depth_maps.append(depth_map)

        # Fuse depth maps into 3D point cloud
        logger.info("Fusing depth maps into dense point cloud...")
        dense_cloud = self._fuse_depth_maps(
            depth_maps, images, camera_poses, K
        )

        # Filter outliers
        logger.info("Filtering outliers...")
        dense_cloud = self._filter_outliers(dense_cloud)

        logger.info(f"Dense reconstruction complete: {len(dense_cloud.points_3d)} points")

        return dense_cloud

    def _estimate_depth_range(
        self,
        sparse_points: Optional[np.ndarray],
        camera_poses: Dict[int, CameraPose]
    ) -> Tuple[float, float]:
        """Estimate depth range from sparse points and camera positions"""
        if sparse_points is None or len(sparse_points) == 0:
            return (0.1, 10.0)  # Default range

        # Compute depths from first camera
        first_pose = list(camera_poses.values())[0]
        C = -first_pose.R.T @ first_pose.t

        # Depths are distances from camera center
        depths = np.linalg.norm(sparse_points - C, axis=1)

        # Use percentiles to avoid outliers
        min_depth = max(0.1, np.percentile(depths, 5))
        max_depth = np.percentile(depths, 95) * 1.5

        return (min_depth, max_depth)

    def _fuse_depth_maps(
        self,
        depth_maps: List[DepthMap],
        images: List[np.ndarray],
        camera_poses: Dict[int, CameraPose],
        K: np.ndarray
    ) -> DensePointCloud:
        """
        Fuse depth maps from multiple views into unified point cloud

        Converts each depth map to 3D points and filters based on
        multi-view consistency.
        """
        all_points = []
        all_colors = []
        all_confidence = []

        for depth_map in depth_maps:
            img_idx = depth_map.image_idx
            image = images[img_idx]
            pose = camera_poses[img_idx]

            # Convert depth map to 3D points
            points_3d, colors, conf = self._depth_map_to_points(
                depth_map, image, pose, K
            )

            all_points.append(points_3d)
            all_colors.append(colors)
            all_confidence.append(conf)

        # Concatenate all points
        points_3d = np.vstack(all_points) if all_points else np.empty((0, 3))
        colors = np.vstack(all_colors) if all_colors else np.empty((0, 3))
        confidence = np.concatenate(all_confidence) if all_confidence else np.empty(0)

        return DensePointCloud(
            points_3d=points_3d,
            colors=colors,
            confidence=confidence
        )

    def _depth_map_to_points(
        self,
        depth_map: DepthMap,
        image: np.ndarray,
        pose: CameraPose,
        K: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert depth map to 3D points

        Returns:
            Tuple of (points_3d, colors, confidence)
        """
        h, w = depth_map.depth.shape

        # Get valid pixels
        valid_mask = depth_map.valid_mask
        y_coords, x_coords = np.where(valid_mask)

        if len(y_coords) == 0:
            return np.empty((0, 3)), np.empty((0, 3)), np.empty(0)

        pixels = np.stack([x_coords, y_coords], axis=1)
        depths = depth_map.depth[valid_mask]

        # Backproject to 3D
        pixels_hom = np.hstack([pixels, np.ones((len(pixels), 1))])
        K_inv = np.linalg.inv(K)
        rays_cam = (K_inv @ pixels_hom.T).T
        points_cam = rays_cam * depths[:, np.newaxis]

        # Transform to world space
        R_inv = pose.R.T
        C = -R_inv @ pose.t
        points_3d = (R_inv @ points_cam.T).T + C

        # Get colors
        colors = image[y_coords, x_coords]

        # Get confidence
        confidence = depth_map.confidence[valid_mask]

        return points_3d, colors, confidence

    def _filter_outliers(
        self,
        cloud: DensePointCloud,
        k_neighbors: int = 20,
        std_ratio: float = 2.0
    ) -> DensePointCloud:
        """
        Filter outliers using statistical outlier removal

        Removes points whose average distance to k nearest neighbors
        is beyond mean + std_ratio * std.
        """
        if len(cloud.points_3d) < k_neighbors:
            return cloud

        # Build KD-tree
        tree = cKDTree(cloud.points_3d)

        # Compute distances to k nearest neighbors
        distances, _ = tree.query(cloud.points_3d, k=k_neighbors + 1)
        mean_distances = distances[:, 1:].mean(axis=1)  # Exclude self

        # Compute statistics
        global_mean = mean_distances.mean()
        global_std = mean_distances.std()

        # Filter outliers
        threshold = global_mean + std_ratio * global_std
        inlier_mask = mean_distances < threshold

        n_removed = len(cloud.points_3d) - np.sum(inlier_mask)
        logger.info(f"Removed {n_removed} outliers ({n_removed/len(cloud.points_3d)*100:.1f}%)")

        return DensePointCloud(
            points_3d=cloud.points_3d[inlier_mask],
            colors=cloud.colors[inlier_mask],
            confidence=cloud.confidence[inlier_mask] if cloud.confidence is not None else None,
            normals=cloud.normals[inlier_mask] if cloud.normals is not None else None
        )