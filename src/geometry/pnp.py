"""
Perspective-n-Point (PnP) Camera Registration

Implements incremental camera registration for Structure from Motion:
- PnP solver for estimating camera pose from 2D-3D correspondences
- RANSAC-based robust pose estimation
- New camera integration into existing reconstruction
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass
from .two_view import CameraPose
from ..utils.logger import logger


@dataclass
class PnPResult:
    """Container for PnP pose estimation result"""
    success: bool
    pose: Optional[CameraPose]
    inlier_mask: np.ndarray
    n_inliers: int
    inlier_ratio: float
    reprojection_error: float


class PnPSolver:
    """
    Perspective-n-Point solver for camera pose estimation

    Estimates camera pose from known 2D-3D correspondences using
    RANSAC for robustness against outliers.
    """

    def __init__(
        self,
        ransac_threshold: float = 2.5,
        confidence: float = 0.99,
        max_iterations: int = 1000,
        min_inliers: int = 30
    ):
        """
        Initialize PnP solver

        Args:
            ransac_threshold: Maximum reprojection error for inliers (pixels)
            confidence: RANSAC confidence level (0-1)
            max_iterations: Maximum RANSAC iterations
            min_inliers: Minimum number of inliers for valid pose
        """
        self.ransac_threshold = ransac_threshold
        self.confidence = confidence
        self.max_iterations = max_iterations
        self.min_inliers = min_inliers

        logger.info(f"Initialized PnPSolver: "
                   f"threshold={ransac_threshold}px, "
                   f"min_inliers={min_inliers}")

    def estimate_pose(
        self,
        points_2d: np.ndarray,
        points_3d: np.ndarray,
        K: np.ndarray
    ) -> PnPResult:
        """
        Estimate camera pose from 2D-3D correspondences

        Args:
            points_2d: Nx2 array of 2D image points
            points_3d: Nx3 array of 3D world points
            K: 3x3 camera intrinsic matrix

        Returns:
            PnPResult with estimated pose and inlier information
        """
        if len(points_2d) < 4:
            logger.warning(f"Too few points for PnP: {len(points_2d)} < 4")
            return self._create_failed_result(len(points_2d))

        # Solve PnP with RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            points_3d,
            points_2d,
            K,
            distCoeffs=None,
            reprojectionError=self.ransac_threshold,
            confidence=self.confidence,
            iterationsCount=self.max_iterations,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success or inliers is None or len(inliers) < self.min_inliers:
            n_inliers = len(inliers) if inliers is not None else 0
            logger.warning(f"PnP failed or insufficient inliers: {n_inliers}")
            return self._create_failed_result(len(points_2d))

        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.ravel()

        # Create inlier mask
        inlier_mask = np.zeros(len(points_2d), dtype=bool)
        inlier_mask[inliers.ravel()] = True

        n_inliers = len(inliers)
        inlier_ratio = n_inliers / len(points_2d)

        # Compute reprojection error for inliers
        reproj_error = self._compute_reprojection_error(
            points_3d[inlier_mask],
            points_2d[inlier_mask],
            R, t, K
        )

        logger.info(f"PnP: {n_inliers}/{len(points_2d)} inliers "
                   f"({inlier_ratio*100:.1f}%), "
                   f"reproj_error={reproj_error:.3f}px")

        return PnPResult(
            success=True,
            pose=CameraPose(R, t),
            inlier_mask=inlier_mask,
            n_inliers=n_inliers,
            inlier_ratio=inlier_ratio,
            reprojection_error=reproj_error
        )

    def refine_pose(
        self,
        pose: CameraPose,
        points_2d: np.ndarray,
        points_3d: np.ndarray,
        K: np.ndarray
    ) -> CameraPose:
        """
        Refine camera pose using all inliers (non-RANSAC)

        Args:
            pose: Initial camera pose
            points_2d: Nx2 array of 2D image points
            points_3d: Nx3 array of 3D world points
            K: 3x3 camera intrinsic matrix

        Returns:
            Refined CameraPose
        """
        # Convert R to rotation vector
        rvec, _ = cv2.Rodrigues(pose.R)
        tvec = pose.t.reshape(3, 1)

        # Refine using all points (no RANSAC)
        success, rvec, tvec = cv2.solvePnP(
            points_3d,
            points_2d,
            K,
            distCoeffs=None,
            rvec=rvec,
            tvec=tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            logger.warning("Pose refinement failed, returning original pose")
            return pose

        R, _ = cv2.Rodrigues(rvec)
        t = tvec.ravel()

        return CameraPose(R, t)

    def _compute_reprojection_error(
        self,
        points_3d: np.ndarray,
        points_2d: np.ndarray,
        R: np.ndarray,
        t: np.ndarray,
        K: np.ndarray
    ) -> float:
        """
        Compute mean reprojection error

        Args:
            points_3d: Nx3 array of 3D points
            points_2d: Nx2 array of 2D points
            R: 3x3 rotation matrix
            t: 3x1 translation vector
            K: 3x3 intrinsic matrix

        Returns:
            Mean reprojection error in pixels
        """
        # Project 3D points to image
        rvec, _ = cv2.Rodrigues(R)
        projected, _ = cv2.projectPoints(
            points_3d,
            rvec,
            t.reshape(3, 1),
            K,
            distCoeffs=None
        )
        projected = projected.reshape(-1, 2)

        # Compute errors
        errors = np.linalg.norm(projected - points_2d, axis=1)
        return errors.mean()

    def _create_failed_result(self, n_points: int) -> PnPResult:
        """Create PnPResult for failed pose estimation"""
        return PnPResult(
            success=False,
            pose=None,
            inlier_mask=np.zeros(n_points, dtype=bool),
            n_inliers=0,
            inlier_ratio=0.0,
            reprojection_error=float('inf')
        )


class IncrementalReconstructor:
    """
    Manages incremental reconstruction by registering cameras sequentially

    Maintains the reconstruction state and coordinates camera registration
    with the existing 3D structure.
    """

    def __init__(
        self,
        K: np.ndarray,
        pnp_solver: Optional[PnPSolver] = None
    ):
        """
        Initialize incremental reconstructor

        Args:
            K: 3x3 camera intrinsic matrix
            pnp_solver: PnP solver (creates default if None)
        """
        self.K = K
        self.pnp_solver = pnp_solver or PnPSolver()

        # Reconstruction state
        self.registered_cameras: Dict[int, CameraPose] = {}
        self.points_3d: np.ndarray = np.empty((0, 3))
        self.point_observations: List[Dict] = []  # Track which images see each point
        self.point_colors: np.ndarray = np.empty((0, 3))

        logger.info("Initialized IncrementalReconstructor")

    def initialize_from_two_view(
        self,
        img1_idx: int,
        img2_idx: int,
        pose1: CameraPose,
        pose2: CameraPose,
        points_3d: np.ndarray,
        points_2d_img1: np.ndarray,
        points_2d_img2: np.ndarray
    ):
        """
        Initialize reconstruction from two-view result

        Args:
            img1_idx: First image index
            img2_idx: Second image index
            pose1: First camera pose
            pose2: Second camera pose
            points_3d: Nx3 triangulated points
            points_2d_img1: Nx2 points in image 1
            points_2d_img2: Nx2 points in image 2
        """
        # Register initial cameras
        self.registered_cameras[img1_idx] = pose1
        self.registered_cameras[img2_idx] = pose2

        # Store 3D points
        self.points_3d = points_3d.copy()

        # Initialize point observations
        self.point_observations = []
        for i in range(len(points_3d)):
            obs = {
                img1_idx: points_2d_img1[i],
                img2_idx: points_2d_img2[i]
            }
            self.point_observations.append(obs)

        # Initialize colors (placeholder - will be set later)
        self.point_colors = np.zeros((len(points_3d), 3), dtype=np.uint8)

        logger.info(f"Initialized reconstruction with {len(points_3d)} points "
                   f"and cameras {img1_idx}, {img2_idx}")

    def register_camera(
        self,
        img_idx: int,
        points_2d: np.ndarray,
        point_indices: np.ndarray
    ) -> PnPResult:
        """
        Register a new camera using existing 3D points

        Args:
            img_idx: Image index to register
            points_2d: Nx2 observed 2D points in the new image
            point_indices: N indices into self.points_3d for correspondences

        Returns:
            PnPResult with estimated camera pose
        """
        if img_idx in self.registered_cameras:
            logger.warning(f"Camera {img_idx} already registered")
            return PnPResult(
                success=False,
                pose=self.registered_cameras[img_idx],
                inlier_mask=np.zeros(len(points_2d), dtype=bool),
                n_inliers=0,
                inlier_ratio=0.0,
                reprojection_error=float('inf')
            )

        # Get corresponding 3D points
        points_3d = self.points_3d[point_indices]

        # Estimate camera pose
        result = self.pnp_solver.estimate_pose(points_2d, points_3d, self.K)

        if result.success:
            # Register camera
            self.registered_cameras[img_idx] = result.pose

            # Update point observations
            inlier_point_indices = point_indices[result.inlier_mask]
            inlier_points_2d = points_2d[result.inlier_mask]

            for pt_idx, pt_2d in zip(inlier_point_indices, inlier_points_2d):
                if pt_idx < len(self.point_observations):
                    self.point_observations[pt_idx][img_idx] = pt_2d

            logger.info(f"Registered camera {img_idx}, "
                       f"total cameras: {len(self.registered_cameras)}")

        return result

    def get_registered_camera_indices(self) -> List[int]:
        """Get list of registered camera indices"""
        return sorted(self.registered_cameras.keys())

    def get_camera_pose(self, img_idx: int) -> Optional[CameraPose]:
        """Get pose for a registered camera"""
        return self.registered_cameras.get(img_idx)

    def get_reconstruction_size(self) -> Tuple[int, int]:
        """Get number of cameras and points in reconstruction"""
        return len(self.registered_cameras), len(self.points_3d)

    def add_points(
        self,
        new_points_3d: np.ndarray,
        observations: List[Dict]
    ):
        """
        Add new 3D points to reconstruction

        Args:
            new_points_3d: Mx3 array of new 3D points
            observations: List of M dicts mapping img_idx -> 2D point
        """
        if len(new_points_3d) == 0:
            return

        self.points_3d = np.vstack([self.points_3d, new_points_3d])
        self.point_observations.extend(observations)
        new_colors = np.zeros((len(new_points_3d), 3), dtype=np.uint8)
        self.point_colors = np.vstack([self.point_colors, new_colors])

        logger.info(f"Added {len(new_points_3d)} new points, "
                   f"total: {len(self.points_3d)}")