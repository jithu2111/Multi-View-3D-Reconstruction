"""
Two-View Initialization Module

Implements initialization of the reconstruction from two views:
- Image pair selection based on baseline and matches
- Fundamental matrix decomposition
- Essential matrix computation
- Camera pose recovery (R, t)
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from ..features.sift_detector import FeatureMatch
from .ransac import RANSACResult
from ..utils.logger import logger


@dataclass
class CameraPose:
    """Container for camera pose (rotation and translation)"""
    R: np.ndarray  # 3x3 rotation matrix
    t: np.ndarray  # 3x1 translation vector

    def to_projection_matrix(self, K: np.ndarray) -> np.ndarray:
        """
        Convert to 3x4 projection matrix P = K[R|t]

        Args:
            K: 3x3 camera intrinsic matrix

        Returns:
            3x4 projection matrix
        """
        Rt = np.hstack([self.R, self.t.reshape(3, 1)])
        return K @ Rt


@dataclass
class TwoViewReconstruction:
    """Container for two-view reconstruction results"""
    img1_idx: int
    img2_idx: int
    K: np.ndarray  # 3x3 intrinsic matrix
    pose1: CameraPose  # First camera pose (typically identity)
    pose2: CameraPose  # Second camera pose
    inlier_points1: np.ndarray  # Nx2 inlier points from image 1
    inlier_points2: np.ndarray  # Nx2 inlier points from image 2
    F: np.ndarray  # 3x3 fundamental matrix
    E: np.ndarray  # 3x3 essential matrix


class TwoViewInitializer:
    """
    Two-view initialization for Structure from Motion

    Selects the best image pair and recovers camera poses to initialize
    the reconstruction.
    """

    def __init__(
        self,
        focal_length: Optional[float] = None,
        min_baseline_ratio: float = 0.1,
        min_triangulation_angle: float = 2.0
    ):
        """
        Initialize two-view reconstructor

        Args:
            focal_length: Focal length in pixels (if None, will be estimated)
            min_baseline_ratio: Minimum baseline/scene_depth ratio for valid pair
            min_triangulation_angle: Minimum angle (degrees) for valid triangulation
        """
        self.focal_length = focal_length
        self.min_baseline_ratio = min_baseline_ratio
        self.min_triangulation_angle = min_triangulation_angle

        logger.info(f"Initialized TwoViewInitializer: "
                   f"focal_length={focal_length}, "
                   f"min_baseline_ratio={min_baseline_ratio}")

    def select_initial_pair(
        self,
        matches_dict: Dict[Tuple[int, int], FeatureMatch],
        ransac_results: Dict[Tuple[int, int], RANSACResult],
        image_shape: Tuple[int, int]
    ) -> Tuple[int, int]:
        """
        Select the best image pair for initialization

        Args:
            matches_dict: Dictionary of feature matches
            ransac_results: Dictionary of RANSAC verification results
            image_shape: (height, width) of images

        Returns:
            Tuple of (img1_idx, img2_idx) for best pair
        """
        best_pair = None
        best_score = -1

        logger.info("Selecting initial image pair...")

        for pair, ransac_result in ransac_results.items():
            if ransac_result.n_inliers < 100:
                continue

            # Score based on number of inliers and distribution
            match = matches_dict[pair]
            score = self._score_image_pair(
                match, ransac_result, image_shape
            )

            if score > best_score:
                best_score = score
                best_pair = pair

        if best_pair is None:
            raise ValueError("No valid image pair found for initialization")

        logger.info(f"Selected initial pair: {best_pair} with score {best_score:.2f}")
        return best_pair

    def _score_image_pair(
        self,
        match: FeatureMatch,
        ransac_result: RANSACResult,
        image_shape: Tuple[int, int]
    ) -> float:
        """
        Score an image pair based on quality metrics

        Args:
            match: Feature matches
            ransac_result: RANSAC result
            image_shape: Image dimensions

        Returns:
            Quality score (higher is better)
        """
        # Number of inliers
        n_inliers = ransac_result.n_inliers

        # Spatial distribution of matches
        pts = ransac_result.inlier_points1
        if len(pts) == 0:
            return 0.0

        # Check coverage across image
        h, w = image_shape
        x_coverage = (pts[:, 0].max() - pts[:, 0].min()) / w
        y_coverage = (pts[:, 1].max() - pts[:, 1].min()) / h
        coverage = (x_coverage + y_coverage) / 2.0

        # Combine metrics
        score = n_inliers * coverage * ransac_result.inlier_ratio
        return score

    def estimate_camera_intrinsics(
        self,
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Estimate camera intrinsic matrix K

        Args:
            image_shape: (height, width) of images

        Returns:
            3x3 camera intrinsic matrix
        """
        h, w = image_shape

        if self.focal_length is None:
            # Estimate focal length as 1.2 * max(w, h) (common heuristic)
            f = 1.2 * max(w, h)
        else:
            f = self.focal_length

        # Principal point at image center
        cx = w / 2.0
        cy = h / 2.0

        K = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        logger.info(f"Estimated camera intrinsics: f={f:.1f}, "
                   f"cx={cx:.1f}, cy={cy:.1f}")

        return K

    def initialize_reconstruction(
        self,
        match: FeatureMatch,
        ransac_result: RANSACResult,
        image_shape: Tuple[int, int]
    ) -> TwoViewReconstruction:
        """
        Initialize reconstruction from two views

        Args:
            match: Feature matches between the two images
            ransac_result: RANSAC verification result
            image_shape: (height, width) of images

        Returns:
            TwoViewReconstruction object with camera poses
        """
        logger.info(f"Initializing reconstruction from images "
                   f"{match.img1_idx} and {match.img2_idx}")

        # Estimate camera intrinsics
        K = self.estimate_camera_intrinsics(image_shape)

        # Get fundamental matrix from RANSAC
        F = ransac_result.matrix

        # Compute essential matrix: E = K^T * F * K
        E = K.T @ F @ K

        # Recover camera poses from essential matrix
        pose1, pose2, mask = self._recover_pose(
            E,
            ransac_result.inlier_points1,
            ransac_result.inlier_points2,
            K
        )
        
        # Filter points that failed cheirality check during pose recovery
        inlier_points1 = ransac_result.inlier_points1[mask]
        inlier_points2 = ransac_result.inlier_points2[mask]

        logger.info(f"Successfully initialized two-view reconstruction with "
                   f"{len(inlier_points1)} valid pose inliers")

        return TwoViewReconstruction(
            img1_idx=match.img1_idx,
            img2_idx=match.img2_idx,
            K=K,
            pose1=pose1,
            pose2=pose2,
            inlier_points1=inlier_points1,
            inlier_points2=inlier_points2,
            F=F,
            E=E
        )

    def _recover_pose(
        self,
        E: np.ndarray,
        pts1: np.ndarray,
        pts2: np.ndarray,
        K: np.ndarray
    ) -> Tuple[CameraPose, CameraPose, np.ndarray]:
        """
        Recover camera poses from essential matrix

        Args:
            E: 3x3 essential matrix
            pts1: Nx2 points from image 1
            pts2: Nx2 points from image 2
            K: 3x3 camera intrinsic matrix

        Returns:
            Tuple of (pose1, pose2, mask) where pose1 is at origin and mask is the inlier mask
        """
        # First camera at origin
        R1 = np.eye(3)
        t1 = np.zeros(3)
        pose1 = CameraPose(R1, t1)

        # Recover second camera pose
        # OpenCV's recoverPose automatically selects the correct solution
        # among the 4 possible decompositions of E
        _, R2, t2, mask = cv2.recoverPose(E, pts1, pts2, K)

        pose2 = CameraPose(R2, t2.ravel())

        logger.debug(f"Recovered pose: R2 determinant = {np.linalg.det(R2):.6f}")
        logger.debug(f"Recovered pose: t2 = {t2.ravel()}")

        return pose1, pose2, mask.ravel().astype(bool)

    def compute_baseline(self, pose2: CameraPose) -> float:
        """
        Compute baseline distance between cameras

        Args:
            pose2: Second camera pose

        Returns:
            Baseline distance (norm of translation)
        """
        return np.linalg.norm(pose2.t)

    def decompose_essential_matrix(
        self,
        E: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose essential matrix into rotation and translation

        Args:
            E: 3x3 essential matrix

        Returns:
            Tuple of (R, t) - there are actually 4 solutions,
            this returns one of them
        """
        # SVD decomposition
        U, S, Vt = np.linalg.svd(E)

        # Ensure proper rotation matrix (det = +1)
        if np.linalg.det(U) < 0:
            U *= -1
        if np.linalg.det(Vt) < 0:
            Vt *= -1

        # W matrix for extracting rotation
        W = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])

        # Two possible rotations
        R1 = U @ W @ Vt
        R2 = U @ W.T @ Vt

        # Translation (up to scale)
        t = U[:, 2]

        # Return one solution (use recoverPose for correct one)
        return R1, t