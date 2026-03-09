"""
Triangulation Module

Implements 3D point triangulation from 2D correspondences:
- DLT (Direct Linear Transform) triangulation
- Cheirality check (points in front of cameras)
- Reprojection error computation
"""

import cv2
import numpy as np
from typing import Tuple, List
from dataclasses import dataclass
from .two_view import CameraPose, TwoViewReconstruction
from ..utils.logger import logger


@dataclass
class TriangulatedPoints:
    """Container for triangulated 3D points"""
    points_3d: np.ndarray  # Nx3 array of 3D points
    points_2d_img1: np.ndarray  # Nx2 2D points in image 1
    points_2d_img2: np.ndarray  # Nx2 2D points in image 2
    reprojection_errors: np.ndarray  # N array of reprojection errors
    valid_mask: np.ndarray  # N boolean array indicating valid points
    img1_idx: int
    img2_idx: int


class Triangulator:
    """
    3D point triangulation from 2D correspondences

    Uses DLT (Direct Linear Transform) method with cheirality checks
    to ensure triangulated points are in front of both cameras.
    """

    def __init__(
        self,
        min_parallax: float = 1.0,
        max_reproj_error: float = 4.0,
        min_depth: float = 0.0
    ):
        """
        Initialize triangulator

        Args:
            min_parallax: Minimum parallax angle (degrees) for valid triangulation
            max_reproj_error: Maximum reprojection error (pixels) for valid points
            min_depth: Minimum depth for points (reject points behind camera)
        """
        self.min_parallax = min_parallax
        self.max_reproj_error = max_reproj_error
        self.min_depth = min_depth

        logger.info(f"Initialized Triangulator: "
                   f"min_parallax={min_parallax}°, "
                   f"max_reproj_error={max_reproj_error}px")

    def triangulate(
        self,
        reconstruction: TwoViewReconstruction
    ) -> TriangulatedPoints:
        """
        Triangulate 3D points from two-view reconstruction

        Args:
            reconstruction: TwoViewReconstruction with camera poses

        Returns:
            TriangulatedPoints with 3D coordinates and validation info
        """
        logger.info(f"Triangulating points from images "
                   f"{reconstruction.img1_idx} and {reconstruction.img2_idx}")

        # Get projection matrices
        P1 = reconstruction.pose1.to_projection_matrix(reconstruction.K)
        P2 = reconstruction.pose2.to_projection_matrix(reconstruction.K)

        # Get 2D points
        pts1 = reconstruction.inlier_points1
        pts2 = reconstruction.inlier_points2

        # Triangulate using OpenCV
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

        # Convert from homogeneous to 3D coordinates
        w = points_4d[3, :]
        # Prevent division by zero for points at infinity
        w_safe = np.where(np.abs(w) < 1e-7, 1e-7 * np.sign(w + 1e-10), w)
        points_3d = points_4d[:3, :] / w_safe
        points_3d = points_3d.T  # Nx3

        # Validate triangulated points
        valid_mask = self._validate_points(
            points_3d, pts1, pts2, P1, P2,
            reconstruction.pose1, reconstruction.pose2
        )

        # Compute reprojection errors
        reproj_errors = self._compute_reprojection_errors(
            points_3d, pts1, pts2, P1, P2
        )

        n_valid = np.sum(valid_mask)
        n_total = len(points_3d)

        logger.info(f"Triangulated {n_total} points, "
                   f"{n_valid} valid ({n_valid/n_total*100:.1f}%)")
        logger.info(f"Mean reprojection error: "
                   f"{reproj_errors[valid_mask].mean():.3f} pixels")

        return TriangulatedPoints(
            points_3d=points_3d,
            points_2d_img1=pts1,
            points_2d_img2=pts2,
            reprojection_errors=reproj_errors,
            valid_mask=valid_mask,
            img1_idx=reconstruction.img1_idx,
            img2_idx=reconstruction.img2_idx
        )

    def _validate_points(
        self,
        points_3d: np.ndarray,
        pts1: np.ndarray,
        pts2: np.ndarray,
        P1: np.ndarray,
        P2: np.ndarray,
        pose1: CameraPose,
        pose2: CameraPose
    ) -> np.ndarray:
        """
        Validate triangulated points using multiple criteria

        Args:
            points_3d: Nx3 triangulated points
            pts1: Nx2 points in image 1
            pts2: Nx2 points in image 2
            P1: 3x4 projection matrix for camera 1
            P2: 3x4 projection matrix for camera 2
            pose1: Camera 1 pose
            pose2: Camera 2 pose

        Returns:
            Boolean mask of valid points
        """
        n_points = len(points_3d)
        valid = np.ones(n_points, dtype=bool)

        # 1. Cheirality check: points must be in front of both cameras
        depths1 = self._compute_depths(points_3d, pose1)
        depths2 = self._compute_depths(points_3d, pose2)

        valid &= (depths1 > self.min_depth)
        valid &= (depths2 > self.min_depth)

        # 2. Reprojection error check
        reproj_errors = self._compute_reprojection_errors(
            points_3d, pts1, pts2, P1, P2
        )
        valid &= (reproj_errors < self.max_reproj_error)

        # 3. Parallax check (sufficient viewing angle)
        if self.min_parallax > 0:
            parallax_angles = self._compute_parallax(
                points_3d, pose1, pose2
            )
            valid &= (parallax_angles > self.min_parallax)

        return valid

    def _compute_depths(
        self,
        points_3d: np.ndarray,
        pose: CameraPose
    ) -> np.ndarray:
        """
        Compute depths of 3D points with respect to camera

        Args:
            points_3d: Nx3 array of 3D points
            pose: Camera pose

        Returns:
            N array of depths
        """
        # Transform points to camera coordinate system
        points_cam = (pose.R @ points_3d.T).T + pose.t

        # Depth is the Z coordinate in camera frame
        depths = points_cam[:, 2]

        return depths

    def _compute_reprojection_errors(
        self,
        points_3d: np.ndarray,
        pts1: np.ndarray,
        pts2: np.ndarray,
        P1: np.ndarray,
        P2: np.ndarray
    ) -> np.ndarray:
        """
        Compute reprojection errors for triangulated points

        Args:
            points_3d: Nx3 triangulated points
            pts1: Nx2 observed points in image 1
            pts2: Nx2 observed points in image 2
            P1: 3x4 projection matrix for camera 1
            P2: 3x4 projection matrix for camera 2

        Returns:
            N array of reprojection errors
        """
        # Convert to homogeneous coordinates
        points_4d = np.hstack([points_3d, np.ones((len(points_3d), 1))])

        # Project to image 1
        proj1 = (P1 @ points_4d.T).T
        proj1 = proj1[:, :2] / proj1[:, 2:3]
        error1 = np.linalg.norm(proj1 - pts1, axis=1)

        # Project to image 2
        proj2 = (P2 @ points_4d.T).T
        proj2 = proj2[:, :2] / proj2[:, 2:3]
        error2 = np.linalg.norm(proj2 - pts2, axis=1)

        # Average error across both views
        errors = (error1 + error2) / 2.0

        return errors

    def _compute_parallax(
        self,
        points_3d: np.ndarray,
        pose1: CameraPose,
        pose2: CameraPose
    ) -> np.ndarray:
        """
        Compute parallax angles for triangulated points

        Args:
            points_3d: Nx3 triangulated points
            pose1: First camera pose
            pose2: Second camera pose

        Returns:
            N array of parallax angles in degrees
        """
        # Camera centers
        C1 = -pose1.R.T @ pose1.t
        C2 = -pose2.R.T @ pose2.t

        # Vectors from each camera to points
        v1 = points_3d - C1
        v2 = points_3d - C2

        # Normalize
        v1_norm = v1 / np.linalg.norm(v1, axis=1, keepdims=True)
        v2_norm = v2 / np.linalg.norm(v2, axis=1, keepdims=True)

        # Compute angle between viewing rays
        cos_angles = np.sum(v1_norm * v2_norm, axis=1)
        cos_angles = np.clip(cos_angles, -1.0, 1.0)
        angles = np.arccos(cos_angles)

        # Convert to degrees
        angles_deg = np.degrees(angles)

        return angles_deg

    def triangulate_points_dlt(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        P1: np.ndarray,
        P2: np.ndarray
    ) -> np.ndarray:
        """
        Triangulate points using Direct Linear Transform (DLT)

        This is a manual implementation for educational purposes.
        In practice, cv2.triangulatePoints is more efficient.

        Args:
            pts1: Nx2 points in image 1
            pts2: Nx2 points in image 2
            P1: 3x4 projection matrix for camera 1
            P2: 3x4 projection matrix for camera 2

        Returns:
            Nx3 array of triangulated 3D points
        """
        n_points = len(pts1)
        points_3d = np.zeros((n_points, 3))

        for i in range(n_points):
            # Build matrix A for DLT
            x1, y1 = pts1[i]
            x2, y2 = pts2[i]

            A = np.array([
                x1 * P1[2] - P1[0],
                y1 * P1[2] - P1[1],
                x2 * P2[2] - P2[0],
                y2 * P2[2] - P2[1]
            ])

            # Solve using SVD
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1]

            # Convert from homogeneous to 3D
            points_3d[i] = X[:3] / X[3]

        return points_3d

    def filter_valid_points(
        self,
        triangulated: TriangulatedPoints
    ) -> TriangulatedPoints:
        """
        Filter to keep only valid triangulated points

        Args:
            triangulated: TriangulatedPoints with validation info

        Returns:
            New TriangulatedPoints with only valid points
        """
        mask = triangulated.valid_mask

        return TriangulatedPoints(
            points_3d=triangulated.points_3d[mask],
            points_2d_img1=triangulated.points_2d_img1[mask],
            points_2d_img2=triangulated.points_2d_img2[mask],
            reprojection_errors=triangulated.reprojection_errors[mask],
            valid_mask=np.ones(np.sum(mask), dtype=bool),
            img1_idx=triangulated.img1_idx,
            img2_idx=triangulated.img2_idx
        )