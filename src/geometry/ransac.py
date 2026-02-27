"""
RANSAC-based Geometric Verification

Implements RANSAC (Random Sample Consensus) for robust estimation of
geometric relationships between views, filtering outliers from feature matches.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
from ..features.sift_detector import FeatureMatch
from ..utils.logger import logger


@dataclass
class RANSACResult:
    """Container for RANSAC verification results"""
    inlier_mask: np.ndarray  # Boolean mask of inliers
    matrix: np.ndarray  # Estimated matrix (F, E, or H)
    inlier_points1: np.ndarray  # Inlier keypoints from image 1
    inlier_points2: np.ndarray  # Inlier keypoints from image 2
    n_inliers: int  # Number of inliers
    inlier_ratio: float  # Ratio of inliers to total matches


class RANSACVerifier:
    """
    RANSAC-based geometric verification for feature matches

    Supports estimation of:
    - Fundamental matrix (uncalibrated cameras)
    - Essential matrix (calibrated cameras)
    - Homography matrix (planar scenes)
    """

    def __init__(
        self,
        method: str = 'fundamental',
        ransac_threshold: float = 1.0,
        confidence: float = 0.99,
        max_iterations: int = 2000,
        min_inliers: int = 50
    ):
        """
        Initialize RANSAC verifier

        Args:
            method: Geometric model to estimate ('fundamental', 'essential', 'homography')
            ransac_threshold: Maximum reprojection error for inliers (in pixels)
            confidence: Confidence level for RANSAC (0-1)
            max_iterations: Maximum number of RANSAC iterations
            min_inliers: Minimum number of inliers required for valid estimation
        """
        self.method = method.lower()
        self.ransac_threshold = ransac_threshold
        self.confidence = confidence
        self.max_iterations = max_iterations
        self.min_inliers = min_inliers

        if self.method not in ['fundamental', 'essential', 'homography']:
            raise ValueError(f"Unknown method: {method}. "
                           "Must be 'fundamental', 'essential', or 'homography'")

        logger.info(f"Initialized RANSAC verifier: method={method}, "
                   f"threshold={ransac_threshold}, min_inliers={min_inliers}")

    def verify_match(
        self,
        match: FeatureMatch,
        camera_matrix: Optional[np.ndarray] = None
    ) -> RANSACResult:
        """
        Verify feature matches using RANSAC

        Args:
            match: FeatureMatch object containing keypoint correspondences
            camera_matrix: Camera intrinsic matrix (required for essential matrix)

        Returns:
            RANSACResult containing inliers and estimated matrix
        """
        pts1 = match.keypoints1
        pts2 = match.keypoints2

        if len(pts1) < 8:
            logger.warning(f"Too few matches ({len(pts1)}) for RANSAC verification")
            return self._create_empty_result()

        # Estimate geometric relationship based on method
        if self.method == 'fundamental':
            matrix, mask = self._estimate_fundamental(pts1, pts2)
        elif self.method == 'essential':
            if camera_matrix is None:
                raise ValueError("Camera matrix required for essential matrix estimation")
            matrix, mask = self._estimate_essential(pts1, pts2, camera_matrix)
        else:  # homography
            matrix, mask = self._estimate_homography(pts1, pts2)

        if matrix is None or mask is None:
            logger.warning("RANSAC failed to estimate matrix")
            return self._create_empty_result()

        # Extract inliers
        mask = mask.ravel().astype(bool)
        inlier_pts1 = pts1[mask]
        inlier_pts2 = pts2[mask]
        n_inliers = np.sum(mask)
        inlier_ratio = n_inliers / len(pts1) if len(pts1) > 0 else 0.0

        logger.debug(f"RANSAC: {n_inliers}/{len(pts1)} inliers "
                    f"({inlier_ratio*100:.1f}%)")

        if n_inliers < self.min_inliers:
            logger.warning(f"Insufficient inliers: {n_inliers} < {self.min_inliers}")
            return self._create_empty_result()

        return RANSACResult(
            inlier_mask=mask,
            matrix=matrix,
            inlier_points1=inlier_pts1,
            inlier_points2=inlier_pts2,
            n_inliers=n_inliers,
            inlier_ratio=inlier_ratio
        )

    def _estimate_fundamental(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Estimate fundamental matrix using RANSAC

        Args:
            pts1: Nx2 array of points from image 1
            pts2: Nx2 array of points from image 2

        Returns:
            Fundamental matrix (3x3) and inlier mask
        """
        F, mask = cv2.findFundamentalMat(
            pts1,
            pts2,
            cv2.FM_RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
            confidence=self.confidence,
            maxIters=self.max_iterations
        )
        return F, mask

    def _estimate_essential(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        camera_matrix: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Estimate essential matrix using RANSAC

        Args:
            pts1: Nx2 array of points from image 1
            pts2: Nx2 array of points from image 2
            camera_matrix: 3x3 camera intrinsic matrix

        Returns:
            Essential matrix (3x3) and inlier mask
        """
        E, mask = cv2.findEssentialMat(
            pts1,
            pts2,
            camera_matrix,
            method=cv2.RANSAC,
            prob=self.confidence,
            threshold=self.ransac_threshold,
            maxIters=self.max_iterations
        )
        return E, mask

    def _estimate_homography(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Estimate homography matrix using RANSAC

        Args:
            pts1: Nx2 array of points from image 1
            pts2: Nx2 array of points from image 2

        Returns:
            Homography matrix (3x3) and inlier mask
        """
        H, mask = cv2.findHomography(
            pts1,
            pts2,
            cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
            confidence=self.confidence,
            maxIters=self.max_iterations
        )
        return H, mask

    def _create_empty_result(self) -> RANSACResult:
        """Create empty RANSACResult for failed verification"""
        return RANSACResult(
            inlier_mask=np.array([], dtype=bool),
            matrix=np.eye(3),
            inlier_points1=np.empty((0, 2), dtype=np.float32),
            inlier_points2=np.empty((0, 2), dtype=np.float32),
            n_inliers=0,
            inlier_ratio=0.0
        )

    def verify_all_matches(
        self,
        matches_dict: dict,
        camera_matrix: Optional[np.ndarray] = None
    ) -> dict:
        """
        Verify all feature matches using RANSAC

        Args:
            matches_dict: Dictionary of (img1_idx, img2_idx) -> FeatureMatch
            camera_matrix: Optional camera intrinsic matrix

        Returns:
            Dictionary of (img1_idx, img2_idx) -> RANSACResult
        """
        verified_matches = {}

        logger.info(f"Verifying {len(matches_dict)} match pairs with RANSAC...")

        for pair, match in matches_dict.items():
            result = self.verify_match(match, camera_matrix)
            verified_matches[pair] = result

            if result.n_inliers >= self.min_inliers:
                logger.debug(f"Pair {pair}: {result.n_inliers} inliers "
                           f"({result.inlier_ratio*100:.1f}%)")

        # Count valid pairs
        valid_pairs = sum(1 for r in verified_matches.values()
                         if r.n_inliers >= self.min_inliers)

        logger.info(f"RANSAC verification complete: "
                   f"{valid_pairs}/{len(matches_dict)} pairs have sufficient inliers")

        return verified_matches

    def filter_matches(
        self,
        match: FeatureMatch,
        ransac_result: RANSACResult
    ) -> FeatureMatch:
        """
        Create filtered FeatureMatch containing only inliers

        Args:
            match: Original FeatureMatch
            ransac_result: RANSACResult from verification

        Returns:
            New FeatureMatch with only inlier correspondences
        """
        if ransac_result.n_inliers == 0:
            return FeatureMatch(
                img1_idx=match.img1_idx,
                img2_idx=match.img2_idx,
                keypoints1=np.empty((0, 2), dtype=np.float32),
                keypoints2=np.empty((0, 2), dtype=np.float32),
                matches=[],
                descriptors1=np.empty((0, 128), dtype=np.float32),
                descriptors2=np.empty((0, 128), dtype=np.float32)
            )

        # Filter matches based on inlier mask
        mask = ransac_result.inlier_mask
        filtered_matches = [m for i, m in enumerate(match.matches) if mask[i]]

        return FeatureMatch(
            img1_idx=match.img1_idx,
            img2_idx=match.img2_idx,
            keypoints1=ransac_result.inlier_points1,
            keypoints2=ransac_result.inlier_points2,
            matches=filtered_matches,
            descriptors1=match.descriptors1,
            descriptors2=match.descriptors2
        )