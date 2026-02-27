"""
SIFT Feature Detection and Matching Module

Implements SIFT (Scale-Invariant Feature Transform) for detecting and matching
keypoints across multiple views in the SfM pipeline.
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
from ..utils.logger import logger


@dataclass
class FeatureMatch:
    """Container for feature matches between two images"""
    img1_idx: int
    img2_idx: int
    keypoints1: np.ndarray  # Nx2 array of (x, y) coordinates
    keypoints2: np.ndarray  # Nx2 array of (x, y) coordinates
    matches: List[cv2.DMatch]
    descriptors1: np.ndarray
    descriptors2: np.ndarray


class SIFTDetector:
    """
    SIFT-based feature detector and matcher for Structure from Motion

    This class handles:
    - Feature detection using SIFT
    - Feature description
    - Feature matching with ratio test
    """

    def __init__(
        self,
        n_features: int = 0,
        n_octave_layers: int = 3,
        contrast_threshold: float = 0.04,
        edge_threshold: float = 10,
        sigma: float = 1.6,
        ratio_threshold: float = 0.75
    ):
        """
        Initialize SIFT detector with parameters

        Args:
            n_features: Number of best features to retain (0 = all features)
            n_octave_layers: Number of layers in each octave
            contrast_threshold: Threshold for filtering weak features
            edge_threshold: Threshold for filtering edge-like features
            sigma: Sigma of the Gaussian applied to input image at octave 0
            ratio_threshold: Lowe's ratio test threshold for matching
        """
        self.n_features = n_features
        self.ratio_threshold = ratio_threshold

        # Initialize SIFT detector
        self.sift = cv2.SIFT_create(
            nfeatures=n_features,
            nOctaveLayers=n_octave_layers,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
            sigma=sigma
        )

        # Initialize matcher (FLANN-based for faster matching)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

        # Storage for detected features
        self.keypoints_list = []
        self.descriptors_list = []

        logger.info(f"Initialized SIFT detector with {n_features} features, "
                   f"ratio_threshold={ratio_threshold}")

    def detect_and_compute(self, images: List[np.ndarray]) -> None:
        """
        Detect and compute SIFT features for all images

        Args:
            images: List of images (RGB numpy arrays)
        """
        self.keypoints_list = []
        self.descriptors_list = []

        logger.info(f"Detecting SIFT features in {len(images)} images...")

        for idx, img in enumerate(images):
            # Convert to grayscale
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img

            # Detect and compute
            keypoints, descriptors = self.sift.detectAndCompute(gray, None)

            self.keypoints_list.append(keypoints)
            self.descriptors_list.append(descriptors)

            logger.debug(f"Image {idx}: Detected {len(keypoints)} keypoints")

        logger.info(f"Feature detection complete. "
                   f"Average: {np.mean([len(kp) for kp in self.keypoints_list]):.0f} keypoints per image")

    def match_features(self, idx1: int, idx2: int) -> FeatureMatch:
        """
        Match features between two images using Lowe's ratio test

        Args:
            idx1: Index of first image
            idx2: Index of second image

        Returns:
            FeatureMatch object containing matched keypoints and descriptors
        """
        if idx1 >= len(self.descriptors_list) or idx2 >= len(self.descriptors_list):
            raise IndexError(f"Image indices out of range: {idx1}, {idx2}")

        desc1 = self.descriptors_list[idx1]
        desc2 = self.descriptors_list[idx2]

        if desc1 is None or desc2 is None:
            logger.warning(f"No descriptors found for images {idx1}, {idx2}")
            return self._create_empty_match(idx1, idx2)

        # Match descriptors using KNN with k=2
        matches = self.matcher.knnMatch(desc1, desc2, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)

        # Extract matched keypoint coordinates
        kp1 = self.keypoints_list[idx1]
        kp2 = self.keypoints_list[idx2]

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        logger.debug(f"Matched {len(good_matches)} features between images {idx1} and {idx2}")

        return FeatureMatch(
            img1_idx=idx1,
            img2_idx=idx2,
            keypoints1=pts1,
            keypoints2=pts2,
            matches=good_matches,
            descriptors1=desc1,
            descriptors2=desc2
        )

    def match_all_pairs(self) -> Dict[Tuple[int, int], FeatureMatch]:
        """
        Match features between all image pairs

        Returns:
            Dictionary mapping (img1_idx, img2_idx) to FeatureMatch objects
        """
        n_images = len(self.descriptors_list)
        all_matches = {}

        logger.info(f"Matching features for all {n_images * (n_images - 1) // 2} image pairs...")

        for i in range(n_images):
            for j in range(i + 1, n_images):
                match = self.match_features(i, j)
                all_matches[(i, j)] = match

        logger.info(f"Matched all image pairs. "
                   f"Average: {np.mean([len(m.matches) for m in all_matches.values()]):.0f} matches per pair")

        return all_matches

    def _create_empty_match(self, idx1: int, idx2: int) -> FeatureMatch:
        """Create an empty FeatureMatch for cases with no valid matches"""
        return FeatureMatch(
            img1_idx=idx1,
            img2_idx=idx2,
            keypoints1=np.empty((0, 2), dtype=np.float32),
            keypoints2=np.empty((0, 2), dtype=np.float32),
            matches=[],
            descriptors1=np.empty((0, 128), dtype=np.float32),
            descriptors2=np.empty((0, 128), dtype=np.float32)
        )

    def visualize_matches(
        self,
        img1: np.ndarray,
        img2: np.ndarray,
        match: FeatureMatch,
        max_matches: int = 50
    ) -> np.ndarray:
        """
        Visualize feature matches between two images

        Args:
            img1: First image
            img2: Second image
            match: FeatureMatch object
            max_matches: Maximum number of matches to draw

        Returns:
            Image showing feature matches
        """
        kp1 = self.keypoints_list[match.img1_idx]
        kp2 = self.keypoints_list[match.img2_idx]

        # Convert RGB to BGR for OpenCV
        img1_bgr = cv2.cvtColor(img1, cv2.COLOR_RGB2BGR)
        img2_bgr = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)

        # Draw matches
        matches_to_draw = match.matches[:max_matches]
        match_img = cv2.drawMatches(
            img1_bgr, kp1,
            img2_bgr, kp2,
            matches_to_draw, None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        # Convert back to RGB
        match_img = cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB)

        return match_img

    def get_keypoints(self, idx: int) -> List[cv2.KeyPoint]:
        """Get keypoints for image at index"""
        return self.keypoints_list[idx]

    def get_descriptors(self, idx: int) -> np.ndarray:
        """Get descriptors for image at index"""
        return self.descriptors_list[idx]