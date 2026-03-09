"""
Phase 1 Test Script: Feature Detection and Matching

Tests the SIFT feature detector and RANSAC verification modules
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.image_loader import ImageLoader
from src.features.sift_detector import SIFTDetector
from src.geometry.ransac import RANSACVerifier
from src.utils.logger import logger


def test_phase1(image_dir: str = "Datasets/dinoRing"):
    """
    Test Phase 1: Feature detection, matching, and RANSAC verification

    Args:
        image_dir: Directory containing test images
    """
    logger.info("=" * 60)
    logger.info("PHASE 1 TEST: Feature Detection and Matching")
    logger.info("=" * 60)

    # Step 1: Load images
    logger.info("\n[1/4] Loading images...")
    loader = ImageLoader(max_dimension=1600)

    try:
        images = loader.load_images(image_dir)
        logger.info(f"Loaded {len(images)} images")
    except Exception as e:
        logger.error(f"Failed to load images: {e}")
        return

    if len(images) < 2:
        logger.error("Need at least 2 images for testing")
        return

    # Step 2: Detect SIFT features
    logger.info("\n[2/4] Detecting SIFT features...")
    detector = SIFTDetector(n_features=2000, ratio_threshold=0.75)
    detector.detect_and_compute(images)

    # Print feature statistics
    for i in range(len(images)):
        n_features = len(detector.get_keypoints(i))
        logger.info(f"  Image {i}: {n_features} keypoints detected")

    # Step 3: Match features between first two images
    logger.info("\n[3/4] Matching features...")
    if len(images) >= 2:
        match = detector.match_features(0, 1)
        logger.info(f"Matched {len(match.matches)} features between images 0 and 1")

        # Visualize matches
        if len(match.matches) > 0:
            logger.info("Creating match visualization...")
            match_img = detector.visualize_matches(images[0], images[1], match, max_matches=100)

            # Save visualization
            output_dir = Path("data/output")
            output_dir.mkdir(exist_ok=True, parents=True)

            plt.figure(figsize=(15, 8))
            plt.imshow(match_img)
            plt.title(f"SIFT Feature Matches: {len(match.matches)} correspondences")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_dir / "phase1_matches_raw.png", dpi=150, bbox_inches='tight')
            logger.info(f"Saved match visualization to {output_dir / 'phase1_matches_raw.png'}")
            plt.close()

    # Step 4: RANSAC verification
    logger.info("\n[4/4] Running RANSAC verification...")
    verifier = RANSACVerifier(
        method='fundamental',
        ransac_threshold=1.0,
        min_inliers=50
    )

    if len(images) >= 2 and len(match.matches) > 0:
        ransac_result = verifier.verify_match(match)

        logger.info(f"\nRANSAC Results:")
        logger.info(f"  Total matches: {len(match.matches)}")
        logger.info(f"  Inliers: {ransac_result.n_inliers}")
        logger.info(f"  Inlier ratio: {ransac_result.inlier_ratio*100:.2f}%")
        logger.info(f"  Outliers removed: {len(match.matches) - ransac_result.n_inliers}")

        # Visualize inlier matches
        if ransac_result.n_inliers > 0:
            filtered_match = verifier.filter_matches(match, ransac_result)
            filtered_img = detector.visualize_matches(
                images[0], images[1], filtered_match, max_matches=100
            )

            plt.figure(figsize=(15, 8))
            plt.imshow(filtered_img)
            plt.title(f"RANSAC Inliers: {ransac_result.n_inliers} verified correspondences "
                     f"({ransac_result.inlier_ratio*100:.1f}%)")
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(output_dir / "phase1_matches_ransac.png", dpi=150, bbox_inches='tight')
            logger.info(f"Saved RANSAC visualization to {output_dir / 'phase1_matches_ransac.png'}")
            plt.close()

    # Match all pairs
    if len(images) > 2:
        logger.info(f"\n[Bonus] Matching all {len(images)*(len(images)-1)//2} image pairs...")
        all_matches = detector.match_all_pairs()
        verified_all = verifier.verify_all_matches(all_matches)

        # Print summary
        logger.info("\nPair-wise matching summary:")
        for pair, result in verified_all.items():
            if result.n_inliers >= 50:
                logger.info(f"  Images {pair}: {result.n_inliers} inliers "
                          f"({result.inlier_ratio*100:.1f}%)")

    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1 TEST COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nOutputs saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Phase 1: Feature Detection and Matching")
    parser.add_argument(
        "--images",
        type=str,
        default="Datasets/dinoRing",
        help="Directory containing input images"
    )

    args = parser.parse_args()

    test_phase1(args.images)