"""
Phase 2 Test Script: Two-View Initialization and Triangulation

Tests the two-view initialization and 3D point triangulation modules
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.image_loader import ImageLoader
from src.features.sift_detector import SIFTDetector
from src.geometry.ransac import RANSACVerifier
from src.geometry.two_view import TwoViewInitializer
from src.geometry.triangulation import Triangulator
from src.utils.logger import logger


def test_phase2(image_dir: str = "Datasets/dinoRing"):
    """
    Test Phase 2: Two-view initialization and triangulation

    Args:
        image_dir: Directory containing test images
    """
    logger.info("=" * 60)
    logger.info("PHASE 2 TEST: Two-View Initialization & Triangulation")
    logger.info("=" * 60)

    # Step 1: Load images
    logger.info("\n[1/6] Loading images...")
    loader = ImageLoader(max_dimension=1600)

    try:
        images = loader.load_images(image_dir)
        logger.info(f"Loaded {len(images)} images")
    except Exception as e:
        logger.error(f"Failed to load images: {e}")
        logger.error(f"Please specify a valid image directory")
        logger.error(f"Example: python test_phase2.py --images Datasets/dinoRing")
        return

    if len(images) < 2:
        logger.error("Need at least 2 images for testing")
        return

    image_shape = images[0].shape[:2]

    # Step 2: Detect SIFT features
    logger.info("\n[2/6] Detecting SIFT features...")
    detector = SIFTDetector(n_features=2000, ratio_threshold=0.75)
    detector.detect_and_compute(images)

    # Step 3: Match all pairs and verify with RANSAC
    logger.info("\n[3/6] Matching features and RANSAC verification...")
    all_matches = detector.match_all_pairs()

    verifier = RANSACVerifier(
        method='fundamental',
        ransac_threshold=1.0,
        min_inliers=100
    )
    verified_matches = verifier.verify_all_matches(all_matches)

    # Step 4: Select initial image pair
    logger.info("\n[4/6] Selecting initial image pair...")
    initializer = TwoViewInitializer(focal_length=None)

    try:
        img1_idx, img2_idx = initializer.select_initial_pair(
            all_matches, verified_matches, image_shape
        )
        logger.info(f"Selected pair: Image {img1_idx} and Image {img2_idx}")
    except ValueError as e:
        logger.error(f"Failed to select initial pair: {e}")
        return

    # Step 5: Initialize reconstruction
    logger.info("\n[5/6] Initializing two-view reconstruction...")
    match = all_matches[(img1_idx, img2_idx)]
    ransac_result = verified_matches[(img1_idx, img2_idx)]

    reconstruction = initializer.initialize_reconstruction(
        match, ransac_result, image_shape
    )

    logger.info(f"\nReconstruction Summary:")
    logger.info(f"  Intrinsic matrix K:")
    logger.info(f"    {reconstruction.K[0]}")
    logger.info(f"    {reconstruction.K[1]}")
    logger.info(f"    {reconstruction.K[2]}")
    logger.info(f"  Camera 1 pose: R=I, t=[0,0,0]")
    logger.info(f"  Camera 2 pose:")
    logger.info(f"    R det = {np.linalg.det(reconstruction.pose2.R):.6f}")
    logger.info(f"    t = {reconstruction.pose2.t}")
    logger.info(f"  Baseline: {np.linalg.norm(reconstruction.pose2.t):.3f}")

    # Step 6: Triangulate 3D points
    logger.info("\n[6/6] Triangulating 3D points...")
    triangulator = Triangulator(
        min_parallax=1.0,
        max_reproj_error=4.0
    )

    triangulated = triangulator.triangulate(reconstruction)

    logger.info(f"\nTriangulation Summary:")
    logger.info(f"  Total points: {len(triangulated.points_3d)}")
    logger.info(f"  Valid points: {np.sum(triangulated.valid_mask)}")
    logger.info(f"  Invalid points: {np.sum(~triangulated.valid_mask)}")

    valid_errors = triangulated.reprojection_errors[triangulated.valid_mask]
    if len(valid_errors) > 0:
        logger.info(f"  Reprojection error (valid points):")
        logger.info(f"    Mean: {valid_errors.mean():.3f} pixels")
        logger.info(f"    Median: {np.median(valid_errors):.3f} pixels")
        logger.info(f"    Max: {valid_errors.max():.3f} pixels")

    # Visualizations
    output_dir = Path("data/output")
    output_dir.mkdir(exist_ok=True, parents=True)

    # Visualize camera poses
    logger.info("\nCreating visualizations...")
    _visualize_cameras_and_points(reconstruction, triangulated, output_dir)

    # Visualize reprojection errors
    _visualize_reprojection_errors(triangulated, output_dir)

    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2 TEST COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nOutputs saved to: {output_dir.absolute()}")


def _visualize_cameras_and_points(reconstruction, triangulated, output_dir):
    """Visualize camera poses and triangulated 3D points"""

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Filter valid points
    valid_mask = triangulated.valid_mask
    points_3d = triangulated.points_3d[valid_mask]

    # Plot 3D points
    ax.scatter(
        points_3d[:, 0],
        points_3d[:, 1],
        points_3d[:, 2],
        c='blue', marker='.', s=1, alpha=0.5
    )

    # Plot camera centers
    C1 = -reconstruction.pose1.R.T @ reconstruction.pose1.t
    C2 = -reconstruction.pose2.R.T @ reconstruction.pose2.t

    ax.scatter(*C1, c='red', marker='o', s=100, label='Camera 1')
    ax.scatter(*C2, c='green', marker='o', s=100, label='Camera 2')

    # Plot camera orientations
    scale = np.linalg.norm(reconstruction.pose2.t) * 0.3

    # Camera 1 axes
    R1 = reconstruction.pose1.R
    for i, color in enumerate(['r', 'g', 'b']):
        axis = R1[:, i] * scale
        ax.plot([C1[0], C1[0] + axis[0]],
               [C1[1], C1[1] + axis[1]],
               [C1[2], C1[2] + axis[2]], color + '-', linewidth=2)

    # Camera 2 axes
    R2 = reconstruction.pose2.R
    for i, color in enumerate(['r', 'g', 'b']):
        axis = R2[:, i] * scale
        ax.plot([C2[0], C2[0] + axis[0]],
               [C2[1], C2[1] + axis[1]],
               [C2[2], C2[2] + axis[2]], color + '-', linewidth=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title(f'Two-View Reconstruction\n{np.sum(valid_mask)} valid 3D points')

    # Equal aspect ratio
    max_range = np.array([
        points_3d[:, 0].max() - points_3d[:, 0].min(),
        points_3d[:, 1].max() - points_3d[:, 1].min(),
        points_3d[:, 2].max() - points_3d[:, 2].min()
    ]).max() / 2.0

    mid_x = (points_3d[:, 0].max() + points_3d[:, 0].min()) * 0.5
    mid_y = (points_3d[:, 1].max() + points_3d[:, 1].min()) * 0.5
    mid_z = (points_3d[:, 2].max() + points_3d[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.savefig(output_dir / "phase2_reconstruction_3d.png", dpi=150, bbox_inches='tight')
    logger.info(f"Saved 3D reconstruction to {output_dir / 'phase2_reconstruction_3d.png'}")
    plt.close()


def _visualize_reprojection_errors(triangulated, output_dir):
    """Visualize reprojection error distribution"""

    valid_errors = triangulated.reprojection_errors[triangulated.valid_mask]

    if len(valid_errors) == 0:
        logger.warning("No valid points to visualize reprojection errors")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    axes[0].hist(valid_errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(valid_errors.mean(), color='r', linestyle='--',
                   label=f'Mean: {valid_errors.mean():.3f}px')
    axes[0].axvline(np.median(valid_errors), color='g', linestyle='--',
                   label=f'Median: {np.median(valid_errors):.3f}px')
    axes[0].set_xlabel('Reprojection Error (pixels)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Reprojection Error Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Cumulative distribution
    sorted_errors = np.sort(valid_errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    axes[1].plot(sorted_errors, cumulative, linewidth=2)
    axes[1].axhline(50, color='r', linestyle='--', alpha=0.5)
    axes[1].axhline(95, color='g', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Reprojection Error (pixels)')
    axes[1].set_ylabel('Cumulative Percentage (%)')
    axes[1].set_title('Cumulative Reprojection Error')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "phase2_reprojection_errors.png", dpi=150, bbox_inches='tight')
    logger.info(f"Saved reprojection error plot to {output_dir / 'phase2_reprojection_errors.png'}")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Phase 2: Two-View Initialization & Triangulation")
    parser.add_argument(
        "--images",
        type=str,
        default="Datasets/dinoRing",
        help="Directory containing input images (e.g., Datasets/dinoRing, Datasets/templeSparseRing)"
    )

    args = parser.parse_args()

    test_phase2(args.images)