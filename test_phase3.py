"""
Phase 3 Test Script: Incremental Reconstruction and Bundle Adjustment

Tests PnP camera registration and Bundle Adjustment optimization
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
from src.geometry.pnp import PnPSolver, IncrementalReconstructor
from src.optimization.bundle_adjustment import BundleAdjuster
from src.utils.logger import logger


def test_phase3(image_dir: str = "Datasets/dinoRing", max_cameras: int = 5):
    """
    Test Phase 3: Incremental reconstruction and Bundle Adjustment

    Args:
        image_dir: Directory containing test images
        max_cameras: Maximum number of cameras to register
    """
    logger.info("=" * 60)
    logger.info("PHASE 3 TEST: Incremental Reconstruction & Bundle Adjustment")
    logger.info("=" * 60)

    # Step 1: Load images
    logger.info("\n[1/8] Loading images...")
    loader = ImageLoader(max_dimension=1600)

    try:
        images = loader.load_images(image_dir)
        logger.info(f"Loaded {len(images)} images")
    except Exception as e:
        logger.error(f"Failed to load images: {e}")
        return

    if len(images) < 3:
        logger.error("Need at least 3 images for incremental reconstruction")
        return

    image_shape = images[0].shape[:2]

    # Step 2: Detect SIFT features
    logger.info("\n[2/8] Detecting SIFT features...")
    detector = SIFTDetector(n_features=2000, ratio_threshold=0.75)
    detector.detect_and_compute(images)

    # Step 3: Match and verify
    logger.info("\n[3/8] Matching features and RANSAC verification...")
    all_matches = detector.match_all_pairs()

    verifier = RANSACVerifier(
        method='fundamental',
        ransac_threshold=1.0,
        min_inliers=50  # Lower threshold for more matches
    )
    verified_matches = verifier.verify_all_matches(all_matches)

    # Step 4: Two-view initialization
    logger.info("\n[4/8] Initializing with two views...")
    initializer = TwoViewInitializer()

    img1_idx, img2_idx = initializer.select_initial_pair(
        all_matches, verified_matches, image_shape
    )

    match = all_matches[(img1_idx, img2_idx)]
    ransac_result = verified_matches[(img1_idx, img2_idx)]

    reconstruction = initializer.initialize_reconstruction(
        match, ransac_result, image_shape
    )

    triangulator = Triangulator(min_parallax=1.0, max_reproj_error=4.0)
    triangulated = triangulator.triangulate(reconstruction)
    triangulated = triangulator.filter_valid_points(triangulated)

    logger.info(f"Initialized with {len(triangulated.points_3d)} valid 3D points")

    # Step 5: Setup incremental reconstructor
    logger.info("\n[5/8] Setting up incremental reconstruction...")
    reconstructor = IncrementalReconstructor(K=reconstruction.K)

    reconstructor.initialize_from_two_view(
        img1_idx, img2_idx,
        reconstruction.pose1, reconstruction.pose2,
        triangulated.points_3d,
        triangulated.points_2d_img1,
        triangulated.points_2d_img2
    )

    # Step 6: Register additional cameras
    logger.info(f"\n[6/8] Registering up to {max_cameras} cameras total...")

    registered_order = [img1_idx, img2_idx]
    n_cameras_to_add = min(max_cameras - 2, len(images) - 2)

    for _ in range(n_cameras_to_add):
        # Find best next camera to register
        best_camera = _find_next_camera(
            reconstructor,
            all_matches,
            verified_matches,
            detector,
            len(images)
        )

        if best_camera is None:
            logger.warning("No more cameras can be registered")
            break

        # Get 2D-3D correspondences
        points_2d, point_indices = _get_2d_3d_correspondences(
            best_camera,
            reconstructor,
            all_matches,
            verified_matches
        )

        if len(points_2d) < 30:
            logger.warning(f"Too few correspondences for camera {best_camera}")
            continue

        # Register camera
        pnp_result = reconstructor.register_camera(
            best_camera, points_2d, point_indices
        )

        if pnp_result.success:
            registered_order.append(best_camera)
            logger.info(f"Successfully registered camera {best_camera}")

            # Triangulate new points if possible
            _triangulate_new_points(
                reconstructor, best_camera, detector,
                all_matches, verified_matches, triangulator
            )

    n_cams, n_pts = reconstructor.get_reconstruction_size()
    logger.info(f"\nIncremental reconstruction: {n_cams} cameras, {n_pts} points")

    # Step 7: Bundle Adjustment
    logger.info("\n[7/8] Running Bundle Adjustment...")

    ba = BundleAdjuster(
        optimize_intrinsics=False,
        loss_function='huber',
        max_nfev=50
    )

    # Get initial error
    initial_error = _compute_mean_reprojection_error(
        reconstructor.registered_cameras,
        reconstructor.points_3d,
        reconstructor.point_observations,
        reconstructor.K
    )

    logger.info(f"Initial mean reprojection error: {initial_error:.3f} pixels")

    optimized_poses, optimized_points, K_opt, final_error = ba.optimize(
        reconstructor.registered_cameras,
        reconstructor.points_3d,
        reconstructor.point_observations,
        reconstructor.K,
        fix_first_camera=True
    )

    logger.info(f"Optimized mean reprojection error: {final_error:.3f} pixels")
    logger.info(f"Improvement: {initial_error - final_error:.3f} pixels "
               f"({(1 - final_error/initial_error)*100:.1f}% reduction)")

    # Update reconstructor with optimized values
    reconstructor.registered_cameras = optimized_poses
    reconstructor.points_3d = optimized_points

    # Step 8: Visualize results
    logger.info("\n[8/8] Creating visualizations...")
    output_dir = Path("data/output")
    output_dir.mkdir(exist_ok=True, parents=True)

    _visualize_reconstruction(reconstructor, registered_order, output_dir)
    _visualize_error_comparison(initial_error, final_error, output_dir)

    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3 TEST COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Final reconstruction: {n_cams} cameras, {n_pts} points")
    logger.info(f"Final reprojection error: {final_error:.3f} pixels")
    logger.info(f"\nOutputs saved to: {output_dir.absolute()}")


def _find_next_camera(reconstructor, all_matches, verified_matches, detector, n_images):
    """Find the next best camera to register"""
    registered = set(reconstructor.get_registered_camera_indices())
    unregistered = set(range(n_images)) - registered

    best_camera = None
    best_score = 0

    for cam_idx in unregistered:
        # Count matches with registered cameras
        n_matches = 0
        for reg_idx in registered:
            pair = tuple(sorted([cam_idx, reg_idx]))
            if pair in verified_matches:
                result = verified_matches[pair]
                n_matches += result.n_inliers

        if n_matches > best_score:
            best_score = n_matches
            best_camera = cam_idx

    return best_camera


def _get_2d_3d_correspondences(camera_idx, reconstructor, all_matches, verified_matches):
    """Get 2D-3D correspondences for a camera"""
    points_2d_list = []
    point_indices_list = []

    registered = reconstructor.get_registered_camera_indices()

    for reg_idx in registered:
        pair = tuple(sorted([camera_idx, reg_idx]))
        if pair not in verified_matches:
            continue

        result = verified_matches[pair]
        if result.n_inliers < 20:
            continue

        match = all_matches[pair]

        # Determine which points are which
        if pair[0] == camera_idx:
            pts_cam = result.inlier_points1
            pts_reg = result.inlier_points2
        else:
            pts_cam = result.inlier_points2
            pts_reg = result.inlier_points1

        # Find corresponding 3D points
        for i, (pt_cam, pt_reg) in enumerate(zip(pts_cam, pts_reg)):
            # Find this 2D point in the registered camera's observations
            for pt_idx, obs in enumerate(reconstructor.point_observations):
                if reg_idx in obs:
                    obs_pt = obs[reg_idx]
                    if np.linalg.norm(obs_pt - pt_reg) < 1.0:  # Match threshold
                        points_2d_list.append(pt_cam)
                        point_indices_list.append(pt_idx)
                        break

    if len(points_2d_list) == 0:
        return np.empty((0, 2)), np.empty(0, dtype=int)

    return np.array(points_2d_list), np.array(point_indices_list)


def _triangulate_new_points(reconstructor, new_cam_idx, detector, all_matches, verified_matches, triangulator):
    """Triangulate new points visible in the new camera"""
    # Get the pose of the newly registered camera
    new_pose = reconstructor.get_camera_pose(new_cam_idx)
    if new_pose is None:
        return

    points_added = 0
    registered_cams = set(reconstructor.get_registered_camera_indices())
    registered_cams.remove(new_cam_idx)

    # Try to triangulate with each already registered camera
    for reg_cam_idx in registered_cams:
        pair = tuple(sorted([new_cam_idx, reg_cam_idx]))
        if pair not in verified_matches:
            continue

        result = verified_matches[pair]
        if result.n_inliers < 30:
            continue

        reg_pose = reconstructor.get_camera_pose(reg_cam_idx)
        
        # Determine point order based on the pair tuple
        if pair[0] == new_cam_idx:
            pts_new = result.inlier_points1
            pts_reg = result.inlier_points2
        else:
            pts_new = result.inlier_points2
            pts_reg = result.inlier_points1

        # Create a temporary TwoViewReconstruction to use the Triangulator
        from src.geometry.two_view import TwoViewReconstruction
        temp_recon = TwoViewReconstruction(
            img1_idx=new_cam_idx,
            img2_idx=reg_cam_idx,
            K=reconstructor.K,
            pose1=new_pose,
            pose2=reg_pose,
            inlier_points1=pts_new,
            inlier_points2=pts_reg,
            F=np.eye(3),  # Not needed for triangulation
            E=np.eye(3)   # Not needed for triangulation
        )

        # Triangulate
        triangulated = triangulator.triangulate(temp_recon)
        filtered = triangulator.filter_valid_points(triangulated)

        if len(filtered.points_3d) > 0:
            # Build observations list matching the expected format
            observations = []
            for i in range(len(filtered.points_3d)):
                obs = {
                    new_cam_idx: filtered.points_2d_img1[i],
                    reg_cam_idx: filtered.points_2d_img2[i]
                }
                observations.append(obs)
            
            reconstructor.add_points(filtered.points_3d, observations)
            points_added += len(filtered.points_3d)

    logger.info(f"Triangulated {points_added} new points using camera {new_cam_idx}")


def _compute_mean_reprojection_error(camera_poses, points_3d, point_observations, K):
    """Compute mean reprojection error"""
    errors = []

    for pt_idx, obs_dict in enumerate(point_observations):
        X = points_3d[pt_idx]

        for img_idx, pt_2d in obs_dict.items():
            if img_idx not in camera_poses:
                continue

            pose = camera_poses[img_idx]

            # Project
            X_cam = pose.R @ X + pose.t
            if X_cam[2] <= 0:
                continue

            x_proj = K @ X_cam
            x_proj = x_proj[:2] / x_proj[2]

            error = np.linalg.norm(pt_2d - x_proj)
            errors.append(error)

    return np.mean(errors) if errors else 0.0


def _visualize_reconstruction(reconstructor, registered_order, output_dir):
    """Visualize the final 3D reconstruction"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot 3D points
    points = reconstructor.points_3d
    ax.scatter(points[:, 0], points[:, 1], points[:, 2],
              c='blue', marker='.', s=1, alpha=0.5, label='3D Points')

    # Plot cameras
    colors = plt.cm.rainbow(np.linspace(0, 1, len(registered_order)))

    for i, img_idx in enumerate(registered_order):
        pose = reconstructor.get_camera_pose(img_idx)
        C = -pose.R.T @ pose.t

        ax.scatter(*C, c=[colors[i]], marker='o', s=100,
                  label=f'Camera {img_idx}' if i < 10 else None)

        # Camera orientation
        scale = np.linalg.norm(points.max(axis=0) - points.min(axis=0)) * 0.05
        for j, color in enumerate(['r', 'g', 'b']):
            axis = pose.R.T[:, j] * scale
            ax.plot([C[0], C[0] + axis[0]],
                   [C[1], C[1] + axis[1]],
                   [C[2], C[2] + axis[2]], color + '-', linewidth=1, alpha=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_title(f'Incremental Reconstruction\n'
                f'{len(registered_order)} cameras, {len(points)} points')

    plt.tight_layout()
    plt.savefig(output_dir / "phase3_reconstruction.png", dpi=150, bbox_inches='tight')
    logger.info(f"Saved reconstruction to {output_dir / 'phase3_reconstruction.png'}")
    plt.close()


def _visualize_error_comparison(initial_error, final_error, output_dir):
    """Visualize error before and after BA"""
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ['Before BA', 'After BA']
    errors = [initial_error, final_error]
    colors = ['#ff6b6b', '#51cf66']

    bars = ax.bar(categories, errors, color=colors, alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for bar, error in zip(bars, errors):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{error:.3f}px',
               ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add target line
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2,
              label='Target: 1.0 pixel', alpha=0.7)

    ax.set_ylabel('Mean Reprojection Error (pixels)', fontsize=12)
    ax.set_title('Bundle Adjustment Optimization Results', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "phase3_ba_comparison.png", dpi=150, bbox_inches='tight')
    logger.info(f"Saved BA comparison to {output_dir / 'phase3_ba_comparison.png'}")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Phase 3: Incremental Reconstruction & BA")
    parser.add_argument(
        "--images",
        type=str,
        default="Datasets/dinoRing",
        help="Directory containing input images"
    )
    parser.add_argument(
        "--max-cameras",
        type=int,
        default=5,
        help="Maximum number of cameras to register"
    )

    args = parser.parse_args()

    test_phase3(args.images, args.max_cameras)