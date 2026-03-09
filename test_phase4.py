"""
Phase 4 Test Script: Densification and Colorization

Tests Multi-View Stereo densification, point cloud colorization, and PLY export
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
from src.geometry.pnp import IncrementalReconstructor
from src.optimization.bundle_adjustment import BundleAdjuster
from src.densification import MVSDensifier, colorize_sparse_points
from src.utils.ply_export import export_reconstruction_to_ply
from src.visualization.mesh_reconstruction import MeshReconstructor
from src.utils.logger import logger


def test_phase4(
    image_dir: str = "Datasets/dinoRing",
    max_cameras: int = 10,
    use_mvs: bool = False  # MVS is computationally expensive
):
    """
    Test Phase 4: Densification and Colorization

    Args:
        image_dir: Directory containing test images
        max_cameras: Maximum number of cameras to register
        use_mvs: Whether to run full MVS densification (slow)
    """
    logger.info("=" * 60)
    logger.info("PHASE 4 TEST: Densification & Colorization")
    logger.info("=" * 60)

    # =========================================================================
    # Step 1-7: Run Phase 3 pipeline (sparse reconstruction + BA)
    # =========================================================================
    logger.info("\n[1/9] Running sparse reconstruction pipeline...")

    # Load images
    loader = ImageLoader(max_dimension=1600)
    try:
        images = loader.load_images(image_dir)
        logger.info(f"Loaded {len(images)} images")
    except Exception as e:
        logger.error(f"Failed to load images: {e}")
        return

    if len(images) < 3:
        logger.error("Need at least 3 images for reconstruction")
        return

    image_shape = images[0].shape[:2]

    # Detect SIFT features
    logger.info("\n[2/9] Detecting SIFT features...")
    detector = SIFTDetector(n_features=2000, ratio_threshold=0.75)
    detector.detect_and_compute(images)

    # Match and verify
    logger.info("\n[3/9] Matching and verifying features...")
    all_matches = detector.match_all_pairs()
    verifier = RANSACVerifier(
        method='fundamental',
        ransac_threshold=1.0,
        min_inliers=50
    )
    verified_matches = verifier.verify_all_matches(all_matches)

    # Two-view initialization
    logger.info("\n[4/9] Two-view initialization...")
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

    logger.info(f"Initial reconstruction: {len(triangulated.points_3d)} points")

    # Incremental reconstruction
    logger.info(f"\n[5/9] Registering up to {max_cameras} cameras...")
    reconstructor = IncrementalReconstructor(K=reconstruction.K)
    reconstructor.initialize_from_two_view(
        img1_idx, img2_idx,
        reconstruction.pose1, reconstruction.pose2,
        triangulated.points_3d,
        triangulated.points_2d_img1,
        triangulated.points_2d_img2
    )

    # Register additional cameras
    registered_order = [img1_idx, img2_idx]
    n_cameras_to_add = min(max_cameras - 2, len(images) - 2)

    for _ in range(n_cameras_to_add):
        best_camera = _find_next_camera(
            reconstructor, all_matches, verified_matches, detector, len(images)
        )

        if best_camera is None:
            logger.warning("No more cameras can be registered")
            break

        points_2d, point_indices = _get_2d_3d_correspondences(
            best_camera, reconstructor, all_matches, verified_matches
        )

        if len(points_2d) < 30:
            logger.warning(f"Too few correspondences for camera {best_camera}")
            continue

        pnp_result = reconstructor.register_camera(
            best_camera, points_2d, point_indices
        )

        if pnp_result.success:
            registered_order.append(best_camera)
            logger.info(f"Registered camera {best_camera}")

    n_cams, n_pts = reconstructor.get_reconstruction_size()
    logger.info(f"Sparse reconstruction: {n_cams} cameras, {n_pts} points")

    # Bundle Adjustment
    logger.info("\n[6/9] Running Bundle Adjustment...")
    ba = BundleAdjuster(
        optimize_intrinsics=False,
        loss_function='huber',
        max_nfev=50
    )

    optimized_poses, optimized_points, K_opt, final_error = ba.optimize(
        reconstructor.registered_cameras,
        reconstructor.points_3d,
        reconstructor.point_observations,
        reconstructor.K,
        fix_first_camera=True
    )

    logger.info(f"Final reprojection error: {final_error:.3f} pixels")

    # Update reconstructor
    reconstructor.registered_cameras = optimized_poses
    reconstructor.points_3d = optimized_points

    # =========================================================================
    # Step 8: Colorization
    # =========================================================================
    logger.info("\n[7/9] Colorizing sparse point cloud...")

    colors = colorize_sparse_points(
        reconstructor.points_3d,
        reconstructor.point_observations,
        images,
        reconstructor.registered_cameras
    )

    logger.info(f"Colored {len(colors)} points")

    # =========================================================================
    # Step 9: MVS Densification (Optional - computationally expensive)
    # =========================================================================
    if use_mvs:
        logger.info("\n[8/9] Running MVS densification (this may take a while)...")

        densifier = MVSDensifier(
            min_views=3,
            consistency_threshold=0.01
        )

        dense_cloud = densifier.densify(
            images,
            reconstructor.registered_cameras,
            reconstructor.K,
            sparse_points=reconstructor.points_3d
        )

        logger.info(f"Dense reconstruction: {len(dense_cloud.points_3d)} points")

        # Use dense cloud for export
        export_points = dense_cloud.points_3d
        export_colors = dense_cloud.colors
        reconstruction_type = "dense"
    else:
        logger.info("\n[8/9] Skipping MVS densification (use --mvs flag to enable)")
        export_points = reconstructor.points_3d
        export_colors = colors
        reconstruction_type = "sparse"

    # =========================================================================
    # Step 10: PLY Export
    # =========================================================================
    logger.info("\n[9/9] Exporting to PLY format...")

    output_dir = Path("data/output")
    output_dir.mkdir(exist_ok=True, parents=True)

    metadata = {
        'n_cameras': n_cams,
        'reprojection_error': final_error,
        'reconstruction_type': reconstruction_type
    }

    export_reconstruction_to_ply(
        str(output_dir),
        export_points,
        export_colors,
        reconstructor.registered_cameras,
        reconstructor.K,
        metadata=metadata,
        prefix="phase4_reconstruction"
    )

    # =========================================================================
    # Surface Reconstruction (Mesh Generation)
    # =========================================================================
    logger.info("\n[10/10] Creating 3D mesh surface...")

    try:
        # Choose reconstruction method based on point count
        if len(export_points) > 10000:
            # Dense point cloud - use Ball Pivoting (more robust)
            logger.info("Running Ball Pivoting surface reconstruction (for dense clouds)...")
            vertices, triangles, vertex_colors = MeshReconstructor.ball_pivoting_reconstruction(
                export_points,
                export_colors
            )
        else:
            # Sparse point cloud - use Poisson
            logger.info("Running Poisson surface reconstruction...")
            vertices, triangles, vertex_colors = MeshReconstructor.poisson_reconstruction(
                export_points,
                export_colors,
                depth=9,  # Octree depth (9-11 recommended)
                density_threshold=0.01
            )

        # Export mesh
        mesh_path = output_dir / "phase4_reconstruction_mesh.ply"
        MeshReconstructor.export_mesh_to_ply(
            str(mesh_path),
            vertices,
            triangles,
            vertex_colors
        )

        logger.info(f"3D MESH CREATED: {mesh_path}")
        logger.info(f"  Vertices: {len(vertices)}")
        logger.info(f"  Triangles: {len(triangles)}")
        logger.info(f"\nYou can now view the 3D dinosaur in:")
        logger.info(f"  - MeshLab: Open {mesh_path.name}")
        logger.info(f"  - CloudCompare: Open {mesh_path.name}")
        logger.info(f"  - Blender: Import PLY -> {mesh_path.name}")

    except Exception as e:
        logger.warning(f"Mesh reconstruction failed: {e}")
        logger.warning("Only point cloud PLY files will be available")

    # =========================================================================
    # Visualization
    # =========================================================================
    logger.info("\nCreating visualizations...")
    _visualize_colored_reconstruction(
        export_points, export_colors, reconstructor.registered_cameras, output_dir
    )

    logger.info("\n" + "=" * 60)
    logger.info("PHASE 4 TEST COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Reconstruction type: {reconstruction_type.upper()}")
    logger.info(f"Points: {len(export_points)}")
    logger.info(f"Cameras: {n_cams}")
    logger.info(f"Reprojection error: {final_error:.3f} pixels")
    logger.info(f"\nOutputs saved to: {output_dir.absolute()}")
    logger.info(f"  - phase4_reconstruction_points.ply (point cloud)")
    logger.info(f"  - phase4_reconstruction_mesh.ply (3D MESH - THE ACTUAL DINOSAUR!)")
    logger.info(f"  - phase4_reconstruction_cameras.ply (camera frustums)")
    logger.info(f"  - phase4_colored_reconstruction.png (visualization)")


def _find_next_camera(reconstructor, all_matches, verified_matches, detector, n_images):
    """Find the next best camera to register"""
    registered = set(reconstructor.get_registered_camera_indices())
    unregistered = set(range(n_images)) - registered

    best_camera = None
    best_score = 0

    for cam_idx in unregistered:
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

        if pair[0] == camera_idx:
            pts_cam = result.inlier_points1
            pts_reg = result.inlier_points2
        else:
            pts_cam = result.inlier_points2
            pts_reg = result.inlier_points1

        for i, (pt_cam, pt_reg) in enumerate(zip(pts_cam, pts_reg)):
            for pt_idx, obs in enumerate(reconstructor.point_observations):
                if reg_idx in obs:
                    obs_pt = obs[reg_idx]
                    if np.linalg.norm(obs_pt - pt_reg) < 1.0:
                        points_2d_list.append(pt_cam)
                        point_indices_list.append(pt_idx)
                        break

    if len(points_2d_list) == 0:
        return np.empty((0, 2)), np.empty(0, dtype=int)

    return np.array(points_2d_list), np.array(point_indices_list)


def _visualize_colored_reconstruction(points_3d, colors, camera_poses, output_dir):
    """Visualize the colored 3D reconstruction"""
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')

    # Plot colored 3D points
    ax.scatter(
        points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
        c=colors / 255.0,  # Normalize to [0, 1]
        marker='.', s=2, alpha=0.6
    )

    # Plot camera centers
    camera_centers = []
    for pose in camera_poses.values():
        C = -pose.R.T @ pose.t
        camera_centers.append(C)
    camera_centers = np.array(camera_centers)

    ax.scatter(
        camera_centers[:, 0],
        camera_centers[:, 1],
        camera_centers[:, 2],
        c='red', marker='o', s=100, label='Cameras', edgecolors='black', linewidths=2
    )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_title(f'Colored 3D Reconstruction\n'
                f'{len(points_3d)} points, {len(camera_poses)} cameras')

    # Set equal aspect ratio
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
    output_path = output_dir / "phase4_colored_reconstruction.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved colored reconstruction to {output_path}")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test Phase 4: Densification & Colorization")
    parser.add_argument(
        "--images",
        type=str,
        default="Datasets/dinoRing",
        help="Directory containing input images"
    )
    parser.add_argument(
        "--max-cameras",
        type=int,
        default=10,
        help="Maximum number of cameras to register"
    )
    parser.add_argument(
        "--mvs",
        action="store_true",
        help="Enable MVS densification (computationally expensive)"
    )

    args = parser.parse_args()

    test_phase4(args.images, args.max_cameras, args.mvs)