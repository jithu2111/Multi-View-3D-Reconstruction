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
from src.utils.calibration_loader import try_load_intrinsics


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

    # Subsample images for large datasets to keep matching tractable
    # Use 3x the camera count to give the pipeline enough candidates for
    # registration, with a reasonable minimum and cap to avoid blowup
    max_images = min(max(max_cameras * 3, 40), len(images))
    if len(images) > max_images:
        # Uniformly sample around the ring to maintain even coverage
        step = len(images) / max_images
        indices = [int(i * step) for i in range(max_images)]
        images = [images[i] for i in indices]
        logger.info(f"Subsampled to {len(images)} images (step={step:.1f}) for tractable matching")

    image_shape = images[0].shape[:2]

    # Detect SIFT features
    logger.info("\n[2/9] Detecting SIFT features...")
    detector = SIFTDetector(n_features=2000, ratio_threshold=0.75)
    detector.detect_and_compute(images)

    # Try to load ground-truth intrinsics from calibration file
    known_K = try_load_intrinsics(image_dir)
    if known_K is not None:
        logger.info("Using ground-truth camera intrinsics from calibration file")
    else:
        logger.info("No calibration file found, using heuristic focal length estimation")

    # Match and verify — use Essential matrix RANSAC when K is known (much more accurate)
    logger.info("\n[3/9] Matching and verifying features...")
    all_matches = detector.match_all_pairs()
    if known_K is not None:
        verifier = RANSACVerifier(
            method='essential',
            ransac_threshold=0.5,
            min_inliers=50
        )
        verified_matches = verifier.verify_all_matches(all_matches, camera_matrix=known_K)
    else:
        verifier = RANSACVerifier(
            method='fundamental',
            ransac_threshold=0.5,
            min_inliers=50
        )
        verified_matches = verifier.verify_all_matches(all_matches)

    # Two-view initialization with auto-detected intrinsics
    logger.info("\n[4/9] Two-view initialization...")

    initializer = TwoViewInitializer(known_K=known_K)
    triangulator = Triangulator(min_parallax=1.0, max_reproj_error=2.0)

    # Rank all candidate pairs and try them in order until one produces valid points
    candidate_pairs = []
    for pair, ransac_result in verified_matches.items():
        if ransac_result.n_inliers >= 100:
            match = all_matches[pair]
            score = initializer._score_image_pair(match, ransac_result, image_shape)
            candidate_pairs.append((score, pair))
    candidate_pairs.sort(reverse=True)  # Best score first

    if not candidate_pairs:
        logger.error("No valid image pairs found for initialization")
        return

    reconstruction = None
    triangulated = None
    img1_idx, img2_idx = None, None

    for rank, (score, pair) in enumerate(candidate_pairs[:10]):  # Try top 10 pairs
        try:
            match = all_matches[pair]
            ransac_result = verified_matches[pair]
            recon = initializer.initialize_reconstruction(match, ransac_result, image_shape)
            tri = triangulator.triangulate(recon)
            tri = triangulator.filter_valid_points(tri)

            if len(tri.points_3d) >= 20:
                reconstruction = recon
                triangulated = tri
                img1_idx, img2_idx = pair
                logger.info(f"Pair {pair} (rank {rank+1}, score {score:.1f}): {len(tri.points_3d)} points ✓")
                break
            else:
                logger.warning(f"Pair {pair} (rank {rank+1}, score {score:.1f}): only {len(tri.points_3d)} points, trying next...")
        except Exception as e:
            logger.warning(f"Pair {pair} (rank {rank+1}) failed: {e}, trying next...")

    if reconstruction is None or triangulated is None:
        logger.error("All candidate initial pairs failed to produce valid triangulation")
        return

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
            _triangulate_new_points(reconstructor, best_camera, detector, all_matches, verified_matches, triangulator)

    n_cams, n_pts = reconstructor.get_reconstruction_size()
    logger.info(f"Sparse reconstruction: {n_cams} cameras, {n_pts} points")

    # Bundle Adjustment (two passes: first cleans up, second refines)
    logger.info("\n[6/9] Running Bundle Adjustment...")
    ba = BundleAdjuster(
        optimize_intrinsics=False,
        loss_function='huber',
        ftol=1e-8,
        max_nfev=500
    )

    # Pass 1: initial optimization
    optimized_poses, optimized_points, K_opt, error_pass1 = ba.optimize(
        reconstructor.registered_cameras,
        reconstructor.points_3d,
        reconstructor.point_observations,
        reconstructor.K,
        fix_first_camera=True
    )
    reconstructor.registered_cameras = optimized_poses
    reconstructor.points_3d = optimized_points
    logger.info(f"BA pass 1 error: {error_pass1:.3f} pixels")

    # Filter outlier points between passes
    _filter_outlier_points(reconstructor)

    # Pass 2: refine with cleaner points
    optimized_poses, optimized_points, K_opt, final_error = ba.optimize(
        reconstructor.registered_cameras,
        reconstructor.points_3d,
        reconstructor.point_observations,
        reconstructor.K,
        fix_first_camera=True
    )
    reconstructor.registered_cameras = optimized_poses
    reconstructor.points_3d = optimized_points
    logger.info(f"BA pass 2 error: {final_error:.3f} pixels")

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


def _filter_outlier_points(reconstructor, std_ratio=2.5):
    """Remove statistical outlier points from the reconstruction"""
    from scipy.spatial import cKDTree

    points = reconstructor.points_3d
    if len(points) < 20:
        return

    # Use KD-tree for nearest neighbor distances
    tree = cKDTree(points)
    k = min(10, len(points) - 1)
    distances, _ = tree.query(points, k=k + 1)
    mean_distances = distances[:, 1:].mean(axis=1)  # Exclude self

    # Statistical filtering
    global_mean = mean_distances.mean()
    global_std = mean_distances.std()
    threshold = global_mean + std_ratio * global_std
    inlier_mask = mean_distances < threshold

    n_removed = len(points) - np.sum(inlier_mask)
    if n_removed > 0:
        logger.info(f"Removed {n_removed} outlier points ({n_removed/len(points)*100:.1f}%)")
        reconstructor.points_3d = points[inlier_mask]
        reconstructor.point_observations = [
            obs for obs, keep in zip(reconstructor.point_observations, inlier_mask) if keep
        ]


def _triangulate_new_points(reconstructor, new_cam_idx, detector, all_matches, verified_matches, triangulator):
    """Triangulate new points visible in the new camera with duplicate detection"""
    new_pose = reconstructor.get_camera_pose(new_cam_idx)
    if new_pose is None:
        return

    points_added = 0
    duplicates_filtered = 0
    registered_cams = set(reconstructor.get_registered_camera_indices())
    registered_cams.remove(new_cam_idx)

    # Build KD-tree for existing points to detect duplicates
    from scipy.spatial import cKDTree
    existing_points = reconstructor.points_3d
    kdtree = None
    if len(existing_points) > 0:
        kdtree = cKDTree(existing_points)

    for reg_cam_idx in registered_cams:
        pair = tuple(sorted([new_cam_idx, reg_cam_idx]))
        if pair not in verified_matches:
            continue

        result = verified_matches[pair]
        if result.n_inliers < 30:
            continue

        reg_pose = reconstructor.get_camera_pose(reg_cam_idx)

        if pair[0] == new_cam_idx:
            pts_new = result.inlier_points1
            pts_reg = result.inlier_points2
        else:
            pts_new = result.inlier_points2
            pts_reg = result.inlier_points1

        from src.geometry.two_view import TwoViewReconstruction
        temp_recon = TwoViewReconstruction(
            img1_idx=new_cam_idx,
            img2_idx=reg_cam_idx,
            K=reconstructor.K,
            pose1=new_pose,
            pose2=reg_pose,
            inlier_points1=pts_new,
            inlier_points2=pts_reg,
            F=np.eye(3),
            E=np.eye(3)
        )

        triangulated = triangulator.triangulate(temp_recon)
        filtered = triangulator.filter_valid_points(triangulated)

        if len(filtered.points_3d) > 0:
            # Filter out duplicate points
            new_points = []
            new_observations = []

            for i in range(len(filtered.points_3d)):
                point_3d = filtered.points_3d[i]

                # Check if this point is too close to existing points
                is_duplicate = False
                if kdtree is not None:
                    # Search for nearby points (adaptive threshold based on scene scale)
                    scene_scale = np.ptp(existing_points, axis=0).max() if len(existing_points) > 1 else 1.0
                    dup_threshold = max(0.001, scene_scale * 0.002)  # 0.2% of scene size
                    distances, indices = kdtree.query(point_3d, k=1, distance_upper_bound=dup_threshold)
                    if distances < dup_threshold:  # Point already exists
                        is_duplicate = True
                        duplicates_filtered += 1

                if not is_duplicate:
                    new_points.append(point_3d)
                    # Start with the two cameras used for triangulation
                    obs = {
                        new_cam_idx: filtered.points_2d_img1[i],
                        reg_cam_idx: filtered.points_2d_img2[i]
                    }
                    new_observations.append(obs)

            # Add non-duplicate points and link multi-view observations
            if len(new_points) > 0:
                # First add the points with their initial observations
                start_idx = len(reconstructor.points_3d)
                reconstructor.add_points(np.array(new_points), new_observations)

                # Now search for additional observations in other registered cameras
                _link_multiview_observations(
                    reconstructor, new_cam_idx, reg_cam_idx,
                    start_idx, np.array(new_points), all_matches, verified_matches
                )

                points_added += len(new_points)

                # Update KD-tree with new points for subsequent iterations
                if kdtree is not None:
                    existing_points = np.vstack([existing_points, new_points])
                else:
                    existing_points = np.array(new_points)
                kdtree = cKDTree(existing_points)

    logger.info(f"Triangulated {points_added} new points using camera {new_cam_idx} "
                f"({duplicates_filtered} duplicates filtered)")


def _link_multiview_observations(
    reconstructor, cam1_idx, cam2_idx, start_point_idx, new_points, all_matches, verified_matches
):
    """
    Link newly triangulated points to observations in other registered cameras

    For each newly added 3D point, project it to all other registered cameras
    and check if there's a matching feature observation. This gives Bundle Adjustment
    more complete information about which cameras see which points.
    """
    registered_cams = set(reconstructor.get_registered_camera_indices())
    # Remove the two cameras already used for triangulation
    other_cams = registered_cams - {cam1_idx, cam2_idx}

    if len(other_cams) == 0:
        return

    n_observations_added = 0
    max_reproj_error = 3.0  # pixels

    for other_cam_idx in other_cams:
        other_pose = reconstructor.get_camera_pose(other_cam_idx)

        # Check matches with cam1
        pair1 = tuple(sorted([cam1_idx, other_cam_idx]))
        if pair1 not in verified_matches:
            continue

        result1 = verified_matches[pair1]
        if result1.n_inliers < 20:
            continue

        # Get the correct point arrays based on ordering
        if pair1[0] == cam1_idx:
            pts_cam1 = result1.inlier_points1
            pts_other = result1.inlier_points2
        else:
            pts_cam1 = result1.inlier_points2
            pts_other = result1.inlier_points1

        # Project new 3D points to the other camera
        P_other = other_pose.to_projection_matrix(reconstructor.K)
        points_hom = np.hstack([new_points, np.ones((len(new_points), 1))])
        projected = (P_other @ points_hom.T).T
        projected_2d = projected[:, :2] / projected[:, 2:3]

        # For each projected point, find if there's a matching observation
        for i, proj_pt in enumerate(projected_2d):
            point_idx = start_point_idx + i

            # Get the cam1 observation for this point
            cam1_obs = reconstructor.point_observations[point_idx].get(cam1_idx)
            if cam1_obs is None:
                continue

            # Find the closest match in pts_cam1 to cam1_obs
            dists = np.linalg.norm(pts_cam1 - cam1_obs, axis=1)
            min_idx = np.argmin(dists)

            if dists[min_idx] < 2.0:  # Found corresponding feature
                # Check if projection is close to the matched feature
                matched_pt_other = pts_other[min_idx]
                reproj_error = np.linalg.norm(proj_pt - matched_pt_other)

                if reproj_error < max_reproj_error:
                    # Add this observation
                    reconstructor.point_observations[point_idx][other_cam_idx] = matched_pt_other
                    n_observations_added += 1

    if n_observations_added > 0:
        logger.info(f"  Linked {n_observations_added} additional observations in other cameras")


def _find_next_camera(reconstructor, all_matches, verified_matches, detector, n_images):
    """Find the next best camera to register based on geometric quality"""
    registered = set(reconstructor.get_registered_camera_indices())
    unregistered = set(range(n_images)) - registered

    # Get camera centers for baseline computation
    camera_centers = {}
    for cam_idx in registered:
        pose = reconstructor.get_camera_pose(cam_idx)
        C = -pose.R.T @ pose.t
        camera_centers[cam_idx] = C

    # Compute mean camera center for diversity scoring
    if len(camera_centers) > 0:
        mean_center = np.mean(list(camera_centers.values()), axis=0)
    else:
        mean_center = np.zeros(3)

    best_camera = None
    best_score = 0

    for cam_idx in unregistered:
        n_matches = 0
        total_baseline = 0
        n_pairs = 0

        for reg_idx in registered:
            pair = tuple(sorted([cam_idx, reg_idx]))
            if pair in verified_matches:
                result = verified_matches[pair]
                n_matches += result.n_inliers

                # Estimate baseline using triangulated points if available
                # (Approximation: use mean point cloud distance)
                if len(reconstructor.points_3d) > 0:
                    # Compute average distance to existing 3D points
                    mean_point = np.mean(reconstructor.points_3d, axis=0)
                    reg_center = camera_centers[reg_idx]
                    baseline = np.linalg.norm(mean_point - reg_center)
                    total_baseline += baseline
                    n_pairs += 1

        if n_pairs > 0:
            avg_baseline = total_baseline / n_pairs
        else:
            avg_baseline = 1.0  # Default if no baseline estimate

        # Score combines:
        # 1. Number of matches (more is better)
        # 2. Average baseline (larger baseline = better geometry)
        # Weight matches more heavily, but use baseline as a tiebreaker
        score = n_matches * (1.0 + 0.1 * avg_baseline)

        if score > best_score:
            best_score = score
            best_camera = cam_idx

    return best_camera


def _get_2d_3d_correspondences(camera_idx, reconstructor, all_matches, verified_matches):
    """Get 2D-3D correspondences for a camera using spatial lookup"""
    points_2d_list = []
    point_indices_list = []

    registered = reconstructor.get_registered_camera_indices()

    # Build spatial lookup structure for all registered cameras
    # Maps (camera_idx, discretized_x, discretized_y) -> list of point indices
    from collections import defaultdict
    point_lookup = defaultdict(list)

    for pt_idx, obs in enumerate(reconstructor.point_observations):
        for cam_idx, pixel in obs.items():
            # Discretize to integer pixels for lookup
            key = (cam_idx, int(np.round(pixel[0])), int(np.round(pixel[1])))
            point_lookup[key].append(pt_idx)

    for reg_idx in registered:
        pair = tuple(sorted([camera_idx, reg_idx]))
        if pair not in verified_matches:
            continue

        result = verified_matches[pair]
        if result.n_inliers < 20:
            continue

        if pair[0] == camera_idx:
            pts_cam = result.inlier_points1
            pts_reg = result.inlier_points2
        else:
            pts_cam = result.inlier_points2
            pts_reg = result.inlier_points1

        # Match each correspondence to existing 3D points
        for pt_cam, pt_reg in zip(pts_cam, pts_reg):
            # Search in a 2x2 pixel window around the registered camera point
            x_base = int(np.round(pt_reg[0]))
            y_base = int(np.round(pt_reg[1]))

            matched = False
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    key = (reg_idx, x_base + dx, y_base + dy)
                    if key in point_lookup:
                        # Use the first match found (closest to discretized location)
                        pt_idx = point_lookup[key][0]
                        points_2d_list.append(pt_cam)
                        point_indices_list.append(pt_idx)
                        matched = True
                        break
                if matched:
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