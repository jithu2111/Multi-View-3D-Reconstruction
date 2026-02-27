"""
Bundle Adjustment Module

Implements non-linear least squares optimization to jointly refine:
- 3D point positions
- Camera poses (rotation and translation)
- Optionally: camera intrinsics

Minimizes total reprojection error: Σ ||x_ij - π(P_j, X_i)||²
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from typing import List, Dict, Tuple, Optional
import cv2
from ..geometry.two_view import CameraPose
from ..utils.logger import logger


class BundleAdjuster:
    """
    Bundle Adjustment optimizer for Structure from Motion

    Jointly optimizes camera poses and 3D point positions to minimize
    reprojection error across all observations.
    """

    def __init__(
        self,
        optimize_intrinsics: bool = False,
        loss_function: str = 'huber',
        ftol: float = 1e-6,
        max_nfev: int = 100
    ):
        """
        Initialize Bundle Adjuster

        Args:
            optimize_intrinsics: Whether to optimize camera intrinsics
            loss_function: Robust loss function ('linear', 'huber', 'soft_l1', 'cauchy')
            ftol: Tolerance for termination by cost function change
            max_nfev: Maximum number of function evaluations
        """
        self.optimize_intrinsics = optimize_intrinsics
        self.loss_function = loss_function
        self.ftol = ftol
        self.max_nfev = max_nfev

        logger.info(f"Initialized BundleAdjuster: "
                   f"optimize_intrinsics={optimize_intrinsics}, "
                   f"loss={loss_function}")

    def optimize(
        self,
        camera_poses: Dict[int, CameraPose],
        points_3d: np.ndarray,
        point_observations: List[Dict],
        K: np.ndarray,
        fix_first_camera: bool = True
    ) -> Tuple[Dict[int, CameraPose], np.ndarray, np.ndarray, float]:
        """
        Run bundle adjustment optimization

        Args:
            camera_poses: Dict mapping img_idx -> CameraPose
            points_3d: Nx3 array of 3D points
            point_observations: List of N dicts {img_idx: (x, y)}
            K: 3x3 camera intrinsic matrix
            fix_first_camera: Whether to fix the first camera (recommended)

        Returns:
            Tuple of (optimized_poses, optimized_points_3d, K_optimized, final_error)
        """
        logger.info(f"Starting Bundle Adjustment: "
                   f"{len(camera_poses)} cameras, {len(points_3d)} points")

        # Prepare data for optimization
        camera_indices = sorted(camera_poses.keys())
        idx_to_cam = {idx: i for i, idx in enumerate(camera_indices)}

        n_cameras = len(camera_indices)
        n_points = len(points_3d)

        # Build observation list
        observations = []
        for pt_idx, obs_dict in enumerate(point_observations):
            for img_idx, pt_2d in obs_dict.items():
                if img_idx in idx_to_cam:
                    cam_idx = idx_to_cam[img_idx]
                    observations.append((cam_idx, pt_idx, pt_2d))

        n_observations = len(observations)

        logger.info(f"Total observations: {n_observations}")

        # Convert to parameter vector
        x0 = self._pack_parameters(
            camera_poses, camera_indices, points_3d, K, fix_first_camera
        )

        # Define bounds (optional - helps stability)
        bounds = self._compute_bounds(n_cameras, n_points, fix_first_camera)

        # Build sparsity structure for efficiency
        A = self._bundle_adjustment_sparsity(
            n_cameras, n_points, observations, fix_first_camera
        )

        # Optimize
        logger.info("Running optimization...")
        result = least_squares(
            fun=self._residuals,
            x0=x0,
            jac_sparsity=A,
            bounds=bounds,
            method='trf',
            ftol=self.ftol,
            max_nfev=self.max_nfev,
            loss=self.loss_function,
            verbose=0,
            args=(n_cameras, n_points, observations, K, fix_first_camera)
        )

        logger.info(f"Optimization complete: "
                   f"success={result.success}, "
                   f"iterations={result.nfev}, "
                   f"cost={result.cost:.6f}")

        # Unpack optimized parameters
        optimized_poses, optimized_points, K_opt = self._unpack_parameters(
            result.x, camera_indices, n_cameras, n_points, K, fix_first_camera
        )

        # Compute final reprojection error
        final_error = np.sqrt(result.cost * 2 / n_observations)

        logger.info(f"Final mean reprojection error: {final_error:.3f} pixels")

        return optimized_poses, optimized_points, K_opt, final_error

    def _pack_parameters(
        self,
        camera_poses: Dict[int, CameraPose],
        camera_indices: List[int],
        points_3d: np.ndarray,
        K: np.ndarray,
        fix_first_camera: bool = True
    ) -> np.ndarray:
        """
        Pack camera poses and 3D points into parameter vector

        Format: [cam1_params, cam2_params, ..., point0_xyz, point1_xyz, ..., intrinsics]
        Camera params: [rvec (3), tvec (3)] = 6 parameters per camera
        Note: If fix_first_camera=True, camera 0 is NOT included
        """
        params = []

        # Pack camera parameters (rotation vector + translation)
        # Skip first camera if fixing it
        start_idx = 1 if fix_first_camera else 0
        for i in range(start_idx, len(camera_indices)):
            img_idx = camera_indices[i]
            pose = camera_poses[img_idx]
            rvec, _ = cv2.Rodrigues(pose.R)
            params.append(rvec.ravel())
            params.append(pose.t)

        # Pack 3D points
        params.append(points_3d.ravel())

        # Pack intrinsics (focal length) if optimizing
        if self.optimize_intrinsics:
            params.append(np.array([K[0, 0]]))  # fx (assuming fx = fy)

        return np.concatenate(params)

    def _unpack_parameters(
        self,
        params: np.ndarray,
        camera_indices: List[int],
        n_cameras: int,
        n_points: int,
        K_initial: np.ndarray,
        fix_first_camera: bool
    ) -> Tuple[Dict[int, CameraPose], np.ndarray, np.ndarray]:
        """Unpack parameter vector into camera poses and 3D points"""
        camera_poses = {}
        offset = 0

        # Unpack camera parameters
        if fix_first_camera:
            # First camera is fixed at identity
            camera_poses[camera_indices[0]] = CameraPose(
                R=np.eye(3),
                t=np.zeros(3)
            )
            start_cam = 1
        else:
            start_cam = 0

        for i in range(start_cam, n_cameras):
            rvec = params[offset:offset+3]
            tvec = params[offset+3:offset+6]
            offset += 6

            R, _ = cv2.Rodrigues(rvec)
            camera_poses[camera_indices[i]] = CameraPose(R, tvec)

        # Unpack 3D points
        points_3d = params[offset:offset+n_points*3].reshape(n_points, 3)
        offset += n_points * 3

        # Unpack intrinsics
        K = K_initial.copy()
        if self.optimize_intrinsics and offset < len(params):
            fx = params[offset]
            K[0, 0] = fx
            K[1, 1] = fx

        return camera_poses, points_3d, K

    def _residuals(
        self,
        params: np.ndarray,
        n_cameras: int,
        n_points: int,
        observations: List[Tuple],
        K: np.ndarray,
        fix_first_camera: bool
    ) -> np.ndarray:
        """
        Compute residuals (reprojection errors) for all observations

        Returns:
            Flattened array of residuals (2 * n_observations,)
        """
        # Unpack parameters
        camera_indices = list(range(n_cameras))
        poses, points_3d, K_opt = self._unpack_parameters(
            params, camera_indices, n_cameras, n_points, K, fix_first_camera
        )

        residuals = []

        for cam_idx, pt_idx, pt_2d in observations:
            # Project 3D point to image
            pose = poses[cam_idx]
            X = points_3d[pt_idx]

            # Transform to camera coordinates
            X_cam = pose.R @ X + pose.t

            # Perspective projection
            x_proj = K_opt @ X_cam
            if x_proj[2] > 0:  # Valid depth
                x_proj = x_proj[:2] / x_proj[2]
            else:
                x_proj = np.array([0.0, 0.0])

            # Residual
            residual = pt_2d - x_proj
            residuals.append(residual)

        return np.concatenate(residuals)

    def _bundle_adjustment_sparsity(
        self,
        n_cameras: int,
        n_points: int,
        observations: List[Tuple],
        fix_first_camera: bool
    ) -> lil_matrix:
        """
        Build sparsity structure for Jacobian

        Each observation affects:
        - 6 camera parameters (or 0 if camera is fixed)
        - 3 point parameters

        Returns:
            Sparse matrix indicating which parameters affect each residual
        """
        n_params_per_camera = 6
        camera_params_start = 0
        if fix_first_camera:
            n_camera_params = (n_cameras - 1) * n_params_per_camera
        else:
            n_camera_params = n_cameras * n_params_per_camera

        points_params_start = n_camera_params
        n_point_params = n_points * 3

        intrinsics_start = points_params_start + n_point_params
        if self.optimize_intrinsics:
            n_total_params = intrinsics_start + 1
        else:
            n_total_params = intrinsics_start

        n_residuals = len(observations) * 2  # 2 residuals per observation (x, y)

        A = lil_matrix((n_residuals, n_total_params), dtype=int)

        for obs_idx, (cam_idx, pt_idx, _) in enumerate(observations):
            residual_idx = obs_idx * 2

            # Camera parameters
            if fix_first_camera:
                if cam_idx > 0:
                    cam_param_idx = (cam_idx - 1) * n_params_per_camera
                    A[residual_idx:residual_idx+2, cam_param_idx:cam_param_idx+6] = 1
            else:
                cam_param_idx = cam_idx * n_params_per_camera
                A[residual_idx:residual_idx+2, cam_param_idx:cam_param_idx+6] = 1

            # Point parameters
            point_param_idx = points_params_start + pt_idx * 3
            A[residual_idx:residual_idx+2, point_param_idx:point_param_idx+3] = 1

            # Intrinsics (if optimizing)
            if self.optimize_intrinsics:
                A[residual_idx:residual_idx+2, intrinsics_start] = 1

        return A

    def _compute_bounds(
        self,
        n_cameras: int,
        n_points: int,
        fix_first_camera: bool
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute parameter bounds for optimization

        Returns:
            Tuple of (lower_bounds, upper_bounds)
        """
        n_params_per_camera = 6
        if fix_first_camera:
            n_camera_params = (n_cameras - 1) * n_params_per_camera
        else:
            n_camera_params = n_cameras * n_params_per_camera

        n_point_params = n_points * 3

        # Camera parameters: no strict bounds (use wide range)
        lower_camera = np.full(n_camera_params, -np.inf)
        upper_camera = np.full(n_camera_params, np.inf)

        # Point parameters: no strict bounds
        lower_points = np.full(n_point_params, -np.inf)
        upper_points = np.full(n_point_params, np.inf)

        lower = np.concatenate([lower_camera, lower_points])
        upper = np.concatenate([upper_camera, upper_points])

        # Intrinsics bounds (if optimizing)
        if self.optimize_intrinsics:
            lower = np.append(lower, 100.0)  # Minimum focal length
            upper = np.append(upper, 10000.0)  # Maximum focal length

        return lower, upper