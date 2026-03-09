"""
Surface Reconstruction Module

Converts point clouds to triangle meshes using:
- Poisson Surface Reconstruction
- Ball Pivoting Algorithm
- Alpha Shapes

Enables solid 3D object rendering instead of just points.
"""

import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Optional, Tuple
from ..utils.logger import logger


class MeshReconstructor:
    """
    Reconstructs triangle mesh surfaces from point clouds

    Converts sparse/dense point clouds into solid 3D meshes
    suitable for rendering and visualization.
    """

    @staticmethod
    def estimate_normals(
        points_3d: np.ndarray,
        k_neighbors: int = 30,
        orient_normals: bool = True
    ) -> np.ndarray:
        """
        Estimate surface normals for point cloud

        Args:
            points_3d: Nx3 point cloud
            k_neighbors: Number of neighbors for normal estimation
            orient_normals: Whether to orient normals consistently

        Returns:
            Nx3 normal vectors
        """
        logger.info(f"Estimating normals for {len(points_3d)} points...")

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)

        # Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k_neighbors)
        )

        # Orient normals consistently
        if orient_normals:
            pcd.orient_normals_consistent_tangent_plane(k=k_neighbors)

        normals = np.asarray(pcd.normals)
        logger.info(f"Estimated {len(normals)} normals")

        return normals

    @staticmethod
    def poisson_reconstruction(
        points_3d: np.ndarray,
        colors: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None,
        depth: int = 9,
        density_threshold: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Poisson Surface Reconstruction

        Creates watertight mesh from oriented point cloud.
        Best for complete, evenly distributed point clouds.

        Args:
            points_3d: Nx3 point cloud
            colors: Optional Nx3 RGB colors (0-255)
            normals: Optional Nx3 surface normals (will be estimated if None)
            depth: Octree depth (higher = more detail, default 9)
            density_threshold: Remove low-density vertices (0.01 = 1%)

        Returns:
            Tuple of (vertices, triangles, vertex_colors)
        """
        logger.info(f"Running Poisson reconstruction (depth={depth})...")

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)

        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

        # Estimate normals if not provided
        if normals is None:
            logger.info("Normals not provided, estimating...")
            normals = MeshReconstructor.estimate_normals(points_3d)

        pcd.normals = o3d.utility.Vector3dVector(normals)

        # Poisson reconstruction
        logger.info("Computing Poisson surface...")
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth, width=0, scale=1.1, linear_fit=False
        )

        # Remove low-density vertices (removes outliers)
        if density_threshold > 0:
            densities = np.asarray(densities)
            density_colors = plt.get_cmap('plasma')(
                (densities - densities.min()) / (densities.max() - densities.min())
            )
            vertices_to_remove = densities < np.quantile(densities, density_threshold)
            mesh.remove_vertices_by_mask(vertices_to_remove)

        logger.info(f"Mesh created: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

        # Extract mesh data
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)

        # Transfer colors to mesh
        if colors is not None and len(mesh.vertex_colors) == 0:
            # Map point colors to mesh vertices using nearest neighbor
            from scipy.spatial import cKDTree
            tree = cKDTree(points_3d)
            distances, indices = tree.query(vertices)
            vertex_colors = colors[indices]
        else:
            vertex_colors = np.asarray(mesh.vertex_colors) * 255 if len(mesh.vertex_colors) > 0 else None

        return vertices, triangles, vertex_colors

    @staticmethod
    def ball_pivoting_reconstruction(
        points_3d: np.ndarray,
        colors: Optional[np.ndarray] = None,
        normals: Optional[np.ndarray] = None,
        radii: Optional[list] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Ball Pivoting Algorithm (BPA) Surface Reconstruction

        Creates mesh by rolling a ball over the point cloud.
        Best for uniformly sampled point clouds without noise.

        Args:
            points_3d: Nx3 point cloud
            colors: Optional Nx3 RGB colors (0-255)
            normals: Optional Nx3 surface normals
            radii: List of ball radii (auto-computed if None)

        Returns:
            Tuple of (vertices, triangles, vertex_colors)
        """
        logger.info("Running Ball Pivoting reconstruction...")

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)

        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

        # Estimate normals if not provided
        if normals is None:
            logger.info("Normals not provided, estimating...")
            normals = MeshReconstructor.estimate_normals(points_3d)

        pcd.normals = o3d.utility.Vector3dVector(normals)

        # Compute radii if not provided
        if radii is None:
            # Estimate average point spacing
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radii = [avg_dist * r for r in [1.5, 2.0, 3.0]]
            logger.info(f"Auto-computed radii: {radii}")

        # Ball pivoting
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )

        logger.info(f"Mesh created: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

        # Extract mesh data
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        vertex_colors = np.asarray(mesh.vertex_colors) * 255 if len(mesh.vertex_colors) > 0 else None

        return vertices, triangles, vertex_colors

    @staticmethod
    def export_mesh_to_ply(
        file_path: str,
        vertices: np.ndarray,
        triangles: np.ndarray,
        colors: Optional[np.ndarray] = None
    ) -> None:
        """
        Export triangle mesh to PLY file

        Args:
            file_path: Output PLY file path
            vertices: Nx3 vertex positions
            triangles: Mx3 triangle indices
            colors: Optional Nx3 RGB colors (0-255)
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Exporting mesh to {file_path}...")

        # Create Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)

        if colors is not None:
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors / 255.0)

        # Compute normals for better rendering
        mesh.compute_vertex_normals()

        # Write PLY
        o3d.io.write_triangle_mesh(str(file_path), mesh, write_ascii=False)

        logger.info(f"Exported mesh: {len(vertices)} vertices, {len(triangles)} triangles")

    @staticmethod
    def visualize_mesh(
        vertices: np.ndarray,
        triangles: np.ndarray,
        colors: Optional[np.ndarray] = None,
        window_name: str = "3D Mesh Reconstruction"
    ) -> None:
        """
        Visualize mesh in Open3D viewer

        Args:
            vertices: Nx3 vertex positions
            triangles: Mx3 triangle indices
            colors: Optional Nx3 RGB colors (0-255)
            window_name: Viewer window title
        """
        logger.info("Launching 3D mesh viewer...")

        # Create mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)

        if colors is not None:
            mesh.vertex_colors = o3d.utility.Vector3dVector(colors / 255.0)

        # Compute normals for better rendering
        mesh.compute_vertex_normals()

        # Visualize
        o3d.visualization.draw_geometries(
            [mesh],
            window_name=window_name,
            width=1280,
            height=720,
            mesh_show_back_face=True
        )


def reconstruct_surface_from_ply(
    ply_path: str,
    output_path: str,
    method: str = "poisson",
    depth: int = 9
) -> None:
    """
    Load point cloud PLY and create surface mesh

    Args:
        ply_path: Input PLY file (point cloud)
        output_path: Output PLY file (mesh)
        method: "poisson" or "ball_pivoting"
        depth: Poisson octree depth (9-11 recommended)
    """
    logger.info(f"Loading point cloud from {ply_path}...")

    # Load PLY
    pcd = o3d.io.read_point_cloud(ply_path)
    points_3d = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) * 255 if pcd.has_colors() else None

    logger.info(f"Loaded {len(points_3d)} points")

    # Reconstruct surface
    if method == "poisson":
        vertices, triangles, vertex_colors = MeshReconstructor.poisson_reconstruction(
            points_3d, colors, depth=depth
        )
    elif method == "ball_pivoting":
        vertices, triangles, vertex_colors = MeshReconstructor.ball_pivoting_reconstruction(
            points_3d, colors
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    # Export mesh
    MeshReconstructor.export_mesh_to_ply(output_path, vertices, triangles, vertex_colors)

    logger.info(f"Mesh saved to {output_path}")


# Avoid matplotlib import error
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None