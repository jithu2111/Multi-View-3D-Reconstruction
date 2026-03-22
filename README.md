# Multi-View 3D Reconstruction Pipeline

A from-scratch Structure from Motion (SfM) and Multi-View Stereo (MVS) pipeline that reconstructs complete 360-degree 3D models from multiple uncalibrated images. Built as a graduate course project for Computer Vision at Georgia State University (Spring 2026).

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Architecture](#pipeline-architecture)
- [Concepts and Theory](#concepts-and-theory)
- [Challenges and Solutions](#challenges-and-solutions)
- [Project Structure](#project-structure)
- [Outputs](#outputs)
- [Datasets](#datasets)

---

## Overview

This project takes a set of overlapping photographs of an object (e.g., the Middlebury Dino or Temple datasets) and automatically produces a colored 3D point cloud and triangle mesh. The pipeline is divided into four phases:

1. **Feature Detection & Matching** -- SIFT keypoints with RANSAC geometric verification
2. **Two-View Initialization & Triangulation** -- Camera pose recovery and initial 3D points
3. **Incremental Reconstruction & Bundle Adjustment** -- Sequential camera registration with global optimization
4. **Densification & Colorization** -- Multi-View Stereo depth maps, colored point clouds, and mesh generation

No deep learning is used. Every component relies on classical computer vision and optimization techniques.

---

## Installation

### Prerequisites

- Python 3.8 or later
- pip

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd Project
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Dependencies

| Package                | Version   | Purpose                                    |
|------------------------|-----------|--------------------------------------------|
| numpy                  | >= 1.24.0 | Matrix operations, linear algebra          |
| opencv-python          | >= 4.8.0  | Image I/O, projective geometry             |
| opencv-contrib-python  | >= 4.8.0  | SIFT detector (patented, in contrib)       |
| scipy                  | >= 1.11.0 | Sparse least-squares (Bundle Adjustment)   |
| matplotlib             | >= 3.7.0  | 2D/3D plotting and visualization           |
| open3d                 | >= 0.18.0 | Point cloud processing, mesh reconstruction|
| tqdm                   | >= 4.65.0 | Progress bars                              |
| pillow                 | >= 10.0.0 | Image format support                       |
| tomlkit                | >= 0.11.0 | Configuration file parsing                 |

---

## Usage

### Prepare a Dataset

Place your images in a directory. The pipeline supports JPG, PNG, BMP, and TIFF formats. If a Middlebury-format calibration file (`*_par.txt`) is present in the image directory, ground-truth camera intrinsics will be loaded automatically; otherwise the focal length is estimated heuristically as `1.2 * max(width, height)`.

Datasets used during development:
- `Datasets/dinoRing` -- 48 images of a toy dinosaur (Middlebury Dino Ring)
- `TempleSparseRing` -- 16 images of a temple model (Middlebury Temple Sparse Ring)

### Phase 1: Feature Detection & Matching

```bash
python test_phase1.py --images <path-to-images>
```

Detects SIFT keypoints in all images, matches features across pairs with Lowe's ratio test, and filters outliers using RANSAC on the fundamental matrix. Outputs visualizations to `data/output/`.

### Phase 2: Two-View Initialization

```bash
python test_phase2.py --images <path-to-images>
```

Selects the best image pair, estimates (or loads) camera intrinsics, recovers relative camera pose from the essential matrix, and triangulates an initial sparse 3D point cloud.

### Phase 3: Incremental Reconstruction & Bundle Adjustment

```bash
python test_phase3.py --images <path-to-images> --max-cameras 5
```

Registers additional cameras via PnP (Perspective-n-Point), triangulates new 3D points from each newly registered view, and runs Bundle Adjustment to jointly minimize reprojection error across all cameras and points.

### Phase 4: Full Pipeline (Densification, Colorization, Mesh)

```bash
python test_phase4.py --images <path-to-images> --max-cameras 10
```

Runs the complete pipeline: sparse reconstruction (Phases 1-3), point cloud colorization, statistical outlier removal, PLY export, and surface mesh generation via Poisson or Ball Pivoting reconstruction.

To enable Multi-View Stereo dense depth map estimation (computationally expensive):

```bash
python test_phase4.py --images <path-to-images> --max-cameras 10 --mvs
```

### Google Colab

A Jupyter notebook is available in `colab/mvs_reconstruction.ipynb` for running the MVS densification in the cloud with GPU acceleration.

---

## Pipeline Architecture

```
Input Images
     |
     v
[SIFT Feature Detection] -----> Keypoints + 128-D descriptors per image
     |
     v
[FLANN Matching + Ratio Test] -> Pairwise feature correspondences
     |
     v
[RANSAC Geometric Verification] -> Fundamental/Essential matrix + inlier matches
     |
     v
[Two-View Initialization] -----> Initial camera pair poses + triangulated 3D points
     |
     v
[Incremental PnP Registration] -> Sequential camera pose estimation from 2D-3D matches
     |                              |
     |                    [Triangulate New Points]
     |                              |
     v                              v
[Bundle Adjustment] ------------> Jointly optimized cameras + 3D points
     |
     +-------> [Sparse Colorization] -> Colored sparse point cloud (.ply)
     |
     +-------> [MVS Plane Sweep Stereo] -> Per-view depth maps
     |              |
     |         [Multi-View Consistency Filter]
     |              |
     |         [Depth Map Fusion] -> Dense colored point cloud (.ply)
     |
     +-------> [Poisson / Ball Pivoting] -> Triangle mesh (.ply)
```

---

## Concepts and Theory

### 1. SIFT (Scale-Invariant Feature Transform)

SIFT detects keypoints that are invariant to scale, rotation, and partially invariant to illumination and viewpoint changes. Each keypoint is described by a 128-dimensional histogram of oriented gradients. We use FLANN (Fast Library for Approximate Nearest Neighbors) with KD-trees for efficient descriptor matching, followed by Lowe's ratio test (threshold = 0.75) to reject ambiguous matches.

### 2. RANSAC (Random Sample Consensus)

Raw feature matches contain outliers from repetitive textures, reflections, or mismatches. RANSAC fits a geometric model (fundamental matrix for uncalibrated cameras, essential matrix when intrinsics are known) to randomly sampled minimal subsets, then counts inliers within a reprojection threshold. After many iterations, the model with the most inliers is selected. This robustly separates correct correspondences from outliers -- typically achieving 60-90% inlier ratios on our datasets.

### 3. Epipolar Geometry: Fundamental and Essential Matrices

The **Fundamental matrix** F (3x3, rank 2) encodes the epipolar constraint between two uncalibrated views: for corresponding points x and x', we have x'^T F x = 0. When camera intrinsics K are known, the **Essential matrix** E = K^T F K provides a direct route to decomposing relative rotation R and translation t between cameras via SVD. Using the essential matrix with known intrinsics yields significantly more accurate pose estimation than the fundamental matrix alone.

### 4. Triangulation (DLT)

Given two camera projection matrices P1 and P2 and corresponding 2D points, triangulation recovers the 3D point X by solving the overconstrained system derived from x = PX. We use OpenCV's `triangulatePoints` (based on Direct Linear Transform), followed by validation checks:
- **Cheirality check**: Points must have positive depth in both cameras
- **Reprojection error**: Projected 3D points must land close to observed 2D points (< 4 pixels)
- **Parallax angle**: Viewing rays must subtend a sufficient angle (> 1 degree) for accurate depth estimation

### 5. Perspective-n-Point (PnP)

PnP estimates the pose of a new camera given known 2D-3D correspondences (2D points in the new image matched to existing 3D points). We use `cv2.solvePnPRansac` with the iterative algorithm, requiring at least 30 inliers for a valid registration. After each new camera is registered, we triangulate additional 3D points visible in the new view to grow the reconstruction incrementally.

### 6. Bundle Adjustment

Bundle Adjustment is the cornerstone of SfM accuracy. It jointly optimizes all camera poses (rotation + translation, parameterized as Rodrigues vectors) and all 3D point positions by minimizing the total reprojection error:

```
minimize  sum_ij  || x_ij - project(K, R_j, t_j, X_i) ||^2
```

We use `scipy.optimize.least_squares` with the Trust Region Reflective (TRF) method, Huber robust loss to downweight outliers, and an analytically computed Jacobian sparsity pattern. The sparsity structure is critical: each observation only depends on one camera (6 parameters) and one point (3 parameters), creating a sparse block structure that enables efficient solving. The first camera is fixed to remove gauge freedom (the reconstruction is defined up to a similarity transform).

### 7. Multi-View Stereo (MVS) -- Plane Sweep Stereo

Plane sweep stereo densifies the sparse reconstruction by estimating per-pixel depth maps:
1. For each reference camera, sweep a set of fronto-parallel depth planes through the scene
2. At each depth, warp source images to the reference view using the known camera geometry
3. Compute a photometric matching cost (windowed Sum of Absolute Differences via box filter)
4. Select the depth with minimum cost at each pixel (winner-takes-all)
5. Compute confidence from the cost distribution and filter low-confidence pixels

Source views are selected based on viewing angle: cameras with a baseline angle of 5-30 degrees (ideal ~10 degrees) provide enough parallax for depth estimation without excessive appearance change.

### 8. Multi-View Consistency Filtering

Individual depth maps are noisy. We enforce consistency by cross-checking: a 3D point backprojected from one depth map must reproject into other depth maps with a matching depth value (relative difference < 1%). Only points confirmed by at least 3 views are retained, which dramatically reduces noise and phantom geometry.

### 9. Point Cloud Colorization

3D points are colored by projecting them into all visible camera views and computing a weighted average of sampled pixel colors. Weights combine:
- **Viewing angle**: Perpendicular views (cos^2 weighting) provide less distorted color
- **Distance**: Gaussian weighting centered on the median distance to avoid overweighting extreme views

### 10. Surface Reconstruction

The final colored point cloud is converted to a triangle mesh using either:
- **Poisson Surface Reconstruction**: Creates watertight meshes by solving a spatial Poisson equation; best for complete point clouds
- **Ball Pivoting Algorithm (BPA)**: Rolls a virtual ball over the point cloud to connect nearby points into triangles; better for sparse or incomplete data

---

## Challenges and Solutions

The development of this pipeline was an iterative process. Below are the major challenges encountered and how they were resolved, in chronological order matching the git history.

### Challenge 1: Bundle Adjustment Sparsity Matrix Error

**Problem**: The initial Bundle Adjustment implementation had an incorrect Jacobian sparsity pattern. When a camera was "fixed" (excluded from optimization to remove gauge freedom), the sparsity matrix indices were not adjusted to account for the missing camera parameters, causing a dimension mismatch that either crashed the optimizer or produced silently wrong results.

**Solution**: Introduced an `effective_cam_idx` calculation that maps internal camera indices to parameter indices, correctly skipping the fixed camera. Additionally, the fixed camera's pose was being hardcoded to identity (R=I, t=0) during unpacking, which was wrong when the fixed camera wasn't the first one or wasn't at the origin. Fixed by storing and restoring the actual pose of the whichever camera is closest to the origin.

### Challenge 2: Fundamental vs. Essential Matrix -- The Math Pipeline

**Problem**: When ground-truth camera intrinsics K were available (from Middlebury calibration files), the pipeline was still using RANSAC to estimate the Fundamental matrix and then computing E = K^T F K. This double transformation introduced numerical errors and produced poor pose recovery, especially for the Dino and Temple datasets where accurate intrinsics are known.

**Solution**: Added a calibration file loader (`calibration_loader.py`) that auto-detects `*_par.txt` files in the dataset directory. When K is known, RANSAC now directly estimates the Essential matrix using calibrated point coordinates, bypassing the fundamental matrix entirely. The `TwoViewInitializer` was updated to accept a `known_K` parameter that switches between the two code paths. This significantly improved pose accuracy and downstream triangulation quality.

### Challenge 3: Depth Representation -- Euclidean Distance vs. Z-Depth

**Problem**: The MVS plane sweep was using inconsistent depth representations across three functions. `_backproject_to_depth` was normalizing camera rays to unit length and multiplying by depth (treating depth as Euclidean distance from the camera center), but `_estimate_depth_range` was computing z-depth (the Z component in camera coordinates), and the depth map fusion was mixing both conventions. This caused barrel distortion at image edges -- pixels far from the principal point were backprojected to incorrect 3D locations because their ray direction differs significantly from the optical axis.

**Solution**: Standardized on **z-depth** (depth along the optical axis) everywhere. Instead of normalizing rays and scaling by Euclidean depth, the code now scales unnormalized rays so that their z-component equals the target depth: `points_cam = rays * (depth / rays[:, 2])`. This is consistent with how plane sweep stereo actually works (sweeping planes perpendicular to the optical axis) and eliminated the edge distortion artifacts.

### Challenge 4: Per-Camera Depth Range Estimation

**Problem**: The depth range for plane sweep was computed once using only the first camera's viewpoint. For a 360-degree ring of cameras, the scene appears at very different depths depending on the camera position. Using a single global depth range meant many cameras were searching in the wrong depth interval, producing empty or garbage depth maps.

**Solution**: Changed `_estimate_depth_range` to accept a single camera pose instead of the entire camera dictionary. Now each camera computes its own depth range by transforming the sparse 3D points into its own coordinate frame and taking the 5th-95th percentile of z-depths. This ensures every camera sweeps through the depth range where the scene actually exists from its perspective.

### Challenge 5: Depth Map Noise from Naive Fusion

**Problem**: The initial depth map fusion simply concatenated all backprojected points from all views without any consistency check. This produced extremely noisy point clouds because:
- Each depth map has per-pixel noise from the winner-takes-all depth selection
- Background pixels with no valid depth were assigned arbitrary values
- Overlapping views contributed redundant or conflicting points

**Solution**: Implemented multi-view reprojection consistency filtering. After backprojecting each depth map's pixels to 3D, the code projects those 3D points into every other depth map and checks whether the observed depth agrees (relative difference < 1%). Only points confirmed by at least 3 independent views survive. This reduced noise by an order of magnitude while preserving genuine surface points.

### Challenge 6: Source View Selection for Stereo

**Problem**: MVS was using all other cameras as source views for stereo matching with each reference view. Cameras on the opposite side of the 360-degree ring (viewing angle > 90 degrees) share almost no visual content with the reference, contributing only noise to the matching cost. This degraded depth map quality and wasted computation.

**Solution**: Added `_select_source_views` which selects up to 5 source cameras based on their viewing angle relative to the reference. Cameras with baseline angles between 5 and 30 degrees are preferred, with a sweet spot around 10 degrees (enough parallax for depth estimation, but not so much that appearance changes drastically). A fallback mechanism relaxes the angle constraints if fewer than 2 candidates pass the filter.

### Challenge 7: Pixel-Level Noise in Matching Cost

**Problem**: The photometric matching cost was computed per-pixel (L1 difference between reference and warped source), which is sensitive to noise, aliasing, and minor calibration errors. This produced noisy depth maps with many single-pixel depth spikes.

**Solution**: Replaced per-pixel cost with **windowed Sum of Absolute Differences (SAD)** using OpenCV's `cv2.boxFilter`. After computing the per-pixel L1 cost, a box filter of size `window_size x window_size` (default 5x5) aggregates costs over a local neighborhood. This provides patch-level matching robustness similar to NCC (Normalized Cross-Correlation) but at a fraction of the computational cost.

### Challenge 8: Dark Background Noise

**Problem**: The datasets have a black background. Since black matches black everywhere (zero photometric cost at any depth), the plane sweep confidently assigned arbitrary depth values to all background pixels. These phantom points formed a dense cloud of noise surrounding the actual object.

**Solution**: Added a foreground mask based on pixel brightness. Pixels with mean intensity below 30 (out of 255, roughly 12%) are masked out before depth map generation. This simple heuristic effectively eliminates background noise for studio-captured datasets with controlled dark backgrounds.

### Challenge 9: Duplicate Points and Missing Observations

**Problem**: When registering new cameras incrementally, newly triangulated points often duplicated existing 3D points (the same physical surface point triangulated from a different camera pair). This inflated the point count and confused Bundle Adjustment. Additionally, observations were only linked to the two cameras used for triangulation, even when the point was visible in other registered cameras, starving Bundle Adjustment of constraints.

**Solution**: Two-part fix:
1. **Duplicate detection**: Before adding new points, a KD-tree of existing points is queried. Points within 0.2% of the scene scale from an existing point are considered duplicates and discarded.
2. **Multi-view observation linking**: After adding new points, `_link_multiview_observations` projects each new 3D point into all other registered cameras, looks up matching 2D features via RANSAC-verified correspondences, and adds confirmed observations (reprojection error < 3 pixels). This gives Bundle Adjustment significantly more information to work with.

### Challenge 10: Scalability with Large Image Sets

**Problem**: The Dino Ring dataset has 48 images, producing 1,128 image pairs for matching. Larger datasets (312 images) produce ~48,000 pairs, making exhaustive matching intractable.

**Solution**: For large datasets, images are uniformly subsampled around the ring to a manageable count (3x the target camera count, minimum 40, capped at the full set). Uniform spacing maintains coverage while keeping the O(N^2) matching cost feasible.

---

## Project Structure

```
.
├── src/
│   ├── features/
│   │   └── sift_detector.py         # SIFT detection, FLANN matching, ratio test
│   ├── geometry/
│   │   ├── ransac.py                # RANSAC for F/E/H matrix estimation
│   │   ├── two_view.py             # Two-view initialization, pose recovery
│   │   ├── triangulation.py        # DLT triangulation, cheirality checks
│   │   └── pnp.py                  # PnP solver, incremental reconstructor
│   ├── optimization/
│   │   └── bundle_adjustment.py    # Sparse least-squares BA with Huber loss
│   ├── densification/
│   │   ├── mvs.py                  # Plane sweep stereo, depth map fusion
│   │   └── colorization.py        # Multi-view weighted colorization
│   ├── visualization/
│   │   └── mesh_reconstruction.py  # Poisson and Ball Pivoting meshing
│   └── utils/
│       ├── logger.py               # Timestamped file + console logging
│       ├── image_loader.py         # Image loading with auto-resize
│       ├── calibration_loader.py   # Middlebury *_par.txt intrinsics loader
│       └── ply_export.py           # ASCII/binary PLY writer
├── test_phase1.py                   # Phase 1 test script
├── test_phase2.py                   # Phase 2 test script
├── test_phase3.py                   # Phase 3 test script
├── test_phase4.py                   # Phase 4 full pipeline script
├── colab/
│   └── mvs_reconstruction.ipynb    # Google Colab notebook
├── data/
│   ├── input/                       # Input images
│   └── output/                      # Generated PLY files and visualizations
├── logs/                            # Timestamped execution logs
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## Outputs

Running the full pipeline produces the following files in `data/output/`:

| File | Description |
|------|-------------|
| `phase4_reconstruction_points.ply` | Colored sparse (or dense) point cloud |
| `phase4_reconstruction_cameras.ply` | Camera frustums for visualization |
| `phase4_reconstruction_mesh.ply` | Triangle mesh surface |
| `phase4_colored_reconstruction.png` | Matplotlib 3D scatter plot |

PLY files can be viewed in:
- **MeshLab** (free, cross-platform)
- **CloudCompare** (free, cross-platform)
- **Blender** (Import PLY)

---

## Datasets

This project was developed and tested using the [Middlebury Multi-View Stereo benchmark](https://vision.middlebury.edu/mview/):

- **dinoRing**: 48 images of a toy dinosaur captured in a ring configuration
- **templeSparseRing**: 16 images of a temple model

These datasets include ground-truth camera calibration files (`*_par.txt`) that the pipeline auto-detects for accurate intrinsics.

To use your own images, simply point `--images` to a directory of overlapping photographs. Without a calibration file, the pipeline will estimate focal length heuristically.