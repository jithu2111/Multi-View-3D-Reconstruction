# Multi-View 3D Reconstruction Pipeline


## Overview

This project implements a comprehensive Structure from Motion (SfM) pipeline for reconstructing complete 360° 3D models from multiple uncalibrated images. The system addresses the fundamental limitations of traditional stereo vision by using global optimization through Bundle Adjustment.

## Problem Statement

Traditional stereo vision is limited to:
- **2.5D depth maps** with severe occlusions
- **Drift accumulation** preventing loop closure for 360° models
- **Lack of global optimization** causing misaligned structures

## Approach

### Front-End (Feature Processing)
- SIFT feature detection and matching across N uncalibrated images
- RANSAC-based geometric verification to filter spurious matches

### Back-End (3D Reconstruction)
- Incremental reconstruction with two-view initialization
- Sequential camera registration using Perspective-n-Point (PnP)
- **Bundle Adjustment**: Non-linear least squares optimization jointly refining 3D points and camera parameters

### Densification
- Multi-View Stereo (MVS) to convert sparse point cloud to dense surface

## Project Structure

```
.
├── src/
│   ├── features/          # Feature detection and matching
│   │   └── sift_detector.py
│   ├── geometry/          # Geometric computations
│   │   ├── ransac.py
│   │   ├── two_view.py
│   │   └── triangulation.py
│   ├── optimization/      # Bundle Adjustment
│   │   └── bundle_adjustment.py
│   ├── densification/     # Multi-View Stereo
│   │   └── mvs.py
│   ├── visualization/     # 3D visualization
│   │   └── viewer.py
│   └── utils/             # Utilities
│       ├── logger.py
│       └── image_loader.py
├── data/
│   ├── input/             # Input images
│   ├── output/            # Output point clouds and visualizations
│   └── benchmarks/        # Middlebury MVS benchmark data
├── tests/                 # Unit tests
├── logs/                  # Execution logs
└── requirements.txt       # Python dependencies
```

## Installation

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Phase 1: Feature Detection and Matching

Test SIFT feature detection and RANSAC verification:

```bash
python test_phase1.py --images Datasets/dinoRing
```

This will:
- Load images from the specified directory
- Detect SIFT features
- Match features between image pairs
- Apply RANSAC verification to filter outliers
- Generate visualization outputs in `data/output/`

### Phase 2: Two-View Initialization and Triangulation

Test two-view initialization and 3D point triangulation:

```bash
python test_phase2.py --images Datasets/dinoRing
```

This will:
- Select the best image pair for initialization
- Estimate camera intrinsics
- Recover camera poses from fundamental matrix
- Triangulate initial 3D points
- Validate points using cheirality and reprojection error
- Generate 3D visualization of cameras and points

### Full Pipeline (Coming Soon)

```bash
python run_sfm.py --images <image_dir> --output <output_dir>
```

## Expected Deliverables

1. **Metric**: Reprojection error < 1 pixel across the dataset
2. **Output**: Colored dense point cloud (.ply format)
3. **Validation**: Comparison against Middlebury MVS benchmark
4. **Visualization**: Interactive 3D viewer demonstrating 360° completeness

## Implementation Phases

- [x] **Phase 1**: Foundation (Feature detection, matching, RANSAC)
- [x] **Phase 2**: Two-view initialization and triangulation
- [ ] **Phase 3**: Incremental reconstruction and Bundle Adjustment
- [ ] **Phase 4**: Densification and colorization
- [ ] **Phase 5**: Validation and visualization

## Dependencies

- Python 3.8+
- OpenCV (with SIFT support)
- NumPy
- SciPy
- Open3D
- Matplotlib

See `requirements.txt` for complete list.

