"""
Calibration File Loader

Loads camera calibration parameters from *_par.txt files (Middlebury MVS format).
Each line contains: imgname.png k11 k12 k13 k21 k22 k23 k31 k32 k33 r11 r12 r13 r21 r22 r23 r31 r32 r33 t1 t2 t3
The projection matrix is P = K * [R | t]
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from .logger import logger


def find_par_file(image_dir: str) -> Optional[str]:
    """
    Auto-detect a *_par.txt calibration file in the given image directory.

    Args:
        image_dir: Path to the image directory

    Returns:
        Path to the par file, or None if not found
    """
    image_path = Path(image_dir)
    par_files = list(image_path.glob("*_par.txt"))
    if par_files:
        logger.info(f"Found calibration file: {par_files[0].name}")
        return str(par_files[0])
    return None


def load_intrinsics_from_par(par_file: str) -> np.ndarray:
    """
    Load camera intrinsic matrix K from a Middlebury-format *_par.txt file.

    All images in these datasets share the same K matrix, so we read from
    the first image entry.

    Args:
        par_file: Path to the *_par.txt calibration file

    Returns:
        3x3 camera intrinsic matrix K
    """
    with open(par_file, 'r') as f:
        lines = f.readlines()

    # First line is the number of images
    # Second line is the first image entry
    parts = lines[1].strip().split()
    # parts[0] = image filename
    # parts[1:10] = K matrix (row-major: k11 k12 k13 k21 k22 k23 k31 k32 k33)
    k_values = [float(x) for x in parts[1:10]]
    K = np.array(k_values).reshape(3, 3)

    logger.info(f"Loaded intrinsics from {Path(par_file).name}: "
               f"fx={K[0,0]:.1f}, fy={K[1,1]:.1f}, "
               f"cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")

    return K


def try_load_intrinsics(image_dir: str) -> Optional[np.ndarray]:
    """
    Try to load ground-truth intrinsics from a calibration file in the dataset.

    This searches for *_par.txt files in the image directory and loads K if found.
    Returns None if no calibration file exists, allowing fallback to heuristic estimation.

    Args:
        image_dir: Path to the image directory

    Returns:
        3x3 intrinsic matrix K, or None if no calibration file found
    """
    par_file = find_par_file(image_dir)
    if par_file is not None:
        return load_intrinsics_from_par(par_file)
    return None
