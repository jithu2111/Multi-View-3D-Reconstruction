"""
Image loading and preprocessing utilities
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from .logger import logger


class ImageLoader:
    """Handle loading and preprocessing of images for SfM pipeline"""

    def __init__(self, max_dimension: Optional[int] = 1600):
        """
        Initialize image loader

        Args:
            max_dimension: Maximum dimension (width or height) for loaded images.
                          Images larger than this will be downscaled while maintaining
                          aspect ratio. Set to None to disable resizing.
        """
        self.max_dimension = max_dimension
        self.images = []
        self.image_paths = []
        self.original_sizes = []
        self.scale_factors = []

    def load_images(self, image_dir: str, extensions: List[str] = None) -> List[np.ndarray]:
        """
        Load all images from a directory

        Args:
            image_dir: Path to directory containing images
            extensions: List of valid image extensions (default: common formats)

        Returns:
            List of loaded images as numpy arrays
        """
        if extensions is None:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']

        image_path = Path(image_dir)
        if not image_path.exists():
            raise FileNotFoundError(f"Image directory not found: {image_dir}")

        # Find all image files
        image_files = []
        for ext in extensions:
            image_files.extend(image_path.glob(f"*{ext}"))
            image_files.extend(image_path.glob(f"*{ext.upper()}"))

        image_files = sorted(image_files)

        if not image_files:
            raise ValueError(f"No images found in {image_dir}")

        logger.info(f"Found {len(image_files)} images in {image_dir}")

        # Load images
        self.images = []
        self.image_paths = []
        self.original_sizes = []
        self.scale_factors = []

        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                logger.warning(f"Failed to load image: {img_path}")
                continue

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Store original size
            original_h, original_w = img.shape[:2]
            self.original_sizes.append((original_h, original_w))

            # Resize if necessary
            if self.max_dimension is not None:
                img, scale = self._resize_image(img, self.max_dimension)
                self.scale_factors.append(scale)
            else:
                self.scale_factors.append(1.0)

            self.images.append(img)
            self.image_paths.append(str(img_path))

            logger.debug(f"Loaded: {img_path.name} - Shape: {img.shape}")

        logger.info(f"Successfully loaded {len(self.images)} images")
        return self.images

    def _resize_image(self, image: np.ndarray, max_dim: int) -> Tuple[np.ndarray, float]:
        """
        Resize image to have maximum dimension max_dim while maintaining aspect ratio

        Args:
            image: Input image
            max_dim: Maximum allowed dimension

        Returns:
            Resized image and scale factor
        """
        h, w = image.shape[:2]

        if max(h, w) <= max_dim:
            return image, 1.0

        if h > w:
            new_h = max_dim
            new_w = int(w * (max_dim / h))
        else:
            new_w = max_dim
            new_h = int(h * (max_dim / w))

        scale = new_h / h  # or new_w / w, should be the same
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return resized, scale

    def get_image(self, idx: int) -> np.ndarray:
        """Get image by index"""
        if idx < 0 or idx >= len(self.images):
            raise IndexError(f"Image index {idx} out of range")
        return self.images[idx]

    def get_image_path(self, idx: int) -> str:
        """Get image path by index"""
        if idx < 0 or idx >= len(self.image_paths):
            raise IndexError(f"Image index {idx} out of range")
        return self.image_paths[idx]

    def get_scale_factor(self, idx: int) -> float:
        """Get scale factor for image at index"""
        if idx < 0 or idx >= len(self.scale_factors):
            raise IndexError(f"Image index {idx} out of range")
        return self.scale_factors[idx]

    def __len__(self) -> int:
        """Return number of loaded images"""
        return len(self.images)

    def __getitem__(self, idx: int) -> np.ndarray:
        """Allow indexing to get images"""
        return self.get_image(idx)