"""Mosaic image generation module for spatial audio features."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import matplotlib.cm as cm
import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont


class SpatialFeatureMosaicGenerator:
    """Generates mosaic images from spatial audio features.

    Creates comprehensive visual representations by arranging multiple
    feature maps in a mosaic layout, with proper normalization and
    consistent sizing for CNN/ViT compatibility.
    """

    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        mosaic_layout: Tuple[int, int] = (4, 6),
        feature_size: Tuple[int, int] = (128, 200),
        bit_depth: int = 8,
        colormap: str = "viridis",
        add_labels: bool = True,
        image_format: str = "png",
    ):
        """Initialize the mosaic generator.

        Args:
            target_size: Final image dimensions (height, width)
            mosaic_layout: Grid layout (rows, cols) for feature arrangement
            feature_size: Size for individual feature maps (height, width)
            bit_depth: Bit depth for output images (8 or 16)
            colormap: Matplotlib colormap for feature visualization
            add_labels: Whether to add text labels to feature blocks
            image_format: Output format ('tiff', 'png') - tiff recommended for max quality
        """
        self.target_size = target_size
        self.mosaic_layout = mosaic_layout
        self.feature_size = feature_size
        self.bit_depth = bit_depth
        self.colormap = colormap
        self.add_labels = add_labels
        self.image_format = image_format.lower()

        self.total_blocks = mosaic_layout[0] * mosaic_layout[1]
        self._setup_feature_layout()

    def _setup_feature_layout(self) -> None:
        """Define the layout and order of features in the mosaic."""
        self.feature_order = [
            "magnitude_L_short",
            "magnitude_R_short",
            "ild_short",
            "ipd_sin_short",
            "ipd_cos_short",
            "iacc_short",
            "magnitude_L_long",
            "magnitude_R_long",
            "ild_long",
            "ipd_sin_long",
            "ipd_cos_long",
            "iacc_long",
            "mid_short",
            "side_short",
            "laterality_short",
            "mid_long",
            "side_long",
            "laterality_long",
            "msc_short",
            "msc_long",
            "gcc_phat",
            "spectral_centroid_L_short",
            "spectral_slope_L_short",
            "late_energy_short",
        ]

        self.feature_labels = {
            "magnitude_L_short": "Mag L (S)",
            "magnitude_R_short": "Mag R (S)",
            "magnitude_L_long": "Mag L (L)",
            "magnitude_R_long": "Mag R (L)",
            "ild_short": "ILD (S)",
            "ild_long": "ILD (L)",
            "ipd_sin_short": "IPD Sin (S)",
            "ipd_cos_short": "IPD Cos (S)",
            "ipd_sin_long": "IPD Sin (L)",
            "ipd_cos_long": "IPD Cos (L)",
            "iacc_short": "IACC (S)",
            "iacc_long": "IACC (L)",
            "mid_short": "Mid (S)",
            "side_short": "Side (S)",
            "mid_long": "Mid (L)",
            "side_long": "Side (L)",
            "laterality_short": "Lat (S)",
            "laterality_long": "Lat (L)",
            "msc_short": "MSC (S)",
            "msc_long": "MSC (L)",
            "gcc_phat": "GCC-PHAT",
            "itd_estimate": "ITD Est",
            "spectral_centroid_L_short": "SC L (S)",
            "spectral_slope_L_short": "SS L (S)",
            "late_energy_short": "Late E (S)",
            "late_energy_long": "Late E (L)",
        }

    def generate_mosaic(
        self,
        features: Dict[str, np.ndarray],
        output_path: Optional[Path] = None,
        save_raw: bool = False,
    ) -> np.ndarray:
        """Generate mosaic image from spatial audio features.

        Args:
            features: Dictionary of extracted spatial features
            output_path: Path to save the mosaic image (optional)
            save_raw: Whether to save raw feature data as HDF5

        Returns:
            Mosaic image as numpy array
        """
        normalized_features = self._normalize_features(features)
        resized_features = self._resize_features(normalized_features)

        mosaic_image = self._create_mosaic_layout(resized_features)

        if output_path:
            self._save_image(mosaic_image, output_path)

            if save_raw:
                # Save raw features only when explicitly requested (disabled by default)
                self._save_raw_features(features, output_path.with_suffix(".h5"))

        return mosaic_image

    def _normalize_features(
        self, features: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Normalize features for consistent visualization."""
        normalized = {}

        for key, feature in features.items():
            if feature.size == 0:
                continue

            feature = self._ensure_2d(feature)

            if "ipd" in key or "iacc" in key:
                normalized[key] = np.clip((feature + 1) / 2, 0, 1)
            elif "ild" in key:
                feature_clipped = np.clip(feature, -20, 20)
                normalized[key] = (feature_clipped + 20) / 40
            elif "gcc_phat" in key:
                normalized[key] = (feature - feature.min()) / (
                    feature.max() - feature.min() + 1e-8
                )
            else:
                feature_norm = feature - np.percentile(feature, 1)
                feature_norm = feature_norm / (np.percentile(feature_norm, 99) + 1e-8)
                normalized[key] = np.clip(feature_norm, 0, 1)

        return normalized

    def _ensure_2d(self, feature: np.ndarray) -> np.ndarray:
        """Ensure feature is 2D for visualization."""
        if feature.ndim == 1:
            return feature[np.newaxis, :]
        elif feature.ndim == 3:
            return feature[0, :, :]
        return feature

    def _resize_features(
        self, features: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        """Resize all features to consistent dimensions."""
        resized = {}

        for key, feature in features.items():
            if feature.size == 0:
                resized[key] = np.zeros(self.feature_size)
                continue

            from skimage.transform import resize

            if feature.shape != self.feature_size:
                resized_feature = resize(
                    feature,
                    self.feature_size,
                    mode="reflect",
                    anti_aliasing=True,
                    preserve_range=True,
                )
                resized[key] = resized_feature
            else:
                resized[key] = feature

        return resized

    def _create_mosaic_layout(self, features: Dict[str, np.ndarray]) -> np.ndarray:
        """Create the final mosaic layout from resized features."""
        rows, cols = self.mosaic_layout
        block_height, block_width = self.feature_size

        mosaic_height = rows * block_height
        mosaic_width = cols * block_width

        mosaic = np.zeros((mosaic_height, mosaic_width, 3), dtype=np.float32)

        cmap = cm.get_cmap(self.colormap)

        for idx, feature_name in enumerate(self.feature_order[: self.total_blocks]):
            row = idx // cols
            col = idx % cols

            if feature_name not in features:
                continue

            feature = features[feature_name]

            if feature.size == 0:
                continue

            colored_feature = cmap(feature)[:, :, :3]

            start_row = row * block_height
            end_row = start_row + block_height
            start_col = col * block_width
            end_col = start_col + block_width

            mosaic[start_row:end_row, start_col:end_col] = colored_feature

            if self.add_labels:
                self._add_text_label(
                    mosaic,
                    self.feature_labels.get(feature_name, feature_name),
                    (start_row + 5, start_col + 5),
                )

        if mosaic.shape[:2] != self.target_size:
            from skimage.transform import resize

            mosaic = resize(mosaic, self.target_size, preserve_range=True)

        return mosaic

    def _add_text_label(
        self, image: np.ndarray, text: str, position: Tuple[int, int]
    ) -> None:
        """Add text label to a specific position in the image."""
        try:
            # Ensure we have proper dimensions for PIL text rendering
            if image.ndim == 3:
                pil_image = Image.fromarray((image * 255).astype(np.uint8))
            else:
                # For 2D arrays, expand to RGB for text rendering
                rgb_image = np.stack([image, image, image], axis=-1)
                pil_image = Image.fromarray((rgb_image * 255).astype(np.uint8))

            draw = ImageDraw.Draw(pil_image)

            try:
                font = ImageFont.truetype("arial.ttf", 12)
            except:
                font = ImageFont.load_default()

            draw.text(position, text, fill=(255, 255, 255), font=font)

            # Update original image
            if image.ndim == 3:
                image[:] = np.array(pil_image) / 255.0
            else:
                # Convert back to grayscale if original was 2D
                gray_back = np.array(pil_image.convert("L")) / 255.0
                image[:] = gray_back

        except Exception:
            pass

    def _save_image(self, image: np.ndarray, output_path: Path) -> None:
        """Save the mosaic image to disk with high quality support."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure image is in [0, 1] range
        image = np.clip(image, 0, 1)

        # Set output format and extension
        if self.image_format == "tiff":
            output_path = output_path.with_suffix(".tiff")
            self._save_as_tiff(image, output_path)
        else:  # png
            output_path = output_path.with_suffix(".png")
            self._save_as_png(image, output_path)

    def _save_as_tiff(self, image: np.ndarray, output_path: Path) -> None:
        """Save image as TIFF format (supports 16-bit RGB natively)."""
        try:
            if self.bit_depth == 16:
                image_16bit = (image * 65535).astype(np.uint16)
                photometric = "rgb" if image.ndim == 3 else "minisblack"
                tifffile.imwrite(str(output_path), image_16bit, photometric=photometric)
            else:
                image_8bit = (image * 255).astype(np.uint8)
                photometric = "rgb" if image.ndim == 3 else "minisblack"
                tifffile.imwrite(str(output_path), image_8bit, photometric=photometric)
        except Exception as e:
            print(f"Warning: TIFF save failed: {e}")
            self._save_with_fallback(image, output_path.with_suffix(".png"))

    def _save_as_png(self, image: np.ndarray, output_path: Path) -> None:
        """Save image as PNG format using OpenCV for 16-bit support."""
        try:
            if self.bit_depth == 16:
                image_16bit = (image * 65535).astype(np.uint16)
                if image.ndim == 3:
                    # OpenCV expects BGR format
                    bgr_image = cv2.cvtColor(image_16bit, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(output_path), bgr_image)
                else:
                    cv2.imwrite(str(output_path), image_16bit)
            else:
                image_8bit = (image * 255).astype(np.uint8)
                if image.ndim == 3:
                    bgr_image = cv2.cvtColor(image_8bit, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(str(output_path), bgr_image)
                else:
                    cv2.imwrite(str(output_path), image_8bit)
        except Exception as e:
            print(f"Warning: OpenCV PNG save failed: {e}")
            self._save_with_fallback(image, output_path)

    def _save_with_fallback(self, image: np.ndarray, output_path: Path) -> None:
        """Fallback method using PIL for compatibility."""
        try:
            image_8bit = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image_8bit)
            pil_image.save(str(output_path))
        except Exception as e:
            print(f"Error: Could not save image {output_path}: {e}")

    def _save_raw_features(
        self, features: Dict[str, np.ndarray], output_path: Path
    ) -> None:
        """Save raw features to HDF5 format for reproducibility."""
        try:
            with h5py.File(str(output_path), "w") as f:
                for key, feature in features.items():
                    if feature.size > 0:
                        f.create_dataset(
                            key, data=feature, compression="gzip", compression_opts=9
                        )
        except Exception as e:
            print(f"Warning: Could not save raw features: {e}")

    def create_class_comparison(
        self, class_samples: Dict[str, List[np.ndarray]], output_dir: Path
    ) -> None:
        """Create comparison images showing differences between classes.

        Args:
            class_samples: Dictionary with class names as keys and lists of mosaic images
            output_dir: Directory to save comparison images
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for class_name, samples in class_samples.items():
            if not samples:
                continue

            mean_image = np.mean(samples, axis=0)
            std_image = np.std(samples, axis=0)

            comparison = np.concatenate([mean_image, std_image], axis=1)

            comparison_path = output_dir / f"{class_name}_comparison.png"
            self._save_image(comparison, comparison_path)

    def validate_features(self, features: Dict[str, np.ndarray]) -> Dict[str, bool]:
        """Validate feature maps for expected characteristics.

        Args:
            features: Dictionary of spatial features

        Returns:
            Dictionary of validation results per feature
        """
        validation_results = {}

        for key, feature in features.items():
            if feature.size == 0:
                validation_results[key] = False
                continue

            has_variation = np.std(feature) > 1e-6
            has_finite_values = np.isfinite(feature).all()

            if "iacc" in key:
                in_range = np.all((feature >= -1) & (feature <= 1))
            elif "ild" in key:
                in_range = np.all((feature >= -40) & (feature <= 40))
            else:
                in_range = True

            validation_results[key] = has_variation and has_finite_values and in_range

        return validation_results
