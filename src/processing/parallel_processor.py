"""Parallel processing module for batch audio-to-image conversion."""

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from ..audio.processor import SpatialAudioProcessor
from ..features.spatial_extractor import BinauralFeatureExtractor
from ..images.mosaic_generator import SpatialFeatureMosaicGenerator


@dataclass
class ProcessingConfig:
    """Configuration for parallel audio processing."""

    sample_rate: int = 48000
    n_fft_short: int = 2048
    n_fft_long: int = 4096
    hop_length_ratio: float = 0.5
    n_mels: int = 128
    n_erb: int = 128
    target_image_size: Tuple[int, int] = (512, 512)
    mosaic_layout: Tuple[int, int] = (4, 6)
    bit_depth: int = 8
    max_workers: Optional[int] = 8
    device: str = "cpu"


class AudioToImageConverter:
    """Orchestrates parallel conversion of audio files to spatial feature images.

    Manages the complete pipeline from audio loading through feature extraction
    to mosaic image generation, with support for parallel processing.
    """

    def __init__(self, config: ProcessingConfig):
        """Initialize the converter with given configuration.

        Args:
            config: Processing configuration parameters
        """
        self.config = config
        self.logger = self._setup_logger()

        if config.max_workers is None:
            self.max_workers = max(1, mp.cpu_count() - 1)
        else:
            self.max_workers = config.max_workers

    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the conversion process."""
        logger = logging.getLogger("AudioToImageConverter")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def convert_dataset(
        self, input_dir: Path, output_dir: Path, file_pattern: str = "*.wav"
    ) -> Dict[str, List[Path]]:
        """Convert entire dataset from input directory to output directory.

        Args:
            input_dir: Directory containing audio files organized by class
            output_dir: Directory to save generated images
            file_pattern: Glob pattern for audio files

        Returns:
            Dictionary mapping class names to lists of generated image paths
        """
        audio_files = self._collect_audio_files(input_dir, file_pattern)

        if not audio_files:
            self.logger.warning("No audio files found in input directory")
            return {}

        self.logger.info(
            f"Found {sum(len(files) for files in audio_files.values())} audio files"
        )
        self.logger.info(f"Using {self.max_workers} workers for parallel processing")

        output_paths = {}

        for class_name, files in audio_files.items():
            self.logger.info(f"Processing class '{class_name}': {len(files)} files")

            class_output_dir = output_dir / class_name
            class_output_dir.mkdir(parents=True, exist_ok=True)

            class_results = self._process_class_parallel(files, class_output_dir)
            output_paths[class_name] = class_results

            self.logger.info(
                f"Completed class '{class_name}': {len(class_results)} images generated"
            )

        return output_paths

    def _collect_audio_files(
        self, input_dir: Path, file_pattern: str
    ) -> Dict[str, List[Path]]:
        """Collect audio files organized by class directories."""
        audio_files = {}

        for class_dir in input_dir.iterdir():
            if not class_dir.is_dir():
                continue

            class_files = list(class_dir.glob(file_pattern))
            if class_files:
                audio_files[class_dir.name] = class_files

        return audio_files

    def _process_class_parallel(
        self, audio_files: List[Path], output_dir: Path
    ) -> List[Path]:
        """Process a single class of audio files in parallel."""
        output_paths = []

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_file = {
                executor.submit(
                    _process_single_audio_file, audio_file, output_dir, self.config
                ): audio_file
                for audio_file in audio_files
            }

            with tqdm(
                total=len(audio_files), desc=f"Processing {output_dir.name}"
            ) as pbar:
                for future in as_completed(future_to_file):
                    audio_file = future_to_file[future]

                    try:
                        result_path = future.result()
                        if result_path:
                            output_paths.append(result_path)

                    except Exception as e:
                        self.logger.error(f"Error processing {audio_file}: {str(e)}")

                    pbar.update(1)

        return output_paths

    def generate_class_statistics(
        self, output_paths: Dict[str, List[Path]], stats_output_dir: Path
    ) -> None:
        """Generate statistics and comparison images for each class.

        Args:
            output_paths: Dictionary of class names to image paths
            stats_output_dir: Directory to save statistics
        """
        stats_output_dir.mkdir(parents=True, exist_ok=True)

        mosaic_generator = SpatialFeatureMosaicGenerator(
            target_size=self.config.target_image_size,
            mosaic_layout=self.config.mosaic_layout,
            bit_depth=self.config.bit_depth,
        )

        class_samples = {}

        for class_name, image_paths in output_paths.items():
            if not image_paths:
                continue

            samples = []
            for image_path in image_paths[:10]:
                try:
                    from PIL import Image

                    image = np.array(Image.open(image_path))
                    samples.append(image / 255.0)
                except Exception as e:
                    self.logger.warning(f"Could not load image {image_path}: {e}")

            class_samples[class_name] = samples

        if class_samples:
            mosaic_generator.create_class_comparison(class_samples, stats_output_dir)
            self.logger.info(f"Class comparison images saved to {stats_output_dir}")

    def validate_outputs(self, output_paths: Dict[str, List[Path]]) -> Dict[str, int]:
        """Validate generated outputs and return statistics.

        Args:
            output_paths: Dictionary of class names to image paths

        Returns:
            Dictionary with validation statistics
        """
        stats = {
            "total_files": 0,
            "valid_files": 0,
            "invalid_files": 0,
            "classes_processed": len(output_paths),
        }

        for class_name, image_paths in output_paths.items():
            stats["total_files"] += len(image_paths)

            for image_path in image_paths:
                try:
                    from PIL import Image

                    with Image.open(image_path) as img:
                        if img.size == self.config.target_image_size[::-1]:
                            stats["valid_files"] += 1
                        else:
                            stats["invalid_files"] += 1
                            self.logger.warning(f"Invalid image size: {image_path}")

                except Exception as e:
                    stats["invalid_files"] += 1
                    self.logger.error(f"Could not validate image {image_path}: {e}")

        self.logger.info(
            f"Validation complete: {stats['valid_files']}/{stats['total_files']} valid images"
        )
        return stats


def _process_single_audio_file(
    audio_file: Path, output_dir: Path, config: ProcessingConfig
) -> Optional[Path]:
    """Process a single audio file (used by parallel workers).

    This function runs in a separate process and must be pickleable.

    Args:
        audio_file: Path to the audio file
        output_dir: Directory to save the generated image
        config: Processing configuration

    Returns:
        Path to generated image or None if processing failed
    """
    try:
        audio_processor = SpatialAudioProcessor(
            sample_rate=config.sample_rate,
            n_fft_short=config.n_fft_short,
            n_fft_long=config.n_fft_long,
            hop_length_ratio=config.hop_length_ratio,
            device=config.device,
        )

        feature_extractor = BinauralFeatureExtractor(
            sample_rate=config.sample_rate, n_mels=config.n_mels, n_erb=config.n_erb
        )

        mosaic_generator = SpatialFeatureMosaicGenerator(
            target_size=config.target_image_size,
            mosaic_layout=config.mosaic_layout,
            bit_depth=config.bit_depth,
        )

        left_channel, right_channel = audio_processor.load_audio(audio_file)

        stft_left_short, stft_right_short, stft_left_long, stft_right_long = (
            audio_processor.compute_stft_pair(left_channel, right_channel)
        )

        features = feature_extractor.extract_all_features(
            stft_left_short, stft_right_short, stft_left_long, stft_right_long
        )

        output_path = output_dir / f"{audio_file.stem}.png"
        # Do not save raw .h5 feature files by default to reduce disk usage and I/O.
        mosaic_generator.generate_mosaic(features, output_path, save_raw=False)

        return output_path

    except Exception as e:
        print(f"Error processing {audio_file}: {str(e)}")
        return None
