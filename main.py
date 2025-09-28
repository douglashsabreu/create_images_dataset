#!/usr/bin/env python3
"""Main script for converting spatial audio dataset to feature images.

This script orchestrates the complete pipeline from stereo audio files
to spatial feature mosaic images, implementing multi-resolution STFT,
binaural feature extraction, and mosaic generation for audio spatial
classification tasks.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

from src.processing.parallel_processor import AudioToImageConverter, ProcessingConfig


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the command line argument parser.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Convert spatial audio dataset to feature images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("organized_audio"),
        help="Input directory containing audio files organized by class",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("spatial_images"),
        help="Output directory for generated feature images",
    )

    parser.add_argument(
        "--stats-dir",
        type=Path,
        default=Path("dataset_stats"),
        help="Output directory for dataset statistics and comparisons",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48000,
        help="Target sample rate for audio processing",
    )

    parser.add_argument(
        "--n-fft-short",
        type=int,
        default=2048,
        help="FFT size for short-time analysis (transients)",
    )

    parser.add_argument(
        "--n-fft-long",
        type=int,
        default=4096,
        help="FFT size for long-time analysis (ambience)",
    )

    parser.add_argument(
        "--hop-ratio",
        type=float,
        default=0.5,
        help="Hop length as ratio of n_fft (0.5 = 50%% overlap)",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[512, 512],
        metavar=("HEIGHT", "WIDTH"),
        help="Target image dimensions",
    )

    parser.add_argument(
        "--mosaic-layout",
        type=int,
        nargs=2,
        default=[4, 6],
        metavar=("ROWS", "COLS"),
        help="Mosaic grid layout for features",
    )

    parser.add_argument(
        "--bit-depth",
        type=int,
        choices=[8, 16],
        default=8,
        help="Bit depth for output images",
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers (default: CPU count - 1)",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda" if __import__("torch").cuda.is_available() else "cpu",
        help="Device for PyTorch computations",
    )

    parser.add_argument(
        "--file-pattern", type=str, default="*.wav", help="Glob pattern for audio files"
    )

    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Skip generation of dataset statistics",
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate existing outputs without processing",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    return parser


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration.

    Args:
        verbose: Enable verbose logging
    """
    import logging

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def validate_input_directory(input_dir: Path) -> bool:
    """Validate input directory structure.

    Args:
        input_dir: Input directory path

    Returns:
        True if directory structure is valid
    """
    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist")
        return False

    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory")
        return False

    class_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    if not class_dirs:
        print(f"Error: No class directories found in {input_dir}")
        return False

    print(f"Found {len(class_dirs)} class directories: {[d.name for d in class_dirs]}")
    return True


def create_processing_config(args: argparse.Namespace) -> ProcessingConfig:
    """Create processing configuration from command line arguments.

    Args:
        args: Parsed command line arguments

    Returns:
        Processing configuration object
    """
    return ProcessingConfig(
        sample_rate=args.sample_rate,
        n_fft_short=args.n_fft_short,
        n_fft_long=args.n_fft_long,
        hop_length_ratio=args.hop_ratio,
        target_image_size=tuple(args.image_size),
        mosaic_layout=tuple(args.mosaic_layout),
        bit_depth=args.bit_depth,
        max_workers=args.max_workers,
        device=args.device,
    )


def print_processing_summary(
    config: ProcessingConfig, args: argparse.Namespace
) -> None:
    """Print processing configuration summary.

    Args:
        config: Processing configuration
        args: Command line arguments
    """
    print("=" * 60)
    print("SPATIAL AUDIO TO IMAGE CONVERSION")
    print("=" * 60)
    print(f"Input directory:     {args.input_dir}")
    print(f"Output directory:    {args.output_dir}")
    print(f"Statistics directory: {args.stats_dir}")
    print()
    print("Audio Processing:")
    print(f"  Sample rate:       {config.sample_rate} Hz")
    print(f"  Duration:          Full audio length")
    print(f"  Short STFT size:   {config.n_fft_short} samples")
    print(f"  Long STFT size:    {config.n_fft_long} samples")
    print(f"  Hop length ratio:  {config.hop_length_ratio}")
    print(f"  Device:            {config.device}")
    print()
    print("Image Generation:")
    print(f"  Image size:        {config.target_image_size}")
    print(f"  Mosaic layout:     {config.mosaic_layout[0]}x{config.mosaic_layout[1]}")
    print(f"  Bit depth:         {config.bit_depth} bits")
    print()
    print("Parallel Processing:")
    print(f"  Max workers:       {config.max_workers or 'auto'}")
    print("=" * 60)


def main() -> int:
    """Main entry point for the spatial audio conversion pipeline.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = create_argument_parser()
    args = parser.parse_args()

    setup_logging(args.verbose)

    if not validate_input_directory(args.input_dir):
        return 1

    config = create_processing_config(args)

    print_processing_summary(config, args)

    try:
        converter = AudioToImageConverter(config)

        if args.validate_only:
            print("Validation mode - checking existing outputs...")
            if args.output_dir.exists():
                from collections import defaultdict

                output_paths = defaultdict(list)

                for class_dir in args.output_dir.iterdir():
                    if class_dir.is_dir():
                        images = list(class_dir.glob("*.png"))
                        output_paths[class_dir.name] = images

                stats = converter.validate_outputs(dict(output_paths))
                print(f"Validation complete: {stats}")
                return 0
            else:
                print(f"Output directory {args.output_dir} does not exist")
                return 1

        print("\\nStarting audio-to-image conversion...")
        output_paths = converter.convert_dataset(
            args.input_dir, args.output_dir, args.file_pattern
        )

        if not output_paths:
            print("No images were generated")
            return 1

        total_images = sum(len(paths) for paths in output_paths.values())
        print(f"\\nConversion complete! Generated {total_images} images")

        for class_name, paths in output_paths.items():
            print(f"  {class_name}: {len(paths)} images")

        if not args.skip_stats:
            print("\\nGenerating dataset statistics...")
            converter.generate_class_statistics(output_paths, args.stats_dir)

        print("\\nValidating generated images...")
        validation_stats = converter.validate_outputs(output_paths)

        success_rate = validation_stats["valid_files"] / max(
            validation_stats["total_files"], 1
        )
        print(f"Success rate: {success_rate:.2%}")

        if success_rate < 0.95:
            print("Warning: Success rate below 95%, check logs for errors")
            return 1

        print("\\nPipeline completed successfully!")
        return 0

    except KeyboardInterrupt:
        print("\\nOperation cancelled by user")
        return 1

    except Exception as e:
        print(f"\\nError: {str(e)}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
