#!/usr/bin/env python3
"""Generate comprehensive normalization analysis report for the spatial audio dataset.

This script analyzes the entire dataset and generates detailed reports
comparing different normalization methods. The reports are suitable for
academic documentation and thesis writing.
"""

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.reports import NormalizationAnalyzer


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the command line argument parser.

    Returns:
        Configured argument parser
    """
    parser = argparse.ArgumentParser(
        description="Generate normalization analysis report for spatial audio dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("organized_audio"),
        help="Path to the organized audio dataset directory",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("normalization_reports"),
        help="Directory to save the generated reports",
    )

    parser.add_argument(
        "--target-level",
        type=float,
        default=-23.0,
        help="Target normalization level in dB",
    )

    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48000,
        help="Target sample rate for analysis",
    )

    parser.add_argument(
        "--report-name",
        type=str,
        default="spatial_audio_normalization",
        help="Base name for generated report files",
    )

    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with limited files (for testing purposes)",
    )

    return parser


def main() -> int:
    """Main entry point for the normalization analysis report generator.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = create_argument_parser()
    args = parser.parse_args()

    if not args.dataset_path.exists():
        print(f"Error: Dataset path does not exist: {args.dataset_path}")
        return 1

    if not args.dataset_path.is_dir():
        print(f"Error: Dataset path is not a directory: {args.dataset_path}")
        return 1

    print("SPATIAL AUDIO NORMALIZATION ANALYSIS")
    print("=" * 50)
    print(f"Dataset path: {args.dataset_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Target level: {args.target_level} dB")
    print(f"Sample rate: {args.sample_rate} Hz")
    print(f"Report name: {args.report_name}")
    print("=" * 50)

    try:
        # Initialize analyzer
        analyzer = NormalizationAnalyzer(
            target_level_db=args.target_level,
            sample_rate=args.sample_rate,
            output_dir=args.output_dir,
        )

        if args.quick_test:
            print("\nRunning quick test mode (limited files)...")
            # Create a temporary dataset path with limited files for testing
            test_files = []
            for class_dir in args.dataset_path.iterdir():
                if class_dir.is_dir():
                    audio_files = list(class_dir.glob("*.wav"))[
                        :2
                    ]  # Limit to 2 files per class
                    test_files.extend(audio_files)

            print(f"Testing with {len(test_files)} files")

        # Analyze the dataset
        print("\nStarting dataset analysis...")
        report = analyzer.analyze_dataset(args.dataset_path)

        # Save the reports
        print("\nSaving reports...")
        output_files = analyzer.save_report(report, args.report_name)

        # Print summary
        print("\n" + "=" * 50)
        print("ANALYSIS SUMMARY")
        print("=" * 50)
        print(f"Total files analyzed: {report.total_files}")
        print(f"Total duration: {report.total_duration_hours:.2f} hours")
        print(f"Audio classes: {len(report.class_statistics)}")

        for class_stat in report.class_statistics:
            print(
                f"  - {class_stat.class_name}: {class_stat.num_files} files, "
                f"{class_stat.total_duration_seconds / 60:.1f} minutes"
            )

        print(f"\nOriginal audio levels:")
        print(
            f"  RMS: {report.overall_original_rms_mean:.1f} ± {report.overall_original_rms_std:.1f} dB"
        )
        print(
            f"  Peak: {report.overall_original_peak_mean:.1f} ± {report.overall_original_peak_std:.1f} dB"
        )
        print(
            f"  LUFS: {report.overall_original_lufs_mean:.1f} ± {report.overall_original_lufs_std:.1f}"
        )

        print(f"\nNormalization effectiveness scores:")
        for method, score in report.normalization_effectiveness_scores.items():
            indicator = " ★" if method == report.best_normalization_method else ""
            print(f"  {method}: {score:.2f}{indicator}")

        print(f"\nBest normalization method: {report.best_normalization_method}")

        print("\nFiles generated:")
        for format_name, path in output_files.items():
            print(f"  - {format_name}: {path.name}")

        print("\nThese files can be used for:")
        print("  - Statistical analysis and plotting with Python/R")
        print("  - Creating figures for thesis documentation")
        print("  - Comparing normalization method effectiveness")
        print("  - Justifying normalization choices in academic papers")

        print(f"\nAnalysis completed successfully!")
        return 0

    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1

    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
