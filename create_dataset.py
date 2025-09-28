#!/usr/bin/env python3
"""Create complete spatial audio image dataset with normalization analysis.

This script executes the complete pipeline:
1. Analyzes normalization characteristics of the raw audio
2. Converts all audio files to feature images with normalization applied
3. Generates comprehensive reports for academic documentation

The output dataset is ready for machine learning applications.
"""

import subprocess
import sys
from pathlib import Path


def run_command(command, description, working_dir=None):
    """Run a command and handle errors."""
    print(f"\n{'=' * 70}")
    print(f"EXECUTING: {description}")
    print(f"{'=' * 70}")
    print(f"Command: {' '.join(command)}")
    if working_dir:
        print(f"Working directory: {working_dir}")
    print()

    try:
        result = subprocess.run(
            command, check=True, capture_output=False, text=True, cwd=working_dir
        )
        print("\n✅ SUCCESS!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: Command failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        print("\n⏹️  Operation cancelled by user")
        return False


def main():
    """Main execution function."""
    print("🎵 SPATIAL AUDIO DATASET CREATION PIPELINE")
    print("=" * 70)
    print("This pipeline will:")
    print("1. Analyze original audio normalization characteristics")
    print("2. Convert all audio files to feature images (with normalization)")
    print("3. Generate academic reports and visualizations")
    print("4. Create complete ML-ready dataset")

    # Verify prerequisites
    dataset_path = Path("organized_audio")
    if not dataset_path.exists():
        print(f"\n❌ ERROR: Dataset directory not found: {dataset_path}")
        print(
            "Please ensure your audio files are organized in the 'organized_audio' directory"
        )
        return 1

    # Count files
    total_files = sum(
        len(list(class_dir.glob("*.wav")))
        for class_dir in dataset_path.iterdir()
        if class_dir.is_dir()
    )

    class_counts = {}
    for class_dir in dataset_path.iterdir():
        if class_dir.is_dir():
            count = len(list(class_dir.glob("*.wav")))
            class_counts[class_dir.name] = count

    print(f"\n📊 DATASET OVERVIEW:")
    print(f"Total files: {total_files}")
    for class_name, count in class_counts.items():
        print(f"  - {class_name}: {count} files")

    estimated_time = total_files * 0.1  # ~0.1 minutes per file
    print(f"\nEstimated processing time: {estimated_time:.1f} minutes")

    # Get user confirmation
    print(f"\n⚠️  This will process {total_files} audio files and may take a while.")
    response = input("Continue? (y/N): ").strip().lower()
    if response != "y":
        print("Operation cancelled.")
        return 0

    # Step 1: Analyze normalization characteristics
    print(f"\n🔍 STEP 1: Analyzing original audio characteristics...")
    analysis_cmd = [
        "uv",
        "run",
        "python",
        "scripts/generate_normalization_report.py",
        "--dataset-path",
        "organized_audio",
        "--report-name",
        "complete_dataset_analysis",
        "--output-dir",
        "normalization_reports",
    ]

    if not run_command(analysis_cmd, "Analyzing audio normalization characteristics"):
        print("❌ Failed to analyze audio characteristics")
        return 1

    # Step 2: Create visualizations
    print(f"\n📈 STEP 2: Creating normalization analysis visualizations...")
    visualization_cmd = [
        "uv",
        "run",
        "python",
        "scripts/visualize_normalization.py",
        "--reports-dir",
        "normalization_reports",
    ]

    if not run_command(visualization_cmd, "Creating analysis visualizations"):
        print("❌ Failed to create visualizations")
        return 1

    # Step 3: Convert audio to images with normalization
    print(f"\n🖼️  STEP 3: Converting audio files to feature images...")

    # Using RMS normalization (can be changed based on analysis results)
    main_cmd = [
        "uv",
        "run",
        "python",
        "main.py",
        "--input-dir",
        "organized_audio",
        "--output-dir",
        "spatial_images_dataset",
        "--stats-dir",
        "dataset_stats",
        "--sample-rate",
        "48000",
        "--n-fft-short",
        "2048",
        "--n-fft-long",
        "4096",
        "--hop-ratio",
        "0.5",
        "--image-size",
        "384",
        "384",
        "--mosaic-layout",
        "4",
        "6",
        "--bit-depth",
        "8",
        "--device",
        "cuda" if __import__("torch").cuda.is_available() else "cpu",
    ]

    if not run_command(main_cmd, "Converting audio to feature images"):
        print("❌ Failed to convert audio to images")
        return 1

    # Success summary
    print("\n" + "=" * 70)
    print("🎉 DATASET CREATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)

    # Check outputs
    output_dir = Path("spatial_images_dataset")
    reports_dir = Path("normalization_reports")
    stats_dir = Path("dataset_stats")

    if output_dir.exists():
        image_count = sum(
            len(list(class_dir.glob("*.png")))
            for class_dir in output_dir.iterdir()
            if class_dir.is_dir()
        )
        print(f"✅ Generated {image_count} feature images in: {output_dir}/")

        for class_dir in output_dir.iterdir():
            if class_dir.is_dir():
                count = len(list(class_dir.glob("*.png")))
                print(f"   - {class_dir.name}: {count} images")

    if reports_dir.exists():
        print(f"\n📊 Normalization analysis reports: {reports_dir}/")
        for file in reports_dir.glob("complete_dataset_analysis*"):
            print(f"   - {file.name}")

        figures_dir = reports_dir / "figures"
        if figures_dir.exists():
            print(f"   - figures/ ({len(list(figures_dir.glob('*.png')))} plots)")

    if stats_dir.exists():
        print(f"\n📈 Dataset statistics: {stats_dir}/")
        for file in stats_dir.glob("*.png"):
            print(f"   - {file.name}")

    print(f"\n🎓 FOR YOUR THESIS:")
    print("   • Use normalization_reports/ for justifying preprocessing choices")
    print("   • Use spatial_images_dataset/ as your ML dataset")
    print("   • Use figures/ for thesis visualizations")
    print("   • All data is quantitatively documented for academic rigor")

    print(f"\n🚀 Your spatial audio image dataset is ready for machine learning!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
