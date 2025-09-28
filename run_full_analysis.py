#!/usr/bin/env python3
"""Run complete normalization analysis on the full spatial audio dataset.

This script executes the complete analysis pipeline:
1. Analyzes all audio files in the dataset
2. Generates comprehensive reports
3. Creates publication-ready visualizations

Results are suitable for academic documentation and thesis writing.
"""

import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{'=' * 60}")
    print(f"EXECUTING: {description}")
    print(f"{'=' * 60}")
    print(f"Command: {' '.join(command)}")

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("SUCCESS!")
        if result.stdout:
            print("Output:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with exit code {e.returncode}")
        if e.stderr:
            print("Error output:")
            print(e.stderr)
        return False


def main():
    """Main execution function."""
    print("SPATIAL AUDIO DATASET: COMPLETE NORMALIZATION ANALYSIS")
    print("=" * 60)
    print("This script will:")
    print("1. Analyze all audio files in 'organized_audio' directory")
    print("2. Generate comprehensive normalization reports")
    print("3. Create publication-ready visualizations")
    print("4. Save all results to 'normalization_reports' directory")

    dataset_path = Path("organized_audio")
    if not dataset_path.exists():
        print(f"\nERROR: Dataset directory not found: {dataset_path}")
        print(
            "Please ensure your audio files are organized in the 'organized_audio' directory"
        )
        return 1

    # Count total files
    total_files = sum(
        len(list(class_dir.glob("*.wav")))
        for class_dir in dataset_path.iterdir()
        if class_dir.is_dir()
    )

    print(f"\nFound {total_files} audio files to analyze")
    print("Estimated processing time: ~5-10 minutes")

    input("\nPress Enter to continue or Ctrl+C to cancel...")

    # Step 1: Generate analysis report
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

    if not run_command(analysis_cmd, "Generating comprehensive analysis report"):
        return 1

    # Step 2: Create visualizations
    visualization_cmd = [
        "uv",
        "run",
        "python",
        "scripts/visualize_normalization.py",
        "--reports-dir",
        "normalization_reports",
    ]

    if not run_command(visualization_cmd, "Creating publication-ready visualizations"):
        return 1

    # Success summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Results saved to: normalization_reports/")
    print("\nGenerated files:")

    reports_dir = Path("normalization_reports")
    if reports_dir.exists():
        print("\nReports:")
        for file in reports_dir.glob("complete_dataset_analysis*"):
            print(f"  - {file.name}")

        figures_dir = reports_dir / "figures"
        if figures_dir.exists():
            print("\nFigures:")
            for file in figures_dir.glob("*.png"):
                print(f"  - figures/{file.name}")

    print("\nThese files can be used for:")
    print("  • Statistical analysis with Python/R/MATLAB")
    print("  • Thesis figures and documentation")
    print("  • Academic paper illustrations")
    print("  • Justifying normalization method choices")
    print("  • Dataset characterization and analysis")

    print(f"\nFor detailed usage instructions, see: README_NORMALIZATION.md")

    return 0


if __name__ == "__main__":
    sys.exit(main())



