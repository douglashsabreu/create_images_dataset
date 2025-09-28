#!/usr/bin/env python3
"""Visualize normalization analysis results for academic documentation.

This script creates publication-ready plots and figures from the normalization
analysis reports, suitable for thesis documentation and academic papers.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import numpy as np
import pandas as pd
import seaborn as sns

# Set publication-ready style
mplstyle.use("default")
plt.rcParams.update(
    {
        "font.size": 12,
        "font.family": "serif",
        "figure.figsize": (10, 6),
        "figure.dpi": 300,
        "axes.linewidth": 1.2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.3,
    }
)


class NormalizationVisualizer:
    """Creates publication-ready visualizations of normalization analysis results."""

    def __init__(self, reports_dir: Path, output_dir: Path = None):
        """Initialize the visualizer.

        Args:
            reports_dir: Directory containing normalization analysis reports
            output_dir: Directory to save visualization outputs
        """
        self.reports_dir = Path(reports_dir)
        self.output_dir = (
            Path(output_dir) if output_dir else self.reports_dir / "figures"
        )
        self.output_dir.mkdir(exist_ok=True)

        # Load data
        self.comparison_data = None
        self.class_summary_data = None
        self.detailed_data = None

        self._load_data()

    def _load_data(self) -> None:
        """Load analysis data from CSV files."""
        try:
            comparison_files = list(self.reports_dir.glob("*_comparison.csv"))
            class_files = list(self.reports_dir.glob("*_class_summary.csv"))
            detailed_files = list(self.reports_dir.glob("*_detailed.csv"))

            if comparison_files:
                self.comparison_data = pd.read_csv(comparison_files[0])
                print(f"Loaded comparison data: {len(self.comparison_data)} rows")

            if class_files:
                self.class_summary_data = pd.read_csv(class_files[0])
                print(
                    f"Loaded class summary data: {len(self.class_summary_data)} classes"
                )

            if detailed_files:
                self.detailed_data = pd.read_csv(detailed_files[0])
                print(f"Loaded detailed data: {len(self.detailed_data)} files")

        except Exception as e:
            print(f"Warning: Could not load some data files: {e}")

    def create_all_visualizations(self) -> List[Path]:
        """Create all standard visualizations.

        Returns:
            List of paths to created figures
        """
        figure_paths = []

        if self.comparison_data is not None:
            # Level distribution comparisons
            fig_path = self.plot_level_distributions()
            if fig_path:
                figure_paths.append(fig_path)

            # Normalization effectiveness
            fig_path = self.plot_normalization_effectiveness()
            if fig_path:
                figure_paths.append(fig_path)

            # Class-wise comparison
            fig_path = self.plot_class_comparison()
            if fig_path:
                figure_paths.append(fig_path)

            # Gain distribution
            fig_path = self.plot_gain_distributions()
            if fig_path:
                figure_paths.append(fig_path)

        if self.class_summary_data is not None:
            # Class statistics overview
            fig_path = self.plot_class_statistics()
            if fig_path:
                figure_paths.append(fig_path)

        return figure_paths

    def plot_level_distributions(self) -> Path:
        """Plot audio level distributions before and after normalization."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            "Audio Level Distributions: Original vs Normalized",
            fontsize=16,
            fontweight="bold",
        )

        methods = ["Original", "RMS", "Peak", "LUFS"]
        metrics = [("rms_db", "RMS Level (dB)"), ("peak_db", "Peak Level (dB)")]

        for i, (metric, ylabel) in enumerate(metrics):
            ax1, ax2 = axes[i]

            # Original vs RMS/Peak
            for method in methods[:3] if i == 0 else ["Original", "Peak"]:
                data = self.comparison_data[
                    self.comparison_data["normalization_method"] == method
                ][metric].dropna()

                ax1.hist(data, bins=30, alpha=0.7, label=method, density=True)

            ax1.set_xlabel(ylabel)
            ax1.set_ylabel("Density")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_title(f"{ylabel} Distribution")

            # LUFS comparison
            if i == 0:  # Only for RMS plot
                for method in ["Original", "LUFS"]:
                    data = self.comparison_data[
                        self.comparison_data["normalization_method"] == method
                    ]["lufs"].dropna()

                    ax2.hist(data, bins=30, alpha=0.7, label=method, density=True)

                ax2.set_xlabel("LUFS")
                ax2.set_ylabel("Density")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.set_title("LUFS Distribution")
            else:
                # Dynamic range comparison
                for method in ["Original", "RMS", "Dynamic"]:
                    data = self.comparison_data[
                        self.comparison_data["normalization_method"] == method
                    ]["dynamic_range_db"].dropna()

                    ax2.hist(data, bins=30, alpha=0.7, label=method, density=True)

                ax2.set_xlabel("Dynamic Range (dB)")
                ax2.set_ylabel("Density")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.set_title("Dynamic Range Distribution")

        plt.tight_layout()

        output_path = self.output_dir / "level_distributions.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved level distributions plot: {output_path}")
        return output_path

    def plot_normalization_effectiveness(self) -> Path:
        """Plot normalization effectiveness by measuring consistency."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            "Normalization Effectiveness: Level Consistency Analysis",
            fontsize=16,
            fontweight="bold",
        )

        metrics = ["rms_db", "peak_db", "lufs", "dynamic_range_db"]
        titles = [
            "RMS Level Consistency",
            "Peak Level Consistency",
            "LUFS Consistency",
            "Dynamic Range Consistency",
        ]

        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i // 2, i % 2]

            # Calculate standard deviation for each method and class
            consistency_data = []
            methods = ["RMS", "Peak", "LUFS", "Dynamic"]

            if metric == "peak_db":
                methods = ["Peak"]  # Peak normalization mainly affects peak levels
            elif metric == "lufs":
                methods = ["LUFS", "RMS"]  # These methods affect LUFS

            for audio_class in self.comparison_data["audio_class"].unique():
                for method in methods:
                    class_method_data = self.comparison_data[
                        (self.comparison_data["audio_class"] == audio_class)
                        & (self.comparison_data["normalization_method"] == method)
                    ][metric].dropna()

                    if len(class_method_data) > 1:
                        std_dev = class_method_data.std()
                        consistency_data.append(
                            {
                                "Class": audio_class,
                                "Method": method,
                                "Standard_Deviation": std_dev,
                            }
                        )

            if consistency_data:
                df = pd.DataFrame(consistency_data)

                # Create grouped bar plot
                classes = df["Class"].unique()
                methods_in_plot = df["Method"].unique()
                x = np.arange(len(classes))
                width = 0.8 / len(methods_in_plot)

                for j, method in enumerate(methods_in_plot):
                    method_data = df[df["Method"] == method]
                    values = [
                        method_data[method_data["Class"] == cls][
                            "Standard_Deviation"
                        ].iloc[0]
                        if not method_data[method_data["Class"] == cls].empty
                        else 0
                        for cls in classes
                    ]

                    ax.bar(x + j * width, values, width, label=method, alpha=0.8)

                ax.set_xlabel("Audio Class")
                ax.set_ylabel(f"Standard Deviation ({metric})")
                ax.set_title(title)
                ax.set_xticks(x + width * (len(methods_in_plot) - 1) / 2)
                ax.set_xticklabels(classes, rotation=45)
                ax.legend()
                ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = self.output_dir / "normalization_effectiveness.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved normalization effectiveness plot: {output_path}")
        return output_path

    def plot_class_comparison(self) -> Path:
        """Plot class-wise audio characteristics comparison."""
        if self.class_summary_data is None:
            print("Class summary data not available")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("Audio Characteristics by Class", fontsize=16, fontweight="bold")

        metrics = [
            ("original_rms_mean", "original_rms_std", "Original RMS Level (dB)"),
            ("original_peak_mean", "original_peak_std", "Original Peak Level (dB)"),
            ("original_lufs_mean", "original_lufs_std", "Original LUFS"),
            (
                "original_dynamic_range_mean",
                "original_dynamic_range_std",
                "Original Dynamic Range (dB)",
            ),
        ]

        for i, (mean_col, std_col, ylabel) in enumerate(metrics):
            ax = axes[i // 2, i % 2]

            classes = self.class_summary_data["class_name"]
            means = self.class_summary_data[mean_col]
            stds = self.class_summary_data[std_col]

            bars = ax.bar(classes, means, yerr=stds, capsize=5, alpha=0.8)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Audio Class")
            ax.set_title(ylabel.replace("Original ", ""))
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{mean:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        plt.tight_layout()

        output_path = self.output_dir / "class_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved class comparison plot: {output_path}")
        return output_path

    def plot_gain_distributions(self) -> Path:
        """Plot gain applied by different normalization methods."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(
            "Gain Applied by Normalization Methods", fontsize=16, fontweight="bold"
        )

        methods = ["RMS", "Peak", "LUFS", "Dynamic"]

        for i, method in enumerate(methods):
            ax = axes[i // 2, i % 2]

            method_data = self.comparison_data[
                self.comparison_data["normalization_method"] == method
            ]

            for audio_class in method_data["audio_class"].unique():
                class_data = method_data[method_data["audio_class"] == audio_class][
                    "gain_applied_db"
                ].dropna()

                if len(class_data) > 0:
                    ax.hist(
                        class_data, bins=20, alpha=0.6, label=audio_class, density=True
                    )

            ax.set_xlabel("Gain Applied (dB)")
            ax.set_ylabel("Density")
            ax.set_title(f"{method} Normalization")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Add statistics text
            all_gains = method_data["gain_applied_db"].dropna()
            if len(all_gains) > 0:
                mean_gain = all_gains.mean()
                std_gain = all_gains.std()
                ax.text(
                    0.05,
                    0.95,
                    f"μ = {mean_gain:.1f} dB\nσ = {std_gain:.1f} dB",
                    transform=ax.transAxes,
                    verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                )

        plt.tight_layout()

        output_path = self.output_dir / "gain_distributions.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved gain distributions plot: {output_path}")
        return output_path

    def plot_class_statistics(self) -> Path:
        """Plot comprehensive class statistics overview."""
        if self.class_summary_data is None:
            return None

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            "Class Statistics: Normalization Consistency Analysis",
            fontsize=16,
            fontweight="bold",
        )

        consistency_metrics = [
            ("rms_norm_consistency", "RMS Norm Consistency (dB)"),
            ("peak_norm_consistency", "Peak Norm Consistency (dB)"),
            ("lufs_norm_consistency", "LUFS Norm Consistency"),
            ("dynamic_norm_consistency", "Dynamic Norm Consistency (dB)"),
        ]

        # Plot consistency metrics
        for i, (metric, title) in enumerate(consistency_metrics):
            ax = axes[i // 3, i % 3]

            classes = self.class_summary_data["class_name"]
            values = self.class_summary_data[metric]

            bars = ax.bar(
                classes,
                values,
                alpha=0.8,
                color=plt.cm.Set3(np.linspace(0, 1, len(classes))),
            )
            ax.set_ylabel("Standard Deviation")
            ax.set_xlabel("Audio Class")
            ax.set_title(title)
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, alpha=0.3)

            # Add value labels
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

        # Add file count and duration info
        ax = axes[1, 2]
        classes = self.class_summary_data["class_name"]
        file_counts = self.class_summary_data["num_files"]
        durations = (
            self.class_summary_data["total_duration_seconds"] / 3600
        )  # Convert to hours

        x = np.arange(len(classes))
        width = 0.35

        ax2 = ax.twinx()
        bars1 = ax.bar(x - width / 2, file_counts, width, label="File Count", alpha=0.8)
        bars2 = ax2.bar(
            x + width / 2,
            durations,
            width,
            label="Duration (hours)",
            alpha=0.8,
            color="orange",
        )

        ax.set_xlabel("Audio Class")
        ax.set_ylabel("File Count")
        ax2.set_ylabel("Duration (hours)")
        ax.set_title("Dataset Composition")
        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=45)

        # Combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = self.output_dir / "class_statistics.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved class statistics plot: {output_path}")
        return output_path

    def create_publication_summary(self) -> Path:
        """Create a single summary figure suitable for publication."""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        fig.suptitle(
            "Spatial Audio Dataset: Normalization Analysis Summary",
            fontsize=18,
            fontweight="bold",
        )

        # 1. Original level distributions
        ax1 = fig.add_subplot(gs[0, :2])
        metrics = ["rms_db", "peak_db", "lufs"]
        colors = ["blue", "red", "green"]
        labels = ["RMS (dB)", "Peak (dB)", "LUFS"]

        original_data = self.comparison_data[
            self.comparison_data["normalization_method"] == "Original"
        ]

        for metric, color, label in zip(metrics, colors, labels):
            data = original_data[metric].dropna()
            ax1.hist(data, bins=25, alpha=0.6, color=color, label=label, density=True)

        ax1.set_xlabel("Audio Level")
        ax1.set_ylabel("Density")
        ax1.set_title("A) Original Audio Level Distributions")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Normalization consistency
        ax2 = fig.add_subplot(gs[0, 2])
        if self.class_summary_data is not None:
            methods = ["RMS", "Peak", "LUFS", "Dynamic"]
            consistency_cols = [
                "rms_norm_consistency",
                "peak_norm_consistency",
                "lufs_norm_consistency",
                "dynamic_norm_consistency",
            ]

            avg_consistency = [
                self.class_summary_data[col].mean() for col in consistency_cols
            ]

            bars = ax2.bar(
                methods,
                avg_consistency,
                alpha=0.8,
                color=["blue", "red", "green", "purple"],
            )
            ax2.set_ylabel("Avg. Consistency (Std Dev)")
            ax2.set_title("B) Normalization\nEffectiveness")
            ax2.tick_params(axis="x", rotation=45)
            ax2.grid(True, alpha=0.3)

            # Highlight best method
            best_idx = np.argmin(avg_consistency)
            bars[best_idx].set_color("gold")

        # 3. Class characteristics
        ax3 = fig.add_subplot(gs[1, :])
        if self.class_summary_data is not None:
            classes = self.class_summary_data["class_name"]
            x = np.arange(len(classes))
            width = 0.25

            rms_means = self.class_summary_data["original_rms_mean"]
            peak_means = self.class_summary_data["original_peak_mean"]
            lufs_means = self.class_summary_data["original_lufs_mean"]

            ax3.bar(x - width, rms_means, width, label="RMS", alpha=0.8)
            ax3.bar(x, peak_means, width, label="Peak", alpha=0.8)
            ax3.bar(x + width, lufs_means, width, label="LUFS", alpha=0.8)

            ax3.set_xlabel("Audio Class")
            ax3.set_ylabel("Level (dB / LUFS)")
            ax3.set_title("C) Original Audio Characteristics by Class")
            ax3.set_xticks(x)
            ax3.set_xticklabels(classes)
            ax3.legend()
            ax3.grid(True, alpha=0.3)

        # 4. Gain statistics
        ax4 = fig.add_subplot(gs[2, :])
        methods = ["RMS", "Peak", "LUFS", "Dynamic"]
        colors = ["blue", "red", "green", "purple"]

        for i, (method, color) in enumerate(zip(methods, colors)):
            method_data = self.comparison_data[
                self.comparison_data["normalization_method"] == method
            ]["gain_applied_db"].dropna()

            if len(method_data) > 0:
                positions = [i + 1]
                violin_parts = ax4.violinplot(
                    [method_data], positions=positions, widths=0.7
                )
                for pc in violin_parts["bodies"]:
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)

        ax4.set_xlabel("Normalization Method")
        ax4.set_ylabel("Gain Applied (dB)")
        ax4.set_title("D) Gain Distribution by Normalization Method")
        ax4.set_xticks(range(1, len(methods) + 1))
        ax4.set_xticklabels(methods)
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0, color="black", linestyle="--", alpha=0.5)

        output_path = self.output_dir / "publication_summary.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved publication summary: {output_path}")
        return output_path


def main():
    """Main entry point for the visualization script."""
    parser = argparse.ArgumentParser(
        description="Create visualizations from normalization analysis reports"
    )

    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("normalization_reports"),
        help="Directory containing analysis reports",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to save figures (default: reports_dir/figures)",
    )

    parser.add_argument(
        "--publication-only",
        action="store_true",
        help="Generate only the publication summary figure",
    )

    args = parser.parse_args()

    if not args.reports_dir.exists():
        print(f"Error: Reports directory does not exist: {args.reports_dir}")
        return 1

    print("NORMALIZATION ANALYSIS VISUALIZATION")
    print("=" * 50)
    print(f"Reports directory: {args.reports_dir}")
    print(f"Output directory: {args.output_dir or args.reports_dir / 'figures'}")
    print("=" * 50)

    try:
        visualizer = NormalizationVisualizer(args.reports_dir, args.output_dir)

        if args.publication_only:
            print("Creating publication summary...")
            figure_path = visualizer.create_publication_summary()
            figure_paths = [figure_path] if figure_path else []
        else:
            print("Creating all visualizations...")
            figure_paths = visualizer.create_all_visualizations()

            # Also create publication summary
            pub_path = visualizer.create_publication_summary()
            if pub_path:
                figure_paths.append(pub_path)

        print(f"\nCreated {len(figure_paths)} figures:")
        for path in figure_paths:
            if path:
                print(f"  - {path}")

        print(f"\nVisualization completed successfully!")

    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
