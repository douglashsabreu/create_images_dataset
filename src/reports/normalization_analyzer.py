"""Audio normalization analysis and reporting module for academic documentation."""

import csv
import math
import multiprocessing as mp
import statistics
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
import torchaudio
from tqdm import tqdm

from ..audio import AudioNormalizer, NormalizationType

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")


@dataclass
class AudioMetrics:
    """Audio metrics for a single file."""

    file_path: str
    audio_class: str
    duration_seconds: float
    sample_rate: int
    num_channels: int

    # Original metrics
    original_rms_db: float
    original_peak_db: float
    original_lufs: float
    original_crest_factor: float
    original_dynamic_range_db: float

    # RMS normalized metrics
    rms_norm_rms_db: float
    rms_norm_peak_db: float
    rms_norm_lufs: float
    rms_norm_crest_factor: float
    rms_norm_dynamic_range_db: float
    rms_norm_gain_applied_db: float

    # Peak normalized metrics
    peak_norm_rms_db: float
    peak_norm_peak_db: float
    peak_norm_lufs: float
    peak_norm_crest_factor: float
    peak_norm_dynamic_range_db: float
    peak_norm_gain_applied_db: float

    # LUFS normalized metrics
    lufs_norm_rms_db: float
    lufs_norm_peak_db: float
    lufs_norm_lufs: float
    lufs_norm_crest_factor: float
    lufs_norm_dynamic_range_db: float
    lufs_norm_gain_applied_db: float

    # Dynamic range normalized metrics
    dynamic_norm_rms_db: float
    dynamic_norm_peak_db: float
    dynamic_norm_lufs: float
    dynamic_norm_crest_factor: float
    dynamic_norm_dynamic_range_db: float
    dynamic_norm_gain_applied_db: float


@dataclass
class ClassStatistics:
    """Statistics for a single audio class."""

    class_name: str
    num_files: int
    total_duration_seconds: float

    # Original statistics
    original_rms_mean: float
    original_rms_std: float
    original_peak_mean: float
    original_peak_std: float
    original_lufs_mean: float
    original_lufs_std: float
    original_crest_mean: float
    original_crest_std: float
    original_dynamic_range_mean: float
    original_dynamic_range_std: float

    # Normalization effectiveness
    rms_norm_consistency: float  # Standard deviation of RMS levels after normalization
    peak_norm_consistency: float
    lufs_norm_consistency: float
    dynamic_norm_consistency: float

    # Gain statistics
    rms_gain_mean: float
    rms_gain_std: float
    peak_gain_mean: float
    peak_gain_std: float
    lufs_gain_mean: float
    lufs_gain_std: float
    dynamic_gain_mean: float
    dynamic_gain_std: float


@dataclass
class NormalizationReport:
    """Complete normalization analysis report."""

    dataset_path: str
    total_files: int
    total_duration_hours: float
    analysis_timestamp: str
    target_level_db: float

    file_metrics: List[AudioMetrics]
    class_statistics: List[ClassStatistics]

    # Overall dataset statistics
    overall_original_rms_mean: float
    overall_original_rms_std: float
    overall_original_peak_mean: float
    overall_original_peak_std: float
    overall_original_lufs_mean: float
    overall_original_lufs_std: float

    # Normalization comparison
    best_normalization_method: str
    normalization_effectiveness_scores: Dict[str, float]


class NormalizationAnalyzer:
    """Analyzes audio normalization effects across a dataset for academic reporting."""

    def __init__(
        self,
        target_level_db: float = -23.0,
        sample_rate: int = 48000,
        output_dir: Path = Path("normalization_reports"),
    ):
        """Initialize the normalization analyzer.

        Args:
            target_level_db: Target normalization level in dB
            sample_rate: Target sample rate for analysis
            output_dir: Directory to save reports
        """
        self.target_level_db = target_level_db
        self.sample_rate = sample_rate
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.normalizers = {
            "RMS": AudioNormalizer(
                target_level_db=target_level_db,
                normalization_type=NormalizationType.RMS,
                sample_rate=sample_rate,
            ),
            "Peak": AudioNormalizer(
                target_level_db=target_level_db,
                normalization_type=NormalizationType.PEAK,
                sample_rate=sample_rate,
            ),
            "LUFS": AudioNormalizer(
                target_level_db=target_level_db,
                normalization_type=NormalizationType.LUFS,
                sample_rate=sample_rate,
            ),
            "Dynamic": AudioNormalizer(
                target_level_db=target_level_db,
                normalization_type=NormalizationType.DYNAMIC_RANGE,
                sample_rate=sample_rate,
            ),
        }

    def analyze_dataset(self, dataset_path: Path) -> NormalizationReport:
        """Analyze normalization effects across the entire dataset.

        Args:
            dataset_path: Path to dataset directory containing audio class subdirectories

        Returns:
            Complete normalization analysis report
        """
        print(f"Analyzing normalization effects for dataset: {dataset_path}")

        file_metrics = []
        audio_files = self._collect_audio_files(dataset_path)

        print(f"Found {len(audio_files)} audio files")

        # Prepare arguments for parallel processing (file_path, class_name)
        analysis_args = [
            (file_path, class_name) for file_path, class_name in audio_files
        ]

        # Use parallel processing with progress bar and per-worker initializer to
        # reuse the NormalizationAnalyzer inside each process (reduces reinit cost)
        max_workers = max(1, mp.cpu_count() - 1)
        print(f"Using {max_workers} workers for parallel analysis")

        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_worker_init,
            initargs=(self.target_level_db, self.sample_rate),
        ) as executor:
            # Submit all tasks
            future_to_args = {
                executor.submit(_analyze_single_file_worker, file_path, class_name): (
                    file_path,
                    class_name,
                )
                for file_path, class_name in analysis_args
            }

            # Process results with progress bar
            for future in tqdm(
                as_completed(future_to_args),
                total=len(analysis_args),
                desc="Analyzing audio files",
            ):
                metrics = future.result()
                if metrics is not None:
                    file_metrics.append(metrics)

        class_stats = self._calculate_class_statistics(file_metrics)
        overall_stats = self._calculate_overall_statistics(file_metrics)

        report = NormalizationReport(
            dataset_path=str(dataset_path),
            total_files=len(file_metrics),
            total_duration_hours=sum(m.duration_seconds for m in file_metrics) / 3600,
            analysis_timestamp=pd.Timestamp.now().isoformat(),
            target_level_db=self.target_level_db,
            file_metrics=file_metrics,
            class_statistics=class_stats,
            **overall_stats,
        )

        return report

    def _collect_audio_files(self, dataset_path: Path) -> List[Tuple[Path, str]]:
        """Collect all audio files from the dataset directory.

        Args:
            dataset_path: Path to dataset directory

        Returns:
            List of (file_path, class_name) tuples
        """
        audio_files: List[Tuple[Path, str]] = []

        # Use recursive search to support nested layouts (e.g. class/audiosInteiros/*.wav)
        for audio_file in dataset_path.rglob("*.wav"):
            try:
                class_name = audio_file.relative_to(dataset_path).parts[0]
            except Exception:
                class_name = audio_file.parent.name
            audio_files.append((audio_file, class_name))

        return audio_files

    def _analyze_file(self, file_path: Path, class_name: str) -> AudioMetrics:
        """Analyze normalization effects for a single audio file.

        Args:
            file_path: Path to audio file
            class_name: Audio class name

        Returns:
            Audio metrics for the file
        """
        waveform, original_sr = torchaudio.load(str(file_path))

        if original_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=original_sr, new_freq=self.sample_rate
            )
            waveform = resampler(waveform)

        duration_seconds = waveform.shape[1] / self.sample_rate

        original_analysis = self.normalizers["RMS"].analyze_audio_levels(waveform)

        normalized_results = {}
        for norm_name, normalizer in self.normalizers.items():
            try:
                normalized_waveform = normalizer.normalize_waveform(waveform)
                normalized_analysis = normalizer.analyze_audio_levels(
                    normalized_waveform
                )

                original_peak = torch.max(torch.abs(waveform))
                normalized_peak = torch.max(torch.abs(normalized_waveform))
                gain_applied = 20 * torch.log10(normalized_peak / original_peak)

                normalized_results[norm_name] = {
                    **normalized_analysis,
                    "gain_applied_db": float(gain_applied),
                }
            except Exception as e:
                print(f"Warning: {norm_name} normalization failed for {file_path}: {e}")
                normalized_results[norm_name] = {
                    "rms_db": 0.0,
                    "peak_db": 0.0,
                    "lufs": 0.0,
                    "crest_factor": 0.0,
                    "dynamic_range_db": 0.0,
                    "gain_applied_db": 0.0,
                }

        return AudioMetrics(
            file_path=str(file_path),
            audio_class=class_name,
            duration_seconds=duration_seconds,
            sample_rate=self.sample_rate,
            num_channels=waveform.shape[0],
            # Original metrics
            original_rms_db=original_analysis["rms_db"],
            original_peak_db=original_analysis["peak_db"],
            original_lufs=original_analysis["lufs"],
            original_crest_factor=original_analysis["crest_factor"],
            original_dynamic_range_db=original_analysis["dynamic_range_db"],
            # RMS normalized metrics
            rms_norm_rms_db=normalized_results["RMS"]["rms_db"],
            rms_norm_peak_db=normalized_results["RMS"]["peak_db"],
            rms_norm_lufs=normalized_results["RMS"]["lufs"],
            rms_norm_crest_factor=normalized_results["RMS"]["crest_factor"],
            rms_norm_dynamic_range_db=normalized_results["RMS"]["dynamic_range_db"],
            rms_norm_gain_applied_db=normalized_results["RMS"]["gain_applied_db"],
            # Peak normalized metrics
            peak_norm_rms_db=normalized_results["Peak"]["rms_db"],
            peak_norm_peak_db=normalized_results["Peak"]["peak_db"],
            peak_norm_lufs=normalized_results["Peak"]["lufs"],
            peak_norm_crest_factor=normalized_results["Peak"]["crest_factor"],
            peak_norm_dynamic_range_db=normalized_results["Peak"]["dynamic_range_db"],
            peak_norm_gain_applied_db=normalized_results["Peak"]["gain_applied_db"],
            # LUFS normalized metrics
            lufs_norm_rms_db=normalized_results["LUFS"]["rms_db"],
            lufs_norm_peak_db=normalized_results["LUFS"]["peak_db"],
            lufs_norm_lufs=normalized_results["LUFS"]["lufs"],
            lufs_norm_crest_factor=normalized_results["LUFS"]["crest_factor"],
            lufs_norm_dynamic_range_db=normalized_results["LUFS"]["dynamic_range_db"],
            lufs_norm_gain_applied_db=normalized_results["LUFS"]["gain_applied_db"],
            # Dynamic normalized metrics
            dynamic_norm_rms_db=normalized_results["Dynamic"]["rms_db"],
            dynamic_norm_peak_db=normalized_results["Dynamic"]["peak_db"],
            dynamic_norm_lufs=normalized_results["Dynamic"]["lufs"],
            dynamic_norm_crest_factor=normalized_results["Dynamic"]["crest_factor"],
            dynamic_norm_dynamic_range_db=normalized_results["Dynamic"][
                "dynamic_range_db"
            ],
            dynamic_norm_gain_applied_db=normalized_results["Dynamic"][
                "gain_applied_db"
            ],
        )

    def _calculate_class_statistics(
        self, file_metrics: List[AudioMetrics]
    ) -> List[ClassStatistics]:
        """Calculate statistics for each audio class.

        Args:
            file_metrics: List of audio metrics for all files

        Returns:
            List of class statistics
        """
        class_groups = {}
        for metric in file_metrics:
            if metric.audio_class not in class_groups:
                class_groups[metric.audio_class] = []
            class_groups[metric.audio_class].append(metric)

        class_stats = []

        def _clean_values(values: List[float]) -> List[float]:
            cleaned: List[float] = []
            for v in values:
                try:
                    fv = float(v)
                except Exception:
                    continue
                if math.isfinite(fv):
                    cleaned.append(fv)
            return cleaned

        for class_name, metrics in class_groups.items():
            original_rms = _clean_values([m.original_rms_db for m in metrics])
            original_peak = _clean_values([m.original_peak_db for m in metrics])
            original_lufs = _clean_values([m.original_lufs for m in metrics])
            original_crest = _clean_values([m.original_crest_factor for m in metrics])
            original_dynamic = _clean_values(
                [m.original_dynamic_range_db for m in metrics]
            )

            rms_norm_levels = _clean_values([m.rms_norm_rms_db for m in metrics])
            peak_norm_levels = _clean_values([m.peak_norm_peak_db for m in metrics])
            lufs_norm_levels = _clean_values([m.lufs_norm_lufs for m in metrics])
            dynamic_norm_levels = _clean_values(
                [m.dynamic_norm_rms_db for m in metrics]
            )

            rms_gains = _clean_values([m.rms_norm_gain_applied_db for m in metrics])
            peak_gains = _clean_values([m.peak_norm_gain_applied_db for m in metrics])
            lufs_gains = _clean_values([m.lufs_norm_gain_applied_db for m in metrics])
            dynamic_gains = _clean_values(
                [m.dynamic_norm_gain_applied_db for m in metrics]
            )

            class_stat = ClassStatistics(
                class_name=class_name,
                num_files=len(metrics),
                total_duration_seconds=sum(m.duration_seconds for m in metrics),
                # Original statistics
                original_rms_mean=statistics.mean(original_rms),
                original_rms_std=statistics.stdev(original_rms)
                if len(original_rms) > 1
                else 0.0,
                original_peak_mean=statistics.mean(original_peak),
                original_peak_std=statistics.stdev(original_peak)
                if len(original_peak) > 1
                else 0.0,
                original_lufs_mean=statistics.mean(original_lufs),
                original_lufs_std=statistics.stdev(original_lufs)
                if len(original_lufs) > 1
                else 0.0,
                original_crest_mean=statistics.mean(original_crest),
                original_crest_std=statistics.stdev(original_crest)
                if len(original_crest) > 1
                else 0.0,
                original_dynamic_range_mean=statistics.mean(original_dynamic),
                original_dynamic_range_std=statistics.stdev(original_dynamic)
                if len(original_dynamic) > 1
                else 0.0,
                # Normalization consistency (lower is better)
                rms_norm_consistency=statistics.stdev(rms_norm_levels)
                if len(rms_norm_levels) > 1
                else 0.0,
                peak_norm_consistency=statistics.stdev(peak_norm_levels)
                if len(peak_norm_levels) > 1
                else 0.0,
                lufs_norm_consistency=statistics.stdev(lufs_norm_levels)
                if len(lufs_norm_levels) > 1
                else 0.0,
                dynamic_norm_consistency=statistics.stdev(dynamic_norm_levels)
                if len(dynamic_norm_levels) > 1
                else 0.0,
                # Gain statistics
                rms_gain_mean=statistics.mean(rms_gains),
                rms_gain_std=statistics.stdev(rms_gains) if len(rms_gains) > 1 else 0.0,
                peak_gain_mean=statistics.mean(peak_gains),
                peak_gain_std=statistics.stdev(peak_gains)
                if len(peak_gains) > 1
                else 0.0,
                lufs_gain_mean=statistics.mean(lufs_gains),
                lufs_gain_std=statistics.stdev(lufs_gains)
                if len(lufs_gains) > 1
                else 0.0,
                dynamic_gain_mean=statistics.mean(dynamic_gains),
                dynamic_gain_std=statistics.stdev(dynamic_gains)
                if len(dynamic_gains) > 1
                else 0.0,
            )

            class_stats.append(class_stat)

        return class_stats

    def _calculate_overall_statistics(self, file_metrics: List[AudioMetrics]) -> dict:
        """Calculate overall dataset statistics.

        Args:
            file_metrics: List of audio metrics for all files

        Returns:
            Dictionary of overall statistics
        """
        # If there are no file metrics, return safe defaults
        if not file_metrics:
            return {
                "overall_original_rms_mean": 0.0,
                "overall_original_rms_std": 0.0,
                "overall_original_peak_mean": 0.0,
                "overall_original_peak_std": 0.0,
                "overall_original_lufs_mean": 0.0,
                "overall_original_lufs_std": 0.0,
                "best_normalization_method": "N/A",
                "normalization_effectiveness_scores": {
                    "RMS": 0.0,
                    "Peak": 0.0,
                    "LUFS": 0.0,
                    "Dynamic": 0.0,
                },
            }

        original_rms = [m.original_rms_db for m in file_metrics]
        original_peak = [m.original_peak_db for m in file_metrics]
        original_lufs = [m.original_lufs for m in file_metrics]

        # Calculate normalization effectiveness (consistency scores)
        rms_norm_levels = [m.rms_norm_rms_db for m in file_metrics]
        peak_norm_levels = [m.peak_norm_peak_db for m in file_metrics]
        lufs_norm_levels = [m.lufs_norm_lufs for m in file_metrics]
        dynamic_norm_levels = [m.dynamic_norm_rms_db for m in file_metrics]

        def safe_stdev(values: List[float]) -> float:
            return statistics.stdev(values) if len(values) > 1 else 0.0

        effectiveness_scores = {
            "RMS": 1.0 / (safe_stdev(rms_norm_levels) + 1e-6),
            "Peak": 1.0 / (safe_stdev(peak_norm_levels) + 1e-6),
            "LUFS": 1.0 / (safe_stdev(lufs_norm_levels) + 1e-6),
            "Dynamic": 1.0 / (safe_stdev(dynamic_norm_levels) + 1e-6),
        }

        best_method = max(
            effectiveness_scores.keys(), key=lambda k: effectiveness_scores[k]
        )

        return {
            "overall_original_rms_mean": statistics.mean(original_rms)
            if original_rms
            else 0.0,
            "overall_original_rms_std": statistics.stdev(original_rms)
            if len(original_rms) > 1
            else 0.0,
            "overall_original_peak_mean": statistics.mean(original_peak)
            if original_peak
            else 0.0,
            "overall_original_peak_std": statistics.stdev(original_peak)
            if len(original_peak) > 1
            else 0.0,
            "overall_original_lufs_mean": statistics.mean(original_lufs)
            if original_lufs
            else 0.0,
            "overall_original_lufs_std": statistics.stdev(original_lufs)
            if len(original_lufs) > 1
            else 0.0,
            "best_normalization_method": best_method,
            "normalization_effectiveness_scores": effectiveness_scores,
        }

    def save_report(
        self, report: NormalizationReport, report_name: str = "normalization_analysis"
    ) -> Dict[str, Path]:
        """Save the normalization report in multiple formats.

        Args:
            report: Normalization analysis report
            report_name: Base name for report files

        Returns:
            Dictionary mapping format names to file paths
        """
        output_files = {}

        # Save detailed CSV file with all metrics
        csv_path = self.output_dir / f"{report_name}_detailed.csv"
        self._save_detailed_csv(report, csv_path)
        output_files["detailed_csv"] = csv_path

        # Save summary CSV file with class statistics
        summary_csv_path = self.output_dir / f"{report_name}_class_summary.csv"
        self._save_class_summary_csv(report, summary_csv_path)
        output_files["class_summary_csv"] = summary_csv_path

        # Save text report for documentation
        txt_path = self.output_dir / f"{report_name}_report.txt"
        self._save_text_report(report, txt_path)
        output_files["text_report"] = txt_path

        # Save comparison CSV for plotting
        comparison_csv_path = self.output_dir / f"{report_name}_comparison.csv"
        self._save_comparison_csv(report, comparison_csv_path)
        output_files["comparison_csv"] = comparison_csv_path

        print(f"\nReports saved to: {self.output_dir}")
        for format_name, path in output_files.items():
            print(f"  {format_name}: {path}")

        return output_files

    def _save_detailed_csv(self, report: NormalizationReport, csv_path: Path) -> None:
        """Save detailed per-file metrics to CSV."""
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(asdict(report.file_metrics[0]).keys())
            )
            writer.writeheader()
            for metric in report.file_metrics:
                writer.writerow(asdict(metric))

    def _save_class_summary_csv(
        self, report: NormalizationReport, csv_path: Path
    ) -> None:
        """Save class statistics summary to CSV."""
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(asdict(report.class_statistics[0]).keys())
            )
            writer.writeheader()
            for stat in report.class_statistics:
                writer.writerow(asdict(stat))

    def _save_comparison_csv(self, report: NormalizationReport, csv_path: Path) -> None:
        """Save normalization method comparison data to CSV."""
        comparison_data = []

        for metric in report.file_metrics:
            base_data = {
                "file_path": metric.file_path,
                "audio_class": metric.audio_class,
                "duration_seconds": metric.duration_seconds,
            }

            # Original
            comparison_data.append(
                {
                    **base_data,
                    "normalization_method": "Original",
                    "rms_db": metric.original_rms_db,
                    "peak_db": metric.original_peak_db,
                    "lufs": metric.original_lufs,
                    "crest_factor": metric.original_crest_factor,
                    "dynamic_range_db": metric.original_dynamic_range_db,
                    "gain_applied_db": 0.0,
                }
            )

            # RMS normalized
            comparison_data.append(
                {
                    **base_data,
                    "normalization_method": "RMS",
                    "rms_db": metric.rms_norm_rms_db,
                    "peak_db": metric.rms_norm_peak_db,
                    "lufs": metric.rms_norm_lufs,
                    "crest_factor": metric.rms_norm_crest_factor,
                    "dynamic_range_db": metric.rms_norm_dynamic_range_db,
                    "gain_applied_db": metric.rms_norm_gain_applied_db,
                }
            )

            # Peak normalized
            comparison_data.append(
                {
                    **base_data,
                    "normalization_method": "Peak",
                    "rms_db": metric.peak_norm_rms_db,
                    "peak_db": metric.peak_norm_peak_db,
                    "lufs": metric.peak_norm_lufs,
                    "crest_factor": metric.peak_norm_crest_factor,
                    "dynamic_range_db": metric.peak_norm_dynamic_range_db,
                    "gain_applied_db": metric.peak_norm_gain_applied_db,
                }
            )

            # LUFS normalized
            comparison_data.append(
                {
                    **base_data,
                    "normalization_method": "LUFS",
                    "rms_db": metric.lufs_norm_rms_db,
                    "peak_db": metric.lufs_norm_peak_db,
                    "lufs": metric.lufs_norm_lufs,
                    "crest_factor": metric.lufs_norm_crest_factor,
                    "dynamic_range_db": metric.lufs_norm_dynamic_range_db,
                    "gain_applied_db": metric.lufs_norm_gain_applied_db,
                }
            )

            # Dynamic normalized
            comparison_data.append(
                {
                    **base_data,
                    "normalization_method": "Dynamic",
                    "rms_db": metric.dynamic_norm_rms_db,
                    "peak_db": metric.dynamic_norm_peak_db,
                    "lufs": metric.dynamic_norm_lufs,
                    "crest_factor": metric.dynamic_norm_crest_factor,
                    "dynamic_range_db": metric.dynamic_norm_dynamic_range_db,
                    "gain_applied_db": metric.dynamic_norm_gain_applied_db,
                }
            )

        with open(csv_path, "w", newline="") as f:
            if comparison_data:
                writer = csv.DictWriter(f, fieldnames=comparison_data[0].keys())
                writer.writeheader()
                writer.writerows(comparison_data)

    def _save_text_report(self, report: NormalizationReport, txt_path: Path) -> None:
        """Save a comprehensive text report for documentation."""
        with open(txt_path, "w") as f:
            f.write("AUDIO NORMALIZATION ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Dataset Path: {report.dataset_path}\n")
            f.write(f"Analysis Date: {report.analysis_timestamp}\n")
            f.write(f"Target Level: {report.target_level_db} dB\n")
            f.write(f"Total Files: {report.total_files}\n")
            f.write(f"Total Duration: {report.total_duration_hours:.2f} hours\n\n")

            f.write("OVERALL DATASET STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(
                f"Original RMS Level: {report.overall_original_rms_mean:.2f} ± {report.overall_original_rms_std:.2f} dB\n"
            )
            f.write(
                f"Original Peak Level: {report.overall_original_peak_mean:.2f} ± {report.overall_original_peak_std:.2f} dB\n"
            )
            f.write(
                f"Original LUFS: {report.overall_original_lufs_mean:.2f} ± {report.overall_original_lufs_std:.2f}\n\n"
            )

            f.write("NORMALIZATION METHOD EFFECTIVENESS\n")
            f.write("-" * 35 + "\n")
            for method, score in report.normalization_effectiveness_scores.items():
                f.write(f"{method:12}: {score:.2f}\n")
            f.write(f"\nBest Method: {report.best_normalization_method}\n\n")

            f.write("CLASS-WISE STATISTICS\n")
            f.write("-" * 25 + "\n")
            for stat in report.class_statistics:
                f.write(f"\nClass: {stat.class_name}\n")
                f.write(f"  Files: {stat.num_files}\n")
                f.write(f"  Duration: {stat.total_duration_seconds / 3600:.2f} hours\n")
                f.write(
                    f"  Original RMS: {stat.original_rms_mean:.2f} ± {stat.original_rms_std:.2f} dB\n"
                )
                f.write(
                    f"  Original Peak: {stat.original_peak_mean:.2f} ± {stat.original_peak_std:.2f} dB\n"
                )
                f.write(
                    f"  Original LUFS: {stat.original_lufs_mean:.2f} ± {stat.original_lufs_std:.2f}\n"
                )
                f.write(
                    f"  RMS Normalization Consistency: {stat.rms_norm_consistency:.2f} dB\n"
                )
                f.write(
                    f"  Peak Normalization Consistency: {stat.peak_norm_consistency:.2f} dB\n"
                )
                f.write(
                    f"  LUFS Normalization Consistency: {stat.lufs_norm_consistency:.2f}\n"
                )
                f.write(
                    f"  Dynamic Normalization Consistency: {stat.dynamic_norm_consistency:.2f} dB\n"
                )


def _analyze_single_file(args: Tuple[Path, str, float, int]) -> Optional[AudioMetrics]:
    """Analyze a single audio file for parallel processing.

    Args:
        args: Tuple of (file_path, class_name, target_level_db, sample_rate)

    Returns:
        AudioMetrics object or None if analysis failed
    """
    file_path, class_name, target_level_db, sample_rate = args

    try:
        analyzer = NormalizationAnalyzer(
            sample_rate=sample_rate,
            target_level_db=target_level_db,
            output_dir=Path("temp"),  # Not used in single file analysis
        )
        return analyzer._analyze_file(file_path, class_name)
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return None


# Optimized worker-based analysis: initialize a NormalizationAnalyzer per process
_WORKER_ANALYZER = None


def _worker_init(target_level_db: float, sample_rate: int) -> None:
    """Initializer for process pool workers to create a shared analyzer."""
    global _WORKER_ANALYZER
    _WORKER_ANALYZER = NormalizationAnalyzer(
        target_level_db=target_level_db,
        sample_rate=sample_rate,
        output_dir=Path("temp"),
    )


def _analyze_single_file_worker(
    file_path: Path, class_name: str
) -> Optional[AudioMetrics]:
    global _WORKER_ANALYZER
    try:
        if _WORKER_ANALYZER is None:
            # Fallback to creating a local analyzer
            local = NormalizationAnalyzer(output_dir=Path("temp"))
            return local._analyze_file(file_path, class_name)

        return _WORKER_ANALYZER._analyze_file(file_path, class_name)
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        return None
