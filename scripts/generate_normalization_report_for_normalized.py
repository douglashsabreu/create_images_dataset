"""Generate normalization analysis report for a normalized dataset.

Usage:
  python scripts/generate_normalization_report_for_normalized.py --input dataset_normalized --output reports_normalization --target -23.0
"""

import importlib.util
from argparse import ArgumentParser
from pathlib import Path

if importlib.util.find_spec("src") is None:
    project_root = Path(__file__).resolve().parents[1]
    import sys

    sys.path.insert(0, str(project_root))

from src.reports.normalization_analyzer import NormalizationAnalyzer


def _parse_args():
    p = ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--target", type=float, default=-23.0)
    p.add_argument("--sample-rate", type=int, default=48000)
    return p.parse_args()


def main():
    args = _parse_args()
    input_root = Path(args.input)
    output_dir = Path(args.output)

    analyzer = NormalizationAnalyzer(
        target_level_db=args.target, sample_rate=args.sample_rate, output_dir=output_dir
    )

    report = analyzer.analyze_dataset(input_root)
    analyzer.save_report(report, report_name="dataset_normalized_report")


if __name__ == "__main__":
    main()


