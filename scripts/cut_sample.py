"""Professional CLI to cut dataset samples using `AudioCutter`.

Features:
  - Robust import handling for local package during development
  - Validates input path and writable output directory
  - Logging with levels, dry-run mode and graceful fallback for unsafe outputs
  - Returns non-zero exit codes on failure for CI integration

Example:
  python scripts/cut_sample.py --input dataset/200/audiosInteiros/stereo1.wav --output temp/chunks
"""

import importlib.util
import logging
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

if importlib.util.find_spec("src") is None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

from src.processing import AudioCutter, CutterConfig


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _parse_args() -> Namespace:
    parser = ArgumentParser(
        description="Cut dataset audio files into fixed-length chunks"
    )
    parser.add_argument(
        "--input", required=True, help="Path to sample file or folder inside dataset"
    )
    parser.add_argument(
        "--output", required=True, help="Directory to save generated chunks"
    )
    parser.add_argument(
        "--chunk", type=float, default=10.0, help="Chunk length in seconds"
    )
    parser.add_argument(
        "--workers", type=int, default=11, help="Number of threads to use"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print actions without writing files",
    )
    parser.add_argument(
        "--log", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    return parser.parse_args()


def _validate_paths(input_path: Path, output_path: Path) -> None:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file() and not input_path.parent.exists():
        raise FileNotFoundError(
            f"Input file's parent does not exist: {input_path.parent}"
        )

    try:
        output_path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise PermissionError(
            f"Cannot create or write to output directory {output_path}: {exc}"
        )


def main() -> int:
    args = _parse_args()

    _configure_logging(args.log)
    logger = logging.getLogger("cut_sample")

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    try:
        _validate_paths(input_path, output_path)
    except Exception as exc:
        logger.error(str(exc))
        return 2

    config = CutterConfig(chunk_seconds=args.chunk, num_workers=args.workers)
    cutter = AudioCutter(config)

    root = input_path.parent if input_path.is_file() else input_path

    logger.info(
        "Starting cut: input=%s output=%s chunk=%.2fs workers=%d",
        root,
        output_path,
        args.chunk,
        args.workers,
    )

    if args.dry_run:
        logger.info("Dry-run: scanning files without writing chunks")
        files = list(root.rglob(cutter.config.file_pattern))
        logger.info("Found %d files that would be processed", len(files))
        return 0

    try:
        generated = cutter.cut_folder(root, output_path)
        if generated:
            logger.info("Generated %d chunks", len(generated))
            for p in generated:
                logger.debug("chunk: %s", p)
            return 0
        else:
            logger.warning("No chunks generated")
            return 1

    except Exception as exc:
        logger.exception("Processing failed: %s", exc)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
