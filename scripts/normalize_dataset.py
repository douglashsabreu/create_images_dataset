"""Normalize all audio files in a cut dataset and save to `dataset_normalized`.

This script walks `dataset_cutted/`, applies `AudioNormalizer` to each file and
saves the normalized audio preserving the folder structure under
`dataset_normalized/`.

Usage:
  python scripts/normalize_dataset.py --input dataset_cutted --output dataset_normalized --method RMS --workers 11
"""

import importlib.util
import logging
import sys
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

if importlib.util.find_spec("src") is None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))

import torchaudio

from src.audio.normalizer import AudioNormalizer, NormalizationType


def _parse_args():
    p = ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--method", default="RMS", choices=[m.value for m in NormalizationType]
    )
    p.add_argument("--sample-rate", type=int, default=48000)
    p.add_argument("--workers", type=int, default=11)
    return p.parse_args()


def _gather_files(root: Path) -> List[Path]:
    return list(root.rglob("*.wav"))


def _process_file(
    file_path: Path, root_in: Path, root_out: Path, normalizer: AudioNormalizer
) -> Path:
    rel = file_path.parent.relative_to(root_in)
    out_dir = root_out / rel
    out_dir.mkdir(parents=True, exist_ok=True)

    waveform, sr = torchaudio.load(str(file_path))
    if sr != normalizer.sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sr, new_freq=normalizer.sample_rate
        )
        waveform = resampler(waveform)

    normalized = normalizer.normalize_waveform(waveform)
    out_path = out_dir / file_path.name
    torchaudio.save(str(out_path), normalized, sample_rate=normalizer.sample_rate)
    return out_path


def main():
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("normalize_dataset")

    input_root = Path(args.input)
    output_root = Path(args.output)

    files = _gather_files(input_root)
    logger.info("Found %d files to normalize", len(files))

    method = NormalizationType(args.method)
    normalizer = AudioNormalizer(
        target_level_db=-23.0, normalization_type=method, sample_rate=args.sample_rate
    )

    processed = []
    with ThreadPoolExecutor(max_workers=args.workers) as exe:
        futs = {
            exe.submit(_process_file, f, input_root, output_root, normalizer): f
            for f in files
        }
        for fut in as_completed(futs):
            try:
                p = fut.result()
                processed.append(p)
            except Exception as e:
                logger.exception("Failed to normalize %s: %s", futs.get(fut), e)

    logger.info("Completed normalization: %d files", len(processed))


if __name__ == "__main__":
    main()
