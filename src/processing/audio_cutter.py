"""Parallel audio cutter for splitting audio files into fixed-length segments.

This module provides the `AudioCutter` class which splits audio files in a
directory into fixed-length chunks (in seconds) using multithreading. The
last chunk of each file is discarded to avoid variable-length tails.
"""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import torchaudio


@dataclass
class CutterConfig:
    """Configuration for audio cutting.

    Attributes:
        chunk_seconds: Length of each chunk in seconds.
        sample_rate: Target sample rate for resampling (no resample if None).
        num_workers: Number of worker threads to use.
        file_pattern: Glob pattern to match audio files.
    """

    chunk_seconds: float = 10.0
    sample_rate: Optional[int] = 48000
    num_workers: int = 11
    max_retries: int = 3
    file_pattern: str = "*.wav"


class AudioCutter:
    """Cut audio files in a directory into fixed-length segments in parallel.

    The cutter writes segmented audio files into an output directory preserving
    the input directory structure. The last partial segment is discarded.
    """

    def __init__(self, config: CutterConfig):
        """Initialize the cutter with a configuration.

        Args:
            config: CutterConfig instance
        """
        self.config = config

    def cut_folder(self, input_dir: Path, output_dir: Path) -> List[Path]:
        """Cut all audio files found under `input_dir` and save to `output_dir`.

        Args:
            input_dir: Root directory containing class subfolders or wav files
            output_dir: Directory where chunks will be saved

        Returns:
            List of paths to generated chunk files
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        files = list(input_dir.rglob(self.config.file_pattern))
        if not files:
            return []

        logger = logging.getLogger(__name__)
        results: List[Path] = []

        with ThreadPoolExecutor(max_workers=self.config.num_workers) as exe:
            futs = {
                exe.submit(self._process_file, f, input_dir, output_dir): f
                for f in files
            }

            for fut in as_completed(futs):
                src_file = futs.get(fut)
                try:
                    out_paths = fut.result()
                    results.extend(out_paths)
                except Exception as exc:
                    logger.exception("Unhandled error processing %s: %s", src_file, exc)

        return results

    def _process_file(
        self, file_path: Path, root_input: Path, root_output: Path
    ) -> List[Path]:
        """Load a file, split into chunks and write them out.

        This method discards the final chunk if it's shorter than the configured
        chunk length.
        """
        logger = logging.getLogger(__name__)
        attempts = 0
        max_attempts = max(1, getattr(self.config, "max_retries", 1))
        while attempts < max_attempts:
            attempts += 1
            try:
                waveform, sr = torchaudio.load(str(file_path))

                if self.config.sample_rate and sr != self.config.sample_rate:
                    resampler = torchaudio.transforms.Resample(
                        orig_freq=sr, new_freq=self.config.sample_rate
                    )
                    waveform = resampler(waveform)
                    sr = self.config.sample_rate

                num_samples_per_chunk = int(self.config.chunk_seconds * sr)
                total_samples = waveform.shape[1]

                num_full_chunks = total_samples // num_samples_per_chunk
                if num_full_chunks == 0:
                    return []

                rel_dir = file_path.parent.relative_to(root_input)
                out_dir = root_output / rel_dir
                out_dir.mkdir(parents=True, exist_ok=True)

                out_paths: List[Path] = []

                for i in range(num_full_chunks):
                    start = i * num_samples_per_chunk
                    end = start + num_samples_per_chunk
                    chunk = waveform[:, start:end]

                    out_name = f"{file_path.stem}_chunk{i:04d}{file_path.suffix}"
                    out_path = out_dir / out_name
                    torchaudio.save(str(out_path), chunk, sample_rate=sr)
                    out_paths.append(out_path)

                return out_paths

            except Exception as exc:
                logger.warning("Attempt %d failed for %s: %s", attempts, file_path, exc)
                time.sleep(0.1 * attempts)

        logger.error("Failed to process %s after %d attempts", file_path, attempts)
        return []
