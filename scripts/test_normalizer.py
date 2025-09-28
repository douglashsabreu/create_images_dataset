#!/usr/bin/env python3
"""Test script for the audio normalizer functionality.

This script demonstrates how to use the AudioNormalizer with different
normalization methods and shows analysis of audio levels before and after
normalization.
"""

import sys
import warnings
from pathlib import Path

import torch
import torchaudio

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

sys.path.append(str(Path(__file__).parent.parent))

from src.audio import AudioNormalizer, NormalizationType


def test_normalizer_with_audio_file(audio_path: Path) -> None:
    """Test the normalizer with a specific audio file.

    Args:
        audio_path: Path to the audio file to test
    """
    print(f"\n{'=' * 60}")
    print(f"Testing with: {audio_path.name}")
    print(f"{'=' * 60}")

    try:
        waveform, sample_rate = torchaudio.load(str(audio_path))
        print(f"Original audio shape: {waveform.shape}")
        print(f"Sample rate: {sample_rate} Hz")

        normalizers = [
            ("RMS", AudioNormalizer(normalization_type=NormalizationType.RMS)),
            ("Peak", AudioNormalizer(normalization_type=NormalizationType.PEAK)),
            ("LUFS", AudioNormalizer(normalization_type=NormalizationType.LUFS)),
            (
                "Dynamic Range",
                AudioNormalizer(normalization_type=NormalizationType.DYNAMIC_RANGE),
            ),
        ]

        print("\nOriginal Audio Analysis:")
        original_analysis = normalizers[0][1].analyze_audio_levels(waveform)
        _print_audio_analysis(original_analysis)

        print("\nNormalization Results:")
        print("-" * 60)

        for name, normalizer in normalizers:
            print(f"\n{name} Normalization:")
            try:
                normalized_waveform = normalizer.normalize_waveform(waveform)
                analysis = normalizer.analyze_audio_levels(normalized_waveform)
                _print_audio_analysis(analysis)

                gain_applied = torch.max(torch.abs(normalized_waveform)) / torch.max(
                    torch.abs(waveform)
                )
                print(f"Gain applied: {20 * torch.log10(gain_applied):.2f} dB")

            except Exception as e:
                print(f"Error with {name} normalization: {e}")

    except Exception as e:
        print(f"Error loading audio file: {e}")


def _print_audio_analysis(analysis: dict) -> None:
    """Print audio analysis results in a formatted way.

    Args:
        analysis: Analysis results dictionary
    """
    print(f"  RMS Level: {analysis['rms_db']:.2f} dB")
    print(f"  Peak Level: {analysis['peak_db']:.2f} dB")
    print(f"  LUFS: {analysis['lufs']:.2f}")
    print(f"  Crest Factor: {analysis['crest_factor']:.2f}")
    print(f"  Dynamic Range: {analysis['dynamic_range_db']:.2f} dB")


def test_normalization_methods() -> None:
    """Test different normalization methods with sample audio files."""
    print("Audio Normalizer Test Suite")
    print("=" * 60)

    test_audio_dir = Path("test_audio_sample")

    if not test_audio_dir.exists():
        print(f"Test audio directory not found: {test_audio_dir}")
        print("Please ensure test audio files are available.")
        return

    audio_files = []
    for subdir in test_audio_dir.iterdir():
        if subdir.is_dir():
            audio_files.extend(list(subdir.glob("*.wav"))[:1])

    if not audio_files:
        print("No audio files found in test directory.")
        return

    print(f"Found {len(audio_files)} test audio files")

    for audio_file in audio_files:
        test_normalizer_with_audio_file(audio_file)


def test_custom_normalization() -> None:
    """Test custom normalization parameters."""
    print(f"\n{'=' * 60}")
    print("Custom Normalization Parameters Test")
    print(f"{'=' * 60}")

    test_audio = Path("test_audio_sample/1.0/mono1_snippet001.wav")

    if not test_audio.exists():
        print(f"Test audio file not found: {test_audio}")
        return

    waveform, _ = torchaudio.load(str(test_audio))

    target_levels = [-12.0, -18.0, -23.0, -30.0]

    for target_level in target_levels:
        print(f"\nTarget Level: {target_level} dB")
        normalizer = AudioNormalizer(
            target_level_db=target_level, normalization_type=NormalizationType.RMS
        )

        normalized = normalizer.normalize_waveform(waveform)
        analysis = normalizer.analyze_audio_levels(normalized)

        print(f"Achieved RMS Level: {analysis['rms_db']:.2f} dB")
        print(f"Difference: {abs(analysis['rms_db'] - target_level):.2f} dB")


def main() -> None:
    """Main test function."""
    try:
        test_normalization_methods()
        test_custom_normalization()

        print(f"\n{'=' * 60}")
        print("All tests completed successfully!")
        print(f"{'=' * 60}")

    except Exception as e:
        print(f"Test failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
