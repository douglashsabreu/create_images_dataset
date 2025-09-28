"""Audio processing module for spatial audio analysis."""

import warnings
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import torch
import torchaudio
from nnAudio import features as nn_features

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio")

from .normalizer import AudioNormalizer, NormalizationType


class SpatialAudioProcessor:
    """Processes stereo audio files for spatial audio analysis.

    Handles loading, preprocessing, and multi-resolution STFT computation
    for stereo audio signals used in spatial audio feature extraction.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        n_fft_short: int = 2048,
        n_fft_long: int = 4096,
        hop_length_ratio: float = 0.5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        normalization_type: NormalizationType = NormalizationType.RMS,
        target_level_db: float = -23.0,
    ):
        """Initialize the spatial audio processor.

        Args:
            sample_rate: Target sample rate for audio processing
            n_fft_short: FFT size for short-time analysis (transients)
            n_fft_long: FFT size for long-time analysis (ambience)
            hop_length_ratio: Hop length as ratio of n_fft (0.5 = 50% overlap)
            device: Device for PyTorch computations ('cuda' or 'cpu')
            normalization_type: Type of audio normalization to apply
            target_level_db: Target level in dB for normalization
        """
        self.sample_rate = sample_rate
        self.n_fft_short = n_fft_short
        self.n_fft_long = n_fft_long
        self.hop_short = int(n_fft_short * hop_length_ratio)
        self.hop_long = int(n_fft_long * hop_length_ratio)
        self.device = device

        self.normalizer = AudioNormalizer(
            target_level_db=target_level_db,
            normalization_type=normalization_type,
            sample_rate=sample_rate,
        )

        self._initialize_stft_layers()

    def _initialize_stft_layers(self) -> None:
        """Initialize STFT layers for multi-resolution analysis."""
        self.stft_short = nn_features.STFT(
            n_fft=self.n_fft_short,
            hop_length=self.hop_short,
            window="hann",
            freq_scale="linear",
            center=True,
            pad_mode="reflect",
            fmin=20,
            fmax=self.sample_rate // 2,
            sr=self.sample_rate,
            trainable=False,
            output_format="Complex",
        ).to(self.device)

        self.stft_long = nn_features.STFT(
            n_fft=self.n_fft_long,
            hop_length=self.hop_long,
            window="hann",
            freq_scale="linear",
            center=True,
            pad_mode="reflect",
            fmin=20,
            fmax=self.sample_rate // 2,
            sr=self.sample_rate,
            trainable=False,
            output_format="Complex",
        ).to(self.device)

    def load_audio(self, file_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load stereo audio file and return left/right channels.

        Args:
            file_path: Path to the audio file

        Returns:
            Tuple of (left_channel, right_channel) tensors

        Raises:
            ValueError: If audio is not stereo or has invalid format
        """
        try:
            waveform, original_sr = torchaudio.load(str(file_path))

            if waveform.shape[0] != 2:
                raise ValueError(
                    f"Audio must be stereo, got {waveform.shape[0]} channels"
                )

            if original_sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(
                    orig_freq=original_sr, new_freq=self.sample_rate
                )
                waveform = resampler(waveform)

            waveform = self._normalize_audio(waveform)

            left_channel = waveform[0].to(self.device)
            right_channel = waveform[1].to(self.device)

            return left_channel, right_channel

        except Exception as e:
            raise ValueError(f"Failed to load audio file {file_path}: {str(e)}")

    def _normalize_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize audio using configured normalization method.
        
        Args:
            waveform: Input waveform tensor with shape (channels, samples)
            
        Returns:
            Normalized waveform tensor
        """
        return self.normalizer.normalize_waveform(waveform)

    def compute_stft_pair(
        self, left_channel: torch.Tensor, right_channel: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute multi-resolution STFTs for both channels.

        Args:
            left_channel: Left channel waveform
            right_channel: Right channel waveform

        Returns:
            Tuple of (stft_left_short, stft_right_short, stft_left_long, stft_right_long)
            Each STFT is complex-valued with shape (freq_bins, time_frames)
        """
        with torch.no_grad():
            stft_left_short = self.stft_short(left_channel.unsqueeze(0))[0]
            stft_right_short = self.stft_short(right_channel.unsqueeze(0))[0]

            stft_left_long = self.stft_long(left_channel.unsqueeze(0))[0]
            stft_right_long = self.stft_long(right_channel.unsqueeze(0))[0]

        # Convert from [freq, time, 2] to complex tensors [freq, time]
        stft_left_short = torch.complex(
            stft_left_short[:, :, 0], stft_left_short[:, :, 1]
        )
        stft_right_short = torch.complex(
            stft_right_short[:, :, 0], stft_right_short[:, :, 1]
        )
        stft_left_long = torch.complex(stft_left_long[:, :, 0], stft_left_long[:, :, 1])
        stft_right_long = torch.complex(
            stft_right_long[:, :, 0], stft_right_long[:, :, 1]
        )

        return stft_left_short, stft_right_short, stft_left_long, stft_right_long

    def get_frequency_bins(self, resolution: str = "short") -> np.ndarray:
        """Get frequency bin centers for given resolution.

        Args:
            resolution: Either 'short' or 'long'

        Returns:
            Array of frequency bin centers in Hz
        """
        n_fft = self.n_fft_short if resolution == "short" else self.n_fft_long
        return np.fft.fftfreq(n_fft, 1 / self.sample_rate)[: n_fft // 2 + 1]

    def analyze_audio_levels(self, waveform: torch.Tensor) -> dict:
        """Analyze audio levels for diagnostic purposes.
        
        Args:
            waveform: Input waveform tensor
            
        Returns:
            Dictionary containing level analysis results
        """
        return self.normalizer.analyze_audio_levels(waveform)

    def set_normalization_type(self, normalization_type: NormalizationType) -> None:
        """Change the normalization type.
        
        Args:
            normalization_type: New normalization type to use
        """
        self.normalizer.normalization_type = normalization_type

    def set_target_level(self, target_level_db: float) -> None:
        """Change the target normalization level.
        
        Args:
            target_level_db: New target level in dB
        """
        self.normalizer.target_level_db = target_level_db
