"""Audio normalization module for consistent audio level processing."""

from enum import Enum
from typing import Optional, Tuple

import numpy as np
import torch
from scipy import signal


class NormalizationType(Enum):
    """Available normalization methods for audio signals."""

    RMS = "rms"
    PEAK = "peak"
    LUFS = "lufs"
    DYNAMIC_RANGE = "dynamic_range"


class AudioNormalizer:
    """Normalizes audio signals to consistent levels using various methods.

    Provides multiple normalization strategies to ensure consistent audio
    levels before processing, which is critical for machine learning
    applications and signal analysis.
    """

    def __init__(
        self,
        target_level_db: float = -23.0,
        normalization_type: NormalizationType = NormalizationType.RMS,
        sample_rate: int = 48000,
        epsilon: float = 1e-8,
    ):
        """Initialize the audio normalizer.

        Args:
            target_level_db: Target level in dB for normalization
            normalization_type: Type of normalization to apply
            sample_rate: Sample rate of input audio
            epsilon: Small value to prevent division by zero
        """
        self.target_level_db = target_level_db
        self.normalization_type = normalization_type
        self.sample_rate = sample_rate
        self.epsilon = epsilon

    def normalize_waveform(
        self,
        waveform: torch.Tensor,
        normalization_type: Optional[NormalizationType] = None,
    ) -> torch.Tensor:
        """Normalize audio waveform using specified method.

        Args:
            waveform: Input audio waveform tensor with shape (channels, samples)
            normalization_type: Override default normalization type

        Returns:
            Normalized waveform tensor
        """
        norm_type = normalization_type or self.normalization_type

        if norm_type == NormalizationType.RMS:
            return self._normalize_by_rms(waveform)
        elif norm_type == NormalizationType.PEAK:
            return self._normalize_by_peak(waveform)
        elif norm_type == NormalizationType.LUFS:
            return self._normalize_by_lufs(waveform)
        elif norm_type == NormalizationType.DYNAMIC_RANGE:
            return self._normalize_dynamic_range(waveform)
        else:
            raise ValueError(f"Unsupported normalization type: {norm_type}")

    def _normalize_by_rms(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize audio by RMS (Root Mean Square) level.

        Args:
            waveform: Input waveform tensor

        Returns:
            RMS-normalized waveform
        """
        rms = self._compute_rms(waveform)
        target_rms = self._db_to_linear(self.target_level_db)
        gain = target_rms / (rms + self.epsilon)

        return self._apply_safe_gain(waveform, gain)

    def _normalize_by_peak(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize audio by peak level.

        Args:
            waveform: Input waveform tensor

        Returns:
            Peak-normalized waveform
        """
        peak = torch.max(torch.abs(waveform))
        target_peak = self._db_to_linear(self.target_level_db)
        gain = target_peak / (peak + self.epsilon)

        return self._apply_safe_gain(waveform, gain)

    def _normalize_by_lufs(self, waveform: torch.Tensor) -> torch.Tensor:
        """Normalize audio by LUFS (Loudness Units relative to Full Scale).

        Args:
            waveform: Input waveform tensor

        Returns:
            LUFS-normalized waveform
        """
        lufs = self._compute_lufs(waveform)
        gain_db = self.target_level_db - lufs
        gain = self._db_to_linear(gain_db)

        return self._apply_safe_gain(waveform, gain)

    def _normalize_dynamic_range(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply dynamic range normalization with soft compression.

        Args:
            waveform: Input waveform tensor

        Returns:
            Dynamic range normalized waveform
        """
        # Apply RMS normalization first, then apply link-compression based on
        # the mono mix envelope to preserve interaural level differences (ILD)
        rms_normalized = self._normalize_by_rms(waveform)
        compressed = self._apply_linked_soft_compression(rms_normalized)

        return compressed

    def _compute_rms(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute RMS level of waveform.

        Args:
            waveform: Input waveform tensor

        Returns:
            RMS level as tensor
        """
        return torch.sqrt(torch.mean(waveform**2))

    def _compute_lufs(self, waveform: torch.Tensor) -> float:
        """Compute LUFS (approximate) for the waveform.

        Args:
            waveform: Input waveform tensor

        Returns:
            LUFS value in dB
        """
        filtered_audio = self._apply_k_weighting_filter(waveform)
        mean_square = torch.mean(filtered_audio**2, dim=-1)
        lufs = -0.691 + 10 * torch.log10(mean_square + self.epsilon)

        return float(torch.mean(lufs))

    def _apply_k_weighting_filter(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply K-weighting filter for LUFS calculation.

        Args:
            waveform: Input waveform tensor

        Returns:
            K-weighted audio
        """
        b, a = self._get_k_weighting_coefficients()

        filtered = torch.zeros_like(waveform)
        for ch in range(waveform.shape[0]):
            audio_np = waveform[ch].cpu().numpy().copy()
            filtered_np = signal.filtfilt(b, a, audio_np)
            filtered[ch] = torch.from_numpy(filtered_np.copy())

        return filtered

    def _get_k_weighting_coefficients(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get K-weighting filter coefficients.

        Returns:
            Tuple of (numerator, denominator) filter coefficients
        """
        shelf_freq = 1681.0
        shelf_gain = 4.0
        hpf_freq = 38.13

        nyquist = self.sample_rate / 2
        shelf_freq_norm = shelf_freq / nyquist
        hpf_freq_norm = hpf_freq / nyquist

        b_shelf, a_shelf = signal.butter(1, shelf_freq_norm, btype="high")
        b_hpf, a_hpf = signal.butter(2, hpf_freq_norm, btype="high")

        b = np.convolve(b_shelf, b_hpf)
        a = np.convolve(a_shelf, a_hpf)

        return b, a

    def _apply_soft_compression(
        self,
        waveform: torch.Tensor,
        threshold_db: float = -12.0,
        ratio: float = 3.0,
        attack_ms: float = 5.0,
        release_ms: float = 50.0,
    ) -> torch.Tensor:
        """Apply soft dynamic range compression.

        Args:
            waveform: Input waveform tensor
            threshold_db: Compression threshold in dB
            ratio: Compression ratio
            attack_ms: Attack time in milliseconds
            release_ms: Release time in milliseconds

        Returns:
            Compressed waveform
        """
        threshold = self._db_to_linear(threshold_db)

        envelope = self._compute_envelope(waveform, attack_ms, release_ms)

        gain_reduction = torch.ones_like(envelope)
        over_threshold = envelope > threshold

        if torch.any(over_threshold):
            excess_db = self._linear_to_db(envelope[over_threshold] / threshold)
            compressed_excess_db = excess_db / ratio
            gain_reduction[over_threshold] = self._db_to_linear(
                -excess_db + compressed_excess_db
            )

        return waveform * gain_reduction

    def _apply_linked_soft_compression(
        self,
        waveform: torch.Tensor,
        threshold_db: float = -12.0,
        ratio: float = 3.0,
        attack_ms: float = 5.0,
        release_ms: float = 50.0,
    ) -> torch.Tensor:
        """Apply soft compression using a mono-linked envelope.

        This computes the envelope from the mono mix (or max across channels)
        and applies the same sample-wise gain reduction to all channels, which
        preserves ILD and other interaural cues.

        Args:
            waveform: Input waveform tensor (channels, samples)
            threshold_db: Compression threshold in dB
            ratio: Compression ratio
            attack_ms: Attack time in milliseconds
            release_ms: Release time in milliseconds

        Returns:
            Compressed waveform with linked gain applied to all channels
        """
        # Create mono-linked signal by taking RMS across channels per sample
        # to represent overall loudness for gain computation
        mono_mix = torch.sqrt(torch.mean(waveform**2, dim=0))

        # Compute envelope from mono mix
        mono_envelope = self._compute_envelope(
            mono_mix.unsqueeze(0), attack_ms, release_ms
        )
        mono_envelope = mono_envelope.squeeze(0)

        threshold = self._db_to_linear(threshold_db)

        gain_reduction = torch.ones_like(mono_envelope)
        over_threshold = mono_envelope > threshold

        if torch.any(over_threshold):
            excess_db = self._linear_to_db(mono_envelope[over_threshold] / threshold)
            compressed_excess_db = excess_db / ratio
            gain_reduction[over_threshold] = self._db_to_linear(
                -excess_db + compressed_excess_db
            )

        # Expand gain_reduction to all channels and apply
        gain_reduction = gain_reduction.unsqueeze(0).expand_as(waveform)

        return waveform * gain_reduction

    def _compute_envelope(
        self, waveform: torch.Tensor, attack_ms: float, release_ms: float
    ) -> torch.Tensor:
        """Compute amplitude envelope for compression.

        Args:
            waveform: Input waveform tensor
            attack_ms: Attack time in milliseconds
            release_ms: Release time in milliseconds

        Returns:
            Amplitude envelope
        """
        abs_signal = torch.abs(waveform)

        attack_coeff = np.exp(-1.0 / (attack_ms * self.sample_rate / 1000))
        release_coeff = np.exp(-1.0 / (release_ms * self.sample_rate / 1000))

        envelope = torch.zeros_like(abs_signal)

        for ch in range(abs_signal.shape[0]):
            env_ch = envelope[ch]
            abs_ch = abs_signal[ch]

            for i in range(1, len(abs_ch)):
                if abs_ch[i] > env_ch[i - 1]:
                    env_ch[i] = (
                        attack_coeff * env_ch[i - 1] + (1 - attack_coeff) * abs_ch[i]
                    )
                else:
                    env_ch[i] = (
                        release_coeff * env_ch[i - 1] + (1 - release_coeff) * abs_ch[i]
                    )

        return envelope

    def _apply_safe_gain(
        self, waveform: torch.Tensor, gain: torch.Tensor
    ) -> torch.Tensor:
        """Apply gain with clipping protection.

        Args:
            waveform: Input waveform tensor
            gain: Gain factor to apply

        Returns:
            Gain-adjusted waveform with clipping protection
        """
        max_gain = 0.99 / (torch.max(torch.abs(waveform)) + self.epsilon)
        safe_gain = torch.min(gain, max_gain)

        return waveform * safe_gain

    def _db_to_linear(self, db_value: float) -> torch.Tensor:
        """Convert dB value to linear amplitude.

        Args:
            db_value: Value in decibels

        Returns:
            Linear amplitude value
        """
        return torch.tensor(10.0 ** (db_value / 20.0), dtype=torch.float32)

    def _linear_to_db(self, linear_value: torch.Tensor) -> torch.Tensor:
        """Convert linear amplitude to dB.

        Args:
            linear_value: Linear amplitude value

        Returns:
            Value in decibels
        """
        return 20 * torch.log10(linear_value + self.epsilon)

    def analyze_audio_levels(self, waveform: torch.Tensor) -> dict:
        """Analyze various audio levels for diagnostic purposes.

        Args:
            waveform: Input waveform tensor

        Returns:
            Dictionary containing level analysis results
        """
        rms = self._compute_rms(waveform)
        peak = torch.max(torch.abs(waveform))
        lufs = self._compute_lufs(waveform)

        return {
            "rms_db": float(self._linear_to_db(rms)),
            "peak_db": float(self._linear_to_db(peak)),
            "lufs": lufs,
            "crest_factor": float(peak / (rms + self.epsilon)),
            "dynamic_range_db": float(
                self._linear_to_db(peak) - self._linear_to_db(rms)
            ),
        }
