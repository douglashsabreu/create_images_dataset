"""Spatial audio feature extraction module."""

from typing import Dict, Optional, Tuple

import librosa
import numpy as np
import torch
from scipy.fft import fft, ifft
from scipy.signal import correlate, hilbert


class BinauralFeatureExtractor:
    """Extracts binaural and spatial features from stereo STFT data.

    Implements various spatial audio features including ILD, IPD, IACC, MSC,
    and other binaural cues used for spatial audio analysis and classification.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        n_mels: int = 128,
        n_erb: int = 128,
        gcc_tau_max: int = 64,
        pcen_time_constant: float = 0.04,
        pcen_gain: float = 0.8,
    ):
        """Initialize the binaural feature extractor.

        Args:
            sample_rate: Sample rate of the audio
            n_mels: Number of Mel filter banks
            n_erb: Number of ERB filter banks
            gcc_tau_max: Maximum delay for GCC-PHAT in samples
            pcen_time_constant: Time constant for PCEN normalization
            pcen_gain: Gain factor for PCEN normalization
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_erb = n_erb
        self.gcc_tau_max = gcc_tau_max
        self.pcen_time_constant = pcen_time_constant
        self.pcen_gain = pcen_gain

        self._setup_filter_banks()

    def _setup_filter_banks(self) -> None:
        """Setup Mel and ERB filter banks for auditory-scale analysis."""
        pass

    def extract_all_features(
        self,
        stft_left_short: torch.Tensor,
        stft_right_short: torch.Tensor,
        stft_left_long: torch.Tensor,
        stft_right_long: torch.Tensor,
    ) -> Dict[str, np.ndarray]:
        """Extract all spatial features from multi-resolution STFTs.

        Args:
            stft_left_short: Short-time STFT of left channel
            stft_right_short: Short-time STFT of right channel
            stft_left_long: Long-time STFT of left channel
            stft_right_long: Long-time STFT of right channel

        Returns:
            Dictionary containing all extracted spatial features
        """
        features = {}

        for resolution, (L, R) in [
            ("short", (stft_left_short, stft_right_short)),
            ("long", (stft_left_long, stft_right_long)),
        ]:
            suffix = f"_{resolution}"

            features[f"magnitude_L{suffix}"] = self._compute_magnitude_spectrum(L)
            features[f"magnitude_R{suffix}"] = self._compute_magnitude_spectrum(R)

            features[f"ild{suffix}"] = self._compute_ild(L, R)
            features[f"ipd_sin{suffix}"], features[f"ipd_cos{suffix}"] = (
                self._compute_ipd(L, R)
            )

            features[f"iacc{suffix}"] = self._compute_iacc(L, R)
            features[f"msc{suffix}"] = self._compute_msc(L, R)

            features[f"mid{suffix}"], features[f"side{suffix}"] = (
                self._compute_mid_side(L, R)
            )
            features[f"laterality{suffix}"] = self._compute_laterality_index(L, R)

            features[f"spectral_centroid_L{suffix}"] = self._compute_spectral_centroid(
                L
            )
            features[f"spectral_centroid_R{suffix}"] = self._compute_spectral_centroid(
                R
            )
            features[f"spectral_slope_L{suffix}"] = self._compute_spectral_slope(L)
            features[f"spectral_slope_R{suffix}"] = self._compute_spectral_slope(R)

            features[f"late_energy{suffix}"] = self._compute_late_energy_ratio(L, R)

        features["gcc_phat"] = self._compute_gcc_phat(stft_left_short, stft_right_short)
        features["itd_estimate"] = self._extract_itd_from_gcc(features["gcc_phat"])

        return features

    def _compute_magnitude_spectrum(self, stft: torch.Tensor) -> np.ndarray:
        """Compute magnitude spectrum with PCEN compression."""
        magnitude = torch.abs(stft).cpu().numpy()
        return self._apply_pcen(magnitude)

    def _apply_pcen(self, magnitude: np.ndarray) -> np.ndarray:
        """Apply Per-Channel Energy Normalization (PCEN)."""
        epsilon = 1e-8
        smoothed = np.zeros_like(magnitude)
        alpha = np.exp(-1.0 / (self.pcen_time_constant * self.sample_rate))

        for t in range(magnitude.shape[1]):
            if t == 0:
                smoothed[:, t] = magnitude[:, t]
            else:
                smoothed[:, t] = (
                    alpha * smoothed[:, t - 1] + (1 - alpha) * magnitude[:, t]
                )

        pcen = magnitude / (smoothed + epsilon) ** self.pcen_gain
        return np.log(pcen + epsilon)

    def _compute_ild(
        self, stft_left: torch.Tensor, stft_right: torch.Tensor
    ) -> np.ndarray:
        """Compute Interaural Level Difference in dB."""
        magnitude_left = torch.abs(stft_left)
        magnitude_right = torch.abs(stft_right)

        epsilon = 1e-8
        ild = 20 * torch.log10((magnitude_left + epsilon) / (magnitude_right + epsilon))
        return ild.cpu().numpy()

    def _compute_ipd(
        self, stft_left: torch.Tensor, stft_right: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Interaural Phase Difference as sine and cosine components."""
        phase_left = torch.angle(stft_left)
        phase_right = torch.angle(stft_right)

        ipd = phase_left - phase_right

        ipd_sin = torch.sin(ipd).cpu().numpy()
        ipd_cos = torch.cos(ipd).cpu().numpy()

        return ipd_sin, ipd_cos

    def _compute_iacc(
        self, stft_left: torch.Tensor, stft_right: torch.Tensor
    ) -> np.ndarray:
        """Compute Interaural Cross-Correlation per frequency band."""
        L = stft_left.cpu().numpy()
        R = stft_right.cpu().numpy()

        iacc = np.zeros((L.shape[0], L.shape[1]))

        for freq_bin in range(L.shape[0]):
            for frame in range(L.shape[1]):
                if frame < 10 or frame >= L.shape[1] - 10:
                    iacc[freq_bin, frame] = 0
                    continue

                l_segment = L[freq_bin, frame - 10 : frame + 10]
                r_segment = R[freq_bin, frame - 10 : frame + 10]

                # Compute Pearson correlation safely to avoid divide-by-zero warnings
                l_real = np.real(l_segment)
                r_real = np.real(r_segment)

                # Remove means
                l_centered = l_real - np.mean(l_real)
                r_centered = r_real - np.mean(r_real)

                denom = np.sqrt(np.sum(l_centered**2) * np.sum(r_centered**2))
                if denom == 0 or not np.isfinite(denom):
                    iacc[freq_bin, frame] = 0.0
                else:
                    iacc[freq_bin, frame] = float(
                        np.sum(l_centered * r_centered) / denom
                    )

        # Replace any NaNs introduced by numerical issues and clip to valid range
        iacc = np.nan_to_num(iacc, nan=0.0, posinf=0.0, neginf=0.0)
        return np.clip(iacc, -1, 1)

    def _compute_msc(
        self, stft_left: torch.Tensor, stft_right: torch.Tensor
    ) -> np.ndarray:
        """Compute Magnitude-Squared Coherence."""
        L = stft_left.cpu().numpy()
        R = stft_right.cpu().numpy()

        cross_spectrum = L * np.conj(R)
        auto_spectrum_L = L * np.conj(L)
        auto_spectrum_R = R * np.conj(R)

        epsilon = 1e-8
        msc = np.abs(cross_spectrum) ** 2 / (
            (auto_spectrum_L + epsilon) * (auto_spectrum_R + epsilon)
        )

        return np.real(msc)

    def _compute_mid_side(
        self, stft_left: torch.Tensor, stft_right: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Mid and Side signals from L/R STFTs."""
        mid = (stft_left + stft_right) / 2
        side = (stft_left - stft_right) / 2

        mid_magnitude = torch.abs(mid).cpu().numpy()
        side_magnitude = torch.abs(side).cpu().numpy()

        return self._apply_pcen(mid_magnitude), self._apply_pcen(side_magnitude)

    def _compute_laterality_index(
        self, stft_left: torch.Tensor, stft_right: torch.Tensor
    ) -> np.ndarray:
        """Compute laterality index (Side/Mid ratio)."""
        mid = torch.abs((stft_left + stft_right) / 2)
        side = torch.abs((stft_left - stft_right) / 2)

        epsilon = 1e-8
        laterality = side / (mid + epsilon)

        return laterality.cpu().numpy()

    def _compute_spectral_centroid(self, stft: torch.Tensor) -> np.ndarray:
        """Compute spectral centroid for elevation cues."""
        magnitude = torch.abs(stft).cpu().numpy()

        # Ensure we have a 2D array
        if magnitude.ndim == 1:
            magnitude = magnitude[:, np.newaxis]

        freq_bins = np.arange(magnitude.shape[0])

        centroids = []
        for frame in range(magnitude.shape[1]):
            spectrum = magnitude[:, frame]
            if np.sum(spectrum) > 0:
                centroid = np.sum(freq_bins * spectrum) / np.sum(spectrum)
            else:
                centroid = 0
            centroids.append(centroid)

        return np.array(centroids)[np.newaxis, :]

    def _compute_spectral_slope(self, stft: torch.Tensor) -> np.ndarray:
        """Compute spectral slope for elevation cues."""
        magnitude = torch.abs(stft).cpu().numpy()

        # Ensure we have a 2D array
        if magnitude.ndim == 1:
            magnitude = magnitude[:, np.newaxis]

        freq_bins = np.arange(magnitude.shape[0])

        slopes = []
        for frame in range(magnitude.shape[1]):
            spectrum = magnitude[:, frame]
            if np.sum(spectrum) > 0:
                log_spectrum = np.log(spectrum + 1e-8)
                slope = np.polyfit(freq_bins, log_spectrum, 1)[0]
            else:
                slope = 0
            slopes.append(slope)

        return np.array(slopes)[np.newaxis, :]

    def _compute_gcc_phat(
        self, stft_left: torch.Tensor, stft_right: torch.Tensor
    ) -> np.ndarray:
        """Compute GCC-PHAT for ITD estimation."""
        L = stft_left.cpu().numpy()
        R = stft_right.cpu().numpy()

        cross_spectrum = L * np.conj(R)

        magnitude = np.abs(cross_spectrum)
        phase = np.angle(cross_spectrum)

        whitened_cross_spectrum = np.exp(1j * phase)

        gcc_phat = np.fft.ifft(whitened_cross_spectrum, axis=0)
        gcc_phat = np.real(np.fft.fftshift(gcc_phat, axes=0))

        freq_bins = gcc_phat.shape[0]
        center = freq_bins // 2
        tau_range = min(self.gcc_tau_max, center)

        return gcc_phat[center - tau_range : center + tau_range + 1, :]

    def _extract_itd_from_gcc(self, gcc_phat: np.ndarray) -> np.ndarray:
        """Extract ITD estimates from GCC-PHAT."""
        itd_samples = []

        for frame in range(gcc_phat.shape[1]):
            gcc_frame = gcc_phat[:, frame]
            peak_idx = np.argmax(np.abs(gcc_frame))

            tau_offset = peak_idx - len(gcc_frame) // 2
            itd_samples.append(tau_offset)

        return np.array(itd_samples)[np.newaxis, :]

    def _compute_late_energy_ratio(
        self, stft_left: torch.Tensor, stft_right: torch.Tensor
    ) -> np.ndarray:
        """Compute late energy ratio for reverberation estimation."""
        magnitude = (torch.abs(stft_left) + torch.abs(stft_right)) / 2
        magnitude = magnitude.cpu().numpy()

        # Ensure we have a 2D array
        if magnitude.ndim == 1:
            magnitude = magnitude[:, np.newaxis]

        late_ratios = []
        window_size = min(10, magnitude.shape[1] // 4)

        for frame in range(magnitude.shape[1]):
            start_frame = max(0, frame - window_size)
            end_frame = min(magnitude.shape[1], frame + window_size)

            if end_frame > start_frame:
                early_energy = np.sum(
                    magnitude[:, start_frame : start_frame + window_size // 2]
                )
                late_energy = np.sum(
                    magnitude[:, end_frame - window_size // 2 : end_frame]
                )

                total_energy = early_energy + late_energy
                if total_energy > 0:
                    ratio = late_energy / total_energy
                else:
                    ratio = 0
            else:
                ratio = 0

            late_ratios.append(ratio)

        return np.array(late_ratios)[np.newaxis, :]
