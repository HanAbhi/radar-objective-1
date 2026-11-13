import numpy as np
import pytest
from conftest import NR, ND


def test_hann_window_coefficients_and_symmetry():
    """Verify Hann window mathematical properties against independent analytical formula."""
    w_r = np.hanning(NR)
    w_d = np.hanning(ND)
    
    # 1. Exact formula: np.hanning(M) uses 0.5 - 0.5 * cos(2*pi*n / (M-1)) for n = 0..M-1
    expected_w_r = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(NR) / (NR - 1))
    assert np.allclose(w_r, expected_w_r, atol=1e-14)
    
    # 2. Symmetry: w[n] == w[N - 1 - n]
    assert np.allclose(w_r, w_r[::-1], atol=1e-14)
    assert np.allclose(w_d, w_d[::-1], atol=1e-14)
    
    # 3. Peak value: for even N, peak sampled value is 0.5 + 0.5*cos(pi/(N-1))
    expected_peak_d = 0.5 + 0.5 * np.cos(np.pi / (ND - 1))
    assert np.isclose(np.max(w_r), 1.0, atol=1e-4)
    assert np.isclose(np.max(w_d), expected_peak_d, atol=1e-6)


def test_hann_window_coherent_and_noise_gains():
    """Verify 1D and 2D coherent gain, noise gain, and processing gain against analytical sums."""
    w_r = np.hanning(NR)
    w_d = np.hanning(ND)
    
    sum_wr = float(np.sum(w_r))
    sum_sq_wr = float(np.sum(w_r ** 2))
    sum_wd = float(np.sum(w_d))
    sum_sq_wd = float(np.sum(w_d ** 2))
    
    # 1D Range Hann (NR = 512): sum = 255.5, sum_sq = 191.625
    assert np.isclose(sum_wr, 255.5, atol=1e-6)
    assert np.isclose(sum_sq_wr, 191.625, atol=1e-6)
    pg_r = (sum_wr ** 2) / sum_sq_wr
    assert np.isclose(pg_r, 340.67, atol=1e-2)
    
    # 1D Doppler Hann (ND = 32): sum = 15.5, sum_sq = 11.625
    assert np.isclose(sum_wd, 15.5, atol=1e-6)
    assert np.isclose(sum_sq_wd, 11.625, atol=1e-6)
    pg_d = (sum_wd ** 2) / sum_sq_wd
    assert np.isclose(pg_d, 20.67, atol=1e-2)
    
    # 2D Product Hann (512 x 32):
    # Coherent Gain G_c = (sum_wr * sum_wd)^2 = (255.5 * 15.5)^2 = 15,683,580.0625
    gc_2d = (sum_wr * sum_wd) ** 2
    gn_2d = sum_sq_wr * sum_sq_wd
    pg_2d = gc_2d / gn_2d
    
    assert np.isclose(gc_2d, 15683580.0625, atol=1e-4)
    assert np.isclose(gn_2d, 2227.640625, atol=1e-4)
    assert np.isclose(pg_2d, 7040.4444, atol=1e-3)
    
    # In dB: 10*log10(PG) = +38.48 dB
    assert np.isclose(10.0 * np.log10(pg_2d), 38.47599, atol=1e-4)


@pytest.mark.parametrize("range_bin, doppler_bin, expected_dop_shift", [
    (40, 1.0, 17),    # +1.0 -> 16 + 1 = 17
    (70, 2.0, 18),    # +2.0 -> 16 + 2 = 18
    (120, -3.0, 13),  # -3.0 -> 16 - 3 = 13
    (200, 0.0, 16),   # 0.0  -> 16 + 0 = 16 (DC)
    (300, 7.0, 23),   # +7.0 -> 16 + 7 = 23
    (450, -8.0, 8),   # -8.0 -> 16 - 8 = 8
])
def test_range_and_doppler_fft_peak_locations(known_tone_generator, range_bin, doppler_bin, expected_dop_shift):
    """Verify that pure on-grid complex tones peak exactly at (expected_dop_shift, range_bin)."""
    tone = known_tone_generator(range_bin, doppler_bin, amplitude=1.0)
    
    # Process through standard Hann windowing and 2D FFT
    w_r = np.hanning(NR)
    w_d = np.hanning(ND)
    
    r_fft = np.fft.fft(tone * w_r[None, :], axis=1)
    d_fft = np.fft.fft(r_fft * w_d[:, None], axis=0)
    rd_map = np.fft.fftshift(d_fft, axes=0)
    power_map = np.abs(rd_map) ** 2
    
    # Find peak coordinates
    peak_idx = np.unravel_index(np.argmax(power_map), power_map.shape)
    peak_dop, peak_rng = peak_idx
    
    assert peak_rng == range_bin
    assert peak_dop == expected_dop_shift


def test_signed_doppler_indexing_and_fftshift():
    """Verify fftshift maps signed Doppler indices [-ND//2 .. ND//2 - 1] to [0 .. ND-1]."""
    # Create impulse in frequency domain at known unshifted and shifted bins
    ND_half = ND // 2  # 16
    
    # DC (Doppler bin 0) in standard FFT is at index 0. After fftshift, it must be at index 16.
    dc_vector = np.zeros(ND)
    dc_vector[0] = 1.0
    shifted_dc = np.fft.fftshift(dc_vector)
    assert np.argmax(shifted_dc) == ND_half
    
    # +1 bin in standard FFT is at index 1. After fftshift, it must be at index 17.
    pos1_vector = np.zeros(ND)
    pos1_vector[1] = 1.0
    shifted_pos1 = np.fft.fftshift(pos1_vector)
    assert np.argmax(shifted_pos1) == ND_half + 1
    
    # -1 bin in standard FFT is at index ND-1 (31). After fftshift, it must be at index 15.
    neg1_vector = np.zeros(ND)
    neg1_vector[ND - 1] = 1.0
    shifted_neg1 = np.fft.fftshift(neg1_vector)
    assert np.argmax(shifted_neg1) == ND_half - 1
