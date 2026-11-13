import numpy as np
import pytest
from conftest import (
    NR, ND, ADC_MAX, ADC_MIN, HEADROOM, EDGE_SIZE,
    CFAR_ALPHA, THRESHOLD_FACTOR, VALID_CELLS_PER_FRAME
)


def run_mini_pipeline(target_amp, clutter_amp, noise_amp, target_r, target_d, clutter_r, clutter_d, seed, fixed_gain=None):
    """Mini helper executing exact frozen radar pipeline with parameter validation."""
    if not 0 <= target_r < NR:
        raise ValueError("TARGET_RANGE_BIN must be between 0 and NR - 1.")
    if not 0 <= clutter_r < NR:
        raise ValueError("CLUTTER_RANGE_BIN must be between 0 and NR - 1.")
    if not -(ND // 2) <= target_d <= (ND - 1) // 2:
        raise ValueError("TARGET_DOPPLER_BIN is outside the valid range.")
    if not -(ND // 2) <= clutter_d <= (ND - 1) // 2:
        raise ValueError("CLUTTER_DOPPLER_BIN is outside the valid range.")

    rng = np.random.RandomState(seed)
    n = np.arange(NR)
    m = np.arange(ND)
    n_grid, m_grid = np.meshgrid(n, m)
    
    t_phase = 2.0 * np.pi * (target_r * n_grid / NR + target_d * m_grid / ND)
    t_sig = target_amp * np.exp(1j * t_phase)
    
    c_phase = 2.0 * np.pi * (clutter_r * n_grid / NR + clutter_d * m_grid / ND)
    c_sig = clutter_amp * np.exp(1j * c_phase)
    
    n_sig = noise_amp * (rng.randn(ND, NR) + 1j * rng.randn(ND, NR))
    
    rx = t_sig + c_sig + n_sig
    
    if fixed_gain is None:
        max_v = max(float(np.max(np.abs(rx.real))), float(np.max(np.abs(rx.imag))))
        scale = (HEADROOM * ADC_MAX / max_v) if max_v > 0 else 1.0
    else:
        scale = fixed_gain
        
    i_samp = np.clip(np.round(rx.real * scale), ADC_MIN, ADC_MAX).astype(np.int16)
    q_samp = np.clip(np.round(rx.imag * scale), ADC_MIN, ADC_MAX).astype(np.int16)
    
    ref_in = (i_samp.astype(np.float64) + 1j * q_samp.astype(np.float64)) / scale
    
    w_r = np.hanning(NR)
    w_d = np.hanning(ND)
    r_fft = np.fft.fft(ref_in * w_r[None, :], axis=1)
    d_fft = np.fft.fft(r_fft * w_d[:, None], axis=0)
    rd_map = np.fft.fftshift(d_fft, axes=0)
    pwr = np.abs(rd_map) ** 2
    
    # Global
    g_thr = THRESHOLD_FACTOR * np.median(pwr)
    g_det = (pwr > g_thr)
    
    # CFAR
    c_det = np.zeros((ND, NR), dtype=bool)
    c_thr = np.zeros((ND, NR))
    for d in range(ND):
        for r in range(EDGE_SIZE, NR - EDGE_SIZE):
            ref = np.concatenate((pwr[d, r-6:r-2], pwr[d, r+3:r+7]))
            loc_t = CFAR_ALPHA * np.mean(ref)
            c_thr[d, r] = loc_t
            c_det[d, r] = (pwr[d, r] > loc_t)
            
    t_d_shift = int(np.round((target_d + ND // 2) % ND))
    
    return {
        "pwr": pwr,
        "g_det": g_det,
        "c_det": c_det,
        "c_thr": c_thr,
        "t_power": float(pwr[t_d_shift, target_r]),
        "c_target_thr": float(c_thr[t_d_shift, target_r]),
        "c_margin_db": float(10.0 * np.log10(pwr[t_d_shift, target_r] / c_thr[t_d_shift, target_r])),
        "target_detected": bool(c_det[t_d_shift, target_r]),
    }


def test_target_present_detection_case():
    """Verify target-present case: weak target At=0.03 at R=120, D=+2.0 is detected with high margin."""
    res = run_mini_pipeline(
        target_amp=0.03, clutter_amp=0.0, noise_amp=0.08,
        target_r=120, target_d=2.0, clutter_r=72, clutter_d=0.0, seed=42
    )
    assert res["target_detected"] is True
    assert res["c_margin_db"] > 15.0  # Typically ~ +19.7 dB


def test_target_absent_noise_only_pfa():
    """Verify target-absent case: noise-only false alarm probability matches ~1.9% empirical expectation."""
    res = run_mini_pipeline(
        target_amp=0.0, clutter_amp=0.0, noise_amp=0.08,
        target_r=70, target_d=2.0, clutter_r=72, clutter_d=0.0, seed=42
    )
    # Target cell should not have detection bias
    valid_c_det = res["c_det"][:, EDGE_SIZE:NR - EDGE_SIZE]
    false_alarms = int(valid_c_det.sum())
    pfa = false_alarms / VALID_CELLS_PER_FRAME
    
    # Empirical Pfa must lie within reasonable statistical bounds for a single frame [1.4% .. 2.5%]
    assert 0.014 <= pfa <= 0.025


def test_parameter_validation_error_cases():
    """Verify that illegal parameter combinations raise expected ValueError exceptions."""
    # 1. Invalid Doppler bin (> ND//2)
    with pytest.raises(ValueError, match="TARGET_DOPPLER_BIN is outside the valid range"):
        run_mini_pipeline(
            target_amp=0.03, clutter_amp=0.0, noise_amp=0.08,
            target_r=70, target_d=20.0, clutter_r=72, clutter_d=0.0, seed=42
        )
            
    # 2. Invalid Range bin (>= NR)
    with pytest.raises(ValueError, match="TARGET_RANGE_BIN must be between 0 and NR - 1"):
        run_mini_pipeline(
            target_amp=0.03, clutter_amp=0.0, noise_amp=0.08,
            target_r=600, target_d=2.0, clutter_r=72, clutter_d=0.0, seed=42
        )
            
    # 3. Invalid Clutter Range bin (>= NR)
    with pytest.raises(ValueError, match="CLUTTER_RANGE_BIN must be between 0 and NR - 1"):
        run_mini_pipeline(
            target_amp=0.03, clutter_amp=0.0, noise_amp=0.08,
            target_r=70, target_d=2.0, clutter_r=800, clutter_d=0.0, seed=42
        )
            
    # 4. Invalid Clutter Doppler bin (< -ND//2)
    with pytest.raises(ValueError, match="CLUTTER_DOPPLER_BIN is outside the valid range"):
        run_mini_pipeline(
            target_amp=0.03, clutter_amp=0.0, noise_amp=0.08,
            target_r=70, target_d=2.0, clutter_r=72, clutter_d=-25.0, seed=42
        )

