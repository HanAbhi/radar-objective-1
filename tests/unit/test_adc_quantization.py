import numpy as np
import pytest
from conftest import ADC_BITS, ADC_MAX, ADC_MIN, HEADROOM


def test_12bit_adc_bounds_and_saturation():
    """Verify signed 12-bit two's-complement range [-2048, +2047] and hard clipping."""
    assert ADC_MAX == 2047
    assert ADC_MIN == -2048
    
    # Test vector with extreme values exceeding 12-bit range
    test_inputs = np.array([-5000.0, -2049.0, -2048.0, 0.0, 2047.0, 2048.0, 10000.0])
    
    quantized = np.clip(np.round(test_inputs), ADC_MIN, ADC_MAX).astype(np.int16)
    
    expected = np.array([-2048, -2048, -2048, 0, 2047, 2047, 2047], dtype=np.int16)
    assert np.array_equal(quantized, expected)
    
    # Ensure all values strictly lie within [-2048, 2047]
    assert np.all(quantized >= ADC_MIN)
    assert np.all(quantized <= ADC_MAX)


def test_adc_rounding_known_vector():
    """Verify IEEE 754 round-to-nearest-even quantisation on floating-point test inputs."""
    floats = np.array([-1.6, -1.5, -1.4, -0.6, -0.5, -0.4, 0.0, 0.4, 0.5, 0.6, 1.4, 1.5, 1.6, 2.5])
    rounded = np.round(floats)
    
    # Standard round-half-to-even: -0.5 -> 0, 0.5 -> 0, 1.5 -> 2, 2.5 -> 2
    expected = np.array([-2.0, -2.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    assert np.array_equal(rounded, expected)


def test_dynamic_scaling_headroom():
    """Verify dynamic headroom scaling factor: 0.90 * ADC_MAX / max_val."""
    # Test signal with peak amplitude 1.0
    sig_peak_1 = np.array([0.5 + 1.0j, -0.8 - 0.2j])
    max_val = max(float(np.max(np.abs(sig_peak_1.real))), float(np.max(np.abs(sig_peak_1.imag))))
    assert max_val == 1.0
    
    scale = HEADROOM * ADC_MAX / max_val
    assert np.isclose(scale, 0.90 * 2047.0)  # 1842.3
    
    scaled = np.round(sig_peak_1.real * scale)
    assert np.max(np.abs(scaled)) <= int(np.ceil(0.90 * ADC_MAX))


def test_fixed_gain_vs_dynamic_gain_metrics():
    """Verify fixed-gain vs dynamic-gain clipping percentage and SQNR calculations."""
    np.random.seed(42)
    # Generate test complex signal with known amplitude
    t = np.linspace(0, 1, 1000, endpoint=False)
    # Sinusoid with peak 1.5 V
    sig = 1.5 * np.exp(1j * 2 * np.pi * 10 * t)
    
    # 1. Dynamic scaling (should never clip)
    max_v = max(float(np.max(np.abs(sig.real))), float(np.max(np.abs(sig.imag))))
    dyn_scale = 0.90 * ADC_MAX / max_v
    i_dyn_unclipped = np.round(sig.real * dyn_scale)
    i_dyn = np.clip(i_dyn_unclipped, ADC_MIN, ADC_MAX)
    
    dyn_clipped = np.sum((i_dyn_unclipped < ADC_MIN) | (i_dyn_unclipped > ADC_MAX))
    assert dyn_clipped == 0
    
    # 2. Fixed gain that causes deliberate 20-30% clipping
    # Since peak is 1.5, choosing gain G = 2500 means peak reaches 3750 >> 2047
    fix_scale = 2500.0
    i_fix_unclipped = np.round(sig.real * fix_scale)
    i_fix = np.clip(i_fix_unclipped, ADC_MIN, ADC_MAX)
    
    fix_clipped = np.sum((i_fix_unclipped < ADC_MIN) | (i_fix_unclipped > ADC_MAX))
    clip_pct = 100.0 * fix_clipped / len(sig)
    assert clip_pct > 20.0  # Must register substantial clipping
    
    # 3. SQNR comparison: dynamic mode must have significantly higher SQNR than clipped fixed mode
    # Error for dynamic:
    e_dyn = (i_dyn / dyn_scale) - sig.real
    pwr_sig = np.mean(sig.real ** 2)
    pwr_e_dyn = np.mean(e_dyn ** 2)
    sqnr_dyn = 10.0 * np.log10(pwr_sig / pwr_e_dyn)
    
    # Error for clipped fixed:
    e_fix = (i_fix / fix_scale) - sig.real
    pwr_e_fix = np.mean(e_fix ** 2)
    sqnr_fix = 10.0 * np.log10(pwr_sig / pwr_e_fix)
    
    # Dynamic SQNR should be > 60 dB, while clipped fixed SQNR collapses below 25 dB
    assert sqnr_dyn > 60.0
    assert sqnr_fix < 25.0
    assert sqnr_dyn - sqnr_fix > 35.0

