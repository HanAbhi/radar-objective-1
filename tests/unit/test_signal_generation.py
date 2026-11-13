import numpy as np
import pytest
from conftest import NR, ND, ADC_MAX, ADC_MIN, HEADROOM


def test_complex_phase_independent_analytical(sample_grids):
    """Verify complex phase formula phi = 2*pi*(r*n/NR + d*m/ND) independently."""
    n_grid, m_grid = sample_grids
    r_target = 70.0
    d_target = 2.0
    
    # Independent calculation for known test indices
    # At chirp m=0, sample n=0: phase must be exactly 0
    phase_0_0 = 2.0 * np.pi * (r_target * 0 / NR + d_target * 0 / ND)
    assert phase_0_0 == 0.0
    
    # At chirp m=ND/4 (m=8), sample n=0: phase = 2*pi*(2*8/32) = 2*pi*(0.5) = pi
    phase_8_0 = 2.0 * np.pi * (r_target * 0 / NR + d_target * 8 / ND)
    assert np.isclose(phase_8_0, np.pi)
    
    # Full grid calculation
    phase_grid = 2.0 * np.pi * (r_target * n_grid / NR + d_target * m_grid / ND)
    sig = np.exp(1j * phase_grid)
    
    # Magnitude of pure tone must be identically 1.0 everywhere
    mag = np.abs(sig)
    assert np.allclose(mag, 1.0, atol=1e-12)
    
    # Specific known complex values
    assert np.isclose(sig[0, 0], 1.0 + 0j)
    assert np.isclose(sig[8, 0], -1.0 + 0j, atol=1e-12)


def test_deterministic_seed_reproducibility():
    """Verify that identical random seeds produce bit-exact identical noise arrays."""
    seed = 42
    rng1 = np.random.RandomState(seed)
    noise1 = 0.08 * (rng1.randn(ND, NR) + 1j * rng1.randn(ND, NR))
    
    rng2 = np.random.RandomState(seed)
    noise2 = 0.08 * (rng2.randn(ND, NR) + 1j * rng2.randn(ND, NR))
    
    assert np.array_equal(noise1, noise2)
    assert np.array_equal(noise1.real, noise2.real)
    assert np.array_equal(noise1.imag, noise2.imag)


def test_different_seeds_produce_different_realizations():
    """Verify that different random seeds produce statistically independent noise."""
    rng1 = np.random.RandomState(1)
    noise1 = rng1.randn(ND, NR)
    rng2 = np.random.RandomState(2)
    noise2 = rng2.randn(ND, NR)
    
    # Must not be equal
    assert not np.array_equal(noise1, noise2)
    # Correlation between independent Gaussian realization must be near 0
    corr = np.corrcoef(noise1.ravel(), noise2.ravel())[0, 1]
    assert abs(corr) < 0.05


def test_input_iq_ordering_and_dimensions(sample_grids):
    """Verify that input I/Q formatting uses chirp-major ordering and exactly 16,384 samples."""
    n_grid, m_grid = sample_grids
    # Synthesize arbitrary recognizable signal
    i_test = (m_grid * 1000 + n_grid).astype(np.int16)
    q_test = (-i_test).astype(np.int16)
    
    iq_data = np.column_stack((i_test.ravel(), q_test.ravel()))
    
    # Total samples must be exactly NR * ND = 16,384
    assert iq_data.shape == (NR * ND, 2)
    assert len(iq_data) == 16384
    
    # Chirp-major ordering verification:
    # First NR samples (0 to 511) must belong to chirp m=0
    for n in range(NR):
        assert iq_data[n, 0] == n  # m=0 -> 0*1000 + n = n
        assert iq_data[n, 1] == -n
        
    # Samples 512 to 1023 must belong to chirp m=1
    for n in range(NR):
        assert iq_data[NR + n, 0] == 1000 + n  # m=1 -> 1*1000 + n
        assert iq_data[NR + n, 1] == -(1000 + n)


def test_zero_signal_and_zero_noise_edge_cases(sample_grids):
    """Verify system behavior under pure zero signal (A_t=0, A_c=0, A_n=0) and pure zero noise."""
    n_grid, m_grid = sample_grids
    
    # Case A: Complete silence (Zero signal + zero noise)
    zero_signal = np.zeros((ND, NR), dtype=complex)
    max_val = max(float(np.max(np.abs(zero_signal.real))), float(np.max(np.abs(zero_signal.imag))))
    assert max_val == 0.0
    
    # Scaling factor must fallback to 1.0 when max_val == 0 to avoid division by zero
    scale = (HEADROOM * ADC_MAX / max_val) if max_val > 0 else 1.0
    assert scale == 1.0
    
    i_samp = np.clip(np.round(zero_signal.real * scale), ADC_MIN, ADC_MAX).astype(np.int16)
    q_samp = np.clip(np.round(zero_signal.imag * scale), ADC_MIN, ADC_MAX).astype(np.int16)
    assert np.all(i_samp == 0)
    assert np.all(q_samp == 0)
    
    # Case B: Zero noise with single tone (A_n = 0)
    tone = 0.50 * np.exp(1j * 2 * np.pi * (70 * n_grid / NR + 2 * m_grid / ND))
    max_tone = max(float(np.max(np.abs(tone.real))), float(np.max(np.abs(tone.imag))))
    assert max_tone > 0.0
    scale_tone = HEADROOM * ADC_MAX / max_tone
    i_tone = np.clip(np.round(tone.real * scale_tone), ADC_MIN, ADC_MAX).astype(np.int16)
    
    # Peak must not exceed 0.90 * ADC_MAX = 1842
    assert np.max(np.abs(i_tone)) <= int(np.ceil(0.90 * ADC_MAX))

