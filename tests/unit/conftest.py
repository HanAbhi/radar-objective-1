import pytest
import numpy as np

# ---------------------------------------------------------
# FROZEN OBJECTIVE I SPECIFICATIONS & CONSTANTS
# ---------------------------------------------------------

NR = 512
ND = 32
ADC_BITS = 12
ADC_MAX = 2 ** (ADC_BITS - 1) - 1   # +2047
ADC_MIN = -(2 ** (ADC_BITS - 1))     # -2048
HEADROOM = 0.90

REFERENCE_CELLS_PER_SIDE = 4
GUARD_CELLS_PER_SIDE = 2
EDGE_SIZE = REFERENCE_CELLS_PER_SIDE + GUARD_CELLS_PER_SIDE  # 6
VALID_CELLS_PER_FRAME = ND * (NR - 2 * EDGE_SIZE)  # 16,000 cells

CFAR_ALPHA = 6.0
THRESHOLD_FACTOR = 6.0


@pytest.fixture
def radar_constants():
    return {
        "NR": NR,
        "ND": ND,
        "ADC_BITS": ADC_BITS,
        "ADC_MAX": ADC_MAX,
        "ADC_MIN": ADC_MIN,
        "HEADROOM": HEADROOM,
        "REFERENCE_CELLS_PER_SIDE": REFERENCE_CELLS_PER_SIDE,
        "GUARD_CELLS_PER_SIDE": GUARD_CELLS_PER_SIDE,
        "EDGE_SIZE": EDGE_SIZE,
        "VALID_CELLS_PER_FRAME": VALID_CELLS_PER_FRAME,
        "CFAR_ALPHA": CFAR_ALPHA,
        "THRESHOLD_FACTOR": THRESHOLD_FACTOR,
    }


@pytest.fixture
def sample_grids():
    n = np.arange(NR)
    m = np.arange(ND)
    n_grid, m_grid = np.meshgrid(n, m)
    return n_grid, m_grid


@pytest.fixture
def known_tone_generator(sample_grids):
    n_grid, m_grid = sample_grids
    def _generator(range_bin: float, doppler_bin: float, amplitude: float = 1.0):
        phase = 2.0 * np.pi * (range_bin * n_grid / NR + doppler_bin * m_grid / ND)
        return amplitude * np.exp(1j * phase)
    return _generator

