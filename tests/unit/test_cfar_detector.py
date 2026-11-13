import numpy as np
import pytest
from conftest import (
    NR, ND, EDGE_SIZE, REFERENCE_CELLS_PER_SIDE, GUARD_CELLS_PER_SIDE,
    VALID_CELLS_PER_FRAME, CFAR_ALPHA, THRESHOLD_FACTOR
)


def test_cfar_guard_and_reference_cell_indices():
    """Verify that guard cells are r +/- [1, 2] and reference cells are r +/- [3, 4, 5, 6]."""
    test_r = 100
    
    # Left reference and guard
    l_start = test_r - GUARD_CELLS_PER_SIDE - REFERENCE_CELLS_PER_SIDE  # 100 - 2 - 4 = 94
    l_stop = test_r - GUARD_CELLS_PER_SIDE                              # 100 - 2 = 98
    left_ref_indices = list(range(l_start, l_stop))
    assert left_ref_indices == [94, 95, 96, 97]  # Exactly test_r - 6, -5, -4, -3
    
    left_guard_indices = list(range(l_stop, test_r))
    assert left_guard_indices == [98, 99]        # Exactly test_r - 2, -1
    
    # Right reference and guard
    r_start = test_r + GUARD_CELLS_PER_SIDE + 1                          # 100 + 2 + 1 = 103
    r_stop = r_start + REFERENCE_CELLS_PER_SIDE                         # 103 + 4 = 107
    right_guard_indices = list(range(test_r + 1, r_start))
    assert right_guard_indices == [101, 102]     # Exactly test_r + 1, +2
    
    right_ref_indices = list(range(r_start, r_stop))
    assert right_ref_indices == [103, 104, 105, 106]  # Exactly test_r + 3, +4, +5, +6
    
    total_ref_cells = len(left_ref_indices) + len(right_ref_indices)
    assert total_ref_cells == 8


def test_cfar_valid_boundary_bins():
    """Verify that exactly bins [0..5] and [506..511] are invalid, yielding 16,000 valid cells."""
    valid_mask = np.zeros((ND, NR), dtype=bool)
    valid_mask[:, EDGE_SIZE:NR - EDGE_SIZE] = True
    
    # Exactly 16,000 valid cells
    assert valid_mask.sum() == VALID_CELLS_PER_FRAME
    assert valid_mask.sum() == 16000
    
    # Edge size must be exactly 6
    assert EDGE_SIZE == 6
    
    # Bins 0 through 5 must be invalid
    assert not valid_mask[:, 0:6].any()
    # Bin 6 must be valid
    assert valid_mask[:, 6].all()
    # Bin 505 must be valid
    assert valid_mask[:, 505].all()
    # Bins 506 through 511 must be invalid
    assert not valid_mask[:, 506:512].any()


def test_cfar_known_answer_test_vector():
    """Verify CA-CFAR local noise estimate and threshold on an independent synthetic test vector."""
    # Create a 1D range slice with known reference cell values
    # Left reference cells: [10, 20, 30, 40], sum = 100
    # Right reference cells: [50, 60, 70, 80], sum = 260
    # Guard cells: arbitrarily set to 999 (must be ignored!)
    # CUT: value = 250
    # Total reference sum = 360, N_ref = 8 -> local_noise = 360 / 8 = 45.0
    # Threshold = CFAR_ALPHA * 45.0 = 6.0 * 45.0 = 270.0
    # Since CUT (250) < Threshold (270), detection must be False.
    r_slice = np.zeros(NR)
    cut_r = 50
    
    # Set left reference cells: cut_r - [6, 5, 4, 3] = [44, 45, 46, 47]
    r_slice[44:48] = [10.0, 20.0, 30.0, 40.0]
    # Set left guard cells: cut_r - [2, 1] = [48, 49]
    r_slice[48:50] = [999.0, 999.0]
    
    # Set CUT
    r_slice[cut_r] = 250.0
    
    # Set right guard cells: cut_r + [1, 2] = [51, 52]
    r_slice[51:53] = [999.0, 999.0]
    # Set right reference cells: cut_r + [3, 4, 5, 6] = [53, 54, 55, 56]
    r_slice[53:57] = [50.0, 60.0, 70.0, 80.0]
    
    # Extract reference cells using exact detector indices
    l_start = cut_r - GUARD_CELLS_PER_SIDE - REFERENCE_CELLS_PER_SIDE
    l_stop = cut_r - GUARD_CELLS_PER_SIDE
    r_start = cut_r + GUARD_CELLS_PER_SIDE + 1
    r_stop = r_start + REFERENCE_CELLS_PER_SIDE
    
    ref = np.concatenate((r_slice[l_start:l_stop], r_slice[r_start:r_stop]))
    assert len(ref) == 8
    
    loc_noise = float(np.mean(ref))
    assert loc_noise == 45.0
    
    thr = CFAR_ALPHA * loc_noise
    assert thr == 270.0
    
    detected = bool(r_slice[cut_r] > thr)
    assert not detected  # 250.0 < 270.0 -> False
    
    # Now increase CUT to 300.0 (> 270.0), detection must be True
    r_slice[cut_r] = 300.0
    detected_strong = bool(r_slice[cut_r] > thr)
    assert detected_strong


def test_target_masking_by_interfering_clutter_in_reference_cells():
    """Verify that a strong interfering peak in reference cells inflates the threshold and suppresses target."""
    r_slice = np.full(NR, 20.0)  # Uniform noise floor = 20.0 (thr = 6.0 * 20 = 120.0)
    cut_r = 70
    r_slice[cut_r] = 200.0       # Target power = 200.0 (in clean noise, 200 > 120 -> detected!)
    
    # In clean noise:
    ref_clean = np.concatenate((r_slice[cut_r-6:cut_r-2], r_slice[cut_r+3:cut_r+7]))
    thr_clean = 6.0 * np.mean(ref_clean)
    assert thr_clean == 120.0
    assert r_slice[cut_r] > thr_clean  # Detected
    
    # Now introduce strong clutter peak at r = 73 (which is in right reference cells: 70 + 3 = 73)
    # Clutter peak = 10,000.0
    r_slice[73] = 10000.0
    ref_masked = np.concatenate((r_slice[cut_r-6:cut_r-2], r_slice[cut_r+3:cut_r+7]))
    thr_masked = 6.0 * np.mean(ref_masked)
    
    # Noise estimate = (7 * 20 + 10000) / 8 = 10140 / 8 = 1267.5
    # Threshold = 6.0 * 1267.5 = 7605.0
    assert thr_masked == 7605.0
    assert r_slice[cut_r] < thr_masked  # Target is masked (200.0 < 7605.0)


def test_zero_noise_median_collapse():
    """Verify zero-noise median collapse: when noise is zero, global median collapses to 0."""
    # When signal contains only an isolated target tone with zero noise
    power_map = np.zeros((ND, NR))
    power_map[18, 70] = 50000.0  # Single isolated peak
    
    # Median across all 16,384 cells
    median_val = np.median(power_map)
    assert median_val == 0.0
    
    # Fixed threshold collapses to 0
    fixed_thr = THRESHOLD_FACTOR * median_val
    assert fixed_thr == 0.0

