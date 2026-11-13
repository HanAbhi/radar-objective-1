import csv
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------
# CONSTANTS & SPECIFICATIONS (OBJECTIVE I FROZEN BASELINE)
# ---------------------------------------------------------

NR = 512
ND = 32

ADC_BITS = 12
ADC_MAX = 2 ** (ADC_BITS - 1) - 1   # +2047
ADC_MIN = -(2 ** (ADC_BITS - 1))     # -2048

REFERENCE_CELLS_PER_SIDE = 4
GUARD_CELLS_PER_SIDE = 2
EDGE_SIZE = REFERENCE_CELLS_PER_SIDE + GUARD_CELLS_PER_SIDE  # 6
CFAR_ALPHA = 6.0
THRESHOLD_FACTOR = 6.0

RANGE_WINDOW = np.hanning(NR)
DOPPLER_WINDOW = np.hanning(ND)

n_idx = np.arange(NR)
m_idx = np.arange(ND)
n_grid, m_grid = np.meshgrid(n_idx, m_idx)

# Test Configurations
TARGET_AMPLITUDE = 0.03
TARGET_RANGE_BIN = 120   # Distant from clutter center to isolate ADC/quantization from proximity masking
TARGET_DOPPLER_BIN = 2.0
CLUTTER_RANGE_BIN = 72
CLUTTER_DOPPLER_BIN = 0.0
NOISE_AMPLITUDE = 0.08

NOMINAL_FIXED_GAIN = 2200.0
GAIN_SWEEP = [500.0, 1000.0, 1500.0, 2047.0, 2500.0, 3500.0, 5000.0, 7500.0, 10000.0]
CLUTTER_SWEEP = [0.0, 0.10, 0.25, 0.50, 0.60, 0.80, 1.00, 1.25, 1.50, 2.00, 3.00]
N_SEEDS = 20
SEED_START = 6001

CFAR_VALID_MASK = np.zeros((ND, NR), dtype=bool)
CFAR_VALID_MASK[:, EDGE_SIZE:NR - EDGE_SIZE] = True


def quantize_and_process(
    received_signal: np.ndarray,
    scale_factor: float,
    mode_name: str,
    target_dop_shift: int,
    target_r_bin: int,
    clutter_r_bin: int,
) -> dict:
    """Quantize to 12-bit signed I/Q using given scale, process RD map, and evaluate detections."""
    # 1. Quantization & Saturation
    i_unclipped = np.round(received_signal.real * scale_factor)
    q_unclipped = np.round(received_signal.imag * scale_factor)

    i_samples = np.clip(i_unclipped, ADC_MIN, ADC_MAX).astype(np.int16)
    q_samples = np.clip(q_unclipped, ADC_MIN, ADC_MAX).astype(np.int16)

    # 2. Clipping Statistics
    i_clipped = (i_unclipped < ADC_MIN) | (i_unclipped > ADC_MAX)
    q_clipped = (q_unclipped < ADC_MIN) | (q_unclipped > ADC_MAX)
    total_components = 2 * ND * NR
    clipped_components = int(i_clipped.sum() + q_clipped.sum())
    clipping_pct = float(100.0 * clipped_components / total_components)

    # 3. Quantization Error & SQNR
    i_recon = i_samples.astype(np.float64) / scale_factor
    q_recon = q_samples.astype(np.float64) / scale_factor
    e_i = i_recon - received_signal.real
    e_q = q_recon - received_signal.imag

    signal_pwr = float(np.mean(received_signal.real ** 2 + received_signal.imag ** 2))
    error_pwr = float(np.mean(e_i ** 2 + e_q ** 2))
    sqnr_db = float(10.0 * np.log10(signal_pwr / error_pwr)) if error_pwr > 0 else 99.0

    # 4. Range-Doppler Processing
    ref_input = i_recon + 1j * q_recon
    win_sig = ref_input * RANGE_WINDOW[None, :]
    r_fft = np.fft.fft(win_sig, axis=1)

    r_fft_win = r_fft * DOPPLER_WINDOW[:, None]
    d_fft = np.fft.fft(r_fft_win, axis=0)
    rd_map = np.fft.fftshift(d_fft, axes=0)
    power_map = np.abs(rd_map) ** 2

    # 5. Global Threshold Detector
    est_noise = float(np.median(power_map))
    fixed_thr = float(THRESHOLD_FACTOR * est_noise)
    fixed_det = (power_map > fixed_thr)

    # 6. CA-CFAR Detector
    cfar_det = np.zeros((ND, NR), dtype=bool)
    cfar_thr_map = np.full((ND, NR), np.nan)
    for d in range(ND):
        for r in range(EDGE_SIZE, NR - EDGE_SIZE):
            l_start = r - GUARD_CELLS_PER_SIDE - REFERENCE_CELLS_PER_SIDE
            l_stop = r - GUARD_CELLS_PER_SIDE
            r_start = r + GUARD_CELLS_PER_SIDE + 1
            r_stop = r_start + REFERENCE_CELLS_PER_SIDE
            ref_cells = np.concatenate((power_map[d, l_start:l_stop], power_map[d, r_start:r_stop]))
            loc_noise = float(np.mean(ref_cells))
            loc_thr = float(CFAR_ALPHA * loc_noise)
            cfar_thr_map[d, r] = loc_thr
            cfar_det[d, r] = (power_map[d, r] > loc_thr)

    # 7. Detection Metrics at Target Cell
    t_power = float(power_map[target_dop_shift, target_r_bin])
    cfar_target_thr = float(cfar_thr_map[target_dop_shift, target_r_bin])

    global_margin_db = float(10.0 * np.log10(t_power / fixed_thr))
    cfar_margin_db = float(10.0 * np.log10(t_power / cfar_target_thr))

    global_pd = int(fixed_det[target_dop_shift, target_r_bin])
    cfar_pd = int(cfar_det[target_dop_shift, target_r_bin])

    # 8. False Alarm Calculation (masking target 3x3 and clutter columns)
    excl_mask = np.zeros((ND, NR), dtype=bool)
    excl_mask[target_dop_shift - 1:target_dop_shift + 2, target_r_bin - 1:target_r_bin + 2] = True
    excl_mask[12:23, clutter_r_bin - 1:clutter_r_bin + 2] = True
    eval_mask = CFAR_VALID_MASK & ~excl_mask
    n_eval = int(eval_mask.sum())

    cfar_false = int((cfar_det & eval_mask).sum())
    cfar_pfa = float(cfar_false / n_eval)

    return {
        "mode": mode_name,
        "scale_factor": scale_factor,
        "clipped_components": clipped_components,
        "clipping_pct": clipping_pct,
        "signal_power": signal_pwr,
        "error_power": error_pwr,
        "sqnr_db": sqnr_db,
        "target_power": t_power,
        "global_threshold": fixed_thr,
        "cfar_target_threshold": cfar_target_thr,
        "global_margin_db": global_margin_db,
        "cfar_margin_db": cfar_margin_db,
        "global_pd": global_pd,
        "cfar_pd": cfar_pd,
        "cfar_pfa": cfar_pfa,
        "power_map": power_map,
    }


def main():
    base_dir = Path(__file__).resolve().parent
    print("=" * 75)
    print("EXTENDED VALIDATION 3: FIXED-GAIN ADC & QUANTISATION ROBUSTNESS")
    print("=" * 75)

    target_dop_shift = int(np.round((TARGET_DOPPLER_BIN + ND // 2) % ND))  # 18
    clutter_dop_shift = int(np.round((CLUTTER_DOPPLER_BIN + ND // 2) % ND))  # 16

    # Target Phase Matrix
    t_phase = 2.0 * np.pi * (TARGET_RANGE_BIN * n_grid / NR + TARGET_DOPPLER_BIN * m_grid / ND)
    target_sig = TARGET_AMPLITUDE * np.exp(1j * t_phase)

    # ---------------------------------------------------------
    # PART 1: FIXED INPUT GAIN SWEEP (Nominal Clutter Ac = 0.60)
    # ---------------------------------------------------------
    print(f"\n[Part 1] Fixed Input Gain Sweep ({len(GAIN_SWEEP)} gain settings, Ac=0.60, {N_SEEDS} seeds)...")
    gain_sweep_rows = []

    c_phase = 2.0 * np.pi * (CLUTTER_RANGE_BIN * n_grid / NR + CLUTTER_DOPPLER_BIN * m_grid / ND)
    clutter_sig_nom = 0.60 * np.exp(1j * c_phase)

    for gain in GAIN_SWEEP:
        gain_results = []
        for s_idx in range(N_SEEDS):
            seed = SEED_START + s_idx
            rng = np.random.RandomState(seed)
            noise_sig = NOISE_AMPLITUDE * (rng.randn(ND, NR) + 1j * rng.randn(ND, NR))
            rx_sig = target_sig + clutter_sig_nom + noise_sig

            res_fixed = quantize_and_process(
                rx_sig, gain, "fixed", target_dop_shift, TARGET_RANGE_BIN, CLUTTER_RANGE_BIN
            )
            # Remove bulky 2D map from rows
            res_fixed.pop("power_map", None)
            res_fixed["gain"] = gain
            res_fixed["seed"] = seed
            gain_results.append(res_fixed)

        # Average across seeds
        avg_clip = float(np.mean([r["clipping_pct"] for r in gain_results]))
        avg_sqnr = float(np.mean([r["sqnr_db"] for r in gain_results]))
        avg_margin = float(np.mean([r["cfar_margin_db"] for r in gain_results]))
        avg_pd = float(np.mean([r["cfar_pd"] for r in gain_results]))
        avg_pfa = float(np.mean([r["cfar_pfa"] for r in gain_results]))
        avg_thr = float(np.mean([r["cfar_target_threshold"] for r in gain_results]))

        gain_sweep_rows.append({
            "gain": gain,
            "clipping_pct_mean": avg_clip,
            "sqnr_db_mean": avg_sqnr,
            "cfar_pd": avg_pd,
            "cfar_margin_db_mean": avg_margin,
            "cfar_target_threshold_mean": avg_thr,
            "cfar_pfa_mean": avg_pfa,
        })
        print(f"  Gain={gain:7.1f} | Clip={avg_clip:6.2f}% | SQNR={avg_sqnr:6.2f} dB | Margin={avg_margin:+6.2f} dB | Pd={avg_pd:4.2f}")

    gain_csv_path = base_dir / "adc_gain_sweep_results.csv"
    with open(gain_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(gain_sweep_rows[0].keys()))
        w.writeheader()
        w.writerows(gain_sweep_rows)
    print(f"Saved: {gain_csv_path}")

    # ---------------------------------------------------------
    # PART 2: CLUTTER AMPLITUDE SWEEP (Fixed Gain 2200 vs Dynamic Scaling)
    # ---------------------------------------------------------
    print(f"\n[Part 2] Clutter Amplitude Sweep ({len(CLUTTER_SWEEP)} clutter levels, {N_SEEDS} seeds per level)...")
    clutter_comp_rows = []
    rd_maps_to_save = {}

    for c_amp in CLUTTER_SWEEP:
        if c_amp > 0:
            clut_sig = c_amp * np.exp(1j * c_phase)
        else:
            clut_sig = np.zeros((ND, NR), dtype=complex)

        dyn_results = []
        fix_results = []

        for s_idx in range(N_SEEDS):
            seed = SEED_START + 100 + s_idx
            rng = np.random.RandomState(seed)
            noise_sig = NOISE_AMPLITUDE * (rng.randn(ND, NR) + 1j * rng.randn(ND, NR))
            rx_sig = target_sig + clut_sig + noise_sig

            # 1. Dynamic Mode
            max_v = max(float(np.max(np.abs(rx_sig.real))), float(np.max(np.abs(rx_sig.imag))))
            dyn_scale = (0.90 * ADC_MAX / max_v) if max_v > 0 else 1.0
            r_dyn = quantize_and_process(
                rx_sig, dyn_scale, "dynamic", target_dop_shift, TARGET_RANGE_BIN, CLUTTER_RANGE_BIN
            )

            # 2. Fixed Gain Mode (Nominal Gain = 2200.0)
            r_fix = quantize_and_process(
                rx_sig, NOMINAL_FIXED_GAIN, "fixed", target_dop_shift, TARGET_RANGE_BIN, CLUTTER_RANGE_BIN
            )

            if s_idx == 0 and c_amp in (0.60, 2.00):
                rd_maps_to_save[f"dyn_c{c_amp:.2f}"] = r_dyn["power_map"]
                rd_maps_to_save[f"fix_c{c_amp:.2f}"] = r_fix["power_map"]

            r_dyn.pop("power_map", None)
            r_fix.pop("power_map", None)
            dyn_results.append(r_dyn)
            fix_results.append(r_fix)

        clutter_comp_rows.append({
            "clutter_amplitude": c_amp,
            "target_amplitude": TARGET_AMPLITUDE,
            "dyn_scale_factor_mean": float(np.mean([r["scale_factor"] for r in dyn_results])),
            "dyn_clipping_pct": float(np.mean([r["clipping_pct"] for r in dyn_results])),
            "dyn_sqnr_db": float(np.mean([r["sqnr_db"] for r in dyn_results])),
            "dyn_cfar_margin_db": float(np.mean([r["cfar_margin_db"] for r in dyn_results])),
            "dyn_cfar_pd": float(np.mean([r["cfar_pd"] for r in dyn_results])),
            "fix_gain": NOMINAL_FIXED_GAIN,
            "fix_clipping_pct": float(np.mean([r["clipping_pct"] for r in fix_results])),
            "fix_sqnr_db": float(np.mean([r["sqnr_db"] for r in fix_results])),
            "fix_cfar_margin_db": float(np.mean([r["cfar_margin_db"] for r in fix_results])),
            "fix_cfar_pd": float(np.mean([r["cfar_pd"] for r in fix_results])),
            "margin_advantage_dyn_vs_fix_db": float(
                np.mean([r["cfar_margin_db"] for r in dyn_results]) - np.mean([r["cfar_margin_db"] for r in fix_results])
            ),
        })
        print(
            f"  Ac={c_amp:4.2f} | Dyn: Margin={clutter_comp_rows[-1]['dyn_cfar_margin_db']:+6.2f}dB (Clip={clutter_comp_rows[-1]['dyn_clipping_pct']:4.1f}%) | "
            f"Fix(2200): Margin={clutter_comp_rows[-1]['fix_cfar_margin_db']:+6.2f}dB (Clip={clutter_comp_rows[-1]['fix_clipping_pct']:5.1f}%, SQNR={clutter_comp_rows[-1]['fix_sqnr_db']:5.1f}dB)"
        )

    clut_csv_path = base_dir / "adc_clutter_sweep_comparison.csv"
    with open(clut_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(clutter_comp_rows[0].keys()))
        w.writeheader()
        w.writerows(clutter_comp_rows)
    print(f"Saved: {clut_csv_path}")

    # ---------------------------------------------------------
    # PART 3: GENERATE PUBLICATION FIGURES
    # ---------------------------------------------------------
    print("\n[Part 3] Generating diagnostic figures...")

    # Figure 1: Clipping % and SQNR vs Fixed Gain
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    gains = [r["gain"] for r in gain_sweep_rows]
    clips = [r["clipping_pct_mean"] for r in gain_sweep_rows]
    sqnrs = [r["sqnr_db_mean"] for r in gain_sweep_rows]

    ax1.plot(gains, clips, "o-", color="#d62728", lw=2, ms=6)
    ax1.axvline(NOMINAL_FIXED_GAIN, color="#1f77b4", linestyle="--", lw=1.2, label=f"Nominal Design Gain ({NOMINAL_FIXED_GAIN})")
    ax1.set_title("ADC Clipping Percentage vs. Fixed Input Gain ($A_c=0.60$)", fontsize=11)
    ax1.set_xlabel("Fixed Input Gain Multiplier $G_{\\text{fixed}}$")
    ax1.set_ylabel("Clipping Percentage (%)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left")

    ax2.plot(gains, sqnrs, "s-", color="#2ca02c", lw=2, ms=6)
    ax2.axvline(NOMINAL_FIXED_GAIN, color="#1f77b4", linestyle="--", lw=1.2, label=f"Nominal Design Gain ({NOMINAL_FIXED_GAIN})")
    ax2.set_title("Signal-to-Quantisation-Noise Ratio (SQNR) vs. Gain", fontsize=11)
    ax2.set_xlabel("Fixed Input Gain Multiplier $G_{\\text{fixed}}$")
    ax2.set_ylabel("SQNR [dB]")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    fig.suptitle("Extended Validation 3: ADC Saturation & Quantisation Dynamics", fontsize=13)
    fig.tight_layout()
    fig1_path = base_dir / "adc_clipping_and_sqnr_vs_gain.png"
    fig.savefig(fig1_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {fig1_path}")

    # Figure 2: Pd vs Clutter Amplitude (Dynamic vs Fixed)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    c_amps = [r["clutter_amplitude"] for r in clutter_comp_rows]
    dyn_pds = [r["dyn_cfar_pd"] for r in clutter_comp_rows]
    fix_pds = [r["fix_cfar_pd"] for r in clutter_comp_rows]
    dyn_margins = [r["dyn_cfar_margin_db"] for r in clutter_comp_rows]
    fix_margins = [r["fix_cfar_margin_db"] for r in clutter_comp_rows]

    ax1.plot(c_amps, dyn_pds, "o-", color="#1f77b4", lw=2, ms=6, label="Ideal Dynamic Scaling")
    ax1.plot(c_amps, fix_pds, "s--", color="#d62728", lw=2, ms=6, label=f"Fixed Gain ($G={NOMINAL_FIXED_GAIN}$)")
    ax1.set_title("Detection Probability $P_d$ vs. Clutter Amplitude", fontsize=11)
    ax1.set_xlabel("Clutter Amplitude ($A_c$)")
    ax1.set_ylabel("Detection Probability $P_d$")
    ax1.set_ylim(-0.05, 1.1)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="lower left")

    ax2.plot(c_amps, dyn_margins, "o-", color="#1f77b4", lw=2, ms=6, label="Ideal Dynamic Scaling")
    ax2.plot(c_amps, fix_margins, "s--", color="#d62728", lw=2, ms=6, label=f"Fixed Gain ($G={NOMINAL_FIXED_GAIN}$)")
    ax2.axhline(0.0, color="black", linestyle=":", lw=1.2, label="Detection Threshold (0 dB)")
    ax2.axvline(0.60, color="gray", linestyle="--", lw=1.0, alpha=0.7, label="Design Clutter ($A_c=0.60$)")
    ax2.set_title("CFAR Detection Margin vs. Clutter Amplitude", fontsize=11)
    ax2.set_xlabel("Clutter Amplitude ($A_c$)")
    ax2.set_ylabel("Target Detection Margin [dB]")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")

    fig.suptitle("Extended Validation 3: Target Detection Under Clutter Growth (Fixed vs Dynamic)", fontsize=13)
    fig.tight_layout()
    fig2_path = base_dir / "adc_margin_vs_clutter_amplitude.png"
    fig.savefig(fig2_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {fig2_path}")

    # Figure 3: Range-Doppler Distortion under Severe Clipping
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    plot_cases = [
        ("dyn_c0.60", "Dynamic Scaling: Nominal Clutter $A_c=0.60$ (No Clipping)"),
        ("fix_c0.60", "Fixed Gain: Nominal Clutter $A_c=0.60$ (Linear Region)"),
        ("dyn_c2.00", "Dynamic Scaling: Severe Clutter $A_c=2.00$ (Quantisation Backoff)"),
        ("fix_c2.00", "Fixed Gain: Severe Clutter $A_c=2.00$ (Severe Clipping > 30%)"),
    ]
    for ax, (case_id, title) in zip(axes.flat, plot_cases):
        pmap = rd_maps_to_save.get(case_id)
        if pmap is not None:
            im = ax.imshow(
                10.0 * np.log10(pmap + 1e-12),
                aspect="auto",
                origin="lower",
                cmap="viridis",
                extent=[0, NR - 1, -ND // 2, ND // 2 - 1],
            )
            ax.set_title(title, fontsize=10.5)
            ax.set_xlabel("Range Bin")
            ax.set_ylabel("Doppler Bin")
            fig.colorbar(im, ax=ax, label="Power [dB]")

    fig.suptitle("ADC Clipping Spectral Regrowth & Intermodulation Splatter in Range-Doppler Domain", fontsize=13)
    fig.tight_layout()
    fig3_path = base_dir / "adc_clipping_spectral_distortion.png"
    fig.savefig(fig3_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {fig3_path}")

    print("\nExperiment 3 execution completed successfully.")


if __name__ == "__main__":
    main()

