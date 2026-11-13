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
# CONSTANTS & DETECTOR SPECIFICATIONS (OBJECTIVE I FROZEN)
# ---------------------------------------------------------

NR = 512
ND = 32

ADC_BITS = 12
ADC_MAX = 2 ** (ADC_BITS - 1) - 1
ADC_MIN = -(2 ** (ADC_BITS - 1))

REFERENCE_CELLS_PER_SIDE = 4
GUARD_CELLS_PER_SIDE = 2
EDGE_SIZE = REFERENCE_CELLS_PER_SIDE + GUARD_CELLS_PER_SIDE  # 6
VALID_CELLS_PER_FRAME = ND * (NR - 2 * EDGE_SIZE)  # Exactly 16,000 valid cells

RANGE_WINDOW = np.hanning(NR)
DOPPLER_WINDOW = np.hanning(ND)

n_idx = np.arange(NR)
m_idx = np.arange(ND)
n_grid, m_grid = np.meshgrid(n_idx, m_idx)

ALPHA_GRID = [5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]
TARGET_AMPLITUDES = [0.001, 0.003, 0.01, 0.03, 0.10]
NOISE_AMPLITUDE = 0.08
CLUTTER_AMPLITUDE = 0.60
CLUTTER_RANGE_BIN = 72


def wilson_score_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """Calculate Wilson score confidence interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    z = 1.959963984540054  # 95% two-sided normal quantile
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2.0 * n)) / denom
    half_width = z * np.sqrt((p * (1.0 - p) + z**2 / (4.0 * n)) / n) / denom
    lower = max(0.0, center - half_width)
    upper = min(1.0, center + half_width)
    return p, lower, upper


def compute_frame_rd_power(
    target_sig: np.ndarray,
    clutter_sig: np.ndarray,
    noise_sig: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Generate unnormalized magnitude-squared power map with 12-bit ADC scaling."""
    received = target_sig + clutter_sig + noise_sig
    max_val = max(
        float(np.max(np.abs(received.real))),
        float(np.max(np.abs(received.imag))),
    )
    scale = (0.90 * ADC_MAX / max_val) if max_val > 0 else 1.0

    i_samp = np.clip(np.round(received.real * scale), ADC_MIN, ADC_MAX).astype(np.int16)
    q_samp = np.clip(np.round(received.imag * scale), ADC_MIN, ADC_MAX).astype(np.int16)

    ref_in = (i_samp.astype(np.float64) + 1j * q_samp.astype(np.float64)) / scale

    win_sig = ref_in * RANGE_WINDOW[None, :]
    r_fft = np.fft.fft(win_sig, axis=1)

    r_fft_win = r_fft * DOPPLER_WINDOW[:, None]
    d_fft = np.fft.fft(r_fft_win, axis=0)
    rd_map = np.fft.fftshift(d_fft, axes=0)
    power_map = np.abs(rd_map) ** 2
    return power_map, scale


def compute_cfar_noise_map(power_map: np.ndarray) -> np.ndarray:
    """Compute 1D range CA-CFAR local noise estimate across all valid cells."""
    noise_map = np.zeros((ND, NR - 2 * EDGE_SIZE), dtype=np.float64)
    # Vectorized / efficient slice computation
    for offset, r in enumerate(range(EDGE_SIZE, NR - EDGE_SIZE)):
        l_start = r - GUARD_CELLS_PER_SIDE - REFERENCE_CELLS_PER_SIDE
        l_stop = r - GUARD_CELLS_PER_SIDE
        r_start = r + GUARD_CELLS_PER_SIDE + 1
        r_stop = r_start + REFERENCE_CELLS_PER_SIDE
        # Sum of 4 left + 4 right reference cells
        left_sum = np.sum(power_map[:, l_start:l_stop], axis=1)
        right_sum = np.sum(power_map[:, r_start:r_stop], axis=1)
        noise_map[:, offset] = (left_sum + right_sum) / 8.0
    return noise_map


def main():
    base_dir = Path(__file__).resolve().parent
    config_file = base_dir.parent.parent / "configs" / "extended_validation" / "cfar_calibration_config.json"
    
    print("=" * 75)
    print("EXTENDED VALIDATION 2: EMPIRICAL CA-CFAR CALIBRATION & ROC ANALYSIS")
    print("=" * 75)
    
    t0_total = time.time()
    
    # ---------------------------------------------------------
    # 1. THRESHOLD CALIBRATION (100 NOISE-ONLY FRAMES)
    # ---------------------------------------------------------
    N_CAL = 100
    SEED_CAL_START = 1001
    print(f"\n[Phase 1] Simulating {N_CAL} noise-only calibration frames (seeds {SEED_CAL_START}..{SEED_CAL_START+N_CAL-1})...")
    
    cal_cell_counts = {alpha: 0 for alpha in ALPHA_GRID}
    total_cal_cells = N_CAL * VALID_CELLS_PER_FRAME
    
    # Store ratios of CUT power to local noise estimate across all valid cells to derive exact empirical alpha for 1%
    # Ratio = CUT_power / local_noise. If Ratio > alpha, false alarm occurs!
    # Therefore, Pfa(alpha) = Prob(Ratio > alpha).
    # For target Pfa = 1.0%, alpha_1% is exactly the 99th percentile of this ratio!
    all_cal_ratios = []
    
    zero_sig = np.zeros((ND, NR), dtype=complex)
    
    t0 = time.time()
    for i in range(N_CAL):
        seed = SEED_CAL_START + i
        rng = np.random.RandomState(seed)
        noise = NOISE_AMPLITUDE * (rng.randn(ND, NR) + 1j * rng.randn(ND, NR))
        power_map, _ = compute_frame_rd_power(zero_sig, zero_sig, noise)
        noise_map = compute_cfar_noise_map(power_map)
        
        valid_cut = power_map[:, EDGE_SIZE:NR - EDGE_SIZE]
        ratios = (valid_cut / noise_map).ravel()
        all_cal_ratios.append(ratios)
        
        for alpha in ALPHA_GRID:
            cal_cell_counts[alpha] += int(np.sum(ratios > alpha))
            
    all_cal_ratios = np.concatenate(all_cal_ratios)
    print(f"Phase 1 completed in {time.time() - t0:.2f}s ({total_cal_cells:,} total cell evaluations).")
    
    # Derive empirical alpha for target Pfa = 1.0% (99.0th percentile of ratio distribution)
    alpha_calibrated_1pct = float(np.percentile(all_cal_ratios, 99.0))
    emp_pfa_at_cal = float(np.mean(all_cal_ratios > alpha_calibrated_1pct))
    
    print(f"\n--- EMPIRICAL CALIBRATION RESULT ---")
    print(f"Target Design Pfa : 1.000%")
    print(f"Calibrated Alpha  : {alpha_calibrated_1pct:.4f}")
    print(f"Calibration Pfa   : {emp_pfa_at_cal * 100:.4f}%")
    
    # ---------------------------------------------------------
    # 2. NOISE-ONLY VALIDATION (200 INDEPENDENT FRAMES)
    # ---------------------------------------------------------
    N_VAL = 200
    SEED_VAL_START = 2001
    print(f"\n[Phase 2] Simulating {N_VAL} independent noise-only validation frames (seeds {SEED_VAL_START}..{SEED_VAL_START+N_VAL-1})...")
    
    eval_alphas = sorted(list(set(ALPHA_GRID + [round(alpha_calibrated_1pct, 4)])))
    val_cell_counts = {alpha: 0 for alpha in eval_alphas}
    total_val_cells = N_VAL * VALID_CELLS_PER_FRAME
    
    t0 = time.time()
    for i in range(N_VAL):
        seed = SEED_VAL_START + i
        rng = np.random.RandomState(seed)
        noise = NOISE_AMPLITUDE * (rng.randn(ND, NR) + 1j * rng.randn(ND, NR))
        power_map, _ = compute_frame_rd_power(zero_sig, zero_sig, noise)
        noise_map = compute_cfar_noise_map(power_map)
        
        valid_cut = power_map[:, EDGE_SIZE:NR - EDGE_SIZE]
        ratios = (valid_cut / noise_map).ravel()
        
        for alpha in eval_alphas:
            val_cell_counts[alpha] += int(np.sum(ratios > alpha))
            
    print(f"Phase 2 completed in {time.time() - t0:.2f}s ({total_val_cells:,} total cell evaluations).")
    
    # ---------------------------------------------------------
    # COMPOSE NOISE-ONLY Pfa RESULTS & COMPARISON
    # ---------------------------------------------------------
    noise_pfa_rows = []
    for alpha in eval_alphas:
        cal_k = cal_cell_counts.get(alpha, int(np.sum(all_cal_ratios > alpha)))
        cal_p, cal_lo, cal_hi = wilson_score_interval(cal_k, total_cal_cells)
        
        val_k = val_cell_counts[alpha]
        val_p, val_lo, val_hi = wilson_score_interval(val_k, total_val_cells)
        
        # Uncorrelated theory: (1 + alpha/8)^(-8)
        pfa_uncorr_theory = (1.0 + alpha / 8.0) ** (-8.0)
        # Hann effective N_eff ~ 4.714 theory:
        pfa_hann_theory = (1.0 + alpha / 4.714) ** (-4.714)
        
        noise_pfa_rows.append({
            "alpha": alpha,
            "is_calibrated_1pct": (alpha == round(alpha_calibrated_1pct, 4)),
            "is_baseline_alpha6": (alpha == 6.0),
            "cal_cells_total": total_cal_cells,
            "cal_false_alarms": cal_k,
            "cal_pfa": cal_p,
            "cal_pfa_ci95_low": cal_lo,
            "cal_pfa_ci95_high": cal_hi,
            "val_cells_total": total_val_cells,
            "val_false_alarms": val_k,
            "val_pfa": val_p,
            "val_pfa_ci95_low": val_lo,
            "val_pfa_ci95_high": val_hi,
            "pfa_uncorr_theory": pfa_uncorr_theory,
            "pfa_hann_effective_theory": pfa_hann_theory,
            "val_vs_cal_rel_diff_pct": 100.0 * (val_p - cal_p) / cal_p if cal_p > 0 else 0.0,
        })
        
    pfa_csv_path = base_dir / "cfar_calibration_noise_pfa.csv"
    with open(pfa_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(noise_pfa_rows[0].keys()))
        w.writeheader()
        w.writerows(noise_pfa_rows)
    print(f"Saved Noise Pfa Results: {pfa_csv_path}")

    print("\n--- PFA CALIBRATION VS VALIDATION SUMMARY ---")
    print(f"{'Alpha':>7s} | {'Cal Pfa (95% CI)':>22s} | {'Val Pfa (95% CI)':>22s} | {'Hann Theory':>11s} | {'Uncorr Theory':>13s} | {'Notes'}")
    print("-" * 95)
    for r in noise_pfa_rows:
        cal_str = f"{r['cal_pfa']*100:.3f}% [{r['cal_pfa_ci95_low']*100:.3f}..{r['cal_pfa_ci95_high']*100:.3f}]"
        val_str = f"{r['val_pfa']*100:.3f}% [{r['val_pfa_ci95_low']*100:.3f}..{r['val_pfa_ci95_high']*100:.3f}]"
        note = "Baseline" if r["is_baseline_alpha6"] else ("Calibrated 1%" if r["is_calibrated_1pct"] else "")
        print(f"{r['alpha']:7.4f} | {cal_str:>22s} | {val_str:>22s} | {r['pfa_hann_effective_theory']*100:10.3f}% | {r['pfa_uncorr_theory']*100:12.3f}% | {note}")

    # ---------------------------------------------------------
    # 3. TARGET-PRESENT DETECTION & ROC ANALYSIS
    # ---------------------------------------------------------
    N_RUNS_PER_AMP = 50
    SEED_TARGET_START = 3001
    print(f"\n[Phase 3] Target-present evaluation ({N_RUNS_PER_AMP} runs per amplitude across 5 amplitudes)...")
    
    # We test two primary operating environments:
    # 1. Clutter-Free: Target at R_t = 70, d_t = +2.0 (Doppler 18)
    # 2. Clutter-Present (Distant from clutter center): Target at R_t = 120, Clutter at R_c = 72, d_c = 0.0
    # 3. Clutter-Present (Adjacent to clutter): Target at R_t = 70, Clutter at R_c = 72, d_c = 0.0
    
    environments = [
        ("clutter_free", 70, 2.0, 0.0, 72, 0.0, "Clutter-Free (Rt=70)"),
        ("clutter_distant", 120, 2.0, 0.60, 72, 0.0, "Clutter-Present Distant (Rt=120, Rc=72)"),
        ("clutter_adjacent", 70, 2.0, 0.60, 72, 0.0, "Clutter-Present Adjacent (Rt=70, Rc=72)"),
    ]
    
    roc_results_rows = []
    
    for env_id, r_target, d_target, c_amp, r_clut, c_dop, env_label in environments:
        t_dop_shift = int(np.round((d_target + ND // 2) % ND))
        c_dop_shift = int(np.round((c_dop + ND // 2) % ND))
        
        # Clutter signal
        if c_amp > 0:
            c_phase = 2.0 * np.pi * (r_clut * n_grid / NR + c_dop * m_grid / ND)
            clut_sig = c_amp * np.exp(1j * c_phase)
        else:
            clut_sig = zero_sig
            
        print(f"  Testing Environment: {env_label}...")
        
        for amp in TARGET_AMPLITUDES:
            t_phase = 2.0 * np.pi * (r_target * n_grid / NR + d_target * m_grid / ND)
            t_sig = amp * np.exp(1j * t_phase)
            
            # Store target power and local noise estimate at target CUT for all runs
            t_powers = []
            loc_noises = []
            
            for run_i in range(N_RUNS_PER_AMP):
                seed = SEED_TARGET_START + run_i
                rng = np.random.RandomState(seed)
                noise = NOISE_AMPLITUDE * (rng.randn(ND, NR) + 1j * rng.randn(ND, NR))
                
                power_map, _ = compute_frame_rd_power(t_sig, clut_sig, noise)
                
                # Extract reference cells around CUT (d=t_dop_shift, r=r_target)
                l_start = r_target - GUARD_CELLS_PER_SIDE - REFERENCE_CELLS_PER_SIDE
                l_stop = r_target - GUARD_CELLS_PER_SIDE
                r_start = r_target + GUARD_CELLS_PER_SIDE + 1
                r_stop = r_start + REFERENCE_CELLS_PER_SIDE
                ref_cells = np.concatenate((
                    power_map[t_dop_shift, l_start:l_stop],
                    power_map[t_dop_shift, r_start:r_stop]
                ))
                loc_noise = float(np.mean(ref_cells))
                t_pwr = float(power_map[t_dop_shift, r_target])
                
                t_powers.append(t_pwr)
                loc_noises.append(loc_noise)
                
            t_powers = np.array(t_powers)
            loc_noises = np.array(loc_noises)
            
            for alpha in eval_alphas:
                thr = alpha * loc_noises
                detected = (t_powers > thr)
                k_det = int(np.sum(detected))
                pd, pd_lo, pd_hi = wilson_score_interval(k_det, N_RUNS_PER_AMP)
                
                margins_db = 10.0 * np.log10(t_powers / thr)
                mean_margin_db = float(np.mean(margins_db))
                min_margin_db = float(np.min(margins_db))
                
                # Get matching validated Pfa for this alpha
                val_pfa = next(r["val_pfa"] for r in noise_pfa_rows if r["alpha"] == alpha)
                
                roc_results_rows.append({
                    "environment_id": env_id,
                    "environment_label": env_label,
                    "target_amplitude": amp,
                    "target_range_bin": r_target,
                    "target_doppler_bin": d_target,
                    "clutter_amplitude": c_amp,
                    "alpha": alpha,
                    "is_calibrated_1pct": (alpha == round(alpha_calibrated_1pct, 4)),
                    "is_baseline_alpha6": (alpha == 6.0),
                    "runs_evaluated": N_RUNS_PER_AMP,
                    "target_detections": k_det,
                    "pd": pd,
                    "pd_ci95_low": pd_lo,
                    "pd_ci95_high": pd_hi,
                    "cfar_margin_db_mean": mean_margin_db,
                    "cfar_margin_db_min": min_margin_db,
                    "val_pfa": val_pfa,
                    "mean_target_power": float(np.mean(t_powers)),
                    "mean_cfar_threshold": float(np.mean(thr)),
                })

    roc_csv_path = base_dir / "cfar_roc_results.csv"
    with open(roc_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(roc_results_rows[0].keys()))
        w.writeheader()
        w.writerows(roc_results_rows)
    print(f"Saved ROC Results CSV: {roc_csv_path}")

    # ---------------------------------------------------------
    # 4. GENERATE PUBLICATION-GRADE VISUALIZATIONS
    # ---------------------------------------------------------
    print("\n[Phase 4] Generating diagnostic figures...")

    # Figure 1: Calibration vs Validation Pfa vs Alpha
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    alphas_plot = [r["alpha"] for r in noise_pfa_rows if not r["is_calibrated_1pct"]]
    cal_pfas = [100.0 * r["cal_pfa"] for r in noise_pfa_rows if not r["is_calibrated_1pct"]]
    val_pfas = [100.0 * r["val_pfa"] for r in noise_pfa_rows if not r["is_calibrated_1pct"]]
    
    # 95% Error bars
    cal_err = [
        [100.0 * (r["cal_pfa"] - r["cal_pfa_ci95_low"]) for r in noise_pfa_rows if not r["is_calibrated_1pct"]],
        [100.0 * (r["cal_pfa_ci95_high"] - r["cal_pfa"]) for r in noise_pfa_rows if not r["is_calibrated_1pct"]],
    ]
    val_err = [
        [100.0 * (r["val_pfa"] - r["val_pfa_ci95_low"]) for r in noise_pfa_rows if not r["is_calibrated_1pct"]],
        [100.0 * (r["val_pfa_ci95_high"] - r["val_pfa"]) for r in noise_pfa_rows if not r["is_calibrated_1pct"]],
    ]
    
    fine_alpha = np.linspace(5.0, 9.0, 100)
    theory_uncorr = 100.0 * (1.0 + fine_alpha / 8.0) ** (-8.0)
    theory_hann = 100.0 * (1.0 + fine_alpha / 4.714) ** (-4.714)
    
    ax.plot(fine_alpha, theory_uncorr, "--", color="#7f7f7f", lw=1.3, label="Uncorrelated Theory ($N=8$, IID)")
    ax.plot(fine_alpha, theory_hann, "-", color="#1f77b4", lw=1.8, label="Hann Effective Theory ($N_{eff}=4.714$)")
    
    ax.errorbar(alphas_plot, cal_pfas, yerr=cal_err, fmt="o", color="#ff7f0e", capsize=4, lw=1.5, ms=6, label=f"Calibration Dataset (100 frames, 1.6M cells)")
    ax.errorbar(alphas_plot, val_pfas, yerr=val_err, fmt="s", color="#2ca02c", capsize=4, lw=1.5, ms=6, label=f"Validation Dataset (200 frames, 3.2M cells)")
    
    # Highlight baseline alpha=6 and calibrated alpha_1%
    row_base = next(r for r in noise_pfa_rows if r["is_baseline_alpha6"])
    row_cal = next(r for r in noise_pfa_rows if r["is_calibrated_1pct"])
    
    ax.axhline(1.0, color="#d62728", linestyle=":", lw=1.4, label=r"Target $P_{fa} = 1.000\%$")
    ax.axvline(6.0, color="#1f77b4", linestyle=":", lw=1.1, alpha=0.7)
    ax.axvline(row_cal["alpha"], color="#d62728", linestyle=":", lw=1.1, alpha=0.7)
    
    ax.scatter([6.0], [row_base['val_pfa']*100], color="#1f77b4", s=90, zorder=5, label=f"Baseline $\\alpha=6.0$ ($P_{{fa}}={row_base['val_pfa']*100:.3f}\\%$)")
    ax.scatter([row_cal["alpha"]], [row_cal['val_pfa']*100], color="#d62728", s=110, marker="*", zorder=5, label=f"Calibrated $\\alpha={row_cal['alpha']:.3f}$ ($P_{{fa}}={row_cal['val_pfa']*100:.3f}\\%$)")
    
    ax.set_title("CA-CFAR False-Alarm Rate Calibration: Theory vs. Empirical Datasets", fontsize=12)
    ax.set_xlabel("CA-CFAR Threshold Multiplier $\\alpha$")
    ax.set_ylabel("False-Alarm Probability $P_{fa}$ (%)")
    ax.set_xlim(5.2, 8.8)
    ax.set_ylim(0.2, 3.2)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8.5)
    fig.tight_layout()
    fig_pfa_path = base_dir / "cfar_calibration_pfa_vs_alpha.png"
    fig.savefig(fig_pfa_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {fig_pfa_path}")

    # Figure 2: ROC Curves (Pd vs Pfa) for Clutter-Free & Clutter-Distant
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    colors = ["#9467bd", "#8c564b", "#1f77b4", "#ff7f0e", "#2ca02c"]
    
    for ax, env_id, env_title in [(ax1, "clutter_free", "Clutter-Free ($R_t=70$)"), (ax2, "clutter_distant", "Clutter-Present ($R_t=120, R_c=72$)")]:
        for amp, color in zip(TARGET_AMPLITUDES, colors):
            sub = [r for r in roc_results_rows if r["environment_id"] == env_id and r["target_amplitude"] == amp]
            pfas = [100.0 * s["val_pfa"] for s in sub]
            pds = [s["pd"] for s in sub]
            ax.plot(pfas, pds, "o-", color=color, lw=2, ms=5, label=f"$A_t = {amp}$")
            
        ax.axvline(row_base['val_pfa']*100, color="#1f77b4", linestyle="--", lw=1.1, label=f"Baseline $\\alpha=6.0$ ($P_{{fa}}={row_base['val_pfa']*100:.2f}\\%$)")
        ax.axvline(row_cal['val_pfa']*100, color="#d62728", linestyle=":", lw=1.3, label=f"Calibrated $\\alpha={row_cal['alpha']:.2f}$ ($P_{{fa}}={row_cal['val_pfa']*100:.2f}\\%$)")
        ax.set_title(f"ROC Curves: {env_title}", fontsize=11)
        ax.set_xlabel("Empirical False-Alarm Probability $P_{fa}$ (%)")
        ax.set_ylabel("Detection Probability $P_d$")
        ax.set_xlim(0.4, 2.8)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8.5)

    fig.suptitle("Extended Validation 2: Receiver Operating Characteristic (ROC) Analysis", fontsize=13)
    fig.tight_layout()
    fig_roc_path = base_dir / "cfar_roc_curves.png"
    fig.savefig(fig_roc_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {fig_roc_path}")

    # Figure 3: Detection Margin vs Alpha & Performance Trade-off
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    
    # Margin vs alpha for Clutter-Free
    for amp, color in zip(TARGET_AMPLITUDES, colors):
        sub = [r for r in roc_results_rows if r["environment_id"] == "clutter_free" and r["target_amplitude"] == amp]
        alphas_val = [s["alpha"] for s in sub]
        margins = [s["cfar_margin_db_mean"] for s in sub]
        ax1.plot(alphas_val, margins, "o-", color=color, lw=2, ms=5, label=f"$A_t = {amp}$")
        
    ax1.axhline(0.0, color="red", linestyle="--", lw=1.2, label="Detection Boundary (0 dB)")
    ax1.axvline(6.0, color="#1f77b4", linestyle=":", lw=1.2, label="Baseline $\\alpha=6.0$")
    ax1.axvline(row_cal["alpha"], color="#d62728", linestyle=":", lw=1.2, label=f"Calibrated $\\alpha={row_cal['alpha']:.2f}$")
    ax1.set_title("CA-CFAR Detection Margin vs. Threshold Multiplier $\\alpha$", fontsize=11)
    ax1.set_xlabel("Threshold Multiplier $\\alpha$")
    ax1.set_ylabel("Mean Detection Margin [dB]")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8.5)
    
    # Trade-off: Pd vs Target Amplitude at alpha=6.0 vs calibrated alpha
    x_indices = np.arange(len(TARGET_AMPLITUDES))
    sub_base = [r for r in roc_results_rows if r["environment_id"] == "clutter_free" and r["alpha"] == 6.0]
    sub_cal = [r for r in roc_results_rows if r["environment_id"] == "clutter_free" and r["alpha"] == round(alpha_calibrated_1pct, 4)]
    
    w_bar = 0.35
    ax2.bar(x_indices - w_bar / 2, [s["pd"] for s in sub_base], w_bar, color="#1f77b4", label=f"Baseline $\\alpha=6.0$ ($P_{{fa}}={row_base['val_pfa']*100:.2f}\\%$)")
    ax2.bar(x_indices + w_bar / 2, [s["pd"] for s in sub_cal], w_bar, color="#d62728", label=f"Calibrated $\\alpha={row_cal['alpha']:.2f}$ ($P_{{fa}}={row_cal['val_pfa']*100:.2f}\\%$)")
    
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels([f"{amp}" for amp in TARGET_AMPLITUDES])
    ax2.set_title("Detection Probability $P_d$ Trade-Off Across Target Amplitudes", fontsize=11)
    ax2.set_xlabel("Target Amplitude ($A_t$)")
    ax2.set_ylabel("Detection Probability $P_d$")
    ax2.set_ylim(0, 1.15)
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(loc="upper left", fontsize=8.5)
    
    fig.suptitle("Extended Validation 2: Operational Sensitivity & Trade-Off Analysis", fontsize=13)
    fig.tight_layout()
    fig_margin_path = base_dir / "cfar_detection_margin_tradeoff.png"
    fig.savefig(fig_margin_path, dpi=180)
    plt.close(fig)
    print(f"Saved: {fig_margin_path}")

    # ---------------------------------------------------------
    # COMPOSE SUMMARY COMPARISON CSV
    # ---------------------------------------------------------
    summary_tradeoff_rows = []
    for amp in TARGET_AMPLITUDES:
        sub_b = next(r for r in roc_results_rows if r["environment_id"] == "clutter_free" and r["target_amplitude"] == amp and r["alpha"] == 6.0)
        sub_c = next(r for r in roc_results_rows if r["environment_id"] == "clutter_free" and r["target_amplitude"] == amp and r["alpha"] == round(alpha_calibrated_1pct, 4))
        
        summary_tradeoff_rows.append({
            "target_amplitude": amp,
            "baseline_alpha": 6.0,
            "baseline_val_pfa": row_base["val_pfa"],
            "baseline_pd": sub_b["pd"],
            "baseline_margin_db": sub_b["cfar_margin_db_mean"],
            "calibrated_alpha": round(alpha_calibrated_1pct, 4),
            "calibrated_val_pfa": row_cal["val_pfa"],
            "calibrated_pd": sub_c["pd"],
            "calibrated_margin_db": sub_c["cfar_margin_db_mean"],
            "delta_margin_db": sub_c["cfar_margin_db_mean"] - sub_b["cfar_margin_db_mean"],
            "delta_pd": sub_c["pd"] - sub_b["pd"],
            "pfa_reduction_factor": row_base["val_pfa"] / row_cal["val_pfa"],
        })
        
    sum_csv_path = base_dir / "cfar_calibration_comparison_summary.csv"
    with open(sum_csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_tradeoff_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_tradeoff_rows)
    print(f"Saved Trade-off Summary: {sum_csv_path}")

    print(f"\nAll tasks completed successfully in {time.time() - t0_total:.2f}s.")


if __name__ == "__main__":
    main()

