import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
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

TARGET_AMPLITUDE = 0.03   # Weak target
CLUTTER_AMPLITUDE = 0.60
NOISE_AMPLITUDE = 0.08

CLUTTER_RANGE_BIN = 72

TARGET_RANGE_BINS = list(range(64, 81))  # 64 through 80 inclusive (17 bins)
TARGET_DOPPLERS = [1.0, 2.0]
CLUTTER_DOPPLERS = [0.0, 0.3]
SEEDS = list(range(1, 11))  # 1 through 10

ADC_BITS = 12
ADC_MAX = 2 ** (ADC_BITS - 1) - 1
ADC_MIN = -(2 ** (ADC_BITS - 1))

THRESHOLD_FACTOR = 6.0
REFERENCE_CELLS_PER_SIDE = 4
GUARD_CELLS_PER_SIDE = 2
CFAR_ALPHA = 6.0
EDGE_SIZE = REFERENCE_CELLS_PER_SIDE + GUARD_CELLS_PER_SIDE  # 6

CFAR_VALID_MASK = np.zeros((ND, NR), dtype=bool)
CFAR_VALID_MASK[:, EDGE_SIZE:NR - EDGE_SIZE] = True  # Exactly 16,000 valid cells

RANGE_WINDOW = np.hanning(NR)
DOPPLER_WINDOW = np.hanning(ND)

n_idx = np.arange(NR)
m_idx = np.arange(ND)
n_grid, m_grid = np.meshgrid(n_idx, m_idx)

# Nominal clean noise floor CFAR threshold (uncontaminated baseline reference ~ 190.0)
NOMINAL_NOISE_CFAR_THRESHOLD = 190.0


def run_single_simulation(
    target_range: int,
    target_doppler: float,
    clutter_doppler: float,
    seed: int,
    output_dir: Path,
    save_plots: bool = True,
) -> dict:
    """Run a single 512x32 radar simulation matching radar_model.py mathematically."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)

    # 1. Synthetic Signal Generation
    target_phase = 2.0 * np.pi * (
        target_range * n_grid / NR + target_doppler * m_grid / ND
    )
    target_sig = TARGET_AMPLITUDE * np.exp(1j * target_phase)

    clutter_phase = 2.0 * np.pi * (
        CLUTTER_RANGE_BIN * n_grid / NR + clutter_doppler * m_grid / ND
    )
    clutter_sig = CLUTTER_AMPLITUDE * np.exp(1j * clutter_phase)

    noise_sig = NOISE_AMPLITUDE * (
        rng.randn(ND, NR) + 1j * rng.randn(ND, NR)
    )

    received_sig = target_sig + clutter_sig + noise_sig

    # 2. Signed 12-bit Quantization
    max_val = max(
        float(np.max(np.abs(received_sig.real))),
        float(np.max(np.abs(received_sig.imag))),
    )
    scaling_factor = (0.90 * ADC_MAX / max_val) if max_val > 0 else 1.0

    i_samples = np.clip(
        np.round(received_sig.real * scaling_factor), ADC_MIN, ADC_MAX
    ).astype(np.int16)
    q_samples = np.clip(
        np.round(received_sig.imag * scaling_factor), ADC_MIN, ADC_MAX
    ).astype(np.int16)

    # 3. Save Raw I/Q input (Chirp-major order)
    iq_data = np.column_stack((i_samples.ravel(), q_samples.ravel()))
    iq_file = output_dir / "radar_input_iq.txt"
    np.savetxt(iq_file, iq_data, fmt="%d")

    # Shifted Doppler coordinates
    target_doppler_shifted = int(np.round((target_doppler + ND // 2) % ND))
    clutter_doppler_shifted = int(np.round((clutter_doppler + ND // 2) % ND))

    # Save metadata
    meta_file = output_dir / "simulation_metadata.npz"
    np.savez(
        meta_file,
        nr=NR,
        nd=ND,
        seed=seed,
        adc_bits=ADC_BITS,
        scaling_factor=scaling_factor,
        target_amplitude=TARGET_AMPLITUDE,
        target_range_bin=target_range,
        target_doppler_bin=target_doppler,
        target_doppler_shifted=target_doppler_shifted,
        clutter_amplitude=CLUTTER_AMPLITUDE,
        clutter_range_bin=CLUTTER_RANGE_BIN,
        clutter_doppler_bin=clutter_doppler,
        clutter_doppler_shifted=clutter_doppler_shifted,
        noise_amplitude=NOISE_AMPLITUDE,
    )

    # 4. Range-Doppler Processing
    ref_input = (
        i_samples.astype(np.float64) + 1j * q_samples.astype(np.float64)
    ) / scaling_factor

    windowed_sig = ref_input * RANGE_WINDOW[None, :]
    range_fft = np.fft.fft(windowed_sig, axis=1)

    range_fft_windowed = range_fft * DOPPLER_WINDOW[:, None]
    doppler_fft = np.fft.fft(range_fft_windowed, axis=0)
    rd_map = np.fft.fftshift(doppler_fft, axes=0)
    power_map = np.abs(rd_map) ** 2

    # 5. Global Threshold Detector
    est_noise_power = float(np.median(power_map))
    fixed_threshold = float(THRESHOLD_FACTOR * est_noise_power)
    fixed_detections = power_map > fixed_threshold

    # 6. CA-CFAR Detector along Range
    cfar_detections = np.zeros((ND, NR), dtype=bool)
    cfar_threshold_map = np.full((ND, NR), np.nan)
    cfar_noise_map = np.full((ND, NR), np.nan)

    for d in range(ND):
        for r in range(EDGE_SIZE, NR - EDGE_SIZE):
            l_start = r - GUARD_CELLS_PER_SIDE - REFERENCE_CELLS_PER_SIDE
            l_stop = r - GUARD_CELLS_PER_SIDE
            r_start = r + GUARD_CELLS_PER_SIDE + 1
            r_stop = r_start + REFERENCE_CELLS_PER_SIDE

            ref_cells = np.concatenate((
                power_map[d, l_start:l_stop],
                power_map[d, r_start:r_stop]
            ))
            loc_noise = float(np.mean(ref_cells))
            loc_thr = float(CFAR_ALPHA * loc_noise)

            cfar_noise_map[d, r] = loc_noise
            cfar_threshold_map[d, r] = loc_thr
            cfar_detections[d, r] = (power_map[d, r] > loc_thr)

    # 7. Detection Metrics & Detection Margins
    target_power = float(power_map[target_doppler_shifted, target_range])
    cfar_target_threshold = float(cfar_threshold_map[target_doppler_shifted, target_range])

    global_margin_db = float(10.0 * np.log10(target_power / fixed_threshold))
    cfar_margin_db = float(10.0 * np.log10(target_power / cfar_target_threshold))

    threshold_inflation_ratio = cfar_target_threshold / NOMINAL_NOISE_CFAR_THRESHOLD
    threshold_inflation_db = float(10.0 * np.log10(threshold_inflation_ratio))

    # Exact-cell Pd
    global_exact_pd = bool(fixed_detections[target_doppler_shifted, target_range])
    cfar_exact_pd = bool(cfar_detections[target_doppler_shifted, target_range])

    # 3x3 Neighborhood Metrics
    d_lo = max(0, target_doppler_shifted - 1)
    d_hi = min(ND, target_doppler_shifted + 2)
    r_lo = max(0, target_range - 1)
    r_hi = min(NR, target_range + 2)

    # Raw 3x3
    global_3x3_raw_pd = bool(fixed_detections[d_lo:d_hi, r_lo:r_hi].any())
    cfar_3x3_raw_pd = bool(cfar_detections[d_lo:d_hi, r_lo:r_hi].any())

    # Clutter-excluded 3x3: exclude clutter core row/columns
    # 1. Any cell with r in [71, 72, 73] where r != target_range is clutter leakage, NOT the target
    # 2. Any cell on the zero-Doppler clutter row (d=16) in range [71..73] is the clutter peak
    sub_fixed = fixed_detections[d_lo:d_hi, r_lo:r_hi].copy()
    sub_cfar = cfar_detections[d_lo:d_hi, r_lo:r_hi].copy()

    for d_idx_rel, d_val in enumerate(range(d_lo, d_hi)):
        for r_idx_rel, r_val in enumerate(range(r_lo, r_hi)):
            if r_val in (71, 72, 73) and r_val != target_range:
                sub_fixed[d_idx_rel, r_idx_rel] = False
                sub_cfar[d_idx_rel, r_idx_rel] = False
            if d_val == 16 and r_val in (71, 72, 73):
                sub_fixed[d_idx_rel, r_idx_rel] = False
                sub_cfar[d_idx_rel, r_idx_rel] = False

    global_3x3_clutter_excl_pd = bool(sub_fixed.any())
    cfar_3x3_clutter_excl_pd = bool(sub_cfar.any())

    # Same-range Doppler straddle (target range bin only: d in [d_lo..d_hi], r = target_range)
    # strictly guarantees detection occurs at the simulated target range bin
    same_range_fixed = fixed_detections[d_lo:d_hi, target_range].copy()
    same_range_cfar = cfar_detections[d_lo:d_hi, target_range].copy()
    for d_idx_rel, d_val in enumerate(range(d_lo, d_hi)):
        if d_val == 16 and target_range in (71, 72, 73):
            same_range_fixed[d_idx_rel] = False
            same_range_cfar[d_idx_rel] = False

    global_same_range_pd = bool(same_range_fixed.any())
    cfar_same_range_pd = bool(same_range_cfar.any())

    # 8. False-Alarm Evaluation with Identical Exclusion Mask
    excl_mask = np.zeros((ND, NR), dtype=bool)
    # Target 3x3 exclusion
    excl_mask[d_lo:d_hi, r_lo:r_hi] = True
    # Clutter ridge exclusion (Doppler 12..22, Range 71..73)
    excl_mask[12:23, 71:74] = True

    eval_mask = CFAR_VALID_MASK & ~excl_mask
    n_eval = int(eval_mask.sum())

    g_false = int((fixed_detections & eval_mask).sum())
    c_false = int((cfar_detections & eval_mask).sum())

    global_pfa = float(g_false / n_eval)
    cfar_pfa = float(c_false / n_eval)

    # 9. Save Summary Text
    summary_text = "\n".join([
        "=== EXTENDED VALIDATION 1: CLUTTER-PROXIMITY SIMULATION SUMMARY ===",
        f"Target Range Bin: {target_range} (delta_r={target_range - CLUTTER_RANGE_BIN:+d} wrt clutter 72)",
        f"Target Doppler Bin: {target_doppler:.1f} (shifted: {target_doppler_shifted})",
        f"Clutter Doppler Bin: {clutter_doppler:.1f} (shifted: {clutter_doppler_shifted})",
        f"Random Seed: {seed}",
        f"Target Power: {target_power:.2f}",
        f"Global Threshold: {fixed_threshold:.2f} (Margin: {global_margin_db:+.2f} dB)",
        f"CFAR Threshold at Target: {cfar_target_threshold:.2f} (Margin: {cfar_margin_db:+.2f} dB)",
        f"Threshold Inflation: {threshold_inflation_ratio:.2f}x ({threshold_inflation_db:+.2f} dB)",
        f"Exact Cell Detected: Global={global_exact_pd}, CFAR={cfar_exact_pd}",
        f"Same-Range Doppler Detected: Global={global_same_range_pd}, CFAR={cfar_same_range_pd}",
        f"3x3 Clutter-Excluded Detected: Global={global_3x3_clutter_excl_pd}, CFAR={cfar_3x3_clutter_excl_pd}",
        f"3x3 Raw Detected: Global={global_3x3_raw_pd}, CFAR={cfar_3x3_raw_pd}",
        f"False Alarms (eval_cells={n_eval}): Global={g_false} ({global_pfa*100:.3f}%), CFAR={c_false} ({cfar_pfa*100:.3f}%)",
    ])
    (output_dir / "simulation_summary.txt").write_text(summary_text, encoding="utf-8")

    # 10. Generate Plots (if requested)
    if save_plots:
        # Range-Doppler Map
        fig, ax = plt.subplots(figsize=(7, 4.5))
        im = ax.imshow(
            10.0 * np.log10(power_map + 1e-12),
            aspect="auto",
            origin="lower",
            cmap="viridis",
            extent=[0, NR - 1, -ND // 2, ND // 2 - 1],
        )
        ax.set_title(f"Range-Doppler Map (Rt={target_range}, Dt=+{target_doppler:.1f}, Dc={clutter_doppler:.1f}, Seed={seed})")
        ax.set_xlabel("Range Bin")
        ax.set_ylabel("Doppler Bin")
        fig.colorbar(im, ax=ax, label="Power [dB]")
        fig.tight_layout()
        fig.savefig(output_dir / "range_doppler_map.png", dpi=140)
        plt.close(fig)

        # Global Detection Map
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.imshow(fixed_detections, aspect="auto", origin="lower", cmap="gray_r", extent=[0, NR - 1, -ND // 2, ND // 2 - 1])
        ax.set_title(f"Global Detection Map (Exact={global_exact_pd}, Margin={global_margin_db:+.2f} dB)")
        ax.set_xlabel("Range Bin")
        ax.set_ylabel("Doppler Bin")
        fig.tight_layout()
        fig.savefig(output_dir / "fixed_threshold_detection.png", dpi=140)
        plt.close(fig)

        # CFAR Detection Map
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.imshow(cfar_detections, aspect="auto", origin="lower", cmap="gray_r", extent=[0, NR - 1, -ND // 2, ND // 2 - 1])
        ax.set_title(f"CA-CFAR Detection Map (Exact={cfar_exact_pd}, Margin={cfar_margin_db:+.2f} dB)")
        ax.set_xlabel("Range Bin")
        ax.set_ylabel("Doppler Bin")
        fig.tight_layout()
        fig.savefig(output_dir / "cfar_detection.png", dpi=140)
        plt.close(fig)

    # Verify I/Q integrity
    line_cnt = len(iq_data)
    iq_valid = bool(
        (iq_data >= ADC_MIN).all() and (iq_data <= ADC_MAX).all() and line_cnt == NR * ND
    )

    return {
        "target_range_bin": target_range,
        "delta_r_clutter": target_range - CLUTTER_RANGE_BIN,
        "target_doppler_bin": target_doppler,
        "target_doppler_shifted": target_doppler_shifted,
        "clutter_doppler_bin": clutter_doppler,
        "clutter_doppler_shifted": clutter_doppler_shifted,
        "seed": seed,
        "scaling_factor": scaling_factor,
        "target_power": target_power,
        "global_threshold": fixed_threshold,
        "cfar_target_threshold": cfar_target_threshold,
        "global_margin_db": global_margin_db,
        "cfar_margin_db": cfar_margin_db,
        "threshold_inflation_ratio": threshold_inflation_ratio,
        "threshold_inflation_db": threshold_inflation_db,
        "global_exact_pd": int(global_exact_pd),
        "cfar_exact_pd": int(cfar_exact_pd),
        "global_same_range_pd": int(global_same_range_pd),
        "cfar_same_range_pd": int(cfar_same_range_pd),
        "global_3x3_raw_pd": int(global_3x3_raw_pd),
        "cfar_3x3_raw_pd": int(cfar_3x3_raw_pd),
        "global_3x3_clutter_excl_pd": int(global_3x3_clutter_excl_pd),
        "cfar_3x3_clutter_excl_pd": int(cfar_3x3_clutter_excl_pd),
        "eval_cells": n_eval,
        "global_false_alarms": g_false,
        "cfar_false_alarms": c_false,
        "global_pfa": global_pfa,
        "cfar_pfa": cfar_pfa,
        "iq_lines": line_cnt,
        "iq_valid_12bit": iq_valid,
        "output_dir": str(output_dir).replace("\\", "/"),
    }


def run_single_worker(args):
    return run_single_simulation(*args)


def main():
    parser = argparse.ArgumentParser(description="Extended Validation 1: Direct Clutter-Proximity Sweep")
    parser.add_argument("--pilot", action="store_true", help="Run 4-run pilot experiment first")
    parser.add_argument("--full", action="store_true", help="Run complete 680-run sweep")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1), help="Number of parallel workers")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent

    if args.pilot:
        print("=== RUNNING 4-RUN PILOT SIMULATION ===")
        pilot_configs = [
            (64, 1.0, 0.0, 1),  # Distant from clutter (dr = -8)
            (70, 1.0, 0.0, 1),  # Inside clutter masking zone (dr = -2)
            (72, 2.0, 0.3, 1),  # Co-located with clutter range (dr = 0)
            (76, 2.0, 0.3, 1),  # Symmetrical right masking zone (dr = +4)
        ]
        pilot_dir = base_dir / "outputs_pilot"
        pilot_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        results = []
        for r_bin, t_dop, c_dop, seed in pilot_configs:
            t_tag = f"{t_dop:.1f}".replace(".", "p")
            c_tag = f"{c_dop:.1f}".replace(".", "p")
            run_dir = pilot_dir / f"range_{r_bin:03d}_tdop_{t_tag}_cdop_{c_tag}_seed_{seed:02d}"
            res = run_single_simulation(r_bin, t_dop, c_dop, seed, run_dir, save_plots=True)
            results.append(res)
            print(
                f"Pilot run: Range={r_bin:2d} (dr={r_bin-72:+d}), Tdop=+{t_dop:.1f}, Cdop={c_dop:.1f}, Seed={seed:02d} | "
                f"CFAR Exact={res['cfar_exact_pd']}, 3x3_Excl={res['cfar_3x3_clutter_excl_pd']} | "
                f"Margin={res['cfar_margin_db']:+.2f} dB, Inflation={res['threshold_inflation_ratio']:.1f}x ({res['threshold_inflation_db']:+.2f} dB) | "
                f"IQ 12-bit Valid={res['iq_valid_12bit']} (16384 lines)"
            )

        print(f"\nPilot completed in {time.time() - t0:.2f}s.")
        pilot_csv = pilot_dir / "pilot_results.csv"
        with open(pilot_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        print(f"Saved pilot results: {pilot_csv}")
        return

    if not args.full:
        print("Please specify either --pilot or --full.")
        sys.exit(1)

    print("=== RUNNING FULL 680-RUN CLUTTER-PROXIMITY SWEEP ===")
    out_root = base_dir / "outputs_680_clutter_proximity"
    out_root.mkdir(parents=True, exist_ok=True)

    tasks = []
    for r_bin in TARGET_RANGE_BINS:
        for t_dop in TARGET_DOPPLERS:
            for c_dop in CLUTTER_DOPPLERS:
                t_tag = f"{t_dop:.1f}".replace(".", "p")
                c_tag = f"{c_dop:.1f}".replace(".", "p")
                for seed in SEEDS:
                    run_dir = (
                        out_root
                        / f"range_{r_bin:03d}_tdop_{t_tag}_cdop_{c_tag}"
                        / f"seed_{seed:02d}"
                    )
                    # For bulk runs, save plots for representative seed_01 only to optimize size
                    save_plot = (seed == 1)
                    tasks.append((r_bin, t_dop, c_dop, seed, run_dir, save_plot))

    print(f"Total simulations to execute: {len(tasks)}")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        all_results = list(pool.map(run_single_worker, tasks))

    print(f"Completed {len(all_results)} simulations in {time.time() - t0:.1f}s.")

    # Save consolidated results CSV
    cons_csv = base_dir / "clutter_proximity_sweep_consolidated_results.csv"
    with open(cons_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
        w.writeheader()
        w.writerows(all_results)
    print(f"Saved consolidated CSV: {cons_csv}")

    # Compute summary grouped by (range_bin, target_doppler, clutter_doppler)
    summary_rows = []
    for r_bin in TARGET_RANGE_BINS:
        for t_dop in TARGET_DOPPLERS:
            for c_dop in CLUTTER_DOPPLERS:
                sub = [
                    r for r in all_results
                    if r["target_range_bin"] == r_bin
                    and r["target_doppler_bin"] == t_dop
                    and r["clutter_doppler_bin"] == c_dop
                ]
                summary_rows.append({
                    "target_range_bin": r_bin,
                    "delta_r_clutter": r_bin - CLUTTER_RANGE_BIN,
                    "target_doppler_bin": t_dop,
                    "clutter_doppler_bin": c_dop,
                    "runs": len(sub),
                    "global_exact_pd": float(np.mean([r["global_exact_pd"] for r in sub])),
                    "cfar_exact_pd": float(np.mean([r["cfar_exact_pd"] for r in sub])),
                    "global_same_range_pd": float(np.mean([r["global_same_range_pd"] for r in sub])),
                    "cfar_same_range_pd": float(np.mean([r["cfar_same_range_pd"] for r in sub])),
                    "global_3x3_clutter_excl_pd": float(np.mean([r["global_3x3_clutter_excl_pd"] for r in sub])),
                    "cfar_3x3_clutter_excl_pd": float(np.mean([r["cfar_3x3_clutter_excl_pd"] for r in sub])),
                    "global_3x3_raw_pd": float(np.mean([r["global_3x3_raw_pd"] for r in sub])),
                    "cfar_3x3_raw_pd": float(np.mean([r["cfar_3x3_raw_pd"] for r in sub])),
                    "global_margin_db_mean": float(np.mean([r["global_margin_db"] for r in sub])),
                    "cfar_margin_db_mean": float(np.mean([r["cfar_margin_db"] for r in sub])),
                    "cfar_margin_db_min": float(np.min([r["cfar_margin_db"] for r in sub])),
                    "cfar_margin_db_max": float(np.max([r["cfar_margin_db"] for r in sub])),
                    "threshold_inflation_ratio_mean": float(np.mean([r["threshold_inflation_ratio"] for r in sub])),
                    "threshold_inflation_db_mean": float(np.mean([r["threshold_inflation_db"] for r in sub])),
                    "global_pfa_mean": float(np.mean([r["global_pfa"] for r in sub])),
                    "cfar_pfa_mean": float(np.mean([r["cfar_pfa"] for r in sub])),
                })

    sum_csv = base_dir / "clutter_proximity_sweep_summary_metrics.csv"
    with open(sum_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"Saved summary metrics CSV: {sum_csv}")

    # Generate Analysis Plots
    # 1. Detection Probability vs Range
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    configs = [
        (1.0, 0.0, "Target Dop +1.0, Clutter Dop 0.0"),
        (1.0, 0.3, "Target Dop +1.0, Clutter Dop 0.3"),
        (2.0, 0.0, "Target Dop +2.0, Clutter Dop 0.0"),
        (2.0, 0.3, "Target Dop +2.0, Clutter Dop 0.3"),
    ]
    for ax, (t_dop, c_dop, title) in zip(axes.flat, configs):
        sub = [s for s in summary_rows if s["target_doppler_bin"] == t_dop and s["clutter_doppler_bin"] == c_dop]
        rngs = [s["target_range_bin"] for s in sub]
        ax.plot(rngs, [s["cfar_exact_pd"] for s in sub], "o-", color="#d62728", lw=2, label="CA-CFAR Exact Pd")
        ax.plot(rngs, [s["cfar_3x3_clutter_excl_pd"] for s in sub], "s--", color="#ff7f0e", lw=1.5, label="CA-CFAR 3x3 (Clutter Excl)")
        ax.plot(rngs, [s["global_exact_pd"] for s in sub], "^-.", color="#1f77b4", lw=1.5, label="Global Exact Pd")
        ax.axvline(CLUTTER_RANGE_BIN, color="gray", linestyle=":", lw=1.2, label="Clutter Center (r=72)")
        ax.axvspan(65, 79, color="red", alpha=0.1, label="CFAR Degraded Zone (|dr|<=7)")
        ax.set_title(title)
        ax.set_xlabel("Target Range Bin")
        ax.set_ylabel("Detection Probability (Pd)")
        ax.set_ylim(-0.05, 1.1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Extended Validation 1: Direct Clutter-Proximity Detection Probability (680 Runs)", fontsize=13)
    fig.tight_layout()
    plot_pd_path = base_dir / "clutter_proximity_pd_vs_range.png"
    fig.savefig(plot_pd_path, dpi=180)
    plt.close(fig)
    print(f"Saved Pd plot: {plot_pd_path}")

    # 2. CFAR Detection Margin vs Range
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    for ax, (t_dop, c_dop, title) in zip(axes.flat, configs):
        sub = [s for s in summary_rows if s["target_doppler_bin"] == t_dop and s["clutter_doppler_bin"] == c_dop]
        rngs = [s["target_range_bin"] for s in sub]
        margins = [s["cfar_margin_db_mean"] for s in sub]
        ax.plot(rngs, margins, "o-", color="#2ca02c", lw=2, label="CA-CFAR Margin (mean)")
        ax.axhline(0.0, color="red", linestyle="--", lw=1.2, label="Detection Boundary (0 dB)")
        ax.axvline(CLUTTER_RANGE_BIN, color="gray", linestyle=":", lw=1.2, label="Clutter Center (r=72)")
        ax.axvspan(65, 79, color="red", alpha=0.1)
        ax.set_title(title)
        ax.set_xlabel("Target Range Bin")
        ax.set_ylabel("CFAR Margin [dB]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="lower right")

    fig.suptitle("Extended Validation 1: CA-CFAR Detection Margin vs Range (680 Runs)", fontsize=13)
    fig.tight_layout()
    plot_margin_path = base_dir / "clutter_proximity_cfar_margin_vs_range.png"
    fig.savefig(plot_margin_path, dpi=180)
    plt.close(fig)
    print(f"Saved margin plot: {plot_margin_path}")

    # 3. Threshold Inflation vs Range
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for t_dop, c_dop, label, color in [
        (1.0, 0.0, "Tdop +1.0, Cdop 0.0", "#1f77b4"),
        (1.0, 0.3, "Tdop +1.0, Cdop 0.3", "#ff7f0e"),
        (2.0, 0.0, "Tdop +2.0, Cdop 0.0", "#2ca02c"),
        (2.0, 0.3, "Tdop +2.0, Cdop 0.3", "#d62728"),
    ]:
        sub = [s for s in summary_rows if s["target_doppler_bin"] == t_dop and s["clutter_doppler_bin"] == c_dop]
        rngs = [s["target_range_bin"] for s in sub]
        inflat = [s["threshold_inflation_db_mean"] for s in sub]
        ax.plot(rngs, inflat, "o-", color=color, lw=2, label=label)

    ax.axvline(CLUTTER_RANGE_BIN, color="gray", linestyle=":", lw=1.2, label="Clutter Center (r=72)")
    ax.axvspan(65, 79, color="red", alpha=0.08, label="Contaminated Region (|dr|<=7)")
    ax.set_title("CA-CFAR Threshold Inflation [dB] Relative to Clean Noise Baseline")
    ax.set_xlabel("Target Range Bin")
    ax.set_ylabel("Threshold Inflation [dB]")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    plot_inflat_path = base_dir / "clutter_proximity_threshold_inflation_vs_range.png"
    fig.savefig(plot_inflat_path, dpi=180)
    plt.close(fig)
    print(f"Saved threshold inflation plot: {plot_inflat_path}")


if __name__ == "__main__":
    main()
