import os
os.environ["MPLBACKEND"] = "Agg"

import csv
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------
# 1. FIXED MODEL & DETECTOR PARAMETERS (UNCHANGED)
# ---------------------------------------------------------

NR = 512
ND = 32

TARGET_AMPLITUDE = 0.03
CLUTTER_RANGE_BIN = 72
CLUTTER_AMPLITUDE = 0.60
NOISE_AMPLITUDE = 0.08

THRESHOLD_FACTOR = 6.0

REFERENCE_CELLS_PER_SIDE = 4
GUARD_CELLS_PER_SIDE = 2
CFAR_ALPHA = 6.0

ADC_BITS = 12
ADC_MAX = 2 ** (ADC_BITS - 1) - 1
ADC_MIN = -(2 ** (ADC_BITS - 1))

EDGE_SIZE = REFERENCE_CELLS_PER_SIDE + GUARD_CELLS_PER_SIDE

RANGE_BINS = [40, 70, 120, 200, 300, 450]
TARGET_DOPPLERS = [1.0, 2.0]
CLUTTER_DOPPLERS = [0.0, 0.3]
SEEDS = [1, 2, 3, 4, 5]

n = np.arange(NR)
m = np.arange(ND)
n_grid, m_grid = np.meshgrid(n, m)

RANGE_WINDOW = np.hanning(NR)
DOPPLER_WINDOW = np.hanning(ND)

CFAR_VALID_MASK = np.zeros((ND, NR), dtype=bool)
CFAR_VALID_MASK[:, EDGE_SIZE:NR - EDGE_SIZE] = True


def _compute_ca_cfar(power_map: np.ndarray):
    """Exact CA-CFAR matching radar_model.py loop arithmetic."""
    cfar_detections = np.zeros((ND, NR), dtype=bool)
    cfar_threshold_map = np.full((ND, NR), np.nan)
    cfar_noise_map = np.full((ND, NR), np.nan)

    # Vectorized over range_index in [EDGE_SIZE, NR - EDGE_SIZE)
    r_idx = np.arange(EDGE_SIZE, NR - EDGE_SIZE)
    left_offsets = np.arange(-GUARD_CELLS_PER_SIDE - REFERENCE_CELLS_PER_SIDE, -GUARD_CELLS_PER_SIDE)
    right_offsets = np.arange(GUARD_CELLS_PER_SIDE + 1, GUARD_CELLS_PER_SIDE + 1 + REFERENCE_CELLS_PER_SIDE)
    offsets = np.concatenate((left_offsets, right_offsets))  # length 8

    ref_window = power_map[:, r_idx[None, :] + offsets[:, None]]  # (ND, 8, N_valid)
    local_noise = np.mean(ref_window, axis=1)
    local_thr = CFAR_ALPHA * local_noise

    cfar_noise_map[:, EDGE_SIZE:NR - EDGE_SIZE] = local_noise
    cfar_threshold_map[:, EDGE_SIZE:NR - EDGE_SIZE] = local_thr
    cfar_detections[:, EDGE_SIZE:NR - EDGE_SIZE] = (
        power_map[:, EDGE_SIZE:NR - EDGE_SIZE] > local_thr
    )
    return cfar_detections, cfar_threshold_map, cfar_noise_map


def run_single_task(args: tuple) -> dict:
    """Execute one range-robustness simulation and save all outputs."""
    target_range_bin, target_doppler_bin, clutter_doppler_bin, seed, output_dir_str = args
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    # 1. Synthesize signal
    target_phase = 2 * np.pi * (
        target_range_bin * n_grid / NR
        + target_doppler_bin * m_grid / ND
    )
    target_signal = TARGET_AMPLITUDE * np.exp(1j * target_phase)

    clutter_phase = 2 * np.pi * (
        CLUTTER_RANGE_BIN * n_grid / NR
        + clutter_doppler_bin * m_grid / ND
    )
    clutter_signal = CLUTTER_AMPLITUDE * np.exp(1j * clutter_phase)

    noise = NOISE_AMPLITUDE * (
        np.random.randn(ND, NR)
        + 1j * np.random.randn(ND, NR)
    )
    received_signal = target_signal + clutter_signal + noise

    # 2. Signed 12-bit I/Q quantization
    maximum_value = max(
        np.max(np.abs(received_signal.real)),
        np.max(np.abs(received_signal.imag)),
    )
    scaling_factor = 0.90 * ADC_MAX / maximum_value if maximum_value > 0 else 1.0

    i_samples = np.clip(
        np.round(received_signal.real * scaling_factor),
        ADC_MIN,
        ADC_MAX,
    ).astype(np.int16)

    q_samples = np.clip(
        np.round(received_signal.imag * scaling_factor),
        ADC_MIN,
        ADC_MAX,
    ).astype(np.int16)

    reference_input = (
        i_samples.astype(np.float64)
        + 1j * q_samples.astype(np.float64)
    ) / scaling_factor

    # Target & clutter shifted coordinates
    target_doppler_shifted = int((target_doppler_bin + ND // 2) % ND)
    clutter_doppler_shifted = float((clutter_doppler_bin + ND // 2) % ND)

    # 3. Save FPGA I/Q input & metadata
    iq_data = np.column_stack((i_samples.ravel(), q_samples.ravel()))
    iq_path = output_dir / "radar_input_iq.txt"
    np.savetxt(iq_path, iq_data, fmt="%d")

    # Verify IQ validity right away
    iq_valid = (
        iq_data.shape == (NR * ND, 2)
        and int(iq_data.min()) >= ADC_MIN
        and int(iq_data.max()) <= ADC_MAX
    )

    np.savez(
        output_dir / "simulation_metadata.npz",
        nr=NR,
        nd=ND,
        adc_bits=ADC_BITS,
        scaling_factor=scaling_factor,
        target_range_bin=target_range_bin,
        target_doppler_bin=target_doppler_bin,
        target_doppler_shifted=target_doppler_shifted,
        clutter_range_bin=CLUTTER_RANGE_BIN,
        clutter_doppler_bin=clutter_doppler_bin,
        clutter_doppler_shifted=clutter_doppler_shifted,
        seed=seed,
    )

    # 4. 2D Hann-windowed FFT
    range_fft = np.fft.fft(reference_input * RANGE_WINDOW[None, :], axis=1)
    doppler_fft = np.fft.fft(range_fft * DOPPLER_WINDOW[:, None], axis=0)
    range_doppler_map = np.fft.fftshift(doppler_fft, axes=0)

    power_map = np.abs(range_doppler_map) ** 2
    power_map_db = 10 * np.log10(power_map + 1e-12)
    target_power = float(power_map[target_doppler_shifted, target_range_bin])

    # 5. Global detector
    estimated_noise_power = float(np.median(power_map))
    fixed_threshold = THRESHOLD_FACTOR * estimated_noise_power
    fixed_detections = power_map > fixed_threshold
    target_detected = bool(fixed_detections[target_doppler_shifted, target_range_bin])

    # 6. CA-CFAR detector
    cfar_detections, cfar_threshold_map, cfar_noise_map = _compute_ca_cfar(power_map)
    cfar_target_detected = bool(cfar_detections[target_doppler_shifted, target_range_bin])
    cfar_target_threshold = float(cfar_threshold_map[target_doppler_shifted, target_range_bin])

    # 7. Save reference CSV maps
    for filename, data in {
        "reference_power_map.csv": power_map,
        "reference_cfar_noise_map.csv": cfar_noise_map,
        "reference_cfar_threshold_map.csv": cfar_threshold_map,
    }.items():
        np.savetxt(output_dir / filename, data, delimiter=",")

    for filename, data in {
        "reference_global_detection_map.csv": fixed_detections,
        "reference_cfar_detection_map.csv": cfar_detections,
        "reference_cfar_valid_mask.csv": CFAR_VALID_MASK,
    }.items():
        np.savetxt(output_dir / filename, data.astype(int), delimiter=",", fmt="%d")

    # 8. Save PNG plots
    def save_map_figure(title: str, filename: str, detections=None) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        image = ax.imshow(power_map_db, aspect="auto", origin="lower", cmap="jet")
        fig.colorbar(image, ax=ax, label="Power (dB, arbitrary reference)")
        if detections is not None:
            d_idx, r_idx = np.where(detections)
            ax.scatter(
                r_idx, d_idx,
                facecolors="none", edgecolors="lime", s=65, linewidths=1.3,
                label="Cells above threshold",
            )
        ax.plot(
            target_range_bin, target_doppler_shifted,
            marker="x", color="white", markersize=11, markeredgewidth=2,
            linestyle="none", label="Expected moving target",
        )
        ax.plot(
            CLUTTER_RANGE_BIN, clutter_doppler_shifted,
            marker="+", color="magenta", markersize=11, markeredgewidth=2,
            linestyle="none", label="Expected clutter",
        )
        ax.set_xlabel("Range Bin")
        ax.set_ylabel("Shifted Doppler Index")
        ax.set_title(title)
        ax.legend(loc="upper right", fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=120)
        plt.close(fig)

    save_map_figure("Radar Range-Doppler Map: Target + Clutter + Noise", "range_doppler_map.png")
    save_map_figure("Simple Global-Threshold Detection", "fixed_threshold_detection.png", fixed_detections)
    save_map_figure("Adaptive CA-CFAR Detection", "cfar_detection.png", cfar_detections)

    summary_text = "\n".join([
        "RADAR SIMULATION SUMMARY",
        f"Frame: {ND} chirps x {NR} samples = {ND * NR} complex samples",
        (
            f"Target: range bin {target_range_bin}, signed Doppler bin {target_doppler_bin}, "
            f"shifted index {target_doppler_shifted}"
        ),
        (
            f"Clutter: range bin {CLUTTER_RANGE_BIN}, signed Doppler bin {clutter_doppler_bin}, "
            f"shifted coord {clutter_doppler_shifted:.2f}"
        ),
        f"Global detector: target={target_detected}, cells above threshold={int(fixed_detections.sum())}",
        f"CFAR detector: target={cfar_target_detected}, cells above threshold={int(cfar_detections.sum())}",
        f"Target power: {target_power:.6f}",
        f"Global threshold: {fixed_threshold:.6f}",
        f"CFAR threshold at target: {cfar_target_threshold:.6f}",
        f"FPGA input: {ADC_BITS}-bit signed I/Q, {ND * NR} lines, scaling factor={scaling_factor:.6f}",
        f"All files saved in: {output_dir}",
    ])
    (output_dir / "simulation_summary.txt").write_text(summary_text, encoding="utf-8")

    # 9. Common valid-cell evaluation mask:
    # Excludes target 3x3 around (target_doppler_shifted, target_range_bin)
    # AND clutter ridge d[12..22], r[71..73]
    excl_mask = np.zeros((ND, NR), dtype=bool)
    excl_mask[
        target_doppler_shifted - 1:target_doppler_shifted + 2,
        target_range_bin - 1:target_range_bin + 2,
    ] = True
    excl_mask[12:23, 71:74] = True

    eval_mask = CFAR_VALID_MASK & ~excl_mask
    n_eval = int(eval_mask.sum())

    g_false = int((fixed_detections & eval_mask).sum())
    c_false = int((cfar_detections & eval_mask).sum())

    # 3x3 target neighborhood (both raw and clutter-masked excluding r in [71..73] when adjacent)
    d_lo, d_hi = target_doppler_shifted - 1, target_doppler_shifted + 2
    r_lo, r_hi = target_range_bin - 1, target_range_bin + 2
    g_3x3 = bool(fixed_detections[d_lo:d_hi, r_lo:r_hi].any())
    c_3x3 = bool(cfar_detections[d_lo:d_hi, r_lo:r_hi].any())

    clutter_col_mask = np.zeros((ND, NR), dtype=bool)
    clutter_col_mask[12:23, 71:74] = True
    t_box_mask = np.zeros((ND, NR), dtype=bool)
    t_box_mask[d_lo:d_hi, r_lo:r_hi] = True
    t_noclutter_mask = t_box_mask & ~clutter_col_mask
    g_3x3_noclutter = bool((fixed_detections & t_noclutter_mask).any())
    c_3x3_noclutter = bool((cfar_detections & t_noclutter_mask).any())

    return {
        "target_range_bin": target_range_bin,
        "target_doppler_bin": target_doppler_bin,
        "target_doppler_shifted": target_doppler_shifted,
        "clutter_doppler_bin": clutter_doppler_bin,
        "seed": seed,
        "iq_lines": int(iq_data.shape[0]),
        "iq_min": int(iq_data.min()),
        "iq_max": int(iq_data.max()),
        "iq_valid_12bit": iq_valid,
        "eval_cells": n_eval,
        "target_power": target_power,
        "global_threshold": fixed_threshold,
        "cfar_threshold_at_target": cfar_target_threshold,
        "global_target_detected_exact": target_detected,
        "cfar_target_detected_exact": cfar_target_detected,
        "global_target_detected_3x3": g_3x3,
        "cfar_target_detected_3x3": c_3x3,
        "global_target_detected_3x3_noclutter": g_3x3_noclutter,
        "cfar_target_detected_3x3_noclutter": c_3x3_noclutter,
        "global_margin_db": float(10.0 * np.log10(target_power / fixed_threshold)),
        "cfar_margin_db": float(10.0 * np.log10(target_power / cfar_target_threshold)),
        "global_total_count": int(fixed_detections.sum()),
        "cfar_total_count": int(cfar_detections.sum()),
        "global_false_detections_eval": g_false,
        "cfar_false_detections_eval": c_false,
        "global_pfa_eval": g_false / n_eval,
        "cfar_pfa_eval": c_false / n_eval,
    }


def main() -> None:
    base = Path(__file__).resolve().parent
    out_root = base / "outputs_512x32_range_robustness"
    out_root.mkdir(parents=True, exist_ok=True)

    tasks = []
    for r_bin in RANGE_BINS:
        for t_dop in TARGET_DOPPLERS:
            for c_dop in CLUTTER_DOPPLERS:
                t_tag = f"{t_dop:.1f}".replace(".", "p")
                c_tag = f"{c_dop:.1f}".replace(".", "p")
                for seed in SEEDS:
                    run_dir = out_root / f"range_{r_bin:03d}_tdop_{t_tag}_cdop_{c_tag}" / f"seed_{seed:02d}"
                    tasks.append((r_bin, t_dop, c_dop, seed, str(run_dir)))

    t0 = time.time()
    max_workers = min(4, os.cpu_count() or 1)
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        rows = list(pool.map(run_single_task, tasks))

    # Verify IQ files on disk
    bad_iq = []
    for r in rows:
        t_tag = f"{r['target_doppler_bin']:.1f}".replace(".", "p")
        c_tag = f"{r['clutter_doppler_bin']:.1f}".replace(".", "p")
        iq_file = out_root / f"range_{r['target_range_bin']:03d}_tdop_{t_tag}_cdop_{c_tag}" / f"seed_{r['seed']:02d}" / "radar_input_iq.txt"
        line_cnt = sum(1 for _ in open(iq_file, encoding="utf-8"))
        if line_cnt != 16384 or not r["iq_valid_12bit"]:
            bad_iq.append(str(iq_file))

    print(f"Completed {len(rows)} simulations in {time.time() - t0:.1f}s (bad IQ files: {len(bad_iq)})")

    # Save consolidated CSV
    csv_path = out_root / "range_robustness_consolidated_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Compute summary metrics per (range_bin, target_doppler, clutter_doppler)
    summary_rows = []
    for r_bin in RANGE_BINS:
        for t_dop in TARGET_DOPPLERS:
            for c_dop in CLUTTER_DOPPLERS:
                sub = [
                    r for r in rows
                    if r["target_range_bin"] == r_bin
                    and r["target_doppler_bin"] == t_dop
                    and r["clutter_doppler_bin"] == c_dop
                ]
                summary_rows.append({
                    "target_range_bin": r_bin,
                    "target_doppler_bin": t_dop,
                    "target_doppler_shifted": sub[0]["target_doppler_shifted"],
                    "clutter_doppler_bin": c_dop,
                    "runs": len(sub),
                    "eval_cells": sub[0]["eval_cells"],
                    "global_pd_exact": np.mean([r["global_target_detected_exact"] for r in sub]),
                    "cfar_pd_exact": np.mean([r["cfar_target_detected_exact"] for r in sub]),
                    "global_pd_3x3": np.mean([r["global_target_detected_3x3"] for r in sub]),
                    "cfar_pd_3x3": np.mean([r["cfar_target_detected_3x3"] for r in sub]),
                    "global_pd_3x3_noclutter": np.mean([r["global_target_detected_3x3_noclutter"] for r in sub]),
                    "cfar_pd_3x3_noclutter": np.mean([r["cfar_target_detected_3x3_noclutter"] for r in sub]),
                    "global_margin_db_mean": np.mean([r["global_margin_db"] for r in sub]),
                    "cfar_margin_db_mean": np.mean([r["cfar_margin_db"] for r in sub]),
                    "cfar_margin_db_min": np.min([r["cfar_margin_db"] for r in sub]),
                    "cfar_margin_db_max": np.max([r["cfar_margin_db"] for r in sub]),
                    "global_false_mean": np.mean([r["global_false_detections_eval"] for r in sub]),
                    "cfar_false_mean": np.mean([r["cfar_false_detections_eval"] for r in sub]),
                    "global_pfa_mean": np.mean([r["global_pfa_eval"] for r in sub]),
                    "global_pfa_std": np.std([r["global_pfa_eval"] for r in sub], ddof=1),
                    "cfar_pfa_mean": np.mean([r["cfar_pfa_eval"] for r in sub]),
                    "cfar_pfa_std": np.std([r["cfar_pfa_eval"] for r in sub], ddof=1),
                })

    sum_path = out_root / "range_robustness_summary_metrics.csv"
    with open(sum_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    # Print concise summary table
    for s in summary_rows:
        print(
            f"R={s['target_range_bin']:3d} tdop={s['target_doppler_bin']:.1f} cdop={s['clutter_doppler_bin']:.1f}: "
            f"Pd_exact(G={s['global_pd_exact']:.2f}, C={s['cfar_pd_exact']:.2f}) "
            f"Pd_3x3_noclut(G={s['global_pd_3x3_noclutter']:.2f}, C={s['cfar_pd_3x3_noclutter']:.2f}) "
            f"margin(G={s['global_margin_db_mean']:+.2f}, C={s['cfar_margin_db_mean']:+.2f}) "
            f"Pfa(G={100*s['global_pfa_mean']:.3f}%, C={100*s['cfar_pfa_mean']:.3f}%)"
        )

    # --- Plot 1: Exact-Cell Pd & CFAR Margin vs Range Bin ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5), sharex=True)
    x = np.arange(len(RANGE_BINS))
    for col_idx, t_dop in enumerate(TARGET_DOPPLERS):
        ax_pd = axes[0, col_idx]
        ax_mg = axes[1, col_idx]
        for c_dop, ls, mk in [(0.0, "-", "o"), (0.3, "--", "s")]:
            sub = [s for s in summary_rows if s["target_doppler_bin"] == t_dop and s["clutter_doppler_bin"] == c_dop]
            ax_pd.plot(x, [s["global_pd_exact"] for s in sub], linestyle=ls, marker=mk, label=f"Global (clut {c_dop})")
            ax_pd.plot(x, [s["cfar_pd_exact"] for s in sub], linestyle=ls, marker=mk, label=f"CA-CFAR (clut {c_dop})")
            ax_mg.plot(x, [s["global_margin_db_mean"] for s in sub], linestyle=ls, marker=mk, label=f"Global (clut {c_dop})")
            ax_mg.plot(x, [s["cfar_margin_db_mean"] for s in sub], linestyle=ls, marker=mk, label=f"CA-CFAR (clut {c_dop})")

        ax_pd.set_title(f"Exact-Cell Pd vs Range Bin (Target Doppler = +{t_dop:.1f})")
        ax_pd.set_ylabel("Empirical Pd (5 seeds)")
        ax_pd.set_ylim(-0.08, 1.15)
        ax_pd.grid(True, alpha=0.3)
        ax_pd.legend(loc="lower right", fontsize=8)

        ax_mg.axhline(0.0, color="red", linestyle=":", linewidth=1.2, label="0 dB threshold")
        ax_mg.set_title(f"Detection Margin vs Range Bin (Target Doppler = +{t_dop:.1f})")
        ax_mg.set_ylabel("Mean Margin [dB]")
        ax_mg.set_xlabel("Target Range Bin (Clutter fixed at bin 72)")
        ax_mg.set_xticks(x)
        ax_mg.set_xticklabels([str(r) for r in RANGE_BINS])
        ax_mg.grid(True, alpha=0.3)
        ax_mg.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_root / "range_robustness_pd_margin.png", dpi=180)
    plt.close(fig)

    # --- Plot 2: Background Pfa vs Range Bin ---
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), sharey=True)
    for col_idx, t_dop in enumerate(TARGET_DOPPLERS):
        ax = axes[col_idx]
        for c_dop, ls, mk in [(0.0, "-", "o"), (0.3, "--", "s")]:
            sub = [s for s in summary_rows if s["target_doppler_bin"] == t_dop and s["clutter_doppler_bin"] == c_dop]
            ax.errorbar(
                x, [100 * s["global_pfa_mean"] for s in sub],
                yerr=[100 * s["global_pfa_std"] for s in sub],
                linestyle=ls, marker=mk, capsize=3, label=f"Global (clut {c_dop})",
            )
            ax.errorbar(
                x, [100 * s["cfar_pfa_mean"] for s in sub],
                yerr=[100 * s["cfar_pfa_std"] for s in sub],
                linestyle=ls, marker=mk, capsize=3, label=f"CA-CFAR (clut {c_dop})",
            )
        ax.axhline(100.0 / 64.0, color="gray", linestyle="--", linewidth=1.0, label="Global theory (1.563%)")
        ax.axhline(100.0 * (1 + 6.0 / 4.78) ** (-4.78), color="black", linestyle=":", linewidth=1.1, label="CFAR Hann theory (2.05%)")
        ax.set_title(f"Background Pfa vs Range Bin (Target Doppler = +{t_dop:.1f})")
        ax.set_xlabel("Target Range Bin")
        ax.set_xticks(x)
        ax.set_xticklabels([str(r) for r in RANGE_BINS])
        ax.set_ylim(0.8, 2.5)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=7.5)
    axes[0].set_ylabel("Empirical Pfa (%)")
    fig.tight_layout()
    fig.savefig(out_root / "range_robustness_pfa.png", dpi=180)
    plt.close(fig)

    print("Saved:", csv_path)
    print("Saved:", sum_path)
    print("Saved plots in:", out_root)


if __name__ == "__main__":
    main()

