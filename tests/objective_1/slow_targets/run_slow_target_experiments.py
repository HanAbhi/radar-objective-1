import os
os.environ["MPLBACKEND"] = "Agg"

import csv
import time
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

TARGET_RANGE_BIN = 70
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

n = np.arange(NR)
m = np.arange(ND)
n_grid, m_grid = np.meshgrid(n, m)

RANGE_WINDOW = np.hanning(NR)
DOPPLER_WINDOW = np.hanning(ND)

CFAR_VALID_MASK = np.zeros((ND, NR), dtype=bool)
CFAR_VALID_MASK[:, EDGE_SIZE:NR - EDGE_SIZE] = True


def run_single_simulation(
    experiment_name: str,
    target_doppler_bin: float,
    clutter_doppler_bin: float,
    seed: int,
    output_dir: Path,
) -> dict:
    """Run one 512x32 simulation without modifying radar_model.py."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    # Synthesize target, clutter, and noise
    target_phase = 2 * np.pi * (
        TARGET_RANGE_BIN * n_grid / NR
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

    # 12-bit I/Q quantization
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

    # Continuous shifted coordinates and nearest discrete FFT Doppler index
    target_doppler_shifted_coord = float((target_doppler_bin + ND // 2) % ND)
    clutter_doppler_shifted_coord = float((clutter_doppler_bin + ND // 2) % ND)
    # Round half up so 16.5 maps to 17 (target side of 0-Doppler); also record bin 16 separately
    target_doppler_idx = int(np.floor(target_doppler_shifted_coord + 0.5)) % ND

    # Save FPGA input & metadata
    iq_data = np.column_stack((i_samples.ravel(), q_samples.ravel()))
    np.savetxt(output_dir / "radar_input_iq.txt", iq_data, fmt="%d")

    np.savez(
        output_dir / "simulation_metadata.npz",
        nr=NR,
        nd=ND,
        adc_bits=ADC_BITS,
        scaling_factor=scaling_factor,
        target_range_bin=TARGET_RANGE_BIN,
        target_doppler_bin=target_doppler_bin,
        target_doppler_shifted=target_doppler_shifted_coord,
        target_doppler_idx=target_doppler_idx,
        clutter_range_bin=CLUTTER_RANGE_BIN,
        clutter_doppler_bin=clutter_doppler_bin,
        clutter_doppler_shifted=clutter_doppler_shifted_coord,
        seed=seed,
    )

    # 2D Hann-windowed FFT
    range_fft = np.fft.fft(reference_input * RANGE_WINDOW[None, :], axis=1)
    doppler_fft = np.fft.fft(range_fft * DOPPLER_WINDOW[:, None], axis=0)
    range_doppler_map = np.fft.fftshift(doppler_fft, axes=0)

    power_map = np.abs(range_doppler_map) ** 2
    power_map_db = 10 * np.log10(power_map + 1e-12)

    target_power = float(power_map[target_doppler_idx, TARGET_RANGE_BIN])
    bin16_power = float(power_map[16, TARGET_RANGE_BIN])

    # Global threshold detector
    estimated_noise_power = float(np.median(power_map))
    fixed_threshold = THRESHOLD_FACTOR * estimated_noise_power
    fixed_detections = power_map > fixed_threshold
    target_detected = bool(fixed_detections[target_doppler_idx, TARGET_RANGE_BIN])

    # CA-CFAR detector along range
    cfar_detections = np.zeros((ND, NR), dtype=bool)
    cfar_threshold_map = np.full((ND, NR), np.nan)
    cfar_noise_map = np.full((ND, NR), np.nan)

    for doppler_index in range(ND):
        for range_index in range(EDGE_SIZE, NR - EDGE_SIZE):
            left_start = range_index - GUARD_CELLS_PER_SIDE - REFERENCE_CELLS_PER_SIDE
            left_stop = range_index - GUARD_CELLS_PER_SIDE
            right_start = range_index + GUARD_CELLS_PER_SIDE + 1
            right_stop = right_start + REFERENCE_CELLS_PER_SIDE

            reference_cells = np.concatenate((
                power_map[doppler_index, left_start:left_stop],
                power_map[doppler_index, right_start:right_stop],
            ))
            local_noise_estimate = float(np.mean(reference_cells))
            local_threshold = CFAR_ALPHA * local_noise_estimate

            cfar_noise_map[doppler_index, range_index] = local_noise_estimate
            cfar_threshold_map[doppler_index, range_index] = local_threshold
            cfar_detections[doppler_index, range_index] = (
                power_map[doppler_index, range_index] > local_threshold
            )

    cfar_target_detected = bool(cfar_detections[target_doppler_idx, TARGET_RANGE_BIN])
    cfar_target_threshold = float(cfar_threshold_map[target_doppler_idx, TARGET_RANGE_BIN])

    # Save reference CSVs
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

    # Save PNG figures
    def save_map_figure(title: str, filename: str, detections=None) -> None:
        fig, ax = plt.subplots(figsize=(12, 6))
        image = ax.imshow(power_map_db, aspect="auto", origin="lower", cmap="jet")
        fig.colorbar(image, ax=ax, label="Power (dB, arbitrary reference)")
        if detections is not None:
            d_idx, r_idx = np.where(detections)
            ax.scatter(
                r_idx, d_idx,
                facecolors="none", edgecolors="lime", s=80, linewidths=1.5,
                label="Cells above threshold",
            )
        ax.plot(
            TARGET_RANGE_BIN, target_doppler_shifted_coord,
            marker="x", color="white", markersize=12, markeredgewidth=2,
            linestyle="none", label="Expected moving target",
        )
        ax.plot(
            CLUTTER_RANGE_BIN, clutter_doppler_shifted_coord,
            marker="+", color="magenta", markersize=12, markeredgewidth=2,
            linestyle="none", label="Expected clutter",
        )
        ax.set_xlabel("Range Bin")
        ax.set_ylabel("Shifted Doppler Index")
        ax.set_title(title)
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=200)
        plt.close(fig)

    save_map_figure("Radar Range-Doppler Map: Target + Clutter + Noise", "range_doppler_map.png")
    save_map_figure("Simple Global-Threshold Detection", "fixed_threshold_detection.png", fixed_detections)
    save_map_figure("Adaptive CA-CFAR Detection", "cfar_detection.png", cfar_detections)

    summary_text = "\n".join([
        "RADAR SIMULATION SUMMARY",
        f"Experiment: {experiment_name} | seed={seed}",
        f"Frame: {ND} chirps x {NR} samples = {ND * NR} complex samples",
        (
            f"Target: range bin {TARGET_RANGE_BIN}, signed Doppler bin {target_doppler_bin}, "
            f"shifted coord {target_doppler_shifted_coord:.2f} (eval bin {target_doppler_idx})"
        ),
        (
            f"Clutter: range bin {CLUTTER_RANGE_BIN}, signed Doppler bin {clutter_doppler_bin}, "
            f"shifted coord {clutter_doppler_shifted_coord:.2f}"
        ),
        f"Global detector: target={target_detected}, cells above threshold={int(fixed_detections.sum())}",
        f"CFAR detector: target={cfar_target_detected}, cells above threshold={int(cfar_detections.sum())}",
        f"Target power (bin {target_doppler_idx}): {target_power:.6f}",
        f"Global threshold: {fixed_threshold:.6f}",
        f"CFAR threshold at target (bin {target_doppler_idx}): {cfar_target_threshold:.6f}",
        f"FPGA input: {ADC_BITS}-bit signed I/Q, {ND * NR} lines, scaling factor={scaling_factor:.6f}",
        f"All files saved in: {output_dir}",
    ])
    (output_dir / "simulation_summary.txt").write_text(summary_text, encoding="utf-8")

    # Identical 15,961-cell evaluation mask for false alarms:
    # Excludes target 3x3 around eval bin + clutter ridge d[12..22], r[71..73] (39 cells total)
    excl_mask = np.zeros((ND, NR), dtype=bool)
    excl_mask[target_doppler_idx - 1:target_doppler_idx + 2, TARGET_RANGE_BIN - 1:TARGET_RANGE_BIN + 2] = True
    excl_mask[12:23, 71:74] = True
    eval_mask = CFAR_VALID_MASK & ~excl_mask
    n_eval = int(eval_mask.sum())

    g_false = int((fixed_detections & eval_mask).sum())
    c_false = int((cfar_detections & eval_mask).sum())

    # 3x3 neighborhood around target_doppler_idx (raw) AND clutter-free range bins r in {69, 70}
    d_lo, d_hi = target_doppler_idx - 1, target_doppler_idx + 2
    g_3x3_raw = bool(fixed_detections[d_lo:d_hi, 69:72].any())
    c_3x3_raw = bool(cfar_detections[d_lo:d_hi, 69:72].any())
    g_3x3_noclutter = bool(fixed_detections[d_lo:d_hi, 69:71].any())
    c_3x3_noclutter = bool(cfar_detections[d_lo:d_hi, 69:71].any())

    g_bin16 = bool(fixed_detections[16, TARGET_RANGE_BIN])
    c_bin16 = bool(cfar_detections[16, TARGET_RANGE_BIN])
    cfar_thr_bin16 = float(cfar_threshold_map[16, TARGET_RANGE_BIN])

    return {
        "experiment": experiment_name,
        "target_doppler_bin": target_doppler_bin,
        "target_doppler_shifted_coord": target_doppler_shifted_coord,
        "eval_doppler_idx": target_doppler_idx,
        "clutter_doppler_bin": clutter_doppler_bin,
        "seed": seed,
        "eval_cells": n_eval,
        "target_power": target_power,
        "bin16_power": bin16_power,
        "global_threshold": fixed_threshold,
        "cfar_threshold_at_target": cfar_target_threshold,
        "cfar_threshold_at_bin16": cfar_thr_bin16,
        "global_target_detected_exact": target_detected,
        "cfar_target_detected_exact": cfar_target_detected,
        "global_bin16_detected": g_bin16,
        "cfar_bin16_detected": c_bin16,
        "global_target_detected_3x3_raw": g_3x3_raw,
        "cfar_target_detected_3x3_raw": c_3x3_raw,
        "global_target_detected_r69_70": g_3x3_noclutter,
        "cfar_target_detected_r69_70": c_3x3_noclutter,
        "global_margin_db": float(10.0 * np.log10(target_power / fixed_threshold)),
        "cfar_margin_db": float(10.0 * np.log10(target_power / cfar_target_threshold)),
        "cfar_margin_bin16_db": float(10.0 * np.log10(bin16_power / cfar_thr_bin16)),
        "global_total_count": int(fixed_detections.sum()),
        "cfar_total_count": int(cfar_detections.sum()),
        "global_false_detections_eval": g_false,
        "cfar_false_detections_eval": c_false,
        "global_pfa_eval": g_false / n_eval,
        "cfar_pfa_eval": c_false / n_eval,
    }


def main() -> None:
    base = Path(__file__).resolve().parent
    out_root = base
    out_root.mkdir(parents=True, exist_ok=True)

    configs = [
        ("T06_Slow_1p0", 1.0, 0.0, "T06_target_dop1p0_clutter_dop0p0"),
        ("T06_Slow_1p0", 1.0, 0.3, "T06_target_dop1p0_clutter_dop0p3"),
        ("T07_VerySlow_0p5", 0.5, 0.0, "T07_target_dop0p5_clutter_dop0p0"),
        ("T07_VerySlow_0p5", 0.5, 0.3, "T07_target_dop0p5_clutter_dop0p3"),
    ]

    t0 = time.time()
    all_rows = []
    run_idx = 0
    for exp_name, tdop, cdop, folder_name in configs:
        for seed in range(1, 11):
            run_dir = out_root / folder_name / f"seed_{seed:02d}"
            row = run_single_simulation(exp_name, tdop, cdop, seed, run_dir)
            all_rows.append(row)
            run_idx += 1
            print(
                f"[{run_idx}/40] {folder_name} seed={seed:02d}: "
                f"eval_d={row['eval_doppler_idx']} G_exact={row['global_target_detected_exact']} "
                f"C_exact={row['cfar_target_detected_exact']} "
                f"P_t={row['target_power']:.1f} C_thr={row['cfar_threshold_at_target']:.1f} "
                f"C_margin={row['cfar_margin_db']:+.2f}dB",
                flush=True,
            )

    # Save consolidated CSV
    csv_path = out_root / "slow_targets_consolidated_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    # Summary CSV
    summary_rows = []
    for exp_name, tdop, cdop, folder_name in configs:
        sub = [r for r in all_rows if r["experiment"] == exp_name and r["clutter_doppler_bin"] == cdop]
        summary_rows.append({
            "experiment": exp_name,
            "target_doppler_bin": tdop,
            "target_doppler_shifted_coord": sub[0]["target_doppler_shifted_coord"],
            "eval_doppler_idx": sub[0]["eval_doppler_idx"],
            "clutter_doppler_bin": cdop,
            "runs": len(sub),
            "eval_cells": sub[0]["eval_cells"],
            "global_pd_exact": np.mean([r["global_target_detected_exact"] for r in sub]),
            "cfar_pd_exact": np.mean([r["cfar_target_detected_exact"] for r in sub]),
            "global_pd_bin16": np.mean([r["global_bin16_detected"] for r in sub]),
            "cfar_pd_bin16": np.mean([r["cfar_bin16_detected"] for r in sub]),
            "global_pd_3x3_raw": np.mean([r["global_target_detected_3x3_raw"] for r in sub]),
            "cfar_pd_3x3_raw": np.mean([r["cfar_target_detected_3x3_raw"] for r in sub]),
            "global_pd_r69_70": np.mean([r["global_target_detected_r69_70"] for r in sub]),
            "cfar_pd_r69_70": np.mean([r["cfar_target_detected_r69_70"] for r in sub]),
            "global_margin_db_mean": np.mean([r["global_margin_db"] for r in sub]),
            "cfar_margin_db_mean": np.mean([r["cfar_margin_db"] for r in sub]),
            "cfar_margin_db_min": np.min([r["cfar_margin_db"] for r in sub]),
            "cfar_margin_db_max": np.max([r["cfar_margin_db"] for r in sub]),
            "global_false_mean": np.mean([r["global_false_detections_eval"] for r in sub]),
            "global_false_std": np.std([r["global_false_detections_eval"] for r in sub], ddof=1),
            "cfar_false_mean": np.mean([r["cfar_false_detections_eval"] for r in sub]),
            "cfar_false_std": np.std([r["cfar_false_detections_eval"] for r in sub], ddof=1),
            "global_pfa_mean": np.mean([r["global_pfa_eval"] for r in sub]),
            "global_pfa_std": np.std([r["global_pfa_eval"] for r in sub], ddof=1),
            "cfar_pfa_mean": np.mean([r["cfar_pfa_eval"] for r in sub]),
            "cfar_pfa_std": np.std([r["cfar_pfa_eval"] for r in sub], ddof=1),
        })

    sum_path = out_root / "slow_targets_summary_metrics.csv"
    with open(sum_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    # Comparison plots: Pd + Margin and Pfa
    labels = ["T06 (1.0)\nClut 0.0", "T06 (1.0)\nClut 0.3", "T07 (0.5)\nClut 0.0", "T07 (0.5)\nClut 0.3"]
    x = np.arange(len(labels))
    w_bar = 0.2

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    ax1.bar(x - 1.5*w_bar, [s["global_pd_exact"] for s in summary_rows], w_bar, label="Global (Exact eval bin)")
    ax1.bar(x - 0.5*w_bar, [s["cfar_pd_exact"] for s in summary_rows], w_bar, label="CA-CFAR (Exact eval bin)")
    ax1.bar(x + 0.5*w_bar, [s["global_pd_r69_70"] for s in summary_rows], w_bar, label="Global (Target side r=69,70)")
    ax1.bar(x + 1.5*w_bar, [s["cfar_pd_r69_70"] for s in summary_rows], w_bar, label="CA-CFAR (Target side r=69,70)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylim(0, 1.18)
    ax1.set_ylabel("Detection Probability (Pd)")
    ax1.set_title("Slow-Target Pd (T06: 1.0, T07: 0.5; Weak Target 0.03)")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8)

    g_m = [[r["global_margin_db"] for r in all_rows if r["experiment"] == e and r["clutter_doppler_bin"] == c] for e, _, c, _ in configs]
    c_m = [[r["cfar_margin_db"] for r in all_rows if r["experiment"] == e and r["clutter_doppler_bin"] == c] for e, _, c, _ in configs]
    bp1 = ax2.boxplot(g_m, positions=x - 0.16, widths=0.28, patch_artist=True, boxprops=dict(facecolor="#4c72b0", alpha=0.7), medianprops=dict(color="black"))
    bp2 = ax2.boxplot(c_m, positions=x + 0.16, widths=0.28, patch_artist=True, boxprops=dict(facecolor="#dd8452", alpha=0.7), medianprops=dict(color="black"))
    ax2.axhline(0.0, color="red", linestyle="--", linewidth=1.2, label="Detection threshold (0 dB)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Target-to-Threshold Margin at Eval Bin [dB]")
    ax2.set_title("Slow-Target Detection Margin (Negative = Clutter Masked)")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend([bp1["boxes"][0], bp2["boxes"][0], ax2.lines[0]], ["Global margin", "CA-CFAR margin", "0 dB boundary"], loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_root / "slow_targets_pd_margin_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    bw = 0.34
    ax.bar(x - bw/2, [100*s["global_pfa_mean"] for s in summary_rows], bw, yerr=[100*s["global_pfa_std"] for s in summary_rows], capsize=4, label="Global Pfa (mean ± 1 SD)")
    ax.bar(x + bw/2, [100*s["cfar_pfa_mean"] for s in summary_rows], bw, yerr=[100*s["cfar_pfa_std"] for s in summary_rows], capsize=4, label="CA-CFAR Pfa (mean ± 1 SD)")
    ax.axhline(100.0 / 64.0, color="#1f77b4", linestyle="--", linewidth=1.1, label="Global theory (1.563%)")
    ax.axhline(100.0 * (1 + 6.0 / 4.78) ** (-4.78), color="#ff7f0e", linestyle=":", linewidth=1.3, label="CFAR Hann theory (2.05%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 2.55)
    ax.set_ylabel("Empirical False-Alarm Probability Pfa (%)")
    ax.set_title("Slow-Target Background Pfa (Identical 15,961-Cell Mask)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_root / "slow_targets_pfa_comparison.png", dpi=180)
    plt.close(fig)

    print(f"\nFinished 40 slow-target runs in {time.time() - t0:.1f}s")
    print("Saved:", csv_path)
    print("Saved:", sum_path)
    for s in summary_rows:
        print(
            f"{s['experiment']} clut={s['clutter_doppler_bin']}: "
            f"Pd_exact(G={s['global_pd_exact']:.2f}, C={s['cfar_pd_exact']:.2f}), "
            f"Pd_3x3_raw(G={s['global_pd_3x3_raw']:.2f}, C={s['cfar_pd_3x3_raw']:.2f}), "
            f"Pd_r69_70(G={s['global_pd_r69_70']:.2f}, C={s['cfar_pd_r69_70']:.2f}), "
            f"margin_dB(G={s['global_margin_db_mean']:+.2f}, C={s['cfar_margin_db_mean']:+.2f}), "
            f"Pfa(G={100*s['global_pfa_mean']:.4f}%, C={100*s['cfar_pfa_mean']:.4f}%)"
        )


if __name__ == "__main__":
    main()

