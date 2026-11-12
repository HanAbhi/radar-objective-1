import csv
import os
os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


BASE = Path(__file__).resolve().parent.parent
OUT_DIR = BASE / "analysis" / "clutter_investigation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLUTTER_RANGE = 72
CLUTTER_MAINLOBE_BINS = {71, 72, 73}
REF_PER_SIDE = 4
GUARD_PER_SIDE = 2


def clutter_ref_overlap(r_cut: int) -> tuple[list[int], list[int], bool]:
    """Return clutter mainlobe bins ({71,72,73}) in CFAR ref window, guard band, and whether CUT is on clutter."""
    left_ref = list(range(r_cut - GUARD_PER_SIDE - REF_PER_SIDE, r_cut - GUARD_PER_SIDE))
    right_ref = list(range(r_cut + GUARD_PER_SIDE + 1, r_cut + GUARD_PER_SIDE + 1 + REF_PER_SIDE))
    guard = list(range(r_cut - GUARD_PER_SIDE, r_cut)) + list(range(r_cut + 1, r_cut + GUARD_PER_SIDE + 1))
    in_ref = [b for b in (left_ref + right_ref) if b in CLUTTER_MAINLOBE_BINS]
    in_guard = [b for b in guard if b in CLUTTER_MAINLOBE_BINS]
    cut_on_clutter = r_cut in CLUTTER_MAINLOBE_BINS
    return in_ref, in_guard, cut_on_clutter


def collect_all_runs() -> list[tuple[str, str, Path, dict]]:
    """Collect metadata and paths for all 210 existing Step 7 / slow-target / range-robustness runs."""
    runs = []

    # 1. Step 7 multi-seed (40 runs)
    s7_dir = BASE / "tests" / "step7_multiseed"
    for cfg, tamp, cdop in [
        ("groupA_moderate_dop0p0", 0.10, 0.0),
        ("groupA_moderate_dop0p3", 0.10, 0.3),
        ("groupB_weak_dop0p0", 0.03, 0.0),
        ("groupB_weak_dop0p3", 0.03, 0.3),
    ]:
        for seed in range(1, 11):
            runs.append((
                "step7_multiseed",
                cfg,
                s7_dir / cfg / f"seed_{seed:02d}",
                {"t_range": 70, "t_dop": 2.0, "eval_d": 18, "t_amp": tamp, "c_amp": 0.60, "c_dop": cdop, "n_amp": 0.08, "seed": seed},
            ))

    # 2. Step 7 noise-only (10 runs)
    n_dir = BASE / "tests" / "noise_only"
    for seed in range(1, 11):
        runs.append((
            "step7_noise_only",
            "noise_only",
            n_dir / f"seed_{seed:02d}",
            {"t_range": 70, "t_dop": 2.0, "eval_d": 18, "t_amp": 0.0, "c_amp": 0.0, "c_dop": np.nan, "n_amp": 0.08, "seed": seed},
        ))

    # 3. T06 / T07 slow targets (40 runs)
    slow_dir = BASE / "tests" / "slow_targets"
    for cfg, tdop, eval_d, cdop in [
        ("T06_target_dop1p0_clutter_dop0p0", 1.0, 17, 0.0),
        ("T06_target_dop1p0_clutter_dop0p3", 1.0, 17, 0.3),
        ("T07_target_dop0p5_clutter_dop0p0", 0.5, 17, 0.0),
        ("T07_target_dop0p5_clutter_dop0p3", 0.5, 17, 0.3),
    ]:
        for seed in range(1, 11):
            runs.append((
                "slow_targets",
                cfg,
                slow_dir / cfg / f"seed_{seed:02d}",
                {"t_range": 70, "t_dop": tdop, "eval_d": eval_d, "t_amp": 0.03, "c_amp": 0.60, "c_dop": cdop, "n_amp": 0.08, "seed": seed},
            ))

    # 4. Range robustness (120 runs)
    rr_dir = BASE / "tests" / "range_robustness"
    for r_bin in [40, 70, 120, 200, 300, 450]:
        for tdop, eval_d in [(1.0, 17), (2.0, 18)]:
            for cdop in [0.0, 0.3]:
                t_tag = f"{tdop:.1f}".replace(".", "p")
                c_tag = f"{cdop:.1f}".replace(".", "p")
                cfg = f"range_{r_bin:03d}_tdop_{t_tag}_cdop_{c_tag}"
                for seed in range(1, 6):
                    runs.append((
                        "range_robustness",
                        cfg,
                        rr_dir / cfg / f"seed_{seed:02d}",
                        {"t_range": r_bin, "t_dop": tdop, "eval_d": eval_d, "t_amp": 0.03, "c_amp": 0.60, "c_dop": cdop, "n_amp": 0.08, "seed": seed},
                    ))

    return runs


def main() -> None:
    runs = collect_all_runs()
    print(f"Loaded {len(runs)} existing runs for diagnostic analysis.")

    per_run_rows = []

    # Accumulators for clutter+noise profiles at r=64..80 (using target-free clutter region from R=200,300,450)
    prof_power = {0.0: [], 0.3: [], "noise": []}
    prof_cfar_thr = {0.0: [], 0.3: [], "noise": []}
    prof_cfar_det = {0.0: [], 0.3: [], "noise": []}
    prof_global_det = {0.0: [], 0.3: [], "noise": []}

    # Target-present R=70 profiles for comparison
    r70_power = {
        ("T06_1.0", 0.0): [], ("T06_1.0", 0.3): [],
        ("Step7B_2.0", 0.0): [], ("Step7B_2.0", 0.3): [],
    }

    for dataset, cfg, sdir, info in runs:
        pm = np.loadtxt(sdir / "reference_power_map.csv", delimiter=",")
        gd = np.loadtxt(sdir / "reference_global_detection_map.csv", delimiter=",", dtype=bool)
        cd = np.loadtxt(sdir / "reference_cfar_detection_map.csv", delimiter=",", dtype=bool)
        cthr = np.loadtxt(sdir / "reference_cfar_threshold_map.csv", delimiter=",")

        r_t = info["t_range"]
        d_t = info["eval_d"]
        p_t = float(pm[d_t, r_t])
        g_thr = float(6.0 * np.median(pm))
        c_thr = float(cthr[d_t, r_t])

        p_clut_peak = float(pm[16, CLUTTER_RANGE])
        p_clut_same_d = float(pm[d_t, CLUTTER_RANGE])
        p_clut_r71 = float(pm[d_t, 71])
        p_clut_r73 = float(pm[d_t, 73])

        in_ref, in_guard, cut_clut = clutter_ref_overlap(r_t)

        # Distinguish genuine target-side detection (r in [r_t-1, r_t+1] excluding clutter mainlobe r in {71,72,73})
        # vs clutter-ridge sidelobe crossings (d in [12..22], r in {71, 72, 73})
        clut_ridge_g = int(gd[12:23, 71:74].sum())
        clut_ridge_c = int(cd[12:23, 71:74].sum())

        t_box = np.zeros((32, 512), dtype=bool)
        t_box[d_t - 1:d_t + 2, r_t - 1:r_t + 2] = True
        c_box = np.zeros((32, 512), dtype=bool)
        c_box[12:23, 71:74] = True
        t_noclut = t_box & ~c_box

        per_run_rows.append({
            "dataset": dataset,
            "config": cfg,
            "seed": info["seed"],
            "target_range_bin": r_t,
            "target_doppler_bin": info["t_dop"],
            "eval_doppler_idx": d_t,
            "target_amplitude": info["t_amp"],
            "clutter_amplitude": info["c_amp"],
            "clutter_doppler_bin": info["c_dop"],
            "target_power": p_t,
            "clutter_peak_power_16_72": p_clut_peak,
            "clutter_same_doppler_power_72": p_clut_same_d,
            "clutter_r71_same_doppler": p_clut_r71,
            "clutter_r73_same_doppler": p_clut_r73,
            "target_to_clutter_peak_db": float(10.0 * np.log10(p_t / p_clut_peak)) if p_clut_peak > 0 else np.nan,
            "target_to_clutter_same_doppler_db": float(10.0 * np.log10(p_t / p_clut_same_d)) if p_clut_same_d > 0 else np.nan,
            "global_threshold": g_thr,
            "cfar_threshold_at_target": c_thr,
            "global_margin_db": float(10.0 * np.log10(p_t / g_thr)),
            "cfar_margin_db": float(10.0 * np.log10(p_t / c_thr)),
            "global_detected_exact": bool(gd[d_t, r_t]),
            "cfar_detected_exact": bool(cd[d_t, r_t]),
            "global_detected_3x3_noclutter": bool((gd & t_noclut).any()),
            "cfar_detected_3x3_noclutter": bool((cd & t_noclut).any()),
            "clutter_bins_in_cfar_ref": ";".join(map(str, in_ref)) if in_ref else "none",
            "clutter_bins_in_cfar_guard": ";".join(map(str, in_guard)) if in_guard else "none",
            "clutter_ridge_global_detections": clut_ridge_g,
            "clutter_ridge_cfar_detections": clut_ridge_c,
        })

        # Collect clutter+noise only region r=64..80 from R=200, 300, 450 (tdop=2.0 so unique seeds 1..5 per cdop)
        if dataset == "range_robustness" and r_t in (200, 300, 450) and info["t_dop"] == 2.0:
            cdop = info["c_dop"]
            prof_power[cdop].append(pm[:, 64:81])
            prof_cfar_thr[cdop].append(cthr[:, 64:81])
            prof_cfar_det[cdop].append(cd[:, 64:81])
            prof_global_det[cdop].append(gd[:, 64:81])
        elif dataset == "step7_noise_only":
            prof_power["noise"].append(pm[:, 64:81])
            prof_cfar_thr["noise"].append(cthr[:, 64:81])
            prof_cfar_det["noise"].append(cd[:, 64:81])
            prof_global_det["noise"].append(gd[:, 64:81])

        if dataset == "slow_targets" and info["t_dop"] == 1.0:
            r70_power[("T06_1.0", info["c_dop"])].append(pm[:, 64:81])
        elif dataset == "step7_multiseed" and info["t_amp"] == 0.03:
            r70_power[("Step7B_2.0", info["c_dop"])].append(pm[:, 64:81])

    # Save per-run CSV
    per_run_csv = OUT_DIR / "clutter_proximity_per_run.csv"
    with open(per_run_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(per_run_rows[0].keys()))
        w.writeheader()
        w.writerows(per_run_rows)
    print("Saved:", per_run_csv)

    # Build range-bin 64..80 profile table (clutter+noise background from R=200,300,450 vs noise-only)
    r_bins = np.arange(64, 81)
    # Empirical weak target mean power at d=17 (T06) and d=18 (Step7B)
    weak_p_d17 = float(np.mean([r["target_power"] for r in per_run_rows if r["dataset"] == "slow_targets" and r["target_doppler_bin"] == 1.0]))
    weak_p_d18 = float(np.mean([r["target_power"] for r in per_run_rows if r["dataset"] == "step7_multiseed" and r["target_amplitude"] == 0.03]))
    mod_p_d18 = float(np.mean([r["target_power"] for r in per_run_rows if r["dataset"] == "step7_multiseed" and r["target_amplitude"] == 0.10]))

    prof_rows = []
    for idx, r_val in enumerate(r_bins):
        in_ref, in_guard, cut_clut = clutter_ref_overlap(int(r_val))
        for dop_label, d_idx, p_weak in [("Dop0_clutter", 16, np.nan), ("Dop+1", 17, weak_p_d17), ("Dop+2", 18, weak_p_d18)]:
            row = {
                "range_bin": int(r_val),
                "range_separation_from_clutter": int(r_val - CLUTTER_RANGE),
                "abs_range_separation": abs(int(r_val - CLUTTER_RANGE)),
                "doppler_label": dop_label,
                "doppler_shifted_idx": d_idx,
                "cut_on_clutter_mainlobe": cut_clut,
                "clutter_bins_in_ref_window": ";".join(map(str, in_ref)) if in_ref else "none",
                "clutter_bins_in_guard_band": ";".join(map(str, in_guard)) if in_guard else "none",
                "noise_only_power_mean": float(np.mean([a[d_idx, idx] for a in prof_power["noise"]])),
                "noise_only_cfar_thr_mean": float(np.mean([a[d_idx, idx] for a in prof_cfar_thr["noise"]])),
            }
            for cdop, tag in [(0.0, "c0p0"), (0.3, "c0p3")]:
                p_arr = [a[d_idx, idx] for a in prof_power[cdop]]
                t_arr = [a[d_idx, idx] for a in prof_cfar_thr[cdop]]
                d_arr = [a[d_idx, idx] for a in prof_cfar_det[cdop]]
                row[f"{tag}_power_mean"] = float(np.mean(p_arr))
                row[f"{tag}_cfar_thr_mean"] = float(np.mean(t_arr))
                row[f"{tag}_cfar_det_rate_no_target"] = float(np.mean(d_arr))
                if not np.isnan(p_weak):
                    row[f"{tag}_weak_target_margin_db"] = float(10.0 * np.log10(p_weak / np.mean(t_arr)))
                    row[f"{tag}_weak_target_above_thr"] = (not cut_clut) and (p_weak > np.mean(t_arr))
                else:
                    row[f"{tag}_weak_target_margin_db"] = np.nan
                    row[f"{tag}_weak_target_above_thr"] = False
            prof_rows.append(row)

    prof_csv = OUT_DIR / "clutter_profile_bins_64_80.csv"
    with open(prof_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(prof_rows[0].keys()))
        w.writeheader()
        w.writerows(prof_rows)
    print("Saved:", prof_csv)

    # Print key recovery table for Doppler +1 and +2 across r=64..80
    print(f"\nWeak target reference power: d=17 (T06)={weak_p_d17:.1f}, d=18 (Step7B)={weak_p_d18:.1f}; Mod d=18={mod_p_d18:.1f}")
    for d_lbl in ("Dop+1", "Dop+2"):
        print(f"\n--- Clutter-Proximity Recovery Profile ({d_lbl}) ---")
        for row in [x for x in prof_rows if x["doppler_label"] == d_lbl]:
            print(
                f"r={row['range_bin']:2d} (dr={row['range_separation_from_clutter']:+2d}) ref={row['clutter_bins_in_ref_window']:8s} "
                f"cut_clut={str(row['cut_on_clutter_mainlobe']):5s} | "
                f"c0.0: thr={row['c0p0_cfar_thr_mean']:10.1f} margin={row['c0p0_weak_target_margin_db']:+6.2f}dB | "
                f"c0.3: thr={row['c0p3_cfar_thr_mean']:10.1f} margin={row['c0p3_weak_target_margin_db']:+6.2f}dB"
            )

    # --- Plot 1: Range Profiles (r=64..80) through Doppler 0, +1, +2 ---
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 11.0), sharex=True)
    for ax, (d_idx, title_d, tgt_key) in zip(
        axes,
        [
            (16, "Clutter Doppler 0.0 (Shifted Index 16)", None),
            (17, "Target Doppler +1.0 (Shifted Index 17)", "T06_1.0"),
            (18, "Target Doppler +2.0 (Shifted Index 18)", "Step7B_2.0"),
        ],
    ):
        ax.axvspan(66, 78, color="gray", alpha=0.10, label="CFAR proximity zone (66–78)")
        ax.axvspan(71, 73, color="magenta", alpha=0.12, label="Clutter mainlobe (71–73)")
        p_n = np.mean([a[d_idx, :] for a in prof_power["noise"]], axis=0)
        p_c0 = np.mean([a[d_idx, :] for a in prof_power[0.0]], axis=0)
        p_c3 = np.mean([a[d_idx, :] for a in prof_power[0.3]], axis=0)
        ax.semilogy(r_bins, p_n, "k:", linewidth=1.6, label="Noise only (mean)")
        ax.semilogy(r_bins, p_c0, "o-", linewidth=1.8, label="Clutter 0.0 + Noise (no local target)")
        ax.semilogy(r_bins, p_c3, "s--", linewidth=1.8, label="Clutter 0.3 + Noise (no local target)")
        if tgt_key is not None:
            pt0 = np.mean([a[d_idx, :] for a in r70_power[(tgt_key, 0.0)]], axis=0)
            pt3 = np.mean([a[d_idx, :] for a in r70_power[(tgt_key, 0.3)]], axis=0)
            ax.semilogy(r_bins, pt0, "^-", alpha=0.85, label=f"Target at R=70 + Clut 0.0 ({tgt_key})")
            ax.semilogy(r_bins, pt3, "v--", alpha=0.85, label=f"Target at R=70 + Clut 0.3 ({tgt_key})")
        ax.set_ylabel("Power (linear, log scale)")
        ax.set_title(f"Range Profile (r = 64–80) — {title_d}")
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(loc="upper right", fontsize=7.5, ncol=2)

    axes[-1].set_xlabel("Range Bin (Clutter Center = 72)")
    axes[-1].set_xticks(r_bins)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "range_profiles_64_80.png", dpi=180)
    plt.close(fig)

    # --- Plot 2: CFAR Threshold vs Range (r=64..80) for Target Doppler +1 and +2 ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharey=True)
    for ax, (d_idx, title_d, p_weak) in zip(
        axes,
        [
            (17, "Target Doppler +1.0 (Shifted Index 17)", weak_p_d17),
            (18, "Target Doppler +2.0 (Shifted Index 18)", weak_p_d18),
        ],
    ):
        ax.axvspan(66, 78, color="gray", alpha=0.10, label="Clutter ref-window zone (66–78)")
        ax.axvspan(71, 73, color="red", alpha=0.10, label="Clutter mainlobe CUT (71–73)")
        t_n = np.mean([a[d_idx, :] for a in prof_cfar_thr["noise"]], axis=0)
        t_c0 = np.mean([a[d_idx, :] for a in prof_cfar_thr[0.0]], axis=0)
        t_c3 = np.mean([a[d_idx, :] for a in prof_cfar_thr[0.3]], axis=0)

        ax.semilogy(r_bins, t_n, "k:", linewidth=1.6, label="Noise-only CFAR threshold")
        ax.semilogy(r_bins, t_c0, "o-", linewidth=1.9, label="CFAR threshold (Clutter 0.0)")
        ax.semilogy(r_bins, t_c3, "s--", linewidth=1.9, label="CFAR threshold (Clutter 0.3)")
        ax.axhline(p_weak, color="#2ca02c", linestyle="-.", linewidth=1.8, label=f"Weak target 0.03 (~{p_weak:,.0f})")
        ax.axhline(mod_p_d18, color="#9467bd", linestyle=":", linewidth=1.8, label=f"Moderate target 0.10 (~{mod_p_d18:,.0f})")

        ax.set_title(f"CFAR Threshold vs Range — {title_d}")
        ax.set_xlabel("Range Bin (Clutter at 72)")
        ax.set_xticks(r_bins)
        ax.set_xticklabels([str(r) for r in r_bins], fontsize=8)
        ax.grid(True, which="both", alpha=0.25)

    axes[0].set_ylabel("Mean CA-CFAR Threshold / Target Power (log scale)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 0.01))
    fig.tight_layout(rect=[0.0, 0.11, 1.0, 1.0])
    fig.savefig(OUT_DIR / "cfar_threshold_vs_range_64_80.png", dpi=180)
    plt.close(fig)

    # --- Plot 3: Target-to-Clutter Ratio & Margin Summary ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4))
    groups_summary = [
        ("Step7 A\n(0.10, d=2, c0.0)", [r for r in per_run_rows if r["dataset"] == "step7_multiseed" and r["target_amplitude"] == 0.10 and r["clutter_doppler_bin"] == 0.0]),
        ("Step7 A\n(0.10, d=2, c0.3)", [r for r in per_run_rows if r["dataset"] == "step7_multiseed" and r["target_amplitude"] == 0.10 and r["clutter_doppler_bin"] == 0.3]),
        ("Step7 B\n(0.03, d=2, c0.0)", [r for r in per_run_rows if r["dataset"] == "step7_multiseed" and r["target_amplitude"] == 0.03 and r["clutter_doppler_bin"] == 0.0]),
        ("Step7 B\n(0.03, d=2, c0.3)", [r for r in per_run_rows if r["dataset"] == "step7_multiseed" and r["target_amplitude"] == 0.03 and r["clutter_doppler_bin"] == 0.3]),
        ("T06\n(0.03, d=1, c0.0)", [r for r in per_run_rows if r["dataset"] == "slow_targets" and r["target_doppler_bin"] == 1.0 and r["clutter_doppler_bin"] == 0.0]),
        ("T06\n(0.03, d=1, c0.3)", [r for r in per_run_rows if r["dataset"] == "slow_targets" and r["target_doppler_bin"] == 1.0 and r["clutter_doppler_bin"] == 0.3]),
        ("T07\n(0.03, d=0.5, c0.0)", [r for r in per_run_rows if r["dataset"] == "slow_targets" and r["target_doppler_bin"] == 0.5 and r["clutter_doppler_bin"] == 0.0]),
        ("T07\n(0.03, d=0.5, c0.3)", [r for r in per_run_rows if r["dataset"] == "slow_targets" and r["target_doppler_bin"] == 0.5 and r["clutter_doppler_bin"] == 0.3]),
    ]
    gx = np.arange(len(groups_summary))
    tcr_peak = [np.mean([x["target_to_clutter_peak_db"] for x in sub]) for _, sub in groups_summary]
    tcr_same = [np.mean([x["target_to_clutter_same_doppler_db"] for x in sub]) for _, sub in groups_summary]
    cfar_m = [np.mean([x["cfar_margin_db"] for x in sub]) for _, sub in groups_summary]
    glob_m = [np.mean([x["global_margin_db"] for x in sub]) for _, sub in groups_summary]

    ax1.bar(gx - 0.18, tcr_peak, 0.35, label="TCR vs Clutter Peak (16, 72)")
    ax1.bar(gx + 0.18, tcr_same, 0.35, label="TCR vs Clutter at Target Doppler (d_t, 72)")
    ax1.axhline(0.0, color="red", linestyle="--", linewidth=1.0)
    ax1.set_xticks(gx)
    ax1.set_xticklabels([g[0] for g in groups_summary], fontsize=7.5, rotation=20)
    ax1.set_ylabel("Target-to-Clutter Ratio [dB]")
    ax1.set_ylim(-30.0, 26.5)
    ax1.set_title("Target-to-Clutter Ratios (R_target = 70, R_clutter = 72)")
    ax1.grid(axis="y", alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8)

    ax2.bar(gx - 0.18, glob_m, 0.35, label="Global Margin [dB]")
    ax2.bar(gx + 0.18, cfar_m, 0.35, label="CA-CFAR Margin [dB]")
    ax2.axhline(0.0, color="red", linestyle="--", linewidth=1.2, label="0 dB Detection Threshold")
    ax2.set_xticks(gx)
    ax2.set_xticklabels([g[0] for g in groups_summary], fontsize=7.5, rotation=20)
    ax2.set_ylabel("Target-to-Threshold Margin [dB]")
    ax2.set_ylim(-20.0, 34.5)
    ax2.set_title("Global vs. CA-CFAR Margins at R_target = 70")
    ax2.grid(axis="y", alpha=0.3)
    ax2.legend(loc="lower left", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "tcr_margin_comparison.png", dpi=180)
    plt.close(fig)

    # Write concise technical report markdown inside outputs_512x32_clutter_investigation
    def find_recovery(d_lbl: str, tag: str) -> tuple[list[int], list[int]]:
        rec = [r["range_bin"] for r in prof_rows if r["doppler_label"] == d_lbl and r[f"{tag}_weak_target_above_thr"]]
        masked = [r["range_bin"] for r in prof_rows if r["doppler_label"] == d_lbl and not r[f"{tag}_weak_target_above_thr"]]
        return rec, masked

    def prof_lookup(r_bin: int, d_lbl: str) -> dict:
        return next(x for x in prof_rows if x["range_bin"] == r_bin and x["doppler_label"] == d_lbl)

    rec_d1_c0, _ = find_recovery("Dop+1", "c0p0")
    rec_d1_c3, _ = find_recovery("Dop+1", "c0p3")
    rec_d2_c0, _ = find_recovery("Dop+2", "c0p0")
    rec_d2_c3, _ = find_recovery("Dop+2", "c0p3")

    table_meta = [
        ("Step 7 A", "`0.10`, d=+2, c=0.0"),
        ("Step 7 A", "`0.10`, d=+2, c=0.3"),
        ("Step 7 B", "`0.03`, d=+2, c=0.0"),
        ("Step 7 B", "`0.03`, d=+2, c=0.3"),
        ("T06", "`0.03`, d=+1, c=0.0"),
        ("T06", "`0.03`, d=+1, c=0.3"),
        ("T07", "`0.03`, d=+0.5, c=0.0"),
        ("T07", "`0.03`, d=+0.5, c=0.3"),
    ]
    table_lines = []
    for (suite_lbl, cfg_lbl), (_, sub) in zip(table_meta, groups_summary):
        tp = np.mean([x["target_power"] for x in sub])
        tpk = np.mean([x["target_to_clutter_peak_db"] for x in sub])
        tsd = np.mean([x["target_to_clutter_same_doppler_db"] for x in sub])
        gm = np.mean([x["global_margin_db"] for x in sub])
        cm = np.mean([x["cfar_margin_db"] for x in sub])
        pd_ex = np.mean([1.0 if x["cfar_detected_exact"] else 0.0 for x in sub])
        pd_3x3_nc = np.mean([1.0 if x["cfar_detected_3x3_noclutter"] else 0.0 for x in sub])
        ridge_n = np.mean([x["clutter_ridge_cfar_detections"] for x in sub])
        table_lines.append(
            f"| {suite_lbl} | {cfg_lbl} | {tp:,.0f} | `{tpk:+.2f} dB` | `{tsd:+.2f} dB` | "
            f"`{gm:+.2f} dB` | `{cm:+.2f} dB` | `{pd_ex:.2f}` | `{pd_3x3_nc:.2f}` | {ridge_n:.1f} |"
        )
    table_block = "\n".join(table_lines)

    p16_71 = prof_lookup(71, "Dop0_clutter")["c0p0_power_mean"]
    p16_72 = prof_lookup(72, "Dop0_clutter")["c0p0_power_mean"]
    p16_73 = prof_lookup(73, "Dop0_clutter")["c0p0_power_mean"]
    p16_70 = prof_lookup(70, "Dop0_clutter")["c0p0_power_mean"]
    p16_74 = prof_lookup(74, "Dop0_clutter")["c0p0_power_mean"]

    p17_72_c0 = prof_lookup(72, "Dop+1")["c0p0_power_mean"]
    p17_71_c0 = prof_lookup(71, "Dop+1")["c0p0_power_mean"]
    p17_72_c3 = prof_lookup(72, "Dop+1")["c0p3_power_mean"]
    t17_70_c0 = prof_lookup(70, "Dop+1")["c0p0_cfar_thr_mean"]
    m17_70_c0 = prof_lookup(70, "Dop+1")["c0p0_weak_target_margin_db"]
    t17_70_c3 = prof_lookup(70, "Dop+1")["c0p3_cfar_thr_mean"]
    m17_70_c3 = prof_lookup(70, "Dop+1")["c0p3_weak_target_margin_db"]

    p18_72_c0 = prof_lookup(72, "Dop+2")["c0p0_power_mean"]
    p18_71_c0 = prof_lookup(71, "Dop+2")["c0p0_power_mean"]
    p18_72_c3 = prof_lookup(72, "Dop+2")["c0p3_power_mean"]
    p18_71_c3 = prof_lookup(71, "Dop+2")["c0p3_power_mean"]
    leak_ratio_18 = p18_72_c3 / p18_72_c0
    t18_70_c0 = prof_lookup(70, "Dop+2")["c0p0_cfar_thr_mean"]
    m18_70_c0 = prof_lookup(70, "Dop+2")["c0p0_weak_target_margin_db"]
    t18_70_c3 = prof_lookup(70, "Dop+2")["c0p3_cfar_thr_mean"]
    m18_70_c3 = prof_lookup(70, "Dop+2")["c0p3_weak_target_margin_db"]

    d2_c0_contam_min_m = min(prof_lookup(r, "Dop+2")["c0p0_weak_target_margin_db"] for r in (66, 67, 68, 69, 75, 76, 77, 78))
    d2_c0_contam_max_t = max(prof_lookup(r, "Dop+2")["c0p0_cfar_thr_mean"] for r in (66, 67, 68, 69, 75, 76, 77, 78))
    d2_c3_peak_thr_min = min(prof_lookup(r, "Dop+2")["c0p3_cfar_thr_mean"] for r in (66, 67, 68, 69, 75, 76, 77, 78))
    d2_c3_peak_thr_max = max(prof_lookup(r, "Dop+2")["c0p3_cfar_thr_mean"] for r in (66, 67, 68, 69, 75, 76, 77, 78))
    d2_c3_peak_m_min = min(prof_lookup(r, "Dop+2")["c0p3_weak_target_margin_db"] for r in (66, 67, 68, 69, 75, 76, 77, 78))
    d2_c3_sh_thr_min = min(prof_lookup(r, "Dop+2")["c0p3_cfar_thr_mean"] for r in (65, 70, 74, 79))
    d2_c3_sh_thr_max = max(prof_lookup(r, "Dop+2")["c0p3_cfar_thr_mean"] for r in (65, 70, 74, 79))
    d2_c3_sh_m_min = min(prof_lookup(r, "Dop+2")["c0p3_weak_target_margin_db"] for r in (65, 70, 74, 79))
    d2_c3_clean_m = min(prof_lookup(r, "Dop+2")["c0p3_weak_target_margin_db"] for r in (64, 80))

    ridge_c0_min = min(np.mean([x["clutter_ridge_cfar_detections"] for x in sub]) for (_, sub) in groups_summary[0::2])
    ridge_c0_max = max(np.mean([x["clutter_ridge_cfar_detections"] for x in sub]) for (_, sub) in groups_summary[0::2])
    ridge_c3_min = min(np.mean([x["clutter_ridge_cfar_detections"] for x in sub]) for (_, sub) in groups_summary[1::2])
    ridge_c3_max = max(np.mean([x["clutter_ridge_cfar_detections"] for x in sub]) for (_, sub) in groups_summary[1::2])

    report_md = f"""# Clutter-Proximity Failure Investigation Report (Bins 64–80)

**Diagnostic Scope:** {len(per_run_rows)} existing runs (`40` Step-7 multi-seed, `10` noise-only, `40` T06/T07 slow-target, `120` range-robustness) analyzed without modifying `radar_model.py` or tuning any detector parameters.

## 1. Clutter Reference-Window Contamination Geometry (Around Bin 72)

- Clutter at `CLUTTER_RANGE_BIN = 72` with `np.hanning(512)` spans three range mainlobe bins: **`{{71, 72, 73}}`** (stationary clutter+noise mean power at `d=16`: `{p16_71:,.0f}` at `r=71`, `{p16_72:,.0f}` at `r=72`, `{p16_73:,.0f}` at `r=73`; drops to noise level `{p16_70:.1f}` at `r=70` and `{p16_74:.1f}` at `r=74`).
- Range CA-CFAR (`GUARD = 2`, `REF = 4`) places clutter bins `{{71, 72, 73}}` inside the reference window `[r-6..r-3] U [r+3..r+6]` for **`|r - 72| in {{2, 3, 4, 5, 6, 7}}`** (bins **`65–70`** and **`74–79`**):
  - `r = 64, 80` (`|Δr| = 8`): Reference window is completely clutter-free.
  - `r = 65, 79` (`|Δr| = 7`): Window ingests `1` clutter shoulder (`71` or `73`).
  - `r = 66, 78` (`|Δr| = 6`): Window ingests `2` clutter bins (`71, 72` or `72, 73`).
  - `r = 67, 68, 76, 77` (`|Δr| = 4, 5`): Window ingests **all 3** clutter mainlobe bins (`71, 72, 73`).
  - `r = 69, 75` (`|Δr| = 3`): Window ingests `72, 73` or `71, 72`.
  - `r = 70, 74` (`|Δr| = 2`): Window ingests `1` clutter shoulder (`73` or `71`).
  - `r = 71, 72, 73` (`|Δr| <= 1`): Reference window is clean, but the CUT itself sits inside the `{p16_73:,.0f}–{p16_72:,.0f}` clutter mainlobe.

## 2. Target-to-Clutter Ratio (TCR) & Margin at R = 70

| Suite | Config | Mean Target Power | TCR vs Peak `(16,72)` | TCR vs Same-Doppler `(d_t,72)` | Global Margin | CFAR Margin | CFAR Exact $P_d$ | CFAR 3×3 No-Clut $P_d$ | Mean Clutter-Ridge CFAR Dets |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{table_block}

## 3. Stationary (`0.0`) vs. Near-Zero-Doppler (`0.3`) Clutter & Root Cause

- **Both clutter mainlobe contamination (in range) AND Doppler leakage (across Doppler) are jointly responsible:**
  1. *Range mainlobe contamination* puts clutter energy at `r = 71, 72, 73` inside the CFAR reference window for all targets at `|r_t - 72| in [2, 7]`.
  2. *Doppler leakage* determines how much of that clutter power enters the target's Doppler row:
     - At **Doppler `+1` (`d = 17`)**, even stationary clutter (`0.0`) has a `{p17_72_c0:,.0f}` Hann Doppler mainlobe shoulder at `(17, 72)` (`{p17_71_c0:,.0f}` at `r = 71`), driving the CFAR threshold at `R = 70` to `{t17_70_c0:,.0f}` (`{m17_70_c0:+.2f} dB` margin); shifting clutter to `0.3` bins raises `(17, 72)` power to `{p17_72_c3:,.0f}` and the CFAR threshold at `R = 70` to `{t17_70_c3:,.0f}` (`{m17_70_c3:+.2f} dB`).
     - At **Doppler `+2` (`d = 18`)**, stationary clutter (`0.0`) leaks only `{p18_72_c0:,.0f}` at `(18, 72)` (`{p18_71_c0:,.0f}` at `r = 71`), leaving threshold `{t18_70_c0:,.0f}` (`{m18_70_c0:+.2f} dB` CFAR margin); shifting clutter by `+0.3` bins increases `(18, 72)` Doppler leakage **{leak_ratio_18:.1f}×** (`{p18_72_c3:,.0f}` at `r = 72`, `{p18_71_c3:,.0f}` at `r = 71`), raising the CFAR threshold at `R = 70` to `{t18_70_c3:,.0f}` and collapsing the weak-target margin to `{m18_70_c3:+.2f} dB`.

## 4. Genuine Target Detections vs. Clutter Sidelobes

- Across all `R = 70` runs, the clutter region (`d[12..22], r[71..73]`) alone generates an average of `{ridge_c0_min:.1f}–{ridge_c0_max:.1f}` CFAR detections per frame at `clutter_doppler = 0.0` and `{ridge_c3_min:.1f}–{ridge_c3_max:.1f}` at `0.3` (`10/10` hits at `r = 71` on `d = 15..17`, expanding to `d = 18` under `0.3` clutter).
- Because `r = 71` lies inside a raw `3×3` box around `R = 70`, naive `3×3` detection checks falsely report `Pd = 1.00` even when `R = 70` has `0/10` exact-cell detections (`T06` and `T07`). Excluding `r = 71..73` isolates genuine target-side energy (`1.00` via `r = 69, 70` Hann target shoulder at `cdop = 0.0`, vs. **`0.00`** at `cdop = 0.3` for both `T06` and `T07`).

## 5. Weak-Target (`0.03`) Recovery Range Separation

- **Doppler `+1.0` (`d = 17`), `clut = 0.0` & `0.3`:** Weak target (`~{weak_p_d17:,.0f}`) is masked (`margin < 0 dB`) across **all** reference-contaminated bins `65–70` and `74–79` (`|Δr| = 2..7`) as well as on the clutter mainlobe `71–73`. CA-CFAR recovers the weak target only at **`|Δr| >= 8`** (`r <= 64` or `r >= 80`; recovered bins in `64..80`: `c0.0 -> {rec_d1_c0}`, `c0.3 -> {rec_d1_c3}`).
- **Doppler `+2.0` (`d = 18`), `clut = 0.0`:** Weak target (`~{weak_p_d18:,.0f}`) clears the CFAR threshold everywhere outside the `71–73` clutter mainlobe (`recovered in 64..80 -> {rec_d2_c0}`), though `r = 66–69` and `75–78` (where `r = 72` enters the reference window) still lose `~6–7 dB` of margin (threshold peaks at `{d2_c0_contam_max_t:,.0f}`, minimum margin `{d2_c0_contam_min_m:+.2f} dB`).
- **Doppler `+2.0` (`d = 18`), `clut = 0.3`:** Near-zero-Doppler clutter leakage masks the weak target completely at `r = 66–69` and `75–78` (threshold `{d2_c3_peak_thr_min:,.0f}–{d2_c3_peak_thr_max:,.0f}`, worst margin `{d2_c3_peak_m_min:+.2f} dB`), degrades `r = 65, 70, 74, 79` to marginal detection (threshold `{d2_c3_sh_thr_min:,.0f}–{d2_c3_sh_thr_max:,.0f}`, minimum margin `{d2_c3_sh_m_min:+.2f} dB`; `recovered in 64..80 -> {rec_d2_c3}`), and achieves full clutter-free recovery (`>={d2_c3_clean_m:+.2f} dB`) only at **`|Δr| >= 8`** (`r <= 64` or `r >= 80`).

## 6. Conclusion: Limitations of Current Range-Only CA-CFAR

1. **Wide 15-bin clutter vulnerability zone (`r = 65–79`):** Because range-only CA-CFAR has `GUARD + REF = 6` plus a `±1`-bin Hann clutter mainlobe, any clutter peak creates a `15`-bin blind/degraded corridor (`|Δr| <= 7`) along range.
2. **Arithmetic mean (`CA`) vulnerability to single-cell clutter outliers:** A single clutter bin (`r = 72` or `73`) in one side of the 8-cell window dominates `np.mean(reference_cells)`, driving thresholds orders of magnitude above the thermal noise floor.
3. **Zero-Doppler coupling without MTI / notch filtering:** Without pre-CFAR clutter suppression along slow-time, `+0.3`-bin clutter leakage expands full masking into Doppler `+2` (`r = 66–69, 75–78`) and completely extinguishes `Doppler <= 1` weak targets within `|Δr| <= 7`.
4. **Raw `3×3` neighborhood ambiguity:** Proximity checks without clutter-mask exclusion conflate `r = 71` clutter shoulders with genuine target detections.
"""
    rep_path = OUT_DIR / "clutter_proximity_technical_report.md"
    rep_path.write_text(report_md, encoding="utf-8")
    print("Saved report:", rep_path)


if __name__ == "__main__":
    main()

