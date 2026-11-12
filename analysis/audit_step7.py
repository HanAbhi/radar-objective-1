import csv
import numpy as np
from pathlib import Path

base = Path(__file__).resolve().parent.parent
step7_multi = base / "tests" / "step7_multiseed"
step7_noise = base / "tests" / "noise_only"

configs = [
    ("Group_A_Moderate", 0.10, 0.0, step7_multi / "groupA_moderate_dop0p0"),
    ("Group_A_Moderate", 0.10, 0.3, step7_multi / "groupA_moderate_dop0p3"),
    ("Group_B_Weak", 0.03, 0.0, step7_multi / "groupB_weak_dop0p0"),
    ("Group_B_Weak", 0.03, 0.3, step7_multi / "groupB_weak_dop0p3"),
]

# Exclusion mask: target 3x3 (d=17..19, r=69..71) union clutter ridge (d=12..22, r=71..73)
excl_mask = np.zeros((32, 512), dtype=bool)
excl_mask[17:20, 69:72] = True
excl_mask[12:23, 71:74] = True
assert int(excl_mask.sum()) == 39

# Load original CSVs to compare against
with open(step7_multi / "step7_consolidated_results.csv", encoding="utf-8") as f:
    orig_40 = {(r["experiment_group"], float(r["clutter_doppler"]), int(r["seed"])): r for r in csv.DictReader(f)}

with open(step7_noise / "step7_noise_only_results.csv", encoding="utf-8") as f:
    orig_10 = {int(r["seed"]): r for r in csv.DictReader(f)}

audited_40 = []
meta_errors = []
iq_errors = []
margin_diffs = []
noise_diffs = []
seed_hashes = {}  # (group, dop) -> list of hashes of noise/iq

for grp, tamp, cdop, grp_dir in configs:
    seeds_seen = []
    hashes = set()
    for seed_dir in sorted(grp_dir.glob("seed_*")):
        seed = int(seed_dir.name.split("_")[1])
        seeds_seen.append(seed)

        # 1. Verify metadata
        meta = np.load(seed_dir / "simulation_metadata.npz")
        req_keys = {
            "nr", "nd", "adc_bits", "scaling_factor",
            "target_range_bin", "target_doppler_bin", "target_doppler_shifted",
            "clutter_range_bin", "clutter_doppler_bin", "clutter_doppler_shifted",
        }
        if not req_keys.issubset(set(meta.files)):
            meta_errors.append(f"{seed_dir}: missing keys {req_keys - set(meta.files)}")
        else:
            if (int(meta["nr"]) != 512 or int(meta["nd"]) != 32 or int(meta["adc_bits"]) != 12
                or int(meta["target_range_bin"]) != 70 or int(meta["target_doppler_bin"]) != 2
                or int(meta["target_doppler_shifted"]) != 18 or int(meta["clutter_range_bin"]) != 72
                or not np.isclose(float(meta["clutter_doppler_bin"]), cdop)
                or not np.isclose(float(meta["clutter_doppler_shifted"]), 16.0 + cdop)
                or not np.isfinite(float(meta["scaling_factor"])) or float(meta["scaling_factor"]) <= 0):
                meta_errors.append(f"{seed_dir}: invalid metadata values")

        # 2. Verify IQ file (16,384 lines, 2 cols, 12-bit signed range)
        iq_path = seed_dir / "radar_input_iq.txt"
        with open(iq_path, encoding="utf-8") as f:
            line_cnt = sum(1 for _ in f)
        iq = np.loadtxt(iq_path, dtype=np.int16)
        if line_cnt != 16384 or iq.shape != (16384, 2) or iq.min() < -2048 or iq.max() > 2047:
            iq_errors.append(f"{seed_dir}: line_cnt={line_cnt}, shape={iq.shape}, min/max=({iq.min()},{iq.max()})")
        hashes.add(hash(iq.tobytes()))

        # 3. Recompute detections, margins, and identical-region Pfa
        pm = np.loadtxt(seed_dir / "reference_power_map.csv", delimiter=",")
        gd = np.loadtxt(seed_dir / "reference_global_detection_map.csv", delimiter=",", dtype=bool)
        cd = np.loadtxt(seed_dir / "reference_cfar_detection_map.csv", delimiter=",", dtype=bool)
        cval = np.loadtxt(seed_dir / "reference_cfar_valid_mask.csv", delimiter=",", dtype=bool)
        cthr = np.loadtxt(seed_dir / "reference_cfar_threshold_map.csv", delimiter=",")

        eval_mask = cval & ~excl_mask
        n_eval = int(eval_mask.sum())
        assert n_eval == 15961

        g_false_identical = int((gd & eval_mask).sum())
        c_false_identical = int((cd & eval_mask).sum())
        g_pfa_identical = g_false_identical / n_eval
        c_pfa_identical = c_false_identical / n_eval

        target_power = float(pm[18, 70])
        g_thr = float(6.0 * np.median(pm))
        c_thr = float(cthr[18, 70])
        g_ratio = target_power / g_thr
        c_ratio = target_power / c_thr
        g_mdb = float(10.0 * np.log10(g_ratio))
        c_mdb = float(10.0 * np.log10(c_ratio))

        orig = orig_40[(grp, cdop, seed)]
        if (not np.isclose(g_mdb, float(orig["global_margin_db"]), atol=1e-12)
            or not np.isclose(c_mdb, float(orig["cfar_margin_db"]), atol=1e-12)
            or not np.isclose(c_pfa_identical, float(orig["cfar_pfa"]), atol=1e-12)):
            margin_diffs.append((grp, cdop, seed))

        audited_40.append({
            "experiment_group": grp,
            "target_amplitude": tamp,
            "clutter_doppler": cdop,
            "seed": seed,
            "eval_cells": n_eval,
            "target_power": target_power,
            "global_threshold": g_thr,
            "cfar_threshold_at_target": c_thr,
            "global_target_detected": bool(gd[18, 70]),
            "cfar_target_detected": bool(cd[18, 70]),
            "global_target_detected_3x3": bool(gd[17:20, 69:72].any()),
            "cfar_target_detected_3x3": bool(cd[17:20, 69:72].any()),
            "global_margin_linear": g_ratio,
            "cfar_margin_linear": c_ratio,
            "global_margin_db": g_mdb,
            "cfar_margin_db": c_mdb,
            "global_total_count": int(gd.sum()),
            "cfar_total_count": int(cd.sum()),
            "global_false_detections_eval": g_false_identical,
            "cfar_false_detections_eval": c_false_identical,
            "global_pfa_eval": g_pfa_identical,
            "cfar_pfa_eval": c_pfa_identical,
            "orig_global_pfa_16345": float(orig["global_pfa"]),
        })

    print(f"Target-present {grp} dop={cdop}: seeds={sorted(seeds_seen)}, unique_iq_realizations={len(hashes)}")
    assert sorted(seeds_seen) == list(range(1, 11))
    assert len(hashes) == 10

# Audit 10 noise-only runs
noise_seeds = []
noise_hashes = set()
audited_10 = []
for seed_dir in sorted(step7_noise.glob("seed_*")):
    seed = int(seed_dir.name.split("_")[1])
    noise_seeds.append(seed)

    meta = np.load(seed_dir / "simulation_metadata.npz")
    if int(meta["nr"]) != 512 or int(meta["nd"]) != 32 or int(meta["adc_bits"]) != 12 or float(meta["scaling_factor"]) <= 0:
        meta_errors.append(f"{seed_dir}: invalid metadata")

    iq_path = seed_dir / "radar_input_iq.txt"
    with open(iq_path, encoding="utf-8") as f:
        line_cnt = sum(1 for _ in f)
    iq = np.loadtxt(iq_path, dtype=np.int16)
    if line_cnt != 16384 or iq.shape != (16384, 2) or iq.min() < -2048 or iq.max() > 2047:
        iq_errors.append(f"{seed_dir}: line_cnt={line_cnt}")
    noise_hashes.add(hash(iq.tobytes()))

    pm = np.loadtxt(seed_dir / "reference_power_map.csv", delimiter=",")
    gd = np.loadtxt(seed_dir / "reference_global_detection_map.csv", delimiter=",", dtype=bool)
    cd = np.loadtxt(seed_dir / "reference_cfar_detection_map.csv", delimiter=",", dtype=bool)
    cval = np.loadtxt(seed_dir / "reference_cfar_valid_mask.csv", delimiter=",", dtype=bool)
    cthr = np.loadtxt(seed_dir / "reference_cfar_threshold_map.csv", delimiter=",")

    n_valid = int(cval.sum())
    assert n_valid == 16000
    g_cnt = int((gd & cval).sum())
    c_cnt = int((cd & cval).sum())
    g_pfa = g_cnt / n_valid
    c_pfa = c_cnt / n_valid
    g_thr = float(6.0 * np.median(pm))
    v_cthr = cthr[cval]

    o = orig_10[seed]
    if (g_cnt != int(o["global_valid_detection_count"])
        or c_cnt != int(o["cfar_detection_count"])
        or not np.isclose(g_pfa, float(o["global_pfa"]), atol=1e-12)
        or not np.isclose(c_pfa, float(o["cfar_pfa"]), atol=1e-12)
        or not np.isclose(g_thr, float(o["global_threshold"]), atol=1e-10)
        or not np.isclose(float(np.mean(v_cthr)), float(o["cfar_threshold_mean"]), atol=1e-10)):
        noise_diffs.append(seed)

    audited_10.append({
        "seed": seed,
        "eval_cells": n_valid,
        "global_threshold": g_thr,
        "cfar_threshold_mean": float(np.mean(v_cthr)),
        "cfar_threshold_std": float(np.std(v_cthr, ddof=1)),
        "global_valid_detections": g_cnt,
        "cfar_valid_detections": c_cnt,
        "global_pfa": g_pfa,
        "cfar_pfa": c_pfa,
    })

print(f"Noise-only: seeds={sorted(noise_seeds)}, unique_iq_realizations={len(noise_hashes)}")
assert sorted(noise_seeds) == list(range(1, 11))
assert len(noise_hashes) == 10

print("Metadata errors:", len(meta_errors), meta_errors)
print("IQ errors:", len(iq_errors), iq_errors)
print("Margin/CFAR-Pfa diffs vs original CSV:", len(margin_diffs))
print("Noise-only diffs vs original CSV:", len(noise_diffs))

# Write corrected consolidated CSVs (keeping originals untouched)
out_40 = step7_multi / "step7_consolidated_results_audited.csv"
with open(out_40, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(audited_40[0].keys()))
    w.writeheader()
    w.writerows(audited_40)

# Combined 50-run audited summary
summary_50 = []
for grp, tamp, cdop, _ in configs:
    sub = [r for r in audited_40 if r["experiment_group"] == grp and r["clutter_doppler"] == cdop]
    summary_50.append({
        "condition": f"{grp}_dop{cdop}",
        "target_amplitude": tamp,
        "clutter_amplitude": 0.60,
        "clutter_doppler": cdop,
        "noise_amplitude": 0.08,
        "runs": len(sub),
        "eval_cells": 15961,
        "global_pd_exact": np.mean([r["global_target_detected"] for r in sub]),
        "cfar_pd_exact": np.mean([r["cfar_target_detected"] for r in sub]),
        "global_pd_3x3": np.mean([r["global_target_detected_3x3"] for r in sub]),
        "cfar_pd_3x3": np.mean([r["cfar_target_detected_3x3"] for r in sub]),
        "global_false_mean": np.mean([r["global_false_detections_eval"] for r in sub]),
        "global_false_std": np.std([r["global_false_detections_eval"] for r in sub], ddof=1),
        "cfar_false_mean": np.mean([r["cfar_false_detections_eval"] for r in sub]),
        "cfar_false_std": np.std([r["cfar_false_detections_eval"] for r in sub], ddof=1),
        "global_pfa_mean": np.mean([r["global_pfa_eval"] for r in sub]),
        "global_pfa_std": np.std([r["global_pfa_eval"] for r in sub], ddof=1),
        "cfar_pfa_mean": np.mean([r["cfar_pfa_eval"] for r in sub]),
        "cfar_pfa_std": np.std([r["cfar_pfa_eval"] for r in sub], ddof=1),
        "global_margin_db_mean": np.mean([r["global_margin_db"] for r in sub]),
        "cfar_margin_db_mean": np.mean([r["cfar_margin_db"] for r in sub]),
        "cfar_margin_db_min": np.min([r["cfar_margin_db"] for r in sub]),
    })

summary_50.append({
    "condition": "Noise_Only",
    "target_amplitude": 0.0,
    "clutter_amplitude": 0.0,
    "clutter_doppler": "NA",
    "noise_amplitude": 0.08,
    "runs": len(audited_10),
    "eval_cells": 16000,
    "global_pd_exact": "NA",
    "cfar_pd_exact": "NA",
    "global_pd_3x3": "NA",
    "cfar_pd_3x3": "NA",
    "global_false_mean": np.mean([r["global_valid_detections"] for r in audited_10]),
    "global_false_std": np.std([r["global_valid_detections"] for r in audited_10], ddof=1),
    "cfar_false_mean": np.mean([r["cfar_valid_detections"] for r in audited_10]),
    "cfar_false_std": np.std([r["cfar_valid_detections"] for r in audited_10], ddof=1),
    "global_pfa_mean": np.mean([r["global_pfa"] for r in audited_10]),
    "global_pfa_std": np.std([r["global_pfa"] for r in audited_10], ddof=1),
    "cfar_pfa_mean": np.mean([r["cfar_pfa"] for r in audited_10]),
    "cfar_pfa_std": np.std([r["cfar_pfa"] for r in audited_10], ddof=1),
    "global_margin_db_mean": "NA",
    "cfar_margin_db_mean": "NA",
    "cfar_margin_db_min": "NA",
})

out_50 = step7_multi / "step7_combined_50runs_audited_summary.csv"
with open(out_50, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(summary_50[0].keys()))
    w.writeheader()
    w.writerows(summary_50)

print("\nSaved audited CSVs:")
print(" ", out_40)
print(" ", out_50)
print("\n--- AUDITED SUMMARY (IDENTICAL DENOMINATOR) ---")
for s in summary_50:
    print(
        f"{s['condition']} (N={s['eval_cells']}): "
        f"G_false={s['global_false_mean']:.2f}+-{s['global_false_std']:.2f}, "
        f"G_Pfa={100*s['global_pfa_mean']:.4f}%+-{100*s['global_pfa_std']:.4f}% | "
        f"C_false={s['cfar_false_mean']:.2f}+-{s['cfar_false_std']:.2f}, "
        f"C_Pfa={100*s['cfar_pfa_mean']:.4f}%+-{100*s['cfar_pfa_std']:.4f}%"
    )
