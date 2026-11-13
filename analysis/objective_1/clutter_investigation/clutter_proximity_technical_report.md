# Clutter-Proximity Failure Investigation Report (Bins 64–80)

**Diagnostic Scope:** 210 existing runs (`40` Step-7 multi-seed, `10` noise-only, `40` T06/T07 slow-target, `120` range-robustness) analyzed without modifying `radar_model.py` or tuning any detector parameters.

## 1. Clutter Reference-Window Contamination Geometry (Around Bin 72)

- Clutter at `CLUTTER_RANGE_BIN = 72` with `np.hanning(512)` spans three range mainlobe bins: **`{71, 72, 73}`** (stationary clutter+noise mean power at `d=16`: `1,425,422` at `r=71`, `5,654,978` at `r=72`, `1,424,511` at `r=73`; drops to noise level `76.2` at `r=70` and `26.0` at `r=74`).
- Range CA-CFAR (`GUARD = 2`, `REF = 4`) places clutter bins `{71, 72, 73}` inside the reference window `[r-6..r-3] U [r+3..r+6]` for **`|r - 72| in {2, 3, 4, 5, 6, 7}`** (bins **`65–70`** and **`74–79`**):
  - `r = 64, 80` (`|Δr| = 8`): Reference window is completely clutter-free.
  - `r = 65, 79` (`|Δr| = 7`): Window ingests `1` clutter shoulder (`71` or `73`).
  - `r = 66, 78` (`|Δr| = 6`): Window ingests `2` clutter bins (`71, 72` or `72, 73`).
  - `r = 67, 68, 76, 77` (`|Δr| = 4, 5`): Window ingests **all 3** clutter mainlobe bins (`71, 72, 73`).
  - `r = 69, 75` (`|Δr| = 3`): Window ingests `72, 73` or `71, 72`.
  - `r = 70, 74` (`|Δr| = 2`): Window ingests `1` clutter shoulder (`73` or `71`).
  - `r = 71, 72, 73` (`|Δr| <= 1`): Reference window is clean, but the CUT itself sits inside the `1,424,511–5,654,978` clutter mainlobe.

## 2. Target-to-Clutter Ratio (TCR) & Margin at R = 70

| Suite | Config | Mean Target Power | TCR vs Peak `(16,72)` | TCR vs Same-Doppler `(d_t,72)` | Global Margin | CFAR Margin | CFAR Exact $P_d$ | CFAR 3×3 No-Clut $P_d$ | Mean Clutter-Ridge CFAR Dets |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Step 7 A | `0.10`, d=+2, c=0.0 | 156,224 | `-15.59 dB` | `+22.98 dB` | `+31.22 dB` | `+27.98 dB` | `1.00` | `1.00` | 14.2 |
| Step 7 A | `0.10`, d=+2, c=0.3 | 156,265 | `-15.11 dB` | `+4.31 dB` | `+31.23 dB` | `+11.54 dB` | `1.00` | `1.00` | 18.1 |
| Step 7 B | `0.03`, d=+2, c=0.0 | 13,954 | `-26.08 dB` | `+12.54 dB` | `+20.73 dB` | `+17.46 dB` | `1.00` | `1.00` | 14.3 |
| Step 7 B | `0.03`, d=+2, c=0.3 | 13,968 | `-25.60 dB` | `-6.18 dB` | `+20.73 dB` | `+1.04 dB` | `1.00` | `1.00` | 20.0 |
| T06 | `0.03`, d=+1, c=0.0 | 14,056 | `-26.05 dB` | `-20.44 dB` | `+20.76 dB` | `-13.20 dB` | `0.00` | `1.00` | 14.0 |
| T06 | `0.03`, d=+1, c=0.3 | 14,104 | `-25.56 dB` | `-23.38 dB` | `+20.77 dB` | `-16.13 dB` | `0.00` | `0.00` | 21.2 |
| T07 | `0.03`, d=+0.5, c=0.0 | 10,728 | `-27.24 dB` | `-21.62 dB` | `+19.57 dB` | `-14.39 dB` | `0.00` | `1.00` | 14.8 |
| T07 | `0.03`, d=+0.5, c=0.3 | 10,915 | `-26.69 dB` | `-24.51 dB` | `+19.65 dB` | `-17.26 dB` | `0.00` | `0.00` | 21.2 |

## 3. Stationary (`0.0`) vs. Near-Zero-Doppler (`0.3`) Clutter & Root Cause

- **Both clutter mainlobe contamination (in range) AND Doppler leakage (across Doppler) are jointly responsible:**
  1. *Range mainlobe contamination* puts clutter energy at `r = 71, 72, 73` inside the CFAR reference window for all targets at `|r_t - 72| in [2, 7]`.
  2. *Doppler leakage* determines how much of that clutter power enters the target's Doppler row:
     - At **Doppler `+1` (`d = 17`)**, even stationary clutter (`0.0`) has a `1,551,666` Hann Doppler mainlobe shoulder at `(17, 72)` (`391,662` at `r = 71`), driving the CFAR threshold at `R = 70` to `293,862` (`-13.20 dB` margin); shifting clutter to `0.3` bins raises `(17, 72)` power to `3,063,214` and the CFAR threshold at `R = 70` to `578,495` (`-16.14 dB`).
     - At **Doppler `+2` (`d = 18`)**, stationary clutter (`0.0`) leaks only `759` at `(18, 72)` (`196` at `r = 71`), leaving threshold `258` (`+17.33 dB` CFAR margin); shifting clutter by `+0.3` bins increases `(18, 72)` Doppler leakage **76.2×** (`57,825` at `r = 72`, `14,563` at `r = 71`), raising the CFAR threshold at `R = 70` to `10,914` and collapsing the weak-target margin to `+1.07 dB`.

## 4. Genuine Target Detections vs. Clutter Sidelobes

- Across all `R = 70` runs, the clutter region (`d[12..22], r[71..73]`) alone generates an average of `14.0–14.8` CFAR detections per frame at `clutter_doppler = 0.0` and `18.1–21.2` at `0.3` (`10/10` hits at `r = 71` on `d = 15..17`, expanding to `d = 18` under `0.3` clutter).
- Because `r = 71` lies inside a raw `3×3` box around `R = 70`, naive `3×3` detection checks falsely report `Pd = 1.00` even when `R = 70` has `0/10` exact-cell detections (`T06` and `T07`). Excluding `r = 71..73` isolates genuine target-side energy (`1.00` via `r = 69, 70` Hann target shoulder at `cdop = 0.0`, vs. **`0.00`** at `cdop = 0.3` for both `T06` and `T07`).

## 5. Weak-Target (`0.03`) Recovery Range Separation

- **Doppler `+1.0` (`d = 17`), `clut = 0.0` & `0.3`:** Weak target (`~14,080`) is masked (`margin < 0 dB`) across **all** reference-contaminated bins `65–70` and `74–79` (`|Δr| = 2..7`) as well as on the clutter mainlobe `71–73`. CA-CFAR recovers the weak target only at **`|Δr| >= 8`** (`r <= 64` or `r >= 80`; recovered bins in `64..80`: `c0.0 -> [64, 80]`, `c0.3 -> [64, 80]`).
- **Doppler `+2.0` (`d = 18`), `clut = 0.0`:** Weak target (`~13,961`) clears the CFAR threshold everywhere outside the `71–73` clutter mainlobe (`recovered in 64..80 -> [64, 65, 66, 67, 68, 69, 70, 74, 75, 76, 77, 78, 79, 80]`), though `r = 66–69` and `75–78` (where `r = 72` enters the reference window) still lose `~6–7 dB` of margin (threshold peaks at `976`, minimum margin `+11.56 dB`).
- **Doppler `+2.0` (`d = 18`), `clut = 0.3`:** Near-zero-Doppler clutter leakage masks the weak target completely at `r = 66–69` and `75–78` (threshold `54,277–65,207`, worst margin `-6.69 dB`), degrades `r = 65, 70, 74, 79` to marginal detection (threshold `10,914–11,119`, minimum margin `+0.99 dB`; `recovered in 64..80 -> [64, 65, 70, 74, 79, 80]`), and achieves full clutter-free recovery (`>=+18.04 dB`) only at **`|Δr| >= 8`** (`r <= 64` or `r >= 80`).

## 6. Conclusion: Limitations of Current Range-Only CA-CFAR

1. **Wide 15-bin clutter vulnerability zone (`r = 65–79`):** Because range-only CA-CFAR has `GUARD + REF = 6` plus a `±1`-bin Hann clutter mainlobe, any clutter peak creates a `15`-bin blind/degraded corridor (`|Δr| <= 7`) along range.
2. **Arithmetic mean (`CA`) vulnerability to single-cell clutter outliers:** A single clutter bin (`r = 72` or `73`) in one side of the 8-cell window dominates `np.mean(reference_cells)`, driving thresholds orders of magnitude above the thermal noise floor.
3. **Zero-Doppler coupling without MTI / notch filtering:** Without pre-CFAR clutter suppression along slow-time, `+0.3`-bin clutter leakage expands full masking into Doppler `+2` (`r = 66–69, 75–78`) and completely extinguishes `Doppler <= 1` weak targets within `|Δr| <= 7`.
4. **Raw `3×3` neighborhood ambiguity:** Proximity checks without clutter-mask exclusion conflate `r = 71` clutter shoulders with genuine target detections.
