# Objective I — One-Page Executive Summary

**Project:** Development and Simulation of a Radar Signal Processing Model for Weak and Slow-Moving Target Detection in Cluttered Environments  
**Scope:** Software Reference Validation Only (`radar_model.py`, `NR = 512 × ND = 32`, signed 12-bit I/Q export) — *No FPGA hardware implementation or measured RTL performance claimed.*

## 1. Baseline System & Signal Model Summary
- **CPI Data Cube:** $x[m, n] = s_t[m, n] + s_c[m, n] + w[m, n]$ (`32 × 512 = 16,384` complex samples); `12`-bit signed I/Q quantization (`90%` dynamic frame full-scale), restored to `float64` for reference DSP.
- **2D Hann FFT Transform Gains:** Coherent gain $G_c = 15.68 \times 10^6$, incoherent noise gain $G_n = 2{,}227.64$, processing gain **$\mathrm{PG} = +38.48\,\mathrm{dB}$** (`7,040.4×`); baseline AWGN ($\sigma_n = 0.08$) cell mean power $\mu_N = 28.51$.
- **Detectors Evaluated (Frozen):**
  - **Global Threshold:** $T_{\text{global}} = 6.0 \times \mathrm{median}(P)$; theoretical noise $P_{fa} = 2^{-6} = \mathbf{1.563\%}$.
  - **Range-Only CA-CFAR:** $G = 2$ guard, $N_{\text{side}} = 4$ ref cells/side ($8$ total along range only), $\alpha_{\text{CFAR}} = 6.0$; `16,000` valid cells (`r = 6..505`). Hann lag-1/lag-2 correlation lowers $N_{\text{eff}} \approx 4.71$, shifting noise $P_{fa}$ from `1.137%` (i.i.d.) to `~2.09%`.

## 2. Empirical Validation Across 217 Simulation Runs

| Experiment Suite | Runs | Exact Global $P_d$ | Exact CA-CFAR $P_d$ | CA-CFAR `3×3` Clut-Excl. $P_d$ | Mean CA-CFAR Margin (dB) | Background $P_{fa}$ (Global / CFAR) |
|---|---:|---:|---:|---:|---:|---|
| **T01–T05** & **C01–C02** (Single seed `42`) | `7` | `1.00` (excl. `T04`); `T03/C01/C02` false* | `1.00` (`T01,T02,T05`) | `1.00` (`T01,T02,T05`) | `+14.56` (`T05`) to `+41.15` (`T02`) | `T04`: `1.475%` / `1.844%` (`16k`) |
| **Step 7 Noise-Only** (`σ_n = 0.08`, `16,000` cells) | `10` | — | — | — | — | **`1.589%`** / **`1.923%`** |
| **Step 7 Group A** (`0.10, R=70, d=+2.0`) | `20` | `1.00` | `1.00` | `1.00` | `+27.98` (`c=0.0`) / `+11.54` (`c=0.3`) | `1.57%` / `1.91%` (`15,961`) |
| **Step 7 Group B** (`0.03, R=70, d=+2.0`) | `20` | `1.00` | `1.00` | `1.00` | `+17.46` (`c=0.0`) / **`+1.04`** (`c=0.3`) | `1.57%` / `1.91%` (`15,961`) |
| **T06 & T07 Slow** (`0.03, R=70, d=+1.0, +0.5`) | `40` | `1.00` | **`0.00`** (`0/40`) | `1.00` (`c=0.0`) / **`0.00`** (`c=0.3`) | **`-13.20` to `-17.26 dB`** (Masked) | `1.57–1.59%` / `1.91%` |
| **Range Robustness** (`R != 70`, `100` runs) | `100` | `1.00` | **`1.00`** | `1.00` | **`+17.32` to `+20.02 dB`** | `1.59–1.60%` / `1.87–1.88%` |
| **Range Robustness** (`R = 70`, `20` runs) | `20` | `1.00` | **`0.50`** (`0/10` at `d=+1`) | `0.75` (`0/5` at `d=1, c=0.3`) | `-16.24` (`d=1`) / `+1.00` (`d=2, c=0.3`) | `1.59%` / `1.88%` |

*Note (*): With $\sigma_n = 0$ (`T03`, `C01`, `C02`), `T_global ≈ 0.000000` causes `>6,200` quantization-residual false alarms (`target=True` at $A_t = 0$). Raw unmasked `3×3` detection falsely reports `1.00` on `T06/T07` due to adjacent `r = 71` clutter hits.*

## 3. Core Technical Conclusions & Clutter-Proximity Breakdown
1. **Validated Core Pipeline:** Transform gain (`+38.48 dB`), clutter-free weak-target detectability (`100/100` across `R = 40, 120, 200, 300, 450`), and calibrated background $P_{fa}$ (`1.59%` Global, `1.92%` CA-CFAR) are statistically proven.
2. **15-Bin Clutter Vulnerability Corridor (`r = 65–79`, `|Δr| <= 7`):** With Hann range mainlobe clutter across `{71, 72, 73}` and range-only CA-CFAR (`G = 2, N_side = 4`), clutter enters the reference window for all targets at `|Δr| = 2..7` and swamps the CUT at `|Δr| <= 1`. (Geometrically predicted from C01/C02 profiles; complete empirical 680-run sweep remains pending).
3. **Joint Failure Mechanism:** Stationary Hann Doppler mainlobe (`d = +1`) plus fractional off-grid Doppler leakage (`d_c = 0.3` boosts `d = +2` power **75×–76×** to `~57,825`) overwhelms arithmetic mean CA-CFAR (`-13.2 to -17.3 dB` masking at $d_t \le 1$; `+1.04 dB` near-failure at $d_t = +2, d_c = 0.3$). Full recovery requires **`|Δr| >= 8`** (`r <= 64` or `r >= 80`).

## 4. Top Unresolved Technical Limitations Carried to Objective II
1. **Range-only CA-CFAR outlier sensitivity:** Single clutter bins dominate $\hat{\mu}_{\text{CFAR}}$; lacks OS-CFAR / censored / 2D Doppler-aware windowing.
2. **No pre-CFAR clutter suppression:** Absent MTI or zero-Doppler notch filtering leaves slow targets (`|d_t| <= 1`) masked and leaves $d_t = +2$ vulnerable to near-zero-Doppler (`0.3`-bin) clutter leakage.
3. **Unquantized float64 DSP & per-frame AGC:** Only the ADC input is 12-bit quantized while FFT, windows, squaring, and CFAR run in `float64` with per-frame `0.90` scaling that masks fixed-gain ADC saturation.
4. **Global detector fragility & metric hazards:** `T_global` collapses at $\sigma_n = 0$ and accepts clutter-ridge triggers; raw `3×3` masks require clutter exclusion.
5. **Model simplifications:** Single point-clutter/target scenario lacks multi-target masking, distributed clutter, and non-coherent integration / binary $M$-of-$N$ confirmation.
