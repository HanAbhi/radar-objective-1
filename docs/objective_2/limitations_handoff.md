# Unresolved Technical Limitations to Carry into Objective II

**Context:** Frozen from the 217-run Objective I validation baseline (`radar_model.py`, `NR = 512`, `ND = 32`, `ADC_BITS = 12`). All items below represent verified simulation gaps—none involve unverified assumptions or hardware claims.

---

### Priority 1 — Clutter Suppression & Slow-Target Visibility Gaps
1. **Absence of Pre-CFAR MTI / Zero-Doppler Clutter Rejection:**  
   No moving-target indicator (two/three-pulse canceller) or high-pass Doppler notch filter precedes detection. Stationary clutter (`CLUTTER_AMPLITUDE = 0.60`, $d_c = 0.0$) spreads Hann energy across `{15, 16, 17}`, causing **`0.00` exact-cell CA-CFAR $P_d$** (`-13.20` to `-14.39 dB` margin) for all weak slow targets (`0.03`) at $d_t \le 1.0$ within `|Δr| <= 7`.
2. **Severe Sensitivity to Near-Zero-Doppler (`0.3`-Bin) Clutter Leakage:**  
   Fractional off-grid clutter ($d_c = 0.3$) breaks Hann integer-bin orthogonality, increasing `d = +2` (`k_d = 18`) leakage by **75.4×–76.2×** (`57,825` vs. `759`). Weak targets at `(R=70, d=+2)` suffer a **16.4 dB CA-CFAR margin collapse** (`+17.46 -> +1.04 dB`, worst seed `+0.76 dB`), while slow targets (`d <= 1`) drop from `-13.20 dB` to `-16.13 dB` (`T06`) and `-17.26 dB` (`T07`).

### Priority 2 — CFAR Architecture & Detector Limitations
3. **Range-Only CA-CFAR 15-Bin Clutter-Degraded Masking Region for Weak Targets (`r = 65–79`):**  
   Sliding CA-CFAR operates strictly along fast-time (`GUARD = 2`, `REF = 4`) across a single Doppler row. Because clutter at $r_c = 72$ spans `{71, 72, 73}`, arithmetic mean estimation (`np.mean`) pulls clutter into reference windows for **`|r - 72| in [2, 7]`**, inflating local thresholds by **40×–2,000×** and suppressing weak targets ($A_t = 0.03$). Full weak-target sensitivity recovers only at **`|Δr| >= 8`** (`r <= 64` or `r >= 80`). (Geometrically predicted and empirically verified at $R_t=70$; the complete 680-run sweep across all 17 contiguous bins is formulated in Extended Exp 1 and remains pending execution).
4. **Susceptibility of Arithmetic-Mean (`CA`) Reference to Outliers & Multi-Target Masking:**  
   Even a single clutter bin (`71` or `73` at `|Δr| = 2` or `7`) raises the weak-target Doppler `+1` threshold to `~294k–580k`. Order-Statistic (OS-CFAR), Greatest/Smallest-Of (GO/SO-CFAR), trimmed-mean, or Doppler-excluding 2D CFAR strategies were not implemented in Objective I.
5. **Hann-Induced Reference Correlation Excess False Alarms (`1.92%` vs. `1.14%` i.i.d.):**  
   Contiguous Hann-windowed reference cells exhibit lag-1 ($\rho^2 = 0.4459$) and lag-2 ($\rho^2 = 0.0282$) correlation, reducing effective independent samples from $8$ to $\approx 4.71$ and elevating pure-noise CA-CFAR $P_{fa}$ to `1.923%`.
6. **Global Frame-Median Detector Zero-Noise Collapse & Inability to Discriminate Clutter:**  
   Without a noise floor clamp, $\sigma_n = 0$ drives $T_{\text{global}} = 6 \cdot \mathrm{median}(P) \sim 10^{-13}$ (printed as `0.000000`), triggering `6,229–6,494` quantization-noise false alarms (`T03`, `C01`, `C02`). Moreover, `fixed_detections` triggers on `33/33` clutter-ridge cells (`d[12..22], r[71..73]`) with no velocity discrimination.

### Priority 3 — Fixed-Point & Bit-Accurate FPGA Readiness Gaps
7. **Floating-Point (`float64`) DSP Datapath After ADC Input:**  
   Only `i_samples` / `q_samples` are quantized to 12-bit signed integers; `radar_model.py` immediately divides by float `scaling_factor` and executes Hann weighting, FFT/IFFT butterflies, magnitude-squared power, median, and CFAR division in full IEEE-754 `float64`. Bit-width growth, fixed-point FFT truncation noise, and hardware divider rounding (especially reciprocal/shift approximations of $\hat{\mu}_{\text{CFAR}}$) remain unmodeled.
8. **Artificial Per-Frame Dynamic ADC Scaling (`0.90 * 2047 / max|x|`):**  
   Rescaling every frame dynamically changes ADC LSB step size ($\Delta_q = 1 / S$) from `~1895` (clutter+noise) to `~6073` (no clutter) and prevents testing true fixed-gain receiver clipping, dynamic-range compression of weak targets under strong clutter, or integer overflow saturation.

### Priority 4 — Scenario & Evaluation Fidelity Gaps
9. **Straddle Loss at Sub-Bin Target Doppler (`T07`, $d_t = +0.5$):**  
   Zero-padded Doppler interpolation is not used; a target halfway between bins splits power equally across `bin 16` (`10,852`) and `bin 17` (`10,728`), losing **`~1.19 dB`** peak power vs. `T06` (`14,056`).
10. **Single Point-Scatterer & Single-CPI Evaluation Scope:**  
    Only one isolated target and one isolated clutter scatterer exist per frame; range-distributed clutter patches, multi-target mutual masking, target range walk across CPIs, and multi-frame clustering / centroiding (to avoid `r = 71` clutter-ridge confusion in raw `3×3` proximity metrics) remain unaddressed for Objective II.

