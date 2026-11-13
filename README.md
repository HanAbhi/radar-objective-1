# Radar Signal Processing: Weak & Slow-Moving Target Detection in Cluttered Environments

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/CI-GitHub%20Actions%20Passing-success.svg)](.github/workflows/radar_ci.yml)
[![Tests: 26/26 Passing](https://img.shields.io/badge/pytest-26%2F26%20passed%20(0.56s)-brightgreen.svg)](tests/unit/)
[![Status: Objective I Frozen](https://img.shields.io/badge/Objective%20I-Validated%20%26%20Frozen-success.svg)](docs/objective_1/final_audit.pdf)
[![Status: Objective II Planned](https://img.shields.io/badge/Objective%20II-RTL%20Architecture%20Planned-blue.svg)](docs/objective_2/limitations_handoff.pdf)

An academic research and engineering portfolio repository presenting the mathematical modeling, algorithmic simulation, statistical validation, and fixed-gain ADC quantization analysis of a coherent pulse-Doppler radar receiver designed for weak and slow-moving target detection in severe clutter environments, with a planned roadmap for fixed-point RTL implementation on FPGA.

---

## Table of Contents

1. [Project Overview & Research Motivation](#1-project-overview--research-motivation)
2. [Problem Statement & Scientific Challenges](#2-problem-statement--scientific-challenges)
3. [End-to-End Signal Processing Architecture](#3-end-to-end-signal-processing-architecture)
4. [Objective I Baseline Methodology](#4-objective-i-baseline-methodology)
5. [Original 217-Run Validation Results](#5-original-217-run-validation-results)
6. [Extended Validation Experiments & Results](#6-extended-validation-experiments--results)
   - [Experiment 1: Clutter-Proximity Sweep](#extended-experiment-1-direct-clutter-proximity-sweep)
   - [Experiment 2: Empirical CA-CFAR Calibration & ROC Analysis](#extended-experiment-2-empirical-ca-cfar-calibration--roc-analysis)
   - [Experiment 3: Fixed-Gain ADC & Quantisation Robustness](#extended-experiment-3-fixed-gain-adc--quantisation-robustness)
   - [Automated Unit Test Suite & CI Automation](#automated-unit-test-suite--ci-automation)
7. [Known Limitations & Failure Modes](#7-known-limitations--failure-modes)
8. [Multi-Phase Research Roadmap (Objectives I, II, and III)](#8-multi-phase-research-roadmap-objectives-i-ii-and-iii)
9. [Repository Architecture](#9-repository-architecture)
10. [Installation & Reproduction Instructions](#10-installation--reproduction-instructions)
11. [Configuration & Seed Management Policy](#11-configuration--seed-management-policy)
12. [Primary Reports, Audit Certificates & Data Links](#12-primary-reports-audit-certificates--data-links)

---

## 1. Project Overview & Research Motivation

Modern radar systems operating in littoral, urban, and low-altitude airspace encounter extreme operational challenges: detecting small, slow-moving targets—such as micro-UAVs, walking personnel, or low-RCS surface targets—masked by high-amplitude stationary ground backscatter and near-zero-Doppler clutter (e.g., wind-blown foliage, rain, and sea surface reflections).

In these operating conditions, the radar cross-section (RCS) of targets is minute compared to backscattering clutter, resulting in severe clutter-to-target amplitude disparities ($>26\,\mathrm{dB}$ to $55\,\mathrm{dB}$). The research objective is to develop, simulate, benchmark, and transition a robust baseband radar signal processing pipeline from mathematical simulation through hardware deployment.

### Clear Distinction of Project Phases

This repository enforces strict scientific and engineering boundaries across three formal project phases:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: OBJECTIVE I — SIMULATION & VALIDATION BASELINE (COMPLETED & FROZEN)     │
│ • Mathematical modeling of target, clutter, AWGN, and 12-bit ADC quantization.   │
│ • 217-run baseline experimental campaign & statistical audit.                    │
│ • Extended validation: CFAR calibration (1,050 frames), fixed-gain ADC (400 runs)│
│ • 26-test unit test suite & multi-platform GitHub Actions CI.                    │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │ Golden Vectors & Architecture Specs
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: OBJECTIVE II — RTL HARDWARE IMPLEMENTATION (PLANNED)                    │
│ • Bit-accurate fixed-point C++/Python reference models.                          │
│ • Pipelined streaming Radix-2/4 Range-Doppler FFT in Verilog/SystemVerilog.      │
│ • Pre-CFAR MTI filter / notch filter & Outlier-robust OS-CFAR logic.             │
│ • Digital AGC module replacing dynamic per-frame software normalization.         │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │ Synthesized Bitstreams & Testbenches
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: OBJECTIVE III — HARDWARE EVALUATION & SILICON TESTING (PLANNED)         │
│ • Deployment to AMD Xilinx Zynq UltraScale+ RFSoC / Artix-7 evaluation boards.   │
│ • Real-time Hardware-in-the-Loop (HIL) latency, resource, and power measurement. │
│ • RF/ADC loopback characterization with physical noise and clutter injectors.     │
└──────────────────────────────────────────────────────────────────────────────────┘
```

> [!IMPORTANT]
> **Scientific Integrity Notice:**  
> • **Objective I** is fully implemented, empirically validated, audited, and mathematically frozen in `src/radar_model.py`.  
> • **Extended Validation Experiments 2 & 3** and the **Pytest Suite** are fully executed, verified, and documented.  
> • **Extended Validation Experiment 1** has completed a 4-run pilot; the full 680-run sweep is automated and configured.  
> • **Objective II** and **Objective III** represent the planned hardware engineering roadmap and are not reported as executed.

---

## 2. Problem Statement & Scientific Challenges

Detecting weak, slow-moving targets under conventional pulse-Doppler processing presents three coupled failure mechanisms:

1. **Spectral Sidelobe Masking:**  
   When clutter possesses near-zero velocity (e.g., wind-blown vegetation with fractional Doppler shift $d_c = 0.3$), the clutter spectrum is not aligned with discrete FFT bins. This off-grid frequency shift degrades Hann window sidelobes, elevating Doppler sidelobe leakage by **$+18.8\,\mathrm{dB}$ ($75.4\times$)**, completely burying weak target returns at adjacent Doppler bins ($d_t = +1.0, +2.0$).

2. **CFAR Threshold Inflation & Target Masking:**  
   Standard Cell-Averaging CFAR (CA-CFAR) estimates the local noise floor by averaging reference cells along range. When a strong clutter mainlobe enters any reference cell, the arithmetic mean inflates the threshold by up to **$+37.8\,\mathrm{dB}$**. This artificially elevates the threshold over a **15-bin region ($|r - 72| \le 7$)**, driving the detection margin of weak targets negative ($-13.2\,\mathrm{dB}$ to $-23.9\,\mathrm{dB}$) and causing total missed detection ($P_d = 0.00$).

3. **ADC Quantization & Dynamic Range Saturation:**  
   Real-world receivers rely on fixed-gain analog front-ends. Under high clutter amplitudes ($A_c \ge 1.5\,\mathrm{V}$), unscaled 12-bit ADCs enter heavy saturation ($>54\%$ clipping), generating non-linear intermodulation harmonics that elevate the global noise floor by $+5.34\,\mathrm{dB}$ and blind weak targets. Conversely, insufficient gain drops Signal-to-Quantization-Noise Ratio (SQNR) to $56.8\,\mathrm{dB}$.

---

## 3. End-to-End Signal Processing Architecture

The simulation models a coherent pulse-Doppler baseband radar receiver operating over a single Coherent Processing Interval (CPI):

```
                                      RADAR RECEIVER SIGNAL CHAIN
                                      
  Target s_t(t) ──┐
 Clutter s_c(t) ──┼──►[ Analog Sum ]──►[ 12-bit ADC ]──►[ 2D Hann Window ]──►[ Range FFT ]──►[ Doppler FFT ]──►[ Power Map ]
   Noise w(t)   ──┘      (V_in)          [-2048, 2047]     w_r(n) * w_d(m)      (512-pt)       (32-pt, shift)     P = |S|^2
                                                                                                                    │
                                    ┌───────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ├──►[ Global Median Detector ]────► Threshold T_glob = 6.0 * Median(P) ──► D_glob
                                    │
                                    └──►[ 1D Range CA-CFAR ]──────────► Threshold T_cfar = α * Mean(P_ref) ──► D_cfar
```

### Complete Parameter Specifications

| Parameter | Symbol | Baseline Value | Units | Description |
|---|:---:|:---:|:---:|---|
| **Fast-Time Range Samples** | $N_R$ | `512` | samples | Number of range samples per transmitted chirp pulse |
| **Slow-Time Pulses** | $N_D$ | `32` | chirps | Number of coherent pulses per CPI ($16,384$ total complex samples) |
| **ADC Resolution** | $B$ | `12` | bits | Signed two's-complement quantizer ($[-2048, +2047]$ LSB) |
| **Headroom Scaling Factor** | $S$ | `0.90` | — | Dynamic scaling headroom applied to prevent digital clipping |
| **Range Windowing** | $w_r[n]$ | `Hann` | — | Symmetric Hann window across $N_R=512$, unnormalized discrete sum |
| **Doppler Windowing** | $w_d[m]$ | `Hann` | — | Symmetric Hann window across $N_D=32$, unnormalized discrete sum |
| **Range Processing** | — | `512-pt FFT` | — | Discrete Fourier Transform along fast-time axis |
| **Doppler Processing** | — | `32-pt FFT + shift`| — | DFT along slow-time axis; zero Doppler centered at bin 16 |
| **Power Metric** | $P[r, d]$ | $|S[r, d]|^2$ | $\mathrm{LSB}^2$ | Unnormalized squared-magnitude power map |
| **Global Threshold Multiplier** | $K_{\text{glob}}$ | `6.0` | — | Multiplier applied to global median of valid power map |
| **CA-CFAR Guard Bins** | $G$ | `2` | bins/side | Guard cells on each side of CUT ($4$ guard cells total) |
| **CA-CFAR Reference Bins** | $N_{\text{side}}$ | `4` | bins/side | Reference cells on each side of CUT ($N_{\text{ref}} = 8$ cells total) |
| **CA-CFAR Window Type** | — | `1D Range-Sliding`| — | Evaluated strictly along the range axis for each Doppler bin |
| **Valid Evaluation Mask** | — | `[6, 505]` $\times$ `[0, 31]` | cells | Excludes boundary reference margins; exactly $16,000$ valid CUT cells |
| **Baseline CFAR Multiplier** | $\alpha$ | `6.0` | — | Theoretical design multiplier for $P_{fa} \approx 10^{-2}$ |

### Window Gains and Mathematical Normalization

Because the discrete Fourier transform executes without scaling ($\frac{1}{N}$ factor omitted), signal and noise powers scale with the 2D window gains:

- **1D Fast-Time Hann ($N_R = 512$):**  
  $\sum w_r = 255.5$, Coherent Gain $G_{c,r} = (\sum w_r)^2 = 65{,}280.25$ ($+48.15\,\mathrm{dB}$), Noise Gain $G_{n,r} = \sum w_r^2 = 191.625$ ($+22.83\,\mathrm{dB}$), Processing Gain $\mathrm{PG}_r = 340.67$ ($+25.32\,\mathrm{dB}$).
- **1D Slow-Time Hann ($N_D = 32$):**  
  $\sum w_d = 15.5$, Coherent Gain $G_{c,d} = 240.25$ ($+23.81\,\mathrm{dB}$), Noise Gain $G_{n,d} = 11.625$ ($+10.65\,\mathrm{dB}$), Processing Gain $\mathrm{PG}_d = 20.67$ ($+13.15\,\mathrm{dB}$).
- **2D Product Hann ($512 \times 32$):**  
  $$G_c = G_{c,r} \times G_{c,d} = 15{,}683{,}580.0625 \quad (+71.95\,\mathrm{dB})$$
  $$G_n = G_{n,r} \times G_{n,d} = 2{,}227.640625 \quad (+33.48\,\mathrm{dB})$$
  $$\mathbf{\mathrm{PG} = \frac{G_c}{G_n} = 7{,}040.4444 \quad (+38.48\,\mathrm{dB})}$$

Peak output power for an on-grid tone of amplitude $A$ is $P_{\text{peak}} = A^2 G_c$. For nominal amplitudes:
- Strong Clutter ($A_c = 0.60$): $P_{\text{peak}} \approx 5.646 \times 10^6\,\mathrm{LSB}^2$ ($+67.52\,\mathrm{dB}$).
- Moderate Target ($A_t = 0.10$): $P_{\text{peak}} \approx 156.84 \times 10^3\,\mathrm{LSB}^2$ ($+51.95\,\mathrm{dB}$).
- Weak Target ($A_t = 0.03$): $P_{\text{peak}} \approx 14.115 \times 10^3\,\mathrm{LSB}^2$ ($+41.50\,\mathrm{dB}$).

---

## 4. Objective I Baseline Methodology

The Objective I baseline established the algorithmic foundation through strict Monte Carlo protocol:
1. **Deterministic Benchmark Suite (T01–T05, C01–C02):** Fixed random seed 42, used to isolate individual signal components, verify linear superposition, calibrate 2D window processing gains, and measure the spectral leakage of fractional Doppler clutter.
2. **Multi-Seed Statistical Campaigns:** Uniform random seeds $1–10$ used to measure statistical variance across independent AWGN realizations.
3. **Detection Decision Metrics:**
   - **Exact-Cell $P_d$:** Binary detection strictly at $(R_t, D_t)$.
   - **Clutter-Excluded $3\times3$ Neighborhood $P_d$:** Binary detection within $[R_t-1..R_t+1] \times [D_t-1..D_t+1]$, with columns $71–73$ (clutter ridge) explicitly masked to prevent false positive attribution.
   - **Detection Margin ($\mathrm{dB}$):** $10 \log_{10}(P_{\text{CUT}} / T_{\text{threshold}})$. A positive margin guarantees detection; negative indicates a miss.

---

## 5. Original 217-Run Validation Results

All 217 baseline simulation runs have been verified and audited against the underlying CSV run logs and raw I/Q data cubes:

| Experiment Suite | Test Directory | Runs | Parameter Range & Conditions | Key Measured Results |
|---|---|:---:|---|---|
| **T01–T05 Benchmarks** | `tests/objective_1/T01_T05/` | 5 | Isolated targets, clutter, and noise; seed 42 | Verified 2D Hann gain ($+38.48\,\mathrm{dB}$). Discovered zero-noise median collapse under $A_n=0.0$. |
| **C01–C02 Clutter Tests** | `tests/objective_1/C01_C02/` | 2 | Stationary ($d_c=0.0$) vs near-zero ($d_c=0.3$) clutter | Off-grid clutter increases Doppler $+2$ leakage by **$75.4\times$ ($+18.8\,\mathrm{dB}$)** ($759 \to 57{,}825$). |
| **Step 7 Multi-Seed** | `tests/objective_1/step7_multiseed/` | 40 | Target+clutter+noise, $R_t=70, d_t=+2.0$, seeds 1–10 | Weak targets suffer a **$16.42\,\mathrm{dB}$ margin loss** under $d_c=0.3$ ($+17.46 \to +1.04\,\mathrm{dB}$). |
| **Step 7 Noise-Only** | `tests/objective_1/noise_only/` | 10 | Pure AWGN ($\sigma_n=0.08$), seeds 1–10, 16,000 cells/run | Global $P_{fa} = 1.589\% \pm 0.162\%$ (Theory: $1.563\%$). CA-CFAR $P_{fa} = 1.923\% \pm 0.136\%$ ($N_{\text{eff}} \approx 4.714$). |
| **Slow Targets (T06–T07)** | `tests/objective_1/slow_targets/` | 40 | $R_t=70, d_t \in \{+1.0, +0.5\}$, seeds 1–10 | **Observed $P_d = 0/10$ ($0.00$)** for weak targets. Clutter mainlobe overlap yields margins of $-13.20$ to $-17.26\,\mathrm{dB}$. |
| **Range Robustness** | `tests/objective_1/range_robustness/` | 120 | 6 range bins $\{40, 70, 120, 200, 300, 450\}$, seeds 1–5 | **Observed 5/5 detection** across all 100 clutter-free runs ($R_t \neq 70$, margins $+17.3$ to $+20.0\,\mathrm{dB}$). Misses strictly confined to $R_t=70$. |
| **Total Baseline Runs** | — | **217** | **100% verified against CSV run logs** | **Audited and frozen in `docs/objective_1/final_audit.pdf`** |

---

## 6. Extended Validation Experiments & Results

To address the limitations discovered during Objective I, three extended validation campaigns and an automated test suite were developed:

### Extended Experiment 1: Direct Clutter-Proximity Sweep
- **Objective:** Measure weak-target detectability across every contiguous range bin from $R_t=64$ through $R_t=80$ (17 bins) across 2 target Dopplers ($+1.0, +2.0$), 2 clutter Dopplers ($0.0, 0.3$), and 10 random seeds (680 total runs).
- **Execution Status:** **Pilot Phase Completed & Audited** (4 runs: $R_t \in \{64, 70, 72, 76\}$); full 680-run sweep automated in [`run_clutter_proximity_sweep.py`](tests/extended_validation/clutter_proximity_sweep/run_clutter_proximity_sweep.py) and ready for execution.
- **Methodology & Geometry Distinction:**
  - *Direct Target-Present Measurements:* Verified at $R_t=70$ in baseline Step 7 and T06/T07 ($P_d = 0.00$), and at $R_t \in \{64, 70, 72, 76\}$ in the 4-run pilot.
  - *Clutter Contamination Geometry:* Stationary clutter at $R_c = 72$ spans mainlobe bins $\{71, 72, 73\}$. With guard cells $G=2$ and reference cells $N_{\text{side}}=4$, reference cells overlap the clutter mainlobe whenever $|r - 72| \in [2, 7]$ ($r \in [65, 70] \cup [74, 79]$), while $|r - 72| \le 1$ ($r \in [71, 73]$) sits directly on the clutter peak. This defines the **15-bin clutter-degraded region ($r \in [65, 79]$)**.
  - *Post-Hoc Analytical Projection:* Evaluates expected threshold inflation and TCR collapse across all 17 bins using measured C01/C02 profiles; full empirical 680-run validation remains configured for execution.
- **Pilot Findings:**
  - At $R_t=70$ and $R_t=72$, clutter mainlobe energy inside the reference window inflates the local CFAR threshold to $3.51 \times 10^6\,\mathrm{LSB}^2$ ($+37.8\,\mathrm{dB}$ above noise), driving exact-cell detection margin to **$-23.96\,\mathrm{dB}$** ($P_d = 0.00$).
  - At $R_t=64$ (outside the 7-bin reference radius), the threshold drops to $252.1\,\mathrm{LSB}^2$, recovering detection margin to **$+17.48\,\mathrm{dB}$** ($P_d = 1.00$).

### Extended Experiment 2: Empirical CA-CFAR Calibration & ROC Analysis
- **Objective:** Calibrate the CFAR multiplier $\alpha$ on independent noise datasets to achieve an exact target false alarm rate of $P_{fa}^* = 1.000\%$, and evaluate ROC detection trade-offs.
- **Execution Status:** **Fully Executed, Audited, and Documented** ([`cfar_calibration_roc_report.md`](docs/objective_1/cfar_calibration_roc_report.md)).
- **Dataset Reconciliation (1,050 Frames):**
  - **100 noise-only calibration frames** (seeds 1001–1100, $1.6 \times 10^6$ CUT cells).
  - **200 independent noise-only validation frames** (seeds 2001–2200, $3.2 \times 10^6$ CUT cells).
  - **750 target-present simulation frames** (3 environments: `clutter_free`, `clutter_distant`, `clutter_adjacent` $\times$ 5 target amplitudes $A_t \in [0.001, 0.10]$ $\times$ 50 Monte Carlo seeds 3001–3050).
  - Total: $100 + 200 + 750 = \mathbf{1{,}050 \text{ unique CPI simulation frames}}$.
- **Key Measured Results & Statistical CI Reconciliation:**
  - **Optimal Calibrated Multiplier:** $\mathbf{\alpha^* = 7.4249}$ achieves empirical validation $P_{fa} = \mathbf{1.000\%}$.
  - **Cluster-Robust 95% Confidence Interval:** Accounting for positive intra-frame cell correlation (Hann window tapering and reference window overlap) across the 200 independent validation frames ($\nu = 199$), the frame-level standard error of the mean is $\mathrm{SEM} = 0.0081\%$, yielding a **$95\%$ Student-$t$ confidence interval of $[0.984\%, 1.016\%]$**.
  - **Naive Binomial Wilson Score Comparison:** Under the assumption of $3.2 \times 10^6$ independent Bernoulli trials (which ignores spatial correlation), the naive Wilson interval is $[0.989\%, 1.011\%]$. The cluster-robust interval appropriately reflects the true sample variance.
  - **Detection Margin Penalty:** Increasing $\alpha$ from $6.0$ to $7.425$ suppresses false alarms by $48\%$, but imposes a **$-0.93\,\mathrm{dB}$ detection margin penalty**.
  - **Weak-Target Sensitivity Collapse:** For weak targets ($A_t=0.03$) at $R_t=70$ under near-zero Doppler clutter ($d_c=0.3$), the margin drops from $+1.04\,\mathrm{dB}$ (marginal detection) to **$-0.07\,\mathrm{dB}$ (missed detection)**.

```
CFAR Multiplier (α) vs. Empirical False-Alarm Probability (Pfa) on 200 Validation Frames
─────────────────────────────────────────────────────────────────────────────────────────────
α = 5.5000 ──► Pfa = 2.413%  [Cluster-Robust 95% CI: 2.390% - 2.436%] (SEM = 0.0118%)
α = 6.0000 ──► Pfa = 1.896%  [Cluster-Robust 95% CI: 1.875% - 1.917%] (SEM = 0.0109%) <-- Baseline
α = 6.5000 ──► Pfa = 1.504%  [Cluster-Robust 95% CI: 1.485% - 1.524%] (SEM = 0.0099%)
α = 7.0000 ──► Pfa = 1.202%  [Cluster-Robust 95% CI: 1.184% - 1.220%] (SEM = 0.0091%)
α = 7.4249 ──► Pfa = 1.000%  [Cluster-Robust 95% CI: 0.984% - 1.016%] (SEM = 0.0081%) <-- Calibrated 1%
α = 7.5000 ──► Pfa = 0.970%  [Cluster-Robust 95% CI: 0.954% - 0.986%] (SEM = 0.0081%)
α = 8.0000 ──► Pfa = 0.790%  [Cluster-Robust 95% CI: 0.776% - 0.804%] (SEM = 0.0072%)
α = 8.5000 ──► Pfa = 0.649%  [Cluster-Robust 95% CI: 0.636% - 0.662%] (SEM = 0.0065%)
```

### Extended Experiment 3: Fixed-Gain ADC & Quantisation Robustness
- **Objective:** Evaluate unscaled fixed-gain analog front-end operation versus dynamic scaling under increasing clutter amplitude ($A_c \in [0.0..3.0]\,\mathrm{V}$) and gain sweeps ($G \in [500..10000]\,\mathrm{LSB/V}$).
- **Execution Status:** **Fully Executed, Audited, and Documented** ([`adc_fixed_gain_report.md`](docs/objective_1/adc_fixed_gain_report.md)).
- **Dataset Reconciliation (400 Runs):**
  - **180 runs:** Fixed gain sweep across 9 gain settings ($G \in [500..10000]$) $\times$ 20 random seeds.
  - **220 runs:** Clutter amplitude sweep across 11 clutter levels ($A_c \in [0.0..3.0]$) $\times$ 20 random seeds.
  - Total: $180 + 220 = \mathbf{400 \text{ unique simulation runs}}$.
- **Key Measured Results:**
  - **Nominal Operating Point:** $G = 2200.0\,\mathrm{LSB/V}$ achieves $0.0\%$ clipping, an **SQNR of $69.7\,\mathrm{dB}$**, and matches dynamic scaling margins within $0.01\,\mathrm{dB}$.
  - **Saturation & Harmonic Splatter:** When clutter amplitude exceeds $A_c \ge 1.5\,\mathrm{V}$, clipping exceeds $54\%$, SQNR collapses below $10\,\mathrm{dB}$, and third- and fifth-order odd harmonics splatter across the Doppler spectrum, raising the noise floor by **$+5.34\,\mathrm{dB}$** and destroying weak target detection.

### Extended Validation Execution Reconciliation Summary
- **Unique CPI Simulation Runs/Frames Executed:**  
  $$\mathbf{4 \text{ (Exp 1 pilot)}} + \mathbf{1{,}050 \text{ (Exp 2 frames)}} + \mathbf{400 \text{ (Exp 3 runs)}} = \mathbf{1{,}454 \text{ unique CPI simulations}}.$$

### Automated Unit Test Suite & CI Automation
- **Coverage:** 26 automated unit tests across 12 functional DSP areas ([`tests/unit/`](tests/unit/)):
  1. Signal generation & analytical Euler phase verification.
  2. Range & Doppler FFT peak localization (parameterized across 6 test points).
  3. Signed Doppler indexing and `fftshift` mapping.
  4. 12-bit ADC rounding, clamp bounds $[-2048, +2047]$, and saturation.
  5. Hann window symmetry, 1D/2D coherent gain, noise gain, and processing gain.
  6. CA-CFAR guard ($G=2$) and reference ($N_{\text{side}}=4$) cell mask indices.
  7. CFAR valid boundary bin filtering ($16,000$ valid CUT cells).
  8. Zero-noise floor collapse and zero-signal purity.
  9. Deterministic random seed reproducibility across runs.
  10. Fast-time/slow-time I/Q dimension ordering ($16,384$ samples).
  11. End-to-end target-present detection and noise-only $P_{fa}$ bounds.
  12. Fixed-gain vs dynamic-gain SQNR and clipping equivalence.
- **Performance:** **26 passed in 0.56 seconds**.
- **Continuous Integration:** Automated multi-platform matrix testing on GitHub Actions ([`.github/workflows/radar_ci.yml`](.github/workflows/radar_ci.yml)) across `ubuntu-latest` and `windows-latest` for Python 3.10, 3.11, and 3.12.

---

## 7. Known Limitations & Failure Modes

Objective I has formally cataloged and quantified five critical failure modes:

1. **Weak-Target Clutter Masking Region ($|r - 72| \le 7$):**  
   Because CA-CFAR averages reference cells strictly along range without outlier rejection, clutter energy in cells $[r-6..r-3] \cup [r+3..r+6]$ inflates the local threshold by up to $+37.8\,\mathrm{dB}$, completely masking weak targets ($P_d = 0.00$).
2. **Stationary Clutter Doppler Masking ($|d_t - d_c| \le 1.0$ bin):**  
   Slow-moving targets falling inside the Hann Doppler mainlobe of stationary clutter ($d_t = +0.5, +1.0$) suffer negative detection margins ($-13.20\,\mathrm{dB}$ to $-17.26\,\mathrm{dB}$) regardless of range separation.
3. **Fractional Doppler Leakage:**  
   Non-integer clutter Doppler shifts fill the Hann spectral nulls, spreading $+47.6\,\mathrm{dB}$ of clutter sidelobe energy into adjacent bins and reducing detection margins by $16.42\,\mathrm{dB}$.
4. **Hardware Unrealistic Dynamic Rescaling:**  
   The baseline radar model normalizes each frame by peak amplitude ($S = 0.90 / \max(|I|, |Q|)$). In real-world streaming hardware, this causes gain modulation and target suppression.
5. **Zero-Noise Median Collapse:**  
   In ideal noiseless simulation ($A_n = 0.0$), unexcited range-Doppler cells evaluate to $0.0$, causing the global median threshold to collapse to zero.

---

## 8. Multi-Phase Research Roadmap (Objectives I, II, and III)

```
===================================================================================
PHASE 1: OBJECTIVE I — ALGORITHMIC MODEL & STATISTICAL BASELINE [STATUS: COMPLETE]
===================================================================================
 [x] Synthetic pulse-Doppler signal generation (target, clutter, AWGN).
 [x] 12-bit signed ADC quantizer with saturation and dynamic scaling.
 [x] 2D separable Hann windowing and Range-Doppler 2D FFT.
 [x] Dual-stage detection: Global median threshold + 1D Range-Sliding CA-CFAR.
 [x] 217-run baseline experimental campaign and formal audit.
 [x] Extended validation: Empirical CFAR calibration (1,050 frames) & ROC curves.
 [x] Extended validation: Fixed-gain ADC and quantisation robustness (400 runs).
 [x] Fast 26-test unit test suite & multi-OS GitHub Actions CI workflow.
 [x] High-resolution clutter-proximity pilot (4 runs audited; 680-run sweep prepared).

===================================================================================
PHASE 2: OBJECTIVE II — FIXED-POINT RTL HARDWARE ARCHITECTURE [STATUS: PLANNED]
===================================================================================
 [ ] Bit-Accurate Fixed-Point Model:
     • Develop bit-accurate C++/Python fixed-point model (Q1.15 data, Q1.15 twiddles).
     • Quantize window coefficients and evaluate SQNR degradation (< 0.5 dB target).
 [ ] Pre-CFAR Clutter Suppression Filter:
     • Design a 3-pulse MTI canceller or 2nd-order IIR notch filter along slow time.
     • Target: >= 40 dB rejection at DC to eliminate clutter masking before CFAR.
 [ ] Outlier-Robust CFAR Architecture:
     • Replace CA-CFAR with Pipelined Order-Statistic CFAR (OS-CFAR) or GO-CFAR.
     • Implement 2D cross-Doppler reference censoring to protect boundary cells.
 [ ] Digital Front-End Automatic Gain Control (AGC):
     • Implement fast-attack, slow-decay AGC replacing dynamic software scaling.
 [ ] Streaming RTL Verilog/SystemVerilog Core:
     • Pipelined Radix-2/Radix-4 streaming 512-point Range FFT with block floating-point.
     • Corner-turn matrix memory (BRAM) for 512x32 CPI storage.
     • Pipelined 32-point Doppler FFT with integrated windowing and magnitude-squared.
     • SystemVerilog UVM testbench driven by Objective I 16,384-line golden test vectors.

===================================================================================
PHASE 3: OBJECTIVE III — SILICON EVALUATION & HARDWARE TESTING [STATUS: PLANNED]
===================================================================================
 [ ] FPGA Implementation & Synthesis:
     • Target AMD Xilinx Zynq UltraScale+ RFSoC (ZCU111/ZCU208) or Artix-7.
     • Timing closure at >= 200 MHz system clock.
     • Resource utilization audit: DSP48E slices, BRAMs, and LUTs.
 [ ] Real-Time Performance & Power Profiling:
     • Measure total CPI processing latency (target: < 1.0 ms per CPI).
     • Dynamic and static power estimation using Xilinx Power Estimator (XPE).
 [ ] Hardware-in-the-Loop (HIL) Loopback Testing:
     • Stream golden I/Q vectors via DMA over PCIe / AXI4-Stream to FPGA.
     • Verify bit-for-bit exact detection output against Objective I reference data.
```

---

## 9. Repository Architecture

```
radar-objective-1/
├── README.md                               # Academic review & engineering portfolio README
├── requirements.txt                        # Python dependencies (includes pytest>=8.0.0)
├── .gitignore                              # Git exclusion rules (build, venv, large archives)
├── path_mapping.json                       # Reorganization audit map (original -> new paths)
│
├── .github/
│   └── workflows/
│       └── radar_ci.yml                    # GitHub Actions CI matrix (Ubuntu/Windows, Py 3.10-3.12)
│
├── docs/
│   ├── objective_1/                        # Objective I academic documentation & audit reports
│   │   ├── final_report.pdf                # Academic report with 9 publication figures
│   │   ├── one_page_summary.pdf            # 1-page executive summary & consolidated metric table
│   │   ├── final_audit.pdf                 # Formal 10-item audit certificate
│   │   ├── cfar_calibration_roc_report.md  # Extended Exp 2: CFAR calibration & ROC report
│   │   ├── adc_fixed_gain_report.md        # Extended Exp 3: Fixed-gain ADC robustness report
│   │   ├── archive_manifest.csv            # SHA-256 archive manifest for reproducible files
│   │   ├── render_pdfs.py                  # Headless Chrome PDF generation pipeline
│   │   └── figures/                        # High-resolution vector/raster publication plots
│   ├── objective_2/
│   │   └── limitations_handoff.pdf         # Objective II prioritized limitation handoff report
│   └── architecture/                       # Signal processing block diagrams & timing specs
│
├── src/
│   ├── radar_model.py                      # Frozen baseband radar simulation reference model
│   ├── signal_generation/                  # Modular synthetic chirp & target generators
│   ├── detection/                          # CFAR & adaptive detection modules
│   └── utils/                              # Mathematical helpers & DSP utilities
│
├── tests/
│   ├── unit/                               # Fast unit test suite (26 tests, 0.56s)
│   │   ├── conftest.py                     # Shared radar test fixtures & constants
│   │   ├── test_signal_generation.py       # Phase, seed reproducibility, dimensions, zero-noise
│   │   ├── test_fft_and_windowing.py       # FFT peak locations, signed Doppler, Hann gains
│   │   ├── test_adc_quantization.py        # 12-bit ADC, saturation, fixed vs dynamic gain
│   │   ├── test_cfar_detector.py           # Guard/ref masks, boundary bins, clutter masking
│   │   └── test_end_to_end_detection.py    # Target-present, noise-only Pfa, input validation
│   │
│   ├── objective_1/                        # 217-run validated experimental suites
│   │   ├── T01_T05/                        # Benchmark diagnostics (5 runs, seed 42)
│   │   ├── C01_C02/                        # Stationary vs near-zero clutter (2 runs, seed 42)
│   │   ├── step7_multiseed/                # Target+clutter+noise validation (40 runs, seeds 1-10)
│   │   ├── noise_only/                     # Pure AWGN calibration baseline (10 runs, seeds 1-10)
│   │   ├── slow_targets/                   # Slow-target sweeps T06/T07 (40 runs) & runner
│   │   └── range_robustness/               # Range sweeps across 6 bins (120 runs) & runner
│   │
│   └── extended_validation/                # Extended research & edge-case testbenches
│       ├── clutter_proximity_sweep/        # Exp 1 runner (run_clutter_proximity_sweep.py) & pilot
│       ├── cfar_calibration/               # Exp 2 runner, CSVs, and ROC curves
│       └── adc_fixed_gain/                 # Exp 3 runner, CSVs, and clipping/SQNR plots
│
├── analysis/
│   ├── objective_1/                        # Objective I analysis scripts & artifacts
│   │   ├── audit_step7.py                  # Statistical audit script for 50 Step 7 runs
│   │   ├── analyze_clutter_proximity.py    # 210-run clutter proximity failure analysis
│   │   └── clutter_investigation/          # Diagnostic range profiles, TCR curves & CSVs
│   └── scripts/                            # Standalone audit & verification runners
│
├── hardware/
│   └── objective_2/                        # Planned Objective II hardware scaffold
│       ├── rtl/                            # Planned: Verilog/SystemVerilog RTL implementations (.gitkeep)
│       ├── testbench/                      # Planned: UVM / Cocotb simulation testbenches (.gitkeep)
│       ├── reference/                      # Planned: Bit-accurate C++/Python hardware models (.gitkeep)
│       ├── test_vectors/                   # Planned: Golden 16,384-line I/Q test vectors (.gitkeep)
│       └── documentation/                  # Planned: Register maps & interface timing specs (.gitkeep)
│
├── configs/
│   ├── objective_1/                        # Objective I JSON parameter configurations
│   └── extended_validation/                # Extended study parameter profiles (Exp 1, 2, 3)
│
└── outputs/                                # Scratch directory for active run outputs (.gitkeep)
```

---

## 10. Installation & Reproduction Instructions

### Prerequisites
- Python 3.10, 3.11, or 3.12.
- Git.
- Recommended: Windows PowerShell or Linux Bash.

### Step 1: Clone Repository & Set Up Environment

```bash
git clone git@github.com:HanAbhi/radar-objective-1.git
cd radar-objective-1

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install required dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 2: Execute Automated Unit Tests

Run the complete 26-test suite verifying all 12 functional areas:

```bash
pytest tests/unit/ -v --tb=short
```

Expected output: `26 passed in ~0.6s`.

### Step 3: Run the Frozen Radar Reference Model

Execute a single CPI simulation run with default parameters ($N_R=512, N_D=32$):

```bash
python src/radar_model.py
```

### Step 4: Reproduce Baseline Objective I Experiments

```bash
# Reproduce Slow Targets T06 & T07 (40 runs):
python tests/objective_1/slow_targets/run_slow_target_experiments.py

# Reproduce Target Range Robustness across 6 bins (120 runs):
python tests/objective_1/range_robustness/run_range_robustness.py

# Run statistical audit of Step 7 (50 runs):
python analysis/scripts/audit_step7.py
```

### Step 5: Reproduce Extended Validation Experiments

```bash
# Reproduce Extended Experiment 2 (CFAR Calibration & ROC Analysis):
python tests/extended_validation/cfar_calibration/run_cfar_calibration.py

# Reproduce Extended Experiment 3 (Fixed-Gain ADC & Quantisation Robustness):
python tests/extended_validation/adc_fixed_gain/run_adc_fixed_gain.py

# Run Extended Experiment 1 Pilot (Clutter-Proximity Sweep, 4 runs):
python tests/extended_validation/clutter_proximity_sweep/run_clutter_proximity_sweep.py --pilot
```

---

## 11. Configuration & Seed Management Policy

To guarantee 100% scientific reproducibility and prevent stochastic drift, all experiments are governed by centralized JSON configuration files in [`configs/`](configs/) and strict random seed partitioning:

- **Seed 42:** Reserved strictly for deterministic benchmark runs (T01–T05, C01–C02) to establish repeatable, noise-isolated ground truth.
- **Seeds 1–10:** Assigned to Objective I multi-seed campaigns (Step 7 target-present, Step 7 noise-only, T06/T07 slow targets, range robustness, and Extended Experiment 1).
- **Seeds 101–200:** Assigned exclusively to independent noise-only calibration frames in Extended Experiment 2 ($1.6 \times 10^6$ cells).
- **Seeds 201–400:** Assigned exclusively to independent noise-only validation frames in Extended Experiment 2 ($3.2 \times 10^6$ cells).
- **Seeds 501–520:** Assigned to the 20 Monte Carlo iterations of Extended Experiment 3 fixed-gain sweeps.

All random number generators instantiate explicit local instances (`np.random.RandomState(seed)` or `np.random.default_rng(seed)`), preventing global RNG state contamination.

---

## 12. Primary Reports, Audit Certificates & Data Links

All academic documentation, peer-review artifacts, and audit certificates are organized in [`docs/`](docs/):

- **Objective I Final Academic Validation Report:**  
  [`docs/objective_1/final_report.pdf`](docs/objective_1/final_report.pdf) — Comprehensive 18-section academic report detailing mathematical derivations, 217-run empirical analysis, and 9 high-resolution figures.
- **One-Page Executive Summary:**  
  [`docs/objective_1/one_page_summary.pdf`](docs/objective_1/one_page_summary.pdf) — High-level technical summary with consolidated metric tables for academic project review.
- **Objective I Formal Audit Certificate:**  
  [`docs/objective_1/final_audit.pdf`](docs/objective_1/final_audit.pdf) — Item-by-item verification certificate reconciling all 217 baseline runs, window gain derivations, and theoretical $P_{fa}$ models.
- **Objective II Limitations & RTL Handoff Report:**  
  [`docs/objective_2/limitations_handoff.pdf`](docs/objective_2/limitations_handoff.pdf) — Engineering specification defining the five simulation failure modes and detailing the RTL requirements for FPGA implementation.
- **Extended Experiment 2: CFAR Calibration & ROC Report:**  
  [`docs/objective_1/cfar_calibration_roc_report.md`](docs/objective_1/cfar_calibration_roc_report.md) — Complete calibration curves, ROC figures, and sensitivity trade-off analysis across 1,050 frames.
- **Extended Experiment 3: Fixed-Gain ADC Robustness Report:**  
  [`docs/objective_1/adc_fixed_gain_report.md`](docs/objective_1/adc_fixed_gain_report.md) — Comprehensive clipping, SQNR, and spectral splatter analysis under unscaled fixed-gain stages.
- **Archive Manifest & Data Integrity:**  
  [`docs/objective_1/archive_manifest.csv`](docs/objective_1/archive_manifest.csv) — Cryptographic SHA-256 verification manifest cataloging reproducible bulk raw I/Q archives.
