# Radar Signal Processing – Objective I: Simulation & Validation Baseline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Objective I Complete](https://img.shields.io/badge/Status-Objective%20I%20Frozen-success.svg)]()

A high-fidelity Python baseband radar signal processing reference model and 217-run experimental validation suite for weak and slow-moving target detection in stationary and near-zero-Doppler clutter environments.

---

## Repository Directory Structure

The repository is organised into a modular research and engineering layout separating the frozen Objective I baseline, documentation, analysis scripts, configurations, and future Objective II hardware/RTL scaffolds:

```
radar-objective-1/
├── README.md                               # Project documentation & reference index
├── requirements.txt                        # Python dependencies
├── .gitignore                              # Git exclusion rules
├── path_mapping.json                       # Comprehensive migration map (old paths -> new paths)
│
├── docs/
│   ├── objective_1/                        # Objective I reports & verification documents
│   │   ├── final_report.pdf                # Comprehensive academic report with 9 figures
│   │   ├── one_page_summary.pdf            # 1-page executive summary & consolidated metric table
│   │   ├── final_audit.pdf                 # Formal 10-item audit & verification certificate
│   │   ├── render_pdfs.py                  # HTML/Chrome headless PDF generation pipeline
│   │   └── figures/                        # High-resolution publication plots (9 figures)
│   ├── objective_2/
│   │   └── limitations_handoff.pdf         # 1-page prioritized technical limitation handoff
│   └── architecture/                       # System block diagrams & specification sheets
│
├── src/
│   ├── radar_model.py                      # Frozen baseband radar simulation reference model
│   ├── signal_generation/                  # Modular synthetic chirp & target generators
│   ├── detection/                          # CFAR & adaptive detection modules
│   └── utils/                              # Mathematical helpers & DSP utilities
│
├── tests/
│   ├── objective_1/                        # 217-run validated experimental suites
│   │   ├── T01_T05/                        # Benchmark diagnostics (5 runs, seed 42)
│   │   ├── C01_C02/                        # Stationary vs near-zero clutter (2 runs, seed 42)
│   │   ├── step7_multiseed/                # Target+clutter+noise validation (40 runs, seeds 1-10)
│   │   ├── noise_only/                     # Pure AWGN calibration baseline (10 runs, seeds 1-10)
│   │   ├── slow_targets/                   # Slow-target sweeps T06/T07 (40 runs) & runner
│   │   └── range_robustness/               # Range sweeps across 6 bins (120 runs) & runner
│   ├── extended_validation/                # Extended research & edge-case testbenches
│   │   ├── clutter_proximity_sweep/
│   │   ├── cfar_calibration/
│   │   ├── adc_fixed_gain/
│   │   ├── boundary_tests/
│   │   ├── fractional_bin_tests/
│   │   ├── multi_target_tests/
│   │   └── detector_comparison/
│   └── unit/                               # Fast unit tests for DSP blocks
│
├── analysis/
│   ├── objective_1/                        # Objective I analysis scripts & artifacts
│   │   ├── audit_step7.py                  # Full statistical audit script for 50 Step 7 runs
│   │   ├── analyze_clutter_proximity.py    # 210-run clutter proximity failure analysis
│   │   └── clutter_investigation/          # Diagnostic range profiles, TCR curves & CSVs
│   ├── extended_validation/                # Extended study analyses
│   └── scripts/                            # Standalone audit & verification runners
│       ├── audit_step7.py
│       └── analyze_clutter_proximity.py
│
├── hardware/
│   └── objective_2/                        # Objective II FPGA/ASIC hardware transition
│       ├── rtl/                            # Verilog/SystemVerilog RTL implementations
│       ├── testbench/                      # UVM / Cocotb simulation testbenches
│       ├── reference/                      # Bit-accurate C/Python hardware models
│       ├── test_vectors/                   # Golden I/Q test vectors & outputs
│       └── documentation/                  # Register maps & interface timing specs
│
├── configs/
│   ├── objective_1/                        # Objective I YAML/JSON parameter configs
│   └── extended_validation/                # Extended study parameter profiles
│
└── outputs/                                # Scratch directory for active run outputs
    ├── .gitkeep
    └── README.md
```

---

## Signal Processing Architecture & Parameters

The simulation models a coherent pulse-Doppler radar receiver operating over a single Coherent Processing Interval (CPI):

| Parameter | Symbol | Value | Description |
|---|:---:|:---:|---|
| Fast-time range samples / FFT size | $N_R$ | `512` | Samples per chirp pulse |
| Slow-time chirps / Doppler FFT size | $N_D$ | `32` | Chirps per Coherent Processing Interval (CPI) |
| Total CPI data cube | $N_R \times N_D$ | `16,384` | Complex baseband samples |
| ADC Quantization | $B$ | `12` bits | Signed two's-complement $[-2048, +2047]$ I/Q |
| Headroom Scaling | $S$ | `0.90` | ADC full-scale utilization |
| Windowing | $w_r, w_d$ | `np.hanning` | Separable symmetric Hann tapering along range and Doppler |
| Global Threshold Multiplier | $K_{\text{glob}}$ | `6.0` | Multiplies global frame median of power |
| CA-CFAR Guard Bins per Side | $G$ | `2` | Total 4 guard cells around CUT |
| CA-CFAR Reference Bins per Side | $N_{\text{side}}$ | `4` | Total $N_{\text{ref}} = 8$ reference cells along range |
| CFAR Evaluation Range | $[E, N_R - E - 1]$ | `[6, 505]` | Exactly `16,000` valid cells evaluated per CPI |
| CA-CFAR Multiplier | $\alpha_{\text{CFAR}}$ | `6.0` | Multiplies local arithmetic mean of reference cells |

### Window Gains & Normalization
Because NumPy's `np.fft.fft` uses unnormalized discrete Fourier transform sums, power values scale with the 2D window coherent gain:
- **1D Range Hann ($N_R = 512$):** $\sum w_r = 255.5$, $G_{c,r} = 65{,}280.25$ ($+48.15\,\mathrm{dB}$), $G_{n,r} = 191.625$ ($+22.83\,\mathrm{dB}$), $\mathrm{PG}_r = 340.67$ ($+25.32\,\mathrm{dB}$).
- **1D Doppler Hann ($N_D = 32$):** $\sum w_d = 15.5$, $G_{c,d} = 240.25$ ($+23.81\,\mathrm{dB}$), $G_{n,d} = 11.625$ ($+10.65\,\mathrm{dB}$), $\mathrm{PG}_d = 20.67$ ($+13.15\,\mathrm{dB}$).
- **2D Product Hann ($512 \times 32$):** $G_c = 15{,}683{,}580.0625$ ($+71.95\,\mathrm{dB}$), $G_n = 2{,}227.640625$ ($+33.48\,\mathrm{dB}$), $\mathbf{\mathrm{PG} = 7{,}040.4444} \;\; (\mathbf{+38.48\,\mathrm{dB}})$.
- **Output Peak Scaling:** An on-grid tone $A e^{j\phi}$ produces peak RD power $P_{\text{peak}} = A^2 G_c$ ($A=0.60 \to 5.65 \times 10^6$; $A=0.10 \to 156.8 \times 10^3$; $A=0.03 \to 14.12 \times 10^3$).

---

## 217-Run Validation Campaign Summary

Across Objective I, a total of **217 simulation runs** were executed and audited (7 deterministic benchmark runs + 210 multi-seed sweep runs). Every single run generated an exact 16,384-line 12-bit signed I/Q file and comprehensive CSV metadata:

| Test Suite | Location | Runs | Parameters Tested | Key Findings |
|---|---|:---:|---|---|
| **T01–T05 Diagnostics** | `tests/objective_1/T01_T05/` | 5 | Isolated target, clutter, and noise baselines (seed 42) | Confirmed linearity, 2D Hann gains, and identified zero-noise median collapse under $A_n = 0$. |
| **C01–C02 Clutter Tests** | `tests/objective_1/C01_C02/` | 2 | Stationary ($d_c = 0.0$) vs. near-zero-Doppler ($d_c = 0.3$) clutter | Fractional clutter shifts off-grid, boosting Doppler $+2$ leakage by **75.4×** ($759 \to 57{,}825$). |
| **Step 7 Target-Present** | `tests/objective_1/step7_multiseed/` | 40 | Moderate ($0.10$) & weak ($0.03$) targets at $R_t=70, d_t=+2.0$, seeds 1–10 | Weak targets suffer a **16.4 dB CFAR margin collapse** under $d_c = 0.3$ ($+17.46 \to +1.04\,\mathrm{dB}$, min $+0.76\,\mathrm{dB}$). |
| **Step 7 Noise-Only** | `tests/objective_1/noise_only/` | 10 | Pure circular complex AWGN ($\sigma_n = 0.08$, seeds 1–10) | Global $P_{fa} = 1.589\% \pm 0.162\%$ (theory $1.563\%$). CFAR $P_{fa} = 1.923\% \pm 0.136\%$ ($N_{\text{eff}} \approx 4.714$ predicts $2.09\%$). |
| **Slow Targets T06–T07** | `tests/objective_1/slow_targets/` | 40 | Weak targets at $d_t = +1.0$ (T06) and $d_t = +0.5$ (T07) at $R_t=70$ | Exact CA-CFAR $P_d = \mathbf{0.00}$ (margins $-13.20$ to $-17.26\,\mathrm{dB}$) due to stationary clutter Hann mainlobe overlap. |
| **Target Range Robustness** | `tests/objective_1/range_robustness/` | 120 | 6 ranges $\{40, 70, 120, 200, 300, 450\}$, 2 target Dop, 2 clutter Dop, 5 seeds | **Observed 5/5 detection** across all 100 clutter-free runs ($R \neq 70$, margins $+17.3$ to $+20.0\,\mathrm{dB}$). Failures strictly isolated to $R_t = 70$. |
| **Clutter Proximity Analysis** | `analysis/objective_1/clutter_investigation/` | — | High-resolution sweep across bins $64–80$ ($210$-run synthesis) | Pinpointed a **15-bin clutter-degraded masking region** for weak targets ($|r - 72| \le 7$, $r \in [65, 79]$); full recovery requires $|r - 72| \ge 8$. |

---

## Key Technical Takeaways & Clutter Degradation Profile

1. **Weak-Target Clutter-Degraded Masking Region ($|r - 72| \le 7$):**  
   Because CA-CFAR averages 8 reference cells strictly along range without outlier censoring, any single clutter mainlobe bin ($\{71, 72, 73\}$) inside $[r-6..r-3] \cup [r+3..r+6]$ inflates the local threshold to $\sim 294{,}000–580{,}000$ (a 40×–2,000× increase). This desensitizes the detector to weak targets ($A_t = 0.03$, power $\approx 14{,}115$), causing negative margins ($-13.2$ to $-23.9\,\mathrm{dB}$). Moderate targets ($A_t = 0.10$) retain positive margin ($+11.54\,\mathrm{dB}$) in the same cells.
2. **Near-Zero-Doppler Clutter Leakage ($d_c = 0.3$):**  
   Moving clutter off the integer grid fills the Hann spectral nulls, spreading $+47.6\,\mathrm{dB}$ of clutter leakage into Doppler bin $+2$, eroding $16.4\,\mathrm{dB}$ of CFAR margin.
3. **Metric Discrimination:**  
   Unmasked $3\times3$ target neighborhood metrics falsely pick up the range 71 clutter shoulder ($P_d = 1.00$). Masking clutter columns $[71..73]$ proves true exact-cell target detection is $0.00$ for slow weak targets.

---

## Getting Started

### 1. Environment Setup

```bash
# Clone repository
git clone git@github.com:HanAbhi/radar-objective-1.git
cd radar-objective-1

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1    # On Windows PowerShell
source .venv/bin/activate       # On Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

### 2. Running the Reference Model

To run the baseline software radar model:
```bash
python src/radar_model.py
```

### 3. Running Test Suites & Experiments

```bash
# Run slow-target validation (T06 & T07, 40 runs)
python tests/objective_1/slow_targets/run_slow_target_experiments.py

# Run target range robustness sweeps (120 runs)
python tests/objective_1/range_robustness/run_range_robustness.py
```

### 4. Running Verification & Audits

```bash
# Run statistical audit of Step 7 (50 runs)
python analysis/scripts/audit_step7.py

# Run clutter proximity failure investigation (210 runs)
python analysis/scripts/analyze_clutter_proximity.py

# Re-render all publication PDFs
python docs/objective_1/render_pdfs.py
```

---

## Academic Documentation & Reports

All formal documentation, executive summaries, and audit reports are available under [`docs/`](docs/):
- **Final Validation Report:** [`docs/objective_1/final_report.pdf`](docs/objective_1/final_report.pdf)
- **One-Page Executive Summary:** [`docs/objective_1/one_page_summary.pdf`](docs/objective_1/one_page_summary.pdf)
- **Comprehensive Audit Report:** [`docs/objective_1/final_audit.pdf`](docs/objective_1/final_audit.pdf)
- **Objective II Limitations Handoff:** [`docs/objective_2/limitations_handoff.pdf`](docs/objective_2/limitations_handoff.pdf)

---

## Transition to Objective II

Objective I is formally frozen. The verified simulation gaps carried into Objective II are:
1. **Pre-CFAR Clutter Suppression:** Implement an MTI (Moving Target Indicator) filter or zero-Doppler notch filter to suppress clutter before Doppler FFT.
2. **Outlier-Robust CFAR:** Implement Order-Statistic CFAR (OS-CFAR), Greatest-Of/Smallest-Of (GO/SO-CFAR), or 2D Doppler-excluding CFAR.
3. **Fixed-Point RTL Architecture:** Design bit-accurate fixed-point FFT/CFAR datapaths for FPGA implementation.
4. **Adaptive CFAR Tuning:** Mitigate Hann window correlation loss and straddle losses.
