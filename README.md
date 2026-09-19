# Radar Signal Processing – Objective I

A Python-based radar simulation project exploring range-Doppler processing, clutter, noise, and target detection.

## Overview

This project generates synthetic radar I/Q signals containing a moving target, stationary clutter, near-zero-Doppler clutter, random clutter, and additive white Gaussian noise (AWGN).

The received signal is processed using range and Doppler FFTs to generate a range-Doppler map. Fixed-threshold and CA-CFAR-style detection methods are used to investigate target detection in the presence of clutter and noise.

## Features

- Synthetic radar I/Q signal generation
- Moving target and clutter simulation
- Additive white Gaussian noise (AWGN)
- 12-bit I/Q quantization
- Range FFT and Doppler FFT processing
- Range-Doppler map visualization
- Fixed-threshold detection
- CA-CFAR-style detection
- Export of reference data for future FPGA implementation

## Configuration

| Parameter | Value |
|---|---:|
| Range FFT size | 256 |
| Doppler FFT size | 16 |
| Total samples | 4096 |
| Target range bin | 70 |
| Target Doppler bin | 2 |
| Target SNR | -5 dB |
| ADC representation | 12-bit |

## Technologies

- Python
- NumPy
- SciPy
- Pandas
- Matplotlib

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/radar-objective-1.git
cd radar-objective-1

2. Create a virtual environment
python -m venv .venv

Activate it on Windows:
.\.venv\Scripts\Activate.ps1

3. Install dependencies
python -m pip install -r requirements.txt

4. Run the simulation
python radar_model.py

Outputs
The simulation generates an outputs/ folder containing:
- Range-Doppler map
- Fixed-threshold detection plot
- CA-CFAR detection plot
- Quantized I/Q input samples
- Packed hexadecimal I/Q data
- Reference range-Doppler results
- Detection results in CSV format
Current Status
Objective I includes synthetic radar signal generation, range-Doppler processing, visualization, and reference data export.
Target detection is currently under investigation. The expected target is not yet detected at its exact range-Doppler coordinate.
Future Work
- Debug and validate target detection
- Evaluate performance under different noise and clutter conditions
- Study probability of detection and false alarm rate
- Prepare reference data for FPGA implementation
- Compare software and FPGA results



4. Replace `YOUR-USERNAME` with your GitHub username.
5. Press `Ctrl + S`.

**Then push it to GitHub:**

```powershell
git add README.md
git commit -m "Add project README"
git push
