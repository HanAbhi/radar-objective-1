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
