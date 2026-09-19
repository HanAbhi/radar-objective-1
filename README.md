# Radar Objective-I Reference Model

## Run

1. Open this folder in VS Code.
2. Create/activate a Python environment if desired.
3. Install packages:

```bash
pip install -r requirements.txt
```

4. Run:

```bash
python radar_model.py
```

The program generates synthetic complex I/Q radar data containing a weak moving target, stationary clutter, near-zero-Doppler clutter, and AWGN. It performs range and Doppler FFT processing, fixed-threshold detection, CA-CFAR detection, plots, CSV references, and FPGA-ready input files.

Generated files are written to `outputs/`.
