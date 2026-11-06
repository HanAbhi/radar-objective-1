import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# ---------------------------------------------------------
# 1. SETTINGS
# ---------------------------------------------------------

NR = 256
ND = 16

TARGET_RANGE_BIN = 70
TARGET_DOPPLER_BIN = 2
TARGET_AMPLITUDE = 0.10

CLUTTER_RANGE_BIN = 72
CLUTTER_AMPLITUDE = 0.60

# Standard deviation of each noise component
NOISE_AMPLITUDE = 0.08

THRESHOLD_FACTOR = 6.0

REFERENCE_CELLS_PER_SIDE = 4
GUARD_CELLS_PER_SIDE = 2
CFAR_ALPHA = 6.0

ADC_BITS = 12

# True: display graphs after all files and results are ready.
# False: save graphs without opening windows.
SHOW_PLOTS = True

np.random.seed(42)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. VALIDATE SETTINGS
# ---------------------------------------------------------

if NR < 4 or ND < 4:
    raise ValueError("NR and ND must each be at least 4.")

if not 0 <= TARGET_RANGE_BIN < NR:
    raise ValueError("TARGET_RANGE_BIN must be between 0 and NR - 1.")

if not 0 <= CLUTTER_RANGE_BIN < NR:
    raise ValueError("CLUTTER_RANGE_BIN must be between 0 and NR - 1.")

if not -(ND // 2) <= TARGET_DOPPLER_BIN <= (ND - 1) // 2:
    raise ValueError("TARGET_DOPPLER_BIN is outside the valid range.")

if REFERENCE_CELLS_PER_SIDE < 1:
    raise ValueError("At least one reference cell per side is required.")

if GUARD_CELLS_PER_SIDE < 0:
    raise ValueError("The number of guard cells cannot be negative.")

if THRESHOLD_FACTOR <= 0 or CFAR_ALPHA <= 0:
    raise ValueError("Threshold multipliers must be positive.")

if not 2 <= ADC_BITS <= 16:
    raise ValueError("ADC_BITS must be between 2 and 16.")

edge_size = (
    REFERENCE_CELLS_PER_SIDE
    + GUARD_CELLS_PER_SIDE
)

if NR <= 2 * edge_size:
    raise ValueError("NR is too small for the selected CFAR window.")


# ---------------------------------------------------------
# 3. CREATE SAMPLE AND CHIRP INDICES
# ---------------------------------------------------------

n = np.arange(NR)
m = np.arange(ND)

n_grid, m_grid = np.meshgrid(n, m)


# ---------------------------------------------------------
# 4. GENERATE TARGET, CLUTTER AND NOISE
# ---------------------------------------------------------

target_phase = 2 * np.pi * (
    TARGET_RANGE_BIN * n_grid / NR
    + TARGET_DOPPLER_BIN * m_grid / ND
)

target_signal = TARGET_AMPLITUDE * np.exp(1j * target_phase)

clutter_phase = 2 * np.pi * (
    CLUTTER_RANGE_BIN * n_grid / NR
)

clutter_signal = CLUTTER_AMPLITUDE * np.exp(1j * clutter_phase)

noise = NOISE_AMPLITUDE * (
    np.random.randn(ND, NR)
    + 1j * np.random.randn(ND, NR)
)

received_signal = target_signal + clutter_signal + noise


# ---------------------------------------------------------
# 5. QUANTIZE INPUT TO SIGNED 12-BIT I/Q
# ---------------------------------------------------------

ADC_MAX = 2 ** (ADC_BITS - 1) - 1
ADC_MIN = -(2 ** (ADC_BITS - 1))

maximum_value = max(
    np.max(np.abs(received_signal.real)),
    np.max(np.abs(received_signal.imag))
)

if maximum_value > 0:
    scaling_factor = 0.90 * ADC_MAX / maximum_value
else:
    scaling_factor = 1.0

i_samples = np.clip(
    np.round(received_signal.real * scaling_factor),
    ADC_MIN,
    ADC_MAX
).astype(np.int16)

q_samples = np.clip(
    np.round(received_signal.imag * scaling_factor),
    ADC_MIN,
    ADC_MAX
).astype(np.int16)

# Use the quantized input for the software reference.
# Dividing by the scale restores approximate original units.
reference_input = (
    i_samples.astype(np.float64)
    + 1j * q_samples.astype(np.float64)
) / scaling_factor


# ---------------------------------------------------------
# 6. SAVE FPGA INPUT AND SCALING INFORMATION
# ---------------------------------------------------------

# Chirp-major ordering:
# all samples of chirp 0, then all samples of chirp 1, etc.
iq_data = np.column_stack((
    i_samples.ravel(),
    q_samples.ravel()
))

np.savetxt(
    OUTPUT_DIR / "radar_input_iq.txt",
    iq_data,
    fmt="%d"
)

np.savez(
    OUTPUT_DIR / "simulation_metadata.npz",
    nr=NR,
    nd=ND,
    adc_bits=ADC_BITS,
    scaling_factor=scaling_factor,
    target_range_bin=TARGET_RANGE_BIN,
    target_doppler_bin=TARGET_DOPPLER_BIN
)


# ---------------------------------------------------------
# 7. RANGE WINDOW AND RANGE FFT
# ---------------------------------------------------------

range_window = np.hanning(NR)

windowed_signal = reference_input * range_window[None, :]

range_fft = np.fft.fft(
    windowed_signal,
    axis=1
)


# ---------------------------------------------------------
# 8. DOPPLER WINDOW AND DOPPLER FFT
# ---------------------------------------------------------

doppler_window = np.hanning(ND)

range_fft_windowed = range_fft * doppler_window[:, None]

doppler_fft = np.fft.fft(
    range_fft_windowed,
    axis=0
)

range_doppler_map = np.fft.fftshift(
    doppler_fft,
    axes=0
)


# ---------------------------------------------------------
# 9. MAGNITUDE-SQUARED POWER
# ---------------------------------------------------------

power_map = np.abs(range_doppler_map) ** 2
power_map_db = 10 * np.log10(power_map + 1e-12)

target_doppler_shifted = (
    TARGET_DOPPLER_BIN + ND // 2
) % ND

clutter_doppler_shifted = ND // 2

target_power = power_map[
    target_doppler_shifted,
    TARGET_RANGE_BIN
]


# ---------------------------------------------------------
# 10. GLOBAL-THRESHOLD DETECTOR
# ---------------------------------------------------------

# Recomputed from each frame; not a constant baseline threshold.
estimated_noise_power = np.median(power_map)

fixed_threshold = THRESHOLD_FACTOR * estimated_noise_power

fixed_detections = power_map > fixed_threshold

target_detected = bool(
    fixed_detections[
        target_doppler_shifted,
        TARGET_RANGE_BIN
    ]
)


# ---------------------------------------------------------
# 11. CA-CFAR DETECTOR ALONG RANGE
# ---------------------------------------------------------

cfar_detections = np.zeros((ND, NR), dtype=bool)
cfar_threshold_map = np.full((ND, NR), np.nan)
cfar_noise_map = np.full((ND, NR), np.nan)

cfar_valid_mask = np.zeros((ND, NR), dtype=bool)
cfar_valid_mask[:, edge_size:NR - edge_size] = True

for doppler_index in range(ND):
    for range_index in range(edge_size, NR - edge_size):

        left_start = (
            range_index
            - GUARD_CELLS_PER_SIDE
            - REFERENCE_CELLS_PER_SIDE
        )

        left_stop = range_index - GUARD_CELLS_PER_SIDE

        right_start = (
            range_index + GUARD_CELLS_PER_SIDE + 1
        )

        right_stop = (
            right_start + REFERENCE_CELLS_PER_SIDE
        )

        reference_cells = np.concatenate((
            power_map[doppler_index, left_start:left_stop],
            power_map[doppler_index, right_start:right_stop]
        ))

        local_noise_estimate = np.mean(reference_cells)
        local_threshold = CFAR_ALPHA * local_noise_estimate

        cfar_noise_map[
            doppler_index, range_index
        ] = local_noise_estimate

        cfar_threshold_map[
            doppler_index, range_index
        ] = local_threshold

        cfar_detections[
            doppler_index, range_index
        ] = (
            power_map[doppler_index, range_index]
            > local_threshold
        )

target_cell_is_tested = bool(
    cfar_valid_mask[
        target_doppler_shifted,
        TARGET_RANGE_BIN
    ]
)

if target_cell_is_tested:
    cfar_target_detected = bool(
        cfar_detections[
            target_doppler_shifted,
            TARGET_RANGE_BIN
        ]
    )

    cfar_target_threshold = cfar_threshold_map[
        target_doppler_shifted,
        TARGET_RANGE_BIN
    ]
else:
    cfar_target_detected = None
    cfar_target_threshold = np.nan


# ---------------------------------------------------------
# 12. SAVE NUMERICAL REFERENCE MAPS
# ---------------------------------------------------------

reference_maps = {
    "reference_power_map.csv": power_map,
    "reference_cfar_noise_map.csv": cfar_noise_map,
    "reference_cfar_threshold_map.csv": cfar_threshold_map
}

for filename, data in reference_maps.items():
    np.savetxt(
        OUTPUT_DIR / filename,
        data,
        delimiter=","
    )

detection_maps = {
    "reference_global_detection_map.csv": fixed_detections,
    "reference_cfar_detection_map.csv": cfar_detections,
    "reference_cfar_valid_mask.csv": cfar_valid_mask
}

for filename, data in detection_maps.items():
    np.savetxt(
        OUTPUT_DIR / filename,
        data.astype(int),
        delimiter=",",
        fmt="%d"
    )


# ---------------------------------------------------------
# 13. CREATE AND SAVE GRAPHS
# ---------------------------------------------------------

def create_map_figure(title, filename, detections=None):
    fig, ax = plt.subplots(figsize=(12, 6))

    image = ax.imshow(
        power_map_db,
        aspect="auto",
        origin="lower",
        cmap="jet"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Power (dB, arbitrary reference)"
    )

    if detections is not None:
        doppler_indices, range_indices = np.where(detections)

        ax.scatter(
            range_indices,
            doppler_indices,
            facecolors="none",
            edgecolors="lime",
            s=80,
            linewidths=1.5,
            label="Cells above threshold"
        )

    ax.plot(
        TARGET_RANGE_BIN,
        target_doppler_shifted,
        marker="x",
        color="white",
        markersize=12,
        markeredgewidth=2,
        linestyle="none",
        label="Expected moving target"
    )

    ax.plot(
        CLUTTER_RANGE_BIN,
        clutter_doppler_shifted,
        marker="+",
        color="magenta",
        markersize=12,
        markeredgewidth=2,
        linestyle="none",
        label="Expected stationary clutter"
    )

    ax.set_xlabel("Range Bin")
    ax.set_ylabel("Shifted Doppler Index")
    ax.set_title(title)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=200)

    return fig


create_map_figure(
    "Radar Range-Doppler Map: Target + Clutter + Noise",
    "range_doppler_map.png"
)

create_map_figure(
    "Simple Global-Threshold Detection",
    "fixed_threshold_detection.png",
    fixed_detections
)

create_map_figure(
    "Adaptive CA-CFAR Detection",
    "cfar_detection.png",
    cfar_detections
)


# ---------------------------------------------------------
# 14. SHORT TERMINAL SUMMARY
# ---------------------------------------------------------

summary_lines = [
    "RADAR SIMULATION SUMMARY",
    f"Frame: {ND} chirps x {NR} samples = {ND * NR} complex samples",
    (
        f"Target: range bin {TARGET_RANGE_BIN}, "
        f"signed Doppler bin {TARGET_DOPPLER_BIN}, "
        f"shifted index {target_doppler_shifted}"
    ),
    (
        f"Global detector: target={target_detected}, "
        f"cells above threshold={int(fixed_detections.sum())}"
    ),
    (
        f"CFAR detector: target="
        f"{cfar_target_detected if target_cell_is_tested else 'NOT TESTED'}, "
        f"cells above threshold={int(cfar_detections.sum())}"
    ),
    f"Target power: {target_power:.6f}",
    f"Global threshold: {fixed_threshold:.6f}",
    f"CFAR threshold at target: {cfar_target_threshold:.6f}",
    (
        f"FPGA input: {ADC_BITS}-bit signed I/Q, "
        f"{ND * NR} lines, scaling factor={scaling_factor:.6f}"
    ),
    "Counts represent cells above threshold, not separate targets.",
    f"All files saved in: {OUTPUT_DIR}"
]

summary_text = "\n".join(summary_lines)

(OUTPUT_DIR / "simulation_summary.txt").write_text(
    summary_text,
    encoding="utf-8"
)

print("\n" + summary_text + "\n", flush=True)


# ---------------------------------------------------------
# 15. DISPLAY GRAPHS ONLY AFTER ALL WORK IS COMPLETE
# ---------------------------------------------------------

if SHOW_PLOTS:
    print(
        "Results are ready. Close the graph windows to exit.",
        flush=True
    )

    try:
        plt.show()
    except KeyboardInterrupt:
        pass
    finally:
        plt.close("all")
else:
    plt.close("all")