import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal.windows import hann

# ---------------- Configuration ----------------
SEED = 42
NR = 256                         # samples per chirp / range FFT size
ND = 16                          # chirps per frame / Doppler FFT size
ADC_BITS = 12
TARGET_RANGE_BIN = 70
TARGET_DOPPLER_BIN = 2
TARGET_AMPLITUDE = 0.08
TARGET_SNR_DB = -5.0
STATIONARY_CLUTTER_RANGE_BIN = 72
STATIONARY_CLUTTER_AMPLITUDE = 0.65
NEAR_ZERO_CLUTTER_RANGE_BIN = 45
NEAR_ZERO_CLUTTER_DOPPLER = 0.30
NEAR_ZERO_CLUTTER_AMPLITUDE = 0.30
NUM_RANDOM_CLUTTER = 4
RANDOM_CLUTTER_AMPLITUDE_MIN = 0.03
RANDOM_CLUTTER_AMPLITUDE_MAX = 0.10
CFAR_REFERENCE_CELLS_PER_SIDE = 4
CFAR_GUARD_CELLS_PER_SIDE = 2
CFAR_ALPHA = 6.0
FIXED_THRESHOLD_FACTOR = 6.0
OUT = Path('outputs')
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(SEED)


def tone(range_bin, doppler_bin, amplitude):
    n = np.arange(NR)[None, :]
    m = np.arange(ND)[:, None]
    phase = 2*np.pi*(range_bin*n/NR + doppler_bin*m/ND)
    return amplitude*np.exp(1j*phase)


def add_awgn(signal):
    signal_power = TARGET_AMPLITUDE**2
    noise_power = signal_power / (10**(TARGET_SNR_DB/10))
    sigma = np.sqrt(noise_power/2)
    noise = sigma*(rng.standard_normal(signal.shape)+1j*rng.standard_normal(signal.shape))
    return signal+noise, noise_power


def quantize_iq(signal):
    max_int = 2**(ADC_BITS-1)-1
    min_int = -(2**(ADC_BITS-1))
    peak = max(np.max(np.abs(signal.real)), np.max(np.abs(signal.imag)), 1e-12)
    scale = 0.90*max_int/peak
    i = np.clip(np.rint(signal.real*scale), min_int, max_int).astype(np.int16)
    q = np.clip(np.rint(signal.imag*scale), min_int, max_int).astype(np.int16)
    return i, q, scale


def pack_iq(i, q):
    # 32-bit word: upper 16 bits I, lower 16 bits Q
    iu = int(i) & 0xffff
    qu = int(q) & 0xffff
    return (iu << 16) | qu


def save_input(i, q):
    rows=[]
    with (OUT/'radar_input_iq.txt').open('w') as f_iq, (OUT/'radar_input_packed_hex.txt').open('w') as f_hex:
        count=0
        for chirp in range(ND):
            for sample in range(NR):
                iv, qv = int(i[chirp,sample]), int(q[chirp,sample])
                word=pack_iq(iv,qv)
                f_iq.write(f'{iv} {qv}\n')
                f_hex.write(f'{word:08X}\n')
                rows.append([count, chirp, sample, iv, qv, f'{word:08X}'])
                count += 1
    pd.DataFrame(rows, columns=['global_sample_number','chirp_index','sample_index','I_signed','Q_signed','packed_32bit_hex']).to_csv(OUT/'radar_input_metadata.csv', index=False)


def cfar(power):
    ref=CFAR_REFERENCE_CELLS_PER_SIDE
    guard=CFAR_GUARD_CELLS_PER_SIDE
    det=np.zeros_like(power,dtype=bool)
    threshold=np.zeros_like(power)
    noise=np.zeros_like(power)
    edge=ref+guard
    for d in range(ND):
        for r in range(edge, NR-edge):
            left=power[d, r-guard-ref:r-guard]
            right=power[d, r+guard+1:r+guard+ref+1]
            estimate=np.mean(np.concatenate((left,right)))
            threshold[d,r]=CFAR_ALPHA*estimate
            noise[d,r]=estimate
            det[d,r]=power[d,r] > threshold[d,r]
    return det, threshold, noise


def local_peaks(power, detections):
    peaks=np.zeros_like(detections)
    for d in range(1, ND-1):
        for r in range(1, NR-1):
            if detections[d,r] and power[d,r] == np.max(power[d-1:d+2,r-1:r+2]):
                peaks[d,r]=True
    return peaks


def plot_map(power, title, filename, detections=None, target_d=None):
    db=10*np.log10(power+1e-12)
    plt.figure(figsize=(12,6))
    plt.imshow(db, origin='lower', aspect='auto', cmap='jet')
    plt.colorbar(label='Power (dB)')
    plt.xlabel('Range bin')
    plt.ylabel('Shifted Doppler bin')
    plt.title(title)
    plt.plot(TARGET_RANGE_BIN, target_d, 'wx', ms=12, mew=2, label='Expected target')
    if detections is not None:
        d,r=np.where(detections)
        plt.scatter(r,d,facecolors='none',edgecolors='lime',s=80,label='Detection')
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT/filename, dpi=200)
    plt.close()


def main():
    target=tone(TARGET_RANGE_BIN,TARGET_DOPPLER_BIN,TARGET_AMPLITUDE)
    stationary=tone(STATIONARY_CLUTTER_RANGE_BIN,0,STATIONARY_CLUTTER_AMPLITUDE)
    near_zero=tone(NEAR_ZERO_CLUTTER_RANGE_BIN,NEAR_ZERO_CLUTTER_DOPPLER,NEAR_ZERO_CLUTTER_AMPLITUDE)
    random_clutter=np.zeros((ND,NR),dtype=complex)
    for _ in range(NUM_RANDOM_CLUTTER):
        random_clutter += tone(rng.integers(5,NR-5),rng.uniform(-1,1),rng.uniform(RANDOM_CLUTTER_AMPLITUDE_MIN,RANDOM_CLUTTER_AMPLITUDE_MAX))
    clean=target+stationary+near_zero+random_clutter
    received, noise_power=add_awgn(clean)
    i,q,scale=quantize_iq(received)
    save_input(i,q)
    quantized=i.astype(float)/scale + 1j*q.astype(float)/scale
    rw=hann(NR,sym=False)
    dw=hann(ND,sym=False)
    range_fft=np.fft.fft(quantized*rw[None,:],axis=1)
    doppler=np.fft.fftshift(np.fft.fft(range_fft*dw[:,None],axis=0),axes=0)
    power=doppler.real**2+doppler.imag**2
    target_d=(TARGET_DOPPLER_BIN+ND//2)%ND
    fixed_noise=np.median(power)
    fixed_threshold=FIXED_THRESHOLD_FACTOR*fixed_noise
    fixed=local_peaks(power,power>fixed_threshold)
    cfar_raw, threshold, noise=cfar(power)
    cfar_peaks=local_peaks(power,cfar_raw)
    complex_rows=[]; power_rows=[]; detections=[]
    for d in range(ND):
        for r in range(NR):
            z=doppler[d,r]
            complex_rows.append([d,r,z.real,z.imag])
            power_rows.append([d,r,power[d,r],noise[d,r],threshold[d,r],int(fixed[d,r]),int(cfar_peaks[d,r])])
            if cfar_peaks[d,r]:
                detections.append([r,d,d-ND//2,power[d,r],noise[d,r],threshold[d,r]])
    pd.DataFrame(complex_rows,columns=['doppler_index_shifted','range_index','real','imag']).to_csv(OUT/'reference_range_doppler_complex.csv',index=False)
    pd.DataFrame(power_rows,columns=['doppler_index_shifted','range_index','power','cfar_noise','cfar_threshold','fixed_detection','cfar_detection']).to_csv(OUT/'reference_range_doppler_power.csv',index=False)
    pd.DataFrame(detections,columns=['range_bin','doppler_bin_shifted','doppler_bin_unshifted','power','noise_estimate','threshold']).to_csv(OUT/'reference_detections.csv',index=False)
    plot_map(power,'Reference Range-Doppler Map','range_doppler_map.png',target_d=target_d)
    plot_map(power,'Fixed-Threshold Detection','fixed_threshold_detection.png',fixed,target_d)
    plot_map(power,'CA-CFAR Adaptive Detection','cfar_detection.png',cfar_peaks,target_d)
    target_power=power[target_d,TARGET_RANGE_BIN]
    print('Objective-I simulation completed.')
    print(f'Configuration: NR={NR}, ND={ND}, total samples={NR*ND}')
    print(f'Expected target: range bin={TARGET_RANGE_BIN}, shifted Doppler bin={target_d}')
    print(f'Target power={target_power:.6g}; fixed target detected={bool(fixed[target_d,TARGET_RANGE_BIN])}; CFAR target detected={bool(cfar_peaks[target_d,TARGET_RANGE_BIN])}')
    print(f'Outputs written to: {OUT.resolve()}')

if __name__ == '__main__':
    main()
