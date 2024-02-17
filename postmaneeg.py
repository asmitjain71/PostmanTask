import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, windows, detrend

eeg_data = np.loadtxt("eeg-data.txt")

fs = 100 # Hz
duration = 30 # seconds

delta = (1, 4) # Hz
theta = (4, 8) # Hz
alpha = (8, 13) # Hz
beta = (13, 30) # Hz

def bandpower(psd, freqs, band):
    idx_band = np.logical_and(freqs >= band[0], freqs <= band[1])
    return np.sum(psd[idx_band])

def relative_bandpower(psd, freqs, band):
    total_power = np.sum(psd)
    band_power = bandpower(psd, freqs, band)
    return band_power / total_power

freqs, psd = welch(eeg_data, fs, nperseg=fs*2, window="hann", detrend="constant")

delta_power = bandpower(psd, freqs, delta)
theta_power = bandpower(psd, freqs, theta)
alpha_power = bandpower(psd, freqs, alpha)
beta_power = bandpower(psd, freqs, beta)

delta_rel_power = relative_bandpower(psd, freqs, delta)
theta_rel_power = relative_bandpower(psd, freqs, theta)
alpha_rel_power = relative_bandpower(psd, freqs, alpha)
beta_rel_power = relative_bandpower(psd, freqs, beta)

print("Absolute bandpowers:")
print(f"Delta: {delta_power:.2f} uV^2")
print(f"Theta: {theta_power:.2f} uV^2")
print(f"Alpha: {alpha_power:.2f} uV^2")
print(f"Beta: {beta_power:.2f} uV^2")
print()
print("Relative bandpowers:")
print(f"Delta: {delta_rel_power:.2f}")
print(f"Theta: {theta_rel_power:.2f}")
print(f"Alpha: {alpha_rel_power:.2f}")
print(f"Beta: {beta_rel_power:.2f}")
print()

bands = ["Delta", "Theta", "Alpha", "Beta"]
rel_powers = [delta_rel_power, theta_rel_power, alpha_rel_power, beta_rel_power]
max_band = bands[np.argmax(rel_powers)]
max_rel_power = np.max(rel_powers)

print(f"The frequency band with the highest relative bandpower is {max_band} with {max_rel_power:.2f} of the total power.")

plt.figure(figsize=(10, 6))
plt.semilogy(freqs, psd)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power spectral density (uV^2/Hz)")
plt.title("EEG data analysis")
plt.xlim(0, 30)
plt.ylim(1e-3, 1e3)
plt.axvspan(delta[0], delta[1], color="blue", alpha=0.2, label="Delta")
plt.axvspan(theta[0], theta[1], color="green", alpha=0.2, label="Theta")
plt.axvspan(alpha[0], alpha[1], color="red", alpha=0.2, label="Alpha")
plt.axvspan(beta[0], beta[1], color="yellow", alpha=0.2, label="Beta")
plt.legend()
plt.show()

from mne.time_frequency import psd_array_multitaper

psd_mt, freqs_mt = psd_array_multitaper(eeg_data, fs, adaptive=True, normalization="full", verbose=False)

delta_rel_power_mt = relative_bandpower(psd_mt, freqs_mt, delta)
theta_rel_power_mt = relative_bandpower(psd_mt, freqs_mt, theta)
alpha_rel_power_mt = relative_bandpower(psd_mt, freqs_mt, alpha)
beta_rel_power_mt = relative_bandpower(psd_mt, freqs_mt, beta)

print("Relative bandpowers using multitaper method:")
print(f"Delta: {delta_rel_power_mt:.2f}")
print(f"Theta: {theta_rel_power_mt:.2f}")
print(f"Alpha: {alpha_rel_power_mt:.2f}")
print(f"Beta: {beta_rel_power_mt:.2f}")
print()

rel_powers_mt = [delta_rel_power_mt, theta_rel_power_mt, alpha_rel_power_mt, beta_rel_power_mt]
max_band_mt = bands[np.argmax(rel_powers_mt)]
max_rel_power_mt = np.max(rel_powers_mt)

print(f"The frequency band with the highest relative bandpower using multitaper method is {max_band_mt} with {max_rel_power_mt:.2f} of the total power.")

plt.figure(figsize=(10, 6))
plt.semilogy(freqs_mt, psd_mt)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power spectral density (uV^2/Hz)")
plt.title("EEG data analysis using multitaper method")
plt.xlim(0, 30)
plt.ylim(1e-3, 1e3)
plt.axvspan(delta[0], delta[1], color="blue", alpha=0.2, label="Delta")
plt.axvspan(theta[0], theta[1], color="green", alpha=0.2, label="Theta")
plt.axvspan(alpha[0], alpha[1], color="red", alpha=0.2, label="Alpha")
plt.axvspan(beta[0], beta[1], color="yellow", alpha=0.2, label="Beta")
plt.legend()
plt.show()
