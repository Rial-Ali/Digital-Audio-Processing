# import packages
import numpy as np
import librosa
import matplotlib.pyplot as plt
import IPython.display as ipd

# %% Load audio file
file_name = '../debussy.wav'
signal, sr = librosa.load(file_name, sr=None)

print(f"the audio file contain sr : {sr}, samples : {len(signal)}")


# %% Time Domin Features

# %%% Amplitude Envelope Feature(AEt)
FRAME_LENGTH = 1024
HOP_LENGTH = 512

# calculate the amplitude envelope


def amplitude_envelope(signal, frame_size, hop_length):
    return(np.array([max(signal[i:i+frame_size])
                     for i in range(0, signal.size, hop_length)]))


ae_signal = amplitude_envelope(signal, FRAME_LENGTH, HOP_LENGTH)

frames = range(0, ae_signal.size)
t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)
plt.figure(figsize=(15, 5))

librosa.display.waveshow(signal, alpha=0.5)
plt.plot(t, ae_signal, color='r')
plt.title('signal Time Domin Features--->Amplitude Envelope (AEt)')
# plt.ylim((-1, 1))
plt.show()

# %%% Root-Mean-Square Energy Feature(RMSt)

rms_signal = librosa.feature.rms(y=signal,
                                 frame_length=FRAME_LENGTH,
                                 hop_length=HOP_LENGTH)[0]
frames = range(0, rms_signal.size)
t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)

plt.figure(figsize=(15, 5))


librosa.display.waveshow(signal, alpha=0.5)
plt.plot(t, rms_signal, color='r')
plt.title('signal Time Domin Features--->Root-Mean-Square Energy (RMSt)')
# plt.ylim((-1, 1))
plt.show()

# %%% Zero Crossing Rate Feature(ZCRt)

zcr_signal = librosa.feature.zero_crossing_rate(y=signal,
                                                frame_length=FRAME_LENGTH,
                                                hop_length=HOP_LENGTH)[0]
frames = range(0, zcr_signal.size)
t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)

plt.figure(figsize=(15, 5))

plt.plot(t, zcr_signal)

plt.ylim((0, 1))
plt.title('signal Time Domin Features--->Zero Crossing Rate (ZCRt)')

plt.show()

# %% freguency Domin Features
# %%% Fast Fourier Transform (fft)


def plot_magnitude_spectrum(signal, title, sr, f_ratio=1):
    ft = np.fft.fft(signal)
    magnitude_spectrum = np.abs(ft)

    frequencies = np.linspace(0, sr, len(magnitude_spectrum))
    num_frequency_bins = int(len(frequencies) * f_ratio)
    plt.figure(figsize=(18, 5))
    plt.plot(frequencies[:num_frequency_bins],
             magnitude_spectrum[:num_frequency_bins])
    plt.xlabel('Frequency (HZ)')
    plt.title(title)

    plt.show()


plot_magnitude_spectrum(signal, 'signal', sr, f_ratio=0.5)

# %%% Short Time Fourier Transform (stft)
ton_spec = librosa.stft(
    y=signal, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
# %%% Band Energy Ratio


def calculate_split_frequency_bin(spectrogram, split_frequency, sample_rate):
    frequency_range = sample_rate / 2
    frequency_delta_per_bin = frequency_range / spectrogram.shape[0]
    split_frequency_bin = np.floor(split_frequency / frequency_delta_per_bin)
    return int(split_frequency_bin)


def calculate_band_energy_ratio(spectrogram, split_frequency, sample_rate):
    split_frequency_bin = calculate_split_frequency_bin(spectrogram,
                                                        split_frequency,
                                                        sample_rate)
    # calculate power spectrogram
    power_spec = np.abs(spectrogram) ** 2
    power_spec = power_spec.T

    band_energy_ratio = []

    # calculate BER for each frame
    for frequencies_in_frame in power_spec:
        sum_power_low_frequency = np.sum(
            frequencies_in_frame[:split_frequency_bin])
        sum_power_high_frequency = np.sum(
            frequencies_in_frame[split_frequency_bin:])
        ber_current_frame = sum_power_low_frequency / sum_power_high_frequency
        band_energy_ratio.append(ber_current_frame)

    return np.array(band_energy_ratio)


ber_signal = calculate_band_energy_ratio(ton_spec, 2000, sr)

#  visualise Band Energy Ratio curves
frames = range(len(ber_signal))
t = librosa.frames_to_time(frames)

plt.Figure(figsize=(25, 5))

plt.plot(t, ber_signal, color='b')
plt.title('freguency Domin Features --->Band Energy Ratio')
plt.show()


# %%% Spectral Centroid
sc_signal = librosa.feature.spectral_centroid(
    y=signal, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]

#  Visualising spectral centroid
frames = range(len(sc_signal))
t = librosa.frames_to_time(frames, hop_length=HOP_LENGTH)

plt.figure(figsize=(25, 10))

plt.plot(t, sc_signal)
plt.title('freguency Domin Features --->Spectral Centroid')

plt.show()

# %%% Bandwidth (Spectral spread)
ban_signal = librosa.feature.spectral_bandwidth(
    y=signal, sr=sr, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)[0]

# Visualising spectral bandwidth
plt.figure(figsize=(25, 5))

plt.plot(t, ban_signal, color='b')
plt.title('freguency Domin Features --->Bandwidth (Spectral spread)')
plt.show()

# %% Frequency-time Domin Features

# %%% Spectrogram
ton_spec = librosa.stft(
    y=signal, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
#  Calculating the magnitude of spectrogram
ton_spec_mag = np.abs(ton_spec) ** 2
# Log-Amplitude Spectrogram
ton_log_spec = librosa.power_to_db(ton_spec_mag)

# Visualize the spectrogram


def plot_spectrogram(Y, sr, hop_length, y_axis='linear'):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(Y,
                             sr=sr,
                             hop_length=hop_length,
                             x_axis='time',
                             y_axis=y_axis)
    plt.colorbar(format='%+2.f')
    plt.title('freguency-time Domin Features --->Spectrogram')


# Log-frequency spectrogram
plot_spectrogram(ton_log_spec, sr, hop_length=HOP_LENGTH, y_axis='log')

# %%% Mel Spectrogram
mel_spectrogram = librosa.feature.melspectrogram(y=signal,
                                                 sr=sr,
                                                 n_fft=FRAME_LENGTH,
                                                 hop_length=HOP_LENGTH,
                                                 n_mels=90)  # 10
log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)

# Visualize the MEL-spectrogram
plt.Figure(figsize=(25, 10))
librosa.display.specshow(log_mel_spectrogram,
                         sr=sr,
                         x_axis='time',
                         y_axis='mel')
plt.colorbar(format='%+2.f')
plt.title('freguency-time Domin Features --->Mel Spectrogram')

plt.show()

# %%% Extract MFCCs, delta and delta2.
mfccs = librosa.feature.mfcc(y=signal,
                             sr=sr,
                             n_mfcc=13)
# visualize MFCCs
plt.Figure(figsize=(25, 10))
librosa.display.specshow(mfccs,
                         sr=sr,
                         x_axis='time')
plt.colorbar(format='%2f')
plt.title('freguency-time Domin Features --->MFCCs')

plt.show()

# calculate delta and delta2 MFCCs
delta_mfccs = librosa.feature.delta(mfccs)
delta2_mfccs = librosa.feature.delta(mfccs, order=2)

# visualize delta_mfccs
plt.Figure(figsize=(25, 10))
librosa.display.specshow(delta_mfccs,
                         sr=sr,
                         x_axis='time')
plt.colorbar(format='%2f')
plt.title('freguency-time Domin Features --->delta_mfccs')

plt.show()

# visualize delta2_mfccs
plt.Figure(figsize=(25, 10))
librosa.display.specshow(delta2_mfccs,
                         sr=sr,
                         x_axis='time')
plt.colorbar(format='%2f')
plt.title('freguency-time Domin Features --->delta_mfccs2')

plt.show()
