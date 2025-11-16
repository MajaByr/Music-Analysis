import librosa
import soundfile as sf
import numpy as np
# from scipy.fft import fft, ifft, fftshift
import scipy as sp
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from functools import partial
from scipy.signal import medfilt
from scipy.io.wavfile import write
import logging

logging.basicConfig(level=logging.INFO)

class SpectralRefiner:
    def plot_fft(fft: np.array, sampling_rate) -> None:
        """
        Plot amplitude spectrum of given fft.
        """
        max_ix = int(len(fft)/2)

        ixs = np.linspace(0, sampling_rate/2, max_ix, endpoint=False)

        y = fft[:max_ix]

        plt.semilogy(ixs, y)
        
        plt.grid()
        plt.show()

    def get_dominant_frequencies_ixs(fft: np.ndarray, min_prominence: float = 30):
        
        as_db = 20 * np.log10(abs(fft) + 1e-12)
        
        N = len(fft)
        as_part = as_db[:N//2]
        
        dominant_frequencies, _ = find_peaks(as_part, prominence=min_prominence)
        dominant_frequencies = [int(df) for df in dominant_frequencies]
        dom_freq_amplitudes = as_part[dominant_frequencies]

        return dominant_frequencies, dom_freq_amplitudes
    
    def plot_fft_with_dominant_peaks(fft, sampling_rate, dominant_frequencies):

        as_db = 20 * np.log10(abs(fft) + 1e-12)
        
        N = len(as_db)
        xs = np.linspace(0, sampling_rate/2, N//2, endpoint=False)
        as_half = as_db[:N//2]
                
        plt.figure(figsize=(10,4))
        plt.plot(xs, as_half, label="Amplitude Spectrum")
        plt.plot(xs[dominant_frequencies], as_half[dominant_frequencies], 'ro', label="Peaks")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude [dB]")
        plt.legend()
        plt.grid(True)
        plt.show()

    def ix_to_hz(sample_ix, fft, sampling_rate) -> int:
        """
        Convert an FFT bin index to its corresponding frequency in Hertz.
        """
        return sample_ix / len(fft) * sampling_rate
    
    def hz_to_ix(f_hz, fft, sampling_rate):
        return round(f_hz / sampling_rate * len(fft))
    
    def filter_fft(fft, sampling_rate, dominant_frequencies_ixs, dom_freq_amplitudes, delta = 15):
        
        max_ix = len(fft)//2

        fft_filtered = fft[:max_ix].copy()

        hz_per_bin = sampling_rate / max_ix
        ix_radius = int(np.ceil(delta / hz_per_bin))

        for idx in dominant_frequencies_ixs:
            start = max(idx - ix_radius, 0)
            end = min(idx + ix_radius + 1, max_ix)
            fft_filtered[start:end] = 0
            fft_filtered[idx] = fft[idx]

        return np.concatenate((fft_filtered, fft_filtered[::-1]))

    def fft_to_wav(fft: np.array, wav_path: str, sampling_rate: int):
        """
        Save waveform to 16-bit PCM
        """

        time_signal = sp.fft.ifft(fft)
        time_signal = np.real(time_signal)
        
        # Normalize to <-1, 1>
        time_signal = time_signal / np.max(np.abs(time_signal))  
        time_signal_int16 = np.int16(time_signal * 32767)

        write(wav_path, sampling_rate, time_signal_int16)
        print(f"Saved WAV file: {wav_path}")
    
    def analyze_file(wav_path, output_path, min_prominence: float = 30):
        
        # Load wav file
        y, sr = librosa.load(wav_path, sr=None)

        # Calculate FFT
        fft = sp.fft.fft(y)
        
        # Get dominant frequencies
        dominant_frequencies_ixs, dom_freq_amplitudes = SpectralRefiner.get_dominant_frequencies_ixs(fft, 35)

        # Filter out all harmonics lying within ±15 Hz of the detected spectral peaks
        filtered_fft = SpectralRefiner.filter_fft(fft, sr, dominant_frequencies_ixs, dom_freq_amplitudes, 15)

        # Save filtered file to .wav
        SpectralRefiner.fft_to_wav(filtered_fft, output_path, sr)
    

if __name__=="__main__":
    dir_path = "spectral_refiner/audio_samples/"
    
    # # Analyze raw violin
    # raw_wav_path = dir_path + 'violin_raw.wav'
    # SpectralRefiner.analyze_file(raw_wav_path, 35)

    # Analyze violin triad
    triad_wav_path = dir_path + 'violin_triad.wav'
    SpectralRefiner.analyze_file(triad_wav_path, 'spectral_refiner/output/violin_triad_filtered.wav', 24)
