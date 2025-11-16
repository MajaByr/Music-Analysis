import librosa
import soundfile as sf
import numpy as np
from scipy.fft import fft, ifft, fftshift
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from functools import partial
from scipy.signal import medfilt
import logging

logging.basicConfig(level=logging.INFO)

class SpectralRefiner:
    def plot_fft(fft: np.array, sampling_rate) -> None:
        """
        Plot amplitude spectrum of given fft.
        """
        max_ix = int(len(fft)/2)

        ixs = np.linspace(0, sampling_rate/2, max_ix, endpoint=False)

        y = fft[1:max_ix]

        plt.semilogy(ixs, y)
        
        plt.grid()
        plt.show()

    def get_dominant_frequencies(fft: np.ndarray, min_prominence: float = 30):
        
        as_db = 20 * np.log10(abs(fft) + 1e-12)
        
        N = len(fft)
        as_part = as_db[:N//2]
        
        dominant_frequencies, _ = find_peaks(as_part, prominence=min_prominence)
        dominant_frequencies = [int(df) for df in dominant_frequencies]
        return dominant_frequencies
    
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
        
    def get_conversion_factor(fft, sr) -> int:
        max = int(len(fft)/2)
        return 1/max * sr/2

    def ix_to_hz(sample_ix, conversion_factor) -> int:
        """
        Convert an FFT bin index to its corresponding frequency in Hertz.
        """
        return sample_ix * conversion_factor
    

if __name__=="__main__":
    dir_path = "spectral_refiner/audio_samples/"
    wav_path = dir_path + 'violin_raw.wav'

    # Load wav file
    y, sr = librosa.load(wav_path, sr=None)

    # Calculate FFT
    fft = fft(y)

    conversion_factor = SpectralRefiner.get_conversion_factor(fft, sr)
    
    # Get dominant frequencies
    dominant_frequencies = SpectralRefiner.get_dominant_frequencies(fft, 35)

    # Filter dominant frequencies
    dominant_frequencies = [df for df in dominant_frequencies if df>100]

    SpectralRefiner.plot_fft_with_dominant_peaks(fft, sr, dominant_frequencies)
    
    print(dominant_frequencies)