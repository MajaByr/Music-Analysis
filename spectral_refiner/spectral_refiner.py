import librosa
import soundfile as sf
import numpy as np
from scipy.fft import fft, ifft, fftshift
import matplotlib.pyplot as plt


class SpectralRefiner:
    def plot_fft(fft: np.array) -> None:
        """
        Plot amplitude spectrum of given fft.
        """
        max_ix = int(len(fft)/2)
        ixs = range(1,max_ix)
        plt.semilogy(ixs, fft[1:max_ix])
        plt.grid()
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

    SpectralRefiner.plot_fft(fft)

    conversion_factor = SpectralRefiner.get_conversion_factor(fft, sr)
    ix_converted = SpectralRefiner.ix_to_hz(260, conversion_factor)
    print(ix_converted)
