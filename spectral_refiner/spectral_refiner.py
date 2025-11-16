import librosa
import soundfile as sf
import numpy as np
from scipy.fft import fft, ifft, fftshift
import matplotlib.pyplot as plt


def plot_fft(fft) -> None:
    # plt.plot(xs, fft)
    max = int(len(fft)/2)
    xs = range(1,max)
    plt.semilogy(xs, fft[1:max])
    plt.grid()
    plt.show()

def get_conversion_factor(fft, sr) -> int:
    max = int(len(fft)/2)
    return 1/max * sr/2

# Konwersja indeksu próbki (piku na hz)
def ix_to_hz(sample_ix, factor) -> int:
    return sample_ix * factor

if __name__=="__main__":
    dir_path = "/home/maja/Studia/Akustyka/AkustykaMuzyczna/Lab_3/AMlab3/"
    wav_path = dir_path + 'skrzypce.wav'

    # Load wav file
    y, sr = librosa.load(wav_path, sr=None)

    # Calculate FFT
    fft = fft(y)

    plot_fft(fft)

    factor = get_conversion_factor(fft, sr)
    ix_converted = ix_to_hz(260)

    

    print("stop")