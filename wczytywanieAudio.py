#$1 - .WAV file

from scipy.interpolate import interp1d
import scipy.fftpack
import sys
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf

def checkScale(wa):
 if wa[int(step*740)] > wa[int(step*698)]: 
    if wa[int(step*554)] < wa[int(step*523)]: return 1;
    if wa[int(step*831)] < wa[int(step*784)]: return 2;
    if wa[int(step*622)] < wa[int(step*587)]: return 3;
    if wa[int(step*932)] < wa[int(step*880)]: return 4;
    else: return 5; 
 elif wa[int(step*932)] > wa[int(step*988)]: 
    if wa[int(step*622)] < wa[int(step*659)]: return -1;
    if wa[int(step*831)] < wa[int(step*880)]: return -2;
    if wa[int(step*1109)] < wa[int(step*587)]: return -3;
    if wa[int(step*740)] < wa[int(step*784)]: return -4;
    else: return -5; 
 else: return 0;

def printScale(code):
    match code:
        case 0: print("C Major/A Minor"); return;
        case 1: print("G Major/E Minor"); return;
        case 2: print("D Major/B Minor"); return;
        case 3: print("A Major/Gb Minor"); return;
        case 4: print("E Major/Db Minor"); return;
        case 5: print("B Major/Ab Minor\nYou are screwed."); return;
        case -1: print("F Major/D Minor"); return;
        case -2: print("Bb Major/G Minor"); return;
        case -3: print("Eb Major/C Minor"); return;
        case -4: print("Ab Major/F Minor"); return;
        case -5: print("Db Major/Bb Minor"); return;


data, Fs = sf.read(sys.argv[1], dtype='float32')

#print(data.dtype)
#print(data.shape)

#sd.play(data,fs)
#status=sd.wait()

N=len(data) #number of samples
t = np.arange(N)/Fs     # time array

#plt.subplot(2,1,1) #pierwszy kanał
#plt.plot(t, data[:,0]) 
#plt.subplot(2,1,2) #drugi kanał
#plt.plot(t, data[:,1]) 

f=np.linspace(-Fs/2, Fs/2, len(t))
XT=np.fft.fftshift(np.fft.fft(data[:,0]))
WA=abs(XT)

#plt.figure()
plt.subplot(2,1,1)
plt.plot(t, data[:,0])
data_interp = interp1d(t, data[:,0])
plt.plot(t, data_interp)


plt.subplot(2,1,2)
plt.plot(f, WA)

step=len(t)/Fs
#WA[int(831*Fs/len(t))])

printScale(checkScale(WA))

#plt.subplot(2,1,1)
#plt.plot(data[:,0])
sd.play(data, Fs)
plt.show()
status=sd.wait()
exit()
