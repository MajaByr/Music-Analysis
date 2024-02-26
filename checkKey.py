#$1 - .WAV file
#Example program call: python3 ./checkKey.py nameOfFile.wav
#Currently program requires manually setting Bar value (line 45). To choose proper value run this program, then look for Y walue at AS plot, which can separate deltas of notes frequencies

import scipy.fftpack
from scipy.signal import medfilt
import sys
import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd
import soundfile as sf

#Reading input file
data, Fs = sf.read(sys.argv[1], dtype='float32')
N=len(data) #number of samples
t = np.arange(N)/Fs     # time array

#Getting amplitude spectrum
f=np.linspace(-Fs/2, Fs/2, len(t))
XT=np.fft.fftshift(np.fft.fft(data[:,0]))
WA=abs(XT)

#Plot of data(t) and WA(f)
plt.subplot(2,1,1) #data(t) plot
plt.plot(t, data[:,0])
plt.subplot(2,1,2) #AS(f) plot
plt.plot(f, WA)

#Function to get frequencies with highest AS value
def importantFrequencies(spectrum, bar, fs):
    aboveBar = (spectrum >= bar).astype(np.int64)
    derivative = np.diff(aboveBar) 
    starts = np.where(derivative == 1)[0] + 1 
    ends = np.where(derivative == -1)[0] + 1 
    maxima = [] 
 for start, end in zip(starts, ends): 
        p = np.argmax(spectrum[start:end]) + start 
        tmp = p/len(t)*fs-fs/2+2.5
        if tmp>0:
            if len(maxima)==0:  maxima.append(tmp) 
            elif ( len(maxima)!=0 and tmp>maxima[len(maxima)-1]+8 ): maxima.append(tmp)
    return maxima

#Get frequencies
Bar=1250 #CHANGE this value depending on input file (louder - greater value)
Maximas = importantFrequencies(WA, Bar, Fs)

#get names of notes
notesNames={0} 
notesNames=[["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"],["110.00", "116.54", "123.47", "130.81", "138.59", "146.83", "155.56", "164.81", "174.61", "185.00", "196.00", "207.65", "220.00", "233.08", "246.94", "261.63", "277.18", "293.66", "311.13", "329.63", "349.23", "369.99", "392.00", "415.30", "440.00", "466.16", "493.88", "523.25", "554.37", "587.33", "622.25", "659.25", "698.46", "739.99", "783.99", "830.61", "880.00", "932.33", "987.77", "1046.50", "1108.73", "1174.66", "1244.51", "1318.51", "1396.91", "1479.98", "1567.98", "1661.22"]] 

def findName(freq):
    min=1000
    minIx=0
    for i in range(len(notesNames[1])):
        temp=float(notesNames[1][i])
        if abs(freq-temp)<min: min=abs(freq-temp); minIx=i;
    return notesNames[0][minIx]

#display notes from data
scaleNotes = []
for m in maksima:
    print('f0 = ',m, " - ", findName(m))
    scaleNotes.append(findName(m))


#get notes in scale
print("\n", sorted(set(scaleNotes)))

#check scale
def app(note): #appears
 return (note in scaleNotes);

def checkScale():
    if not ( app('F#') or app('C#') or app('G#') or app('D#') or app('A#') ): print("C Major / A Minor"); return;
    if app('F#'):
        if not ( app('C#') or app('G#') or app('D#') or app('A#') ): print("G Major / E Minor"); return;
        if not ( app('G') or app('D') or app('B') or app('E') or app('A') ): print("Db Major / Bb Minor. You are screwed."); return;
    if app('A#'):
        if not ( app('F#') or app('C#') or app('G#') or app('D#') ): print("F Major / D Minor"); return;
        if not ( app('A') or app('G') or app('F') or app('D') or app('C') ): print("B Major / Ab Minor"); return;
    if app('C#'):
        if not ( app('C') or app('G#') or app('A#') or app('D#') ): print("D Major / B Minor"); return;
        if not ( app("D") or app("F#") or app("A") or app("E") or app("B") ): print("Ab Major / F Minor"); return;
    if app("D#"):
        if not ( app("E") or app("F#") or app("C#") or app("G#") ): print("Bb Major / G Minor"); return;
        if not ( app("D") or app("A#") or app("C") or app("F") or app("G") ): print("E Major / Db Minor"); return;
    if app("G#"):
        if not ( app("G") or app("D#") or app("A#") or app("F") or app("C") ): print("A Major / Gb Minor. You are screwed."); return;
        if not ( app("A") or app("F#") or app("C#") or app("B") or app("E") ): print("Eb Major / C Minor"); return;
    else: print("Unconventional scale"); #This message may also appear, when invalid Bar value is choosen

checkScale();

plt.show()
exit()
