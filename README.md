# :musical_keyboard: Music-Analysis :musical_keyboard: 

Scripts to modify and get characteristics of audio files such as key, notes & tempo.

## Set up

Run:
```
./setup.sh
```

## Available scripts

### Key Detection

Automatic key detection of given audio segment.

Run:
```
./key_detection/key_detection.py fileName.wav
```

### Spectral Refiner

Refine spectrum of given audio segment, choosing one of available methods:
* Remove all harmonics lying within ±15 Hz of the detected spectral peaks;
* Remove all harmonics whose frequency offset from each spectral peak is greater than 11 Hz but smaller than 1/4 ERB;
* Retain only the harmonics that are local maxima within a 30 Hz frequency window.

```
python3 spectral_refiner/spectral_refiner.py
```

---
Project sponsored by cats :cat2:
