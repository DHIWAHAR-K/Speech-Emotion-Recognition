# feature_extraction.py

import librosa
import numpy as np
from data_augmentation import noise, stretch, pitch

def extract_features(data, sr):
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    stft = np.abs(librosa.stft(data))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sr).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sr).T, axis=0)
    return np.hstack((zcr, chroma, mfcc, rms, mel))

def get_features(path):
    data, sr = librosa.load(path, duration=2.5, offset=0.6)
    res1 = extract_features(data, sr)
    res2 = extract_features(noise(data), sr)
    res3 = extract_features(pitch(stretch(data, sr), sr), sr)
    return np.vstack([res1, res2, res3])