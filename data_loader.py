# data_loader.py

import os
import pandas as pd
from config import RAVDESS_PATH, CREMA_PATH, TESS_PATH, SAVEE_PATH, EMOTION_LABELS

def load_ravdess():
    emotions, paths = [], []
    for actor in os.listdir(RAVDESS_PATH):
        actor_path = os.path.join(RAVDESS_PATH, actor)
        if not os.path.isdir(actor_path): continue
        for file in os.listdir(actor_path):
            parts = file.split('-')
            if len(parts) > 2:
                emotion_code = int(parts[2])
                emotion = EMOTION_LABELS.get(emotion_code)
                if emotion:
                    emotions.append(emotion)
                    paths.append(os.path.join(actor_path, file))
    return pd.DataFrame({'Emotions': emotions, 'Path': paths})

def load_crema():
    emotions, paths = [], []
    for file in os.listdir(CREMA_PATH):
        parts = file.split('_')
        code = parts[2]
        emotion_map = {'SAD': 'sad', 'ANG': 'angry', 'DIS': 'disgust',
                       'FEA': 'fear', 'HAP': 'happy', 'NEU': 'neutral'}
        emotion = emotion_map.get(code, 'Unknown')
        if emotion != 'Unknown':
            emotions.append(emotion)
            paths.append(os.path.join(CREMA_PATH, file))
    return pd.DataFrame({'Emotions': emotions, 'Path': paths})

def load_tess():
    emotions, paths = [], []
    for folder in os.listdir(TESS_PATH):
        for file in os.listdir(os.path.join(TESS_PATH, folder)):
            emotion = file.split('_')[-1].split('.')[0]
            emotion = 'surprise' if emotion == 'ps' else emotion
            emotions.append(emotion)
            paths.append(os.path.join(TESS_PATH, folder, file))
    return pd.DataFrame({'Emotions': emotions, 'Path': paths})

def load_savee():
    emotions, paths = [], []
    emotion_map = {'a': 'angry', 'd': 'disgust', 'f': 'fear', 'h': 'happy',
                   'n': 'neutral', 'sa': 'sad', 'su': 'surprise'}
    for file in os.listdir(SAVEE_PATH):
        code = file.split('_')[1][:-6]
        emotion = emotion_map.get(code, 'surprise')
        emotions.append(emotion)
        paths.append(os.path.join(SAVEE_PATH, file))
    return pd.DataFrame({'Emotions': emotions, 'Path': paths})

def get_combined_dataset():
    df = pd.concat([load_ravdess(), load_crema(), load_tess(), load_savee()], axis=0)
    df.reset_index(drop=True, inplace=True)
    return df