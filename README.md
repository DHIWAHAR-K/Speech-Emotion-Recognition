# Speech Emotion Recognition

This project performs **Speech Emotion Recognition (SER)** using a **1D Convolutional Neural Network (Conv1D)** on features extracted from multiple audio datasets. The model is trained to classify emotions from audio recordings using augmented MFCC-based features. This project supports **RAVDESS**, **TESS**, **SAVEE**, and **CREMA-D** datasets.

---

## Project Structure

- `config.py` – Stores paths and emotion mappings.
- `data_loader.py` – Loads emotion-labeled audio file paths from RAVDESS, CREMA-D, TESS, and SAVEE.
- `data_augmentation.py` – Applies noise, pitch, shift, and stretch to augment audio signals.
- `feature_extraction.py` – Extracts MFCC, Mel, RMS, ZCR, and Chroma features from audio signals.
- `model.py` – Defines a 1D CNN for multi-class classification.
- `utils.py` – Utilities to visualize waveplots and spectrograms.
- `evaluate.py` – Functions to visualize training history, confusion matrix, and classification report.
- `train.py` – The main pipeline to load data, extract features, train the model, evaluate performance, and save the trained model.

---

## Installation

Ensure you have Python 3.7+ and install the following libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn keras tensorflow librosa
```

## Datasets:

Download and place the following datasets in their respective paths:

| Dataset  | Path                                                  |
|----------|-------------------------------------------------------|
| RAVDESS  | `./data_2/`                                           |
| CREMA-D  | `./data_4/`                                           |
| TESS     | `./data_3/`                                           |
| SAVEE    | `./data_1/`                                           |

## How to Run:

Run the training pipeline:

```bash
python train.py
```

This will:

	•	Load and combine all datasets.

	•	Apply data augmentation (noise, pitch, stretch).

	•	Extract features.

	•	Train a 1D CNN.

	•	Evaluate on test data.

	•	Display confusion matrix and classification report.

	•	Save the trained model to emotion_recognition_model.h5.


## Model Architecture:

	•	Three convolutional blocks with:

	•	Conv1D → BatchNorm → MaxPooling1D → Dropout

	•	GlobalAveragePooling1D

	•	Dense layer (128 units) + Dropout

	•	Softmax output layer for multi-class classification

    •   Loss: categorical_crossentropy

    •   Optimizer: Adam

    •   Metrics: accuracy

## Evaluation:

After training, the model will:

	•	Print training and validation accuracy/loss curves.

	•	Show a confusion matrix for all emotions.

	•	Output a classification report with precision, recall, and F1-score.

## Emotions Covered:

The model can classify the following 8 emotions:

	1. Neutral

	2. Calm

	3. Happy

	4. Sad

	5. Angry

	6. Fear

	7. Disgust

	8. Surprise


## License:

This project is intended for educational and research purposes. Feel free to use, modify, and build upon this work with credit.