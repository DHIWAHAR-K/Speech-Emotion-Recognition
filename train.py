# train.py

import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from data_loader import get_combined_dataset
from feature_extraction import get_features
from model import build_model
from evaluate import plot_training_history, show_confusion_matrix, print_classification

# Step 1: Load data
print("[INFO] Loading dataset...")
data_df = get_combined_dataset()

# Step 2: Extract features
print("[INFO] Extracting features...")
X, Y = [], []
for path, emotion in zip(data_df.Path, data_df.Emotions):
    features = get_features(path)
    for f in features:
        X.append(f)
        Y.append(emotion)

X = np.array(X)
Y = np.array(Y)

# Step 3: One-Hot Encode labels
encoder = OneHotEncoder()
Y_encoded = encoder.fit_transform(Y.reshape(-1, 1)).toarray()

# Step 4: Split data
x_train, x_test, y_train, y_test = train_test_split(X, Y_encoded, test_size=0.2, random_state=42, shuffle=True)

# Step 5: Standardize features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Step 6: Reshape for Conv1D
x_train = np.expand_dims(x_train, axis=2)
x_test = np.expand_dims(x_test, axis=2)

# Step 7: Build model
print("[INFO] Building model...")
model = build_model(input_shape=(x_train.shape[1], 1), num_classes=y_train.shape[1])
model.summary()

# Step 8: Train model
rlrp = ReduceLROnPlateau(monitor='loss', factor=0.4, verbose=1, patience=2, min_lr=1e-7)
history = model.fit(x_train, y_train, batch_size=64, epochs=50,
                    validation_data=(x_test, y_test), callbacks=[rlrp])

# Step 9: Evaluate model
print("[INFO] Evaluating model...")
score = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {score[1]*100:.2f}%")

# Step 10: Visualize training
plot_training_history(history)

# Step 11: Predict and evaluate
y_pred = encoder.inverse_transform(model.predict(x_test))
y_true = encoder.inverse_transform(y_test)

# Step 12: Show confusion matrix and classification report
labels = encoder.categories_[0]
show_confusion_matrix(y_true, y_pred, labels)
print_classification(y_true, y_pred)

# Step 13: Save model
model.save("emotion_recognition_model.h5")
print("[INFO] Model saved as 'emotion_recognition_model.h5'")