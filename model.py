# model.py

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dropout, BatchNormalization, GlobalAveragePooling1D, Dense

def build_model(input_shape, num_classes):
    model = Sequential()

    # Block 1
    model.add(Conv1D(64, 3, activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Conv1D(64, 3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.25))

    # Block 2
    model.add(Conv1D(128, 3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv1D(128, 3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.25))

    # Block 3
    model.add(Conv1D(256, 3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv1D(256, 3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))
    model.add(Dropout(0.5))

    model.add(GlobalAveragePooling1D())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model