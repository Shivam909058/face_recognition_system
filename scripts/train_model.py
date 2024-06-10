from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    model = build_model()
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = datagen.flow_from_directory(
        '../data/',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset='training'
    )
    validation_generator = datagen.flow_from_directory(
        '../data/',
        target_size=(128, 128),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )
    model.fit(train_generator, epochs=10, validation_data=validation_generator)
    model.save('../models/face_detection_model.h5')

if __name__ == "__main__":
    train_model()
