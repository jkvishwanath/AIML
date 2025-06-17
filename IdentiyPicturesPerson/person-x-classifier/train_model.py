import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import os
import glob
# Configuration
DATASET_DIR = os.path.join(os.getcwd(), 'dataset')
#DATASET_DIR = 'E:/person-x-classifier/dataset'       # Dataset folder path
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 10
MODEL_FILENAME = "person_x_classifier.h5"


images = glob.glob(os.path.join(DATASET_DIR, '*/*.jpg'))
print(f"Found {len(images)} images.")

# Data loading with augmentation
datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.0,
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.1
)

train_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=IMG_SIZE + (3,)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary output
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Training
#model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)
if train_generator.samples == 0:
    raise ValueError("🚫 No training images found.")
if val_generator.samples == 0:
    print("⚠️ Warning: No validation images found. Skipping validation.")
    model.fit(train_generator, epochs=EPOCHS)
else:
    model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS)

# Save model
model.save(MODEL_FILENAME)
print(f"✅ Model trained and saved as: {MODEL_FILENAME}")



