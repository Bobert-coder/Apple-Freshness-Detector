import os

from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization
)
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint


# Dataset Path to get the images loaded in



DATASET_PATH = "Fruit Freshness Dataset/Apple"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

classes = os.listdir(DATASET_PATH)
num_classes = len(classes)


# Data Generators  to modify the images

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

val_data = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)



# Model

model = Sequential([
    Conv2D(16, (3,3), activation="relu", input_shape=(128,128,3)),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(32, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(num_classes, activation="softmax")
])


# Compile tells model how to learn

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)






# Training

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=30,

)


# Evaluation to get accuracy

loss, accuracy = model.evaluate(val_data)
print(f"Validation Accuracy: {accuracy:.4f}")


# Save Final Model

model.save("fruits_model_final.h5")
