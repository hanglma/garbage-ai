import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from typing import Tuple
import os
import sys

IMG_SIZE = 180
BATCH_SIZE = 32

DATA_DIR = "archive/garbage_classification/"

VALIDATION_SPLIT = 0.2
EPOCHS = 10
LEARNING_RATE = 0.001

CLASSES = 12

AUTOTUNE = tf.data.AUTOTUNE

def load_data_sets() -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    if not os.path.exists(DATA_DIR):
        print("Data directory doesnt exist")
        sys.exit(1)
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
    ).cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=123,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
    ).cache().prefetch(buffer_size=AUTOTUNE)
    return (train_ds, val_ds)

def main():
    (train_ds, val_ds) = load_data_sets()

    model = models.Sequential([
        Rescaling(1./255, input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        Conv2D(16, 3, activation="relu"),
        MaxPooling2D(),
        Conv2D(32, 3, activation="relu"),
        MaxPooling2D(),
        Conv2D(64, 3, activation="relu"),
        MaxPooling2D(),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(CLASSES, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    model.save("garbage-slayer-v1")

if __name__ == "__main__":
    main()
