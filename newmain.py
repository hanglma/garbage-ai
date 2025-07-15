import tensorflow as tf
from keras import models
from keras.layers import *
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
from keras.applications import MobileNetV2
from keras import regularizers
from typing import Tuple
import matplotlib.pyplot as plt
import os
import sys

IMG_SIZE = 180
BATCH_SIZE = 32

DATA_DIR = "medium_archive/garbage_classification/"

VALIDATION_SPLIT = 0.2
EPOCHS = 10
LEARNING_RATE = 0.001

CLASSES = 8

AUTOTUNE = tf.data.AUTOTUNE

L2_REGULATION = 0.001

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

def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

def main():
    (train_ds, val_ds) = load_data_sets()

    base_model = MobileNetV2(weights="imagenet", include_top=False,input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model.trainable = False

    model = models.Sequential([
        Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        Rescaling(1./255),
        base_model,
        Flatten(),
        Dense(64, activation="relu"),
        Dense(CLASSES, activation="softmax")
    ])

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    plot_accuracy(history)

    model.save("garbage_slayer_v1.keras")

if __name__ == "__main__":
    main()
