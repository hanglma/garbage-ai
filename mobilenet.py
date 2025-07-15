import os
import warnings
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="Your `PyDataset` class should call `super().__init__")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Paths
DATASET_DIR = 'garbage_classification'
IMG_SIZE = (128, 128)
BATCH_SIZE = 64
EPOCHS = 10

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# Model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model summary
model.summary()

# Training
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)


# Fine-tuning: unfreeze last 20 layers and continue training
for layer in base_model.layers[-20:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

fine_tune_epochs = 5
total_epochs = EPOCHS + fine_tune_epochs

history_fine = model.fit(
    train_generator,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1] + 1,
    validation_data=val_generator
)

# Save the model
model.save('mobilenetv2_garbage_classifier.h5')
print('Model saved as mobilenetv2_garbage_classifier.h5')

# Plotting
acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
plt.figure(figsize=(8, 5))
plt.plot(acc, label='accuracy')
plt.plot(val_acc, label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy (with Fine-tuning)')
# Mark the start of fine-tuning
plt.axvline(x=EPOCHS, color='red', linestyle='--', label='Fine-tuning start')
# Save plot as JPG
plt.savefig('training_accuracy_plot.jpg', format='jpg', dpi=200)
plt.show()
print('Plot saved as training_accuracy_plot.jpg')
