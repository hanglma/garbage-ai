import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the trained model
model = tf.keras.models.load_model('mobilenetv2_garbage_classifier.h5')

# Actual class names for your categories
class_names = [
    'battery',        # 0
    'biological',     # 1
    'brown-glass',    # 2
    'cardboard',      # 3
    'green-glass',    # 4
    'metal',          # 5
    'paper',          # 6
    'plastic',        # 7
    'trash',          # 8
    'white-glass'     # 9
]

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# Validation data generator (no augmentation, just rescale)
val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
val_generator = val_datagen.flow_from_directory(
    'garbage_classification',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Get predictions and true labels
Y_pred = model.predict(val_generator)
y_pred = np.argmax(Y_pred, axis=1)
y_true = val_generator.classes

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
accuracy = np.trace(cm) / np.sum(cm)
print(f'Validation accuracy: {accuracy*100:.2f}%')
cmd = ConfusionMatrixDisplay(cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10, 10))
cmd.plot(ax=ax, cmap=plt.cm.Blues, xticks_rotation=45)
plt.title(f'Confusion Matrix - Validation Set\nAccuracy: {accuracy*100:.2f}%')
plt.tight_layout()
plt.savefig('confusion_matrix_validation.jpg', format='jpg', dpi=200)
plt.show()
print('Confusion matrix saved as confusion_matrix_validation.jpg')
