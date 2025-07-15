import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('mobilenetv2_garbage_classifier.h5')

# print model summary
model.summary()

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

# Open the default camera
cap = cv2.VideoCapture(2)
if not cap.isOpened():
    print('Cannot open camera')
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print('Failed to grab frame')
        break

    # Preprocess the frame
    img = cv2.resize(frame, IMG_SIZE)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    preds = model.predict(img)
    class_idx = np.argmax(preds)
    prob = float(np.max(preds))
    label = f'{class_names[class_idx]}: {prob*100:.1f}%'

    # Display prediction on the frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Trash Classifier', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
