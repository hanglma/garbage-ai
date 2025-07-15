# Danach laden wir das Model hier wenden es auf die Live Kamera an
import tensorflow as tf
import numpy as np
import cv2

class_names = ["battery", "biological", "brown-glass", "cardboard", "clothes", "green-glass", "metal", "paper", "plastic", "shoes", "trash", "white-glass"]

# Bilddaten
img_width = 180
img_height = 180

# Hier laden wir vorgefertige Filter
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Abrufen der Modeldetails für input und output
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Objekterkennung - Schritt 1: Datenaufbereitung wie im Training
    # Bringt den Frame auf die Größe 32x32
    input_frame = cv2.resize(frame, (img_width, img_height))
    # Bringt die Pixel auf den Wert von 0-1
    input_frame = input_frame.astype(np.float32) / 255.0

    # Gleiche Daten als im Training
    input_tensor = np.expand_dims(input_frame, axis=0)
    input_tensor = input_tensor.astype(input_details[0]['dtype'])

    # Wir definieren die Informationen des Models, basierend auf unseren Trainingsdaten
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    # Und erzeugen die Objekterkennungsvorhersagen
    output_data = interpreter.get_tensor(output_details[0]['index'])[0]

    # Wir suchen das Objekt bei welchem sich das Model am sichersten ist (größter Wert)
    found_object_index = np.argmax(output_data)
    # Hier schreiben wir das am eindeutigste erkannte Objekt in das live Video
    label = f"Gefunden: {class_names[found_object_index]} ({output_data[found_object_index]:.2f})"
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Objekterkennung", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
