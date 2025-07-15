import tensorflow as tf

model = tf.keras.models.load_model("garbage_slayer_v1.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("garbage_slayer_v1.tflite", "wb") as f:
    f.write(tflite_model)