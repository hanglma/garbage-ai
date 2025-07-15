import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Größe der Bilder (,3 für die 3 Farben)
inputs = Input(shape=(180, 180, 3))
Klassen = 12

# Unser Modell. Conv2D sind die Filter/Muster, welche automatisch gelernt werden
x = Conv2D(180, (3, 3), activation='relu')(inputs)
# Pooling verringert die Größe der Filterergebnisse --> Damit fokussieren wir uns nur auf das wesentliche
x = MaxPooling2D((2, 2))(x)
# Wir geben immer die vorhergehenden Ergebnisse in die nächste Stufe, hier eine weitere Filter/Muster Ebene.
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
# Hier machen wir einfach aus den Ergebnissbildern eine lange "Tabellenspalte" mit den extrem wichtigen, 
# über mehreren Ebenen gefundenen Filter/Musterergebnissen...
x = Flatten()(x)
# ... und zwingen das Modell dazu, diese Ergebnisse in 10 unterschiedliche Klassen zu unterteilen
outputs = Dense(Klassen, activation='softmax')(x) 

# Hier bauen wir alles zusammen. Die Bilder (inputs) und die Ausgabe (ein Wert von 1-10, je nachdem was das Bild zeigt).
model = Model(inputs=inputs, outputs=outputs)
model.summary()

path = "archive/garbage_classification/"

val_split = 0.2
train_ds = tf.keras.utils.image_dataset_from_directory(
    path,
    validation_split=val_split,
    subset="training",
    seed=123,
    image_size=(180, 180),
    batch_size=32
)

for images, labels in train_ds.take(1):
    print(labels.numpy())

val_ds = tf.keras.utils.image_dataset_from_directory(
    path,
    validation_split=val_split,
    subset="validation",
    seed=123,
    image_size=(180, 180),
    batch_size=32
)


normalization = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization(x), y))
val_ds = val_ds.map(lambda x, y: (normalization(x), y))

# Prefetching für Performance
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
history = model.fit(train_ds, epochs=10, validation_data=val_ds)