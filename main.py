import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Größe der Bilder (,3 für die 3 Farben)
inputs = Input(shape=(32, 32, 3))
Klassen = 10

# Unser Modell. Conv2D sind die Filter/Muster, welche automatisch gelernt werden
x = Conv2D(32, (3, 3), activation='relu')(inputs)
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
