import numpy as np
import tensorflow as tf

# Definir los datos de entrada y salida de la compuerta AND
x_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [0], [0], [1]])

# Crear el modelo de la red neuronal
model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, input_shape=(2,), activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(x_train, y_train, epochs=1000)

# Hacer predicciones
x_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predictions = model.predict(x_test)

# Redondear las predicciones a 0 o 1
binary_predictions = np.round(predictions)

# Imprimir las predicciones binarias
for i in range(len(x_test)):
    print(f"Entrada: {x_test[i]}, Predicción binaria: {binary_predictions[i]}")
