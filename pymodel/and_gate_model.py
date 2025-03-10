import tensorflow as tf
import numpy as np

# Datos de entrenamiento para una puerta AND
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
y = np.array([[0], [0], [0], [1]], dtype=np.float32)

# Crear el modelo
model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, input_dim=2, activation='relu'),  # Capa oculta con 2 neuronas
    tf.keras.layers.Dense(1, activation='sigmoid')            # Capa de salida con activación sigmoidea
])

# Compilar el modelo
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X, y, epochs=1000, verbose=0)

# Evaluar el modelo
loss, accuracy = model.evaluate(X, y)
print(f"Precisión del modelo: {accuracy * 100}%")

# Hacer predicciones
predictions = model.predict(X)
print("Predicciones para las combinaciones de entradas AND:")
print(predictions)

# Guardar el modelo en un archivo
model.save('and_gate_model.h5')

# Convertir el modelo a TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Guardar el modelo TFLite
with open('and_gate_model.tflite', 'wb') as f:
    f.write(tflite_model)
