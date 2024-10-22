
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO messages
import tensorflow as tf
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt


mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),      
    layers.Dropout(0.2),                      
    layers.Dense(10, activation='softmax')     
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')

# Predict on test data
predictions = model.predict(x_test)

# Visualize the first test image and its predicted label
plt.imshow(x_test[0], cmap=plt.cm.binary)
plt.title(f'Predicted label: {np.argmax(predictions[0])}')
plt.show()

