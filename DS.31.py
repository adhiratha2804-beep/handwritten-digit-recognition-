# ===============================
# 1. IMPORT LIBRARIES
# ===============================
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import accuracy_score, f1_score


# ===============================
# 2. LOAD & PREPROCESS DATA
# ===============================
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

y_test_original = y_test.copy()


# ===============================
# 3. BASELINE MODEL (MLP)
# ===============================
x_train_flat = x_train.reshape(-1, 784)
x_test_flat = x_test.reshape(-1, 784)

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

mlp_model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

mlp_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

mlp_model.fit(x_train_flat, y_train_cat, epochs=5, batch_size=32)

mlp_preds = mlp_model.predict(x_test_flat)
mlp_pred_labels = np.argmax(mlp_preds, axis=1)

print("MLP Accuracy:", accuracy_score(y_test_original, mlp_pred_labels))
print("MLP F1 Score:", f1_score(y_test_original, mlp_pred_labels, average='weighted'))


# ===============================
# 4. CNN MODEL
# ===============================
x_train_cnn = x_train.reshape(-1, 28, 28, 1)
x_test_cnn = x_test.reshape(-1, 28, 28, 1)

cnn_model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

cnn_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

cnn_model.fit(x_train_cnn, y_train_cat, epochs=5, batch_size=32)


# ===============================
# 5. CNN EVALUATION
# ===============================
cnn_preds = cnn_model.predict(x_test_cnn)
cnn_pred_labels = np.argmax(cnn_preds, axis=1)

print("CNN Accuracy:", accuracy_score(y_test_original, cnn_pred_labels))
print("CNN F1 Score:", f1_score(y_test_original, cnn_pred_labels, average='weighted'))


# ===============================
# 6. VISUALIZE PREDICTIONS
# ===============================
plt.figure(figsize=(8,4))
for i in range(6):
    plt.subplot(2,3,i+1)
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Predicted: {cnn_pred_labels[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()


# ===============================
# 7. SAVE MODEL
# ===============================
cnn_model.save("digit_recognition_cnn_model.h5")
print("Model saved successfully!")