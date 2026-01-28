# handwritten-digit-recognition-
📌 Project Overview

The objectives of this project are:

Load and preprocess the MNIST handwritten digit dataset

Train a baseline MLP model

Train an advanced CNN model

Compare model performance using accuracy and F1-score

Visualize predictions on test images

Save the trained CNN model for future use

📂 Dataset

MNIST Dataset

70,000 grayscale images of handwritten digits (0–9)

60,000 training images

10,000 test images

Image size: 28 × 28 pixels

Dataset is loaded directly using:

from tensorflow.keras.datasets import mnist

⚙️ Libraries Used

numpy – numerical computation

matplotlib – visualization

tensorflow / keras – deep learning models

scikit-learn – evaluation metrics

🔄 Data Preprocessing

Pixel values normalized to the range [0, 1]

Labels converted to one-hot encoding

Images reshaped based on model type:

MLP: flattened to 784 features

CNN: reshaped to (28, 28, 1)

🧠 Model 1: Multi-Layer Perceptron (MLP)
Architecture

Input Layer: 784 neurons

Hidden Layers:

Dense (128 units, ReLU)

Dense (64 units, ReLU)

Output Layer:

Dense (10 units, Softmax)

Training

Optimizer: Adam

Loss Function: Categorical Crossentropy

Epochs: 5

Batch Size: 32

Evaluation Metrics

Accuracy

Weighted F1 Score

🧠 Model 2: Convolutional Neural Network (CNN)
Architecture

Conv2D (32 filters, 3×3, ReLU)

MaxPooling2D (2×2)

Conv2D (64 filters, 3×3, ReLU)

MaxPooling2D (2×2)

Flatten

Dense (128 units, ReLU)

Dense (10 units, Softmax)

Training

Optimizer: Adam

Loss Function: Categorical Crossentropy

Epochs: 5

Batch Size: 32

📊 Model Evaluation

Both models are evaluated on the test dataset using:

Accuracy Score

Weighted F1 Score

The CNN significantly outperforms the MLP due to its ability to capture spatial features in images.

🖼️ Prediction Visualization

Sample test images are displayed along with their predicted digit labels using the CNN model. This helps visually assess the model’s prediction quality.

💾 Model Saving
