# Install dependencies if needed
# !pip install lime shap scikit-image

import os
import numpy as np
import matplotlib.pyplot as plt
import shap
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from lime import lime_image
from skimage.segmentation import mark_boundaries
from skimage.color import gray2rgb, rgb2gray

# Create a "reports" directory if it doesn't exist
reports_dir = "reports"
os.makedirs(reports_dir, exist_ok=True)

# Load dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize pixel values to range [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# Flatten images for simpler modeling
x_train_flat = x_train.reshape(x_train.shape[0], -1)
x_test_flat = x_test.reshape(x_test.shape[0], -1)

# Standardize features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_flat)
x_test_scaled = scaler.transform(x_test_flat)

# Convert labels to one-hot encoding
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# Define a simple neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(784,)),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train_scaled, y_train_cat, epochs=5, batch_size=32, validation_data=(x_test_scaled, y_test_cat))

# ---- SHAP Explainability ----

# SHAP Kernel Explainer
background = x_train_scaled[np.random.choice(x_train_scaled.shape[0], 100, replace=False)]  # 100 background samples
 
explainer_shap = shap.KernelExplainer(model.predict, background)

# Compute SHAP values for first 10 test samples
shap_values = explainer_shap.shap_values(x_test_scaled[:10], nsamples=1000)

# Reshape SHAP values
shap_values_array = np.array(shap_values).reshape(-1, 28, 28)

# Save SHAP visualizations
shap_plot_path = os.path.join(reports_dir, "shap_explanation.png")

plt.figure(figsize=(10, 5))
shap.image_plot(shap_values_array, -x_test[:10], show=False)
plt.savefig(shap_plot_path)
plt.close()
print(f"SHAP visualization saved to {shap_plot_path}")

# ---- LIME Explainability ----

# Function to predict using the trained model
def predict_fn(images):
    if images.shape[-1] == 3:  # Convert RGB back to grayscale
        images = rgb2gray(images)  # Shape: (num_samples, 28, 28)

    images = images.reshape(images.shape[0], -1)  # Flatten to (num_samples, 784)
    return model.predict(images)  # Returns shape (num_samples, 10)

# Initialize LIME explainer
explainer_lime = lime_image.LimeImageExplainer()

# Select a test image
x_test_rgb = gray2rgb(x_test[0])  # Convert grayscale to RGB for LIME

# Generate explanation
explanation = explainer_lime.explain_instance(
    x_test_rgb,  # Pass 3-channel image
    predict_fn,
    top_labels=5,
    hide_color=0,
    num_samples=1000
)

# Visualize explanation
image, mask = explanation.get_image_and_mask(
    explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=False
)

# Save LIME visualization
lime_plot_path = os.path.join(reports_dir, "lime_explanation.png")

plt.figure(figsize=(6, 6))
plt.imshow(mark_boundaries(image, mask))
plt.axis("off")
plt.title("LIME Explanation")
plt.savefig(lime_plot_path)
plt.close()
print(f"LIME visualization saved to {lime_plot_path}")
