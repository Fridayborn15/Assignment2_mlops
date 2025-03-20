# Install ydata-profiling if not installed
# pip install ydata-profiling

import pandas as pd
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from ydata_profiling import ProfileReport

# Load the Fashion MNIST dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Combine training and test data
all_images = tf.concat([train_images, test_images], axis=0).numpy()
all_labels = tf.concat([train_labels, test_labels], axis=0).numpy()

# Resize images to (6,6) to reduce dimensionality
new_size = (6, 6)
all_images_resized = tf.image.resize(tf.expand_dims(all_images, axis=-1), new_size, method='area').numpy().squeeze()

# Flatten the images
all_images_flattened = all_images_resized.reshape(all_images.shape[0], -1)

# Create a Pandas DataFrame
df_all = pd.DataFrame(all_images_flattened)
df_all['label'] = all_labels  # Add labels as a column

# Display dataset info
print(df_all.head())
print(f"Dataset shape: {df_all.shape}")

# Generate the profiling report
profile = ProfileReport(df_all, title="Fashion MNIST Combined Dataset Profiling Report", explorative=True)

# Save the report as an HTML file
profile.to_file("reports/fashion_mnist_report.html")

print("Report successfully generated and saved as 'fashion_mnist_report.html'.")
