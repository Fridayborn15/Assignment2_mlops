import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import subprocess
import wandb  # Import Weights & Biases
from evidently.report import Report
from evidently.metrics import DataDriftTable
import os
import json

wandb.login(key="8c479eaf9bacd90790128eb5107b85cb7c84c2ca")

# Initialize W&B for tracking
wandb.init(project="fashion-mnist-drift", name="drift-detection-run")

# Load Fashion-MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize pixel values to [0,1]
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

# Introduce Feature Drift (Increase Brightness for T-Shirts (0) and Sneakers (7))
drift_classes = [0, 7, 9, 16]  # Classes with drift
factor = 1.5  # Increase brightness by 50%

def introduce_feature_drift(images, labels, drift_classes, factor):
    drift_images = images.copy()
    for i in range(len(drift_images)):
        if labels[i] in drift_classes:
            drift_images[i] = np.clip(drift_images[i] * factor, 0, 1)  # Keep values in range [0,1]
    return drift_images

# Apply drift
x_test_drifted = introduce_feature_drift(x_test, y_test, drift_classes, factor)

# Ensure column names are consistent and explicitly set
column_names = [f'pixel_{i}' for i in range(x_test.shape[1] * x_test.shape[2])]

reference_data = pd.DataFrame(x_test.reshape(x_test.shape[0], -1), columns=column_names)
new_data = pd.DataFrame(x_test_drifted.reshape(x_test_drifted.shape[0], -1), columns=column_names)

# Run drift detection
data_drift_report = Report(metrics=[DataDriftTable()])
data_drift_report.run(reference_data=reference_data, current_data=new_data)

# Extract drift results
drift_results = data_drift_report.as_dict()
drift_summary = drift_results["metrics"][0]["result"]["share_of_drifted_columns"]
print("Observed Drift Summary ----- ", drift_summary)

# Log Drift Summary to W&B
wandb.log({"drift_share": drift_summary})

# Save W&B logs to a file
log_data = {"drift_share": drift_summary}
with open("wandb_logs.json", "w") as f:
    json.dump(log_data, f)

# Log the saved file as an artifact
artifact = wandb.Artifact("drift_logs", type="dataset")
artifact.add_file("wandb_logs.json")
wandb.log_artifact(artifact)

# Set Drift Threshold (20%)
DRIFT_THRESHOLD = 0.2  

# Define Ensemble Stacked Model
def build_stacked_model():
    input_layer = tf.keras.layers.Input(shape=(28, 28))
    
    # First Base Model
    base_model1 = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Second Base Model
    base_model2 = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Get Outputs from Base Models
    output1 = base_model1(input_layer)
    output2 = base_model2(input_layer)
    
    # Concatenate Outputs
    merged = tf.keras.layers.Concatenate()([output1, output2])
    
    # Final Output Layer
    output_layer = tf.keras.layers.Dense(10, activation='softmax')(merged)
    
    stacked_model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
    
    stacked_model.compile(optimizer='adam',
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
    return stacked_model

# Trigger Model Retraining if Drift is Detected
if drift_summary > DRIFT_THRESHOLD:
    print("Significant data drift detected! Retraining the model...")
    wandb.alert(title="Drift Alert", text="Drift exceeds threshold, triggering retraining.")

    # Train a new ensemble stacked model
    def train_and_save_model():
        wandb.init(project="fashion-mnist-drift", name="model-retraining-run")

        model = build_stacked_model()

        # Train with Drifted Data and Log Performance
        for epoch in range(5):
            history = model.fit(x_test_drifted, y_test, epochs=1, validation_data=(x_test, y_test), verbose=1)

            # Log training metrics to W&B
            wandb.log({
                "epoch": epoch + 1,
                "loss": history.history["loss"][0],
                "val_loss": history.history["val_loss"][0],
                "accuracy": history.history["accuracy"][0],
                "val_accuracy": history.history["val_accuracy"][0]
            })

        # Save Updated Model
        model.save("fashion_mnist_updated_model.h5")
        wandb.save("fashion_mnist_updated_model.h5")  # Log model artifact to W&B
        print("Ensemble stacked model retrained and saved as 'fashion_mnist_updated_model.h5'")

        wandb.finish()  # Finish W&B tracking

    train_and_save_model()
else:
    print("No significant drift detected. Model retraining not required.")

wandb.finish()  # Finish W&B logging
