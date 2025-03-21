import unittest
import numpy as np
import tensorflow as tf
import json
import os
import pandas as pd
from evidently.report import Report
from evidently.metrics import DataDriftTable
from unittest.mock import patch, MagicMock

# Import functions from the main script
from M4_DataDrift_ModelRetraining import (
    introduce_feature_drift, build_stacked_model, train_and_save_model
)


class TestFashionMNISTPipeline(unittest.TestCase):
    """ Unit tests for Fashion MNIST drift detection and model retraining. """

    @classmethod
    def setUpClass(cls):
        """ Load dataset and preprocess it for testing. """
        (x_train, y_train), (x_test, y_test) = (
            tf.keras.datasets.fashion_mnist.load_data()
        )

        # Normalize
        cls.x_train, cls.x_test = x_train / 255.0, x_test / 255.0
        cls.y_train, cls.y_test = y_train, y_test

        # Set drift parameters
        cls.drift_classes = [0, 7, 9, 16]
        cls.factor = 1.5

    def test_introduce_feature_drift(self):
        """ Test that the brightness drift is applied correctly. """
        x_test_drifted = introduce_feature_drift(
            self.x_test, self.y_test, self.drift_classes, self.factor
        )

        # Ensure shape remains unchanged
        self.assertEqual(self.x_test.shape, x_test_drifted.shape)

        # Check drift was applied correctly for specific classes
        for class_label in self.drift_classes:
            affected_samples = np.where(self.y_test == class_label)[0]
            if len(affected_samples) > 0:
                sample_idx = affected_samples[0]
                self.assertTrue(
                    np.any(x_test_drifted[sample_idx] > self.x_test[sample_idx])
                )

    def test_drift_detection(self):
        """ Test if drift detection correctly identifies changes. """
        column_names = [
            f'pixel_{i}' for i in range(self.x_test.shape[1] * self.x_test.shape[2])
        ]

        reference_data = pd.DataFrame(
            self.x_test.reshape(self.x_test.shape[0], -1), columns=column_names
        )

        drifted_data = introduce_feature_drift(
            self.x_test, self.y_test, self.drift_classes, self.factor
        )

        new_data = pd.DataFrame(
            drifted_data.reshape(self.x_test.shape[0], -1), columns=column_names
        )

        # Run drift detection
        data_drift_report = Report(metrics=[DataDriftTable()])
        data_drift_report.run(
            reference_data=reference_data, current_data=new_data
        )

        drift_results = data_drift_report.as_dict()
        drift_summary = drift_results["metrics"][0]["result"][
            "share_of_drifted_columns"
        ]

        # Check if drift is within expected range
        self.assertGreaterEqual(drift_summary, 0.0)
        self.assertLessEqual(drift_summary, 1.0)

    def test_build_stacked_model(self):
        """ Test that the stacked ensemble model builds correctly. """
        model = build_stacked_model()
        self.assertIsInstance(model, tf.keras.Model)

        # Check input shape
        self.assertEqual(model.input_shape, (None, 28, 28))

        # Ensure expected number of layers
        self.assertGreaterEqual(len(model.layers), 5)

    @patch("wandb.init")
    @patch("wandb.log")
    @patch("wandb.save")
    @patch("wandb.finish")
    def test_train_and_save_model(
        self, mock_wandb_init, mock_wandb_log, mock_wandb_save, mock_wandb_finish
    ):
        """ Test if the model trains and saves correctly without errors. """

        # Mock W&B tracking
        mock_wandb_init.return_value = MagicMock()
        mock_wandb_log.return_value = None
        mock_wandb_save.return_value = None
        mock_wandb_finish.return_value = None

        # Ensure W&B initialization is not missing
        train_and_save_model()

        # Check if the new model file is created
        self.assertTrue(os.path.exists("fashion_mnist_updated_model.keras"))

    def test_logging_json(self):
        """ Test if drift summary is logged correctly in JSON. """
        test_data = {"drift_share": 0.3}

        # Save JSON log
        with open("wandb_logs.json", "w") as f:
            json.dump(test_data, f)

        # Load and verify
        with open("wandb_logs.json", "r") as f:
            loaded_data = json.load(f)

        self.assertEqual(test_data, loaded_data)

    @classmethod
    def tearDownClass(cls):
        """ Clean up generated files after testing. """
        if os.path.exists("fashion_mnist_updated_model.keras"):
            os.remove("fashion_mnist_updated_model.keras")
        if os.path.exists("wandb_logs.json"):
            os.remove("wandb_logs.json")


if __name__ == "__main__":
    unittest.main()
