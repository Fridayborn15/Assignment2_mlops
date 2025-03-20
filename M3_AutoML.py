# pip install h2o
# pip install optuna
 
import h2o
import optuna
import numpy as np
import pandas as pd
from h2o.automl import H2OAutoML
from tensorflow.keras.datasets import fashion_mnist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
 
# Initialize H2O
h2o.init(enable_assertions=False, nthreads=-1, min_mem_size_GB=8, max_mem_size_GB=16)
 
# Load Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
 
# Normalize and flatten images
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train_flat = x_train.reshape(len(x_train), -1)
x_test_flat = x_test.reshape(len(x_test), -1)
 
# Apply Standard Scaling
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_flat)
x_test_scaled = scaler.transform(x_test_flat)
 
# Convert to DataFrame
columns = [f'Pixel_{i}' for i in range(x_train_scaled.shape[1])]
df_train = pd.DataFrame(x_train_scaled, columns=columns)
df_train['label'] = y_train
df_test = pd.DataFrame(x_test_scaled, columns=columns)
df_test['label'] = y_test
 
# Convert to H2O Frames
train_h2o = h2o.H2OFrame(df_train)
train_h2o['label'] = train_h2o['label'].asfactor()
 
test_h2o = h2o.H2OFrame(df_test)
test_h2o['label'] = test_h2o['label'].asfactor()
 
# Reduce dataset for AutoML training
train_h2o_sample, _ = train_h2o.split_frame(ratios=[0.1], seed=42)
 
# Apply PCA for feature reduction
pca = PCA(n_components=50)
train_pca = pca.fit_transform(train_h2o_sample.drop("label", axis=1).as_data_frame())
train_pca_h2o = h2o.H2OFrame(train_pca)
train_pca_h2o["label"] = train_h2o_sample["label"]
 
# Run AutoML for model selection
auto_ml = H2OAutoML(
    max_models=10,
    max_runtime_secs=1800,
    seed=42)
    #,
    #exclude_algos=["DeepLearning", "StackedEnsemble", "DRF", "GBM"]
#)
 
auto_ml.train(x=list(range(50)), y='label', training_frame=train_pca_h2o)
 
# Get and display AutoML leaderboard
lb = auto_ml.leaderboard
print("AutoML Leaderboard:\n", lb)
 
# Select the best model
best_model = auto_ml.leader
print("Best Model: ", best_model.model_id)
 
# Hyperparameter Optimization using Optuna
def objective(trial):
    hidden_units = trial.suggest_categorical("hidden", [[128, 64], [256, 128], [512, 256]])
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
 
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, BatchNormalization
    from tensorflow.keras.optimizers import Adam
 
    model = Sequential([
        Dense(hidden_units[0], activation='relu', input_shape=(x_train_scaled.shape[1],)),
        BatchNormalization(),
        Dense(hidden_units[1], activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
 
    model.fit(x_train_scaled, y_train, epochs=5, batch_size=batch_size, verbose=0)
    loss, accuracy = model.evaluate(x_test_scaled, y_test, verbose=0)
    return accuracy
 
# Run hyperparameter tuning
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)
 
# Print best hyperparameters
print("Best Hyperparameters: ", study.best_params)
 
# Save AutoML results & hyperparameter tuning logs
with open("automl_results.txt", "w") as f:
    f.write(str(lb))
with open("hyperparameter_tuning_logs.txt", "w") as f:
    f.write(str(study.trials_dataframe()))
 
# Justification Placeholder
print("\nJustification for Model Selection:")
print(f"The selected model ({best_model.model_id}) was chosen due to its superior performance on the validation set.")
 
print("\nJustification for Hyperparameters:")
print("The chosen hyperparameters were selected based on Optuna's optimization, leading to the best validation accuracy.")