import subprocess
import sys

# Ensure required dependencies are installed
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install("sagemaker")

import sagemaker
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os

# Load the dataset
data_path = "/opt/ml/input/data/train"
red_wine = pd.read_csv(os.path.join(data_path, "winequality-red.csv"), sep=";")
white_wine = pd.read_csv(os.path.join(data_path, "winequality-white.csv"), sep=";")

# Combine datasets
data = pd.concat([red_wine, white_wine], axis=0)

# Define features and target
X = data.drop(columns=["quality"])
y = data["quality"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Save model
model_path = "/opt/ml/model/wine_quality_model.pkl"
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

