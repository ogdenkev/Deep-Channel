"""Predict Idealized States from DeepChannel Model
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

import tensorflow as tf
from tensorflow.keras.models import load_model

def load_data(path):
    return pd.read_csv(
        path,
        header=None,
        names=["time", "current", "state"],
        dtype={"time": np.float64, "current": np.float64, "state": np.int32},
    )


MODEL = Path.cwd() / "deepchannel-model-20200716T033708Z.h5"
INFERENCE_PATH = Path.cwd() / "nmdar_data" / "clean-gibb-data"

batch_size = 256

# Load the model
print(f"Loading model from {MODEL}")
model = load_model(MODEL)

# Load data
for data_path in INFERENCE_PATH.glob("*.csv"):
    print(f"Loading data from {str(data_path)}")
    val_data = load_data(data_path)
    
    # Transform the data
    # MinMaxScaler cannot handle outliers, such as large current spike artifacts in the recordings
    # Instead of the minimum and maximum, try the 0.5 percentile and the 99.5 percentile
    # which will cover 99% of the range of x
    # scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = RobustScaler(with_centering=False, quantile_range=(0.5, 99.5))
    current_val = scaler.fit_transform(val_data[["current"]])
    current_val = -current_val
    x_val = current_val.reshape(-1, 1, 1, 1)

    # Make predictions
    predict_val = model.predict(x_val, batch_size=batch_size, verbose=1)
    predict_val = pd.DataFrame(
        predict_val,
        columns=[f"prob_state_{i}" for i in range(predict_val.shape[1])]
    )

    out = pd.concat([val_data, predict_val, pd.DataFrame(current_val, columns=["scaled_current"])], axis=1, ignore_index=False)

    out_path = data_path.with_name(data_path.stem + "_pred.csv.gz")
    print(f"Saving data to {str(out_path)}")
    out.to_csv(out_path, index=False)
