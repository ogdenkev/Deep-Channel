"""Predict Idealized States from DeepChannel Model
"""

import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("model", metavar="MODEL", help="Path to a trained DeepLearning model")
parser.add_argument("predict", metavar="PREDICT", nargs="+", help="Path(s) to the single molecule data for inferences. The file(s) must be in CSV format with three columns.")
parser.add_argument("--out", "-o", metavar="OUT", help="Path to save the predictions. By default the predictions will be written to standard output.")
parser.add_argument("--scaler", "-s", metavar="SCALER", help="Path to the saved MinMaxScaler to use to transform data before predictions. Bf default a new scaler is trained on the prediction data")
args = parser.parse_args()


import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import load_model


def load_data(path):
    return pd.read_csv(
        path,
        header=None,
        names=["time", "current", "state"],
        dtype={"time": np.float64, "current": np.float64, "state": np.int32},
    )


batch_size = 256

# Load data
print(f"Loading data from {args.predict}")
dfs = map(load_data, args.predict)
val_data = pd.concat(dfs, axis=0, ignore_index=True)

# Transform the data
if args.scaler:
    with open(args.scaler, "rb") as fd:
        scaler = pickle.load(fd)
else:
    # MinMaxScaler cannot handle outliers, such as large current spike artifacts in the recordings
    # Instead of the minimum and maximum, try the 0.5 percentile and the 99.5 percentile
    # which will cover 99% of the range of x
    # scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = RobustScaler(with_centering=False, quantile_range=(0.5, 99.5))
    scaler.fit(val_data[["current"]])

current_val = scaler.transform(val_data[["current"]])
x_val = current_val.reshape(-1, 1, 1, 1)

# Load the model
print(f"Loading model from {args.predict}")
model = load_model(args.model)

# Make predictions
predict_val = model.predict(x_val, batch_size=batch_size, verbose=1)
predict_val = pd.DataFrame(
    predict_val,
    columns=[f"prob_state_{i}" for i in range(predict_val.shape[1])]
)

out = pd.concat([val_data, predict_val, pd.DataFrame(current_val, columns=["scaled_current"])], axis=1, ignore_index=False)

if args.out:
    out.to_csv(args.out, index=False)
else:
    print(out.to_csv(index=False))
