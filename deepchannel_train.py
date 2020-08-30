# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:10:07 2019

@author: ncelik34
"""

import argparse

parser = argparse.ArgumentParser(description="Train a deep learning network to detect events in single molecule data")
parser.add_argument("train", metavar="TRAIN", nargs="+", help="Path(s) to the single molecule data for training. The file(s) must be in CSV format with three columns.")
parser.add_argument("--levels", default="0,-5.4", help="A comma-separated list giving the amplitude level for each state [default: %(default)s]")
parser.add_argument("--out", "-o", metavar="OUT", help="Path to save the trained model. By default the model will be saved to deepchannel-model-<TIMESTAMP>.h5")
parser.add_argument("--scaler", help="Path to which the trained MinMaxScaler will be saved. By default, the scaler will be saved to deepchannel-scaler-<TIMESTAMP>.pkl")
parser.add_argument("--history", help="Path to save the training history. By default, the history will be saved to deepchannel-training-history-<TIMESTAMP>.pkl")
args = parser.parse_args()

# Importing the libraries
import datetime
import math
import pickle
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (
    Dense, Dropout, Flatten, Reshape, Activation, LSTM, BatchNormalization,
    TimeDistributed, Conv1D, MaxPooling1D
)
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical

# from tensorflow_addons.metrics import F1Score


def step_decay(epoch):
    # Learning rate scheduler object
    initial_lrate = 0.001
    drop = 0.001
    epochs_drop = 3.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


def _load_data(state_levels, path):
    df = pd.read_csv(
        path,
        header=None,
        names=["time", "current", "state"],
        dtype={"time": np.float64, "current": np.float64, "state": np.float64},
    )
    # df["state"] = df["state"].astype("category").cat.codes
    states = np.ones(df.shape[0], dtype=np.int32)
    state_in_levels = np.zeros(df.shape[0], dtype=bool)
    for i, lev in enumerate(state_levels):
        m = np.isclose(df["state"], lev)
        if np.any(state_in_levels & m):
            raise ValueError("Some of the states were similar to more than one of the amplitude levels")
        state_in_levels |= m
        states[m] = i
    if not np.all(state_in_levels):
        unmatched_levels = df.loc[~state_in_levels, "state"].sample(3).to_list()
        raise ValueError(f"Some of the states, e.g. {unmatched_levels}, were not close to any of the levels")
    df.loc[:, "state"] = states
    return df


'''
############# SET UP RUN HERE ####################
'''

batch_size = 256

DATA_PATH = args.train  # "nmdar_data/NMDA_1.csv"
state_levels = [float(x) for x in args.levels.split(",")]
load_data = partial(_load_data, state_levels)

print(f"Loading data from {DATA_PATH} with state levels {state_levels}")
dfs = map(load_data, DATA_PATH)
train_data = pd.concat(dfs, axis=0, ignore_index=True)

max_channels = train_data["state"].max()
print("Number of channels:", max_channels)

print("Scaling data to between 0 and 1")
scaler = MinMaxScaler(feature_range=(0, 1))
current = scaler.fit_transform(train_data[["current"]])

print("Creating train/test split")
x_train, x_test, y_train, y_test = train_test_split(
    current.reshape(-1, 1, 1, 1),
    to_categorical(train_data[["state"]].to_numpy(), num_classes=max_channels + 1),
    train_size=0.8,
    shuffle=False
)
print("x_train shape", x_train.shape)
print("y_train shape", y_train.shape)
print("x_test shape", x_test.shape)
print("y_test shape", y_test.shape)

# model starts..
print("Building the model")

newmodel = Sequential()
timestep = 1
input_dim = 1
newmodel.add(
    TimeDistributed(
        Conv1D(filters=64, kernel_size=1, activation='relu'),
        input_shape=(None, timestep, input_dim)
    )
)
newmodel.add(TimeDistributed(MaxPooling1D(pool_size=1)))
newmodel.add(TimeDistributed(Flatten()))

newmodel.add(LSTM(256, activation='relu', return_sequences=True))
newmodel.add(BatchNormalization())
newmodel.add(Dropout(0.2))

newmodel.add(LSTM(256, activation='relu', return_sequences=True))
newmodel.add(BatchNormalization())
newmodel.add(Dropout(0.2))

newmodel.add(LSTM(256, activation='relu'))
newmodel.add(BatchNormalization())
newmodel.add(Dropout(0.2))

newmodel.add(Dense(max_channels + 1))
newmodel.add(Activation('softmax'))


newmodel.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.SGD(lr=0.001, momentum=0.9, nesterov=False),
    metrics=[
        'accuracy', Precision(), Recall(),
        # F1Score(num_classes=max_channels + 1, average='micro')
    ]
)

print(newmodel.summary())

lrate = LearningRateScheduler(step_decay)

print("Training the model")
epochers = 2
history = newmodel.fit(
    x=x_train,
    y=y_train,
    initial_epoch=0,
    epochs=epochers,
    batch_size=batch_size,
    callbacks=[lrate],
    verbose=1,
    shuffle=False,
    validation_data=(x_test, y_test)
)

TIMESTAMP = datetime.datetime.now(datetime.timezone.utc)

# Save trained model
if args.out:
    MODEL_PATH = args.out
else:
    MODEL_PATH = TIMESTAMP.strftime("deepchannel-model-%Y%m%dT%H%M%SZ.h5")

print(f"Saving model to {MODEL_PATH}")
newmodel.save(MODEL_PATH)

# Save training history
if args.history:
    HISTORY_PATH = args.history
else:
    HISTORY_PATH = TIMESTAMP.strftime("deepchannel-training-history-%Y%m%dT%H%M%SZ.pkl")

print(f"Saving training history to {HISTORY_PATH}")
with open(HISTORY_PATH, "wb") as fd:
    pickle.dump(history.history, fd)

if args.scaler:
    SCALER_PATH = args.scaler
else:
    SCALER_PATH = TIMESTAMP.strftime("deepchannel-scaler-%Y%m%dT%H%M%SZ.pkl")

with open(SCALER_PATH, "wb") as fd:
    pickle.dump(scaler, fd)
