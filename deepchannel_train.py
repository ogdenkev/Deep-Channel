# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:10:07 2019

@author: ncelik34
"""


# Importing the libraries
import datetime
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE

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

from tensorflow_addons.metrics import F1Score


def step_decay(epoch):
    # Learning rate scheduler object
    initial_lrate = 0.001
    drop = 0.001
    epochs_drop = 3.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


'''
############# SET UP RUN HERE ####################
'''

batch_size = 256

DATA_PATH = "nmdar_data/NMDA_1.csv"

print(f"Loading data from {DATA_PATH}")
df = pd.read_csv(
    DATA_PATH,
    header=None,
    names=["time", "current", "state"],
    dtype={"time": np.float64, "current": np.float64, "state": np.int32},
)

idataset = df["state"]
maxchannels = idataset.max()

print("Scaling data to between 0 and 1")
scaler = MinMaxScaler(feature_range=(0, 1))
current = scaler.fit_transform(df[["current"]])

print("Creating train/test split")
x_train, x_test, y_train, y_test = train_test_split(
    current,
    idataset,
    train_size=0.8,
    shuffle=False
)

print("Creating synthetic minority oversampled data")
sm = SMOTE(sampling_strategy='auto', random_state=42)
X_res, Y_res = sm.fit_resample(x_train, y_train)

print("Finalizing data prep for model")
yy_res = Y_res.to_numpy().reshape((len(Y_res), 1))
yy_res = to_categorical(yy_res, num_classes=maxchannels+1)
xx_res, yy_res = shuffle(X_res, yy_res)

in_train = xx_res
in_test = x_test
target_train = yy_res
target_test = to_categorical(y_test, num_classes=maxchannels+1)
in_train = in_train.reshape(len(in_train), 1, 1, 1)
in_test = in_test.reshape(len(in_test), 1, 1, 1)


# validation set!!
VAL_DATA_PATH = "./nmdar_data/NMDA_2.csv"

print(f"Loading validation data from {VAL_DATA_PATH}")
df_val = df = pd.read_csv(
    VAL_DATA_PATH,
    header=None,
    names=["time", "current", "state"],
    dtype={"time": np.float64, "current": np.float64, "state": np.int32},
)

idataset2 = df_val["state"]

val_set = df_val[["current"]]
val_set = scaler.transform(val_set)
val_set = val_set.reshape(len(val_set), 1, 1, 1)
val_target = to_categorical(idataset2, num_classes=maxchannels+1)

# model starts..
print("Building the model")

newmodel = Sequential()
timestep = 1
input_dim = 1
newmodel.add(TimeDistributed(Conv1D(filters=64, kernel_size=1,
                                    activation='relu'), input_shape=(None, timestep, input_dim)))
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

newmodel.add(Dense(maxchannels+1))
newmodel.add(Activation('softmax'))


newmodel.compile(
    loss='categorical_crossentropy',
    optimizer=optimizers.SGD(lr=0.001, momentum=0.9, nesterov=False),
    metrics=[
        'accuracy', Precision(), Recall(),
        F1Score(num_classes=maxchannels+1, average='micro')
    ]
)

print(newmodel.summary())

lrate = LearningRateScheduler(step_decay)

print("Training the model")
epochers = 2
history = newmodel.fit(x=in_train, y=target_train, initial_epoch=0, epochs=epochers, batch_size=batch_size, callbacks=[
                       lrate], verbose=1, shuffle=False, validation_data=(in_test, target_test))


# prediction for test set
predict = newmodel.predict(in_test, batch_size=batch_size)

# prediction for val set
predict_val = newmodel.predict(val_set, batch_size=batch_size)


class_predict = np.argmax(predict, axis=-1)
class_predict_val = np.argmax(predict_val, axis=-1)
class_target = np.argmax(target_test, axis=-1)
class_target_val = np.argmax(val_target, axis=-1)


cm_test = confusion_matrix(class_target, class_predict)
cm_val = confusion_matrix(idataset2, class_predict_val)

print("Confusion Matrix on Validation Set")
print(cm_val)

rnd = 1
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='lower right')
plt.savefig(str(rnd)+'acc.png')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.savefig(str(rnd)+'loss.png')
plt.show()


MODEL_PATH = datetime.datetime.now(datetime.timezone.utc).strftime("deepchannel-model-%Y%m%dT%H%M%SZ.h5")

print(f"Saving model to {MODEL_PATH}")
newmodel.save(MODEL_PATH)
