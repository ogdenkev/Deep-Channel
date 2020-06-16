"""Train a CNN to idealize single channel data"""

# Importing the libraries
import os
import numpy
import time
import random
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.utils import shuffle

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, Activation, LSTM, BatchNormalization, TimeDistributed, Conv1D, MaxPooling1D
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical

from tensorflow_addons.metrics import F1Score


def mcor(y_true, y_pred):
    # Matthews correlation
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


def make_roc(true, predicted):

    # roc curve plotting for multiple

    n_classesi = predicted.shape[1]

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classesi):
        fpr[i], tpr[i], _ = roc_curve(true[:, i], predicted[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

    plt.figure(2)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    colors = ['aqua', 'darkorange', 'cornflowerblue',
                    'red', 'black', 'yellow']
    for i in range(n_classesi):
        plt.plot(fpr[i], tpr[i], color=color[i], lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Zooom in View: Some extension of ROC to multi-class')
    plt.legend(loc="lower right")
    plt.show()


def step_decay(epoch):
    # Learning rate scheduler object
    initial_lrate = 0.001
    drop = 0.001
    epochs_drop = 3.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    return lrate


def _load_training_data(base_dir):
    data_path = os.path.join(base_dir, "train_data.csv")
    df = pd.read_csv(
        data_path,
        header=None,
        names=["time", "signal", "idealized"],
        dtype={"time": np.float64, "signal": np.float64, "idealized": np.int64},
    )
    scaler = MinMaxScaler(feature_range=(0, 1))
    signal = scaler.fit_transform(df.signal)
    return signal, df.idealized


def _resample_data(X, y):
    sm = SMOTE(sampling_strategy='auto', random_state=42)
    X_res, Y_res = sm.fit_sample(X, y)
    return X_res, Y_res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    
    # Hyperparameters
    parser.add_argument("--batch-size", type=int, default=25)
    
    # Input data and model directories
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--sm-model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.envrion.get("SM_CHANNEL_TEST"))
    parser.add_argument("--eval", type=str, default=os.environ.get("SM_CHANNEL_EVAL"))
    
    args, unknown_args = parser.parse_known_args()

    X_train, y_train = _load_training_data(args.train)
    max_channels = y_train.max() + 1
    X_train, y_train = _resample_data(X_train, y_train)
    
    yy_res = y_train.reshape(-1, 1)
    yy_res = to_categorical(yy_res, num_classes=max_channels)
    xx_res, yy_res = shuffle(X_res, yy_res)

    trainy_size = int(len(xx_res) * 0.80)
    modder = math.floor(trainy_size/batch_size)
    trainy_size = int(modder*batch_size)
    testy_size = int(len(xx_res) - trainy_size)
    modder = math.floor(testy_size/batch_size)
    testy_size = int(modder*batch_size)

    in_train = xx_res[0:trainy_size, 0]
    in_test = xx_res[trainy_size:trainy_size+testy_size, 0]
    target_train = yy_res[0:trainy_size, :]
    target_test = yy_res[trainy_size:trainy_size+testy_size, :]
    in_train = in_train.reshape(len(in_train), 1, 1, 1)
    in_test = in_test.reshape(len(in_test), 1, 1, 1)


# validation set!!
df_val = pd.read_csv('outfinaltest78.csv', header=None)
data_val = df_val.values.astype('float64')

idataset2 = data_val[:, 2].astype(int)

val_set = data_val[:, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
val_set = scaler.fit_transform(val_set.reshape(-1,1))
val_set = val_set.reshape(len(val_set), 1, 1, 1)
val_target = data_val[:, 2]
val_target = to_categorical(val_target, num_classes=maxchannels+1)


# model starts..

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


newmodel.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=0.001, momentum=0.9, nesterov=False), metrics=[
                 'accuracy', Precision(), Recall(), F1Score(num_classes=maxchannels+1, average='micro')])


lrate = LearningRateScheduler(step_decay)


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


plotlen = test_size
lenny = 2000

plt.figure(figsize=(30, 6))
plt.subplot(2, 1, 1)
# temp=scaler.inverse_transform(dataset)
plt.plot(xx_res[trainy_size:trainy_size+lenny, 0],
         color='blue', label="some raw data")
plt.title("The raw test")

plt.subplot(2, 1, 2)
plt.plot(class_target[:lenny], color='black', label="the actual idealisation", drawstyle='steps-mid')

line, = plt.plot(class_predict[:lenny], color='red',
                 label="predicted idealisation", drawstyle='steps-mid')
plt.setp(line, linestyle='--')
plt.xlabel('timepoint')
plt.ylabel('current')
# plt.savefig(str(rnd)+'data.png')
plt.legend()
plt.show()


# newmodel.save('nmn_oversampled_deepchanel6_5.h5')

make_roc(val_target, predicted_val)
