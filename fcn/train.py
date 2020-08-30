"""
Train a Fully Convolutional Network to Detect Single Molecule Events

The Fully Convolutional Network (FCN) is adapted from https://arxiv.org/pdf/1611.06455.pdf

This script is compatible with the SageMaker Python SDK using the TensorFlow Estimator.

Examples
--------
>>> import sagemaker
>>> from sagemaker import get_execution_role
>>> from sagemaker.tensorflow import TensorFlow
>>> sagemaker_session = sagemaker.Session()
>>> role = get_execution_role()
>>> region = sagemaker_session.boto_session.region_name
>>> MAX_RUN_TIME_SECONDS = 4 * 3600
>>> # ml.p2.xlarge = $1.26 per hour
>>> MODEL_HPS = {
...     "num-classes": 2,
...     "epochs": 20,
...     "batch-size": 128,
...     "max-sequence-length": 33,
...     "scaler": True,
...     "undersample": False,
...     "keras-verbosity": 2
... }
>>> tf_estimator = TensorFlow(
...     entry_point="train.py",
...     source_dir="fcn",
...     hyperparameters=MODEL_HPS,
...     output_path="s3://deepchannel-sagemaker",
...     base_job_name="fcn-gibb",
...     role=role,
...     instance_count=1,
...     instance_type="ml.p2.xlarge",
...     max_run=MAX_RUN_TIME_SECONDS,
...     framework_version="2.2",
...     py_version="py37",
...     container_log_level=logging.DEBUG
... )
>>> tf_estimator.fit(
...     inputs={
...         "train": "s3://deepchannel-sagemaker/data/train",
...         "test": "s3://deepchannel-sagemaker/data/test"
...     },
...     wait=True
... )
"""

import pathlib
import datetime
import argparse
import os
# import json
import pickle

import numpy as np
import pandas as pd

import timeseries

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Reshape,
    Activation,
    LSTM,
    BatchNormalization,
    TimeDistributed,
    Conv1D,
    MaxPooling1D,
    Input,
    GlobalAveragePooling1D,
)
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import (
    Precision,
    Recall,
    SparseCategoricalAccuracy,
)
from tensorflow.keras import optimizers

BATCH_SIZE = 64
MAX_SEQUENCE_LENGTH = 33
NUM_CLASSES = 2
NUM_EPOCHS = 10

TRAIN_DIR = pathlib.Path(os.environ.get("SM_CHANNEL_TRAIN", os.getcwd()))
TEST_DIR = pathlib.Path(os.environ.get("SM_CHANNEL_TEST", os.getcwd()))
OUTPUT_DIR = pathlib.Path(os.environ.get("SM_MODEL_DIR", os.getcwd()))
ARTIFACT_DIR = pathlib.Path(os.environ.get("SM_OUTPUT_DATA_DIR", os.getcwd()))
# TRAINING_ENV = json.loads(os.environ.get("SM_TRAINING_ENV", "{}"))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train a deep learning network to detect events in single molecule data"
    )

    # Model and training output
    parser.add_argument(
        "--out",
        "-o",
        metavar="OUT",
        help=("Path to save the trained model. "
              f"By default the model will be saved to {OUTPUT_DIR}/deepchannel-model-<TIMESTAMP>.h5"),
    )
    parser.add_argument(
        "--scaler-out",
        help=("Path to which the trained MinMaxScaler will be saved. "
              f"By default, the scaler will be saved to {ARTIFACT_DIR}/deepchannel-scaler-<TIMESTAMP>.pkl"),
    )
    parser.add_argument(
        "--history-out",
        help=("Path to save the training history. "
              f"By default, the history will be saved to {ARTIFACT_DIR}/deepchannel-training-history-<TIMESTAMP>.pkl"),
    )

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument(
        "--num-classes",
        type=int,
        default=NUM_CLASSES,
        help="Number of output classes from the model [default: %(default)s]"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=NUM_EPOCHS,
        help="Number of training epochs [default: %(default)s]",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size [default: %(default)s]",
    )
    parser.add_argument(
        "--max-sequence-length",
        type=int,
        default=MAX_SEQUENCE_LENGTH,
        help="Sequence feature window length [default: %(default)s]",
    )
    parser.add_argument(
        "--scaler",
        type=str,
        default="true",
        help=("Whether or not to scale the data. "
              "If True then a ``sklearn.preprocessing.RobostScaler`` will be used "
              "with ``quantile_range=(0.5, 99.5)`` [default: %(default)s]")
    )
    parser.add_argument(
        "--undersample",
        type=str,
        default="false",
        help=("Whether to undersample the different classes of single channel states. [default: %(default)s]")
    )
    parser.add_argument(
        "--keras-verbosity",
        type=int,
        default=2,
        help="Verbosity mode for ``tf.keras.Model.fit``. See https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit"
    )

    # input data and model directories
    parser.add_argument(
        "--model_dir",
        type=str,
        help=("Location where the checkpoint data and models can be exported to during training "
              "The location is based on your training configuration. "
              "See https://sagemaker.readthedocs.io/en/stable/frameworks/tensorflow/sagemaker.tensorflow.html")
    )
    parser.add_argument(
        "--train",
        default=TRAIN_DIR,
        type=pathlib.Path,
        help="Path to the directory that contains the input data for training [default: %(default)s].",
    )
    parser.add_argument(
        "--test",
        default=TEST_DIR,
        type=pathlib.Path,
        help="Path to the directory that contains the input data for testing [default: %(default)s].",
    )

    args, _ = parser.parse_known_args()

    # Log GPU info
    gpu_devs = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available:", len(gpu_devs))
    print("GPU Devices:", gpu_devs)
    
    # Training data
    training_paths = sorted(args.train.glob("*.csv"))
    print("Training files:", *[str(p) for p in training_paths], sep="\n")
    
    print(f"""Data Processing Info:
    Max Sequence Length: {args.max_sequence_length}
    Batch Size: {args.batch_size}
    Undersample: {args.undersample}
    Scaler: {args.scaler}""")
    
    if args.scaler.lower() == "true":
        args.scaler = True
    elif args.scaler.lower() == "false":
        args.scaler = False
    else:
        raise ValueError(f"The `scaler` argument must be either 'true' or 'false', but was {args.scaler}")
    
    if args.undersample.lower() == "true":
        args.undersample = True
    elif args.undersample.lower() == "false":
        args.undersample = False
    else:
        raise ValueError(f"The `undersample` argument must be either 'true' or 'false', not '{args.undersample}'")

    data_gen_train = timeseries.TimeSeries(
        training_paths,
        max_sequence_length=args.max_sequence_length,
        batch_size=args.batch_size,
        undersample=args.undersample,
        scaler=args.scaler,
        random_state=88,
    )

    # Test data
    testing_paths = sorted(args.test.glob("*.csv"))
    print("Testing files:", *[str(p) for p in testing_paths], sep="\n")

    data_gen_test = timeseries.TimeSeries(
        testing_paths,
        max_sequence_length=args.max_sequence_length,
        batch_size=args.batch_size,
        undersample=False,
        scaler=args.scaler,
        random_state=8675309,
    )

    # ## Build the Model

    # Fully Convolutional Network (FCN) from https://arxiv.org/pdf/1611.06455.pdf

    inputs = Input(shape=(args.max_sequence_length, 1))

    filters = (8, 5, 3)
    # filters = (16, 10, 6)

    x = Conv1D(
        128, filters[0], padding="same", kernel_initializer="he_uniform"
    )(inputs)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    x = Conv1D(
        256, filters[1], padding="same", kernel_initializer="he_uniform"
    )(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    x = Conv1D(
        128, filters[2], padding="same", kernel_initializer="he_uniform"
    )(x)
    x = Activation("relu")(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling1D()(x)

    output = Dense(args.num_classes, activation="softmax")(x)

    model = Model(inputs, output)

    model.summary()

    model.compile(
        optimizer=optimizers.Adam(
            learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08
        ),
        loss=SparseCategoricalCrossentropy(),
        metrics=["sparse_categorical_accuracy"],
    )

    train_data = tf.data.Dataset.from_generator(
        data_gen_train,
        (tf.float32, tf.float32),
        (
            tf.TensorShape([args.batch_size, args.max_sequence_length, 1]),
            tf.TensorShape([args.batch_size]),
        ),
    )
    test_data = tf.data.Dataset.from_generator(
        data_gen_test,
        (tf.float32, tf.float32),
        (
            tf.TensorShape([args.batch_size, args.max_sequence_length, 1]),
            tf.TensorShape([args.batch_size]),
        ),
    )

    # ## Fit the Model

    print("Fitting the model ...")
    history = model.fit(
        train_data,
        epochs=args.epochs,
        steps_per_epoch=None,
        validation_data=test_data,
        verbose=args.keras_verbosity
    )

    # ## Save the results

    TIMESTAMP = datetime.datetime.now(datetime.timezone.utc)

    # Save trained model
    if args.out:
        MODEL_PATH = args.out
    else:
        MODEL_PATH = OUTPUT_DIR / TIMESTAMP.strftime(
            "fcn-v20200829-%Y%m%dT%H%M%SZ.h5"
        )

    print(f"Saving model to {MODEL_PATH}")
    model.save(MODEL_PATH)

    # Save training history
    if args.history_out:
        HISTORY_PATH = args.history_out
    else:
        HISTORY_PATH = ARTIFACT_DIR / TIMESTAMP.strftime(
            "deepchannel-training-history-%Y%m%dT%H%M%SZ.pkl"
        )

    print(f"Saving training history to {HISTORY_PATH}")
    with open(HISTORY_PATH, "wb") as fd:
        pickle.dump(history.history, fd)

    if data_gen_train.scaler:
        if args.scaler_out:
            SCALER_PATH = args.scaler_out
        else:
            SCALER_PATH = ARTIFACT_DIR / TIMESTAMP.strftime(
                "deepchannel-scaler-%Y%m%dT%H%M%SZ.pkl"
            )

        print(f"Saving scaler to {SCALER_PATH}")
        with open(SCALER_PATH, "wb") as fd:
            pickle.dump(data_gen_train.scaler, fd)
    elif args.scaler_out:
        print("No scaler was used during training, so I cannot save one")
