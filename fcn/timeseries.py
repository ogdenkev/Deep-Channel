import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from sklearn.preprocessing import MinMaxScaler, RobustScaler

multi_state_categories = CategoricalDtype(
    np.array([0, 2, 4.7, 6], dtype=np.float64)
)


class TimeSeries:
    """
    Create a window of features from a time series

    Parameters
    ----------
    data_paths: array-like
        A list of paths from which to read the data
    max_sequence_length: int, default=128
        The maximum sequence length of the features. When all data has been
        exhausted from a given file, then the length of the returned feature
        window may be less than this number.
    batch_size: int, default=64
        The size of the batches that will be returned upon calling the
        instance's `next` method.
    undersample: bool, default=False
        Whether to undersample the different classes of single channel states.
    scaler: bool or sklearn Transformer
        Control how the data are scaled or transformed. If True, the default,
        then a RobostScaler will be used with `quantile_range=(0.5, 99.5)`.
    random_state: int, default=None
    shuffle: bool, default=False

    Examples
    --------
    >>> data_gen_train = TimeSeries(DATA_PATH, random_state=88)
    >>> my_train_data = tf.data.Dataset.from_generator(
    ...     data_gen_train,
    ...     (tf.float32, tf.float32),
    ...     (
    ...         tf.TensorShape([BATCH_SIZE, MAX_SEQUENCE_LENGTH, 1]),
    ...         tf.TensorShape([BATCH_SIZE])
    ...     )
    ... )
    >>> history = model.fit(
    ...     my_train_data,
    ...     epochs=NUM_EPOCHS,
    ...     steps_per_epoch=None,
    ... )
    """

    def __init__(
        self,
        data_paths,
        max_sequence_length=128,
        batch_size=64,
        undersample=False,
        scaler=True,
        random_state=None,
        shuffle=False,
    ):
        self.data_paths = data_paths
        self.max_sequence_length = max_sequence_length
        self.padding = max_sequence_length // 2
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.scaler = scaler
        self.data = None
        self.target = None
        self.target_categories = None
        self.random_state = random_state
        self.undersample = undersample

    def __iter__(self):
        self.index = 0
        self.file_index = 0
        self.load_data(self.file_index)
        self.rng = np.random.default_rng(seed=self.random_state)
        return self

    def __next__(self):
        if self.index >= self.data.shape[0]:
            self.file_index += 1
            if self.file_index >= len(self.data_paths):
                raise StopIteration
            self.load_data(self.file_index)

        batch_index = 0
        batch_data = np.zeros(
            (self.batch_size, self.max_sequence_length, self.data.shape[1])
        )
        batch_target = np.zeros((self.batch_size,))
        while batch_index < self.batch_size:
            if self.index >= self.data.shape[0]:
                self.file_index += 1
                if self.file_index >= len(self.data_paths):
                    break
                self.load_data(self.file_index)

            x, y = self._getwindow(self.index)
            self.index += 1

            # Random undersampling
            if self.undersample and self.rng.random() > self.keep_probs[y]:
                continue

            batch_data[batch_index, :, :], batch_target[batch_index] = x, y
            batch_index += 1

        return batch_data, batch_target

    def _getwindow(self, index):
        if self.data is None:
            raise KeyError("Data has not been loaded yet")

        out = np.zeros((self.max_sequence_length, self.data.shape[1]))

        start = index - self.padding
        end = start + self.max_sequence_length
        out_start = max(0, -start)
        start = max(0, start)

        out_end = self.max_sequence_length - max(0, end - self.data.shape[0])
        end = min(self.data.shape[0], end)

        out[out_start:out_end, :] = self.data[start:end, :]

        return out, self.target[index]

    def load_data(self, file_index):
        raw_data = pd.read_csv(
            self.data_paths[file_index],
            header=None,
            names=["time", "current", "state"],
            usecols=["current", "state"],
            dtype={"current": np.float64, "state": np.float64},
        )
        self.data = self.scale_data(raw_data[["current"]].to_numpy())
        if self.target_categories is None:
            states = raw_data["state"].astype("category")
            self.target_categories = states.dtype
        else:
            states = raw_data["state"].astype(self.target_categories)
        self.target = states.cat.codes.to_numpy()
        self.index = 0

        # Set up for random undersampling
        self.target_unique_values, self.target_counts = np.unique(
            self.target, return_counts=True
        )
        self.minority_class = self.target_unique_values[
            self.target_counts.argmin()
        ]
        self.keep_probs = {
            v: self.target_counts.min() / c
            for v, c in zip(self.target_unique_values, self.target_counts)
        }
        return None

    def scale_data(self, data, reset_scaler=True):
        """
        Scale or transform data

        Parameters
        ----------
        data: array-like
            The data to scale
        reset_scaler: bool, default=True
            If True, the default, then re-fit the scaler to data.
            If False, then use the current scaler that has already been fit.

        Returns
        -------
        scaled_data: array-like
            The scaled or transformed data
        """

        if not self.scaler:
            return data
        if not hasattr(self.scaler, "fit"):
            self.scaler = RobustScaler(
                with_centering=False, quantile_range=(0.5, 99.5)
            )
            self.scaler.fit(data)
        elif reset_scaler:
            self.scaler = self.scaler.fit(data)

        return self.scaler.transform(data)

    def __call__(self):
        for batch in self:
            yield batch
