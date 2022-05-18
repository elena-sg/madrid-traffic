import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


def expand_result(result, n_samples, seq_len):
    r1 = np.repeat(result, seq_len, axis=0)
    r2 = np.repeat([r1], n_samples, axis=0)
    return np.expand_dims(r2, axis=3)


class MeanRegressor(BaseEstimator):
    def __init__(self):
        self.mean_ = None

    def fit(self, X, y):
        y = y[:, :, :, [0]]
        self.mean_ = np.mean(y, axis=(0, 1)).T

        # Return the classifier
        return self

    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)

        return expand_result(self.mean_, X.shape[0], X.shape[1])


class MedianRegressor(BaseEstimator):
    def __init__(self):
        self.median_ = None

    def fit(self, X, y):
        y = y[:, :, :, [0]]
        self.median_ = np.median(y, axis=(0, 1)).T

        # Return the classifier
        return self

    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)

        return expand_result(self.median_, X.shape[0], X.shape[1])


class RepeatRegressor(BaseEstimator):
    def __init__(self):
        self.fitted_ = None

    def fit(self, X, y):
        self.fitted_ = True
        return self

    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)

        return X[..., [0]]


class RepeatLastRegressor(BaseEstimator):
    def __init__(self):
        self.fitted_ = None

    def fit(self, X, y):
        self.fitted_ = True
        return self

    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)

        seq_len = X.shape[1]

        last_values = np.repeat([X[..., -1, :, 0]], seq_len, axis=0)
        last_values_moved = np.moveaxis(last_values, [0, 1, 2], [1, 0, 2])

        return np.expand_dims(last_values_moved, axis=3)


class DaytimeRegressor(BaseEstimator):
    def __init__(self, agg="mean", by_working_day=False):
        self.seasonal_values_ = None
        self.agg = agg
        self.by_working_day = by_working_day

    def fit(self, X, y):
        new_x = np.concatenate([y[..., [0]], X[..., 1:]], axis=3)
        x_train_by_sensor = new_x.reshape(3, -1, 2+1*self.by_working_day)
        if not self.by_working_day:
            df = pd.DataFrame([], index=np.linspace(0, 23.75, 24 * 4), columns=range(X.shape[2]))
            for sensor in range(X.shape[2]):
                df[sensor] = pd.DataFrame(x_train_by_sensor[sensor]).groupby(1).agg(self.agg)

        else:
            index = pd.MultiIndex.from_product([[0, 1], np.linspace(0, 23.75, 24 * 4)],
                                               names=["working_day", "hour"])
            df = pd.DataFrame([], index=index, columns=range(X.shape[2]))
            for sensor in range(X.shape[2]):
                df[sensor] = pd.DataFrame(x_train_by_sensor[sensor]).groupby([2, 1]).agg(self.agg)
        self.seasonal_values_ = df.fillna(method="ffill").fillna(method="bfill")
        return self

    def predict(self, X):
        # Check if fit has been called
        check_is_fitted(self)

        prediction = np.full_like(X, fill_value=np.nan)[:, :, :, [0]]
        if not self.by_working_day:
            for t1 in range(X.shape[0]):
                prediction[t1, :, :, 0] = self.seasonal_values_.loc[X[t1, :, 0, 1]]
        else:
            for t1 in range(X.shape[0]):
                working_day = X[t1, 0, 0, 2]
                prediction[t1, :, :, 0] = self.seasonal_values_.loc[working_day].loc[X[t1, :, 0, 1]]
        return prediction


class DriftRegressor(BaseEstimator):
    def __int__(self):
        self.fitted_ = None

    def fit(self, X, y):
        self.fitted_ = True
        return self

    def predict(self, X):
        prediction = np.full_like(X, fill_value=np.nan)[:, :, :, [0]]
        increment = (X[:, -1, :, 0] - X[:, 0, :, 0]) / X.shape[1]
        for i in range(X.shape[1]):
            prediction[:, i, :, 0] = X[:, -1, :, 0] + (i+1) * increment

        prediction[prediction < 0] = 0

        return prediction
