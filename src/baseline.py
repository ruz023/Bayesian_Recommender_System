import numpy as np
from src.eval import rmse


def split_title(title):
    """Change "BaselineMethod" to "Baseline Method"."""
    words = []
    tmp = [title[0]]
    for c in title[1:]:
        if c.isupper():
            words.append("".join(tmp))
            tmp = [c]
        else:
            tmp.append(c)
    words.append("".join(tmp))
    return " ".join(words)


class Baseline:
    """Calculate baseline predictions."""

    def __init__(self, train_data: np.ndarray):
        """Simple heuristic-based transductive learning to fill in missing
        values in data matrix."""
        self.predict(train_data.copy())

    def predict(self, train_data: np.ndarray):
        raise NotImplementedError("baseline prediction not implemented for base class")

    def rmse(self, test_data: np.ndarray):
        """Calculate root mean squared error for predictions on test data."""
        return rmse(test_data, self.predicted)

    def __str__(self):
        return split_title(self.__class__.__name__)


class UniformRandomBaseline(Baseline):
    """Fill missing values with uniform random values."""
    def predict(self, train_data: np.ndarray):
        nan_mask = np.isnan(train_data)
        masked_train = np.ma.masked_array(train_data, nan_mask)
        pmin, pmax = masked_train.min(), masked_train.max()
        N = nan_mask.sum()
        train_data[nan_mask] = np.random.uniform(pmin, pmax, N)
        self.predicted = train_data


class GlobalMeanBaseline(Baseline):
    """Fill in missing values using the global mean."""
    def predict(self, train_data: np.ndarray):
        nan_mask = np.isnan(train_data)
        train_data[nan_mask] = train_data[~nan_mask].mean()
        self.predicted = train_data


class MeanOfMeansBaseline(Baseline):
    """Fill in missing values using mean of user/item/global means."""
    def predict(self, train_data: np.ndarray):
        # Define array for storing predictions
        predicted = np.zeros_like(train_data)

        # nan_mask to locate missing values in train
        # not_reviewed_movie_mask to locate movies with no ratings at all
        nan_mask = np.isnan(train_data)
        not_reviewed_movie_mask = (nan_mask.sum(axis=0) == train_data.shape[0]) 

        # Calculate global/user/item means for imputation
        global_mean = np.nanmean(train_data)
        user_means = np.nanmean(train_data, axis=1)
        item_means = np.nanmean(train_data, axis=0)

        # Imputation
        predicted[:, :] = global_mean
        predicted += user_means.reshape(-1, 1)
        predicted[:, ~not_reviewed_movie_mask] += item_means[~not_reviewed_movie_mask].reshape(1, -1) 
        predicted[:, ~not_reviewed_movie_mask] /= 3.  # average of 3 quantities (global, user, item)
        predicted[:, not_reviewed_movie_mask] /= 2.  # average of 2 quantities (global, user) 

        # Put back non-missing values
        predicted[~nan_mask] = train_data[~nan_mask]
        self.predicted = predicted