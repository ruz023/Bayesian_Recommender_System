from typing import *
import numpy as np


def train_test_split(data: np.ndarray, frac_test: float) -> Tuple[np.ndarray, np.ndarray]:
    """Split the data into train/test sets.

    Args:
        data (np.ndarray): (num_users x num_movies) matrix containing ratings
        percent_test (float): mask this fraction of data

    Returns:
        Tuple[np.ndarray, np.ndarray]: train, test matrices of the same shape as data
    """
    assert (0. < frac_test < 1.), "frac_test must be between 0 and 1!"
    n, m = data.shape  # # users, # movies
    N = n * m  # # cells in matrix

    # Prepare train/test ndarrays.
    train = data.copy()
    test = np.ones(data.shape) * np.nan

    # Draw random sample of training data to use for testing.
    tosample = np.where(~np.isnan(train))  # ignore nan values in data
    idx_pairs = list(zip(tosample[0], tosample[1]))  # tuples of row/col index pairs

    test_size = int(len(idx_pairs) * frac_test)  # use (frac_test * 100)% of data as test set
    train_size = len(idx_pairs) - test_size  # and remainder for training

    indices = np.arange(len(idx_pairs))  # indices of index pairs
    sample = np.random.choice(indices, replace=False, size=test_size)

    # Transfer random sample from train set to test set.
    for idx in sample:
        idx_pair = idx_pairs[idx]
        test[idx_pair] = train[idx_pair]  # transfer to test set
        train[idx_pair] = np.nan  # remove from train set

    # Verify everything worked properly
    assert train_size == N - np.isnan(train).sum()
    assert test_size == N - np.isnan(test).sum()

    # Return train set and test set
    return train, test