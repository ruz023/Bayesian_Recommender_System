from typing import *

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Define our evaluation function.
def rmse(test_data: np.ndarray, predicted: np.ndarray) -> float:
    """Calculate root mean squared error.
    Ignoring missing values in the test data.
    """
    I = ~np.isnan(test_data)  # indicator for missing values
    N = I.sum()  # number of non-missing values
    sqerror = abs(test_data - predicted) ** 2  # squared error array
    mse = sqerror[I].sum() / N  # mean squared error
    return np.sqrt(mse)  # RMSE


def eval_map(pmf_model, train: np.ndarray, test: np.ndarray) -> float:
    U = pmf_model.map["U"]
    V = pmf_model.map["V"]

    # Make predictions and calculate RMSE on train and test sets.
    predictions = pmf_model.predict(U, V)
    train_rmse, test_rmse = list(map(rmse, (train, test), (predictions, predictions)))
    overfit = test_rmse - train_rmse

    # Print report.
    print("PMF MAP training RMSE: %.5f" % train_rmse)
    print("PMF MAP testing RMSE:  %.5f" % test_rmse)
    print("Train/test difference: %.5f" % overfit)
    return test_rmse


def norm_traceplot(pmf_model):
    """ Plot Frobenius norms of U and V as a function of sample #. """
    def _norms(pmf_model, monitor: str, ord="fro"):
        for sample in pmf_model.trace:
            yield np.linalg.norm(sample[monitor], ord)
    
    trace_norms = {}
    trace_norms["U"], trace_norms["V"] = list(map(_norms, (pmf_model, pmf_model), ("U", "V")))
 
    u_series = pd.Series(trace_norms["U"])
    v_series = pd.Series(trace_norms["V"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    u_series.plot(kind="line", ax=ax1, grid=False, title=r"$\|U\|_{Fro}^2$ at Each Sample")
    v_series.plot(kind="line", ax=ax2, grid=False, title=r"$\|V\|_{Fro}^2$ at Each Sample")
    ax1.set_xlabel("Sample Number")
    ax2.set_xlabel("Sample Number")


def running_rmse(pmf_model, train_data: np.ndarray, test_data: np.ndarray, burn_in: int = 0, plot: bool = True):
    """Calculate RMSE for each step of the trace to monitor convergence."""
    burn_in = burn_in if len(pmf_model.trace) >= burn_in else 0
    results = {"per-step-train": [], "running-train": [], "per-step-test": [], "running-test": []}
    R = np.zeros(test_data.shape)
    for cnt, sample in enumerate(pmf_model.trace[burn_in:]):
        sample_R = pmf_model.predict(sample["U"], sample["V"])
        R += sample_R
        running_R = R / (cnt + 1)
        results["per-step-train"].append(rmse(train_data, sample_R))
        results["running-train"].append(rmse(train_data, running_R))
        results["per-step-test"].append(rmse(test_data, sample_R))
        results["running-test"].append(rmse(test_data, running_R))

    results = pd.DataFrame(results)

    if plot:
        results.plot(
            kind="line",
            grid=False,
            figsize=(15, 7),
            title="Per-step and Running RMSE From Posterior Predictive",
        )

    # Return the final predictions, and the RMSE calculations
    return running_R, results


def jax_predict_linear(U: np.ndarray, V: np.ndarray, pmf_model, rng_key: jax.random.PRNGKey) -> jnp.array:
    """ 
        Predict ratings using one instantiation of U and V. To be vectorized by jax.vmap.
        Linear, because U @ V.T directly gives mean rating. Non-linearity example: sigmoid(U @ V.T).
    """
    UV = jnp.matmul(U, V.T)
    R = pmf_model.std * jax.random.normal(key=rng_key, shape=UV.shape) + UV
    return R


def compare_frequent_infrequent_users(
    pmf_model,
    train_data: np.ndarray, 
    test_data: np.ndarray,
    burn_in: int = 0, 
    user_frac: float = 0.1,
    jax_predict_fn: Callable = None, 
    jax_rng_key: jax.random.PRNGKey = jax.random.PRNGKey(42),
) -> Dict[str, np.ndarray]:
    """ Compare RMSE for the most and least frequent (user_frac * 100%) of reviewers

    Args:
        pmf_model ([type]): fitted PMF model.
        train_data (np.ndarray): (num_users x num_movies).
        test_data (np.ndarray): (num_users x num_movies), train with randomly-masked elements.
        burnin (int, optional): Ignore this number of posterior samples. Defaults to 0.
        user_frac (float, optional): consider the top and bottom this fraction of users.
        jax_predict_fn (Callable, optional). Accelerate prediction with external function + JAX.
    """

    assert (0 < user_frac < 0.5)
    burn_in = burn_in if len(pmf_model.trace) >= burn_in else 0

    # Use MCMC samples to predict ratings
    if jax_predict_fn:
        print("JAX prediction")
        jax_rng_subkeys = jax.random.split(jax_rng_key, len(pmf_model.trace)-burn_in)
        pred_R = jax_predict_fn(
            pmf_model.trace["U"][burn_in:].astype(np.float32),
            pmf_model.trace["V"][burn_in:].astype(np.float32),
            pmf_model,
            jax_rng_subkeys
        )
        pred_R = jnp.mean(pred_R, axis=0).block_until_ready()
    else:
        print("Sequential prediction")
        # Tried accelerating with multiprocessing.Pool, but overhead overwhelms benefits
        pred_R = map(
            pmf_model.predict, 
            pmf_model.trace["U"][burn_in:].astype(np.float32), 
            pmf_model.trace["V"][burn_in:].astype(np.float32)
        )
        pred_R = np.mean(tuple(pred_R), axis=0)

    # Use MAP to predict ratings
    map_R = pmf_model.predict(pmf_model.map["U"], pmf_model.map["V"])

    # Identify users who gave the most and least amounts of ratings
    num_ratings_per_user = (~np.isnan(train_data)).sum(axis=1)
    low_thres, high_thres = np.quantile(
        num_ratings_per_user, 
        [user_frac, 1-user_frac], 
        interpolation="nearest"
    )
    low_thres_mask = (num_ratings_per_user <= low_thres)
    high_thres_mask = (num_ratings_per_user >= high_thres)

    # RMSE for EACH USER
    def per_user_rmse(Ra: np.ndarray, Rb: np.ndarray, user_mask: np.ndarray) -> np.ndarray:
        return np.sqrt(np.nanmean(((Ra - Rb)**2)[user_mask], axis=1))

    return {
        "mcmc_rmse_frequent_train": per_user_rmse(train_data, pred_R, high_thres_mask), 
        "mcmc_rmse_frequent_test": per_user_rmse(test_data, pred_R, high_thres_mask),
        "mcmc_rmse_infrequent_train": per_user_rmse(train_data, pred_R, low_thres_mask),
        "mcmc_rmse_infrequent_test": per_user_rmse(test_data, pred_R, low_thres_mask),
        "map_rmse_frequent_train": per_user_rmse(train_data, map_R, high_thres_mask),
        "map_rmse_frequent_test": per_user_rmse(test_data, map_R, high_thres_mask),
        "map_rmse_infrequent_train": per_user_rmse(train_data, map_R, low_thres_mask),
        "map_rmse_infrequent_test": per_user_rmse(test_data, map_R, low_thres_mask)
    }
