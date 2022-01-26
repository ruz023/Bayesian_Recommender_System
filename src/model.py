import logging
from multiprocessing import Pool
from typing import *
import time

import numpy as np
import pymc3 as pm
import theano

# Enable on-the-fly graph computations, but ignore
# absence of intermediate test values.
theano.config.compute_test_value = "ignore"


class PMF:
    """Probabilistic Matrix Factorization model using pymc3."""

    def __init__(self, train: np.ndarray, dim: int, alpha: float = 2., std: float = 0.01, bounds: Tuple[int, int] = (1, 5)):
        """Build the Probabilistic Matrix Factorization model using pymc3.

        :param train: The training data to use for learning the model.
        :param dim: Dimensionality of the model; number of latent factors.
        :param alpha: Fixed precision for the likelihood function.
        :param std: Amount of noise to use for model initialization.
        :param bounds: (lower, upper) bound of ratings.
            These bounds will simply be used to cap the estimates produced for R.

        """
        self.dim = dim
        self.alpha = alpha
        self.std = np.sqrt(1.0 / alpha)
        self.bounds = bounds
        self.data = train.copy()
        n, m = self.data.shape

        # Perform mean value imputation
        nan_mask = np.isnan(self.data)
        self.data[nan_mask] = np.mean(self.data[~nan_mask])

        # Low precision reflects uncertainty; prevents overfitting.
        # Set to the mean variance across users and items.
        self.alpha_u = 1 / self.data.var(axis=1).mean()
        self.alpha_v = 1 / self.data.var(axis=0).mean()

        # Specify the model.
        logging.info("building the PMF model")
        with pm.Model() as pmf:
            U = pm.MvNormal(
                "U",
                mu=0,
                tau=self.alpha_u * np.eye(dim),
                shape=(n, dim),
                testval=np.random.randn(n, dim) * std,
            )
            V = pm.MvNormal(
                "V",
                mu=0,
                tau=self.alpha_v * np.eye(dim),
                shape=(m, dim),
                testval=np.random.randn(m, dim) * std,
            )
            R = pm.Normal(
                "R", mu=(U @ V.T)[~nan_mask], tau=self.alpha, observed=self.data[~nan_mask]
            )
        logging.info("done building the PMF model")
        self.model = pmf
    
    @property
    def map(self):
        """ Return MAP estimate if already computed. Otherwise, compute it! """
        try:
            return self._map
        except:
            return self.find_map()

    def find_map(self):
        """Find mode of posterior using L-BFGS-B optimization."""
        tstart = time.time()
        with self.model:
            logging.info("finding PMF MAP using L-BFGS-B optimization...")
            self._map = pm.find_MAP(method="L-BFGS-B")
        elapsed = int(time.time() - tstart)
        logging.info("found PMF MAP in %d seconds" % elapsed)
        return self._map

    def draw_samples(self, **kwargs):
        """ Draw MCMC samples """
        kwargs.setdefault("chains", 1)
        with self.model:
            self.trace = pm.sample(**kwargs)

    def predict(self, U: np.ndarray, V: np.ndarray) -> np.ndarray:
        """ Estimate R from the given values of U and V """
        R = np.dot(U, V.T)
        sample_R = np.random.normal(R, self.std)
        # bound ratings
        low, high = self.bounds
        sample_R[sample_R < low] = low
        sample_R[sample_R > high] = high
        return sample_R

    def __str__(self):
        return self.name
