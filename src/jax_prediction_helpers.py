import jax
import jax.numpy as jnp
import numpy as np


def jax_predict_linear(U: np.ndarray, V: np.ndarray, pmf_model, rng_key: jax.random.PRNGKey) -> jnp.array:
    """ 
        Predict ratings using one instantiation of U and V. To be vectorized by jax.vmap.
        Linear, because U @ V.T directly gives mean rating. Non-linearity example: sigmoid(U @ V.T).
    """
    UV = jnp.matmul(U, V.T)
    R = pmf_model.std * jax.random.normal(key=rng_key, shape=UV.shape) + UV
    return R
