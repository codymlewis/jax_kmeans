from typing import Callable, Dict, Tuple
import jax
import jax.numpy as jnp
import chex


def naive_init(samples: chex.Array, k: int = 8, seed: int = 0) -> Dict[str, chex.Array]:
    centroids_idx = jax.random.choice(jax.random.PRNGKey(seed), jnp.arange(samples.shape[0]), shape=(k,), replace=False)
    return {"centroids": samples[centroids_idx]}


def plusplus_init(samples: chex.Array, k: int = 8, seed: int = 0) -> Dict[str, chex.Array]:
    "K-Means++ initialisation algorithm from https://dl.acm.org/doi/10.5555/1283383.1283494"
    rngkeys = jax.random.split(jax.random.PRNGKey(seed), k)
    num_samples = samples.shape[0]
    centroid_idx = jax.random.choice(rngkeys[0], jnp.arange(num_samples))
    centroids = jnp.concatenate(
        (jnp.expand_dims(samples[centroid_idx], axis=0), jnp.full((k,) + samples.shape[1:], jnp.inf))
    )

    def find_centroid(centroids, i):
        dists = jax.vmap(lambda s: jax.vmap(lambda c: jnp.linalg.norm(c - s))(centroids))(samples)
        weights = jnp.min(dists, axis=1)
        centroid_idx = jax.random.choice(rngkeys[i], jnp.arange(num_samples), p=weights**2)
        centroid = samples[centroid_idx]
        centroids = jax.lax.dynamic_update_slice_in_dim(centroids, jnp.expand_dims(centroid, axis=0), i, axis=1)
        return centroids, centroid

    _, centroids = jax.lax.scan(find_centroid, centroids, jnp.arange(1, k))

    return {"centroids": centroids}


def lloyds(
        params: Dict[str, chex.Array], samples: chex.Array, num_iterations: int = 300
) -> Tuple[chex.Array, Dict[str, chex.Array]]:
    "Lloyd's algorithm for cluster finding from https://ieeexplore.ieee.org/document/1056489"
    def iteration(centroids, i):
        dists = jax.vmap(lambda s: jax.vmap(lambda c: jnp.linalg.norm(c - s))(centroids))(samples)
        clusters = jnp.argmin(dists, axis=1)

        def find_centroids(unused, cluster):
            cluster_vals = jnp.where(
                jnp.tile(clusters == cluster, samples.shape[-1:0:-1] + (1,)).T,
                samples,
                jnp.zeros_like(samples, dtype=samples.dtype),
            )
            cluster_size = jnp.maximum(1, jnp.sum(clusters == cluster))
            return unused, jnp.sum(cluster_vals, axis=0) / cluster_size

        _, new_centroids = jax.lax.scan(find_centroids, None, jnp.arange(centroids.shape[0]))
        return new_centroids, jnp.linalg.norm(new_centroids - centroids)

    centroids, losses = jax.lax.scan(iteration, params['centroids'], jnp.arange(num_iterations))
    return losses, {"centroids": centroids}


def fit(
    samples: chex.Array,
    k: int = 8,
    num_iterations: int = 300,
    init_fn: Callable[[chex.Array, int, int], Dict[str, chex.Array]] = plusplus_init,
    seed: int = 0
) -> Dict[str, chex.Array]:
    params = init_fn(samples, k, seed)
    losses, params = lloyds(params, samples, num_iterations)
    return params


def transform(params: Dict[str, chex.Array], samples: chex.Array) -> chex.Array:
    dists = jax.vmap(lambda s: jax.vmap(lambda c: jnp.linalg.norm(c - s))(params["centroids"]))(samples)
    return jnp.argmin(dists, axis=1)


def fit_transform(samples: chex.Array, k: int = 8, num_iterations: int = 300, seed: int = 0) -> chex.Array:
    return transform(fit(samples, k, num_iterations, seed=seed), samples)
