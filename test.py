import jax
import matplotlib.pyplot as plt

import clax


if __name__ == "__main__":
    seed = 42
    samples = jax.random.normal(jax.random.PRNGKey(seed), shape=(100, 2))
    clusters = clax.kmeans.fit_transform(samples, seed=seed)
    plt.scatter(samples[:, 0], samples[:, 1], c=clusters)
    plt.show()
