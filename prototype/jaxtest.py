import jax
import jax.numpy as jnp

print("JAX devices:", jax.devices())
x = jnp.array([1.0, 2.0, 3.0])
print("Array on device:", x.device)
