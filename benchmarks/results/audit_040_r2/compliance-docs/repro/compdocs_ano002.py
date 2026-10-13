"""MADD-ANO-002 re-verification: does the 0.4.0 constructor accept the
unstable timestep the anomaly's own re-verification paragraph describes?"""
import warnings, math
import jax.numpy as jnp
from maddening.nodes.heat import HeatNode

n, L, alpha = 20, 1.0, 0.01
dx = L / n
for Fo in (0.4, 0.5, 5.0):
    dt = Fo * dx**2 / alpha
    print(f"--- Fo={Fo}  dt={dt}")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            node = HeatNode(name="rod", n_cells=n, length=L,
                            thermal_diffusivity=alpha, timestep=dt)
        except Exception as exc:
            print(f"    CONSTRUCTOR RAISED {type(exc).__name__}: {exc}")
            continue
        print(f"    constructed ok; warnings={[str(x.message) for x in w]}")
    st = node.initial_state()
    x = (jnp.arange(n) + 0.5) * dx
    st = dict(st); st["temperature"] = jnp.sin(jnp.pi * x)
    for _ in range(60):
        st = node.update(st, {}, dt)
    print(f"    max|T| after 60 steps = {float(jnp.max(jnp.abs(st['temperature']))):.6g}")
