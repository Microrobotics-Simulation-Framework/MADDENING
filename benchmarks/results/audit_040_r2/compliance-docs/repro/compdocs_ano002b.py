import warnings
import jax.numpy as jnp
from maddening.nodes.heat import HeatNode
n, L, alpha = 20, 1.0, 0.01
dx = L / n
x = (jnp.arange(n) + 0.5) * dx
def run(Fo, bcs):
    dt = Fo * dx**2 / alpha
    node = HeatNode(name="rod", n_cells=n, length=L, thermal_diffusivity=alpha,
                    timestep=min(dt, 0.5*dx**2/alpha))
    st = dict(node.initial_state()); st["temperature"] = jnp.sin(jnp.pi * x)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        for _ in range(60):
            st = node.update(st, bcs, dt)          # dt passed to update()
        nw = len(w)
    return float(jnp.max(jnp.abs(st["temperature"]))), nw
for Fo in (0.4, 0.5, 5.0):
    for label, bcs in (("no bc", {}),
                       ("dirichlet 0", {"left_temperature": jnp.array(0.0),
                                        "right_temperature": jnp.array(0.0)})):
        v, nw = run(Fo, bcs)
        print(f"Fo={Fo:<5} {label:<12} max|T| after 60 update() steps = {v:<12.6g} warnings={nw}")
