"""MADD-ANO-009's description says, at 0.4.0.dev0: stencil_order=4, 20 cells,
alpha=1.0, L=1.0 -- "Fo = 0.37 converges normally, Fo = 0.40 ... diverges"."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from maddening.nodes.heat import HeatNode, MAX_FOURIER_NUMBER
print("MAX_FOURIER_NUMBER =", MAX_FOURIER_NUMBER)
n, L, alpha = 20, 1.0, 1.0
dx = L / n
for Fo in (0.37, 0.40):
    dt = Fo * dx**2 / alpha
    try:
        HeatNode("rod4", n_cells=n, length=L, thermal_diffusivity=alpha,
                 timestep=dt, stencil_order=4)
        print(f"  Fo={Fo}: constructed")
    except Exception as e:
        print(f"  Fo={Fo}: {type(e).__name__}: {str(e)[:110]}")
