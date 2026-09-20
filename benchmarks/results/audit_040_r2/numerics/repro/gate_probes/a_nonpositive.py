from maddening.nodes.heat import HeatNode
# non-positive alpha: the gate skips the Fourier test but still counts it
n1 = HeatNode("a", timestep=1.0, n_cells=10, length=1.0, thermal_diffusivity=0.0)
n2 = HeatNode("b", timestep=0.0, n_cells=10, length=1.0, thermal_diffusivity=100.0)
