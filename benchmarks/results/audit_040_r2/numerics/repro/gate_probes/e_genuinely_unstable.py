from maddening.nodes.heat import HeatNode
n = HeatNode("f", timestep=1e-4, n_cells=257, length=1.0,
             thermal_diffusivity=0.1, stencil_order=4)
