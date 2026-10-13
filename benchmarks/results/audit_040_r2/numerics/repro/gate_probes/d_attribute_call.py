import maddening.nodes.heat as heat
# spelled as an attribute: Fourier = 0.66 on a 4th-order rod
n = heat.HeatNode("e", timestep=1e-4, n_cells=257, length=1.0,
                  thermal_diffusivity=0.1, stencil_order=4)
