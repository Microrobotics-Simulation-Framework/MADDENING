from maddening.nodes.heat import HeatNode
# stencil_order the limit table does not know: skipped, still counted
n = HeatNode("c", timestep=1.0, n_cells=10, length=1.0,
             thermal_diffusivity=100.0, stencil_order=3)
