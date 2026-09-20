from maddening.nodes.heat import HeatNode
# stencil_order passed POSITIONALLY (7th arg).  Fourier = 0.4:
#   stable for order 2 (limit 0.5), UNSTABLE for order 4 (limit 0.3125).
n = HeatNode("d", 0.004, 10, 1.0, 1.0, 0.0, 4)
