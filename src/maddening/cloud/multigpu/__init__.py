"""Multi-GPU Jacobi coupling for MADDENING.

Distributes coupling group computation across multiple GPUs using
JAX device meshes and ``shard_map``.
"""
