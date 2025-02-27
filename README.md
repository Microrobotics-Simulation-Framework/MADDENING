# MADDENING
Modular Automatic Differentiation and Data-Enhanced Neural-network INteracting Graph (MADDENING). It is designed to work as a stand-alone framework, but was specifically desigend to be the backbone of MIME (MIcrorobotics Multiphysics Engine).

## What MADDENING is intended to do, and an explanation of the Acronym.
MADDENING is a framework designed for soft real-time high-fidelity simulation of complex systems. It is designed to run on HPC server environments. 
The framework works by managing a graph (hence the Graph part of the acronym), where each node focuses on simulating a specific aspect of the overall simulation (hence the Modular part of the acronym).

The edges of the graph represent the coupling between simulations.

Nodes are written in JAX (hence the Automatic Differentiation part of the acronym), and can be reduced to Physics Informed Neural Networks (PINNs) for real time simulation. The training of PINNs can also be augmented by real-world data (hence the Data-Enhanced Neural Network part of the acronym).
