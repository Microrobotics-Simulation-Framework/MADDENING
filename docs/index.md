# MADDENING

**Modular Acausal Dataflow Differential Equation Node Network** — a pure-JAX,
autodifferentiable framework for composing physical models from typed nodes
connected by unit-aware edges. MADDENING is the base framework on which
[MIME](../mime/) and the [MICROROBOTICA](../) IDE are built.

```{toctree}
:caption: User Guide
:maxdepth: 2

user_guide/installation
user_guide/quickstart
```

```{toctree}
:caption: Algorithm Guide
:maxdepth: 2
:glob:

algorithm_guide/nodes/*
algorithm_guide/coupling/*
algorithm_guide/uq/*
```

```{toctree}
:caption: Developer Guide
:maxdepth: 2

developer_guide/node_authoring
developer_guide/testing_standards
developer_guide/documentation_standards
```

```{toctree}
:caption: Validation
:maxdepth: 2
:glob:

validation/cou_template
validation/framework_verification
validation/soup_package
validation/node_verification/*
```

```{toctree}
:caption: Regulatory
:maxdepth: 2

regulatory/intended_use
regulatory/iec62304_mapping
regulatory/eu_mdr_guidelines
regulatory/mdcg_2019_11
regulatory/downstream_integration
```
