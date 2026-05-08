# MADDENING — Intended Use and Regulatory Positioning

## Platform Positioning Statement

MADDENING (Modular Automatic Differentiation and Data Enhanced Neural-network INteracting Graph) is a **general-purpose computational framework** for multi-physics simulation. It provides infrastructure for composing, coupling, and differentiating physics simulations using JAX, with support for neural surrogates and uncertainty quantification.

**MADDENING is not a medical device** as defined by EU MDR (EU 2017/745) Article 2(1). It does not have a medical purpose, does not make clinical predictions, and does not provide diagnostic, therapeutic, or monitoring functionality. Per MDCG 2019-11 Section 3.2, MADDENING is a general-purpose computing tool comparable to MATLAB, NumPy, or FEniCS — it can be used to build medical device software, but is not itself a medical device.

When used in a regulated product, MADDENING is classified as **SOUP (Software of Unknown Provenance)** under IEC 62304:2006+AMD1:2015. The downstream commercial manufacturer is responsible for performing the SOUP assessment appropriate to their device's safety class.

## Layered Responsibility Model

| Responsibility | MADDENING | Downstream Manufacturer | End User |
|---|---|---|---|
| Algorithm correctness | Verification evidence (benchmarks, tests) | Validate for specific COU | N/A |
| Numerical stability | Document stability conditions, hazard hints | Enforce stability conditions in integration layer | Follow IFU |
| Parameter validation | Document validated regimes (`NodeMeta`) | Implement runtime validation for clinical parameters | Use within validated ranges |
| Clinical risk assessment | Not provided (technical hints only) | Full ISO 14971 risk management | Follow clinical guidance |
| Regulatory submission | SOUP documentation package | Technical file, CE marking | N/A |
| Post-market surveillance | Anomaly registry, security updates | PMS plan, PSUR, vigilance reporting | Report incidents |

## Commercial Boundary Statement

The regulated clinical product — the thing that actually gets CE-marked and used in a clinical setting — will be built by a **downstream commercial entity** (spin-out, licensee, or partner company) on top of MADDENING and its ecosystem libraries (MIME, MICROBOTICA). MADDENING, MIME, and MICROBOTICA are all open-source research tools. They do not seek CE marking, do not claim medical purpose, and do not take on manufacturer obligations under EU MDR.

The downstream commercial entity is the EU MDR **manufacturer** as defined in Article 2(30). They are solely responsible for: establishing and maintaining an ISO 13485 QMS, performing the conformity assessment procedure, engaging and satisfying the Notified Body, clinical evaluation and post-market surveillance, and bearing manufacturer liability.

## Cybersecurity Boundary Statement (MDCG 2019-16)

MADDENING **assumes trusted inputs**. All data passed to the framework — simulation parameters, graph topology, boundary conditions, external inputs — is assumed to be valid and non-malicious.

The commercial integration layer is responsible for:
- Input sanitization and parameter validation before data reaches MADDENING
- Graph topology validation (ensuring only intended node configurations are used)
- Network security for any API endpoints (the FastAPI server is a development tool, not a production deployment surface)
- Authentication and access control for simulation services

MADDENING's SBOM (Section 14 of the documentation architecture) provides the complete dependency tree needed for the manufacturer's cybersecurity documentation.

## LGPL Replaceability Statement

MADDENING is licensed under LGPL-3.0-or-later. The LGPL "replaceability" obligation is satisfied via Python's module system: the end user can replace the `maddening` package with a modified version by installing it into the same Python environment, with no special linking or build steps required. This is a standard Python package replacement mechanism and does not require any special technical provisions from the downstream product.
