# MADDENING Documentation Architecture Plan

## Executive Summary

**MADDENING** (Modular Automatic Differentiation and Data-Enhanced Neural-network INteracting Graph) is a JAX-based HPC framework for multiphysics simulation. It occupies a dual-purpose position that demands careful documentation architecture from the start:

1. **Research tool** — used directly by scientists and engineers to build, train, and run multiphysics simulations with automatic differentiation and neural surrogates.
2. **Auditable foundation** — the computational backbone of **MIME** (MIcrorobotics Multiphysics Engine), which in turn powers **MICROBOTICA** (MICROROBOTs Iterative Simulation for Clinical Adoption), an open-source research simulator for microrobot-assisted drug delivery in cerebrospinal fluid and other confined biological geometries. MADDENING, MIME, and MICROBOTICA are all open-source research tools. The regulated clinical product — the thing that actually gets CE-marked and used in a clinical setting — will be built by a downstream commercial entity (spin-out, licensee, or partner company) on top of these open-source tools.

This dual purpose means MADDENING's documentation must serve two audiences simultaneously: researchers who need to understand, extend, and publish with the framework, and downstream commercial manufacturers who need to cite MADDENING's verification record in regulatory submissions — without MADDENING itself making any clinical claims.

**Regulatory context**: The developer is EU-based (Netherlands) and intends to work with EU-based research labs. **EU MDR (EU 2017/745)** is therefore the primary regulatory framework, with FDA alignment as a secondary consideration. A commercial product built on top of MICROBOTICA would face EU MDR classification depending on its intended use mode: research and pre-clinical simulation requires no classification; **intraoperative digital twin** use (real-time surgical decision support) would be **Class IIb at minimum, likely Class III** since erroneous output during a live procedure in CSF spaces adjacent to the striatum and substantia nigra could directly cause irreversible neurological harm or death; and **reinforcement learning policy generation or validation for intraoperative microrobot control** would be **Class III**, as the software directly informs an active control loop for a device operating in the central nervous system. The documentation architecture must be designed to support the most demanding plausible classification (Class III) so that less demanding use cases are automatically covered.

Under **IEC 62304**, when any downstream commercial manufacturer incorporates MADDENING into a regulated product, MADDENING would be classified as SOUP (Software of Unknown Provenance) — a third-party software component whose lifecycle the manufacturer did not control. Critically, IEC 62304 assigns SOUP safety classification based on the safety class of the software item that *uses* the SOUP. For a Class III device where software failure could result in death or serious injury, MADDENING will almost certainly be classified as **IEC 62304 Class C SOUP**, which carries the most stringent lifecycle documentation requirements — including detailed architectural design, unit verification, and integration testing evidence. MADDENING's documentation must therefore be structured to satisfy Class C SOUP assessment requirements for any downstream commercial manufacturer, regardless of who they are or what specific clinical application they are building.

This plan treats documentation architecture as a first-class engineering concern. The design is informed by systematic analysis of three case studies — **3D Slicer**, **ITK (Insight Toolkit)**, and **iMSTK** — as well as the requirements of **ASME V&V 40**, **FDA computational modeling guidance (2016, 2023)**, **IEC 62304:2006+AMD1:2015**, **EU MDR (EU 2017/745)**, **MDCG 2019-11**, **MDCG 2019-16**, **IMDRF SaMD N10/N12/N23**, and **EN ISO 13485:2016**.

---

## Table of Contents

1. [Documentation Structure](#1-documentation-structure)
2. [Regulatory Boundary Language](#2-regulatory-boundary-language)
3. [Algorithm and Model Documentation Standards](#3-algorithm-and-model-documentation-standards)
4. [V&V Documentation Hooks](#4-vv-documentation-hooks)
5. [Versioning and API Stability](#5-versioning-and-api-stability)
6. [Uncertainty Quantification Documentation](#6-uncertainty-quantification-documentation)
7. [Contributor and Extensibility Documentation](#7-contributor-and-extensibility-documentation)
8. [README and Repository-Level Documentation](#8-readme-and-repository-level-documentation)
9. [Code-Embedded Documentation and Structural Hooks](#9-code-embedded-documentation-and-structural-hooks)
   - 9.1 [Node and Edge Metadata Schemas](#91-node-and-edge-metadata-schemas)
   - 9.2 [Provenance and Reproducibility Hooks](#92-provenance-and-reproducibility-hooks)
   - 9.3 [Validation Test Registration](#93-validation-test-registration)
   - 9.4 [UQ Interface Hooks](#94-uq-interface-hooks)
   - 9.5 [Deprecation and Stability Machinery](#95-deprecation-and-stability-machinery)
   - 9.6 [Audit Logging Interface](#96-audit-logging-interface)
   - 9.7 [Anomaly Management System](#97-anomaly-management-system)
   - 9.8 [ISO 14971 Risk Management Hooks](#98-iso-14971-risk-management-hooks)
   - 9.9 [Usability Engineering Considerations](#99-usability-engineering-considerations)
10. [IEC 62304 SOUP Compliance Package](#10-iec-62304-soup-compliance-package)
11. [IEC 62304 Software Lifecycle Documentation Mapping](#11-iec-62304-software-lifecycle-documentation-mapping)
12. [EU MDR Annex I and Annex II Alignment](#12-eu-mdr-annex-i-and-annex-ii-alignment)
13. [MDCG 2019-11: Qualification and Classification](#13-mdcg-2019-11-qualification-and-classification)
14. [Configuration Management](#14-configuration-management)
15. [QMS Compatibility](#15-qms-compatibility)
16. [Downstream Framework Inheritance](#16-downstream-framework-inheritance)

**Appendices:**

- [A: Case Study Summary Table](#appendix-a-case-study-summary-table)
- [B: Implementation Priority](#appendix-b-implementation-priority)
- [C: Relationship to Existing Documentation](#appendix-c-relationship-to-existing-documentation)
- [D: EU Regulatory Roadmap for the MADDENING Ecosystem](#appendix-d-eu-regulatory-roadmap-for-the-maddening-ecosystem)
- [E: Project Setup Checklist](#appendix-e-project-setup-checklist)
- [F: MIME / Downstream Library Bootstrap Checklist](#appendix-f-mime--downstream-library-bootstrap-checklist)

---

## 1. Documentation Structure

### Case Study Patterns

**ITK** uses a three-tier documentation architecture that separates concerns cleanly:
- **Tier 1**: In-code API documentation (Doxygen with LaTeX equations and a centralized `doxygen.bib`)
- **Tier 2**: The ITK Software Guide (two-volume LaTeX document — architecture + algorithms)
- **Tier 3**: Executable examples (Sphinx + Jupyter, each self-contained with documentation, code, test data)

**3D Slicer** splits documentation into User Guide and Developer Guide audiences, with per-module documentation following a consistent template (description, use cases, panels, limitations, references).

**iMSTK** uses a single monolithic overview page as an entry point with deep-dive subpages per subsystem, plus Doxygen-generated API reference.

### MADDENING Documentation Structure

MADDENING should adopt ITK's three-tier model, adapted for a Python/JAX project:

```
MADDENING/
├── README.md                          # Project overview + disclaimers
├── DESIGN.md                          # Architecture decisions (existing)
├── ROADMAP.md                         # Feature roadmap (existing)
├── CHANGELOG.md                       # Structured changelog (new)
├── CITATION.cff                       # Academic citation metadata (new)
├── GOVERNANCE.md                      # Decision-making process (new)
├── CONTRIBUTING.md                    # Contributor guide (new)
├── SECURITY.md                        # Security reporting (new)
├── LICENSE                            # LGPL-3.0-or-later (existing)
│
├── docs/                              # Tier 2: Comprehensive documentation
│   ├── conf.py                        # Sphinx configuration
│   ├── index.rst                      # Documentation root
│   │
│   ├── user_guide/                    # For simulation users
│   │   ├── installation.md
│   │   ├── quickstart.md
│   │   ├── concepts.md                # Functional state, graph, edges, etc.
│   │   ├── tutorials/                 # Step-by-step tutorials
│   │   └── faq.md
│   │
│   ├── developer_guide/               # For framework contributors
│   │   ├── architecture.md            # System architecture overview
│   │   ├── node_authoring.md          # How to write a SimulationNode
│   │   ├── surrogate_authoring.md     # How to write a SurrogateArchitecture
│   │   ├── testing_standards.md       # Test requirements
│   │   ├── documentation_standards.md # Doc requirements for contributions
│   │   └── code_style.md
│   │
│   ├── algorithm_guide/               # Mathematical documentation (Tier 2)
│   │   ├── index.md                   # Algorithm documentation overview
│   │   ├── nodes/                     # One document per node type
│   │   │   ├── ball_node.md
│   │   │   ├── heat_node.md
│   │   │   ├── lbm_pipe_node.md
│   │   │   ├── spring_damper_node.md
│   │   │   ├── rigid_body_2d_node.md
│   │   │   └── _template.md           # Template for new node docs
│   │   ├── surrogates/                # One document per architecture
│   │   │   ├── mlp.md
│   │   │   ├── deeponet.md
│   │   │   ├── fno.md
│   │   │   └── _template.md
│   │   ├── coupling/                  # Coupling and scheduling algorithms
│   │   │   ├── gauss_seidel.md
│   │   │   ├── adaptive_timestepping.md
│   │   │   └── multi_rate.md
│   │   └── numerical_methods/         # Shared numerical methods
│   │       ├── richardson_extrapolation.md
│   │       ├── pi_controller.md
│   │       └── euler_rk4_integrators.md
│   │
│   ├── validation/                    # V&V documentation
│   │   ├── index.md                   # V&V philosophy and scope
│   │   ├── framework_verification.md  # Framework-level verification evidence
│   │   ├── node_verification/         # Per-node verification reports
│   │   │   ├── heat_node.md           # Analytical comparison, convergence
│   │   │   ├── lbm_pipe_node.md       # Benchmark comparisons
│   │   │   └── _template.md
│   │   ├── benchmarks/                # Benchmark problem documentation
│   │   │   └── index.md
│   │   ├── known_anomalies.yaml       # Machine-readable anomaly registry
│   │   ├── soup_package.md            # IEC 62304 SOUP package document
│   │   └── cou_template.md            # Context-of-use template for users
│   │
│   ├── regulatory/                    # Regulatory context documentation
│   │   ├── intended_use.md            # Platform positioning statement
│   │   ├── eu_mdr_guidelines.md       # EU MDR alignment guide (primary)
│   │   ├── fda_guidelines.md          # FDA alignment guide (secondary)
│   │   ├── iec62304_mapping.md        # IEC 62304 lifecycle mapping
│   │   ├── mdcg_2019_11.md            # Software qualification guidance
│   │   └── downstream_integration.md  # MIME/MICROBOTICA relationship
│   │
│   ├── api_reference/                 # Tier 1: Auto-generated API docs
│   │   └── (sphinx-autodoc output)
│   │
│   ├── releases/                      # Per-version release notes
│   │   ├── v0.1.0.md
│   │   └── _template.md
│   │
│   └── bibliography.bib              # Centralized academic references
│
├── examples/                          # Tier 3: Executable examples
│   ├── (existing examples, enhanced with docs)
│   └── notebooks/                     # Jupyter notebooks
│
└── tests/                             # Test suite (existing)
    ├── verification/                  # Verification benchmarks (new)
    │   ├── test_heat_analytical.py
    │   ├── test_lbm_poiseuille.py
    │   └── ...
    └── (existing test files)
```

**Rationale**: ITK's three-tier separation keeps API reference, mathematical documentation, and practical examples independently maintainable. The `docs/algorithm_guide/` directory mirrors ITK's Software Guide Book 2. The `docs/validation/` directory is absent from all three case studies and represents a differentiation opportunity. The `docs/regulatory/` directory now includes EU MDR-specific documents alongside FDA guidance, reflecting the EU-primary regulatory strategy.

**Tooling**: Sphinx with MyST-Parser (Markdown with LaTeX math support via `$...$` and `$$...$$`), sphinx-autodoc for API reference, and `sphinxcontrib-bibtex` for centralized bibliography.

---

## 2. Regulatory Boundary Language

### Case Study Patterns

**3D Slicer** establishes a clear "platform, not product" boundary. The software is explicitly NOT FDA-cleared; it is a research platform. Derivative products are built by commercial integrators who perform their own validation. The BSD license is chosen specifically to enable this.

**ITK** takes an educational approach: a dedicated `fda_sw_development_guidelines.md` page in the `learn/` section teaches integrators how to use ITK in FDA-regulated workflows. ITK also maintains a `third_party_applications.md` documenting real-world FDA-cleared products built on ITK, establishing credibility without making claims.

**iMSTK** has almost no regulatory language — only implicit framing through words like "prototyping" and "training." This is a significant gap for a medical simulation toolkit.

### MADDENING Approach

MADDENING should follow ITK's educational model while being more explicit than any of the case studies about the layered relationship between framework and downstream tools. Because the developer is EU-based and EU MDR is the primary regulatory target, the regulatory language must address both EU and US frameworks.

#### `docs/regulatory/intended_use.md`

This document should contain the following elements:

**Platform Positioning Statement** — clear, concise, and legally precise:

> MADDENING is a general-purpose computational framework for multiphysics simulation and surrogate model training. It is research software distributed under the LGPL-3.0-or-later license.
>
> MADDENING is NOT a medical device as defined by EU MDR (EU 2017/745) Article 2(1), nor is it a medical device under US FDA regulations. It does not have a medical purpose. It is not intended for direct clinical use, clinical decision-making, or patient diagnosis. It has not been CE-marked, cleared, or approved by any regulatory body.
>
> MADDENING is a general-purpose computational tool — analogous to a mathematical library or a finite element solver. Under the qualification criteria of MDCG 2019-11, software without a medical purpose is not a medical device regardless of whether it is subsequently used within a medical device. MADDENING performs no clinical interpretation, provides no diagnostic output, and makes no therapeutic recommendations.
>
> MADDENING is designed to be a credible, auditable computational foundation that downstream commercial manufacturers may build upon. When incorporated into a regulated medical device by any commercial entity, MADDENING is classified as SOUP (Software of Unknown Provenance) under IEC 62304:2006+AMD1:2015. The device manufacturer bears full responsibility for assessing MADDENING's suitability for their specific context of use and for performing all required verification and validation activities.

**Cybersecurity Boundary Statement** (MDCG 2019-16):

> MADDENING is a mathematical computation library that assumes trusted inputs. It performs no input sanitization, authentication, authorisation, or network security functions. All simulation parameters, graph topologies, external inputs, and boundary conditions are assumed to be provided by a trusted caller.
>
> When MADDENING is incorporated into a regulated product, the commercial integration layer is solely responsible for: validating and sanitizing all simulation parameters before they reach MADDENING; ensuring graph topologies are well-formed and represent physically meaningful configurations; protecting the simulation server (if deployed via `SimulationServer`) with appropriate network security, authentication, and access controls; and preventing injection of malicious parameters or graph configurations through any user-facing interface.
>
> This boundary is analogous to how a mathematical library like NumPy or LAPACK assumes its callers provide valid inputs — the security perimeter belongs to the application, not the library.

**LGPL Replaceability Statement**:

> LGPL-3.0 Section 4 requires that the end user be able to replace the LGPL-licensed component with a modified version. For MADDENING, this obligation is satisfied via Python's standard module system: the end user can replace the `maddening` package by installing a modified version into the same Python environment (e.g., `pip install ./my-modified-maddening`). No special linking, build steps, or binary compatibility mechanisms are required. The commercial product need not provide its own source code or build tooling — only the ability to substitute the MADDENING package.

**Layered Responsibility Model** (updated for EU MDR context):

| Responsibility | Owner | Applicable Standard | Evidence |
|---|---|---|---|
| Code verification (algorithms correct) | MADDENING project | IEC 62304 Clause 5.6 (via SOUP assessment) | Test suite, benchmarks, analytical comparisons |
| Software quality assurance | MADDENING project | IEC 62304 Clause 8 (configuration management) | Version control, CI, code review, issue tracking |
| SOUP assessment | Downstream manufacturer | IEC 62304 Clause 5.3.3, 5.3.4 | Using MADDENING's SOUP package document |
| Known anomaly evaluation | Downstream manufacturer | IEC 62304 Clause 7.1.3 | Using MADDENING's known anomalies registry |
| Calculation verification | Downstream user | ASME V&V 40 | Mesh convergence, solver settings, error estimation |
| Validation for specific COU | Device manufacturer | EU MDR Annex I Section 17.2 | Experimental comparisons for their COU |
| Risk management | Device manufacturer | ISO 14971, EU MDR Annex I Chapter I | Risk management file for their device |
| Technical documentation | Device manufacturer | EU MDR Annex II | Full technical file for CE marking |
| Clinical evidence | Device manufacturer | EU MDR Annex XIV, MEDDEV 2.7/1 rev. 4 | Clinical evaluation report; PMCF plan (Class III) |
| Regulatory submission | Device manufacturer | EU MDR Article 52+ | CE marking, Notified Body review (Class IIb/III) |

#### `docs/regulatory/eu_mdr_guidelines.md`

The primary regulatory guidance document (replacing the US-centric approach), explaining:

- What EU MDR requires for software in medical devices (Annex I Section 17)
- How IEC 62304 applies to the MADDENING ecosystem
- What MDCG 2019-11 means for MADDENING's classification status
- How to reference MADDENING's verification evidence in a technical file
- SOUP assessment workflow using MADDENING's provided documentation
- SBOM (Software Bill of Materials) and cybersecurity considerations per MDCG 2019-16

#### `docs/regulatory/fda_guidelines.md`

Secondary regulatory guidance for US-market alignment, explaining:

- What ASME V&V 40 requires and how MADDENING's verification evidence supports it
- FDA computational modeling guidance (2016, 2023) requirements
- How to cite MADDENING's verification record in an FDA submission

#### `docs/regulatory/downstream_integration.md`

Documents the specific relationship chain:

```
MADDENING (framework, LGPL, open source research tool, not a medical device)
    └── MIME (physics engine for microrobotics, open source, not a medical device)
        └── MICROBOTICA (research simulator, open source, not a medical device)
            └── [Commercial Product] (SaMD, EU MDR Class IIb–III, CE marking required)
                 Built by a commercial entity taking on manufacturer obligations
```

This document explains what each layer provides and what each layer is responsible for. It specifically addresses:
- Why MADDENING, MIME, and MICROBOTICA are all open-source research tools with no medical purpose
- Why none of them are medical devices or seek CE marking
- Why the regulated clinical product is built by a downstream commercial entity on top of MICROBOTICA
- How that commercial product's classification varies by intended use mode (research: unregulated; intraoperative digital twin: Class IIb/III; RL policy generation for active control: Class III)
- Why the documentation architecture targets Class III readiness so that lower-class use cases are automatically covered
- What MADDENING provides to any downstream commercial manufacturer's technical file

**Phase 1 skeleton**: A skeleton version of this file should be established in Phase 1 (alongside `intended_use.md`) containing at minimum: the four-layer dependency chain shown above, a one-paragraph statement that the commercial layer is responsible for EU MDR manufacturer obligations (ISO 13485 QMS, Notified Body relationship, post-market surveillance, and manufacturer liability), and a forward reference to the full content to be completed in Phase 3. This ensures the commercial boundary is documented from the first release, even before the full details are needed.

### The Commercial Boundary

A strategic decision underpins the entire documentation architecture: **MADDENING, MIME, and MICROBOTICA are open-source research tools. None of them will seek CE marking or FDA clearance. None of them are medical devices. None of them carry manufacturer liability under EU MDR.**

The regulated clinical product — the thing that actually gets CE-marked and used in a clinical setting — will be built by a downstream commercial entity (spin-out, licensee, or partner company) on top of these open-source tools. That entity takes on the QMS, the Notified Body relationship, post-market surveillance obligations, and EU MDR manufacturer liability. They have the organisational structure and resources to do so appropriately.

#### Why the Regulated Product Must Be a Commercial Entity

EU MDR Article 10 places extensive obligations on the manufacturer of a medical device. These are incompatible with open-source research development:

- **ISO 13485 QMS**: Requires documented quality management system with management reviews, internal audits, corrective and preventive action (CAPA) systems, and designated personnel with defined roles. A solo researcher or open-source project cannot credibly maintain this.
- **Notified Body costs**: Initial conformity assessment audit for Class III is substantial (tens of thousands of euros), with annual surveillance audits thereafter. These are ongoing financial commitments.
- **Post-market surveillance**: EU MDR requires active, systematic monitoring of clinical performance (PMS plan), annual Periodic Safety Update Reports (PSUR) for Class III, Post-Market Clinical Follow-up (PMCF) studies, and vigilance reporting to competent authorities within statutory timeframes (serious incidents within 15 days). These are legal obligations with personal liability.
- **Manufacturer liability**: EU MDR Article 10 holds the manufacturer personally liable. Under EU product liability law, the manufacturer bears strict liability for harm caused by defective products. This requires corporate structure, insurance, and legal resources.

This is not a limitation but a deliberate and well-precedented architectural decision. It mirrors how **ITK**, **VTK**, and **3D Slicer** relate to the commercial products built on them: **Brainlab**, **Materialise**, **Synopsys**, and others build FDA-cleared and CE-marked products on these open-source foundations. The open-source project provides the auditable, peer-reviewed computational foundation; commercial manufacturers handle the regulatory pathway.

#### What MADDENING's Open-Source Nature Provides to Commercial Manufacturers

MADDENING's open-source nature is an asset, not a liability, in this model:

- **Full inspectability**: Notified Body reviewers can examine the complete source code. There are no black-box concerns — every algorithm, every numerical method, every line of code is available for scrutiny.
- **Published research credibility**: Broad adoption in published academic research builds an independent verification record. Peer-reviewed publications citing MADDENING constitute third-party evidence of computational credibility that is difficult to manufacture commercially.
- **No vendor lock-in**: The commercial entity is not dependent on a single vendor for their computational foundation. The LGPL licence ensures they can always fork, modify, and maintain MADDENING independently if needed.
- **Community-surfaced anomalies**: A large research user base surfaces bugs and edge cases faster than any internal QA team. Every research user running MADDENING in novel configurations is effectively performing exploratory testing.

#### Responsibility Allocation

| Responsibility | MADDENING (open source) | Commercial Manufacturer |
|---|---|---|
| Auditable computational foundation | **Provides** | Uses |
| SOUP package documentation | **Provides** | References in technical file |
| Verification evidence (analytical benchmarks) | **Provides** | Evaluates and supplements |
| Known anomalies registry | **Provides** | Evaluates for safety impact in COU |
| ISO 14971 risk management file | Does not provide | **Must create** |
| Clinical evidence (CER, PMCF) | Does not provide | **Must generate** |
| ISO 13485 QMS | Does not operate under | **Must certify** |
| Notified Body relationship | None | **Must establish and maintain** |
| Post-market surveillance (PMS, PSUR, vigilance) | None | **Must conduct** |
| EU MDR manufacturer liability | None | **Bears** |

Any number of commercial entities may independently build regulated products on top of MADDENING, MIME, and MICROBOTICA. The documentation architecture is designed to serve all of them equally — any manufacturer performing a Class C SOUP assessment of MADDENING should find everything they need in the SOUP package, regardless of who they are or what specific clinical application they are building.

#### LGPL Licence and the Commercial Model

MADDENING is licensed under LGPL-3.0-or-later, which is deliberately chosen to enable this model:

- **Commercial use permitted**: The LGPL explicitly permits commercial use, including building proprietary products that link against MADDENING.
- **Weak copyleft**: Unlike the GPL, the LGPL's "weak copyleft" means proprietary products can link against MADDENING without being required to open-source their own code. Only modifications to MADDENING itself must be released under LGPL.
- **Distribution requirements**: Commercial entities distributing products that include MADDENING must provide the LGPL-licensed MADDENING source (or a written offer to do so) and allow the user to replace the LGPL component. This is standard practice and does not impede commercial deployment.
- **Patent grant**: LGPL-3.0 includes an implicit patent licence, reducing IP risk for commercial adopters.

Commercial entities building on MADDENING should assess LGPL compatibility with their own licensing strategy. The LGPL is widely accepted in medical device software — it imposes no restriction on the proprietary status of the clinical interpretation layer, user interface, or any other code that does not modify MADDENING itself.

**Future commercial offerings**: A future commercial version, dual licensing arrangement, or services offering around MADDENING itself is not ruled out. If the developer or a future organisation chooses to offer commercial licences, paid support, hosted infrastructure, or other services around MADDENING, MIME, or MICROBOTICA, the documentation architecture described in this document supports that future without requiring structural changes. The LGPL explicitly permits this kind of dual-track open source / commercial offering. This possibility is acknowledged as a natural consequence of the licence model, not as a commitment or plan.

#### Research Adoption as Regulatory Strategy

MADDENING's value to any commercial entity increases with the size and diversity of the research community using it. Broad adoption in published research:

- Builds the verification record through independent, peer-reviewed use
- Surfaces anomalies faster through diverse usage patterns
- Improves the algorithms through community contributions
- Establishes the kind of independent, peer-reviewed credibility that Notified Body reviewers find most persuasive

This is why keeping the tools fully open source and maximising research adoption is the right strategy — it directly strengthens the regulatory case for any downstream commercial product, whoever builds it.

---

## 3. Algorithm and Model Documentation Standards

### Case Study Patterns

**ITK** sets the gold standard: every filter has Doxygen documentation with LaTeX equations, references to a centralized `.bib` file, assumptions, limitations, and cross-references. The Software Guide provides chapter-length treatments of major algorithms with derivations from first principles.

**iMSTK** documents physics models at three levels (conceptual overview, academic citations, Doxygen + MathJax) but lacks convergence properties, stability conditions, and formal validation data.

**3D Slicer** follows a "reference, don't reproduce" pattern — citing foundational papers rather than reproducing derivations.

### MADDENING Algorithm Documentation Standard

Every node type, physics model, and numerical method in MADDENING must have a corresponding document in `docs/algorithm_guide/` following this template. The template draws from ITK's filter documentation pattern and iMSTK's three-level approach, extended with V&V 40 and IEC 62304 requirements.

#### Node Algorithm Documentation Template (`docs/algorithm_guide/nodes/_template.md`)

```markdown
---
bibliography: ../../bibliography.bib
---

# [Node Name]

**Module**: `maddening.nodes.[module]`
**Stability**: [experimental | stable | deprecated]
**Algorithm ID**: `MADD-NODE-[XXX]`
**Version**: [semantic version of this algorithm implementation]

## Summary

[1-2 sentence description of what this node simulates.]

## Governing Equations

[Full mathematical formulation. Use LaTeX math blocks.]

$$
\frac{\partial T}{\partial t} = \alpha \nabla^2 T + S
$$

## Discretization

[How the continuous equations are discretized. Explicit/implicit,
finite difference/finite element/lattice Boltzmann, order of accuracy.]

## Implementation Mapping

[Trace every term in the governing equations and discretization to the
specific Python/JAX function that implements it. This is mandatory for
IEC 62304 Class C detailed design traceability (Clause 5.4).]

| Equation Term | Implementation | Notes |
|---------------|---------------|-------|
| $\alpha \nabla^2 T$ (diffusion) | `HeatNode._diffusion_step()` | 2nd-order central FD stencil |
| $S$ (source term) | `HeatNode._apply_source()` | Added after diffusion |
| Time integration ($\partial T / \partial t$) | Forward Euler in `HeatNode.update()` | 1st-order explicit |
| Boundary conditions | `state.at[0].set(left_T)` | JAX primitive: `jax.numpy.ndarray.at[].set()` |

[If a term is handled by a JAX primitive or third-party function for
which no explicit MADDENING function exists, document the primitive
and its calling convention. Every governing equation term must appear
in this table — silent omissions are not acceptable.]

[Example for a spectral method node:]

| Equation Term | Implementation | Notes |
|---------------|---------------|-------|
| Forward FFT | JAX primitive: `jnp.fft.rfftn()` (jax ≥ 0.4.20) | No MADDENING wrapper; called directly in `FNOLayer.forward()` |

## Assumptions and Simplifications

[Numbered list of every physical and mathematical assumption.]

1. [e.g., "Incompressible flow (Mach number << 1)"]
2. [e.g., "Uniform grid spacing"]
3. [e.g., "Constant material properties (no temperature dependence)"]

## Validated Physical Regimes

[Quantitative bounds on where this model has been verified/validated.]

| Parameter | Verified Range | Notes |
|-----------|---------------|-------|
| Reynolds number | 0 – 100 | Validated against Poiseuille analytical solution |
| Grid resolution | 16³ – 128³ | Convergence study in verification report |
| CFL number | < 0.5 | Stability limit for explicit scheme |

## Known Limitations and Failure Modes

[Specific conditions where this model is known to produce incorrect or
unreliable results. This section feeds directly into IEC 62304 SOUP
anomaly assessment — downstream tools must evaluate each limitation
for safety impact in their context of use.]

1. [e.g., "CFL > 1 causes numerical instability"]
2. [e.g., "Does not capture turbulent transition (Re > ~2000)"]

## Stability Conditions

[Analytical or empirical stability bounds for the numerical scheme.]

$$
\Delta t < \frac{\Delta x^2}{2 \alpha d}
$$

where $d$ is the spatial dimension.

## State Variables

| Field | Shape | Units | Description |
|-------|-------|-------|-------------|
| temperature | (n_cells,) | K | Nodal temperatures |

## Parameters

| Parameter | Type | Default | Units | Description |
|-----------|------|---------|-------|-------------|
| thermal_diffusivity | float | 0.01 | m²/s | Thermal diffusivity α |

## Boundary Inputs

| Field | Shape | Default | Description |
|-------|-------|---------|-------------|
| left_temperature | scalar | current T[0] | Dirichlet BC, left end |

## References

[Cite using Pandoc-style `[@Key]` syntax, where `Key` matches an entry
in `docs/bibliography.bib`.  Multiple citations: `[@Key1; @Key2]`.
CI validates all cited keys exist via `scripts/check_citations.py`.]

- [@AuthorYear] Author, A. (Year). *Title*. Publisher. — Brief note on relevance to this node.

## Verification Evidence

[Link to the verification report and specific test files.]

- Verification report: [](../validation/node_verification/heat_node.md)
- Test files: `tests/verification/test_heat_analytical.py`

## Changelog

| Version | Date | Change |
|---------|------|--------|
| 1.0.0 | 2025-XX-XX | Initial implementation |
```

**Rationale**: This template goes beyond what any of the three case studies provide. The **Implementation Mapping** section is mandatory — it provides the equation-to-code traceability that IEC 62304 Clause 5.4 (detailed design) requires for Class C SOUP assessment. Every governing equation term must be traceable to a specific function; terms handled by JAX primitives are documented as such rather than silently omitted. The **Known Limitations** section explicitly notes its role in IEC 62304 SOUP anomaly assessment, creating a direct link between algorithm documentation and regulatory workflow. The **Algorithm ID** (`MADD-NODE-XXX`) follows the practice of numbered identifiers for traceability in regulatory contexts.

**`NodeMeta.implementation_map` migration path**: The Phase 2 requirement is a Markdown table maintained manually in each algorithm guide document. The recommended migration path for Phase 3/4 is to move the Implementation Mapping into a machine-readable `NodeMeta.implementation_map` field — a dictionary mapping equation term descriptions to Python function qualified names (e.g., `{"α∇²T (diffusion)": "maddening.nodes.heat.HeatNode._diffusion_step"}`). Once this field exists, a Sphinx documentation build step can verify via `getattr()` that every function name in the mapping resolves to an existing callable in the source code, failing the documentation build if a function is renamed or removed without updating the mapping. This closes the traceability loop: the algorithm guide's Markdown table is auto-generated from `NodeMeta.implementation_map`, the Sphinx build verifies that every mapping target exists, and the mapping itself is a machine-readable artifact that downstream SOUP assessors can harvest programmatically. Until the migration is complete, the manually maintained Markdown table remains authoritative.

**Phase 2 CI bridge (documentation rot prevention)**: Even before the full `NodeMeta.implementation_map` migration in Phase 3, the manually maintained Markdown table is vulnerable to documentation rot — a renamed internal function will silently break the traceability chain. To mitigate this, a lightweight CI check should be introduced in Phase 2: a script that parses each algorithm guide's Implementation Mapping table, extracts the function qualified names from the "Implementation" column, and verifies via `importlib` and `getattr()` that each name resolves to an existing callable. This script does not require Sphinx infrastructure — it is a standalone Python script (`scripts/check_impl_mapping.py`) that reads Markdown files directly. The check runs on every push and fails if any function name is stale. This is a bridge measure: it provides rot detection without requiring the full `NodeMeta.implementation_map` machinery, and it becomes redundant once the Phase 3 migration is complete.

**Bibliography citation system**: Algorithm guide documents cite references using Pandoc-style `[@Key]` syntax (e.g., `[@Crank1975]`, `[@Key1; @Key2]` for multiple citations). All cited keys must correspond to BibTeX entries in `docs/bibliography.bib`. A CI script (`scripts/check_citations.py`) parses the `.bib` file for valid entry keys and scans all algorithm guide Markdown files for `[@Key]` patterns, failing the build if any cited key is missing from the bibliography. Unused bibliography entries are reported as warnings (non-blocking). Template files prefixed with `_` are excluded from scanning.

**Citation rendering**: The `[@Key]` syntax is not natively rendered by GitHub or VS Code Markdown previews — it appears as literal text. This is by design: each References section includes a human-readable inline description alongside the citation key (e.g., `[@Crank1975] Crank, J. (1975). *The Mathematics of Diffusion*. ...`), so the document remains fully readable without Pandoc processing. For rendered output, Pandoc can process the citations locally via `pandoc --citeproc --bibliography=docs/bibliography.bib input.md -o output.pdf`. Each algorithm guide file includes a YAML frontmatter block specifying `bibliography: ../../bibliography.bib` so that Pandoc can locate the `.bib` file automatically. In Phase 5 (Sphinx documentation site), `sphinxcontrib-bibtex` or a Pandoc-based build step will render citations into the published documentation with proper formatting and hyperlinked reference lists. A GitHub Pages deployment with Pandoc rendering is feasible via the `pandoc/actions` GitHub Action but is deferred to Phase 5 to avoid premature CI complexity.

---

## 4. V&V Documentation Hooks

### Regulatory Context

**ASME V&V 40-2018** defines a risk-informed credibility assessment framework. The key insight is that a general-purpose framework cannot itself have a Context of Use (COU) — only a specific application can.

**FDA guidance (2023)** aligns with V&V 40 and defines a nine-step credibility assessment process. **EU MDR Annex I Section 17** requires that software be developed and manufactured in accordance with the state of the art, taking into account principles of the development lifecycle, risk management, information security, verification, and validation. IEC 62304 provides the lifecycle framework that satisfies this requirement.

MADDENING's V&V documentation should provide the framework-level evidence that downstream tools reference, structured to align with both EU and US standards.

### V&V Documentation Structure

#### `docs/validation/index.md` — V&V Philosophy and Scope

This document should explain:

- MADDENING's scope within V&V 40 (code verification + SQA only)
- MADDENING's scope within IEC 62304 (SOUP provider — provides documentation to support downstream SOUP assessment)
- What downstream tools are responsible for (calculation verification, validation, applicability, UQ)
- How to reference MADDENING's verification evidence in a regulatory submission (EU technical file or FDA submission)
- Version-pinning requirements (every citation must reference a specific tagged release)

#### `docs/validation/framework_verification.md` — Framework-Level Evidence

A living document aggregating verification evidence across the entire framework:

- **Software quality assurance**: version control (Git), CI pipeline, code review process, issue tracking, test coverage metrics
- **Test suite summary**: number of tests, categories, pass rates per release
- **Platform coverage**: OS/Python/JAX versions tested in CI
- **Numerical code verification**: list of all analytical benchmarks with pass/fail status per release

This maps to V&V 40 Tables 4, 7, 8 and supports IEC 62304 Clause 5.6 (software unit verification) evidence.

#### `docs/validation/node_verification/` — Per-Node Verification Reports

Each node that has been verified gets a report documenting:

- **Benchmark problem description** (analytical solution or reference data)
- **Comparison methodology** (metrics, tolerances)
- **Results** (plots, tables, error norms) — generated automatically from CI
- **Convergence studies** (if applicable)
- **Version tested** and **date**

#### `docs/validation/cou_template.md` — Context of Use Template

A template helping downstream users define their COU per V&V 40:

1. **Question of interest**: what the model predicts
2. **Quantities of interest**: specific output variables
3. **Model influence**: how much the model drives the decision
4. **Decision consequence**: severity of harm if wrong
5. **Risk assessment**: model influence × decision consequence
6. **Credibility goals**: required evidence level per V&V 40
7. **Evidence plan**: what verification, validation, and UQ are needed

### V&V Boundary: Where MADDENING Ends and Clinical Evidence Begins

For the most demanding downstream use modes (Class IIb/III under EU MDR Rule 11), clinical evaluation must meet a correspondingly high evidentiary standard. It is essential that MADDENING's documentation is honest about what its V&V record does and does not demonstrate.

**What MADDENING's V&V provides**:
- **Code verification**: proof that the numerical implementation correctly solves the intended mathematical equations (analytical benchmarks, convergence studies, manufactured solutions)
- **Software quality assurance**: evidence of systematic development (CI, tests, version control, code review)
- **Algorithm documentation**: mathematical formulation, discretization, assumptions, validated physical regimes, known limitations

**What MADDENING's V&V does NOT provide**:
- **Calculation verification for a specific COU**: MADDENING cannot verify that a specific simulation setup (mesh, parameters, boundary conditions) is adequate for a specific clinical question. That is the downstream user's responsibility per ASME V&V 40.
- **Validation against clinical reality**: MADDENING does not demonstrate that any simulation output matches what happens in a living patient. Clinical validation — comparing model predictions to patient outcomes or to in vivo/ex vivo experimental data — is the commercial manufacturer's responsibility and must be part of their clinical evaluation report.
- **Clinical evidence**: EU MDR Annex XIV requires clinical evidence demonstrating safety and performance. For Class III devices, this requires a clinical evaluation report (CER) that meets the highest evidentiary standard. MADDENING's verification benchmarks (e.g., Poiseuille flow convergence) are computational verification, not clinical evidence. They support the credibility argument but do not substitute for clinical data.
- **Post-market clinical follow-up (PMCF)**: Class III devices require PMCF plans and studies. These are entirely the commercial manufacturer's responsibility and cannot be delegated to or supported by MADDENING.

**What MADDENING's V&V evidence enables**: Although MADDENING's verification evidence is not clinical evidence, it provides the commercial manufacturer with a strong foundation for their clinical evaluation report. The manufacturer is not starting from zero — they have documented, peer-reviewed numerical methods with analytical benchmarks, convergence studies, and published algorithm documentation. This establishes that the computational foundation is sound, allowing the manufacturer to focus their clinical evaluation on the question that matters: does the complete simulation chain produce clinically accurate predictions for their specific intended use?

**Practical implication for MADDENING's documentation**: Every V&V artifact should clearly state its scope. Verification reports should include a "Scope and Limitations" section explicitly noting that computational verification does not constitute clinical validation. The `docs/validation/cou_template.md` should guide downstream users in identifying the additional validation evidence they must generate for their specific clinical context.

### Verification Benchmark Integration with CI

Verification benchmarks should run as part of CI and produce structured output (JSON) that can be automatically aggregated into the living V&V documentation. See Section 9.3 for the code-level registration pattern.

---

## 5. Versioning and API Stability

### Case Study Patterns

**ITK** uses full semantic versioning with alpha/beta/RC pre-releases, dedicated migration guides per major version, and a Compatibility module for backward-compat shims.

**iMSTK** documents API changes in release notes with explicit rename/replacement/removal guidance and uses commit prefixes (BUG/ENH/REFAC/DOC) for automated changelog generation.

### MADDENING Versioning Policy

#### Semantic Versioning

MADDENING should follow strict semantic versioning (SemVer 2.0):

- **MAJOR** (X.0.0): Breaking API changes, removed features, changed behavior
- **MINOR** (0.X.0): New features, new nodes, new architectures (backward-compatible)
- **PATCH** (0.0.X): Bug fixes, documentation updates, performance improvements

Pre-release versions: `X.Y.Z-alpha.N`, `X.Y.Z-beta.N`, `X.Y.Z-rc.N`

#### `CHANGELOG.md`

Following [Keep a Changelog](https://keepachangelog.com/) with iMSTK-style categorization:

```markdown
# Changelog

All notable changes to MADDENING are documented in this file.

## [Unreleased]

### Added
### Changed
### Deprecated
### Removed
### Fixed
### Verification
- [Tracks changes to V&V status]
### Security
- [Required by MDCG 2019-16 cybersecurity guidance]
### Known Anomalies
- [Changes to known_anomalies.yaml — required for IEC 62304 SOUP]
```

The **Verification**, **Security**, and **Known Anomalies** sections are unique to MADDENING and directly support EU regulatory workflows.

#### Commit Message Convention

| Prefix | Meaning |
|--------|---------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `refactor:` | Code restructuring (no behavior change) |
| `docs:` | Documentation only |
| `test:` | Test additions or changes |
| `perf:` | Performance improvement |
| `verify:` | Verification/validation evidence |
| `break:` | Breaking change |
| `deprecate:` | Deprecation notice |
| `security:` | Security-relevant change |

#### Migration Guides

For each major version, produce a dedicated `docs/releases/migration_vX.md` documenting every breaking change with before/after code examples.

#### API Stability Communication

Each public API surface carries a stability level (see Section 9.5 for code-level machinery):

| Level | Meaning | SemVer Guarantee |
|-------|---------|-----------------|
| **stable** | Covered by SemVer; breaking changes only in major versions | Full |
| **provisional** | API may change in minor versions with deprecation warnings | One minor version notice |
| **experimental** | API may change without notice | None |
| **deprecated** | Scheduled for removal; use alternative | Removed in next major |

---

## 6. Uncertainty Quantification Documentation

### Regulatory Context

ASME V&V 40 requires UQ at four levels: input parameter uncertainties, sensitivity analysis, model form uncertainty, and propagated output uncertainty. EU MDR Annex I Section 17.1 requires that software be designed to ensure "repeatability, reliability and performance in line with its intended use," which implicitly requires understanding of output uncertainty.

### Current State

MADDENING already has foundations that support UQ workflows:
- `run_sweep()` via `jax.vmap` enables ensemble evaluation over parameter ranges
- `jax.grad` enables sensitivity computation
- Functional purity makes Monte Carlo sampling straightforward
- Adaptive timestepping provides numerical error estimation via Richardson extrapolation

### Documentation Architecture for UQ

Even before UQ is implemented as a first-class feature, the documentation should:

#### 1. Reserve the namespace

Create `docs/algorithm_guide/uq/` with an `index.md` that defines the roadmap and maps existing capabilities to V&V 40 UQ requirements.

#### 2. Document existing UQ-relevant capabilities

| V&V 40 UQ Requirement | MADDENING Capability | Status |
|---|---|---|
| Input parameter sensitivity | `jax.grad` through full simulation | Available now |
| Parameter sweep / ensemble | `run_sweep()` via `jax.vmap` | Available now |
| Numerical error estimation | Richardson extrapolation in `run_adaptive()` | Available now |
| Output uncertainty bounds | Manual via sweep + post-processing | Manual workflow |
| Bayesian inference | Not yet implemented | Planned (Tier 5) |
| Conformal prediction | Not yet implemented | Planned (Tier 5) |

#### 3. Require UQ-readiness in node documentation

Every node's algorithm document must include input parameter uncertainty sources, sensitivity characterization, and numerical error behavior.

#### 4. UQ interface hooks in code

See Section 9.4 for the code-level interface.

---

## 7. Contributor and Extensibility Documentation

### Case Study Patterns

**ITK** provides the most comprehensive contributor standards: `WriteAFilter.tex` and `CreateAModule.tex`, 18 files in `contributing/`, mandatory test baselines, and identical module anatomy.

**iMSTK** has a detailed coding guide with naming conventions and commit message conventions. They explicitly exempt "highly math-based code" from naming rules to maintain correspondence with published formulas.

### MADDENING Contributor Standards

#### `CONTRIBUTING.md` (Repository Root)

Top-level contributor guide covering: development environment setup, branching model, commit convention, code style, and links to detailed guides.

#### `docs/developer_guide/node_authoring.md`

Comprehensive guide for writing a new `SimulationNode`:

1. **The contract**: pure functions, JAX-traceability, parameters vs. state
2. **Directory structure**: where the file lives, naming conventions
3. **Required documentation**: docstring format + algorithm guide document
4. **Required tests**: unit tests + at least one verification benchmark
5. **Required metadata**: `NodeMeta` dataclass (Section 9.1)
6. **Required validation cases**: at least one registered benchmark (Section 9.3)
7. **Checklist**:

```markdown
### New Node Checklist

- [ ] `SimulationNode` subclass with `initial_state()` and `update()`
- [ ] `NodeMeta` metadata attached (algorithm ID, stability, references)
- [ ] NumPy-style docstring with Parameters, State fields, Boundary inputs
- [ ] Algorithm guide document in `docs/algorithm_guide/nodes/`
- [ ] Governing equations with LaTeX
- [ ] Assumptions and simplifications listed
- [ ] Validated physical regimes documented
- [ ] Known limitations and failure modes documented
- [ ] At least one registered verification benchmark
- [ ] Unit tests covering normal operation, edge cases, JAX-traceability
- [ ] CI passes on all platforms
- [ ] Entry in `docs/bibliography.bib` for primary reference
- [ ] Known limitations entered in `docs/validation/known_anomalies.yaml`
```

The final checklist item is new — it ensures that every known limitation is captured in the machine-readable anomaly registry from day one, which directly supports IEC 62304 SOUP assessment.

#### `docs/developer_guide/documentation_standards.md`

- **Docstring format**: NumPy-style (already used throughout codebase)
- **Math in docstrings**: ASCII-art equations in docstrings, LaTeX in Sphinx docs
- **Bibliography**: all academic references go in `docs/bibliography.bib`; cite using Pandoc-style `[@Key]` syntax in algorithm guide documents; CI validates all cited keys exist via `scripts/check_citations.py` (Section 3)
- **Math-heavy code exception**: mathematical variable names may use short names matching published formulas (e.g., `tau`, `f_eq`, `dx`)

#### `docs/developer_guide/testing_standards.md`

Test requirements at three levels:

1. **Unit tests** (mandatory): test each public method, verify JAX-traceability (jit, grad, vmap)
2. **Integration tests** (mandatory for nodes): test the node within a GraphManager
3. **Verification benchmarks** (mandatory for physics nodes): comparison to analytical solution or published reference data, registered via the validation test pattern (Section 9.3)

---

## 8. README and Repository-Level Documentation

### MADDENING README.md

```markdown
# MADDENING

Modular Automatic Differentiation and Data-Enhanced Neural-network
INteracting Graph.

[Badges: CI status, test count, coverage, PyPI version, license]

## What is MADDENING?

MADDENING is a JAX-based framework for soft real-time, high-fidelity
multiphysics simulation on HPC infrastructure. It manages a graph of
simulation nodes, where each node models a specific physical phenomenon
and edges represent coupling between them. Nodes can be reduced to neural
surrogates (PINNs) for accelerated inference, with training augmented by
experimental data.

MADDENING is designed as a standalone framework and as the computational
backbone of MIME (MIcrorobotics Multiphysics Engine).

## Intended Use and Disclaimers

> **MADDENING is research software.** It is not a medical device as
> defined by EU MDR (EU 2017/745) or US FDA regulations. It has no
> medical purpose. It is not intended for clinical use, clinical
> decision-making, or patient diagnosis. It has not been CE-marked,
> cleared, or approved by any regulatory body.
>
> When used as a component within regulated medical software,
> MADDENING is classified as SOUP (Software of Unknown Provenance)
> under IEC 62304. The device manufacturer is responsible for
> assessing MADDENING's suitability and performing all required
> verification and validation. See
> [Regulatory Documentation](docs/regulatory/) for details.

## Quick Start

[Installation, minimal example]

## Documentation

| Document | Description |
|----------|-------------|
| [User Guide](docs/user_guide/) | Installation, tutorials, concepts |
| [Algorithm Guide](docs/algorithm_guide/) | Mathematical docs per node/method |
| [Developer Guide](docs/developer_guide/) | Contributing, node authoring, testing |
| [Validation](docs/validation/) | V&V evidence, benchmarks, SOUP package |
| [Regulatory](docs/regulatory/) | Intended use, EU MDR/FDA guidance, IEC 62304 |
| [API Reference](docs/api_reference/) | Auto-generated API documentation |
| [DESIGN.md](DESIGN.md) | Architecture decisions |
| [ROADMAP.md](ROADMAP.md) | Feature roadmap |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

## Citation

If you use MADDENING in academic work, please cite:

[BibTeX entry]

## License

LGPL-3.0-or-later. See [LICENSE](LICENSE).
```

### Root-Level Governance Files

| File | Purpose |
|------|---------|
| `CITATION.cff` | Machine-readable citation metadata + configuration management artifact |
| `GOVERNANCE.md` | Decision-making process, maintainer roles |
| `CONTRIBUTING.md` | Quick-start for contributors |
| `SECURITY.md` | Vulnerability reporting (supports MDCG 2019-16 cybersecurity) |
| `CODE_OF_CONDUCT.md` | Community standards |

---

## 9. Code-Embedded Documentation and Structural Hooks

This section proposes concrete code-level mechanisms that make compliance, auditability, and documentation structural properties of the framework. Each subsection includes a design sketch sufficient for a developer to begin implementation.

### 9.1 Node and Edge Metadata Schemas

#### Problem

Currently, `SimulationNode` carries only `name`, `delta_t`, and `params`. There is no structured, machine-readable metadata about what algorithm a node implements, what physical regimes it has been validated for, or what its stability status is.

#### Design

```python
# maddening/core/metadata.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class StabilityLevel(Enum):
    """API/algorithm stability classification."""
    EXPERIMENTAL = "experimental"
    PROVISIONAL = "provisional"
    STABLE = "stable"
    DEPRECATED = "deprecated"


class UQReadiness(Enum):
    """UQ integration readiness."""
    NOT_READY = "not_ready"
    PARAMETER_SWEEP = "sweep"
    FULL = "full"


@dataclass(frozen=True)
class ValidatedRegime:
    """A quantitative bound on a validated parameter range."""
    parameter: str
    min_value: float
    max_value: float
    units: str = ""
    evidence: str = ""


@dataclass(frozen=True)
class Reference:
    """A citable academic or technical reference."""
    key: str                        # BibTeX key in bibliography.bib
    description: str = ""


@dataclass(frozen=True)
class NodeMeta:
    """Structured metadata for a SimulationNode.

    Machine-readable and automatically harvestable for V&V reports,
    changelogs, capability matrices, and IEC 62304 SOUP package documents.
    """
    algorithm_id: str               # e.g., "MADD-NODE-003"
    algorithm_version: str          # e.g., "1.0.0"
    stability: StabilityLevel = StabilityLevel.EXPERIMENTAL
    description: str = ""
    governing_equations: str = ""   # LaTeX string
    discretization: str = ""
    assumptions: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    validated_regimes: tuple[ValidatedRegime, ...] = ()
    references: tuple[Reference, ...] = ()
    uq_readiness: UQReadiness = UQReadiness.NOT_READY
    deprecation_notice: Optional[str] = None


@dataclass(frozen=True)
class EdgeMeta:
    """Structured metadata for an edge (coupling) specification."""
    coupling_type: str
    description: str = ""
    assumptions: tuple[str, ...] = ()
    references: tuple[Reference, ...] = ()
    stability: StabilityLevel = StabilityLevel.EXPERIMENTAL
```

#### Scope Distinction: `validated_regimes` vs. `hazard_hints`

Two `NodeMeta` fields carry risk-relevant information. Their scopes are non-overlapping:

| Field | Nature | Scope | Example | Consumer |
|---|---|---|---|---|
| `validated_regimes` | **Quantitative, parameter-bound** | Defines the envelope within which the node has been verified against analytical solutions or reference data. Each entry specifies a named parameter, a numeric min/max range, and the evidence source. | `ValidatedRegime("Re", 0, 100, evidence="Poiseuille analytical")` | Runtime parameter validators, downstream `HealthCheckNode` bound configs, algorithm guide "Validated Physical Regimes" table |
| `hazard_hints` | **Qualitative, non-parameter-bound** | Describes technical conditions that cannot be reduced to a single parameter range — algorithmic limitations, modelling assumptions, platform-specific issues, and failure modes that depend on combinations of parameters or on factors outside the simulation state. | `"Wall bounce-back assumes rigid walls; deformable walls not modelled"` | ISO 14971 hazard identification input, SOUP package "Known Limitations" section |

The distinction is sharp: if a risk can be expressed as "parameter X must be within [a, b]," it belongs in `validated_regimes`. If it cannot — because it involves a modelling assumption, a platform constraint, a multi-parameter interaction, or a qualitative limitation — it belongs in `hazard_hints`. A given risk should appear in exactly one of the two fields, never both. This prevents duplication and ensures that downstream consumers know exactly where to look for each type of information.

#### Integration with SimulationNode

```python
class SimulationNode(ABC):
    # ... existing interface ...
    meta: ClassVar[Optional[NodeMeta]] = None

    @classmethod
    def get_meta(cls) -> Optional[NodeMeta]:
        """Return structured metadata for this node type."""
        return cls.meta
```

#### Usage Example

```python
class HeatNode(SimulationNode):
    meta = NodeMeta(
        algorithm_id="MADD-NODE-002",
        algorithm_version="1.0.0",
        stability=StabilityLevel.STABLE,
        description="1D heat diffusion with explicit finite differences",
        governing_equations=r"\frac{\partial T}{\partial t} = \alpha \nabla^2 T + S",
        discretization="Explicit FD, 2nd-order central, 1st-order Euler time",
        assumptions=(
            "Uniform grid spacing",
            "Constant thermal diffusivity",
            "Dirichlet boundary conditions only",
            "1D geometry (rod)",
        ),
        limitations=(
            "CFL stability limit: dt < dx^2 / (2 * alpha)",
            "No adaptive mesh refinement",
            "No radiative or convective boundary conditions",
        ),
        validated_regimes=(
            ValidatedRegime("n_cells", 4, 1000, evidence="Convergence study"),
            ValidatedRegime("thermal_diffusivity", 1e-6, 1.0, "m^2/s",
                          "Analytical comparison"),
        ),
        references=(
            Reference("LeVeque2007", "Finite Differences for DEs"),
        ),
        uq_readiness=UQReadiness.PARAMETER_SWEEP,
    )
```

#### Automatic Harvesting

```python
def collect_node_metadata() -> dict[str, NodeMeta]:
    """Discover all SimulationNode subclasses and collect their metadata."""
    result = {}
    for cls in SimulationNode.__subclasses__():
        result[cls.__name__] = cls.meta
    return result
```

This enables automated generation of capability matrices, V&V status reports, changelog entries, deprecation reports, and IEC 62304 SOUP package content.

---

### 9.1.1 Unit Awareness and Edge Transforms

#### Problem

When nodes use different physical unit systems (e.g. LBM lattice units vs SI), the unit conversion applied on a coupling edge is invisible to inspection -- it is an opaque callable. A regulator reviewing the simulation graph cannot determine whether the conversion is correct, what units are expected on each side, or whether a conversion was accidentally omitted.

#### Design

Edges now support optional unit annotations alongside the existing ``transform`` callable:

- ``EdgeSpec.source_units: str | None`` -- physical units of the source field (e.g. ``"lattice"``, ``"Pa"``).
- ``EdgeSpec.target_units: str | None`` -- physical units after the transform is applied (e.g. ``"N"``).

Nodes declare expected units on their boundary inputs and flux outputs:

- ``BoundaryInputSpec.expected_units: str | None`` -- what units this input expects.
- ``BoundaryFluxSpec.output_units: str | None`` -- what units this flux output produces.

At ``compile()`` time, ``GraphManager.validate()`` checks for two classes of mismatch:

1. **Explicit mismatch**: edge declares ``target_units`` that differs from the target node's ``expected_units``.
2. **Missing transform**: edge has ``source_units`` different from ``expected_units`` but no ``transform`` is set.

Both produce ``WARNING``-level messages that surface via ``warnings.warn()`` without blocking compilation.

Standard LBM-to-SI conversion factories are provided in ``maddening.core.transforms_unit`` for common physical quantity conversions (force, torque, velocity, pressure, length). Each factory returns a JAX-traceable pure function suitable for ``EdgeSpec.transform``.

#### IEC 62304 Relevance

This directly supports **IEC 62304 Clause 5.4** (detailed design) traceability. A regulator reviewing a MADDENING simulation graph can now:

- See, at the edge level, what unit conversion is being applied.
- Verify that the declared units match the source node's output and the target node's expectation.
- Trace the conversion formula to the factory function in ``transforms_unit.py``.
- Detect missing or incorrect conversions via the compile-time validation warnings.

This creates a machine-readable audit trail of all unit conversions in a coupled simulation, which is essential for Class C SOUP assessment of multi-physics graphs that span different unit systems.

---

### 9.2 Provenance and Reproducibility Hooks

#### Problem

Regulatory audit trails require complete records of what was computed, with what parameters, using what software version, such that the computation can be reproduced exactly.

#### Design

```python
# maddening/core/provenance.py

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
import hashlib, json, jax, jax.numpy as jnp


@dataclass
class SimulationProvenance:
    """Complete, serializable record of a simulation run.

    Captures everything needed to reproduce the run exactly and to
    serve as an audit trail in a regulatory context. The
    maddening_version field satisfies IEC 62304 Clause 8 configuration
    identification requirements for SOUP traceability.
    """
    # Software identity
    maddening_version: str          # IEC 62304 Clause 8: SOUP identification
    jax_version: str
    jaxlib_version: str
    python_version: str
    platform: str

    # Graph topology
    graph_config: dict              # GraphManager.to_dict()
    node_metadata: dict[str, dict]

    # Execution parameters
    run_method: str
    n_steps: Optional[int] = None
    dt: Optional[float] = None
    external_inputs_hash: Optional[str] = None

    # Random state (JAX explicit PRNG)
    rng_seed: Optional[int] = None
    rng_key_state: Optional[tuple] = None

    # Initial conditions
    initial_state_hash: str = ""

    # Surrogate-specific
    surrogate_weights_hashes: dict[str, str] = field(default_factory=dict)
    training_data_hash: Optional[str] = None

    # Timestamps
    started_at: str = ""
    completed_at: str = ""

    # Reproducibility checksum
    final_state_hash: str = ""

    def to_dict(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str) -> "SimulationProvenance":
        with open(path) as f:
            return cls(**json.load(f))
```

#### Integration with GraphManager

```python
class GraphManager:
    def __init__(self, *, track_provenance: bool = False):
        self._track_provenance = track_provenance
        self._provenance: Optional[SimulationProvenance] = None

    # Provenance captured at run start, finalized at run end.
    # Opt-in: zero overhead for research users.
```

---

### 9.3 Validation Test Registration

#### Problem

Validation tests live in `tests/` alongside unit tests with no structural distinction. There is no way to automatically discover which tests are verification benchmarks.

#### Design

```python
# maddening/core/validation.py

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class BenchmarkType(Enum):
    ANALYTICAL = "analytical"
    MANUFACTURED = "manufactured"
    CONVERGENCE = "convergence"
    REFERENCE_DATA = "reference"
    REGRESSION = "regression"


@dataclass(frozen=True)
class ValidationBenchmark:
    """A registered verification benchmark for a node or algorithm."""
    benchmark_id: str
    node_class: str
    benchmark_type: BenchmarkType
    description: str
    reference: str
    tolerance: float
    test_function: str
    expected_order: Optional[float] = None


_BENCHMARK_REGISTRY: dict[str, ValidationBenchmark] = {}


def verification_benchmark(benchmark_id, node_class, benchmark_type,
                           description, reference="analytical",
                           tolerance=1e-3, expected_order=None):
    """Decorator that registers a test function as a verification benchmark."""
    def decorator(func):
        _BENCHMARK_REGISTRY[benchmark_id] = ValidationBenchmark(
            benchmark_id=benchmark_id, node_class=node_class,
            benchmark_type=benchmark_type, description=description,
            reference=reference, tolerance=tolerance,
            test_function=f"{func.__module__}.{func.__qualname__}",
            expected_order=expected_order,
        )
        return func
    return decorator
```

#### Usage

```python
@verification_benchmark(
    benchmark_id="MADD-VER-001",
    node_class="HeatNode",
    benchmark_type=BenchmarkType.ANALYTICAL,
    description="1D heat equation: steady-state linear profile",
    tolerance=1e-6,
)
def test_heat_steady_state_linear():
    ...
```

CI runs all registered benchmarks and auto-generates verification reports for `docs/validation/`.

---

### 9.4 UQ Interface Hooks

```python
# maddening/core/uq.py

from dataclasses import dataclass
from typing import Optional
from enum import Enum


class DistributionType(Enum):
    UNIFORM = "uniform"
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    TRUNCATED_NORMAL = "truncated_normal"
    FIXED = "fixed"


@dataclass(frozen=True)
class UncertainParameter:
    name: str
    distribution: DistributionType
    nominal: float
    lower: Optional[float] = None
    upper: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    units: str = ""
    source: str = ""
    sensitivity_rank: Optional[int] = None


@dataclass(frozen=True)
class UncertaintySpec:
    parameters: tuple[UncertainParameter, ...]
    notes: str = ""
```

Nodes expose `uncertainty_spec() -> Optional[UncertaintySpec]`. UQ wrappers use this to sample parameters via `jax.vmap` without modifying node internals.

---

### 9.5 Deprecation and Stability Machinery

```python
# maddening/core/stability.py

import functools, warnings
from maddening.core.compliance.metadata import StabilityLevel

_STABILITY_REGISTRY: dict[str, dict] = {}

def stability(level: StabilityLevel, since: str = "",
              notice: str = "", alternative: str = ""):
    """Decorator marking a class/function with a stability level.

    Registers in global registry (machine-readable for automated
    stability reports) and emits runtime warnings for deprecated/
    experimental items.
    """
    def decorator(obj):
        qualname = f"{obj.__module__}.{obj.__qualname__}"
        _STABILITY_REGISTRY[qualname] = {
            "level": level, "since": since,
            "notice": notice, "alternative": alternative,
        }
        if level == StabilityLevel.DEPRECATED and isinstance(obj, type):
            original_init = obj.__init__
            @functools.wraps(original_init)
            def warned_init(self, *args, **kwargs):
                warnings.warn(
                    f"{obj.__name__} deprecated since {since}. {notice}",
                    DeprecationWarning, stacklevel=2)
                return original_init(self, *args, **kwargs)
            obj.__init__ = warned_init
        return obj
    return decorator

def generate_stability_report() -> str:
    """Generate Markdown stability report from registry."""
    ...
```

---

### 9.6 Audit Logging Interface

```python
# maddening/core/audit.py

from typing import Any, Optional, Protocol
from datetime import datetime, timezone
import json


class AuditSink(Protocol):
    def write_event(self, event: dict) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


class NullSink:
    """No-op sink. Zero overhead when auditing is off."""
    def write_event(self, event: dict) -> None: pass
    def flush(self) -> None: pass
    def close(self) -> None: pass


class JSONFileSink:
    """Writes audit events as JSON Lines."""
    def __init__(self, path: str):
        self._file = open(path, "a")
    def write_event(self, event: dict) -> None:
        self._file.write(json.dumps(event, default=str) + "\n")
    def flush(self) -> None: self._file.flush()
    def close(self) -> None: self._file.close()


class AuditLogger:
    """Lightweight audit logger. Activated by downstream tools when
    regulatory audit trails are needed. Uses NullSink by default."""

    def __init__(self, sink: Optional[AuditSink] = None):
        self._sink = sink or NullSink()
        self._session_id = None

    def start_session(self, session_id, metadata=None): ...
    def end_session(self): ...
    def log_step(self, step_number, sim_time, state_hash=""): ...
    def log_surrogate_swap(self, node, surrogate_class, weights_hash=""): ...
    def log_parameter_change(self, node, param, old_value, new_value): ...
    def observer_callback(self, event, data):
        """GraphManager observer callback for automatic logging."""
        ...
```

---

### 9.7 Anomaly Management System

#### Problem

IEC 62304 Clause 7.1.3 requires that when SOUP is used in a medical device, the manufacturer must "evaluate the list of known anomalies of the SOUP item" and determine whether any are "unacceptable relative to the safety of the device." This means MADDENING must maintain a structured, versioned list of known anomalies that downstream tools can assess for safety impact.

IEC 62304 Clause 9 (Software problem resolution process) further requires that problems be reported, investigated, and resolved with documented rationale. While MADDENING is not itself subject to IEC 62304, proactively maintaining this documentation dramatically reduces the burden on downstream manufacturers.

**Class III context**: For MICROBOTICA's Class III use cases, the known anomalies registry is a **Notified Body-facing artifact**. The Notified Body will review it as part of the full technical file assessment. Every entry must be complete, internally consistent, and defensible under external scrutiny. The safety relevance rationale must be specific enough that a reviewer unfamiliar with the codebase can understand why the assessment was made.

#### Design

##### Anomaly Record Schema

```python
# maddening/core/anomaly.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import date


class AnomalySeverity(Enum):
    """Severity classification aligned with IEC 62304 problem reporting."""
    CRITICAL = "critical"   # Causes incorrect results with no workaround
    MAJOR = "major"         # Causes incorrect results but has workaround
    MINOR = "minor"         # Cosmetic or minor inconvenience
    ENHANCEMENT = "enhancement"  # Not a defect; a feature request


class SafetyRelevance(Enum):
    """Assessment of whether the anomaly could affect safety-critical
    computations in a downstream medical device context."""
    SAFETY_RELEVANT = "safety_relevant"
    NOT_SAFETY_RELEVANT = "not_safety_relevant"
    CONTEXT_DEPENDENT = "context_dependent"  # Depends on downstream COU


class ResolutionStatus(Enum):
    OPEN = "open"
    WORKAROUND_AVAILABLE = "workaround"
    FIXED = "fixed"
    WONT_FIX = "wont_fix"          # By design or out of scope
    DEFERRED = "deferred"


@dataclass
class AnomalyRecord:
    """A known anomaly in MADDENING.

    This record is the atomic unit of the known anomalies registry.
    Downstream IEC 62304-compliant projects evaluate each record for
    safety impact in their specific context of use.
    """
    anomaly_id: str                     # e.g., "MADD-ANO-001"
    title: str                          # Short description
    description: str                    # Detailed description
    affected_components: tuple[str, ...]  # e.g., ("HeatNode", "LBMPipeNode")
    affected_versions: str              # e.g., "0.1.0 – current"
    severity: AnomalySeverity
    safety_relevance: SafetyRelevance
    safety_relevance_rationale: str     # Why this assessment was made
    workaround: str = ""                # If available
    resolution_status: ResolutionStatus = ResolutionStatus.OPEN
    resolved_in_version: Optional[str] = None
    github_issue: Optional[str] = None  # e.g., "#42"
    date_reported: Optional[str] = None # ISO 8601 date
    date_resolved: Optional[str] = None
```

##### YAML Registry Format

The known anomalies registry lives at `docs/validation/known_anomalies.yaml`:

```yaml
# MADDENING Known Anomalies Registry
# Updated with each release per IEC 62304 SOUP anomaly assessment requirements
# Schema version: 1.0
# Last updated: 2025-XX-XX
# MADDENING version: 0.1.0

schema_version: "1.0"
maddening_version: "0.1.0"
generated_date: "2025-XX-XX"

anomalies:
  - anomaly_id: "MADD-ANO-001"
    title: "LBM D3Q19 lattice segfault on CUDA 12.2 with jaxlib 0.5.1"
    description: >
      The LBMPipeNode crashes with a segmentation fault when running on
      CUDA 12.2 with jaxlib 0.5.1 due to an XLA compilation bug with
      certain einsum patterns. The CPU backend is unaffected.
    affected_components: ["LBMPipeNode"]
    affected_versions: "0.1.0 – current"
    severity: "critical"
    safety_relevance: "context_dependent"
    safety_relevance_rationale: >
      Only affects GPU execution path. CPU execution produces correct
      results. Downstream tools must verify their execution platform.
    workaround: "Use CPU backend (JAX_PLATFORMS=cpu) or jaxlib >= 0.5.2"
    resolution_status: "workaround"
    github_issue: null
    date_reported: "2025-01-15"

  - anomaly_id: "MADD-ANO-002"
    title: "HeatNode CFL stability not enforced at runtime"
    description: >
      HeatNode does not check whether the user-provided timestep
      satisfies the CFL stability condition dt < dx^2 / (2*alpha).
      Unstable timesteps silently produce incorrect results.
    affected_components: ["HeatNode"]
    affected_versions: "0.1.0 – current"
    severity: "major"
    safety_relevance: "safety_relevant"
    safety_relevance_rationale: >
      Silently incorrect results from an unstable numerical scheme
      could propagate to downstream safety-critical computations
      if the timestep is not validated by the caller.
    workaround: "Manually verify dt < dx^2 / (2*alpha*d) before running"
    resolution_status: "open"
    github_issue: null
    date_reported: "2025-01-20"
```

##### GitHub Issues Label Taxonomy

| Label | Color | Meaning |
|-------|-------|---------|
| `anomaly:critical` | Red | Incorrect results, no workaround |
| `anomaly:major` | Orange | Incorrect results, workaround available |
| `anomaly:minor` | Yellow | Cosmetic or minor issue |
| `safety-relevant` | Purple | Could affect safety-critical downstream use |
| `soup-assessment` | Blue | Relevant to IEC 62304 SOUP evaluation |
| `known-anomaly` | Gray | Tracked in known_anomalies.yaml |

##### Anomaly Lifecycle: Three-Phase Model

Anomalies follow a three-phase lifecycle that satisfies IEC 62304 Clause 8 (configuration management) and Clause 9 (problem resolution). The key design decision is that the transition from Phase 1 to Phase 2 is a **deliberate, manual sign-off** — the YAML registry is the authoritative compliance artifact, and entries are never auto-generated from issues.

**Phase 1 — Discovery (GitHub Issue)**

An anomaly is first reported as a GitHub Issue using the structured issue template at `.github/ISSUE_TEMPLATE/anomaly.md`. The template enforces mandatory fields:

```yaml
name: Known Anomaly Report
description: Report a known anomaly for tracking in the compliance registry
labels: ["known-anomaly"]
body:
  - type: input
    id: title
    attributes:
      label: Anomaly title
    validations:
      required: true
  - type: dropdown
    id: severity
    attributes:
      label: Severity
      options: [critical, major, minor, enhancement]
    validations:
      required: true
  - type: dropdown
    id: safety_relevance
    attributes:
      label: Safety relevance
      options: [safety_relevant, not_safety_relevant, context_dependent]
    validations:
      required: true
  - type: textarea
    id: safety_rationale
    attributes:
      label: Safety relevance rationale
      description: >
        Why this severity and safety relevance were chosen. Must be
        understandable by someone unfamiliar with the codebase.
    validations:
      required: true
  - type: input
    id: affected_components
    attributes:
      label: Affected components
      description: "Comma-separated list (e.g., HeatNode, LBMPipeNode)"
    validations:
      required: true
  - type: input
    id: affected_versions
    attributes:
      label: Affected versions
      description: "e.g., 0.1.0 – current"
    validations:
      required: true
  - type: textarea
    id: workaround
    attributes:
      label: Workaround (if any)
```

The issue is the engineering discussion space: reproduction steps, root cause analysis, fix proposals, and technical debate happen here. The issue is labeled `known-anomaly` and triaged with the appropriate severity labels.

**Phase 2 — Formalization (YAML entry)**

When the anomaly is sufficiently understood, a developer **manually** creates or updates the corresponding entry in `known_anomalies.yaml`. This manual step is intentional — it is a regulatory sign-off act:

- The developer reviews the issue discussion and distils it into a precise, Notified-Body-readable record
- The severity and safety relevance assessments are confirmed (they may have changed during investigation)
- The `safety_relevance_rationale` is written for an external reviewer, not for internal consumption
- The `github_issue` field links back to the issue for full engineering context

This transition must not be automated. An automated pipeline that converts issues to YAML entries would bypass the human judgement required for a compliance artifact. The YAML registry is what a Notified Body reads; the GitHub issue is the engineering evidence behind it.

**Phase 3 — Verification (CI consistency check)**

CI enforces consistency between the two artifacts:

- Every issue labeled `known-anomaly` that has been open for more than one release cycle should have a corresponding YAML entry (CI warning, not blocking — some issues may still be in Phase 1 investigation)
- Every YAML entry with a `github_issue` field must reference a valid, existing issue
- No YAML entries may be silently deleted (only `resolution_status` changes are permitted)
- The YAML schema is validated on every push

The three-phase model ensures that engineering discussion (Phase 1) and regulatory documentation (Phase 2) are clearly separated, with CI (Phase 3) enforcing consistency without automating the regulatory judgement.

##### CI Integration and Release Gate Harmony

CI enforcement must distinguish between anomalies that are still under investigation (Phase 1) and those that have been resolved without proper formalization. The guiding principle: **never block active investigation, always block incomplete compliance**.

**On every push (CI check):**

1. The YAML schema is validated (`schema_version`, required fields, valid enums, unique IDs)
2. Every YAML entry with a `github_issue` field must reference a valid, existing issue
3. No YAML entries have been silently deleted (only `resolution_status` changes are permitted)
4. The `maddening_version` field in `known_anomalies.yaml` matches the current release

**CI warnings (non-blocking):**

5. Open issues labeled `known-anomaly` that have been open for more than one release cycle but do not yet have a corresponding YAML entry — these are Phase 1 anomalies still under investigation, and blocking CI would penalise active triage

**Release gate (blocking — enforced before `git tag`), three tiers:**

The release gate uses a three-tier model that allocates clerical effort proportionally to safety impact. A solo developer should not spend equal formalization effort on a cosmetic numerical precision note and a known safety-relevant instability.

**Tier 1 — `safety-relevant` label (any severity):** Hard gate, no grace period. Any closed issue labeled `safety-relevant` must have a corresponding YAML entry before the release is tagged. No exceptions. This tier exists because safety-relevant anomalies are Notified Body-facing artifacts for Class III — they must be formalized immediately upon resolution.

**Tier 2 — `anomaly:critical` or `anomaly:major` (without `safety-relevant`):** No grace period. Must be formalized in YAML before the release in which they are closed. These are significant anomalies that affect numerical correctness and must be visible to downstream SOUP assessors, even if their safety relevance is "not_safety_relevant" or "context_dependent."

**Tier 3 — `anomaly:minor` (without `safety-relevant`):** Two-cycle grace period. May remain in Phase 1 (GitHub Issue only) for up to two release cycles after being closed before a YAML entry is required. CI emits a warning after one release cycle ("minor anomaly closed in previous release without YAML formalization"). The gate blocks only at the second release after closure. This prevents minor cosmetic or documentation anomalies from creating a formalization bottleneck, while still ensuring they are eventually captured.

6. Tier 1 gate: every **closed** issue labeled `safety-relevant` must have a YAML entry — no grace period
7. Tier 2 gate: every **closed** issue labeled `anomaly:critical` or `anomaly:major` (without `safety-relevant`) must have a YAML entry — no grace period
8. Tier 3 warning (one cycle): every **closed** issue labeled `anomaly:minor` (without `safety-relevant`) that was closed in the previous release cycle but lacks a YAML entry — CI warning
9. Tier 3 gate (two cycles): every **closed** issue labeled `anomaly:minor` (without `safety-relevant`) that was closed two or more release cycles ago without a YAML entry — CI blocks

This three-tier model prevents the following failure modes: (a) CI blocks that discourage developers from filing anomaly issues in the first place, (b) issues closed as "fixed" without a corresponding compliance record, (c) releases shipped with unformalized safety-relevant anomalies, and (d) disproportionate clerical burden on minor issues that slows development velocity without meaningful compliance benefit.

**Automated cycle counting**: The Tier 3 grace period tracking must be automated by `scripts/check_anomalies.py` — manual tracking of how many release cycles a specific issue has been open is an unsustainable clerical burden for a solo developer. The script must compare each closed `anomaly:minor` issue's close date (via the GitHub API) against the release timestamps in `CHANGELOG.md` or git tags. If the issue was closed more than one release cycle ago without a YAML entry, the script emits a warning. If two or more release cycles have passed, the script fails the release gate. This automation is a prerequisite for the three-tier model to function; without it, Tier 3 grace periods will be tracked inconsistently or not at all.

```python
# scripts/check_anomalies.py (CI validation script)

import yaml
import sys

def validate_anomalies(path="docs/validation/known_anomalies.yaml"):
    with open(path) as f:
        data = yaml.safe_load(f)

    errors = []
    ids_seen = set()
    for a in data.get("anomalies", []):
        # Uniqueness
        if a["anomaly_id"] in ids_seen:
            errors.append(f"Duplicate anomaly_id: {a['anomaly_id']}")
        ids_seen.add(a["anomaly_id"])

        # Required fields
        for field in ("anomaly_id", "title", "description",
                      "severity", "safety_relevance",
                      "safety_relevance_rationale"):
            if not a.get(field):
                errors.append(f"{a['anomaly_id']}: missing {field}")

        # Valid enums
        if a["severity"] not in ("critical", "major", "minor", "enhancement"):
            errors.append(f"{a['anomaly_id']}: invalid severity")

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"OK: {len(ids_seen)} anomalies validated")
```

##### Release Process Integration

The release checklist should include:

1. Review all open GitHub issues labeled `known-anomaly`
2. Update `known_anomalies.yaml` with any new anomalies or resolution status changes
3. Update the `maddening_version` and `generated_date` fields
4. Run `scripts/check_anomalies.py` to validate
5. Include a "Known Anomalies" section in the release notes

---

### 9.8 ISO 14971 Risk Management Hooks

#### Problem

ISO 14971 is the risk management standard that any downstream commercial manufacturer will use to build their risk management file. MADDENING can make that process substantially easier by structuring certain information explicitly for risk management consumption, rather than requiring risk managers to extract it from algorithm documentation and anomaly registries themselves.

#### How NodeMeta Feeds Into ISO 14971 Hazard Identification

The existing `NodeMeta` fields already contain risk-relevant information:

- **`NodeMeta.limitations`**: Each limitation is a potential failure mode. "CFL stability limit: dt < dx^2 / (2 * alpha)" means that violating this condition produces silently incorrect results — a failure mode that must appear in the manufacturer's hazard analysis.
- **`NodeMeta.validated_regimes`**: The boundary of each validated regime defines the region outside which the model's behaviour is uncharacterised. Operating outside a validated regime is a hazard contributor that the manufacturer must assess. These are **quantitative, parameter-bound** risks (see Section 9.1 scope distinction).
- **`NodeMeta.assumptions`**: Each assumption defines a condition that must hold for the model to be valid. Violation of an assumption (e.g., "Incompressible flow" applied to a compressible scenario) is a potential hazard source.
- **`NodeMeta.hazard_hints`**: Qualitative, non-parameter-bound technical conditions that the risk manager should evaluate. These complement `validated_regimes` with non-overlapping scope — see the scope distinction table in Section 9.1.

This connection should be documented explicitly in `docs/regulatory/eu_mdr_guidelines.md` and in the SOUP package document, so that risk managers performing ISO 14971 hazard identification know where to look.

#### Technical Hazard Hints vs. Clinical Risk Assessments

`hazard_hints` provides technical hazard hints only. Each hint describes a condition under which MADDENING's numerical behaviour is uncharacterised, unstable, or known to produce incorrect results. These are technical conditions, not clinical risk assessments. MADDENING does not assess the clinical significance of any condition it documents — that is the sole responsibility of the downstream commercial manufacturer under ISO 14971. The transition from a technical hint (e.g., "behaviour uncharacterised at Re > 100") to a clinical hazard (e.g., "risk of incorrect drug delivery calculation") requires the manufacturer to supply the clinical context, device-specific scenario, patient population, and ISO 14971 probability/severity estimation that MADDENING cannot provide.

#### Design: HazardHint Field

Add an optional `hazard_hints` field to `NodeMeta` — structured inputs to a manufacturer's hazard analysis.

**The technical-vs-clinical boundary**: MADDENING's hazard hints describe **technical conditions** only — numerical instability thresholds, unvalidated parameter regimes, algorithmic limitations, and known failure modes. They are inputs to ISO 14971 hazard *identification* (Clause 5.4), not hazard *evaluation* (Clause 5.5). A hazard hint such as "behaviour uncharacterised at Re > 100" tells the risk manager that operating above Re 100 is a candidate hazard contributor; it does not assess the probability that a patient is harmed or the severity of that harm. The clinical risk estimation (probability × severity), risk evaluation (acceptable / unacceptable), and risk control measures are the manufacturer's sole responsibility under ISO 14971. This boundary is deliberate and non-negotiable: a general-purpose computation library cannot make clinical risk judgements because it does not know the clinical context. Blurring this boundary — e.g., by labelling a hazard hint "high risk" — would be both technically incorrect (risk depends on context of use) and regulatorily dangerous (it would imply MADDENING is performing risk assessment, which is a manufacturer obligation).

```python
# Addition to maddening/core/metadata.py

@dataclass(frozen=True)
class NodeMeta:
    # ... existing fields ...

    hazard_hints: tuple[str, ...] = ()
    """Technical hazard hints for downstream ISO 14971 hazard identification.

    Each string describes a **technical condition** — numerical instability,
    unvalidated parameter regime, algorithmic limitation — that a risk
    manager should consider as a potential hazard contributor. These are
    strictly technical hazard hints, NOT clinical risk assessments.

    MADDENING provides: technical conditions (e.g., "CFL > 1 causes
    numerical instability", "behaviour uncharacterised at Re > 100").

    MADDENING does NOT provide: clinical risk assessments (e.g.,
    "could cause patient harm", "risk of neurological damage"). Clinical
    risk estimation (probability × severity) and risk evaluation
    (acceptable / unacceptable) are the manufacturer's sole
    responsibility under ISO 14971.

    Examples:
        ("Behaviour uncharacterised at Re > 100",
         "No turbulence model — do not use above laminar-turbulent transition",
         "Wall bounce-back assumes rigid walls; deformable walls not modelled")
    """
```

#### Usage Example

```python
class LBMPipeNode(SimulationNode):
    meta = NodeMeta(
        algorithm_id="MADD-NODE-006",
        algorithm_version="1.0.0",
        stability=StabilityLevel.EXPERIMENTAL,
        description="D3Q19 Lattice Boltzmann in cylindrical pipe with propeller actuator",
        # ... existing fields ...
        hazard_hints=(
            "Behaviour uncharacterised at Re > 100; validated only for laminar flow",
            "No turbulence model — do not use above laminar-turbulent transition (Re ~2000)",
            "Wall bounce-back assumes rigid, impermeable walls; deformable or porous walls not modelled",
            "Gravity applied uniformly — no spatially varying body forces",
            "CUDA 12.2 + jaxlib 0.5.1 causes segfault on GPU (MADD-ANO-001); CPU unaffected",
            "Passive scalar tracer uses D3Q7 with separate tau — accuracy degrades at high Peclet number",
        ),
    )
```

#### Automatic Harvesting for Risk Management

```python
def collect_hazard_hints() -> dict[str, list[str]]:
    """Collect all hazard hints across all nodes for risk management input."""
    result = {}
    for cls in SimulationNode.__subclasses__():
        if cls.meta and cls.meta.hazard_hints:
            result[cls.__name__] = list(cls.meta.hazard_hints)
    return result
```

This enables automated generation of a "MADDENING Hazard Input Summary" that risk managers can use as a starting point for their hazard identification.

#### Functional Purity as a Risk Management Asset

MADDENING's JAX-based functional architecture is a meaningful safety argument that downstream manufacturers should document explicitly in their architectural risk analysis:

- **No hidden state**: MADDENING nodes are pure functions — `update(state, boundary_inputs, dt) -> new_state`. There is no mutable state that could silently accumulate errors across timesteps.
- **No side effects**: The update function cannot modify external state, write to files, or communicate with external systems. Its output depends solely on its inputs.
- **Deterministic outputs**: For identical inputs (modulo floating-point non-determinism across platforms), MADDENING produces identical outputs. Failure modes are reproducible and testable.
- **Explicit data flow**: The graph architecture makes all data dependencies visible. There are no implicit couplings or hidden channels between nodes.

These properties mean that MADDENING's failure modes are observable, reproducible, and testable — not dependent on runtime state that cannot be examined. This supports segregation arguments (IEC 62304 Clause 5.3.5) and simplifies failure mode analysis (ISO 14971).

**The XLA Shadow — qualification of the functional purity argument**: The functional architecture described above protects the **data-flow layer**: given the same inputs, MADDENING's Python-level logic produces the same outputs, with no hidden state or side effects. However, this guarantee does not extend to the **execution layer**. MADDENING's pure functions are compiled by JAX's XLA compiler and executed on CPU or GPU hardware. XLA compiler bugs (such as the segfault in MADD-ANO-001), GPU hardware errors (silent data corruption from faulty VRAM, cosmic-ray bit flips, thermal throttling artefacts), and floating-point non-determinism across platforms can all produce incorrect results that MADDENING's functional architecture cannot detect or prevent.

This means there are two distinct failure domains:

| Layer | What MADDENING Guarantees | What MADDENING Cannot Guarantee |
|---|---|---|
| **Data-flow** (Python/JAX logic) | Pure functions, no hidden state, deterministic logic, explicit data flow | — |
| **Execution** (XLA compiler + hardware) | — | Correct compilation, correct hardware execution, cross-platform bit-identical results |

**Manufacturer obligation**: The downstream commercial manufacturer must implement **execution-layer defences** that MADDENING's functional purity cannot provide. These are listed in priority order — lightweight, per-step monitors first; expensive offline checks second:

**Primary monitors (lightweight, every-step):**

- **NaN / Inf trapping**: check every output state array for non-finite values after each step. This is the single most effective execution-layer defence — NaN propagation is the dominant observable symptom of XLA compiler bugs, hardware memory errors, and numerical instability. Cost: one `jnp.isfinite()` call per state field per step.
- **Physical boundary violation checks**: verify that key output quantities remain within physically plausible bounds (e.g., density > 0, temperature > 0 K, concentration ∈ [0, 1], velocity < speed of sound). These bounds are domain-specific and must be defined by the manufacturer for their context of use, but MADDENING's `NodeMeta.validated_regimes` (Section 9.1) provides the quantitative bounds that inform them.
- **Statistical moment monitoring**: track running mean and variance of key output fields; flag sudden jumps (e.g., mean density changing by more than 10% in one step) that indicate a simulation entering an unphysical regime. Unlike bound checks, moment monitors detect *drift* — a slowly diverging simulation that remains within bounds individually but is clearly non-physical in aggregate.

**Secondary monitors (expensive, periodic or offline):**

- **CPU / GPU cross-validation**: run the same simulation on both CPU and GPU backends and compare results within a tolerance; divergence indicates an execution-layer fault. This is the gold-standard execution-layer check but is expensive (2× compute cost) and should be used for critical validation runs, not routine production. Frequency: at commissioning, after platform changes, and periodically during operation. **Intraoperative feasibility note**: full CPU/GPU cross-validation is **not viable for real-time intraoperative use**. Re-running a high-fidelity simulation (e.g., 3D LBM at clinical resolution) on a second backend within surgical time constraints is computationally infeasible. The primary monitoring suite — NaN/Inf trapping, physical boundary checks, and statistical moment monitoring via `HealthCheckNode` — is the practical safeguard for intraoperative contexts. CPU/GPU cross-validation is appropriate for offline pre-operative planning, hardware qualification testing, and development regression testing. Documentation and compliance arguments must not present cross-validation as a feasible intraoperative safeguard, as this would set a compliance bar the manufacturer cannot meet.
- **Hardware qualification**: document the specific CPU/GPU hardware, driver versions, JAX version, and jaxlib version used in production; maintain a tested platform matrix and requalify after upgrades.

#### HealthCheck Node Infrastructure

MADDENING provides a **HealthCheck base node** — a `SimulationNode` subclass designed to be instantiated and configured by downstream libraries. The HealthCheck node participates in the graph as a regular node: it receives state from monitored nodes via edges, performs the checks described above, and writes diagnostic outputs (pass/fail flags, check values) into its own state. This means health monitoring is part of the JIT-compiled graph step and benefits from the same functional purity guarantees as physics nodes.

```python
# maddening/nodes/health_check.py (design sketch)

class HealthCheckNode(SimulationNode):
    """Base health monitor that downstream libraries configure.

    Receives state from one or more monitored nodes via edges.
    Performs configurable checks and writes results to its own state.

    Parameters
    ----------
    checks : dict
        Mapping of field names to check configurations:
        {"density": {"finite": True, "min": 0.0, "max": 10.0},
         "velocity": {"finite": True, "max_abs": 100.0}}
    """

    meta = NodeMeta(
        algorithm_id="MADD-NODE-HEALTH-001",
        algorithm_version="1.0.0",
        stability=StabilityLevel.EXPERIMENTAL,
        description="Base health monitor for execution-layer fault detection",
        assumptions=(
            "Monitored state is received via boundary inputs (edges)",
            "Check results are outputs only — halting logic belongs to the application layer",
        ),
        limitations=(
            "Cannot detect errors that produce in-range but incorrect values",
            "Moment checks require historical state — first step has no baseline",
        ),
        hazard_hints=(
            "Reliability depends on using numerical primitives independent of the "
            "monitored node — if both share the same XLA compilation path, a compiler "
            "bug may corrupt both simultaneously (see Section 9.8 Algorithmic Diversity "
            "Principle)",
            "Does not halt the simulation — check results are outputs only; halting "
            "logic belongs to the application layer",
            "Cannot detect errors that occur before the monitored state is passed "
            "via boundary inputs",
        ),
    )

    def update(self, state, boundary_inputs, dt):
        # For each monitored field received via boundary_inputs:
        #   1. NaN/Inf check (if configured)
        #   2. Bound check (if configured)
        #   3. Moment check (if configured and historical state available)
        # Returns state with check results (all_passed, per_field_flags, values)
        ...
```

**Algorithmic Diversity Principle** (mandatory for Class C contexts): HealthCheck monitors **must** use fundamentally different numerical primitives than the physics nodes they monitor. For example, a mass conservation check for an LBM node must compute total mass via a simple `jnp.sum` over the density field rather than anything derived from the LBM distribution functions or stencil weights. A velocity magnitude check must use `jnp.linalg.norm` rather than any quantity computed by the monitored node's `update()` function. The rationale is correlated failure risk: if a physics node and its HealthCheck share the same JIT compilation path — the same XLA-lowered HLO operations, potentially fused by the same compiler optimisation passes — a single XLA compiler bug could corrupt both the physics output and the health monitor simultaneously, causing the monitor to report "pass" on corrupted data. Using independent primitives reduces this correlation by ensuring the HealthCheck's computation follows a distinct compilation path. This principle is a **mandatory design requirement** for any `HealthCheckNode` implementation used in a Class C software context (IEC 62304 Clause 5.3.5) — not merely a recommendation. MADDENING cannot enforce this at runtime (the ABC cannot inspect what primitives a subclass's `update()` calls), but it is enforceable at the review and audit level. The first `hazard_hint` in the `NodeMeta` above documents it as a structured warning, and the following CI-checkable heuristic provides a concrete verification strategy:

**CI-checkable heuristic**: A `HealthCheckNode` subclass's `update()` function must not call any function that is also called by the monitored node's `update()` function, excluding JAX/NumPy fundamentals (`jnp.sum`, `jnp.max`, `jnp.isfinite`, etc.) that are so primitive they do not share meaningful compilation paths with domain-specific physics code. This can be verified by static analysis (AST inspection of function call targets) or by code review checklist. The heuristic is not perfect — two functions may call `jnp.dot` for entirely different purposes — but it catches the most dangerous case: a HealthCheck that reuses the monitored node's internal helper functions or stencil operations.

The HealthCheck node is a **schema-layer** contribution: MADDENING provides the base class and the graph integration; the downstream library provides the configuration (which fields to monitor, what bounds to enforce, what moment thresholds to use). The manufacturer's risk management file documents which checks are active and why, connecting back to ISO 14971 risk control measures.

The HealthCheck node does **not** halt the simulation — halting is a control decision that belongs to the downstream application layer, not to a graph node. The node's output state contains check results that the application layer can act on (halt, warn, log, switch to fallback). This separation preserves MADDENING's role as a computation library that does not make clinical decisions.

These are the manufacturer's responsibility — MADDENING provides the mathematical guarantees and the monitoring infrastructure; the manufacturer provides the execution-layer configuration and response logic. This boundary is the same one described in the two-mode verification inheritance model (Section 16): even when upstream mathematical verification is inherited, the execution-layer responsibility always belongs to the downstream manufacturer.

#### Anomaly Registry as ISO 14971 Input

Each anomaly in `known_anomalies.yaml` with `safety_relevance: context_dependent` is a candidate hazard contributor that the downstream manufacturer must assess. The workflow is:

1. The manufacturer reviews all `context_dependent` anomalies in `known_anomalies.yaml`
2. For each, the `safety_relevance_rationale` provides the technical description needed for a hazard analysis entry
3. The manufacturer provides the risk estimation (probability × severity) and risk evaluation (acceptable / unacceptable) for their specific COU
4. Unacceptable risks require risk control measures — these are the manufacturer's responsibility
5. The completed assessment is documented in the manufacturer's risk management file (ISO 14971)

MADDENING provides the technical input; the manufacturer provides the clinical context and risk judgement.

---

### 9.9 Usability Engineering Considerations

#### Scope

IEC 62366-1:2015 (Application of usability engineering to medical devices) applies to the commercial manufacturer's product, not to MADDENING itself. MADDENING is not a medical device and is not subject to usability engineering requirements. However, MADDENING's API design, parameter handling, and validation behaviour directly affect the **use error risks** that the downstream manufacturer must assess under IEC 62366-1.

The relevant use error class here is not end-user misoperation of a clinical interface — that is the commercial manufacturer's concern. The relevant use error class is a **developer or integrator misconfiguring a simulation**: selecting the wrong timestep, setting parameters outside the validated regime, choosing an inappropriate node type for the physical regime, or using a node in a UQ workflow it does not support. These are genuine hazard contributors that the manufacturer must assess in their usability engineering file, and MADDENING's design choices can either mitigate or exacerbate these risks.

#### MADDENING Design Choices That Reduce Use Error Risk

The following design elements in MADDENING are usability engineering assets that downstream commercial manufacturers should document in their technical file:

**1. Runtime parameter validation**

Nodes that enforce runtime checks on parameter validity — such as CFL stability conditions, Reynolds number bounds, or positivity constraints — prevent a class of use errors entirely. Anomaly MADD-ANO-002 (HeatNode CFL stability not enforced at runtime) is an explicit example of the risk when this validation is absent: unstable timesteps silently produce incorrect results. The fix for MADD-ANO-002 — adding runtime CFL checks — is a direct, positive usability engineering contribution.

MADDENING should progressively add runtime validation to nodes. This is not a regulatory obligation — MADDENING is not subject to IEC 62366-1 — but it is good software engineering that reduces use error risk for all downstream tools. Where a node can detect that its parameters violate a known stability or validity condition, it should raise a clear error or warning rather than producing silently incorrect results. Each such validation, once implemented, strengthens the usability engineering argument in any commercial product's technical file.

**2. NodeMeta.validated_regimes**

The `validated_regimes` field (Section 9.1) provides machine-readable quantitative bounds on the parameter ranges within which a node has been verified. A well-designed commercial integration layer can use these bounds to warn operators when parameters approach or exceed validated limits, or to refuse execution outside validated regimes entirely. This transforms what would otherwise be a silent use error (operating outside validated bounds) into an explicit, preventable condition.

**3. NodeMeta.hazard_hints**

The `hazard_hints` field (Section 9.8) provides structured warnings that a commercial product's integration layer should surface to the operator. A well-designed UI can present these hints contextually — for example, displaying "Behaviour uncharacterised at Re > 100" when the operator configures a simulation that will exceed this regime. This supports the IEC 62366-1 principle of designing user interfaces that prevent foreseeable misuse.

**4. UQReadiness flag**

The `uq_readiness` field (Section 9.1) communicates whether a node supports uncertainty quantification workflows. This prevents the silent use error of incorporating a node into a UQ pipeline when it cannot meaningfully participate — for example, using a node with `UQReadiness.NOT_READY` in an ensemble analysis that assumes all nodes propagate parameter uncertainty. A commercial integration layer should check this flag and refuse or warn when UQ is requested for a node that does not support it.

#### Recommendations

**For MADDENING**: Progressively add runtime parameter validation to all physics nodes, prioritised by the severity of the failure mode when validation is absent. Each validation check should:
- Produce a clear, actionable error message (not a generic assertion failure)
- Reference the specific stability condition or validity bound being violated
- Be documented in the node's `NodeMeta.limitations` or `hazard_hints` so the check is discoverable

This is recommended as good software engineering practice that benefits all users — researchers and commercial integrators alike — not as a regulatory obligation.

**For downstream commercial manufacturers**: Document MADDENING's parameter validation behaviour (or absence thereof) for each node used in the product, in the usability engineering file (IEC 62366-1). Specifically:
- For each MADDENING node used: list which parameters have runtime validation and which do not
- For parameters without runtime validation: document the compensating controls in the commercial product's integration layer (e.g., UI input validation, pre-simulation parameter checks against `validated_regimes`)
- Reference `NodeMeta` fields (`validated_regimes`, `hazard_hints`, `limitations`, `uq_readiness`) as the structured information source for this assessment
- Document any additional validation logic added in the integration layer that supplements MADDENING's built-in checks

The `collect_node_metadata()` and `collect_hazard_hints()` harvesting functions (Sections 9.1, 9.8) provide the structured data needed to perform this assessment systematically across all nodes.

---

## 10. IEC 62304 SOUP Compliance Package

### Background: What is SOUP?

IEC 62304:2006+AMD1:2015 defines **SOUP** (Software of Unknown Provenance) as "a software item that is already developed and generally available and that has not been developed for the purpose of being incorporated into the medical device, or software previously developed for which adequate records of the development process are not available." When a medical device manufacturer incorporates SOUP, they must perform specific assessment and documentation activities whose rigor depends on the safety classification of the software item that uses the SOUP.

MADDENING is SOUP. It was not developed under a medical device QMS, and its development records (while thorough for a research project) do not follow IEC 62304's prescribed lifecycle. This is normal and expected — the Linux kernel, FreeRTOS, OpenSSL, NumPy, and countless other widely-used open-source components are SOUP in every medical device that uses them.

### IEC 62304 SOUP Requirements

IEC 62304 imposes these specific requirements on the manufacturer who uses SOUP (not on the SOUP provider — but MADDENING proactively providing the necessary documentation dramatically reduces the manufacturer's burden):

**Clause 5.3.3 — Identify SOUP**: The manufacturer must identify each SOUP item with its name, manufacturer (or project), and a unique designator (version number, release date, or configuration identifier).

**Clause 5.3.4 — Specify functional and performance requirements for SOUP**: For each SOUP item, the manufacturer must specify the functional and performance requirements that are relevant to the intended use of the medical device.

**Clause 7.1.2 — Identify known anomalies**: The manufacturer must identify known anomalies in the SOUP that could affect the medical device.

**Clause 7.1.3 — Evaluate anomalies**: The manufacturer must evaluate the known anomalies to determine whether they are acceptable relative to the safety of the device.

**Clause 8 — Configuration management**: The SOUP item must be identified with sufficient precision to retrieve the exact version used.

### Safety Classification and SOUP

IEC 62304 classifies software into three safety classes:

| Class | Description | SOUP Requirements |
|-------|-------------|-------------------|
| **A** | No injury or damage to health is possible | Identify SOUP, document version, evaluate known anomalies |
| **B** | Non-serious injury is possible | Class A + functional/performance requirements, verify SOUP operation in architecture |
| **C** | Death or serious injury is possible | Class B + detailed architectural design, unit verification, integration testing, segregation analysis |

**Note on Class A**: Amendment 1 (2015) clarified that SOUP identification (Clause 8.1.2) and anomaly list evaluation (Clause 7.1.2/7.1.3) apply to **all** safety classes including Class A. Class A is not documentation-free.

**How this applies to MADDENING**: The safety class applies to the *software item that uses the SOUP*, not to the SOUP itself. MADDENING does not have an inherent safety class — it is classified based on how the downstream tool uses it:

| MICROBOTICA Intended Use | EU MDR Class | IEC 62304 Safety Class for MADDENING (as SOUP) | SOUP Documentation Required |
|---|---|---|---|
| Research / pre-clinical simulation | Not a medical device | N/A | N/A |
| Intraoperative digital twin (surgical decision support) | Class IIb–III | **Class C** (failure could cause death/serious injury) | Full: detailed design, unit verification, integration testing, anomaly evaluation, segregation analysis |
| RL policy generation/validation for microrobot control | Class III | **Class C** (failure could cause death/serious injury) | Full: as above |

For MICROBOTICA's clinically relevant use modes (intraoperative digital twin, RL policy validation), erroneous simulation output could directly contribute to decisions causing irreversible neurological harm or death in the CSF operating environment (adjacent to the striatum and substantia nigra). This means MADDENING will be classified as **IEC 62304 Class C SOUP** for these use cases.

**What Class C SOUP means practically for MADDENING's documentation**:

1. **Detailed architectural design evidence** (Clause 5.3): The downstream manufacturer must demonstrate that they understand MADDENING's architecture well enough to assess failure modes. MADDENING must provide `DESIGN.md`, algorithm documentation, and architecture diagrams at a level of detail sufficient for a Notified Body reviewer to follow.

2. **Unit verification evidence** (Clause 5.5): The downstream manufacturer must verify that MADDENING's software units function correctly. MADDENING's test suite, registered verification benchmarks, and analytical comparisons provide this evidence. For Class C, the bar is higher: verification must be traceable to requirements, and acceptance criteria must be explicit.

3. **Integration testing evidence** (Clause 5.6): The downstream manufacturer must test MADDENING's integration with MIME and MICROBOTICA. MADDENING's integration tests and graph-level tests provide foundational evidence, but the manufacturer must supplement with their own integration tests covering their specific configurations.

4. **Segregation analysis** (Clause 5.3.5, Class C only): If MADDENING components are used alongside safety-critical components, the manufacturer must demonstrate adequate segregation — that a MADDENING failure cannot corrupt safety-critical functionality in an undetected way. MADDENING's functional purity (pure functions, immutable state) supports segregation arguments.

5. **Anomaly evaluation at highest rigor** (Clause 7.1.3): Every known anomaly in `known_anomalies.yaml` must be evaluated for its potential to contribute to a hazardous situation in the Class C context. Anomalies marked `context_dependent` require explicit resolution for the specific intended use.

This means MADDENING's SOUP package must provide enough documentation for Class C assessment, so that any downstream tool can use it regardless of their safety class. **The documentation architecture described in this plan is therefore calibrated to Class C throughout.**

### SOUP Package Document Template

MADDENING should publish a `docs/validation/soup_package.md` with each release, structured as follows. This document is what a downstream IEC 62304-compliant project would reference in their SOUP assessment.

**Notified Body context**: For Class III devices, the Notified Body performs a full review of the technical file, including SOUP documentation. This means the SOUP package document is not just an internal QMS artifact — it will be directly read and assessed by Notified Body reviewers. It must be written to a standard that withstands external technical scrutiny: clear, complete, internally consistent, and traceable. Every claim must be backed by evidence that the reviewer can verify.

```markdown
# MADDENING SOUP Package — v[X.Y.Z]

## 1. Software Identification (IEC 62304 Clause 5.3.3)

| Field | Value |
|-------|-------|
| Name | MADDENING |
| Full name | Modular Automatic Differentiation and Data-Enhanced Neural-network INteracting Graph |
| Version | [X.Y.Z] |
| Release date | [YYYY-MM-DD] |
| License | LGPL-3.0-or-later |
| Source repository | https://github.com/[org]/MADDENING |
| PyPI package | https://pypi.org/project/maddening/[X.Y.Z]/ |
| SHA-256 (sdist) | [hash] |
| SHA-256 (wheel) | [hash] |
| Python version | >= 3.10 |
| Primary dependencies | JAX [version], jaxlib [version], NumPy [version] |

## 2. Functional Description (IEC 62304 Clause 5.3.4 support)

MADDENING provides the following capabilities:

### Core Framework
- Graph-based multiphysics simulation orchestration
- JIT-compiled graph step via JAX (pure function: state → state)
- Multi-rate timestepping, Gauss-Seidel iterative coupling, adaptive dt
- Parameter sweeps via jax.vmap, state checkpointing (NPZ)

### Physics Nodes
[List of all node types with brief descriptions and stability levels]

### Infrastructure Nodes
- **HealthCheckNode** (`maddening.nodes.health_check`): Base health monitor for execution-layer fault detection (NaN/Inf trapping, physical boundary checks, statistical moment monitoring). Downstream libraries instantiate and configure this node for their domain-specific monitoring needs. Note: downstream libraries that instantiate `HealthCheckNode` depend on this component as part of their MADDENING SOUP dependency — it is not a separate SOUP item, but must be explicitly identified in the downstream library's SOUP assessment as a MADDENING component in use.

### Surrogate Framework
[List of all architectures with brief descriptions]

### Capabilities NOT provided
- MADDENING does not interpret simulation results
- MADDENING does not provide clinical recommendations
- MADDENING does not validate that simulation parameters match any
  real physical system
- MADDENING does not enforce safety limits on user-provided parameters

## 3. Known Anomalies (IEC 62304 Clauses 7.1.2, 7.1.3)

NOTE: For Class III devices, the Notified Body will review this
anomaly list as part of the technical file assessment. Every
anomaly must have a clear severity, safety relevance assessment,
and documented rationale. The downstream manufacturer must
evaluate each anomaly for acceptability in their specific COU
and document their assessment in their risk management file.

See `docs/validation/known_anomalies.yaml` for the complete,
machine-readable known anomalies registry for this release.

Summary of [anomaly count] known anomalies:
- Critical: [N] (of which safety-relevant: [N])
- Major: [N] (of which safety-relevant: [N])
- Minor: [N]
- Context-dependent safety relevance: [N]

[Table of all anomalies with IDs, severity, safety relevance,
affected components, workaround status, and resolution status]

### Anomalies Requiring Class C Assessment
[Subset table: anomalies marked safety_relevant or
context_dependent that require explicit evaluation by the
downstream manufacturer for Class C (death/serious injury)
contexts. Each entry includes the safety relevance rationale
and recommended risk control measures.]

## 4. Verification Evidence (IEC 62304 Clause 5.5/5.6 support)

NOTE: This section provides evidence at the level required for
Class C SOUP assessment. Notified Body reviewers for Class III
devices will evaluate this evidence directly.

### Test Suite Summary
- Total tests: [N]
- Verification benchmarks (analytical): [N]
- Verification benchmarks (convergence): [N]
- Regression tests: [N]
- Integration tests: [N]
- Pass rate: [N]% (release CI)
- Platforms tested: [list]
- CI system: [description, link]

### Unit Verification Traceability (Class C)
[Table mapping each software unit to its verification evidence,
with explicit acceptance criteria and pass/fail per release.
This traceability is required for Class C SOUP assessment.]

| Software Unit | Verification Test | Acceptance Criterion | Result | CI Link |
|---|---|---|---|---|
| HeatNode.update() | test_heat_steady_state_linear | L2 error < 1e-6 vs analytical | PASS | [link] |
| LBMPipeNode.update() | test_lbm_poiseuille | L2 error < 1e-3 vs Poiseuille | PASS | [link] |

### Analytical Benchmarks
[Table of all registered verification benchmarks with pass/fail]

### Convergence Studies
[Table of convergence order verifications with measured vs expected order]

### Scope and Limitations of Verification Evidence
This verification evidence demonstrates that MADDENING's numerical
implementations correctly solve their intended mathematical equations
within the documented validated regimes. This is computational
verification, NOT clinical validation. It does not demonstrate that
any simulation output matches patient physiology or clinical outcomes.
Clinical validation is the responsibility of the device manufacturer.

## 5. IEC 62304 Lifecycle Activities Performed

[See Section 11 for the full mapping. Summary table here.]

## 6. Configuration Management

- All source code in Git with complete history
- Tagged releases on GitHub with signed commits
- PyPI releases with SHA-256 hashes
- SBOM available at [path]
- CITATION.cff for academic citation and version identification

## 7. Anomaly Management Policy

- Known anomalies tracked in `docs/validation/known_anomalies.yaml`
- GitHub Issues used for bug reporting with severity labels
- Safety-relevant anomalies evaluated at each release
- Anomaly registry updated with each release
- See CHANGELOG.md for fix history

## 8. Dependencies (SOUP of SOUP)

| Dependency | Version | License | Purpose |
|------------|---------|---------|---------|
| jax | [X.Y.Z] | Apache-2.0 | Automatic differentiation, JIT compilation |
| jaxlib | [X.Y.Z] | Apache-2.0 | XLA backend |
| numpy | [X.Y.Z] | BSD-3-Clause | Array operations |
| [optional deps listed with their install group] |

NOTE: These dependencies are themselves SOUP when MADDENING is
used in a regulated product. See the SOUP-of-SOUP assessment
policy below. The manufacturer performing Class C SOUP assessment
of MADDENING must in turn assess these dependencies as their own
SOUP components. MADDENING's SBOM (Section 14) provides the
complete dependency tree needed for this assessment.
```

### SOUP-of-SOUP Assessment Policy

MADDENING depends on several core libraries that are themselves SOUP when MADDENING is incorporated into a regulated product. The downstream commercial manufacturer must assess these as part of their complete SOUP chain. MADDENING provides the following to support that assessment:

**Dependency monitoring**: MADDENING monitors its core dependencies (JAX, jaxlib, NumPy) for known security vulnerabilities and breaking changes using two mechanisms: (1) **GitHub Dependabot** is enabled on the repository and automatically monitors PyPI dependencies for published CVEs, creating pull requests for security-relevant version bumps; (2) **manual changelog review** of JAX ecosystem libraries (JAX, jaxlib, Equinox, Optax) at each MADDENING release, since these libraries are not comprehensively covered by CVE databases and may introduce correctness-affecting changes (XLA compiler changes, numerical behaviour changes) that are not classified as security vulnerabilities. Both mechanisms are documented in `SECURITY.md`. Security-relevant dependency updates are flagged in the `Security` section of `CHANGELOG.md`. Downstream commercial manufacturers building Class III products will need more rigorous post-market surveillance of SOUP dependencies — that is their PMS obligation under EU MDR Article 83, not MADDENING's.

**Version pinning**: JAX is a rapidly evolving library with frequent releases. MADDENING pins tested version ranges in `pyproject.toml` and runs CI against pinned versions for each release. The exact versions tested are recorded in the SOUP package document and SBOM.

**Dependency credibility assessment**:

| Dependency | Maintainer | Licence | Community | Anomaly Tracking | Assessment |
|---|---|---|---|---|---|
| **JAX** | Google DeepMind | Apache-2.0 (permissive, no copyleft) | Large, active open-source community | GitHub Issues, active triage | Widely used in production ML systems; active security response; permissive licence poses no copyleft concerns for commercial products |
| **jaxlib** | Google DeepMind | Apache-2.0 | Same as JAX | Same as JAX | XLA compiler backend; tightly coupled to JAX versioning |
| **NumPy** | NumPy community (NumFOCUS) | BSD-3-Clause (permissive) | One of the most widely used scientific libraries globally | GitHub Issues, CVE tracking | Decades of production use; routinely accepted as SOUP in regulated products |

**SBOM support**: MADDENING's SBOM (Section 14) provides the complete transitive dependency tree, enabling the commercial manufacturer to identify all SOUP components — including dependencies of dependencies — in a single machine-readable artifact.

### SOUP Anomaly Management Policy

MADDENING adopts the following anomaly management policy to support downstream IEC 62304 compliance:

1. **Reporting**: All bugs are reported via GitHub Issues. Issues that represent known anomalies (incorrect computational results, unexpected behavior, numerical instability) are labeled `known-anomaly` and triaged for severity.

2. **Safety assessment**: Each known anomaly receives a `safety_relevance` assessment (safety-relevant, not-safety-relevant, context-dependent) with documented rationale. This assessment is conservative: if in doubt, it is marked `context_dependent`.

3. **Registry maintenance**: The `known_anomalies.yaml` file is updated with each release. It is never silently reduced — anomalies are only marked as resolved, never deleted.

4. **Communication**: Safety-relevant anomalies are highlighted in release notes under the "Known Anomalies" section. Critical anomalies may trigger out-of-band notifications to known downstream users.

5. **Resolution**: Bug fixes are prioritized based on severity and safety relevance. Critical safety-relevant anomalies receive the highest priority.

6. **Class C readiness**: All anomaly records are written to a standard that withstands Notified Body review. This means: precise technical descriptions, reproducible conditions, explicit scope of impact, and rationale for safety relevance that does not require codebase familiarity to understand. The `context_dependent` safety relevance category exists to be resolved by the downstream manufacturer for their specific COU — MADDENING provides the technical description, the manufacturer provides the risk assessment.

### Guidance for Downstream Safety Classification

When any downstream commercial manufacturer performs their IEC 62304 SOUP assessment of MADDENING, they should:

1. Identify which MADDENING capabilities they depend on (using Section 2 of the SOUP package)
2. Determine the safety class of the software item that uses MADDENING (based on their risk management per ISO 14971)
3. Review all known anomalies (Section 3) and evaluate whether any are unacceptable for their safety class
4. Document their functional and performance requirements for MADDENING
5. Pin to a specific MADDENING version and reference the SOUP package for that version
6. Include MADDENING in their SOUP list with version, known anomaly evaluation, and acceptance rationale

#### Class C SOUP Assessment (Required for Class IIb/III Commercial Products)

For any commercial product built on MADDENING with clinically relevant intended use modes (intraoperative digital twin, RL policy generation), MADDENING will be classified as **IEC 62304 Class C SOUP**. This triggers additional requirements beyond the basic SOUP assessment above:

7. **Architectural analysis** (Clause 5.3): Document how MADDENING is integrated into the product's architecture. Identify all interfaces, data flows, and failure propagation paths. Demonstrate understanding of MADDENING's internal architecture (referencing `DESIGN.md` and algorithm documentation).

8. **Segregation assessment** (Clause 5.3.5): If MADDENING operates alongside other components whose failure could contribute to different hazards, demonstrate that MADDENING failures cannot corrupt those components undetected. MADDENING's functional purity (pure functions, no shared mutable state) supports this argument, but it must be explicitly documented.

9. **Detailed unit verification** (Clause 5.5): Verify that each MADDENING unit used by the product functions correctly for the specific parameter ranges and configurations used. MADDENING's registered verification benchmarks provide baseline evidence, but the manufacturer may need to supplement with tests covering their specific operating regime (e.g., CSF viscosity, microrobot geometry, neural tissue boundary conditions).

10. **Integration testing** (Clause 5.6): Test MADDENING's integration with the product's other software components at all interfaces. Verify data exchange correctness, error handling, and behavior under boundary conditions. Integration test evidence must be traceable to architectural design.

11. **All anomalies with `context_dependent` safety relevance must be resolved**: For Class C, it is not acceptable to leave safety relevance as "context-dependent." Each such anomaly must receive an explicit assessment — safety-relevant or not — for the specific intended use, with documented rationale in the risk management file.

12. **Notified Body readiness**: All of the above evidence will be reviewed by the Notified Body as part of the Class III conformity assessment. Documentation must be written for an external technical reviewer, not just for internal QMS compliance.

---

## 11. IEC 62304 Software Lifecycle Documentation Mapping

### Scope

**MADDENING is not subject to IEC 62304.** It is an open-source research tool, not a medical device, and it is not developed under a medical device QMS. This mapping is provided voluntarily to support downstream SOUP assessment — it documents what MADDENING *does* provide so that any downstream commercial manufacturer subject to IEC 62304 can reference it with confidence when performing their SOUP assessment.

The downstream commercial manufacturer is the entity subject to IEC 62304 for their own software. MADDENING voluntarily documents its lifecycle activities to the same standard to make that SOUP assessment as straightforward as possible. This distinction matters legally: providing this documentation does not make MADDENING subject to IEC 62304, nor does it create any obligation to maintain IEC 62304 compliance.

### Purpose

This section maps MADDENING's existing documentation and development practices to each IEC 62304 lifecycle phase.

### Mapping

| IEC 62304 Phase | Clause | What the Standard Requires | What MADDENING Provides | Gaps |
|---|---|---|---|---|
| **Software development planning** | 5.1 | Development plan, standards, tools, configuration management plan | `DESIGN.md` (architecture decisions), `ROADMAP.md` (planned features), `CONTRIBUTING.md` (development process), Git + GitHub CI | No formal development plan document in IEC 62304 format. Not required — MADDENING is not subject to IEC 62304. |
| **Software requirements analysis** | 5.2 | Documented software requirements including functional, performance, and interface requirements | `DESIGN.md` Sections 1-5 (node contract, GraphManager API, scheduling), module READMEs, test suite (tests encode requirements implicitly) | Requirements are documented but not in a formal requirements specification with IDs and traceability. For SOUP, the downstream manufacturer defines their own requirements for MADDENING (Clause 5.3.4). |
| **Software architectural design** | 5.3 | Software architecture document, identification of SOUP items, segregation analysis (Class C: 5.3.5) | `DESIGN.md` (full architecture), `maddening/core/README.md`, graph-based architecture is self-documenting via `GraphManager.to_dict()`. Functional purity supports segregation arguments (Section 9.8). | Architecture is well-documented. MADDENING's own SOUP dependencies (JAX, NumPy) are listed in `pyproject.toml` and the SOUP package. See Clause 5.3.5 row below for segregation-specific analysis. |
| **↳ Segregation analysis** (Class C only) | 5.3.5 | Demonstrate that a failure in one software item cannot corrupt safety-critical functionality in another, or that such corruption is detected before harm occurs | **Data-flow layer**: MADDENING's functional purity (pure functions, no shared mutable state, explicit data flow) ensures that a node failure cannot silently corrupt another node's computation — the `GraphManager` passes immutable state dicts between nodes with no shared references, supporting a strong segregation argument at the data-flow level. **Execution layer (XLA Shadow, Section 9.8)**: Functional purity does **not** extend below the JAX tracing boundary. XLA compiler bugs (e.g., MADD-ANO-001), GPU hardware faults, and floating-point non-determinism across platforms can produce incorrect results that MADDENING's architecture cannot detect or prevent. The manufacturer must treat the execution layer as a **separate segregation domain** requiring its own defences: NaN/Inf trapping, physical boundary checks, statistical moment monitoring, and optional CPU/GPU cross-validation (see HealthCheck infrastructure, Section 9.8). **Algorithmic Diversity Principle** (mandatory for Class C): any `HealthCheckNode` implementation used in a Class C context must use fundamentally different numerical primitives than the physics node it monitors — this is a mandatory design requirement, not a recommendation, because the segregation argument collapses if a shared XLA compilation path corrupts both the physics output and the health monitor simultaneously (see Section 9.8 for the full rationale and CI-checkable heuristic). | MADDENING provides the data-flow segregation evidence; the manufacturer provides the execution-layer segregation evidence. The HealthCheck base node (Section 9.8) provides the infrastructure for execution-layer monitoring, but the manufacturer must instantiate and configure monitors appropriate to their context of use. The Algorithmic Diversity Principle (Section 9.8) is a mandatory design requirement for Class C contexts, enforceable at review/audit level and via CI heuristic. The two-mode verification inheritance model (Section 16) applies the same mathematical/execution boundary to downstream libraries. |
| **Software detailed design** | 5.4 | Detailed design for each software unit (Class B, C only) | Per-node algorithm documentation in `docs/algorithm_guide/`, docstrings with mathematical formulations, `NodeMeta` metadata | Per-node algorithm docs serve as detailed design. Coverage will grow as algorithm guide is populated. |
| **Software unit implementation** | 5.5 | Implement each software unit per detailed design, coding standards | Source code in `maddening/`, NumPy-style docstrings, consistent coding patterns | Code is well-structured. Formal coding standard document planned (`docs/developer_guide/code_style.md`). |
| **Software unit verification** | 5.6 | Verify each unit against detailed design and requirements | `tests/` with 545+ tests, registered verification benchmarks in `tests/verification/`, analytical comparisons, convergence studies | Good coverage. Registered benchmarks (Section 9.3) formalize verification. |
| **Software integration and integration testing** | 5.7 | Integration testing of software units | Integration tests in `tests/test_integration.py`, graph-level tests that exercise node coupling and scheduling | Present. The graph architecture naturally exercises integration. |
| **Software system testing** | 5.8 | System-level testing against requirements | End-to-end example scripts (`maddening/examples/`), surrogate training + validation pipeline tests | Examples serve as system tests. No formal system test plan. **Manufacturer recommendation**: the manufacturer's system test plan should explicitly include **cross-backend numerical agreement tests** — running the same simulation configuration on both CPU and GPU backends and asserting that outputs agree within a documented tolerance. This demonstrates that algorithmic correctness is not an artefact of a single XLA compiler path, directly addressing the execution-layer risk described in the XLA Shadow analysis (Section 9.8). This is a manufacturer system testing obligation, not a MADDENING obligation — MADDENING provides the infrastructure (HealthCheck nodes, dual-backend support) but cannot itself test every downstream configuration. |
| **Software release** | 5.9 | Release documentation, known anomalies list, verification summary | `CHANGELOG.md`, tagged GitHub releases, PyPI packages, `known_anomalies.yaml`, SOUP package document | Once SOUP package and anomaly registry are implemented, this is well-covered. |

### Recommendations for Gap Closure

The gaps identified above are acceptable because MADDENING is SOUP, not regulated software. However, the following additions (already planned in this document) would make the gap analysis even stronger:

1. **SOUP package document** (Section 10) — provides the release-level documentation that IEC 62304 Clause 5.9 expects
2. **Known anomalies registry** (Section 9.7) — satisfies the anomaly documentation that downstream tools need for Clause 7.1.2
3. **Verification benchmark registry** (Section 9.3) — formalizes unit verification evidence for Clause 5.6
4. **Algorithm guide** (Section 3) — serves as detailed design documentation for Clause 5.4

---

## 12. EU MDR Annex I and Annex II Alignment

### Background: EU MDR Requirements for Software

EU MDR (EU 2017/745) establishes the regulatory framework for medical devices in the EU. Annex I sets General Safety and Performance Requirements (GSPRs) that all medical devices must satisfy. Annex II specifies the Technical Documentation structure. MADDENING is not a medical device, but downstream commercial manufacturers building regulated products on top of MADDENING will need to satisfy these requirements. MADDENING's documentation is structured to provide specific, identified inputs to their technical files.

### Annex I: General Safety and Performance Requirements

#### Chapter I — General Requirements

The general requirements establish that devices must be safe and perform as intended. Key requirements relevant to software include:

- **GSPR 1**: Devices shall achieve the performance intended by their manufacturer and be safe.
- **GSPR 3**: Devices shall achieve their stated performance during their intended lifetime.
- **GSPR 4**: Risk management measures must follow the hierarchy: eliminate risk by design, then protect, then inform.

**How MADDENING supports this**: MADDENING's algorithm documentation (Section 3) documents validated physical regimes and known limitations. This allows any downstream commercial manufacturer's risk management file to identify risks arising from MADDENING's computational limitations and apply appropriate mitigations.

#### Section 17 — Software Requirements (GSPR 17)

Section 17 contains requirements specific to software in medical devices:

**17.1**: Software shall be designed and manufactured in accordance with the state of the art, taking into account the principles of the development lifecycle, risk management, including information security, verification, and validation.

**How MADDENING supports "state of the art"**: MADDENING's published verification evidence (analytical benchmarks, convergence studies), algorithm documentation with mathematical derivations, and use of established numerical methods (finite differences, LBM, neural surrogates with published architectures) constitute state-of-the-art evidence. The `docs/bibliography.bib` with citations to peer-reviewed literature provides the academic foundation. The development lifecycle documentation (Section 11 IEC 62304 mapping) demonstrates systematic development practices.

**17.2**: Software that is intended to be used in combination with mobile computing platforms shall be designed and manufactured taking into account the specific features of the mobile platform.

**Not applicable to MADDENING** (HPC server-based framework).

**17.3**: Manufacturers shall set out minimum requirements concerning hardware, IT networks characteristics and IT security measures, including protection against unauthorised access, necessary to run the software as intended.

**How MADDENING supports this**: The SOUP package document (Section 10) includes hardware/OS requirements and dependency versions. `SECURITY.md` documents vulnerability reporting. SBOM artifacts (Section 14) support cybersecurity assessment per MDCG 2019-16.

**17.4**: Manufacturers shall ensure that software that is a device itself shall be classified using applicable classification rules.

**Not applicable to MADDENING** (not a device). Applicable to any downstream commercial product — see Section 13.

### Annex II: Technical Documentation

When a downstream commercial manufacturer prepares their EU MDR Technical Documentation (required for CE marking), they will reference MADDENING as a SOUP component. MADDENING provides the following specific inputs to each section of the technical file:

#### Section 2 — Information to be supplied by the manufacturer

The technical file must include the device's intended purpose and a description of its components. MADDENING's contribution:
- `docs/regulatory/intended_use.md` — MADDENING's platform positioning statement clarifies that MADDENING is a general-purpose tool with no medical purpose
- `docs/regulatory/downstream_integration.md` — documents the relationship between MADDENING and any downstream regulated product
- `README.md` disclaimer — clear statement that MADDENING is not a medical device

#### Section 4 — Design and manufacturing information

The technical file must include a complete description of the design stages applied to the device, including software design and development. For Class III, the Notified Body performs a full review of the technical file including this section — MADDENING's SOUP documentation will be directly scrutinized, not just audited. MADDENING's contribution:
- `DESIGN.md` — complete architecture documentation (functional state pattern, GraphManager, scheduling, coupling)
- `docs/algorithm_guide/` — per-node mathematical documentation including governing equations, discretization, assumptions (serves as detailed design evidence for Class C SOUP assessment per IEC 62304 Clause 5.4)
- `docs/validation/framework_verification.md` — verification evidence summary with unit-level traceability (required for Class C per IEC 62304 Clause 5.5)
- `docs/validation/soup_package.md` — structured SOUP package document written for Notified Body review
- IEC 62304 lifecycle mapping (Section 11) — demonstrates systematic development
- Source code available under LGPL — full transparency for Notified Body review

#### Section 6 — Benefit-risk analysis and risk management

The technical file must include the benefit-risk analysis and risk management output per ISO 14971. MADDENING's contribution:
- Per-node **Known Limitations and Failure Modes** (Section 3 template) — feeds directly into the manufacturer's risk management file
- `docs/validation/known_anomalies.yaml` — structured anomaly data for risk assessment
- Per-node **Validated Physical Regimes** — bounds on where MADDENING's results are reliable
- `NodeMeta.limitations` (Section 9.1) — machine-readable limitations for automated risk identification

### MDCG 2019-16: Cybersecurity

MDCG 2019-16 provides guidance on cybersecurity for medical devices. Key requirements that affect MADDENING as a SOUP component:

- **SBOM**: The manufacturer should be able to provide a Software Bill of Materials. MADDENING supports this via SBOM generation (Section 14).
- **Vulnerability management**: The manufacturer must monitor for vulnerabilities in SOUP components. MADDENING supports this via `SECURITY.md`, the anomaly registry, and GitHub security advisories.
- **Secure development**: SOUP should be evaluated for secure development practices. MADDENING's CI, code review, and dependency management support this assessment.

---

## 13. MDCG 2019-11: Qualification and Classification

### What is MDCG 2019-11?

MDCG 2019-11 (rev. 2023) provides guidance on the qualification and classification of software under EU MDR. It answers two questions: (1) Is this software a medical device? (2) If so, what class?

The guidance is critical for the MADDENING ecosystem because it draws the line between general-purpose tools (not medical devices) and clinical software (medical devices requiring CE marking).

### Step 1: Is MADDENING a Medical Device?

MDCG 2019-11 defines a medical device as software that has a **medical purpose** as defined in EU MDR Article 2(1). The guidance provides a step-by-step qualification flowchart.

**MADDENING is NOT a medical device** because:

1. **No medical purpose**: MADDENING is a general-purpose computational framework for multiphysics simulation. It performs numerical computation — solving PDEs, running lattice Boltzmann methods, training neural surrogates. It does not perform any function listed in Article 2(1): it does not diagnose, prevent, monitor, predict, prognose, treat, or alleviate disease.

2. **General-purpose tool**: MDCG 2019-11 Section 3.2 explicitly states that "generic tools or general purpose software [...] are not in themselves medical devices." MADDENING is analogous to MATLAB, NumPy, or OpenFOAM — a computational tool that *can be used* within medical applications but has no inherent medical purpose.

3. **No clinical interpretation**: MADDENING produces numerical simulation outputs (arrays of physical quantities). It does not interpret these outputs in a clinical context, provide diagnostic conclusions, or make therapeutic recommendations.

The appropriate language for MADDENING's documentation:

> MADDENING is a general-purpose computational tool as described in MDCG 2019-11 Section 3.2. It performs numerical simulation of physical systems using established mathematical methods. It has no medical purpose and does not qualify as a medical device under EU MDR Article 2(1).

### Step 2: How Would a Commercial Product Built on MICROBOTICA Be Classified?

MICROBOTICA itself is an open-source research simulator with no medical purpose and no manufacturer seeking CE marking. However, a commercial product built on top of MICROBOTICA that *does* have a medical purpose (e.g., informing treatment decisions for microrobot-assisted drug delivery in CSF) would be assessed as follows:

1. **Qualification**: Such a commercial product would be Software as a Medical Device (SaMD) because it has a medical purpose independent of any hardware.

2. **Classification under EU MDR Rule 11**: Software intended to provide information used to take decisions with diagnosis or therapeutic purposes is classified based on the clinical significance of the information:
   - **Class IIa**: If the information may cause non-serious deterioration of health or surgical intervention
   - **Class IIb**: If the information may cause serious deterioration of health or surgical intervention
   - **Class III**: If the information may cause death or irreversible deterioration of health

3. **The commercial product's classification depends on intended use mode**:

   | Intended Use Mode | Rule 11 Analysis | EU MDR Class |
   |---|---|---|
   | **Research / pre-clinical simulation** | No medical purpose; not a medical device | Not classified |
   | **Intraoperative digital twin** (real-time surgical decision support) | Provides information for therapeutic decisions; erroneous output during a live procedure in CSF spaces adjacent to the striatum and substantia nigra could directly cause irreversible neurological harm or death | **Class IIb minimum; likely Class III** |
   | **RL policy generation/validation for intraoperative microrobot control** | Software directly informs an active control loop for a device operating in the central nervous system; failure could cause death or serious injury | **Class III** |

   The operating environment is critical to the classification analysis: microrobot drug delivery in CSF operates in proximity to the striatum and substantia nigra. Erroneous simulation output (incorrect flow prediction, wrong drug concentration profile, flawed trajectory planning) during a live intraoperative procedure could directly cause irreversible neurological damage. This pushes classification to Class III under Rule 11's "death or irreversible deterioration of health" criterion.

4. **IMDRF SaMD alignment**: The IMDRF framework (N10, N12, N23) classifies SaMD by combining the *significance of information* (treat, diagnose, drive clinical management, inform clinical management) with the *healthcare situation* (critical, serious, non-serious). For the clinically relevant use modes of a commercial product built on MICROBOTICA:
   - **Intraoperative digital twin**: "drive clinical management" × "critical" = **IMDRF Category III**, mapping to EU MDR **Class IIb/III**
   - **RL policy generation for active control**: "treat or diagnose" × "critical" = **IMDRF Category IV**, mapping to EU MDR **Class III**

5. **Design principle**: The documentation architecture must support the most demanding plausible classification (Class III). Designing for Class III means that Class IIb and IIa use cases are automatically covered — the reverse is not true.

### What MADDENING Must Provide for a Downstream Manufacturer's Classification Dossier

When a downstream commercial manufacturer prepares their classification dossier, they must describe their software components and their roles. For MADDENING, they should include:

1. **MADDENING's SOUP package document** — identifies MADDENING as a SOUP component with version, capabilities, and known anomalies (written to withstand Notified Body review for Class III)
2. **MADDENING's intended use statement** — confirms MADDENING has no medical purpose
3. **MADDENING's algorithm documentation** — demonstrates the mathematical rigor of the computational methods, with governing equations, discretization details, and validated regimes documented at a level a Notified Body reviewer can assess
4. **MADDENING's verification evidence** — demonstrates that the computational methods produce correct results within validated regimes, with explicit traceability from requirements to test results
5. **MADDENING's architectural documentation** — sufficient for the downstream manufacturer to perform the Class C architectural analysis (Clause 5.3) and segregation assessment (Clause 5.3.5) required for Class III

The classification dossier should explicitly state that MADDENING is a general-purpose computational library (not SaMD) and that the medical purpose resides entirely in the commercial product's clinical interpretation layer. For Class III conformity assessment, the Notified Body will review the full SOUP assessment including all of the above artifacts.

---

## 14. Configuration Management

### IEC 62304 Requirements

IEC 62304 Clause 8 requires that SOUP components be identified with sufficient precision that the exact version used in a medical device can be determined and retrieved. Specifically:

- **Clause 8.1.1**: A configuration management system must be established
- **Clause 8.1.2**: Configuration items must be identified (including SOUP)
- **Clause 8.1.3**: Change control must be documented
- **Clause 8.3**: A mechanism must exist to retrieve previous versions

### What MADDENING Provides

#### Stable, Content-Addressed Releases

- **PyPI releases**: Each version published to PyPI with SHA-256 hashes for both sdist and wheel distributions. PyPI is immutable — once a version is published, it cannot be changed.
- **GitHub release tags**: Each version tagged in Git. Tags should reference a specific commit hash.
- **Signed commits** (recommended): GPG-signed tags provide cryptographic proof of provenance.

#### CITATION.cff as Configuration Management Artifact

The `CITATION.cff` file serves dual purposes — academic citation and configuration management:

```yaml
cff-version: 1.2.0
message: "If you use MADDENING, please cite it as below."
title: "MADDENING"
type: software
version: "0.1.0"                    # Updated with each release
date-released: "2025-XX-XX"         # Updated with each release
license: "LGPL-3.0-or-later"
repository-code: "https://github.com/[org]/MADDENING"
identifiers:
  - type: doi
    value: "10.XXXX/zenodo.XXXXXXX"  # Zenodo DOI for archival
  - type: other
    value: "sha256:[sdist-hash]"      # PyPI sdist hash
    description: "PyPI sdist SHA-256"
authors:
  - family-names: "[Author]"
    given-names: "[Author]"
    orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
```

The `identifiers` section includes the PyPI hash, making `CITATION.cff` a configuration management artifact that can be used to verify the exact source of a MADDENING installation.

#### SimulationProvenance Configuration Traceability

The `SimulationProvenance` dataclass (Section 9.2) captures `maddening_version` at runtime, satisfying the IEC 62304 requirement that the exact SOUP version used in any computation can be determined.

#### Software Bill of Materials (SBOM)

EU MDR (via MDCG 2019-16) and FDA cybersecurity guidance both increasingly expect SBOMs for software components in medical devices. MADDENING should publish SBOM artifacts with each release.

**Recommendation**: Generate CycloneDX or SPDX SBOMs as part of the release process. Tools like `cyclonedx-py` can generate these automatically from `pyproject.toml`.

```bash
# Generate SBOM (add to release process)
pip install cyclonedx-bom
cyclonedx-py environment -o sbom.json --format json
```

The SBOM should be:
- Published alongside each GitHub release as a release artifact
- Referenced in the SOUP package document
- Available for downstream tools to incorporate into their own SBOMs

---

## 15. QMS Compatibility

### Context

Any downstream commercial manufacturer building a regulated product on top of MADDENING must operate within a Quality Management System (QMS) certified to EN ISO 13485:2016. Within this QMS, MADDENING is treated as a **purchased product** or **externally provided component** — specifically, as SOUP under IEC 62304.

### Why MADDENING Does Not Operate Under a QMS

MADDENING is a solo research project. It does not operate under an ISO 13485 QMS, and this is stated honestly, not apologetically.

Formal QMS certification requires organisational infrastructure that is incompatible with solo open-source development:

- **Management reviews**: ISO 13485 requires periodic management reviews of the QMS by "top management." A solo developer reviewing their own work does not constitute a management review.
- **Internal audits**: The QMS must be audited internally by personnel independent of the area being audited. A single person cannot audit themselves.
- **Corrective and preventive action (CAPA)**: Formal CAPA procedures require defined investigation, root cause analysis, and effectiveness check workflows. While MADDENING performs these activities informally (via GitHub Issues and the anomaly registry), they are not documented in the formal structure ISO 13485 requires.
- **Designated personnel**: ISO 13485 requires documented roles, responsibilities, and authorities. A solo project has one person in all roles.
- **Document control**: Formal document control procedures with review, approval, and distribution records. Git provides version control, but not the formal approval workflow ISO 13485 expects.

This is entirely normal for open-source SOUP. The Linux kernel, FreeRTOS, OpenSSL, ITK, VTK, NumPy, and countless other widely-used open-source components are routinely accepted in FDA-cleared and CE-marked medical products without their developers holding ISO 13485 certification.

### What Substitutes for QMS Rigour in MADDENING

MADDENING's development practices provide equivalent assurance without formal certification:

| ISO 13485 Requirement | MADDENING Equivalent |
|---|---|
| Version control and document control | Git with full commit history, tagged releases, immutable PyPI packages |
| Verification and testing | CI with mandatory passing tests (545+ tests), registered verification benchmarks, analytical comparisons |
| Change control | GitHub PRs with review, `CHANGELOG.md`, structured commit messages |
| Corrective action | GitHub Issues with severity labels, anomaly registry with resolution tracking |
| Design input/output traceability | Algorithm documentation with governing equations → implementation → test; centralized bibliography with CI-validated citations; implementation mapping with CI-validated function references |
| Supplier evaluation (of own SOUP) | Dependency monitoring, version pinning, SBOM generation |
| Independent validation | Peer-reviewed publications citing MADDENING, independent research use |

The combination of open-source transparency, CI automation, structured anomaly management, and published research use provides a credibility argument that is well-understood by Notified Body reviewers.

### How SOUP is Handled in the Manufacturer's ISO 13485 QMS

ISO 13485 Clause 7.4 (Purchasing) requires that externally provided components be evaluated, selected, and verified. For SOUP software components, this means:

1. **Supplier evaluation**: The QMS must evaluate MADDENING as a supplier. Since MADDENING is open-source with no formal supplier relationship, the evaluation is based on objective evidence: published source code, test results, documentation quality, issue tracking transparency, and community activity.

2. **Incoming inspection**: The QMS must verify that the received SOUP meets requirements. For MADDENING, this means: downloading a pinned version from PyPI, verifying the SHA-256 hash, running the verification benchmark suite against the downloaded version, and comparing results to the SOUP package document.

3. **Approved supplier list**: MADDENING must be listed on the QMS's approved supplier/SOUP list with version, acceptance criteria, and review date.

### What MADDENING Provides to Support QMS Integration

| QMS Requirement | MADDENING Documentation |
|---|---|
| Supplier identification | `CITATION.cff`, `pyproject.toml`, GitHub repository |
| Version identification | Semantic versioning, PyPI releases with hashes, Git tags |
| Functional specification | SOUP package document Section 2 |
| Known anomalies | `known_anomalies.yaml`, SOUP package document Section 3 |
| Verification evidence | `docs/validation/`, SOUP package document Section 4 |
| Release documentation | `CHANGELOG.md`, release notes |
| Change control evidence | Git history, GitHub PRs, `CHANGELOG.md` |
| Anomaly management | GitHub Issues with label taxonomy, `known_anomalies.yaml` |

### What the Commercial Manufacturer Must Add

The commercial entity building on MADDENING must operate under ISO 13485. Their QMS must cover:

- **Their own development activities**: IEC 62304-compliant software lifecycle for the clinical interpretation layer, user interface, and any other code they write
- **SOUP assessment procedures**: Documented procedures for evaluating, accepting, and monitoring SOUP components (including MADDENING and its dependencies)
- **Risk management process**: ISO 14971-compliant risk management covering the complete product including SOUP failure modes
- **Post-market surveillance system**: PMS plan, PSUR generation, vigilance reporting, PMCF — these are legal obligations that require organisational capacity

MADDENING's documentation reduces the effort required for the SOUP assessment component of their QMS but does not substitute for the QMS itself.

### Transition Planning

If the developer is involved in founding or advising a spin-out commercial entity, there will be a point at which some version of MADDENING becomes a component of that entity's ISO 13485 QMS. The documentation architecture described in this document is designed to make that transition as smooth as possible:

- The **SOUP package documents** are already structured to slot directly into a QMS SOUP assessment procedure
- The **anomaly registry** provides the known anomaly list required by IEC 62304 Clause 7.1.2
- The **lifecycle mapping** (Section 11) pre-answers the questions a QMS auditor will ask about SOUP development practices
- The **verification benchmarks** provide the evidence needed for incoming inspection of the SOUP component
- The **SBOM** feeds directly into the manufacturer's cybersecurity documentation

The commercial entity would need to establish: ISO 13485-certified QMS, regulatory consultant relationship, Notified Body selection, initial risk management file (ISO 14971), and SOUP assessment procedures. MADDENING's documentation provides the SOUP-specific inputs to all of these.

The transition pathway described above applies equally to any commercial entity — whether a spin-out founded by the developer, an independent licensee, a research hospital commercialising a product built on MICROBOTICA, or any other organisation. MADDENING's documentation is designed to serve all of them without modification; no single entity receives privileged access to documentation or support.

### Class III Escalation

For Class III devices, the Notified Body performs a **full technical file review** (not just a QMS audit). This means the Notified Body reviewer will directly examine MADDENING's SOUP package document, known anomalies registry, algorithm documentation, and verification evidence. The quality bar is correspondingly higher than for Class IIa, where SOUP documentation is primarily an internal QMS artifact reviewed during scheduled audits. Every artifact described in this document should be written with the assumption that a Notified Body technical reviewer — not just the internal QMS team — will read it. This means: clear structure, explicit traceability, no unexplained gaps, and honest acknowledgment of limitations.

---

## 16. Downstream Framework Inheritance

This section describes how a downstream physics engine — primarily MIME, but applicable to any library built on MADDENING — can inherit and extend the compliance architecture described in Sections 1–15, rather than rebuilding it from scratch.

The goal is practical: a developer starting MIME should be able to read this section and know exactly what they get for free, what they must provide themselves, and how the split between MADDENING's compliance infrastructure and their own works.

### The Schema/Instance Split

MADDENING's compliance architecture has two distinct layers that downstream libraries must treat differently:

**The schema layer** — data structures, tooling, and conventions that are generic and reusable by any downstream project without modification:

- **Data structures**: `NodeMeta`, `EdgeMeta`, `ValidatedRegime`, `Reference`, `AnomalyRecord`, `AnomalySeverity`, `SafetyRelevance`, `ResolutionStatus`, `ValidationBenchmark`, `BenchmarkType`, `SimulationProvenance`, `StabilityLevel`, `UQReadiness`, `UncertaintySpec`, `UncertainParameter`
- **Decorators**: `@verification_benchmark`, `@stability`
- **Harvesting utilities**: `collect_node_metadata()`, `collect_hazard_hints()`
- **Validation tooling**: anomaly registry validator (schema validation for `known_anomalies.yaml`)
- **Templates**: algorithm guide template, SOUP package template, COU template
- **Conventions**: ID namespacing, CHANGELOG structure, anomaly severity taxonomy, GitHub label taxonomy

**The instance layer** — MADDENING-specific content that belongs to MADDENING and must not be inherited:

- MADDENING's own node implementations and their `NodeMeta` instances
- MADDENING's own `known_anomalies.yaml` (containing `MADD-ANO-*` entries)
- MADDENING's own `soup_package.md`
- MADDENING's own verification benchmarks and their results
- MADDENING's own algorithm guide documents
- MADDENING's own `CHANGELOG.md`, `CITATION.cff`, `intended_use.md`

A downstream library **imports the schema layer** and **creates its own instance layer**. The schema provides the structure; the instances provide the content. This is the fundamental model: **inherit the schema, not the evidence**.

### The `maddening.compliance` Namespace

The schema layer is accessible via a dedicated, stable subpackage that serves as the single import surface for downstream libraries:

```python
# maddening/compliance/__init__.py
"""
Compliance schema and tooling for MADDENING and downstream libraries.

This module re-exports all data structures, decorators, and utilities
that downstream libraries need to participate in MADDENING's compliance
architecture. Import from here — not from maddening.core internals.

Usage (from a downstream library like MIME):

    from maddening.compliance import (
        # Metadata schemas
        NodeMeta, EdgeMeta, ValidatedRegime, Reference,
        StabilityLevel, UQReadiness,
        # Anomaly management
        AnomalyRecord, AnomalySeverity, SafetyRelevance, ResolutionStatus,
        # Validation infrastructure
        ValidationBenchmark, BenchmarkType, verification_benchmark,
        # Stability
        stability,
        # Harvesting
        collect_node_metadata, collect_hazard_hints,
        # Registry validation
        validate_anomaly_registry,
    )
"""

# Metadata schemas
from maddening.core.compliance.metadata import (
    NodeMeta, EdgeMeta, ValidatedRegime, Reference,
    StabilityLevel, UQReadiness,
    collect_node_metadata,
    collect_hazard_hints,
)

# Anomaly management
from maddening.core.compliance.anomaly import (
    AnomalyRecord, AnomalySeverity, SafetyRelevance, ResolutionStatus,
)

# Validation infrastructure
from maddening.core.compliance.validation import (
    ValidationBenchmark, BenchmarkType,
    verification_benchmark,
)

# Stability — no-op stub until Phase 4
# In Phase 0–3, `stability` is an identity decorator (applies the marker
# attribute but does not populate a registry or generate reports).  The
# functional implementation lands in Phase 4 (Appendix B item 30).
# This avoids a broken import while still allowing downstream libraries
# to annotate their API surfaces with `@stability(StabilityLevel.STABLE)`
# from day one — the annotations are inert until the machinery exists.
from maddening.core.compliance.stability import stability

# Registry validation
from maddening.compliance._validate import validate_anomaly_registry

# Provenance (available after Phase 3)
# from maddening.core.compliance.provenance import SimulationProvenance

# UQ (available after Phase 4)
# from maddening.core.compliance.uq import UncertaintySpec, UncertainParameter
```

#### Design Decisions

**No JAX dependency**: Everything in `maddening.compliance` must be importable without JAX installed. The schema layer is pure Python dataclasses, enums, and validation logic. This means a downstream library's CI can validate its anomaly registry and check its `NodeMeta` without installing JAX — important for lightweight CI jobs, documentation builds, and for downstream projects that may use a different computation backend for some tasks.

**Part of the base install**: The compliance namespace is included in the base `pip install maddening` — no extras required. The schema types are lightweight (standard library only: `dataclasses`, `enum`, `typing`). If a downstream library needs only the compliance schema and not the simulation framework, it can `pip install maddening` and import from `maddening.compliance` without triggering JAX installation, provided it does not import from `maddening.core` or `maddening.nodes`. A separate `pip install maddening[compliance]` extra is not needed at this time, but could be introduced later if the base install grows unwieldy.

**Stability guarantee**: The `maddening.compliance` namespace is the stable API surface for downstream libraries. It follows strict semantic versioning: breaking changes to schema types only in major versions. New types may be added in minor versions. Downstream libraries pin to a MADDENING version range and import from this namespace; they are insulated from internal refactoring of `maddening.core`. The `@stability(StabilityLevel.STABLE)` decorator (Section 9.5) should be applied to `maddening.compliance` exports once the stability machinery is implemented (Phase 4).

### Shared Tooling

#### Anomaly Registry Validation

The CI script `scripts/check_anomalies.py` (Section 9.7) validates a `known_anomalies.yaml` file against the schema. Rather than requiring downstream libraries to copy this script, MADDENING exposes the validation logic as both a library function and a command-line tool:

```python
# maddening/compliance/_validate.py

import yaml
import sys
from typing import Optional


def validate_anomaly_registry(
    path: str,
    *,
    prefix: str = "",
) -> list[str]:
    """Validate a known_anomalies.yaml file against the anomaly schema.

    Parameters
    ----------
    path : str
        Path to the YAML file to validate.
    prefix : str, optional
        Expected anomaly ID prefix (e.g., "MIME-ANO-"). If provided,
        all anomaly IDs must start with this prefix. If empty, any
        prefix is accepted.

    Returns
    -------
    list[str]
        List of validation errors. Empty list means the file is valid.
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    errors = []

    # Top-level structure
    if not isinstance(data, dict):
        return ["File must contain a YAML mapping"]
    for required in ("schema_version", "generated_date"):
        if required not in data:
            errors.append(f"Missing top-level field: {required}")

    # Anomaly entries
    ids_seen = set()
    for a in data.get("anomalies", []):
        aid = a.get("anomaly_id", "<missing>")

        # Uniqueness
        if aid in ids_seen:
            errors.append(f"Duplicate anomaly_id: {aid}")
        ids_seen.add(aid)

        # Prefix enforcement
        if prefix and not aid.startswith(prefix):
            errors.append(f"{aid}: does not match required prefix '{prefix}'")

        # Required fields
        for field in ("anomaly_id", "title", "description",
                      "severity", "safety_relevance",
                      "safety_relevance_rationale"):
            if not a.get(field):
                errors.append(f"{aid}: missing required field '{field}'")

        # Valid enums
        if a.get("severity") not in (
            "critical", "major", "minor", "enhancement", None
        ):
            errors.append(f"{aid}: invalid severity '{a.get('severity')}'")
        if a.get("safety_relevance") not in (
            "safety_relevant", "not_safety_relevant",
            "context_dependent", None
        ):
            errors.append(
                f"{aid}: invalid safety_relevance '{a.get('safety_relevance')}'"
            )

    return errors
```

```python
# maddening/compliance/__main__.py
"""
Command-line interface for MADDENING compliance tooling.

Usage:
    python -m maddening.compliance check-anomalies <path> [--prefix PREFIX]
    python -m maddening.compliance check-meta <module> [--require-fields FIELDS]

Examples:
    # Validate MADDENING's own registry
    python -m maddening.compliance check-anomalies docs/validation/known_anomalies.yaml

    # Validate MIME's registry with namespace enforcement
    python -m maddening.compliance check-anomalies docs/validation/known_anomalies.yaml --prefix MIME-ANO-
"""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        prog="python -m maddening.compliance",
        description="MADDENING compliance tooling",
    )
    sub = parser.add_subparsers(dest="command")

    # check-anomalies
    ca = sub.add_parser("check-anomalies", help="Validate an anomaly registry")
    ca.add_argument("path", help="Path to known_anomalies.yaml")
    ca.add_argument("--prefix", default="",
                    help="Required anomaly ID prefix (e.g., MIME-ANO-)")

    args = parser.parse_args()

    if args.command == "check-anomalies":
        from maddening.compliance._validate import validate_anomaly_registry
        errors = validate_anomaly_registry(args.path, prefix=args.prefix)
        if errors:
            for e in errors:
                print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        print(f"OK: registry at {args.path} is valid")
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
```

The original `scripts/check_anomalies.py` (Section 9.7) should be refactored to delegate to `maddening.compliance.validate_anomaly_registry()`, so the validation logic exists in exactly one place:

```python
# scripts/check_anomalies.py (updated)
"""CI script — thin wrapper around the compliance validator."""
import sys
from maddening.compliance._validate import validate_anomaly_registry

errors = validate_anomaly_registry("docs/validation/known_anomalies.yaml")
if errors:
    for e in errors:
        print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)
print(f"OK: anomaly registry is valid")
```

#### Other Shared Tooling

The following utilities are schema-driven and path-agnostic, making them reusable by any downstream library:

| Tool | Import | Downstream Usage |
|---|---|---|
| Anomaly registry validator | `maddening.compliance.validate_anomaly_registry()` | Validate downstream registry against same schema |
| NodeMeta harvester | `maddening.compliance.collect_node_metadata()` | Harvest metadata from downstream node classes (works on any `SimulationNode` subclass) |
| Hazard hints harvester | `maddening.compliance.collect_hazard_hints()` | Harvest hazard hints across downstream nodes for risk management input |
| Stability report generator | `maddening.compliance.generate_stability_report()` | Generate stability report covering downstream API surfaces. **(Phase 4 — not yet available)**: depends on `@stability` decorator infrastructure (Section 9.5, Appendix B item 29). Do not import until Phase 4 is complete. |
| Verification benchmark registry | `maddening.compliance.verification_benchmark` | Register downstream benchmarks in the same registry pattern |
| Bibliography citation validator | `scripts/check_citations.py` (standalone) | Downstream libraries can adapt the script to validate their own algorithm guides against their own `.bib` file (or MADDENING's); set `BIB_PATH` env var to override the default path |

The harvesting utilities (`collect_node_metadata()`, `collect_hazard_hints()`) use `SimulationNode.__subclasses__()` to discover all node classes in the current Python process. When MIME imports both MADDENING and MIME nodes, these utilities will harvest metadata from both — automatically producing a combined capability matrix and hazard summary. This is by design: a commercial manufacturer assessing the full MADDENING+MIME stack can call these utilities once and get a complete picture.

### Algorithm ID and Anomaly ID Namespacing

MADDENING uses `MADD-NODE-XXX` and `MADD-ANO-XXX` prefixes. Downstream libraries must use their own prefix to avoid collisions and to enable unambiguous cross-referencing across the ecosystem.

#### Convention

Each project chooses a short, uppercase prefix (3–5 characters) derived from its name. The prefix is followed by a category tag and a zero-padded numeric sequence:

```
<PREFIX>-<CATEGORY>-<NUMBER>

Where:
  PREFIX   = project identifier (e.g., MADD, MIME, MBOT)
  CATEGORY = NODE, ANO, VER (algorithm, anomaly, verification benchmark)
  NUMBER   = zero-padded, monotonically increasing within the project
```

| Project | Prefix | Node IDs | Anomaly IDs | Benchmark IDs |
|---|---|---|---|---|
| MADDENING | `MADD` | `MADD-NODE-001` | `MADD-ANO-001` | `MADD-VER-001` |
| MIME | `MIME` | `MIME-NODE-001` | `MIME-ANO-001` | `MIME-VER-001` |
| MICROBOTICA | `MBOT` | `MBOT-NODE-001` | `MBOT-ANO-001` | `MBOT-VER-001` |

The prefix is documented in the project's `CONTRIBUTING.md` and enforced by the anomaly registry validator (`--prefix` flag).

#### Cross-Referencing

Anomalies in downstream libraries may trace to known limitations in upstream libraries. The convention is to reference the upstream anomaly ID in the `description` or `safety_relevance_rationale` field:

```yaml
# MIME known_anomalies.yaml
anomalies:
  - anomaly_id: "MIME-ANO-003"
    title: "CSF flow solver inherits LBM GPU segfault from MADDENING"
    description: >
      MIME's CSFFlowNode uses MADDENING's LBMPipeNode internally.
      The CUDA 12.2 segfault documented in MADD-ANO-001 affects
      MIME's CSF flow solver when running on GPU.
    safety_relevance: "context_dependent"
    safety_relevance_rationale: >
      Inherits from MADD-ANO-001. See MADDENING known_anomalies.yaml
      v0.3.2 for the upstream analysis. MIME adds no additional risk
      beyond the upstream anomaly. Workaround: use CPU backend.
```

Cross-references are string-based (not machine-enforced). The namespace convention ensures that references are unambiguous even when anomaly registries from multiple projects are collected into a single commercial manufacturer's risk management file. A Notified Body reviewer seeing `MADD-ANO-001` in MIME's anomaly registry can unambiguously trace it to MADDENING's known anomalies registry.

#### Extensibility

When a new downstream project joins the ecosystem:

1. Choose a unique prefix (document it in `CONTRIBUTING.md`)
2. Use the prefix consistently across all ID categories
3. Enforce the prefix in CI via `--prefix` flag on the anomaly validator

No coordination with MADDENING is required — the convention is self-enforcing as long as prefixes are unique within the ecosystem. If the ecosystem grows large enough that prefix collision becomes a concern, a simple registry file (e.g., `known_prefixes.yaml` in the MADDENING repository) can be added — but this is not needed now and should not be over-engineered.

### SOUP-of-SOUP Chain Documentation

When MIME is assessed as SOUP by a downstream commercial manufacturer, that manufacturer must also assess MADDENING (because MIME depends on it). This creates a SOUP chain: Commercial Product → MIME (SOUP) → MADDENING (SOUP of SOUP). MIME's own SOUP package document must make this dependency chain explicit and navigable.

#### What MIME's SOUP Package Section 8 (Dependencies) Must Contain

```markdown
## 8. Dependencies (SOUP of SOUP)

### 8.1 MADDENING

| Field | Value |
|-------|-------|
| Name | MADDENING |
| Version | 0.3.2 (pinned) |
| Licence | LGPL-3.0-or-later |
| SOUP package | https://github.com/[org]/MADDENING/blob/v0.3.2/docs/validation/soup_package.md |
| Known anomalies | https://github.com/[org]/MADDENING/blob/v0.3.2/docs/validation/known_anomalies.yaml |
| SHA-256 (PyPI sdist) | [hash of maddening-0.3.2.tar.gz] |
| SBOM | https://github.com/[org]/MADDENING/releases/download/v0.3.2/sbom.json |

#### MADDENING Anomalies Relevant to MIME

The following MADDENING known anomalies affect MIME's operation.
Other MADDENING anomalies (e.g., those affecting nodes not used
by MIME) are listed in MADDENING's own registry but do not affect
MIME and are not repeated here.

| MADDENING Anomaly | Affects MIME? | MIME Impact | MIME Mitigation |
|---|---|---|---|
| MADD-ANO-001 (LBM GPU segfault) | Yes | CSFFlowNode crashes on CUDA 12.2 | CI tests on CPU; GPU users warned |
| MADD-ANO-002 (HeatNode CFL not enforced) | No | MIME does not use HeatNode | N/A |

#### MADDENING Version Update Policy

MIME pins to a specific MADDENING release in `pyproject.toml`. When
MADDENING publishes a new version, the following process applies
before updating the pin:

1. Review MADDENING's CHANGELOG.md — specifically the Known Anomalies,
   Security, and Verification sections
2. Review new/changed entries in MADDENING's known_anomalies.yaml and
   assess each for impact on MIME's operation
3. Run MIME's full test suite (including verification benchmarks)
   against the candidate MADDENING version
4. Update the version pin in pyproject.toml
5. Update this section (Section 8.1): version, SHA-256, anomaly table
6. Document the update in MIME's CHANGELOG.md under the appropriate
   release, noting any anomaly impact changes
```

This pattern chains cleanly: when a commercial manufacturer assesses MIME as SOUP, they see MADDENING listed as a dependency with a pinned version, direct links to MADDENING's own SOUP package (version-tagged, so the link is immutable), and an explicit assessment of which MADDENING anomalies affect MIME. The manufacturer can follow the link to MADDENING's SOUP package and perform their own upstream assessment — or, if MIME's assessment is sufficiently thorough, accept it as part of MIME's own SOUP evaluation.

The same pattern applies to MIME's other dependencies (JAX, jaxlib, NumPy). MIME does not need to re-document these — MADDENING's SOUP package Section 8 already covers them — but MIME should note that its transitive dependencies are documented in MADDENING's SOUP package and SBOM.

### Reference Nodes vs. Production Nodes

MADDENING's built-in nodes (`BallNode`, `HeatNode`, `LBMPipeNode`, etc.) are **reference implementations**: demonstrations of correctness, baselines for testing, and proof-of-concept use cases for the framework's capabilities. They are not production physics engines optimised for specific application domains.

This distinction is important for downstream libraries and has concrete implications:

**What downstream libraries inherit from MADDENING's reference nodes**:

- The `SimulationNode` ABC and its contract (pure functions, JAX-traceability, immutable state)
- The `NodeMeta` schema and all compliance infrastructure (imported from `maddening.compliance`)
- The framework machinery: `GraphManager`, scheduling, coupling, adaptive timestepping, parameter sweeps
- Patterns and conventions demonstrated by the reference implementations (how to structure `update()`, how to handle boundary inputs, how to write `initial_state()`)
- The surrogate framework: `SurrogateArchitecture`, `SurrogateNode`, `SurrogateTrainer`, `DatasetGenerator`

**What downstream libraries must provide themselves**:

- Their own `NodeMeta` instances with downstream-namespaced `algorithm_id` values (e.g., `MIME-NODE-001`, not `MADD-NODE-*`)
- Their own anomaly entries in a downstream-namespaced anomaly registry
- Their own algorithm guide documents following the template from Section 3
- Their own verification evidence, subject to the two-mode model below

#### Two-Mode Verification Inheritance

Downstream nodes fall into exactly one of two modes. The mode determines whether upstream verification evidence may be cited.

**Mode 1 — Wrapping (unmodified upstream node)**

A downstream node may cite MADDENING's verification evidence if and only if it wraps an upstream MADDENING node **without modifying its numerical logic**. "Unmodified" means the upstream node's `update()` function is called with no changes to its mathematical behaviour — docstring edits, type annotations, internal variable renames, or logging additions do not invalidate the claim; changes to discretization, boundary handling, forcing terms, or numerical parameters do.

When Mode 1 applies, the downstream library's SOUP package must include a **Scope Statement** for each wrapped node that cites:

1. The exact MADDENING version pin (e.g., `maddening==0.3.2`)
2. The specific upstream verification ID (e.g., `MADD-VER-001: Poiseuille flow analytical benchmark`)
3. A declaration that the upstream node's numerical logic is unmodified

```markdown
### CSFFlowNode — Verification Scope Statement

CSFFlowNode wraps MADDENING's LBMPipeNode (MADD-NODE-006) from
MADDENING v0.3.2 without modification to numerical logic.
Upstream verification evidence cited:
- MADD-VER-003: Poiseuille flow analytical benchmark (L2 < 1e-3)
- MADD-VER-004: LBM mass conservation regression test

CSFFlowNode adds CSF-specific parameter presets and boundary
condition configuration, but does not alter the underlying
LBM collision, streaming, or forcing implementations.

**Additional MIME-specific verification** (required even in Mode 1):
- MIME-VER-001: CSF viscosity regime validation (ν = 0.7e-6 m²/s)
```

Even in Mode 1, the downstream library should supplement upstream evidence with its own verification in the specific parameter regimes relevant to its domain. Upstream evidence proves the algorithm is correctly implemented; downstream evidence proves it is correctly applied.

**Mode 1 regression test requirement**: The claim "unmodified numerical logic" must be supported by a concrete test, not merely asserted in prose. The downstream library must include at least one regression test per Mode 1 node that demonstrates identical numerical outputs for identical inputs across the upstream MADDENING node and the downstream wrapper. "Identical" means floating-point equality of the output state arrays for a fixed input state, fixed boundary inputs, and fixed dt — not bit-identical XLA execution graphs or identical intermediate representations. The test must pin the MADDENING version and fail if the output diverges after a MADDENING upgrade, which is the signal to either re-verify the wrapping claim or reclassify the node as Mode 2. This test is the evidentiary anchor for the Scope Statement; without it, the Scope Statement is an unsubstantiated assertion.

**Tolerance specification**: The numerical tolerance used in the regression test must be either (a) explicitly stated in the Mode 1 Scope Statement as a named value with justification (e.g., "atol=1e-12, chosen because this is the expected floating-point rounding bound for this computation on float64"), or (b) directly referenced from the upstream node's `NodeMeta.validated_regimes` table entry for the relevant parameter regime. Undocumented or implicit tolerances — such as a bare `np.allclose()` call relying on default parameters — are not acceptable. The tolerance specification is part of the auditable evidence that the "unmodified" claim means what it says: a tolerance of zero asserts exact equality; a nonzero tolerance asserts near-equality and must explain why exact equality is not expected (e.g., JAX platform-dependent operator fusion order). The Scope Statement must document which case applies.

**Tolerance derivation methodology**: To prevent arbitrary tolerance choices that a Notified Body may challenge, Mode 1 Scope Statements must derive their regression test tolerance using one of the following three tiers:

| Tier | When to Use | Derivation | Typical Value (float64) |
|---|---|---|---|
| **Exact reproducibility** (tolerance = 0) | The upstream node is used unmodified on the same platform (OS, JAX version, hardware) as the upstream verification evidence. No platform-dependent compilation differences are expected. | Assert bitwise equality of output arrays. | `atol=0, rtol=0` |
| **Platform-dependent rounding** | The upstream node is unmodified but the downstream test may run on a different platform (e.g., different GPU, different XLA version) where operator fusion order may differ. | Derive from IEEE 754 ULP (unit in the last place) analysis for the computation's critical path. Typically 10–100× machine epsilon for the accumulated floating-point operations in the node's `update()` function. | `atol=1e-12` to `atol=1e-10` |
| **Algorithmic tolerance** | The upstream node uses an inherently approximate algorithm (e.g., iterative solver, LBM relaxation) where the result depends on convergence criteria rather than exact arithmetic. | Derive from the algorithm's convergence criterion or from the upstream node's `NodeMeta.validated_regimes` entry for the relevant parameter regime. | Varies; must match the upstream convergence bound |

Each Mode 1 Scope Statement must declare which tier applies, state the chosen tolerance value(s), and provide the derivation. Tier 1 requires no further justification beyond confirming platform identity. Tier 2 requires citing the number of floating-point operations in the critical path and the expected ULP accumulation. Tier 3 requires referencing the specific convergence criterion or `validated_regimes` entry. A Scope Statement that states a tolerance without a tier classification or derivation is incomplete.

**Mode 2 — Independent (modified or new physics)**

If the downstream node modifies the upstream node's numerical logic in any way — or implements new physics not present in MADDENING — no upstream verification evidence may be cited. The downstream library must provide its own complete verification evidence: its own benchmarks, its own convergence studies, its own analytical comparisons, registered via `@verification_benchmark` from `maddening.compliance`.

This is the default mode. Mode 1 is the exception that must be explicitly justified.

| Aspect | Mode 1 (Wrapping) | Mode 2 (Independent) |
|---|---|---|
| Upstream verification citable? | Yes, with Scope Statement | No |
| Downstream verification required? | Supplementary (domain-specific regimes) | Full (no upstream evidence available) |
| Downstream NodeMeta required? | Yes (downstream-namespaced) | Yes (downstream-namespaced) |
| Downstream anomaly entries required? | Yes (may cross-reference upstream IDs) | Yes |
| Algorithm guide required? | Yes (may reference upstream doc) | Yes (standalone) |

**Regulatory implication**: A downstream commercial manufacturer assessing MIME as SOUP must understand which MIME nodes operate in Mode 1 vs. Mode 2. For Mode 1 nodes, the manufacturer's SOUP assessment can reference both MIME's and MADDENING's verification evidence — but only if the Scope Statement is present and the MADDENING version pin is exact. For Mode 2 nodes, only MIME's own verification evidence applies. The manufacturer must assess MADDENING and MIME as separate SOUP components regardless of mode.

**Connection to the XLA Shadow (Sections 9.8 and 11)**: The two-mode model defines where MADDENING's *mathematical* verification boundary lies. Even in Mode 1, where upstream mathematical verification is inherited, the execution-layer boundary described in Section 9.8 (and cross-referenced in Section 11, Clause 5.3 row) still applies: MADDENING's functional purity guarantees protect the data-flow layer but not the XLA compilation or GPU hardware layer. The manufacturer's runtime monitoring obligations — including the HealthCheck infrastructure described in Section 9.8 — exist in both modes.

### The Downstream Library's Own Documentation Architecture Document

MIME (and any future downstream library) should have its own `DOCUMENTATION_ARCHITECTURE.md`. Rather than duplicating this entire document, MIME's version should **inherit by reference** and document only what is different or additional.

#### Recommended Structure

```markdown
# MIME Documentation Architecture Plan

## Inheritance Statement

MIME inherits the MADDENING documentation architecture and compliance
schema as described in MADDENING's DOCUMENTATION_ARCHITECTURE.md,
Sections 1–16. This document describes only what MIME adds, overrides,
or extends. For any topic not addressed below, the MADDENING
documentation architecture applies unchanged.

MIME imports all compliance types and tooling from
`maddening.compliance`. MIME uses the same data structures, the same
anomaly registry schema, the same verification benchmark patterns,
and the same SOUP package template as MADDENING.

## What MIME Inherits (by reference — not repeated here)

- Documentation structure and Sphinx tooling (MADDENING Section 1)
- Algorithm documentation template and citation conventions (MADDENING Section 3)
- Bibliography citation system: Pandoc-style `[@Key]` syntax, CI validation via `scripts/check_citations.py`, `bibliography.bib` as centralized reference store (MADDENING Section 3)
- V&V documentation hooks and boundary language (MADDENING Section 4)
- Versioning policy and CHANGELOG conventions (MADDENING Section 5)
- Contributor standards and node authoring checklist (MADDENING Section 7)
- NodeMeta schema and all code-embedded hooks (MADDENING Section 9)
- Anomaly management schema and process (MADDENING Section 9.7)
- SOUP package template (MADDENING Section 10)
- IEC 62304 lifecycle mapping structure (MADDENING Section 11)
- EU MDR alignment approach (MADDENING Section 12)
- QMS compatibility model (MADDENING Section 15)

## What MIME Provides (documented below)

### MIME-Specific Regulatory Boundary Language

[MIME's own intended_use.md — "MIME is a physics engine for
microrobotics research, not a medical device..."]

[MIME's own downstream_integration.md — the MIME → MICROBOTICA
→ Commercial Product chain, with MADDENING as an upstream dependency]

### MIME-Specific Nodes and Algorithm Documentation

[List of MIME nodes, each with MIME-NODE-* algorithm IDs]
[One algorithm guide document per MIME node in docs/algorithm_guide/nodes/]
[NodeMeta on every node with hazard_hints relevant to CSF dynamics]

### MIME-Specific Verification Evidence

[MIME's own verification benchmarks for CSF-relevant parameter regimes]
[MIME's own known_anomalies.yaml with MIME-ANO-* entries]
[Benchmark problems specific to microrobot physics: CSF flow validation,
drug transport verification, magnetic field coupling tests]

### MIME-Specific SOUP Package

[MIME's own soup_package.md]
[Section 8 lists MADDENING at a pinned version with full dependency chain]

### Differences from MADDENING Architecture

[Any areas where MIME's requirements differ from or extend MADDENING's,
e.g., additional UQ requirements for clinical-regime predictions,
domain-specific benchmark standards, additional regulatory standards]
```

#### What Can Genuinely Be Omitted

MIME's document does **not** need to repeat:

- The case study analysis (Section 1 patterns) — these informed MADDENING's design, not MIME's
- The algorithm documentation template itself — just reference it and use it
- The anomaly record schema definition — imported from `maddening.compliance`
- The full IEC 62304 clause-by-clause mapping — reference MADDENING's; MIME follows the same structure
- The EU MDR Annex I/II walkthrough — same principles apply; reference MADDENING's
- The MDCG 2019-11 qualification analysis — MIME uses the same "general-purpose tool, not a medical device" argument
- The QMS compatibility discussion — same model (open-source research tool, commercial entity takes on QMS)
- The commercial boundary model — MIME inherits this from MADDENING's Section 2; it only needs to document MIME's specific position in the chain

#### What MIME Must Write

MIME's document **must** include:

- **MIME's own regulatory boundary statement**: MIME has its own scope (microrobotics physics) that is distinct from MADDENING's (general multiphysics). Its intended use statement must describe what MIME specifically does and does not do.
- **The MIME → MADDENING dependency relationship**: How MIME uses MADDENING, which MADDENING components MIME depends on, and how the SOUP chain is documented.
- **MIME-specific nodes and verification requirements**: What benchmark problems are relevant to CSF dynamics? What parameter regimes must be validated? What acceptance criteria are appropriate?
- **Any extensions to MADDENING's architecture**: If MIME needs additional compliance infrastructure (e.g., domain-specific UQ requirements, biomechanics-specific validation standards, additional regulatory frameworks), those extensions are documented here.
- **MIME's own implementation priority / bootstrap checklist**: Appendix F of this document provides the template; MIME's document should reference it and note completion status.

---

## Appendix A: Case Study Summary Table

| Aspect | 3D Slicer | ITK | iMSTK | MADDENING (Proposed) |
|---|---|---|---|---|
| **Doc tiers** | User + Developer | API + Software Guide + Examples | User docs + Doxygen | User + Algorithm + Developer + Validation + Regulatory + API |
| **Math documentation** | Cite papers | LaTeX in Software Guide + Doxygen | Informal (SPH citations) | LaTeX in Sphinx (MyST) + docstrings |
| **Regulatory language** | "Platform, not product" BSD | FDA guidelines page (educational) | Almost none | EU MDR + FDA + IEC 62304 + MDCG layered model |
| **V&V** | CTest/CDash, per-module tests | Baseline regression, content-addressed data | Absent | Registered benchmarks, auto-generated reports |
| **SOUP readiness** | None | Implicit via governance | None | Explicit SOUP package + anomaly registry + lifecycle mapping |
| **EU MDR alignment** | Not addressed | FDA focus only | Not addressed | Annex I/II mapping, MDCG 2019-11 qualification |
| **IEC 62304 mapping** | Not addressed | Not addressed | Not addressed | Full lifecycle phase mapping |
| **Stability markers** | Community norms | Compat module, migration guides | Inline deprecation notices | Code-level decorators + machine-readable registry |
| **UQ** | N/A | N/A | N/A | Interface hooks from day one |
| **Audit logging** | N/A | CDash (CI level) | CDash (CI level) | Application-level, opt-in |
| **SBOM** | Not published | Not published | Not published | CycloneDX with each release |
| **Anomaly management** | GitHub Issues | GitHub Issues | GitLab Issues | Structured YAML registry + GitHub label taxonomy |
| **Bibliography** | Per-module | Centralized `doxygen.bib` | Sparse inline | Centralized `bibliography.bib` |
| **QMS compatibility** | Informal | Informal (NumFOCUS governance) | Informal | Explicit SOUP/QMS documentation |

---

## Appendix B: Implementation Priority

The mechanisms described in this document should be implemented in this order, balancing immediate regulatory value with effort.

**Philosophy**: Phase 1-2 items build MADDENING's credibility as a research tool and SOUP foundation. Phase 3-4 items specifically prepare MADDENING for Class C SOUP assessment by downstream commercial manufacturers. Phase 5 items support the long-term commercial relationship and broader ecosystem.

### Phase 0: Compliance Tooling Bootstrap (implement first)

These items establish the shared compliance infrastructure that all subsequent phases depend on. The `maddening.compliance` namespace is also the stable API surface for downstream libraries (Section 16).

See Appendix E Phase 0 for the verifiable checklist corresponding to these items.

0a. **`maddening/compliance/` namespace** (Section 16) — create the compliance subpackage with `__init__.py` re-exporting all schema types (`NodeMeta`, `AnomalyRecord`, `ValidationBenchmark`, etc.), decorators (`@verification_benchmark`, `@stability`), and harvesting utilities. Must be importable without JAX. **Note on `@stability`**: In Phase 0–3, the `stability` export is a **no-op identity decorator** — it attaches the stability level as a marker attribute but does not populate a registry or generate reports. This avoids a broken import or missing-module error during Phases 0–3 while allowing downstream libraries to annotate their API surfaces immediately. The functional implementation (registry population, `generate_stability_report()`) lands in Phase 4 (item 30).
0b. **`maddening/compliance/_validate.py` and `__main__.py`** (Section 16) — anomaly registry validator accepting a path argument and optional `--prefix` flag. Expose as `python -m maddening.compliance check-anomalies <path>`.

### Phase 1: Regulatory Foundation (implement now)

These items are essential for EU regulatory readiness and should be prioritized above all other documentation work. All artifacts in this phase are calibrated to **Class III / IEC 62304 Class C** requirements — the most demanding plausible downstream classification — so that they do not need to be retroactively strengthened later.

1. **`docs/validation/known_anomalies.yaml`** (Section 9.7) — start the anomaly registry immediately. This is the single most important artifact for IEC 62304 SOUP assessment.
2. **`docs/validation/soup_package.md`** (Section 10) — start the SOUP package document now, even in skeletal form. A placeholder SOUP package for v0.1.0 that covers Software Identification, Functional Description, and Known Anomalies is more valuable than a perfect one that arrives in Phase 3, because it establishes the pattern and discipline from the start. Flesh out the remaining sections as they become available.
3. **`docs/regulatory/intended_use.md`** (Section 2) — platform positioning statement with EU MDR and MDCG 2019-11 language, explicitly reflecting the commercial boundary model
4. **README disclaimer** (Section 8) — add EU MDR-aware intended use statement now
5. **`CHANGELOG.md`** (Section 5) — start structured changelog with Known Anomalies section
6. **`CITATION.cff`** (Section 14) — trivial to add; serves as both citation and CM artifact
7. **`NodeMeta` dataclass, `StabilityLevel` enum, and `hazard_hints` field** (Sections 9.1, 9.8) — attach to existing nodes. The `hazard_hints` field is a trivial addition to `NodeMeta` and the connection to ISO 14971 hazard identification is high-value and low-cost.
8. **Skeleton `docs/regulatory/downstream_integration.md`** (Section 2) — create skeleton containing: the four-layer chain (MADDENING → MIME → MICROBOTICA → Commercial Product), a one-paragraph commercial responsibility statement, and a forward reference to full content in Phase 3
9. **GitHub Issues label taxonomy** (Section 9.7) — set up anomaly:critical/major/minor + safety-relevant labels
10. **`.github/ISSUE_TEMPLATE/anomaly.md`** (Section 9.7) — create the structured anomaly issue template with mandatory fields (severity, safety relevance, rationale, affected components, affected versions, workaround). This template is the Phase 1 entry point for the three-phase anomaly lifecycle.

### Phase 2: Verification Infrastructure (next milestone)

11. **Algorithm guide template** (Section 3) — write for HeatNode and LBMPipeNode first. Each algorithm guide must include the Implementation Mapping subsection (equation term → Python/JAX function). Code-driven generation of the Implementation Mapping table is recommended but not mandated — the mapping may be maintained manually provided it is complete and reviewed at each release. **CI bridge**: add `scripts/check_impl_mapping.py` — a lightweight script that parses each algorithm guide's Implementation Mapping Markdown table and verifies via `importlib` + `getattr()` that every function qualified name resolves to an existing callable. This runs on every push and catches documentation rot (renamed/removed functions) before the full `NodeMeta.implementation_map` migration in Phase 3.
12. **Verification benchmark decorator and registry** (Section 9.3) — retrofit existing tests
13. **`tests/verification/`** directory — separate verification benchmarks from unit tests
14. **`docs/validation/framework_verification.md`** (Section 4) — aggregate existing evidence
15. **Centralized `bibliography.bib` and citation validation** (Section 3) — collect all academic references in `docs/bibliography.bib`; algorithm guide documents cite using Pandoc-style `[@Key]` syntax with YAML frontmatter `bibliography: ../../bibliography.bib`; **CI bridge**: add `scripts/check_citations.py` — validates that every `[@Key]` citation in algorithm guides resolves to a BibTeX entry in the `.bib` file. Unused entries are warned, dangling citations fail the build.
16. **`docs/regulatory/iec62304_mapping.md`** (Section 11) — IEC 62304 lifecycle mapping (include the Clause 5.3.5 segregation row)
17. **ISO 14971 risk management documentation** (Section 9.8) — document the NodeMeta → hazard identification connection, functional purity safety argument, XLA Shadow qualification, HealthCheck node infrastructure, and anomaly → risk management workflow
18. **`HealthCheckNode` base class** (Section 9.8) — implement the HealthCheck base node in `maddening/nodes/health_check.py`. Provides NaN/Inf trapping, physical boundary checks, and statistical moment monitoring as a graph-integrated node. Downstream libraries instantiate and configure for their context of use.
19. **Algorithm guides declare Mode 1 / Mode 2 status** (Section 16) — when a downstream library's algorithm guide describes a node that wraps an upstream MADDENING node, the guide must explicitly state whether the node operates in Mode 1 (Wrapping) or Mode 2 (Independent), and the MADDENING version pin if Mode 1. This declaration is part of the algorithm guide template, not a separate artifact.

### Phase 3: Audit, Provenance, and EU Regulatory Docs (when downstream commercial interest materialises)

20. **`NodeMeta.implementation_map`** (Section 3) — migrate the Implementation Mapping from a manually maintained Markdown table to a machine-readable `NodeMeta.implementation_map` dictionary. Add a Sphinx build step that verifies via `getattr()` that every function name in the mapping resolves to an existing callable in the source code, failing the documentation build on stale references. This closes the traceability loop between algorithm documentation and source code.
21. **`SimulationProvenance`** (Section 9.2) — opt-in provenance tracking
22. **`AuditLogger`** (Section 9.6) — opt-in audit logging
23. **`docs/regulatory/eu_mdr_guidelines.md`** (Section 12) — EU MDR Annex I/II alignment guide
24. **`docs/regulatory/mdcg_2019_11.md`** (Section 13) — qualification and classification guide
25. **Complete and expand `docs/regulatory/downstream_integration.md`** (Section 2) — flesh out the Phase 1 skeleton with full per-layer responsibility documentation, classification-by-use-mode analysis, and complete commercial boundary content
26. **`docs/validation/cou_template.md`** (Section 4) — downstream user COU template
27. **SBOM generation** (Section 14) — add CycloneDX to release process
28. **`SECURITY.md`** (Section 8) — vulnerability reporting process

### Phase 4: UQ, Stability, and Commercial Readiness (when surrogate framework matures)

29. **`UncertaintySpec` and `uncertainty_spec()`** (Section 9.4) — UQ interface hooks
30. **`@stability` decorator** (Section 9.5) — code-level stability markers
31. **`docs/algorithm_guide/uq/`** (Section 6) — UQ documentation namespace
32. **`docs/regulatory/fda_guidelines.md`** (Section 2) — FDA alignment guide (secondary market)

### Phase 5: Full Documentation Site (when user base grows)

33. **Sphinx documentation build** — full `docs/` site with auto-generated API reference
34. **Per-node verification reports** — auto-generated from CI benchmark results
35. **`GOVERNANCE.md`** (Section 8) — decision-making process

---

## Appendix C: Relationship to Existing Documentation

This plan builds on MADDENING's existing documentation:

| Existing Document | Status | Proposed Changes |
|---|---|---|
| `README.md` | Good | Add EU MDR-aware disclaimer, citation, documentation table |
| `DESIGN.md` | Excellent (615 lines) | Keep as-is; reference from `docs/developer_guide/architecture.md` and IEC 62304 mapping |
| `ROADMAP.md` | Excellent | Keep as-is; reference from docs |
| Module READMEs | Good | Migrate content to `docs/` over time; keep READMEs as pointers |
| Node docstrings | Good (NumPy style) | Add `meta` classvar; keep docstrings for IDE/help() |
| Test suite (545+ tests) | Good | Add `tests/verification/` for formal benchmarks; retrofit decorators |
| `pyproject.toml` | Good | Add `docs` extra for sphinx dependencies |

---

## Appendix D: EU Regulatory Roadmap for the MADDENING Ecosystem

This appendix provides a concise, practical roadmap for the full ecosystem from research to clinical use, structured around the three-track model: open-source research tools evolving independently, evidence generation using those tools, and the commercial regulatory pathway.

### Track 1: Open-Source Research Tools

MADDENING, MIME, and MICROBOTICA are all open-source research tools that evolve independently of any regulatory timeline.

#### Layer 1: MADDENING — Computational Foundation

**Status**: Research software, LGPL licence, no medical purpose.

**Regulatory requirement**: None. MADDENING is not a medical device.

**Documentation requirement**: Maintain SOUP-quality documentation (this document) so any downstream commercial manufacturer can perform their regulatory obligations.

**Key standards**: IEC 62304 (as SOUP provider), ASME V&V 40 (code verification)

**Deliverables per release**:
- SOUP package document
- Known anomalies registry
- Verification benchmark results
- CHANGELOG with Known Anomalies section
- SBOM

#### Layer 2: MIME — Physics Engine

**Status**: Open-source physics engine for microrobotics, built on MADDENING. LGPL licence. No medical purpose.

**Regulatory requirement**: None. MIME is not a medical device.

**MIME's relationship to MADDENING**: MIME uses MADDENING as a dependency. When MIME is itself used as SOUP in a regulated product, the manufacturer must assess both MIME and MADDENING as SOUP components.

#### Layer 3: MICROBOTICA — Research Simulator

**Status**: Open-source research simulator for microrobot-assisted drug delivery in CSF. No medical purpose as an open-source tool. Not a medical device. Does not seek CE marking.

**Regulatory requirement**: None. MICROBOTICA is a research tool.

**MICROBOTICA's role**: MICROBOTICA generates pre-clinical simulation data, validates algorithms against experimental results, and provides a platform for researchers to publish with. It is the open-source foundation on which a commercial clinical product may be built.

### Track 2: Research-to-Evidence

This track describes the process by which the open-source tools are used to generate the pre-clinical and clinical evidence that a future commercial product will need.

| Phase | Activity | Output |
|---|---|---|
| **Algorithm validation** | Compare simulation predictions to bench-top experiments (phantom CSF, microfluidic models) | Published papers validating numerical methods against physical measurements |
| **In vitro validation** | Compare to controlled in vitro experiments with microrobots in CSF-analog fluids | Validation dataset for specific parameter regimes |
| **Ex vivo / animal studies** | Compare simulation predictions to ex vivo brain tissue / animal model results | Pre-clinical evidence establishing simulation-reality correlation |
| **Clinical feasibility** | Pilot use in clinical settings (under research ethics approval, not as a medical device) | Feasibility data informing the commercial product's clinical evaluation |

All of this work is research — it does not require regulatory approval (beyond standard research ethics), it does not create a medical device, and it does not require a QMS. But the data generated will be essential for any future commercial product's clinical evaluation report.

### Track 3: Commercial Regulatory Pathway

This track begins when a commercial entity (spin-out, licensee, or partner company) decides to build a regulated clinical product on top of the MICROBOTICA ecosystem.

#### The Commercial Product

**Status**: SaMD with medical purpose, built on MADDENING/MIME/MICROBOTICA by a commercial entity that takes on EU MDR manufacturer obligations.

**Classification depends on intended use mode**:

| Intended Use Mode | EU MDR Class | IEC 62304 Class | IMDRF Category |
|---|---|---|---|
| Intraoperative digital twin (surgical decision support) | **Class IIb minimum; likely Class III** | Class C | III–IV |
| RL policy generation/validation for microrobot control | **Class III** | Class C | IV |

The classification analysis is driven by the operating environment: microrobot drug delivery in CSF spaces adjacent to the striatum and substantia nigra. Erroneous simulation output during a live intraoperative procedure could directly cause irreversible neurological harm or death.

#### Class III Conformity Assessment Pathway

**Conformity assessment route** (EU MDR Article 52):

| Class | Conformity Assessment Route | Notified Body Involvement |
|-------|---------------------------|------------------------|
| **IIb** | Annex IX (Chapter I + II) or Annex X + XI (Part A) | QMS audit + technical file sampling |
| **III** | Annex IX (Chapter I + II) or Annex X + XI (Part A) | **Full QMS audit + full technical file review** |

For **Class III**, the Notified Body performs:
- **Full QMS audit** per ISO 13485 (initial + annual surveillance)
- **Full technical file review** — every section of the technical file is examined, not sampled. This includes the complete SOUP assessment of MADDENING: SOUP package document, known anomalies evaluation, architectural analysis, verification evidence, segregation analysis.
- **Assessment of clinical evidence** — the clinical evaluation report and PMCF plan are reviewed in detail
- **Annual surveillance audits** including review of post-market surveillance data and PMCF results

#### Clinical Evaluation Requirements for Class III

EU MDR Annex XIV and MEDDEV 2.7/1 rev. 4 define clinical evaluation requirements. For Class III SaMD:

1. **Clinical Evaluation Report (CER)**: Must demonstrate that the device achieves its intended performance and that residual risks are acceptable in the context of the clinical benefits. For simulation software, this means demonstrating that the commercial product's predictions are clinically accurate — that simulated CSF flow, drug concentration profiles, and microrobot trajectories correlate with reality.

2. **Clinical evidence sources**: Clinical evidence may come from clinical investigations, published literature, and/or equivalence to predicate devices. For a novel SaMD, equivalence arguments will be difficult; clinical investigations (likely prospective studies comparing simulation predictions to intraoperative measurements) will be needed. Published research using MICROBOTICA (Track 2) provides supporting evidence but likely does not substitute for the manufacturer's own clinical investigations.

3. **MADDENING's role in clinical evidence**: MADDENING provides the computational foundation but does not itself generate clinical evidence. MADDENING's verification evidence (analytical benchmarks, convergence studies) establishes that the numerical methods are correctly implemented. Clinical evidence — demonstrating that the simulated physics matches real patient physiology — is the commercial manufacturer's responsibility.

4. **Post-Market Clinical Follow-up (PMCF)**: Class III devices require a PMCF plan and ongoing PMCF studies. This is a long-term commitment requiring organisational capacity for continuous clinical data collection.

#### Post-Market Surveillance for Class III

EU MDR Article 83 and Annex III require:
- **Post-market surveillance plan**: active, systematic monitoring of clinical performance
- **Periodic Safety Update Report (PSUR)**: annual for Class III, summarizing PMS data and conclusions
- **Vigilance reporting**: serious incidents must be reported to competent authorities within defined timeframes
- **Trend reporting**: statistically significant increases in incidents must be reported
- **MADDENING's contribution**: MADDENING's anomaly management system (Section 9.7) and structured release process support the commercial manufacturer's ability to trace post-market findings back to specific SOUP versions and components

#### Key Standards for the Commercial Product

- EU MDR (EU 2017/745) — the regulation itself
- EN ISO 13485:2016 — QMS
- IEC 62304:2006+AMD1:2015 — software lifecycle
- ISO 14971:2019 — risk management
- IEC 62366-1:2015 — usability engineering
- MDCG 2019-11 — software qualification and classification
- MDCG 2019-16 — cybersecurity
- IMDRF SaMD N10/N12/N23 — SaMD risk framework
- MEDDEV 2.7/1 rev. 4 — clinical evaluation guidance
- MDCG 2020-1 — clinical evaluation guidance for MDR

#### Technical File Contents (EU MDR Annex II)

1. Device description and specification (including MADDENING/MIME as SOUP components)
2. Information supplied by the manufacturer (labels, IFU)
3. Design and manufacturing information (software architecture, IEC 62304 lifecycle, SOUP list including MADDENING with Class C assessment)
4. General safety and performance requirements checklist (GSPR mapping)
5. Benefit-risk analysis and risk management (ISO 14971 risk management file, including SOUP failure mode analysis)
6. Product verification and validation (clinical evaluation report, PMCF plan, computational V&V evidence)
7. Post-market surveillance plan (including PSUR schedule, vigilance reporting procedures)

### Timeline

This timeline is illustrative. The three tracks operate independently — the open-source tools evolve on their own schedule, and the commercial pathway begins when a commercial entity is ready.

| Phase | Timeframe | MADDENING / MIME / MICROBOTICA (open source) | Commercial Product |
|-------|-----------|---|---|
| **Research** | Now | Develop frameworks, establish SOUP documentation, publish research | (Does not yet exist) |
| **Evidence generation** | +1-3 years | Mature V&V evidence, expand node library, establish Class C-ready SOUP packages | (Does not yet exist — research publications build the evidence base) |
| **Commercial formation** | +2-4 years | Continue independent development, provide SOUP documentation | Entity formed; establish ISO 13485 QMS, engage regulatory consultant, select Notified Body |
| **Clinical evidence** | +3-5 years | Maintain version stability for pinned commercial releases | Design and conduct clinical investigations (prospective studies) |
| **Conformity assessment** | +5-7 years | Provide release-locked SOUP package, respond to NB information requests | Full Notified Body review: QMS audit + technical file + clinical evidence |
| **CE marking** | +5-7 years | Maintain SOUP quality releases | CE mark obtained, market placement |
| **Post-market** | Ongoing | Monitor anomalies, security updates, maintain SOUP packages | PMCF studies, PSUR (annual), vigilance reporting |

**Note on timeline**: The +5-7 year timeline to CE marking for a Class III SaMD is realistic, not pessimistic. The clinical evidence generation phase is the longest and most uncertain. Starting MADDENING's documentation architecture now — before any commercial entity exists — is a strategic decision: it avoids the costly situation of needing to retrospectively generate SOUP documentation under time pressure during conformity assessment.

### Key Standards Applied at Each Layer

| Standard | MADDENING / MIME / MICROBOTICA | Commercial Product |
|----------|---|---|
| **EU MDR (2017/745)** | Not applicable (not medical devices) | Full compliance required (Class III pathway) |
| **IEC 62304** | SOUP provider (Class C-ready documentation) | Full lifecycle (Class C) |
| **ISO 14971** | Not applicable (hazard hints provided voluntarily) | Full risk management file |
| **ISO 13485** | Not applicable (no QMS) | QMS certification required (NB-audited annually) |
| **ASME V&V 40** | Code verification evidence | Full credibility assessment |
| **MDCG 2019-11** | Not medical devices | SaMD classification (Class IIb–III by use mode) |
| **MDCG 2019-16** | SBOM, SECURITY.md | Full cybersecurity documentation |
| **IMDRF SaMD N10/N12/N23** | Not applicable | Risk classification (Category III–IV) |
| **MEDDEV 2.7/1 rev. 4** | Not applicable | Clinical evaluation guidance |
| **EU MDR Annex XIV** | Not applicable | Clinical evidence requirements (Class III) |

### What MADDENING's Documentation Feeds Into

```
MADDENING Documentation                  Commercial Product Technical File
─────────────────────────                 ──────────────────────────────────
SOUP package document          ───►       SOUP list (Annex II §4)
known_anomalies.yaml           ───►       Risk management file (ISO 14971)
Algorithm guide (equations)    ───►       Design documentation (Annex II §4)
Verification benchmarks        ───►       V&V evidence (Annex II §6)
SBOM                           ───►       Cybersecurity documentation
intended_use.md                ───►       Component description (Annex II §1)
CITATION.cff + version hashes  ───►       Configuration management (IEC 62304 §8)
CHANGELOG.md                   ───►       Change control evidence
NodeMeta.hazard_hints          ───►       Hazard identification (ISO 14971)
```

This traceability chain is the fundamental reason why documentation architecture is a first-class concern for MADDENING: every document described in this plan has a specific downstream consumer in the regulatory process, regardless of which commercial entity ultimately builds the regulated product.

---

## Appendix E: Project Setup Checklist

This appendix provides a concrete, verifiable checklist of everything that must be in place in the MADDENING repository to support the documentation architecture described in this document. Each item is phrased as a yes/no verification that can be confirmed by inspecting the repository. Items are organised by implementation phase (matching Appendix B) so it is clear which items are required immediately versus later.

> **Consistency guardrail**: This checklist, Section 11 (IEC 62304 mapping), and Appendix B (Implementation Priority) are tightly coupled. When editing any section of the main document that introduces a new artifact, mechanism, or requirement, the editor **must** verify that corresponding entries exist in all three locations: (1) Appendix B lists the implementation item in the correct phase, (2) Appendix E contains a verifiable checklist item for the artifact, and (3) Section 11 references the artifact in the appropriate IEC 62304 lifecycle phase row if it provides evidence for that phase. Failure to maintain this three-way consistency will cause gaps that surface during downstream SOUP assessment or Notified Body review. Appendix F (Downstream Bootstrap) should also be checked whenever a new inheritable schema-layer artifact is introduced.

### Phase 0 — Compliance Tooling Bootstrap

These items establish the shared compliance infrastructure that all subsequent phases — and all downstream libraries (Section 16) — depend on. They must be completed before Phase 1 items can be verified.

**Compliance namespace:**

- [x] `maddening/compliance/__init__.py` exists and re-exports at minimum: `NodeMeta`, `EdgeMeta`, `ValidatedRegime`, `Reference`, `StabilityLevel`, `UQReadiness`, `AnomalyRecord`, `AnomalySeverity`, `SafetyRelevance`, `ResolutionStatus`, `ValidationBenchmark`, `BenchmarkType`, `verification_benchmark`, `stability` (Section 16)
- [x] `from maddening.compliance import NodeMeta` succeeds without importing JAX (Section 16)
- [x] `from maddening.compliance import validate_anomaly_registry` succeeds (Section 16)
- [x] `from maddening.compliance import stability` succeeds and `stability` is a **no-op identity decorator** — applying `@stability(StabilityLevel.STABLE)` to a class or function succeeds without error but does not populate a registry or generate reports. The functional implementation lands in Phase 4 (Appendix B item 30). (Section 16)

**Anomaly registry validator:**

- [x] `maddening/compliance/_validate.py` exists with `validate_anomaly_registry(path, *, prefix="") -> list[str]` function (Section 16)
- [x] `python -m maddening.compliance check-anomalies <path>` is a valid invocation that validates a YAML file and exits 0 on success, nonzero on failure (Section 16)
- [x] Validator accepts a `--prefix` argument for downstream namespace enforcement (e.g., `--prefix MIME-ANO-`) (Section 16)
- [x] Validator correctly accepts a minimal valid `known_anomalies.yaml` containing: `schema_version`, a version field, `generated_date`, and an empty `anomalies` list (Section 16)

### Phase 1 — Regulatory Foundation

**Repository root files:**

- [x] `CHANGELOG.md` exists at repository root and follows [Keep a Changelog](https://keepachangelog.com/) format with the additional sections: `Verification`, `Security`, and `Known Anomalies` (Section 5)
- [x] `CITATION.cff` exists at repository root and contains at minimum: `cff-version`, `title`, `type: software`, `version`, `date-released`, `license`, `repository-code`, and at least one entry in `authors` with `family-names` and `given-names` (Section 14)
- [x] `README.md` contains the EU MDR-aware disclaimer block: "MADDENING is research software. It is not a medical device as defined by EU MDR (EU 2017/745)..." and references `docs/regulatory/` for details (Section 8)

**docs/ directory structure:**

- [x] `docs/` directory exists
- [x] `docs/regulatory/` directory exists
- [x] `docs/validation/` directory exists

**docs/regulatory/intended_use.md:**

- [x] File exists at `docs/regulatory/intended_use.md` (Section 2)
- [x] Contains a **Platform Positioning Statement** that identifies MADDENING as a general-purpose computational framework, states it is not a medical device under EU MDR Article 2(1), references MDCG 2019-11 Section 3.2 general-purpose tool language, and states MADDENING is SOUP under IEC 62304 when used in a regulated product (Section 2)
- [x] Contains a **Layered Responsibility Model** table assigning responsibilities between MADDENING, downstream manufacturer, and downstream user (Section 2)
- [x] Contains a **Commercial Boundary Statement** explaining that the regulated clinical product is built by a downstream commercial entity, not by the MADDENING project (Section 2)
- [x] Contains a **Cybersecurity Boundary Statement** (MDCG 2019-16): MADDENING assumes trusted inputs; the commercial integration layer is responsible for input sanitization, parameter validation, and graph topology validation before data reaches the framework (Section 2)
- [x] Contains an **LGPL Replaceability Statement**: LGPL-3.0 "replaceability" obligation is satisfied via Python's module system — the end user can replace the `maddening` package with a modified version by installing it into the same Python environment, with no special linking or build steps required (Section 2)

**Anomaly lifecycle (three-phase model):**

- [x] `.github/ISSUE_TEMPLATE/anomaly.md` exists with mandatory fields: title, severity (critical/major/minor), safety relevance (safety_relevant/not_safety_relevant/context_dependent), safety relevance rationale, affected components, affected versions, workaround (if any) (Section 9.7)
- [x] The three-phase anomaly lifecycle (Discovery → Formalization → Verification) is documented in `CONTRIBUTING.md` or `docs/developer_guide/testing_standards.md` (Section 9.7)

**docs/regulatory/downstream_integration.md (skeleton):**

- [x] File exists at `docs/regulatory/downstream_integration.md` (Section 2)
- [x] Contains the four-layer dependency chain: MADDENING → MIME → MICROBOTICA → [Commercial Product], with each layer's status (open source / regulated) clearly stated (Section 2)
- [x] Contains a one-paragraph statement that the commercial layer is responsible for EU MDR manufacturer obligations including ISO 13485 QMS, Notified Body relationship, post-market surveillance, and manufacturer liability (Section 2)
- [x] Contains a forward reference noting that full content (per-layer responsibility details, classification analysis, commercial boundary documentation) will be completed in Phase 3 (Section 2, Appendix B)

**docs/validation/known_anomalies.yaml:**

- [x] File exists at `docs/validation/known_anomalies.yaml` (Section 9.7)
- [x] File is valid YAML (parseable by `yaml.safe_load()`)
- [x] Contains `schema_version` field (e.g., `"1.0"`)
- [x] Contains `maddening_version` field matching the current release version
- [x] Contains `generated_date` field in ISO 8601 format
- [x] Contains `anomalies` key (list; may be empty for an initial release, but the key must exist)
- [x] Every anomaly entry (if any) contains all required fields: `anomaly_id`, `title`, `description`, `severity`, `safety_relevance`, `safety_relevance_rationale` (Section 9.7)
- [x] No duplicate `anomaly_id` values across entries

**docs/validation/soup_package.md (skeleton):**

- [x] File exists at `docs/validation/soup_package.md` (Section 10)
- [x] Contains **Section 1: Software Identification** with at minimum: name, version, release date, licence, source repository, Python version, primary dependencies (Section 10)
- [x] Contains **Section 2: Functional Description** with at minimum: a brief description of core framework capabilities and a "Capabilities NOT provided" list (Section 10)
- [x] Contains **Section 3: Known Anomalies** with at minimum: a reference to `known_anomalies.yaml` and a summary count table (Section 10)
- [x] Remaining sections (4–8) may be stubs with "To be completed in Phase 2/3" notes

**NodeMeta dataclass:**

- [x] `NodeMeta` dataclass is defined in `maddening/core/metadata.py` (Section 9.1)
- [x] `NodeMeta` includes all fields: `algorithm_id`, `algorithm_version`, `stability`, `description`, `governing_equations`, `discretization`, `assumptions`, `limitations`, `validated_regimes`, `references`, `uq_readiness`, `deprecation_notice`, `hazard_hints` (Sections 9.1, 9.8)
- [x] `StabilityLevel` enum is defined with values: `EXPERIMENTAL`, `PROVISIONAL`, `STABLE`, `DEPRECATED` (Section 9.1)
- [x] `UQReadiness` enum is defined with values: `NOT_READY`, `PARAMETER_SWEEP`, `FULL` (Section 9.1)
- [x] `SimulationNode` has a `meta: ClassVar[Optional[NodeMeta]] = None` attribute (Section 9.1)
- [x] At least one existing `SimulationNode` subclass has `meta` attached as a ClassVar with at minimum: `algorithm_id`, `stability`, `description`, `assumptions`, `limitations`, `hazard_hints` (Sections 9.1, 9.8, 9.9)

**GitHub repository settings:**

- [x] GitHub Issues label `anomaly:critical` exists (red) (Section 9.7)
- [x] GitHub Issues label `anomaly:major` exists (orange) (Section 9.7)
- [x] GitHub Issues label `anomaly:minor` exists (yellow) (Section 9.7)
- [x] GitHub Issues label `safety-relevant` exists (purple) (Section 9.7)
- [x] GitHub Issues label `soup-assessment` exists (blue) (Section 9.7)
- [x] GitHub Issues label `known-anomaly` exists (gray) (Section 9.7)

**CI integration:**

- [x] `scripts/check_anomalies.py` exists and validates `known_anomalies.yaml` (Section 9.7)
- [x] CI pipeline runs `scripts/check_anomalies.py` on every push and fails if YAML schema validation errors are found (Section 9.7)
- [ ] CI emits **warnings** (non-blocking) for open `known-anomaly` issues lacking a YAML entry (Phase 1 anomalies still under investigation) (Section 9.7)
- [ ] Release gate implements the **three-tier model** (Section 9.7):
  - [ ] **Tier 1**: every closed issue labeled `safety-relevant` must have a YAML entry — no grace period, hard gate
  - [ ] **Tier 2**: every closed issue labeled `anomaly:critical` or `anomaly:major` (without `safety-relevant`) must have a YAML entry — no grace period
  - [ ] **Tier 3**: closed `anomaly:minor` issues (without `safety-relevant`) — CI warning after one release cycle, blocking gate after two cycles without YAML formalization
- [ ] `scripts/check_anomalies.py` **automates Tier 3 cycle counting**: the script compares each closed `anomaly:minor` issue's close date (via GitHub API) against release timestamps in `CHANGELOG.md` or git tags to determine grace period status — manual cycle tracking is not acceptable (Section 9.7)

### Phase 2 — Verification Infrastructure

**Verification test directory:**

- [x] `tests/verification/` directory exists and is separate from unit tests (Section 4)
- [x] At least one test file exists in `tests/verification/` (Section 4)

**Algorithm guide documents:**

- [x] `docs/algorithm_guide/` directory exists (Section 3)
- [x] `docs/algorithm_guide/nodes/` directory exists (Section 3)
- [x] `docs/algorithm_guide/nodes/_template.md` exists and matches the template in Section 3 (Section 3)
- [x] At least one node has a complete algorithm guide document (recommended: `heat_node.md` or `lbm_pipe_node.md`) containing all template sections: Summary, Governing Equations, Discretization, Implementation Mapping, Assumptions, Validated Physical Regimes, Known Limitations, Stability Conditions, State Variables, Parameters, Boundary Inputs, References, Verification Evidence, Changelog (Section 3)
- [x] The Implementation Mapping subsection traces every governing equation term to a specific Python/JAX function, with no silent omissions — terms handled by JAX primitives are documented as such (Section 3)

**Implementation Mapping CI bridge (documentation rot prevention):**

- [x] `scripts/check_impl_mapping.py` exists and parses each algorithm guide's Implementation Mapping Markdown table, extracting function qualified names from the "Implementation" column (Section 3)
- [x] The script verifies via `importlib` + `getattr()` that every extracted function name resolves to an existing callable in the source code (Section 3)
- [x] CI runs `scripts/check_impl_mapping.py` on every push and fails if any function name is stale (renamed or removed without updating the algorithm guide) (Section 3)

**Verification benchmarks:**

- [x] `maddening/core/validation.py` exists with `ValidationBenchmark` dataclass, `_BENCHMARK_REGISTRY`, and `@verification_benchmark` decorator (Section 9.3)
- [x] At least one test is decorated with `@verification_benchmark` and registered in the benchmark registry (Section 9.3)

**Framework verification document:**

- [x] `docs/validation/framework_verification.md` exists (Section 4)
- [x] Contains at minimum: link to CI system, total test count, platforms tested (OS/Python/JAX versions) (Section 4)

**Bibliography and citation validation:**

- [x] `docs/bibliography.bib` exists at `docs/bibliography.bib` (Section 3)
- [x] Contains at least one BibTeX entry referenced by a node's algorithm guide document (Section 3)
- [x] Algorithm guide documents use Pandoc-style `[@Key]` citation syntax where `Key` matches a BibTeX entry in `docs/bibliography.bib` (Section 3)
- [x] Algorithm guide documents include YAML frontmatter with `bibliography: ../../bibliography.bib` for Pandoc `--citeproc` processing (Section 3)
- [x] Each References section includes human-readable inline descriptions alongside `[@Key]` markers so documents are readable without Pandoc rendering (Section 3)
- [x] `scripts/check_citations.py` exists and validates that every `[@Key]` citation in algorithm guide Markdown files resolves to a BibTeX entry in `docs/bibliography.bib` (Section 3)
- [x] CI runs `scripts/check_citations.py` on every push and fails on dangling citations (cited key not in `.bib` file) (Section 3)
- [x] Template files prefixed with `_` (e.g., `_template.md`) are excluded from citation scanning to avoid false positives from example keys (Section 3)

**IEC 62304 lifecycle mapping:**

- [x] `docs/regulatory/iec62304_mapping.md` exists (Section 11)
- [x] Contains the lifecycle phase mapping table from Section 11 covering all IEC 62304 phases: development planning (5.1), requirements analysis (5.2), architectural design (5.3) including the Clause 5.3.5 segregation row, detailed design (5.4), unit implementation (5.5), unit verification (5.6), integration testing (5.7), system testing (5.8), release (5.9) (Section 11)

**NodeMeta coverage:**

- [x] Every existing `SimulationNode` subclass (BallNode, TableNode, SpringDamperNode, RigidBody2DNode, HeatNode, LBMPipeNode) has `meta` attached as a ClassVar (Section 9.1)
- [x] Each node's `meta` contains at minimum: `algorithm_id`, `stability`, `description`, `assumptions`, `limitations`, `hazard_hints` (Sections 9.1, 9.8)
- [x] `validated_regimes` and `hazard_hints` have non-overlapping scope per the distinction in Section 9.1: quantitative parameter-bound risks in `validated_regimes`, qualitative non-parameter-bound risks in `hazard_hints` (Section 9.1)

**HealthCheck base node:**

- [x] `maddening/nodes/health_check.py` exists with `HealthCheckNode(SimulationNode)` base class (Section 9.8)
- [x] `HealthCheckNode` accepts check configuration (finite checks, bound checks, moment checks) via parameters (Section 9.8)
- [x] `HealthCheckNode.update()` receives monitored state via boundary inputs (edges from monitored nodes) and writes check results (pass/fail flags, check values) to its own state (Section 9.8)
- [x] `HealthCheckNode` does not halt the simulation — check results are outputs that the application layer acts on (Section 9.8)
- [x] `HealthCheckNode` has a `meta` ClassVar with `algorithm_id="MADD-NODE-HEALTH-001"`, `stability=EXPERIMENTAL`, and `hazard_hints` including the Algorithmic Diversity Principle warning (Section 9.8)
- [x] At least one test demonstrates `HealthCheckNode` detecting a NaN in a monitored field (Section 9.8)

**Algorithm guide Implementation Mapping and Mode declaration:**

- [x] The algorithm guide template (`_template.md`) includes the Implementation Mapping subsection as mandatory (Section 3)
- [x] Code-driven generation of the Implementation Mapping table is noted as recommended but not mandated in the template (Section 3, Appendix B)
- [x] The algorithm guide template includes a field or section for declaring Mode 1 (Wrapping) or Mode 2 (Independent) status when the guide describes a downstream node (Section 16). For MADDENING's own nodes, this field is omitted or set to "N/A — upstream reference implementation."

### Phase 3 — Audit, Provenance, and Full Regulatory Docs

**NodeMeta.implementation_map (migration from Markdown table):**

- [x] `NodeMeta` includes an `implementation_map` field: `dict[str, str]` mapping equation term descriptions to Python function qualified names (Section 3)
- [x] At least one node's `meta.implementation_map` is populated and matches the content of its algorithm guide's Implementation Mapping Markdown table (Section 3)
- [ ] A Sphinx build step verifies via `getattr()` that every function name in `implementation_map` resolves to an existing callable — the documentation build fails if a function is renamed without updating the mapping (Section 3)

**Audit and provenance modules:**

- [x] `maddening/core/audit.py` exists with `AuditLogger`, `AuditSink` (Protocol), `NullSink`, `JSONFileSink` as described in Section 9.6 (Section 9.6)
- [x] `maddening/core/provenance.py` exists with `SimulationProvenance` dataclass as described in Section 9.2 (Section 9.2)

**Full regulatory documents:**

- [x] `docs/regulatory/eu_mdr_guidelines.md` exists with all sections described in Section 12: Annex I GSPR mapping (including GSPR 1, 3, 4, Section 17), Annex II technical file mapping, MDCG 2019-16 cybersecurity (Section 12)
- [x] `docs/regulatory/mdcg_2019_11.md` exists with: Step 1 qualification analysis (why MADDENING is not a medical device), Step 2 classification analysis (how a downstream commercial product would be classified), Rule 11 analysis table, IMDRF SaMD alignment (Section 13)
- [x] `docs/regulatory/downstream_integration.md` is complete (not skeleton): contains full per-layer responsibility documentation, classification-by-use-mode analysis, commercial boundary details, and all content described in Section 2 (Section 2, Appendix B)

**Validation documents:**

- [x] `docs/validation/cou_template.md` exists with the Context of Use template from Section 4: question of interest, quantities of interest, model influence, decision consequence, risk assessment, credibility goals, evidence plan (Section 4)

**Security:**

- [x] `SECURITY.md` exists at repository root (Section 8)
- [x] Contains: vulnerability reporting contact (email or mechanism), response timeline commitment, supported versions table listing which MADDENING versions receive security updates (Section 8)

**SBOM:**

- [ ] SBOM generation is integrated into the release process using CycloneDX format (Section 14)
- [ ] SBOM artifact (JSON or XML) is attached to each GitHub release as a release asset (Section 14)

### Phase 4 — UQ and Stability Machinery

**UQ interface:**

- [x] `UncertaintySpec` dataclass and `UncertainParameter` dataclass are defined in `maddening/core/uq.py` (Section 9.4)
- [x] `SimulationNode` exposes `uncertainty_spec() -> Optional[UncertaintySpec]` method (Section 9.4)

**Stability decorator:**

- [x] `@stability` decorator is implemented in `maddening/core/stability.py` (Section 9.5)
- [x] `_STABILITY_REGISTRY` global dict is maintained by the decorator (Section 9.5)
- [x] `generate_stability_report() -> str` function produces valid Markdown output (Section 9.5)
- [x] `@stability` is applied to all public API surfaces (SimulationNode subclasses, GraphManager, SurrogateArchitecture subclasses, SurrogateTrainer, DatasetGenerator) (Section 9.5)

**UQ documentation:**

- [x] `docs/algorithm_guide/uq/` directory exists (Section 6)
- [x] `docs/algorithm_guide/uq/index.md` exists with: UQ roadmap, capability mapping table (V&V 40 requirement → MADDENING capability → status) from Section 6 (Section 6)

### Phase 5 — Full Documentation Site

**Sphinx build:**

- [ ] `docs/conf.py` exists with Sphinx configuration (Section 1)
- [ ] `sphinx-build -b html docs/ docs/_build/` completes without errors (Section 1)
- [ ] `docs/index.rst` exists and serves as documentation root (Section 1)

**Complete algorithm guide coverage:**

- [ ] Every `SimulationNode` subclass has a corresponding algorithm guide document in `docs/algorithm_guide/nodes/` (Section 3)
- [ ] Every surrogate architecture has a corresponding document in `docs/algorithm_guide/surrogates/` (Section 3)

**Complete verification coverage:**

- [ ] Every physics node has at least one registered `@verification_benchmark` test (Section 9.3)
- [ ] CI generates per-node verification reports in `docs/validation/node_verification/` from benchmark results (Section 4)
- [ ] Each verification report in `docs/validation/node_verification/` contains: benchmark description, comparison methodology, results (plots/tables/error norms), version tested, date (Section 4)

**Complete SOUP package:**

- [ ] `docs/validation/soup_package.md` is fully populated with all 8 sections: Software Identification, Functional Description, Known Anomalies, Verification Evidence (with unit verification traceability table), IEC 62304 Lifecycle Activities, Configuration Management, Anomaly Management Policy, Dependencies (Section 10)

**FDA alignment (secondary market):**

- [ ] `docs/regulatory/fda_guidelines.md` exists with: ASME V&V 40 requirements, FDA computational modeling guidance alignment, how to cite MADDENING in an FDA submission (Section 2)

### The Spin-Out / Licensing Pathway

For reference, the practical steps to transition from "research using MICROBOTICA" to "commercial product built on MICROBOTICA" would typically involve:

1. **Entity formation**: Incorporate a company (BV in the Netherlands, or equivalent). This entity becomes the EU MDR manufacturer with all associated legal obligations.
2. **QMS establishment**: Implement ISO 13485 QMS. This requires: quality manual, documented procedures, management reviews, internal audits, CAPA system, supplier management (including SOUP assessment procedures).
3. **Regulatory consultant**: Engage a regulatory affairs consultant experienced in SaMD and EU MDR. Class III SaMD is specialist territory.
4. **Notified Body selection**: Identify and engage a Notified Body with SaMD competence (not all Notified Bodies accept Class III SaMD).
5. **Initial risk management file**: Build the ISO 14971 risk management file, including SOUP failure mode analysis using MADDENING's hazard hints and anomaly registry.
6. **SOUP assessment**: Perform formal SOUP assessment of MADDENING (and all dependencies) per IEC 62304 Clause 5.3.3/5.3.4, using MADDENING's SOUP package document.
7. **Clinical investigation plan**: Design the clinical studies needed for the CER. This requires ethics approval and may involve multi-site studies.

MADDENING's documentation architecture provides the SOUP-specific inputs to steps 2, 5, and 6. The quality of that documentation directly affects the cost and speed of the commercial entity's regulatory pathway.

---

## Appendix F: MIME / Downstream Library Bootstrap Checklist

This checklist is the downstream-library equivalent of Appendix E. It covers what a physics engine built on MADDENING — primarily MIME, but applicable to any future library — must do to correctly inherit and extend the MADDENING compliance architecture described in Section 16.

The checklist is organised around the schema/instance split: verify that the inherited infrastructure works in the downstream environment, then create the project-specific instances that the schema requires.

### Inherited Infrastructure (verify it works)

- [ ] `from maddening.compliance import NodeMeta, EdgeMeta, ValidatedRegime, Reference` succeeds in the downstream project's Python environment (Section 16)
- [ ] `from maddening.compliance import AnomalyRecord, AnomalySeverity, SafetyRelevance, ResolutionStatus` succeeds (Section 16)
- [ ] `from maddening.compliance import verification_benchmark, stability` succeeds (Section 16)
- [ ] `from maddening.compliance import validate_anomaly_registry` succeeds (Section 16)
- [ ] `python -m maddening.compliance check-anomalies --help` runs successfully, confirming the CLI is available (Section 16)
- [ ] The downstream project's `pyproject.toml` lists `maddening` as a dependency with a pinned version and lower bound (e.g., `maddening>=0.3.2,<0.4`) (Section 16)

### Namespace and Convention Setup

- [ ] A downstream ID prefix is chosen (e.g., `MIME-NODE-`, `MIME-ANO-`, `MIME-VER-`) and documented in the project's `CONTRIBUTING.md` (Section 16)
- [ ] The prefix is distinct from `MADD-` and from any other known downstream project's prefix (Section 16)
- [ ] The prefix convention covers all three ID types: node algorithm IDs (`MIME-NODE-`), anomaly IDs (`MIME-ANO-`), and verification benchmark IDs (`MIME-VER-`) (Section 16)

### Anomaly Lifecycle (create fresh)

- [ ] `.github/ISSUE_TEMPLATE/anomaly.md` exists in the downstream repository, matching or extending MADDENING's template with the same mandatory fields: title, severity, safety relevance, safety rationale, affected components, affected versions, workaround (Section 9.7)
- [ ] `docs/validation/known_anomalies.yaml` exists in the downstream repository (Sections 9.7, 16)
- [ ] The file uses the same schema as MADDENING's registry: `schema_version`, a project version field (e.g., `mime_version`), `generated_date`, and `anomalies` list (Section 9.7)
- [ ] All anomaly IDs use the downstream prefix (e.g., `MIME-ANO-001`, not `MADD-ANO-001`) (Section 16)
- [ ] `python -m maddening.compliance check-anomalies docs/validation/known_anomalies.yaml --prefix MIME-ANO-` passes with zero errors (Section 16)
- [ ] CI is configured to run the anomaly validator on every push and fail on YAML schema errors (Sections 9.7, 16)
- [ ] Release gate implements MADDENING's three-tier model (Section 9.7): Tier 1 hard gate for `safety-relevant`, Tier 2 no-grace-period gate for `anomaly:critical`/`anomaly:major`, Tier 3 two-cycle grace period for `anomaly:minor`
- [ ] Tier 3 cycle counting is **automated** by the release gate script (comparing issue close dates to release timestamps) — manual cycle tracking is not acceptable (Section 9.7)

### Node Metadata (create fresh)

- [ ] Every downstream `SimulationNode` subclass has `meta` attached as a ClassVar (Section 9.1)
- [ ] Each node's `meta.algorithm_id` uses the downstream prefix (e.g., `MIME-NODE-001`) — never a `MADD-NODE-*` ID (Section 16)
- [ ] Each node's `meta` contains at minimum: `algorithm_id`, `stability`, `description`, `assumptions`, `limitations`, `hazard_hints` (Sections 9.1, 9.8, 9.9)

### Verification Benchmarks (create fresh)

- [ ] At least one downstream node has a `@verification_benchmark` (imported from `maddening.compliance`) registered and passing in CI (Sections 9.3, 16)
- [ ] Verification benchmark IDs use the downstream prefix (e.g., `MIME-VER-001`) (Section 16)
- [ ] Verification benchmarks test the downstream node in its own physical regime — not simply re-running MADDENING's benchmarks against MADDENING's reference nodes (Section 16)
- [ ] For each Mode 1 (Wrapping) node: a **regression test** exists that asserts floating-point equality between the downstream wrapper's output and the upstream MADDENING node's output for identical inputs. The test pins the MADDENING version. (Section 16)

### HealthCheck Configuration (configure from upstream base)

- [ ] At least one `HealthCheckNode` instance is configured in the downstream graph with domain-appropriate checks (e.g., CSF density bounds, velocity magnitude limits) (Section 9.8)
- [ ] The HealthCheck node receives monitored state via edges from the relevant physics nodes (Section 9.8)
- [ ] The downstream application layer acts on HealthCheck output (halt, warn, log) — the HealthCheck node itself does not halt the simulation (Section 9.8)
- [ ] HealthCheck configurations are documented in the downstream risk management documentation, connecting each check to a specific ISO 14971 risk control measure (Section 9.8)
- [ ] For Class C contexts: each `HealthCheckNode` subclass's `update()` function uses fundamentally different numerical primitives than the monitored physics node's `update()` function — this is a **mandatory design requirement** (Algorithmic Diversity Principle, Section 9.8, Section 11 Clause 5.3.5)

### Algorithm Guide Mode Declaration (create fresh)

- [ ] Each downstream algorithm guide document declares the node's verification mode: Mode 1 (Wrapping) or Mode 2 (Independent) (Section 16)
- [ ] Mode 1 declarations cite the specific MADDENING version pin and upstream verification IDs (Section 16)
- [ ] Mode 2 declarations note that no upstream verification evidence is cited and that full independent verification is provided (Section 16)

### Bibliography and Citation Validation (create fresh or extend upstream)

- [ ] The downstream project maintains its own `docs/bibliography.bib` containing domain-specific references (e.g., CSF dynamics papers for MIME) (Section 3)
- [ ] Downstream algorithm guide documents use Pandoc-style `[@Key]` citation syntax consistent with MADDENING's convention (Section 3)
- [ ] Downstream algorithm guide documents include YAML frontmatter with `bibliography:` pointing to the downstream `.bib` file (Section 3)
- [ ] If citing references that also appear in MADDENING's `bibliography.bib`, the downstream project either (a) copies the relevant entries into its own `.bib` file, or (b) merges both `.bib` files during the documentation build. The choice must be documented in the downstream `DOCUMENTATION_ARCHITECTURE.md` (Section 3)
- [ ] A citation validation script (adapted from MADDENING's `scripts/check_citations.py` or using `BIB_PATH` env var override) runs in the downstream CI and fails on dangling citations (Section 3)
- [ ] Each References section includes human-readable inline descriptions alongside `[@Key]` markers so documents are readable without Pandoc rendering (Section 3)

### SOUP Package Document (create fresh)

- [ ] `docs/validation/soup_package.md` exists in the downstream repository, following the same template as MADDENING's (Section 10) (Sections 10, 16)
- [ ] Section 1 (Software Identification) identifies the downstream library, not MADDENING (Section 10)
- [ ] Section 8 (Dependencies) lists MADDENING at a pinned version with: version number, licence, SOUP package URL (version-tagged), known anomalies URL (version-tagged), SHA-256 hash (PyPI sdist), SBOM URL (Section 16)
- [ ] Section 8 includes a "MADDENING Anomalies Relevant to [Downstream Project]" table mapping each relevant MADDENING anomaly to its downstream impact and mitigation (Section 16)
- [ ] Section 8 documents the MADDENING version update policy: review CHANGELOG, assess new anomalies, run test suite, update pin, update SOUP package, document in downstream CHANGELOG (Section 16)
- [ ] For each downstream node that wraps an unmodified MADDENING node (Mode 1 — Wrapping), the SOUP package contains a **Scope Statement** citing both the MADDENING version pin AND the specific upstream verification ID (e.g., "CSFFlowNode wraps LBMPipeNode from MADDENING v0.3.2; upstream verification: MADD-VER-003") (Section 16)
- [ ] The SOUP package document identifies `maddening.nodes.health_check.HealthCheckNode` as a MADDENING component in use, with the MADDENING version pin covering it — `HealthCheckNode` is part of the MADDENING SOUP dependency, not a separate SOUP item (Section 10, 9.8)

### Repository Root Files (create fresh)

- [ ] The downstream project's `CHANGELOG.md` exists and follows [Keep a Changelog](https://keepachangelog.com/) format with the additional sections: `Verification`, `Security`, and `Known Anomalies` — matching MADDENING's convention (Section 5)

### Regulatory Boundary (create fresh)

- [ ] The downstream project's `README.md` contains its own regulatory disclaimer (e.g., "MIME is research software. It is not a medical device...") — not just a pointer to MADDENING's disclaimer (Sections 8, 16)
- [ ] `docs/regulatory/intended_use.md` exists in the downstream repository and contains the downstream project's own platform positioning statement (Section 2)
- [ ] The intended use statement notes that the downstream project is not a medical device and references both its own scope and its dependency on MADDENING (Sections 2, 16)

### Documentation Architecture (create fresh)

- [ ] A `DOCUMENTATION_ARCHITECTURE.md` (or equivalent) exists in the downstream repository (Section 16)
- [ ] The document opens with an inheritance statement referencing MADDENING's documentation architecture by name (Section 16)
- [ ] The document identifies what the downstream project inherits by reference (schema, templates, conventions) and what it provides itself (nodes, evidence, anomalies, SOUP package) (Section 16)
- [ ] The document covers at minimum: downstream-specific regulatory boundary language, downstream-specific nodes and their verification requirements, and the SOUP-of-SOUP relationship with MADDENING (Section 16)

### Keeping in Sync with MADDENING

- [ ] A documented process exists (in `CONTRIBUTING.md` or the documentation architecture document) for reviewing MADDENING releases and deciding whether to update the downstream dependency pin (Section 16)
- [ ] The downstream SOUP package Section 8 is updated whenever the MADDENING version pin changes (Section 16)
- [ ] New MADDENING known anomalies are assessed for downstream impact whenever the MADDENING pin is updated, and the "Anomalies Relevant to [Downstream Project]" table is updated accordingly (Section 16)
