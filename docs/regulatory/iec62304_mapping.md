# IEC 62304 Software Lifecycle Documentation Mapping

## Scope

**MADDENING is not subject to {term}`IEC 62304`.** It is an open-source research tool, not a medical device, and it is not developed under a medical device QMS. This mapping is provided voluntarily to support downstream {term}`SOUP` assessment.

## Mapping

| IEC 62304 Phase | Clause | What the Standard Requires | What MADDENING Provides | Gaps |
|---|---|---|---|---|
| **Software development planning** | 5.1 | Development plan, standards, tools, configuration management plan | `DESIGN.md`, `ROADMAP.md`, `CONTRIBUTING.md`, Git + GitHub CI | No formal development plan in IEC 62304 format. Not required — MADDENING is not subject to IEC 62304. |
| **Software requirements analysis** | 5.2 | Documented software requirements | `DESIGN.md` Sections 1-5, module READMEs, test suite | Requirements documented but not in formal specification with IDs and traceability. |
| **Software architectural design** | 5.3 | Software architecture document, SOUP identification, segregation analysis | `DESIGN.md`, graph-based architecture, functional purity. See Clause 5.3.5 below. | Architecture well-documented. |
| **↳ Segregation analysis** (Class C) | 5.3.5 | Demonstrate failure isolation or detection | **Data-flow layer**: functional purity (pure functions, no shared mutable state, explicit data flow). **Execution layer ({term}`XLA` Shadow)**: XLA compiler bugs, GPU hardware faults cannot be detected by MADDENING's architecture. Manufacturer must implement execution-layer defences via HealthCheck infrastructure. **Algorithmic Diversity**: mandatory for Class C — HealthCheck monitors must use different numerical primitives than monitored physics nodes. | MADDENING provides data-flow segregation evidence; manufacturer provides execution-layer segregation evidence. |
| **Software detailed design** | 5.4 | Detailed design for each software unit | Per-node algorithm documentation in `docs/algorithm_guide/`, `NodeMeta` metadata, Implementation Mapping tables | Coverage grows as algorithm guides are populated. |
| **Software unit implementation** | 5.5 | Implement per detailed design, coding standards | Source code in `src/maddening/`, NumPy-style docstrings | Formal coding standard planned. |
| **Software unit verification** | 5.6 | Verify each unit against design and requirements | `tests/` with 500+ tests, registered verification benchmarks in `tests/verification/` | Good coverage. Registered benchmarks formalize verification. |
| **Software integration and integration testing** | 5.7 | Integration testing | Integration tests in `tests/core/test_integration.py`, graph-level tests | Present. Graph architecture naturally exercises integration. |
| **Software system testing** | 5.8 | System-level testing against requirements | End-to-end example scripts, surrogate training + validation pipeline tests. **Manufacturer recommendation**: include cross-backend numerical agreement tests (CPU vs GPU). | No formal system test plan. Cross-backend testing is manufacturer obligation. |
| **Software release** | 5.9 | Release documentation, known anomalies, verification summary | `CHANGELOG.md`, tagged releases, `known_anomalies.yaml`, SOUP package document | Well-covered once SOUP package is complete. |
