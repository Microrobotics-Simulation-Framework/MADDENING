# Context of Use (COU) Template

This template follows ASME V&V 40-2018 for defining the context of use for a specific application of MADDENING. MADDENING as a general-purpose framework does not itself have a COU — only a specific downstream application can.

## 1. Question of Interest

*What specific question is the simulation answering?*

## 2. Quantities of Interest (QOIs)

*What output quantities are used to answer the question?*

| QOI | Units | Required Accuracy | Justification |
|-----|-------|-------------------|---------------|
| | | | |

## 3. Model Influence

*How does the simulation output influence the decision?*

- [ ] Informational only (supports but does not determine decision)
- [ ] Decision-critical (simulation output directly determines action)

## 4. Decision Consequence

*What is the consequence of an incorrect simulation prediction?*

| Consequence | Severity | Likelihood |
|-------------|----------|------------|
| | | |

## 5. Risk Assessment

*Based on model influence and decision consequence, what is the risk level?*

- [ ] Low (informational, reversible consequences)
- [ ] Medium (decision-critical, reversible consequences)
- [ ] High (decision-critical, irreversible consequences)

## 6. Credibility Goals

*What level of V&V evidence is needed for each QOI?*

| QOI | Verification Level | Validation Level |
|-----|-------------------|-----------------|
| | | |

## 7. Evidence Plan

*How will the credibility goals be achieved?*

- Code verification: [reference MADDENING benchmarks]
- Calculation verification: [mesh convergence, timestep convergence]
- Validation: [comparison to experimental data]
- Applicability: [parameter regime analysis]
- UQ: [uncertainty propagation method]
