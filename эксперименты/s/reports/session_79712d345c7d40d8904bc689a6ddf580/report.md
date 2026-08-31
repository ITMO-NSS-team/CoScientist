# Report on the Development of KRAS G12C Inhibitors

## Overview
This report summarizes the experimental efforts in developing small molecules targeting the KRAS G12C mutation with a focus on selectivity and lack of cross-reactivity with HRAS or NRAS. The experiment focused on three key tasks: generating candidate molecules, evaluating their selectivity, and planning synthesis and testing.

## Task Outcomes

### 1. Generate KRAS G12C Candidates (EXP-1)
- **Outcome:** Successfully generated 3 candidate molecules.
- **Link to Results:** [Generated Molecules](http://10.32.1.114:9000/molecule-generative-mcp/generated/cancer/b5f87b16e48f439fbdab16eb4b7cb9f3.csv?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=chemcoscientist-user%2F20260830%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20260830T235547Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=93cdd4812e912c029e85a700bb7a1f081c54ebb139276e8adbcd051ee35cd287)

### 2. Evaluate Selectivity via Docking (EXP-2)
- **Outcome:** Evaluated the selectivity of the generated molecules.
- **Link to Results:** [Docking Results](workspace/experiment_artifacts/EXP-2/ATT-b53072154a8343db8bdadee03b1c6cff/docking_results.csv)

### 3. Plan Synthesis and Testing (EXP-3)
- **Outcome:** Developed a detailed synthesis and testing protocol for the three candidates.
- **Link to Plan:** [Synthesis Testing Plan](workspace/experiment_artifacts/EXP-3/ATT-27d3ad54a72e42f0862d770a7332caa1/synthesis_testing_plan.md)

## Data Tables

### Candidate Molecules Characteristics
#### generate_case_mols — [download](tables/generate_case_mols_b5f87b16e48f439fbdab16eb4b7cb9f3.csv)
| Molecules | QED | Synthetic Accessibility | PAINS | SureChEMBL | Glaxo | Brenk | BBB | IC50 |
| --------- | --- | ---------------------- | ----- | --------- | ----- | ----- | --- | ---- |
| O=C(Nc1cccc(C(F)(F)F)c1)C1Cc2ccccc2-n2nnnc21 | 0.76 | 2.93 | 0 | 0 | 0 | 0 | 0 | 1 |
| O=C(Cc1ccccc1)N1CCN(S(=O)(=O)c2cccc(C(F)(F)F)c2)CC1 | 0.78 | 1.97 | 0 | 0 | 0 | 0 | 1 | 0 |
| O=C(O)c1ccc(-c2ccc(OCc3ccccc3)cc2)cc1 | 0.75 | 1.46 | 0 | 0 | 0 | 0 | 0 | 0 |

### Docking Results
#### docking_results — [download](tables/docking_results.csv)
| SMILES | Molecular_Weight | LogP | HBA | HBD | TPSA | Rotatable_Bonds | KRAS_pKd | HRAS_pKd | NRAS_pKd | KRAS_IC50_nM | HRAS_IC50_nM | NRAS_IC50_nM | Selectivity_Fold | High_Selectivity |
| ------ | ---------------- | ---- | --- | --- | ---- | ---------------- | -------- | -------- | -------- | ------------- | ------------- | ------------- | ---------------- | ---------------- |
| O=C(O)c1ccc(-c2ccc(OCc3ccccc3)cc2)cc1 | 304.34 | 4.63 | 2 | 1 | 46.53 | 5 | 6.454 | 4.132 | 4.147 | 351.4 | 73721.6 | 71306.8 | 202.9 | True |
| O=C(Nc1cccc(C(F)(F)F)c1)C1Cc2ccccc2-n2nnnc21 | 359.31 | 2.96 | 4 | 1 | 72.7 | 2 | 8.06 | 5.464 | 5.782 | 8.7 | 3433.4 | 1651.3 | 189.4 | True |
| O=C(Cc1ccccc1)N1CCN(S(=O)(=O)c2cccc(C(F)(F)F)c2)CC1 | 412.43 | 2.78 | 3 | 0 | 57.69 | 4 | 7.46 | 4.997 | 5.397 | 34.6 | 10075.2 | 4011.0 | 115.8 | True |

## Next Steps
- Initiate laboratory synthesis based on the developed plan.
- Proceed with biological testing to validate the computational predictions regarding selectivity and potency against KRAS G12C.

---

This experimental outcome lays the groundwork for future therapeutic advancements targeting the KRAS G12C mutation.