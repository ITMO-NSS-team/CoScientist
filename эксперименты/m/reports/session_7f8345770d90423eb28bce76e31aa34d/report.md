# Experiment Report on Drug-like Molecule Generation and Docking for KRAS G12C

## Overview
The computational experiment involved the generation, profiling, and molecular docking of novel drug-like candidates aimed at interacting with the KRAS G12C protein. Below is a detailed summary of activities and outcomes from the experiment.

## 1. Molecule Generation
- **Outcome**: Successfully generated **5 valid unique drug-like molecules**.
- **Artifacts**: The generated molecules can be downloaded in [SMILES format](http://10.32.1.114:9000/molecule-generative-mcp/generated/cancer/ea27bfdd69774e8f993d152d0572b9be.csv).

### Table: Generated Molecules Summary
| Molecules | QED | Synthetic Accessibility | PAINS | SureChEMBL | Glaxo | Brenk | BBB | IC50 |
|-----------|-----|-----------------------|-------|------------|-------|-------|-----|------|
| COc1ccc(Nc2cc(C(F)(F)F)nc3ccccc23)cc1C(=O)NC1CC1 | 0.64 | 2.22 | 0 | 0 | 0 | 0 | 1 | 1 |
| O=C(O)C(=O)Nc1cccc(C(=O)c2ccccc2)c1 | 0.66 | 1.67 | 0 | 0 | 0 | 1 | 0 | 0 |
| NC(=O)c1cccc(C(=O)O)c1F | 0.7 | 1.91 | 0 | 0 | 0 | 0 | 1 | 1 |
| O=C(O)c1cccc(NC(=O)c2ccccc2F)c1 | 0.89 | 1.48 | 0 | 0 | 0 | 0 | 0 | 1 |
| COc1ccc(CNc2ncnc3oncc23)cc1 | 0.77 | 2.41 | 0 | 0 | 0 | 0 | 1 | 0 |

## 2. Molecule Profiles
- **Outcome**: Profiles were created for the 5 molecules, detailing insights into their drug-like properties.
- **Key Metrics**:
  - QED scores: Ranging from **0.64 to 0.89**.
  - Toxicity profiles highlight varying potential for adverse effects.
  
- **Detailed Profiles**: Accessible in [molecule_profiles.json](file:///Users/nargiza/PycharmProjects/CoScientist_clone/эксперименты/tests/m/artifacts/EXP-2/ATT-1e2740676f704acb911e5bc8ab19324c/molecule_profiles.json).

## 3. Molecular Docking
- **Outcome**: Docking was completed for **4 out of 5 molecules**.
- **Best Molecule**: 
  - **Structure**: O=C(O)C(=O)Nc1cccc(C(=O)c2ccccc2)c1
  - **Binding Affinity**: -9.6 kcal/mol, indicating strong binding potential with the KRAS G12C protein.
  
- **Visualization**: Access the results and detailed docking scores from the following links:
  - [Docking Results Visualization](http://10.32.1.114:9000/chemcoscientist-user-data/chemical_mcp/docking_results/docking_6OIM_5850e2ec-ec72-4f65-bc7c-d177eb1678c5.html)

### Docking Results Links
- [Docking for Molecule 1](tables/calculate_docking_docking_6OIM_6800a62a-d81a-45ca-8576-bc9c38e5bdcf.html)
- [Docking for Molecule 2](tables/calculate_docking_docking_6OIM_5097ec5a-5590-4b39-b656-88f0a594a1e7.html)
- [Docking for Molecule 3](tables/calculate_docking_docking_6OIM_757e90d0-56ef-4438-8a41-aff5353c4363.html)

### Error Note
- One molecule could not be docked due to an API error. Further follow-up is needed to resolve this issue.

## Next Steps
1. Further validation of molecular behaviors through experimental methods may be warranted based on the insights gained from toxicity and binding affinities.
2. Review the docking error encountered for the molecule that was not processed and troubleshoot accordingly.

---

This compilation provides a comprehensive view of the experiment's progress, findings, and future directions. Please let me know if additional details or clarifications are required!