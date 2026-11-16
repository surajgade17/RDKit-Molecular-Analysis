# RDKit Molecular Analysis Portfolio Project

## Project overview
Analyse 15 example molecules with RDKit. Calculate molecular weight, calculated LogP, TPSA, hydrogen-bond donors/acceptors, rotatable bonds and heavy atom count. Apply a simple Lipinski Rule of Five check, create charts, and export CSV and PDF outputs.

## Setup
```bash
python -m venv .venv
# Windows:
.venv\\Scripts\\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
jupyter notebook
```

Open `RDKit_Molecular_Analysis.ipynb` and choose **Run All**.

Alternatively:
```bash
python rdkit_molecular_analysis.py
```

## Outputs
Generated in `output/`:
- `molecular_analysis_results.csv`
- `molecular_weight.png`
- `logp.png`
- `tpsa.png`
- `hbond_counts.png`
- `molecule_structures.png`
- `RDKit_Molecular_Analysis_Report.pdf`

## Methods
- Molecular weight: `rdkit.Chem.Descriptors.MolWt`
- LogP: `rdkit.Chem.Crippen.MolLogP`
- TPSA: `rdkit.Chem.rdMolDescriptors.CalcTPSA`
- HBD/HBA: RDKit Lipinski descriptor functions

## Important limitations
This is a demonstration project using calculated descriptors. Lipinski's Rule of Five is a heuristic, not proof of drug-likeness, safety, efficacy, oral bioavailability, or clinical suitability. Duplicate compounds in the example dataset are included as a data-quality teaching point; consider removing duplicates for a production analysis. Check chemical structures and outputs before client delivery.


