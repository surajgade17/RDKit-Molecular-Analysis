"""
RDKit Molecular Analysis — standalone script
Install: pip install -r requirements.txt
Run: python rdkit_molecular_analysis.py
"""
# STEP 1 — Install dependencies (run in a terminal, not as Python code)
# pip install -r requirements.txt

import os
import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, Lipinski
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import display

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Imports successful. Output folder:", OUTPUT_DIR)

molecule_data = [
    {"Name": "Ethanol", "SMILES": "CCO"},
    {"Name": "Aspirin", "SMILES": "CC(=O)Oc1ccccc1C(=O)O"},
    {"Name": "Benzene", "SMILES": "c1ccccc1"},
    {"Name": "Caffeine", "SMILES": "Cn1c(=O)c2c(ncn2C)n(C)c1=O"},
    {"Name": "Ibuprofen", "SMILES": "CC(C)Cc1ccc(cc1)C(C)C(=O)O"},
    {"Name": "Paracetamol", "SMILES": "CC(=O)Nc1ccc(O)cc1"},
    {"Name": "Nicotinamide", "SMILES": "NC(=O)c1cccnc1"},
    {"Name": "Lidocaine", "SMILES": "CCN(CC)CC(=O)Nc1c(C)cccc1C"},
    {"Name": "Warfarin", "SMILES": "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O"},
    {"Name": "Acetaminophen", "SMILES": "CC(=O)Nc1ccc(O)cc1"},
    {"Name": "Urea", "SMILES": "NC(=O)N"},
    {"Name": "Benzoic acid", "SMILES": "O=C(O)c1ccccc1"},
    {"Name": "Anisole", "SMILES": "COc1ccccc1"},
    {"Name": "Phenacetin", "SMILES": "CCOc1ccc(NC(C)=O)cc1"},
    {"Name": "Metformin", "SMILES": "CN(C)C(=N)NC(=N)N"}
]

molecules_df = pd.DataFrame(molecule_data)
display(molecules_df)

def parse_smiles(smiles):
    """Return an RDKit molecule, or None if the SMILES cannot be parsed."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles)

molecules_df["Molecule"] = molecules_df["SMILES"].apply(parse_smiles)
molecules_df["SMILES_Valid"] = molecules_df["Molecule"].notna()

print("Valid molecules:", int(molecules_df["SMILES_Valid"].sum()))
print("Invalid molecules:", int((~molecules_df["SMILES_Valid"]).sum()))
display(molecules_df[["Name", "SMILES", "SMILES_Valid"]])

def calculate_properties(mol):
    """Calculate selected 2D molecular descriptors for an RDKit molecule."""
    if mol is None:
        return None
    return {
        "Molecular_Weight": Descriptors.MolWt(mol),
        "LogP": Crippen.MolLogP(mol),
        "TPSA": rdMolDescriptors.CalcTPSA(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "Rotatable_Bonds": Lipinski.NumRotatableBonds(mol),
        "Heavy_Atom_Count": mol.GetNumHeavyAtoms()
    }

property_rows = molecules_df["Molecule"].apply(calculate_properties)
properties_df = pd.DataFrame(property_rows.tolist(), index=molecules_df.index)
results_df = pd.concat(
    [molecules_df[["Name", "SMILES", "SMILES_Valid"]], properties_df],
    axis=1
)
results_df = results_df[results_df["SMILES_Valid"]].copy()
results_df = results_df.reset_index(drop=True)

display(results_df.round(3))

def lipinski_violations(row):
    violations = 0
    violations += row["Molecular_Weight"] > 500
    violations += row["LogP"] > 5
    violations += row["HBD"] > 5
    violations += row["HBA"] > 10
    return int(violations)

results_df["Lipinski_Violations"] = results_df.apply(lipinski_violations, axis=1)
results_df["Lipinski_Result"] = results_df["Lipinski_Violations"].apply(
    lambda n: "Pass (0–1 violations)" if n <= 1 else "More than 1 violation"
)

display(results_df[[
    "Name", "Molecular_Weight", "LogP", "HBD", "HBA",
    "Lipinski_Violations", "Lipinski_Result"
]].round(3))

csv_path = os.path.join(OUTPUT_DIR, "molecular_analysis_results.csv")
results_df.drop(columns=[], errors="ignore").to_csv(csv_path, index=False)
print("Saved:", csv_path)

# Chart 1: Molecular weight
plt.figure(figsize=(11, 5))
plt.bar(results_df["Name"], results_df["Molecular_Weight"])
plt.title("Molecular Weight of Example Compounds")
plt.xlabel("Molecule")
plt.ylabel("Molecular weight (g/mol)")
plt.xticks(rotation=55, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "molecular_weight.png"), dpi=300)
plt.show()

# Chart 2: LogP
plt.figure(figsize=(11, 5))
plt.bar(results_df["Name"], results_df["LogP"])
plt.title("Calculated LogP of Example Compounds")
plt.xlabel("Molecule")
plt.ylabel("LogP")
plt.xticks(rotation=55, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "logp.png"), dpi=300)
plt.show()

# Chart 3: TPSA
plt.figure(figsize=(11, 5))
plt.bar(results_df["Name"], results_df["TPSA"])
plt.title("Topological Polar Surface Area")
plt.xlabel("Molecule")
plt.ylabel("TPSA (Å²)")
plt.xticks(rotation=55, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "tpsa.png"), dpi=300)
plt.show()

# Chart 4: H-bond donors and acceptors
x = range(len(results_df))
width = 0.38
plt.figure(figsize=(11, 5))
plt.bar([i - width/2 for i in x], results_df["HBD"], width, label="HBD")
plt.bar([i + width/2 for i in x], results_df["HBA"], width, label="HBA")
plt.title("Hydrogen-Bond Donors and Acceptors")
plt.xlabel("Molecule")
plt.ylabel("Count")
plt.xticks(list(x), results_df["Name"], rotation=55, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hbond_counts.png"), dpi=300)
plt.show()

from rdkit.Chem import Draw

valid_mols = [m for m in molecules_df["Molecule"].tolist() if m is not None]
valid_names = molecules_df.loc[molecules_df["Molecule"].notna(), "Name"].tolist()

molecule_grid = Draw.MolsToGridImage(
    valid_mols,
    molsPerRow=4,
    subImgSize=(250, 180),
    legends=valid_names
)
structure_path = os.path.join(OUTPUT_DIR, "molecule_structures.png")
molecule_grid.save(structure_path)
display(molecule_grid)
print("Saved:", structure_path)

from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import mm

pdf_path = os.path.join(OUTPUT_DIR, "RDKit_Molecular_Analysis_Report.pdf")
doc = SimpleDocTemplate(
    pdf_path, pagesize=A4,
    rightMargin=15*mm, leftMargin=15*mm,
    topMargin=15*mm, bottomMargin=15*mm
)
styles = getSampleStyleSheet()
story = [
    Paragraph("RDKit Molecular Analysis Report", styles["Title"]),
    Spacer(1, 6),
    Paragraph(
        "Educational demonstration using RDKit-calculated 2D molecular descriptors. "
        "Values are computational estimates, not experimental measurements.",
        styles["BodyText"]
    ),
    Spacer(1, 10),
    Paragraph(f"Number of valid molecules analysed: {len(results_df)}", styles["Heading2"]),
    Paragraph(
        f"Molecules with 0–1 Lipinski violations: "
        f"{int((results_df['Lipinski_Violations'] <= 1).sum())}",
        styles["BodyText"]
    ),
    Spacer(1, 10)
]

table_data = [[
    "Molecule", "MW", "LogP", "TPSA", "HBD", "HBA", "Violations"
]]
for _, r in results_df.iterrows():
    table_data.append([
        r["Name"], f"{r['Molecular_Weight']:.1f}", f"{r['LogP']:.2f}",
        f"{r['TPSA']:.1f}", str(r["HBD"]), str(r["HBA"]),
        str(r["Lipinski_Violations"])
    ])

tbl = Table(table_data, repeatRows=1, colWidths=[42*mm, 19*mm, 17*mm, 19*mm, 14*mm, 14*mm, 23*mm])
tbl.setStyle(TableStyle([
    ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1F4E79")),
    ("TEXTCOLOR", (0,0), (-1,0), colors.white),
    ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
    ("FONTSIZE", (0,0), (-1,-1), 7),
    ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#EAF1F8")]),
    ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
]))
story.append(tbl)
story.append(PageBreak())

for title, filename in [
    ("Molecular Weight", "molecular_weight.png"),
    ("Calculated LogP", "logp.png"),
    ("Topological Polar Surface Area", "tpsa.png"),
    ("Hydrogen-Bond Counts", "hbond_counts.png"),
    ("2D Molecular Structures", "molecule_structures.png")
]:
    story.append(Paragraph(title, styles["Heading2"]))
    img_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(img_path):
        story.append(Image(img_path, width=175*mm, height=80*mm, kind="proportional"))
        story.append(Spacer(1, 5))

story.append(Paragraph("Interpretation and limitations", styles["Heading2"]))
story.append(Paragraph(
    "Lipinski's Rule of Five is a heuristic often used in early oral-drug-likeness assessment. "
    "A pass does not establish safety, efficacy, absorption, or clinical suitability. "
    "The descriptor calculations depend on the supplied molecular structures and RDKit algorithms. "
    "Results should be checked by a qualified researcher before scientific or commercial use.",
    styles["BodyText"]
))
doc.build(story)
print("PDF report created:", pdf_path)

print("PROJECT SUMMARY")
print("=" * 50)
print("Valid molecules analysed:", len(results_df))
print("Average molecular weight:", round(results_df["Molecular_Weight"].mean(), 2), "g/mol")
print("Average LogP:", round(results_df["LogP"].mean(), 2))
print("Average TPSA:", round(results_df["TPSA"].mean(), 2), "Å²")
print("Lipinski pass (0–1 violations):",
      int((results_df["Lipinski_Violations"] <= 1).sum()))
print("Output files:", sorted(os.listdir(OUTPUT_DIR)))