"""Utilities for ligand parameterization and handling."""

import io
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Water residue names to exclude from ligand processing
WATER_RESIDUES = {"HOH", "WAT", "SOL", "TIP3", "TIP4", "SPC"}


@dataclass
class LigandInfo:
    """Information about a ligand extracted from PDB."""

    resname: str
    chain_id: str
    resnum: int
    pdb_lines: List[str]
    smiles: Optional[str] = None


def extract_ligands_from_pdb(pdb_string: str) -> Tuple[str, List[LigandInfo]]:
    """
    Separate protein ATOM records from ligand HETATM records.

    Args:
        pdb_string: Full PDB string with protein and ligands

    Returns:
        Tuple of (protein_only_pdb, list_of_ligand_info)
    """
    protein_lines = []
    ligand_lines_by_residue = {}

    for line in pdb_string.split("\n"):
        if line.startswith("HETATM"):
            resname = line[17:20].strip()
            chain_id = line[21] if len(line) > 21 else " "
            try:
                resnum = int(line[22:26].strip())
            except ValueError:
                resnum = 0

            # Skip water
            if resname in WATER_RESIDUES:
                continue

            key = (chain_id, resnum, resname)
            if key not in ligand_lines_by_residue:
                ligand_lines_by_residue[key] = []
            ligand_lines_by_residue[key].append(line)
        elif line.startswith("END"):
            pass  # Skip, will add back
        else:
            protein_lines.append(line)

    protein_pdb = "\n".join(protein_lines) + "\nEND\n"

    ligands = []
    for (chain_id, resnum, resname), lines in ligand_lines_by_residue.items():
        ligands.append(
            LigandInfo(
                resname=resname,
                chain_id=chain_id,
                resnum=resnum,
                pdb_lines=lines,
            )
        )

    return protein_pdb, ligands


def get_common_ligand_smiles() -> Dict[str, str]:
    """
    Return SMILES for commonly encountered ligands.

    This helps with ligands where automatic bond perception may fail.
    """
    return {
        # Heme and porphyrins
        "HEM": (
            "[Fe+2]12([N-]3C(=C4C(=C([N-]1C(=C5C(C(=C([N-]2C(=C3C)C(=C)"
            "C)C=C5C)CCC(=O)O)C)C=C6[N-]4C(=C(C)C6C=C)C(C)=C)CCC(=O)O)C)C=C)C"
        ),
        "HEC": (
            "[Fe+2]12([N-]3C(=C4C(=C([N-]1C(=C5C(C(=C([N-]2C(=C3C)C(=C)"
            "C)C=C5C)CCC(=O)O)C)C=C6[N-]4C(=C(C)C6C=C)C(C)=C)CCC(=O)O)C)C=C)C"
        ),
        # Common cofactors
        "NAD": (
            "NC(=O)c1ccc[n+](c1)[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])"
            "OC[C@H]2O[C@H]([C@H](O)[C@@H]2O)n2cnc3c(N)ncnc23)[C@@H](O)[C@H]1O"
        ),
        "NAP": (  # NADP
            "NC(=O)c1ccc[n+](c1)[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])"
            "OC[C@H]2O[C@H]([C@H](OP(=O)([O-])[O-])[C@@H]2O)n2cnc3c(N)ncnc23)"
            "[C@@H](O)[C@H]1O"
        ),
        "FAD": (
            "Cc1cc2nc3c(=O)[nH]c(=O)nc-3n(C[C@H](O)[C@H](O)[C@H](O)"
            "COP(=O)([O-])OP(=O)([O-])OC[C@H]3O[C@H]([C@H](O)[C@@H]3O)"
            "n3cnc4c(N)ncnc43)c2cc1C"
        ),
        "FMN": (
            "Cc1cc2nc3c(=O)[nH]c(=O)nc-3n(C[C@H](O)[C@H](O)[C@H](O)"
            "COP(=O)([O-])[O-])c2cc1C"
        ),
        "ATP": (
            "Nc1ncnc2n(cnc12)[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])"
            "OP(=O)([O-])[O-])[C@@H](O)[C@H]1O"
        ),
        "ADP": (
            "Nc1ncnc2n(cnc12)[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])[O-])"
            "[C@@H](O)[C@H]1O"
        ),
        "AMP": (
            "Nc1ncnc2n(cnc12)[C@@H]1O[C@H](COP(=O)([O-])[O-])[C@@H](O)[C@H]1O"
        ),
        "GTP": (
            "Nc1nc2n(cnc2c(=O)[nH]1)[C@@H]1O[C@H](COP(=O)([O-])OP(=O)([O-])"
            "OP(=O)([O-])[O-])[C@@H](O)[C@H]1O"
        ),
        # Common ions (simple)
        "ZN": "[Zn+2]",
        "MG": "[Mg+2]",
        "CA": "[Ca+2]",
        "FE": "[Fe+2]",
        "FE2": "[Fe+2]",
        "MN": "[Mn+2]",
        "CU": "[Cu+2]",
        "CO": "[Co+2]",
        # Common small molecules
        "ACE": "CC(=O)",  # Acetyl
        "NME": "NC",  # N-methyl
        "ACT": "CC(=O)[O-]",  # Acetate
        "GOL": "OCC(O)CO",  # Glycerol
        "EDO": "OCCO",  # Ethylene glycol
        "PEG": "COCCOCCOCCO",  # PEG fragment
        "SO4": "[O-]S(=O)(=O)[O-]",  # Sulfate
        "PO4": "[O-]P(=O)([O-])[O-]",  # Phosphate
        "CL": "[Cl-]",  # Chloride
    }


def create_openff_molecule(ligand: LigandInfo, smiles: Optional[str] = None):
    """
    Create an OpenFF Toolkit Molecule from ligand info.

    Args:
        ligand: LigandInfo with PDB coordinates
        smiles: Optional SMILES string (if known)

    Returns:
        openff.toolkit.Molecule
    """
    from openff.toolkit import Molecule

    if smiles:
        # Create from SMILES
        try:
            mol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
            logger.debug(f"Created molecule for {ligand.resname} from SMILES")
            return mol
        except Exception as e:
            logger.warning(f"Failed to create from SMILES: {e}")

    # Try to create from PDB block
    pdb_block = "\n".join(ligand.pdb_lines) + "\nEND\n"

    try:
        # Try direct PDB parsing
        mol = Molecule.from_pdb_file(
            io.StringIO(pdb_block),
            allow_undefined_stereo=True,
        )
        logger.debug(f"Created molecule for {ligand.resname} from PDB")
        return mol
    except Exception as e:
        logger.debug(f"Direct PDB parsing failed: {e}")

    # Fallback: use RDKit
    try:
        mol = _create_molecule_via_rdkit(pdb_block)
        logger.debug(f"Created molecule for {ligand.resname} via RDKit")
        return mol
    except Exception as e:
        raise ValueError(
            f"Could not create molecule for ligand {ligand.resname}: {e}"
        )


def _create_molecule_via_rdkit(pdb_block: str):
    """Create OpenFF Molecule via RDKit from PDB block."""
    from openff.toolkit import Molecule
    from rdkit import Chem
    from rdkit.Chem import AllChem

    # Parse PDB with RDKit
    mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False, sanitize=False)

    if mol is None:
        raise ValueError("RDKit could not parse PDB block")

    # Try to sanitize
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        # Try without hydrogens and re-add them
        mol = Chem.MolFromPDBBlock(pdb_block, removeHs=True, sanitize=True)
        if mol is None:
            raise ValueError("RDKit sanitization failed")
        mol = Chem.AddHs(mol, addCoords=True)
        AllChem.EmbedMolecule(mol, randomSeed=42)

    # Convert to OpenFF Molecule
    return Molecule.from_rdkit(mol, allow_undefined_stereo=True)


def ligand_pdb_to_topology(ligand: LigandInfo):
    """
    Convert ligand PDB lines to OpenMM topology and positions.

    Args:
        ligand: LigandInfo with PDB lines

    Returns:
        Tuple of (topology, positions)
    """
    from openmm import app as openmm_app

    pdb_block = "\n".join(ligand.pdb_lines) + "\nEND\n"
    pdb_file = openmm_app.PDBFile(io.StringIO(pdb_block))
    return pdb_file.topology, pdb_file.positions
