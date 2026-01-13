"""Utilities for ligand parameterization and handling."""

import io
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from openmm import app as openmm_app

# Ligand parameterization dependencies (conda-forge only)
# conda install -c conda-forge openff-toolkit=0.14.0 rdkit=2023.09.1
try:
    from openff.toolkit import Molecule
    from rdkit import Chem
    from rdkit.Chem import AllChem

    LIGAND_DEPS_AVAILABLE = True
except ImportError:
    LIGAND_DEPS_AVAILABLE = False
    Molecule = None
    Chem = None
    AllChem = None

logger = logging.getLogger(__name__)


def _check_ligand_deps():
    """Raise ImportError if ligand dependencies are not available."""
    if not LIGAND_DEPS_AVAILABLE:
        raise ImportError(
            "Ligand parameterization requires openff-toolkit and rdkit.\n"
            "These packages are only available via conda-forge:\n\n"
            "  conda install -c conda-forge "
            "openff-toolkit=0.14.0 rdkit=2023.09.1\n\n"
            "See the README for full installation instructions."
        )


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


def get_ion_smiles() -> Dict[str, str]:
    """
    Return SMILES for common ions.

    Ions are single atoms that cannot be parsed from PDB coordinates,
    so we need explicit SMILES for them.
    """
    return {
        "ZN": "[Zn+2]",
        "MG": "[Mg+2]",
        "CA": "[Ca+2]",
        "FE": "[Fe+2]",
        "FE2": "[Fe+2]",
        "MN": "[Mn+2]",
        "CU": "[Cu+2]",
        "CO": "[Co+2]",
        "NA": "[Na+]",
        "K": "[K+]",
        "CL": "[Cl-]",
    }


def is_single_atom_ligand(ligand: LigandInfo) -> bool:
    """Check if ligand is a single atom (ion)."""
    return len(ligand.pdb_lines) == 1


def create_openff_molecule(ligand: LigandInfo, smiles: Optional[str] = None):
    """
    Create an OpenFF Toolkit Molecule from ligand info.

    Attempts to create the molecule in this order:
    1. From user-provided SMILES (if given)
    2. From ion lookup table (for single-atom ligands)
    3. From PDB coordinates via RDKit bond perception
    4. From direct OpenFF PDB parsing

    Requires openff-toolkit and rdkit (install via conda-forge).

    Args:
        ligand: LigandInfo with PDB coordinates
        smiles: Optional SMILES string (overrides automatic detection)

    Returns:
        openff.toolkit.Molecule

    Raises:
        ImportError: If openff-toolkit or rdkit are not installed.
    """
    _check_ligand_deps()

    # 1. User-provided SMILES takes precedence
    if smiles:
        try:
            mol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
            logger.debug(f"Created molecule for {ligand.resname} from SMILES")
            return mol
        except Exception as e:
            logger.warning(f"Failed to create from provided SMILES: {e}")

    # 2. Handle ions (single atoms) - need explicit SMILES
    if is_single_atom_ligand(ligand):
        ion_smiles = get_ion_smiles()
        if ligand.resname in ion_smiles:
            mol = Molecule.from_smiles(
                ion_smiles[ligand.resname], allow_undefined_stereo=True
            )
            logger.debug(f"Created ion molecule for {ligand.resname}")
            return mol
        else:
            raise ValueError(
                f"Unknown ion '{ligand.resname}'. Provide SMILES via "
                f"ligand_smiles={{'{ligand.resname}': '[Element+charge]'}}"
            )

    # 3. Try RDKit bond perception from PDB coordinates
    pdb_block = "\n".join(ligand.pdb_lines) + "\nEND\n"
    try:
        mol = _create_molecule_via_rdkit(pdb_block)
        logger.debug(f"Created molecule for {ligand.resname} via RDKit")
        return mol
    except Exception as e:
        logger.debug(f"RDKit parsing failed: {e}")

    # 4. Fallback: direct OpenFF PDB parsing
    try:
        mol = Molecule.from_pdb_file(
            io.StringIO(pdb_block),
            allow_undefined_stereo=True,
        )
        logger.debug(f"Created molecule for {ligand.resname} from PDB")
        return mol
    except Exception as e:
        raise ValueError(
            f"Could not create molecule for ligand {ligand.resname}: {e}"
        )


def _create_molecule_via_rdkit(pdb_block: str):
    """Create OpenFF Molecule via RDKit from PDB block."""
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
    pdb_block = "\n".join(ligand.pdb_lines) + "\nEND\n"
    pdb_file = openmm_app.PDBFile(io.StringIO(pdb_block))
    return pdb_file.topology, pdb_file.positions
