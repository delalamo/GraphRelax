"""Utilities for ligand parameterization and handling."""

import io
import json
import logging
import urllib.error
import urllib.request
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

# Import WATER_RESIDUES from artifacts to avoid duplication
from graphrelax.artifacts import WATER_RESIDUES  # noqa: E402


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


@dataclass
class LigandInfo:
    """Information about a ligand extracted from PDB."""

    resname: str
    chain_id: str
    resnum: int
    pdb_lines: List[str]
    smiles: Optional[str] = None


def extract_ligands_from_pdb(
    pdb_string: str,
    exclude_artifacts: bool = True,
) -> Tuple[str, List[LigandInfo]]:
    """
    Separate protein ATOM records from ligand HETATM records.

    By default, crystallography artifacts (buffers, cryoprotectants, detergents,
    lipids) are excluded from the ligand list since they cannot be meaningfully
    parameterized for minimization.

    Args:
        pdb_string: Full PDB string with protein and ligands
        exclude_artifacts: If True, skip known crystallography artifacts

    Returns:
        Tuple of (protein_only_pdb, list_of_ligand_info)
    """
    # Import here to avoid circular imports
    if exclude_artifacts:
        from graphrelax.artifacts import CRYSTALLOGRAPHY_ARTIFACTS
    else:
        CRYSTALLOGRAPHY_ARTIFACTS = set()

    protein_lines = []
    ligand_lines_by_residue = {}

    for line in pdb_string.split("\n"):
        if line.startswith("HETATM"):
            resname = line[17:20].strip().upper()
            chain_id = line[21] if len(line) > 21 else " "
            try:
                resnum = int(line[22:26].strip())
            except ValueError:
                resnum = 0

            # Skip water
            if resname in WATER_RESIDUES:
                continue

            # Skip known artifacts (they won't be parameterized)
            if resname in CRYSTALLOGRAPHY_ARTIFACTS:
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


# Cofactors that contain metals or unusual chemistry that cannot be
# parameterized by standard force fields (GAFF, OpenFF, etc.)
# These ligands will be excluded from minimization and restored unchanged.
UNPARAMETERIZABLE_COFACTORS = {
    # Heme and porphyrins (contain Fe)
    "HEM",
    "HEC",  # Heme C
    "HEA",  # Heme A
    "HEB",  # Heme B
    "1HE",  # Heme variant
    "2HE",
    "DHE",  # Deuteroheme
    "HAS",  # Heme-AS
    "HDD",  # Hydroxyheme
    "HEO",  # Heme O
    "HNI",  # Heme N
    "SRM",  # Siroheme
    # Iron-sulfur clusters
    "SF4",  # 4Fe-4S cluster
    "FES",  # 2Fe-2S cluster
    "F3S",  # 3Fe-4S cluster
    # Other metallocofactors
    "CLA",  # Chlorophyll A
    "CLB",  # Chlorophyll B
    "BCL",  # Bacteriochlorophyll
    "BPH",  # Bacteriopheophytin
    "PHO",  # Pheophytin
    "CHL",  # Chlorophyll
    "B12",  # Vitamin B12 / Cobalamin
    "COB",  # Cobalamin
    "PQQ",  # Pyrroloquinoline quinone
    "MTE",  # Methanopterin
    "F43",  # Coenzyme F430 (Ni)
    "MO7",  # Molybdopterin
    "MGD",  # Molybdopterin guanine dinucleotide
    # Copper centers
    "CU1",  # Copper site
    "CUA",  # CuA center
    "CUB",  # CuB center
}


def is_unparameterizable_cofactor(resname: str) -> bool:
    """Check if a residue is a known unparameterizable cofactor."""
    return resname.upper() in UNPARAMETERIZABLE_COFACTORS


# In-memory cache for PDBe SMILES lookups
_PDBE_SMILES_CACHE = {}


def fetch_pdbe_smiles(resname: str) -> Optional[str]:
    """
    Fetch SMILES from PDBe Chemical Component Dictionary.

    Args:
        resname: Three-letter ligand code (e.g., "ATP", "3JD")

    Returns:
        SMILES string if found, None otherwise
    """
    resname_upper = resname.upper()

    # Check cache first
    if resname_upper in _PDBE_SMILES_CACHE:
        cached = _PDBE_SMILES_CACHE[resname_upper]
        if cached is not None:
            logger.debug(f"Using cached SMILES for {resname_upper}")
        return cached

    url = f"https://www.ebi.ac.uk/pdbe/api/pdb/compound/summary/{resname_upper}"

    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))

        if resname_upper in data and data[resname_upper]:
            compound_data = data[resname_upper][0]
            if "smiles" in compound_data and compound_data["smiles"]:
                smiles = compound_data["smiles"][0]["name"]
                _PDBE_SMILES_CACHE[resname_upper] = smiles
                logger.info(f"Fetched SMILES for {resname_upper} from PDBe")
                return smiles

    except urllib.error.HTTPError as e:
        if e.code == 404:
            logger.debug(f"Ligand {resname_upper} not found in PDBe CCD")
        else:
            logger.warning(
                f"HTTP error fetching SMILES for {resname_upper}: {e}"
            )
    except urllib.error.URLError as e:
        logger.warning(
            f"Network error fetching SMILES for {resname_upper}: {e}"
        )
    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse error for {resname_upper}: {e}")
    except Exception as e:
        logger.warning(
            f"Unexpected error fetching SMILES for {resname_upper}: {e}"
        )

    # Cache the failure too
    _PDBE_SMILES_CACHE[resname_upper] = None
    return None


def is_single_atom_ligand(ligand: LigandInfo) -> bool:
    """Check if ligand is a single atom (ion)."""
    return len(ligand.pdb_lines) == 1


def create_openff_molecule(
    ligand: LigandInfo,
    smiles: Optional[str] = None,
    fetch_pdbe: bool = True,
):
    """
    Create an OpenFF Toolkit Molecule from ligand info.

    Attempts to create the molecule in this order:
    1. From user-provided SMILES (if given)
    2. From ion lookup table (for single-atom ligands)
    3. From PDBe Chemical Component Dictionary (if fetch_pdbe=True)
    4. From PDB coordinates via RDKit bond perception (fallback)

    Note: PDBe lookup is preferred over RDKit because RDKit's bond perception
    from 3D coordinates often fails or produces incorrect molecules for complex
    organic ligands. PDBe has correct SMILES for all standard PDB ligands.

    Requires openff-toolkit and rdkit (install via conda-forge).

    Args:
        ligand: LigandInfo with PDB coordinates
        smiles: Optional SMILES string (overrides automatic detection)
        fetch_pdbe: If True, try PDBe CCD before RDKit (recommended)

    Returns:
        openff.toolkit.Molecule

    Raises:
        ImportError: If openff-toolkit or rdkit are not installed.
        ValueError: If molecule cannot be created by any method.
    """
    _check_ligand_deps()

    # 1. User-provided SMILES takes precedence
    if smiles:
        try:
            mol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
            mol.name = ligand.resname  # Required for openmmforcefields matching
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
            mol.name = ligand.resname  # Required for openmmforcefields matching
            logger.debug(f"Created ion molecule for {ligand.resname}")
            return mol
        else:
            raise ValueError(
                f"Unknown ion '{ligand.resname}'. Provide SMILES via "
                f"ligand_smiles={{'{ligand.resname}': '[Element+charge]'}}"
            )

    # 3. Try PDBe Chemical Component Dictionary first (most reliable)
    if fetch_pdbe:
        pdbe_smiles = fetch_pdbe_smiles(ligand.resname)
        if pdbe_smiles:
            try:
                mol = Molecule.from_smiles(
                    pdbe_smiles, allow_undefined_stereo=True
                )
                mol.name = (
                    ligand.resname
                )  # Required for openmmforcefields matching
                logger.info(
                    f"Created molecule for {ligand.resname} using PDBe SMILES"
                )
                return mol
            except Exception as e:
                logger.warning(f"Failed to create from PDBe SMILES: {e}")

    # 4. Fallback: Try RDKit bond perception from PDB coordinates
    # Note: This often produces incorrect molecules for complex organic ligands
    pdb_block = "\n".join(ligand.pdb_lines) + "\nEND\n"
    rdkit_error = None
    try:
        mol = _create_molecule_via_rdkit(pdb_block)
        mol.name = ligand.resname  # Required for openmmforcefields matching
        logger.debug(f"Created molecule for {ligand.resname} via RDKit")
        return mol
    except Exception as e:
        rdkit_error = e
        logger.debug(f"RDKit parsing failed: {e}")

    # 5. Last resort: Try direct OpenFF PDB parsing
    openff_error = None
    try:
        mol = Molecule.from_pdb_file(
            io.StringIO(pdb_block),
            allow_undefined_stereo=True,
        )
        mol.name = ligand.resname  # Required for openmmforcefields matching
        logger.debug(f"Created molecule for {ligand.resname} from PDB")
        return mol
    except Exception as e:
        openff_error = e
        logger.debug(f"OpenFF PDB parsing failed: {e}")

    # All methods failed
    raise ValueError(
        f"Could not create molecule for ligand {ligand.resname}.\n"
        f"RDKit error: {rdkit_error}\n"
        f"OpenFF error: {openff_error}\n\n"
        f"You can provide SMILES manually via --ligand-smiles "
        f"'{ligand.resname}:YOUR_SMILES_HERE'"
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


def ligand_pdb_to_topology(ligand: LigandInfo, strip_hydrogens: bool = False):
    """
    Convert ligand PDB lines to OpenMM topology and positions.

    Args:
        ligand: LigandInfo with PDB lines
        strip_hydrogens: If True, remove hydrogen atoms from the topology.
            Useful when hydrogens will be added later by addHydrogens().

    Returns:
        Tuple of (topology, positions)
    """
    if strip_hydrogens:
        # Filter out hydrogen atoms (element column is at position 77-78)
        heavy_atom_lines = []
        for line in ligand.pdb_lines:
            if len(line) >= 78:
                element = line[76:78].strip()
                if element != "H":
                    heavy_atom_lines.append(line)
            else:
                # Try to detect hydrogen from atom name (starts with H)
                atom_name = line[12:16].strip() if len(line) >= 16 else ""
                if not atom_name.startswith("H"):
                    heavy_atom_lines.append(line)
        pdb_block = "\n".join(heavy_atom_lines) + "\nEND\n"
    else:
        pdb_block = "\n".join(ligand.pdb_lines) + "\nEND\n"

    pdb_file = openmm_app.PDBFile(io.StringIO(pdb_block))
    return pdb_file.topology, pdb_file.positions
