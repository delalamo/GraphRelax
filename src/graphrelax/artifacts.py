"""Constants and utilities for crystallography artifact detection and removal.

Common crystallography and cryo-EM artifacts (buffers, cryoprotectants,
detergents, lipids) are stripped by default during preprocessing.
Use --keep-all-ligands to preserve them.
"""

import logging
from collections import defaultdict
from typing import Dict, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# =============================================================================
# Artifact Categories
# =============================================================================

# Buffer components commonly found in crystallization conditions
BUFFER_ARTIFACTS = frozenset(
    {
        # Sulfate/Phosphate
        "SO4",
        "PO4",
        "PO3",
        "PI",
        "2HP",
        "2PO",
        "SPO",
        "SPH",
        # Organic acids
        "CIT",
        "FLC",
        "ACT",
        "ACE",
        "FMT",
        "FOR",
        "NO3",
        "AZI",
        # HEPES, MES, Tris, etc.
        "MES",
        "EPE",
        "TRS",
        "CAC",
        "BIS",
        "HEZ",
        "MOH",
        "BTB",
        "TAM",
        "MRD",
        "PGO",
        "144",
        "IPS",
        # Malonate, Tartrate, etc.
        "MLI",
        "TAR",
        "MAL",
        "SUC",
        "FUM",
        # Borate
        "BO3",
        "BO4",
        # Imidazole (from His-tag purification)
        "IMD",
        "1MZ",
    }
)

# Cryoprotectants used for flash-cooling crystals
CRYOPROTECTANT_ARTIFACTS = frozenset(
    {
        # Glycerol and glycols
        "GOL",
        "EDO",
        "EGL",
        "PGR",
        "PGQ",
        "GLC",
        # MPD (2-Methyl-2,4-pentanediol)
        "MPD",
        "1PG",
        "PDO",
        # PEG variants (polyethylene glycol fragments)
        "PEG",
        "PGE",
        "PG4",
        "1PE",
        "P6G",
        "P33",
        "PE4",
        "PG0",
        "2PE",
        "PEU",
        "PE8",
        "PE5",
        "XPE",
        "12P",
        "15P",
        "PG5",
        "PG6",
        "PEE",
        "PE3",
        "P4G",
        "P2E",
        # DMSO
        "DMS",
        # Isopropanol
        "IPA",
    }
)

# Detergents used for membrane protein crystallization
DETERGENT_ARTIFACTS = frozenset(
    {
        # Maltosides (DDM, DM, UDM, etc.)
        "LMT",
        "MLA",
        "BMA",
        "TRE",
        "DDM",
        "DXM",
        # Glucosides (OG, NG, etc.)
        "BOG",
        "BGC",
        "NDG",
        "HTG",
        "OGA",
        "NGS",
        # LDAO (Lauryldimethylamine-N-oxide)
        "LDA",
        "DAO",
        # CHAPS/CHAPSO
        "CPS",
        "CHT",
        "SDS",
        "CHP",
        # Triton, digitonin
        "TRT",
        "T3A",
        "D10",
        "D12",
        "DGT",
        # LMNG, cymals
        "MNG",
        "CYC",
        "LMG",
        # C12E8/9 polyoxyethylene
        "CE9",
        "C8E",
        "C10",
        "C12",
        # Octyl glucoside
        "OLC",
    }
)

# Lipids and fatty acids (from LCP crystallization or membrane proteins)
LIPID_ARTIFACTS = frozenset(
    {
        # Common fatty acids
        "PLM",
        "MYR",
        "OLA",
        "STE",
        "PAL",
        "LNL",
        "ARA",
        "DCA",
        "UND",
        "MYS",
        "MYA",
        "LNO",
        "EOA",
        "PEF",
        # Monoolein (lipidic cubic phase crystallization)
        "OLB",
        "9OL",
        "MPG",
        "OLI",
        # Phospholipids and fragments
        "PC",
        "PE",
        "PG",
        "PS",
        "PLC",
        "EPH",
        "CDL",
        # Cholesterol
        "CLR",
        "CHO",
    }
)

# Reducing agents from protein preparation
REDUCING_AGENT_ARTIFACTS = frozenset(
    {
        "BME",
        "DTT",
        "DTU",
        "TCE",
        "TRO",
        "GSH",
    }
)

# Halide ions (often crystallization additives, not biologically relevant)
HALIDE_ARTIFACTS = frozenset(
    {
        "CL",
        "BR",
        "IOD",
        "F",
    }
)

# Unknown/placeholder atoms
UNKNOWN_ARTIFACTS = frozenset(
    {
        "UNX",
        "UNL",
        "UNK",
        "DUM",
    }
)

# =============================================================================
# Combined Sets
# =============================================================================

# Master set of all artifacts to remove by default
CRYSTALLOGRAPHY_ARTIFACTS = (
    BUFFER_ARTIFACTS
    | CRYOPROTECTANT_ARTIFACTS
    | DETERGENT_ARTIFACTS
    | LIPID_ARTIFACTS
    | REDUCING_AGENT_ARTIFACTS
    | HALIDE_ARTIFACTS
    | UNKNOWN_ARTIFACTS
)

# Biologically relevant ions - NOT stripped by default
# These are often structurally/functionally important
BIOLOGICALLY_RELEVANT_IONS = frozenset(
    {
        "ZN",
        "MG",
        "CA",
        "FE",
        "FE2",
        "MN",
        "CU",
        "CO",
        "NI",
        "MO",
        "NA",
        "K",  # Often biologically relevant in channels/pumps
    }
)

# Water residues (handled separately by remove_waters)
WATER_RESIDUES = frozenset({"HOH", "WAT", "SOL", "TIP3", "TIP4", "SPC"})


# =============================================================================
# Removal Functions
# =============================================================================


def remove_artifacts(
    pdb_string: str,
    keep_residues: Optional[Set[str]] = None,
) -> Tuple[str, Dict[str, int]]:
    """
    Remove crystallography artifacts from PDB string.

    Artifacts are identified by their residue name (columns 17-20 in PDB
    format). HETATM records with residue names in CRYSTALLOGRAPHY_ARTIFACTS
    are removed unless they appear in keep_residues.

    Args:
        pdb_string: PDB file contents as string
        keep_residues: Set of residue names to preserve (whitelist)

    Returns:
        Tuple of:
            - filtered_pdb: PDB string with artifacts removed
            - removed_counts: Dict mapping residue name to atom count
    """
    if keep_residues is None:
        keep_residues = set()

    # Normalize whitelist to uppercase
    keep_residues = {r.upper() for r in keep_residues}

    # Residues to remove = artifacts minus whitelist
    to_remove = CRYSTALLOGRAPHY_ARTIFACTS - keep_residues

    kept_lines = []
    removed_counts = defaultdict(int)

    for line in pdb_string.split("\n"):
        if line.startswith("HETATM"):
            resname = line[17:20].strip().upper()
            if resname in to_remove:
                removed_counts[resname] += 1
                continue
        kept_lines.append(line)

    filtered_pdb = "\n".join(kept_lines)

    return filtered_pdb, dict(removed_counts)


def is_artifact(resname: str) -> bool:
    """
    Check if a residue name is a known crystallography artifact.

    Args:
        resname: Three-letter residue code

    Returns:
        True if the residue is in CRYSTALLOGRAPHY_ARTIFACTS
    """
    return resname.upper() in CRYSTALLOGRAPHY_ARTIFACTS


def is_biologically_relevant_ion(resname: str) -> bool:
    """
    Check if a residue name is a biologically relevant ion.

    Args:
        resname: Three-letter residue code

    Returns:
        True if the residue is in BIOLOGICALLY_RELEVANT_IONS
    """
    return resname.upper() in BIOLOGICALLY_RELEVANT_IONS
