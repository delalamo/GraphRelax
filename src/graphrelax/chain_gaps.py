"""Detection and handling of chain gaps in protein structures.

This module provides functions to detect missing residues (gaps) in protein
chains and to split chains at those gaps prior to minimization. This prevents
OpenMM minimization from artificially closing gaps by creating unrealistic
peptide bonds.
"""

import io
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from Bio.PDB import PDBParser

logger = logging.getLogger(__name__)

# Maximum C-N peptide bond distance in Angstroms
# Typical C-N bond is ~1.33 A, anything > 2.0 A indicates a break
MAX_PEPTIDE_BOND_DISTANCE = 2.0

# Maximum residue number gap that's considered sequential
# Gap > 1 indicates missing residues
MAX_SEQUENTIAL_GAP = 1


@dataclass
class ChainGap:
    """Represents a gap in a protein chain."""

    chain_id: str
    residue_before: int  # Residue number before the gap
    residue_after: int  # Residue number after the gap
    distance: Optional[float] = None  # C-N distance if available
    icode_before: str = ""
    icode_after: str = ""

    def __str__(self) -> str:
        gap_size = self.residue_after - self.residue_before - 1
        dist_str = f", dist={self.distance:.2f}A" if self.distance else ""
        return (
            f"Chain {self.chain_id}: gap of {gap_size} residues "
            f"between {self.residue_before}{self.icode_before} and "
            f"{self.residue_after}{self.icode_after}{dist_str}"
        )


def detect_chain_gaps(
    pdb_string: str,
    check_distance: bool = True,
    max_bond_distance: float = MAX_PEPTIDE_BOND_DISTANCE,
) -> List[ChainGap]:
    """
    Detect gaps in protein chains.

    Gaps are detected by two methods:
    1. Residue numbering discontinuities (missing residue numbers)
    2. Large C-N distances between consecutive residues (if check_distance=True)

    Args:
        pdb_string: PDB file contents as string
        check_distance: Whether to also check C-N bond distances
        max_bond_distance: Maximum allowed C-N distance in Angstroms

    Returns:
        List of ChainGap objects describing each detected gap
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", io.StringIO(pdb_string))

    gaps = []

    for model in structure:
        for chain in model:
            residues = [r for r in chain.get_residues() if r.id[0] == " "]
            if len(residues) < 2:
                continue

            for i in range(len(residues) - 1):
                res1 = residues[i]
                res2 = residues[i + 1]

                resnum1 = res1.id[1]
                resnum2 = res2.id[1]
                icode1 = res1.id[2].strip()
                icode2 = res2.id[2].strip()

                # Check for numbering gap
                numbering_gap = resnum2 - resnum1 > MAX_SEQUENTIAL_GAP

                # Check C-N distance if requested
                distance = None
                distance_gap = False
                if check_distance:
                    try:
                        c_atom = res1["C"]
                        n_atom = res2["N"]
                        distance = c_atom - n_atom
                        distance_gap = distance > max_bond_distance
                    except KeyError:
                        # Missing backbone atoms - treat as potential gap
                        distance_gap = True

                if numbering_gap or distance_gap:
                    gap = ChainGap(
                        chain_id=chain.id,
                        residue_before=resnum1,
                        residue_after=resnum2,
                        distance=distance,
                        icode_before=icode1,
                        icode_after=icode2,
                    )
                    gaps.append(gap)
                    logger.debug(f"Detected gap: {gap}")

    if gaps:
        logger.info(f"Detected {len(gaps)} chain gap(s) in structure")
    else:
        logger.debug("No chain gaps detected")

    return gaps


def split_chains_at_gaps(
    pdb_string: str,
    gaps: Optional[List[ChainGap]] = None,
) -> Tuple[str, Dict[str, str]]:
    """
    Split chains at detected gaps by assigning new chain IDs.

    Each continuous segment gets a unique chain ID. This prevents OpenMM
    from creating peptide bonds across gaps during minimization.

    Args:
        pdb_string: PDB file contents as string
        gaps: Pre-detected gaps (if None, will detect automatically)

    Returns:
        Tuple of:
        - Modified PDB string with split chains
        - Mapping from new chain IDs to original chain IDs
    """
    if gaps is None:
        gaps = detect_chain_gaps(pdb_string)

    if not gaps:
        return pdb_string, {}

    # Build a set of (chain_id, residue_after) for gap locations
    # We'll start a new chain at each gap
    gap_starts = {(g.chain_id, g.residue_after, g.icode_after) for g in gaps}

    # Available chain IDs (A-Z, a-z, 0-9)
    all_chain_ids = (
        list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        + list("abcdefghijklmnopqrstuvwxyz")
        + list("0123456789")
    )

    # Parse to find existing chain IDs
    existing_chains = set()
    for line in pdb_string.split("\n"):
        if line.startswith(("ATOM", "HETATM")) and len(line) > 21:
            existing_chains.add(line[21])

    # Create pool of available new chain IDs
    available_ids = [c for c in all_chain_ids if c not in existing_chains]

    # Track chain ID assignments
    chain_mapping = {}  # new_chain_id -> original_chain_id
    current_chain_map = {}  # original_chain_id -> current_new_chain_id
    next_id_idx = 0

    # Track which gap starts we've already processed
    processed_gap_starts = set()

    # Process PDB line by line
    output_lines = []

    for line in pdb_string.split("\n"):
        if line.startswith(("ATOM", "HETATM")) and len(line) > 26:
            chain_id = line[21]
            resnum = int(line[22:26].strip())
            icode = line[26].strip() if len(line) > 26 else ""

            gap_key = (chain_id, resnum, icode)

            # Check if this residue starts a new segment (at a gap)
            # Only process each gap start once
            if gap_key in gap_starts and gap_key not in processed_gap_starts:
                processed_gap_starts.add(gap_key)
                # Assign a new chain ID for this segment
                if next_id_idx < len(available_ids):
                    new_chain_id = available_ids[next_id_idx]
                    next_id_idx += 1
                    current_chain_map[chain_id] = new_chain_id
                    chain_mapping[new_chain_id] = chain_id
                    logger.debug(
                        f"Assigned new chain {new_chain_id} for segment "
                        f"starting at {chain_id}:{resnum}{icode}"
                    )
                else:
                    logger.warning(
                        "Ran out of available chain IDs for gap splitting"
                    )

            # Initialize chain mapping if not seen before
            if chain_id not in current_chain_map:
                current_chain_map[chain_id] = chain_id
                # Original chains map to themselves
                if chain_id not in chain_mapping:
                    chain_mapping[chain_id] = chain_id

            # Replace chain ID in line
            new_chain_id = current_chain_map[chain_id]
            line = line[:21] + new_chain_id + line[22:]

        elif line.startswith("TER") and len(line) > 21:
            # Update TER record chain ID too
            chain_id = line[21]
            if chain_id in current_chain_map:
                new_chain_id = current_chain_map[chain_id]
                line = line[:21] + new_chain_id + line[22:]

        output_lines.append(line)

    result = "\n".join(output_lines)

    if chain_mapping:
        # Only log non-trivial mappings
        non_trivial = {k: v for k, v in chain_mapping.items() if k != v}
        if non_trivial:
            logger.info(
                f"Split chains at {len(gaps)} gap(s), "
                f"created {len(non_trivial)} new chain segment(s)"
            )

    return result, chain_mapping


def restore_chain_ids(
    pdb_string: str,
    chain_mapping: Dict[str, str],
) -> str:
    """
    Restore original chain IDs after minimization.

    Args:
        pdb_string: PDB string with split chains
        chain_mapping: Mapping from current chain IDs to original chain IDs

    Returns:
        PDB string with original chain IDs restored
    """
    if not chain_mapping:
        return pdb_string

    # Build reverse mapping (new -> original)
    # Note: chain_mapping is new_id -> original_id
    output_lines = []

    for line in pdb_string.split("\n"):
        if line.startswith(("ATOM", "HETATM")) and len(line) > 21:
            current_chain = line[21]
            if current_chain in chain_mapping:
                original_chain = chain_mapping[current_chain]
                line = line[:21] + original_chain + line[22:]

        elif line.startswith("TER") and len(line) > 21:
            current_chain = line[21]
            if current_chain in chain_mapping:
                original_chain = chain_mapping[current_chain]
                line = line[:21] + original_chain + line[22:]

        output_lines.append(line)

    logger.debug("Restored original chain IDs")
    return "\n".join(output_lines)


def add_ter_records_at_gaps(pdb_string: str, gaps: List[ChainGap]) -> str:
    """
    Add TER records at gap locations to signal chain breaks.

    This is an alternative to chain splitting that works with force fields
    that recognize TER as chain termination.

    Args:
        pdb_string: PDB file contents as string
        gaps: List of detected gaps

    Returns:
        PDB string with TER records inserted at gap locations
    """
    if not gaps:
        return pdb_string

    # Build set of (chain_id, resnum, icode) where TER should be inserted
    # TER goes after residue_before
    ter_locations = {
        (g.chain_id, g.residue_before, g.icode_before) for g in gaps
    }

    output_lines = []
    prev_chain = None
    prev_resnum = None
    prev_icode = None

    for line in pdb_string.split("\n"):
        if line.startswith(("ATOM", "HETATM")) and len(line) > 26:
            chain_id = line[21]
            resnum = int(line[22:26].strip())
            icode = line[26].strip() if len(line) > 26 else ""

            # Check if we need to insert TER before this line
            # (i.e., previous residue was at a gap location)
            if (
                prev_chain is not None
                and (prev_chain, prev_resnum, prev_icode) in ter_locations
                and (chain_id != prev_chain or resnum != prev_resnum)
            ):
                # Insert TER record
                ter_line = "TER   "
                output_lines.append(ter_line)
                logger.debug(
                    f"Inserted TER after {prev_chain}:{prev_resnum}{prev_icode}"
                )

            prev_chain = chain_id
            prev_resnum = resnum
            prev_icode = icode

        output_lines.append(line)

    return "\n".join(output_lines)


def get_gap_summary(gaps: List[ChainGap]) -> str:
    """
    Generate a human-readable summary of detected gaps.

    Args:
        gaps: List of detected gaps

    Returns:
        Formatted string describing all gaps
    """
    if not gaps:
        return "No chain gaps detected"

    lines = [f"Detected {len(gaps)} chain gap(s):"]
    for gap in gaps:
        lines.append(f"  - {gap}")

    return "\n".join(lines)
