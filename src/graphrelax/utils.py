"""Utility functions for scoring and I/O."""

import logging
import math
from pathlib import Path
from typing import Optional

from graphrelax.structure_io import StructureFormat

logger = logging.getLogger(__name__)


def remove_waters(structure_string: str, fmt: StructureFormat = None) -> str:
    """
    Remove water molecules from a structure string.

    Supports both PDB and CIF formats. If format is not specified,
    assumes PDB format for backwards compatibility.

    Args:
        structure_string: Structure file contents as a string
        fmt: Structure format (PDB or CIF). Defaults to PDB.

    Returns:
        Structure string with water molecules removed
    """
    if fmt == StructureFormat.CIF:
        return _remove_waters_cif(structure_string)
    return _remove_waters_pdb(structure_string)


def _remove_waters_pdb(pdb_string: str) -> str:
    """
    Remove water molecules (HOH, WAT, SOL) from a PDB string.

    Args:
        pdb_string: PDB file contents as a string

    Returns:
        PDB string with water molecules removed
    """
    water_residues = {"HOH", "WAT", "SOL", "TIP3", "TIP4", "SPC"}
    filtered_lines = []

    for line in pdb_string.splitlines():
        # Check ATOM/HETATM records
        if line.startswith(("ATOM", "HETATM")):
            # Residue name is in columns 17-20 (0-indexed: 17:20)
            if len(line) >= 20:
                resname = line[17:20].strip()
                if resname in water_residues:
                    continue
        # Check TER records that might reference water
        elif line.startswith("TER"):
            if len(line) >= 20:
                resname = line[17:20].strip()
                if resname in water_residues:
                    continue

        filtered_lines.append(line)

    return "\n".join(filtered_lines)


def _remove_waters_cif(cif_string: str) -> str:
    """
    Remove water molecules from a CIF string.

    Args:
        cif_string: CIF file contents as a string

    Returns:
        CIF string with water molecules removed
    """
    import io
    import tempfile

    from Bio.PDB import MMCIFIO, MMCIFParser, Select

    water_residues = {"HOH", "WAT", "SOL", "TIP3", "TIP4", "SPC"}

    class WaterRemover(Select):
        def accept_residue(self, residue):
            return residue.get_resname().strip() not in water_residues

    # MMCIFParser requires a file path
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".cif", delete=False
    ) as tmp:
        tmp.write(cif_string)
        tmp_path = tmp.name

    try:
        from pathlib import Path

        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("structure", tmp_path)

        cif_io = MMCIFIO()
        cif_io.set_structure(structure)

        output = io.StringIO()
        cif_io.save(output, WaterRemover())
        return output.getvalue()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def compute_sequence_recovery(seq1: str, seq2: str) -> float:
    """
    Compute fraction of identical residues between two sequences.

    Args:
        seq1: First sequence
        seq2: Second sequence

    Returns:
        Fraction of identical positions (0-1)
    """
    if len(seq1) != len(seq2):
        min_len = min(len(seq1), len(seq2))
        seq1 = seq1[:min_len]
        seq2 = seq2[:min_len]

    if len(seq1) == 0:
        return 0.0

    return sum(a == b for a, b in zip(seq1, seq2)) / len(seq1)


def write_scorefile(path: Path, scores: list, header: Optional[list] = None):
    """
    Write a Rosetta-style scorefile.

    Args:
        path: Output file path
        scores: List of dictionaries with score values
        header: Optional list of column names (inferred if not provided)
    """
    if not scores:
        return

    # Infer columns from first score dict if header not provided
    if header is None:
        header = list(scores[0].keys())

    # Build format string for alignment
    col_widths = {}
    for col in header:
        max_width = len(col)
        for score_dict in scores:
            val = score_dict.get(col, "")
            if isinstance(val, float):
                val_str = f"{val:.4f}"
            else:
                val_str = str(val)
            max_width = max(max_width, len(val_str))
        col_widths[col] = max_width + 2

    with open(path, "w") as f:
        # Write header
        header_line = "SCORE: "
        for col in header:
            header_line += f"{col:>{col_widths[col]}}"
        f.write(header_line + "\n")

        # Write data rows
        for score_dict in scores:
            row = "SCORE: "
            for col in header:
                val = score_dict.get(col, "")
                if isinstance(val, float):
                    val_str = f"{val:.4f}"
                else:
                    val_str = str(val)
                row += f"{val_str:>{col_widths[col]}}"
            f.write(row + "\n")

    logger.info(f"Wrote scorefile to {path}")


def compute_ligandmpnn_score(loss: float) -> float:
    """
    Convert LigandMPNN loss to a confidence score.

    Args:
        loss: Average negative log probability from LigandMPNN

    Returns:
        Confidence score (exp(-loss))
    """
    return math.exp(-loss)


def format_output_path(base_path: Path, index: int, n_outputs: int) -> Path:
    """
    Format output path with index suffix if generating multiple outputs.

    Args:
        base_path: Base output path (e.g., output.pdb)
        index: Output index (1-indexed)
        n_outputs: Total number of outputs

    Returns:
        Formatted path (e.g., output_1.pdb if n_outputs > 1)
    """
    if n_outputs == 1:
        return base_path

    stem = base_path.stem
    suffix = base_path.suffix
    return base_path.parent / f"{stem}_{index}{suffix}"


def save_pdb_string(pdb_string: str, path: Path):
    """Save PDB string to file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(pdb_string)
    logger.info(f"Saved structure to {path}")


def format_sequence_alignment(native: str, designed: str) -> str:
    """
    Format a sequence alignment showing changes between native and designed.

    Args:
        native: Original sequence
        designed: Designed sequence

    Returns:
        Multi-line string showing alignment with:
        - Native sequence on top
        - Change indicator (. for same, letter for changed)
        - Designed sequence on bottom
    """
    if len(native) != len(designed):
        min_len = min(len(native), len(designed))
        native = native[:min_len]
        designed = designed[:min_len]

    # Build change line: dot if same, new AA if changed
    changes = ""
    for n, d in zip(native, designed):
        if n == d:
            changes += "."
        else:
            changes += d

    # Count mutations
    n_mutations = sum(1 for n, d in zip(native, designed) if n != d)

    # Format with position markers every 10 residues
    lines = []
    chunk_size = 50

    for i in range(0, len(native), chunk_size):
        end = min(i + chunk_size, len(native))
        pos_start = i + 1
        pos_end = end

        # Position header
        pos_line = f"{pos_start:>4}"
        pos_line += " " * (chunk_size - len(str(pos_start)) - len(str(pos_end)))
        pos_line += f"{pos_end}"

        lines.append(f"         {pos_line}")
        lines.append(f"  Native {native[i:end]}")
        lines.append(f"         {changes[i:end]}")
        lines.append(f"Designed {designed[i:end]}")
        lines.append("")

    # Summary
    recovery = compute_sequence_recovery(native, designed)
    lines.append(
        f"  Mutations: {n_mutations}/{len(native)} "
        f"({100*(1-recovery):.1f}% changed, {100*recovery:.1f}% recovered)"
    )

    return "\n".join(lines)
