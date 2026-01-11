"""Utility functions for scoring and I/O."""

import logging
import math
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


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
