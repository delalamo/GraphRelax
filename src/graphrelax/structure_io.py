"""Unified structure I/O utilities for PDB and CIF formats."""

import io
import logging
import tempfile
from enum import Enum
from pathlib import Path

from Bio.PDB import MMCIFIO, PDBIO, MMCIFParser, PDBParser

logger = logging.getLogger(__name__)


class StructureFormat(Enum):
    """Supported structure file formats."""

    PDB = "pdb"
    CIF = "cif"


def detect_format(path: Path) -> StructureFormat:
    """
    Auto-detect structure format from file extension.

    Args:
        path: Path to structure file

    Returns:
        StructureFormat enum value

    Raises:
        ValueError: If extension is not recognized
    """
    suffix = path.suffix.lower()
    if suffix == ".pdb":
        return StructureFormat.PDB
    elif suffix in (".cif", ".mmcif"):
        return StructureFormat.CIF
    else:
        raise ValueError(
            f"Unknown structure format for extension '{suffix}'. "
            "Supported: .pdb, .cif, .mmcif"
        )


def read_structure(path: Path) -> str:
    """
    Read structure file contents as string.

    Args:
        path: Path to structure file

    Returns:
        File contents as string
    """
    with open(path) as f:
        return f.read()


def write_structure(content: str, path: Path, fmt: StructureFormat = None):
    """
    Write structure string to file.

    Args:
        content: Structure content as string
        path: Output file path
        fmt: Target format (auto-detected from path if not provided)
    """
    if fmt is None:
        fmt = detect_format(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)
    logger.info(f"Saved structure to {path}")


def convert_pdb_to_cif(pdb_string: str) -> str:
    """
    Convert PDB format string to CIF format string.

    Args:
        pdb_string: Structure in PDB format

    Returns:
        Structure in CIF format
    """
    parser = PDBParser(QUIET=True)
    handle = io.StringIO(pdb_string)
    structure = parser.get_structure("structure", handle)

    cif_io = MMCIFIO()
    cif_io.set_structure(structure)

    output = io.StringIO()
    cif_io.save(output)
    return output.getvalue()


def convert_cif_to_pdb(cif_string: str) -> str:
    """
    Convert CIF format string to PDB format string.

    Args:
        cif_string: Structure in CIF format

    Returns:
        Structure in PDB format
    """
    # MMCIFParser requires a file path, so use temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".cif", delete=False
    ) as tmp:
        tmp.write(cif_string)
        tmp_path = tmp.name

    try:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("structure", tmp_path)

        pdb_io = PDBIO()
        pdb_io.set_structure(structure)

        output = io.StringIO()
        pdb_io.save(output)
        return output.getvalue()
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def ensure_pdb_format(content: str, source_path: Path) -> str:
    """
    Ensure structure content is in PDB format.

    If source is CIF, converts to PDB. Otherwise returns as-is.

    Args:
        content: Structure content string
        source_path: Original file path (for format detection)

    Returns:
        Structure in PDB format
    """
    fmt = detect_format(source_path)
    if fmt == StructureFormat.CIF:
        logger.debug("Converting CIF to PDB for internal processing")
        return convert_cif_to_pdb(content)
    return content


def convert_to_format(pdb_string: str, target_format: StructureFormat) -> str:
    """
    Convert PDB string to target format.

    Args:
        pdb_string: Structure in PDB format
        target_format: Desired output format

    Returns:
        Structure in target format
    """
    if target_format == StructureFormat.PDB:
        return pdb_string
    elif target_format == StructureFormat.CIF:
        return convert_pdb_to_cif(pdb_string)
    else:
        raise ValueError(f"Unknown target format: {target_format}")


def get_output_format(
    input_path: Path,
    output_path: Path,
) -> StructureFormat:
    """
    Determine output format based on output path extension.

    Falls back to input format if output path has no recognized extension.

    Args:
        input_path: Input file path
        output_path: Output file path

    Returns:
        StructureFormat for output
    """
    # Try to detect from output path extension
    try:
        return detect_format(output_path)
    except ValueError:
        pass

    # Fall back to input format
    return detect_format(input_path)
