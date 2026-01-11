"""Rosetta-style resfile parser for residue-specific design control."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Set, TextIO, Union

# Amino acid sets for property-based selection
POLAR_AAS = set("DEHKNQRST")
APOLAR_AAS = set("ACFGILMPVWY")
ALL_AAS = set("ACDEFGHIKLMNPQRSTVWY")


class ResidueMode(Enum):
    """Mode for handling a residue during design."""

    NATRO = "NATRO"  # Fixed completely (no design, no repacking)
    NATAA = "NATAA"  # Repack only (same amino acid, optimize rotamers)
    ALLAA = "ALLAA"  # Design with all 20 amino acids
    PIKAA = "PIKAA"  # Design with only specified amino acids
    NOTAA = "NOTAA"  # Design excluding specified amino acids
    POLAR = "POLAR"  # Design with polar residues only
    APOLAR = "APOLAR"  # Design with nonpolar residues only


@dataclass
class ResidueSpec:
    """Specification for a single residue."""

    chain: str
    resnum: int
    icode: str = ""
    mode: ResidueMode = ResidueMode.NATAA
    allowed_aas: Optional[Set[str]] = None  # For PIKAA/NOTAA modes

    def get_allowed_aas(self) -> Set[str]:
        """Return allowed amino acids based on mode."""
        if self.mode == ResidueMode.ALLAA:
            return ALL_AAS.copy()
        elif self.mode == ResidueMode.PIKAA:
            return (
                self.allowed_aas.copy() if self.allowed_aas else ALL_AAS.copy()
            )
        elif self.mode == ResidueMode.NOTAA:
            return ALL_AAS - (self.allowed_aas or set())
        elif self.mode == ResidueMode.POLAR:
            return POLAR_AAS.copy()
        elif self.mode == ResidueMode.APOLAR:
            return APOLAR_AAS.copy()
        else:  # NATRO, NATAA
            return set()  # Not designed

    def is_designable(self) -> bool:
        """Return True if this residue should be designed (sequence changed)."""
        return self.mode in (
            ResidueMode.ALLAA,
            ResidueMode.PIKAA,
            ResidueMode.NOTAA,
            ResidueMode.POLAR,
            ResidueMode.APOLAR,
        )

    def is_repackable(self) -> bool:
        """Return True if this residue should be repacked."""
        return self.mode != ResidueMode.NATRO

    @property
    def key(self) -> str:
        """LigandMPNN-style residue key (e.g., 'A12' or 'A12A' with icode)."""
        return f"{self.chain}{self.resnum}{self.icode}"


@dataclass
class DesignSpec:
    """Complete design specification for a structure."""

    residue_specs: dict  # str -> ResidueSpec
    default_mode: ResidueMode = ResidueMode.NATAA

    def get_spec(self, chain: str, resnum: int, icode: str = "") -> ResidueSpec:
        """Get spec for a residue, returning default if not specified."""
        key = f"{chain}{resnum}{icode}"
        if key in self.residue_specs:
            return self.residue_specs[key]
        # Return a spec with default mode
        return ResidueSpec(
            chain=chain, resnum=resnum, icode=icode, mode=self.default_mode
        )

    def get_designable_keys(self) -> list:
        """Return list of residue keys that should be designed."""
        return [k for k, v in self.residue_specs.items() if v.is_designable()]

    def get_fixed_keys(self) -> list:
        """Return list of residue keys that are completely fixed (NATRO)."""
        return [
            k
            for k, v in self.residue_specs.items()
            if v.mode == ResidueMode.NATRO
        ]


class ResfileParser:
    """
    Parse Rosetta-style resfiles.

    Format:
    - Lines before START: set default behavior for all residues
    - START keyword: marks beginning of per-residue section
    - Lines after START: per-residue commands
    - Comments: lines starting with #
    - Format: <resnum> <chain> <command> [args]

    Supported commands:
    - NATRO: Keep completely fixed (no design, no repacking)
    - NATAA: Repack only (default)
    - ALLAA: Design with all 20 amino acids
    - PIKAA <aas>: Design with only specified amino acids
    - NOTAA <aas>: Design excluding specified amino acids
    - POLAR: Design with polar residues only (DEHKNQRST)
    - APOLAR: Design with nonpolar residues only (ACFGILMPVWY)

    Example resfile:
        # Default to repack-only
        NATAA
        START
        10 A ALLAA          # Design position 10
        15 A PIKAA HYW      # Only allow H, Y, W
        20 A NOTAA CP       # Exclude C and P
        30 A POLAR          # Only polar residues
        40 A NATRO          # Keep completely fixed
    """

    COMMANDS = dict(ResidueMode.__members__.items())

    def parse(self, resfile: Union[str, Path, TextIO]) -> DesignSpec:
        """Parse a resfile and return DesignSpec."""
        if isinstance(resfile, (str, Path)):
            with open(resfile) as f:
                return self._parse_lines(f)
        return self._parse_lines(resfile)

    def _parse_lines(self, lines) -> DesignSpec:
        """Parse resfile lines."""
        default_mode = ResidueMode.NATAA
        residue_specs = {}
        in_body = False

        for line in lines:
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith("#"):
                continue

            # Remove inline comments
            if "#" in line:
                line = line[: line.index("#")].strip()

            tokens = line.split()
            if not tokens:
                continue

            # Check for START keyword
            if tokens[0].upper() == "START":
                in_body = True
                continue

            if not in_body:
                # Header section - set defaults
                cmd = tokens[0].upper()
                if cmd in self.COMMANDS:
                    default_mode = self.COMMANDS[cmd]
            else:
                # Body section - residue-specific commands
                # Format: <resnum> <chain> <command> [args]
                try:
                    resnum = int(tokens[0])
                    chain = tokens[1]
                    cmd = tokens[2].upper()

                    if cmd not in self.COMMANDS:
                        continue

                    mode = self.COMMANDS[cmd]
                    allowed_aas = None

                    # Parse amino acid list for PIKAA/NOTAA
                    if cmd in ("PIKAA", "NOTAA") and len(tokens) > 3:
                        allowed_aas = set(tokens[3].upper())

                    key = f"{chain}{resnum}"
                    residue_specs[key] = ResidueSpec(
                        chain=chain,
                        resnum=resnum,
                        mode=mode,
                        allowed_aas=allowed_aas,
                    )
                except (ValueError, IndexError):
                    # Skip malformed lines
                    continue

        return DesignSpec(
            residue_specs=residue_specs,
            default_mode=default_mode,
        )
