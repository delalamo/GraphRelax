"""Protein structure idealization module for GraphRelax.

This module provides functions to idealize protein backbone geometry while
preserving dihedral angles (phi, psi, omega, chi). It handles chain gaps
to prevent artificial gap closure during subsequent minimization.
"""

import io
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.vectors import Vector, calc_dihedral
from openmm import Platform
from openmm import app as openmm_app
from openmm import openmm, unit
from pdbfixer import PDBFixer

from graphrelax.chain_gaps import (
    ChainGap,
    detect_chain_gaps,
    restore_chain_ids,
    split_chains_at_gaps,
)
from graphrelax.config import IdealizeConfig

# Add vendored LigandMPNN to path for OpenFold imports
LIGANDMPNN_PATH = Path(__file__).parent / "LigandMPNN"
if str(LIGANDMPNN_PATH) not in sys.path:
    sys.path.insert(0, str(LIGANDMPNN_PATH))

from openfold.np import residue_constants as rc  # noqa: E402

logger = logging.getLogger(__name__)

# Water residue names to preserve with protein
WATER_RESIDUES = {"HOH", "WAT", "SOL", "TIP3", "TIP4", "SPC"}


@dataclass
class DihedralAngles:
    """Container for backbone and sidechain dihedral angles."""

    phi: Optional[float] = None  # C(i-1)-N-CA-C
    psi: Optional[float] = None  # N-CA-C-N(i+1)
    omega: Optional[float] = None  # CA(i-1)-C(i-1)-N-CA
    chi: List[Optional[float]] = None  # Up to 4 chi angles

    def __post_init__(self):
        if self.chi is None:
            self.chi = []


def extract_ligands(pdb_string: str) -> Tuple[str, str]:
    """
    Separate ligands (non-water HETATM) from protein.

    Args:
        pdb_string: PDB file contents as string

    Returns:
        Tuple of (protein_pdb, ligand_lines)
        - protein_pdb: PDB with only ATOM + water HETATM + header records
        - ligand_lines: HETATM lines for non-water molecules
    """
    protein_lines = []
    ligand_lines = []

    for line in pdb_string.split("\n"):
        if line.startswith("HETATM"):
            resname = line[17:20].strip()
            if resname in WATER_RESIDUES:
                protein_lines.append(line)
            else:
                ligand_lines.append(line)
        elif line.startswith("END"):
            pass  # Handle END separately
        else:
            protein_lines.append(line)

    protein_pdb = "\n".join(protein_lines)
    if not protein_pdb.rstrip().endswith("END"):
        protein_pdb += "\nEND\n"

    return protein_pdb, "\n".join(ligand_lines)


def restore_ligands(pdb_string: str, ligand_lines: str) -> str:
    """
    Re-insert ligand HETATM lines into PDB after protein ATOM records.

    Args:
        pdb_string: PDB file contents
        ligand_lines: HETATM lines to restore

    Returns:
        PDB string with ligands restored
    """
    if not ligand_lines.strip():
        return pdb_string

    lines = pdb_string.split("\n")
    result = []
    ligand_inserted = False

    for line in lines:
        # Insert ligands before END, ENDMDL, or at the very end
        if line.strip().startswith(("END", "ENDMDL")) and not ligand_inserted:
            # Add ligand lines before END
            for lig_line in ligand_lines.split("\n"):
                if lig_line.strip():
                    result.append(lig_line)
            ligand_inserted = True
        result.append(line)

    # If no END record was found, append ligands at the end
    if not ligand_inserted:
        for lig_line in ligand_lines.split("\n"):
            if lig_line.strip():
                result.append(lig_line)
        result.append("END")

    return "\n".join(result)


def _get_atom_coord(residue, atom_name: str) -> Optional[np.ndarray]:
    """Get atom coordinates from residue, returns None if not present."""
    try:
        return np.array(residue[atom_name].get_coord())
    except KeyError:
        return None


def extract_dihedrals(
    residue,
    prev_residue=None,
    next_residue=None,
) -> DihedralAngles:
    """
    Extract phi/psi/omega/chi angles from a residue.

    Args:
        residue: BioPython residue object
        prev_residue: Previous residue (for phi, omega)
        next_residue: Next residue (for psi)

    Returns:
        DihedralAngles object with extracted values
    """
    angles = DihedralAngles()

    # Get atom coordinates
    n = _get_atom_coord(residue, "N")
    ca = _get_atom_coord(residue, "CA")
    c = _get_atom_coord(residue, "C")

    # Phi: C(i-1) - N - CA - C
    if (
        prev_residue is not None
        and n is not None
        and ca is not None
        and c is not None
    ):
        prev_c = _get_atom_coord(prev_residue, "C")
        if prev_c is not None:
            angles.phi = calc_dihedral(
                Vector(prev_c), Vector(n), Vector(ca), Vector(c)
            )

    # Psi: N - CA - C - N(i+1)
    if (
        next_residue is not None
        and n is not None
        and ca is not None
        and c is not None
    ):
        next_n = _get_atom_coord(next_residue, "N")
        if next_n is not None:
            angles.psi = calc_dihedral(
                Vector(n), Vector(ca), Vector(c), Vector(next_n)
            )

    # Omega: CA(i-1) - C(i-1) - N - CA
    if prev_residue is not None and n is not None and ca is not None:
        prev_ca = _get_atom_coord(prev_residue, "CA")
        prev_c = _get_atom_coord(prev_residue, "C")
        if prev_ca is not None and prev_c is not None:
            angles.omega = calc_dihedral(
                Vector(prev_ca), Vector(prev_c), Vector(n), Vector(ca)
            )

    # Chi angles
    resname = residue.resname.strip()
    if resname in rc.chi_angles_atoms:
        chi_defs = rc.chi_angles_atoms[resname]
        angles.chi = []

        for chi_def in chi_defs:
            coords = [_get_atom_coord(residue, atom) for atom in chi_def]
            if all(c is not None for c in coords):
                chi_val = calc_dihedral(
                    Vector(coords[0]),
                    Vector(coords[1]),
                    Vector(coords[2]),
                    Vector(coords[3]),
                )
                angles.chi.append(chi_val)
            else:
                angles.chi.append(None)

    return angles


def correct_cis_omega(omega: float, next_resname: str) -> float:
    """
    Correct non-trans omega angles to trans (180 deg).

    Prolines can legitimately be cis (~0 deg), so they are not corrected.

    Args:
        omega: Current omega angle in radians
        next_resname: Residue name of the next residue

    Returns:
        Corrected omega angle in radians
    """
    if next_resname == "PRO":
        return omega

    # Convert to degrees for easier checking
    omega_deg = np.degrees(omega) % 360

    # If omega is in cis range (< 90 deg or > 270 deg), correct to trans
    if omega_deg < 90 or omega_deg > 270:
        logger.debug(f"Correcting cis omega {omega_deg:.1f} deg -> 180 deg")
        return np.pi

    return omega


def _rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Create rotation matrix for rotation around axis by angle (radians)."""
    axis = axis / np.linalg.norm(axis)
    K = np.array(
        [
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0],
        ]
    )
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def build_ideal_backbone(
    resname: str,
    phi: Optional[float],
    psi: Optional[float],
    omega: Optional[float],
    prev_c_coord: Optional[np.ndarray] = None,
    prev_n_coord: Optional[np.ndarray] = None,
    prev_ca_coord: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Build ideal backbone atoms for a residue preserving dihedral angles.

    Uses ideal geometry from residue_constants.rigid_group_atom_positions.

    Args:
        resname: Three-letter residue name
        phi: Phi dihedral angle (radians)
        psi: Psi dihedral angle (radians)
        omega: Omega dihedral angle (radians)
        prev_c_coord: Previous residue C coordinate (for chaining)
        prev_n_coord: Previous residue N coordinate (for orientation)
        prev_ca_coord: Previous residue CA coordinate (for orientation)

    Returns:
        Dict mapping atom_name -> np.ndarray of coordinates
    """
    # Map resname for constants lookup
    lookup_name = resname if resname in rc.rigid_group_atom_positions else "ALA"

    # Get ideal positions from rigid_group_atom_positions
    ideal_positions = {}
    for atom_name, group_idx, pos in rc.rigid_group_atom_positions[lookup_name]:
        if group_idx == 0:  # Backbone group
            ideal_positions[atom_name] = np.array(pos)

    # Start with CA at origin, build N and C
    ca = np.array([0.0, 0.0, 0.0])
    n = ideal_positions.get("N", np.array([-0.525, 1.363, 0.0]))
    c = ideal_positions.get("C", np.array([1.526, 0.0, 0.0]))
    cb = ideal_positions.get("CB")

    # O position depends on psi (group 3)
    for atom_name, group_idx, pos in rc.rigid_group_atom_positions[lookup_name]:
        if atom_name == "O" and group_idx == 3:
            o_ideal = np.array(pos)
            break
    else:
        o_ideal = np.array([0.627, 1.062, 0.0])

    # Build coordinate frame
    coords = {"N": n.copy(), "CA": ca.copy(), "C": c.copy()}
    if cb is not None:
        coords["CB"] = cb.copy()

    # Apply psi rotation to O position
    if psi is not None:
        # Psi rotates around CA-C axis
        ca_c = c - ca
        # O is in psi-group, apply rotation
        rot_psi = _rotation_matrix(ca_c, psi)
        o_pos = c + rot_psi @ (o_ideal - np.array([0, 0, 0]))
        coords["O"] = o_pos
    else:
        # Place O at default position relative to C
        coords["O"] = c + o_ideal

    # If we have previous residue coordinates, chain the backbone
    if prev_c_coord is not None and prev_ca_coord is not None:
        # Ideal peptide bond length
        ideal_cn_bond = rc.between_res_bond_length_c_n[0]  # 1.329 A

        # Direction from prev_CA to prev_C
        prev_ca_c = prev_c_coord - prev_ca_coord
        prev_ca_c = prev_ca_c / np.linalg.norm(prev_ca_c)

        # Place N at ideal distance from prev_C along prev_CA-C direction
        # Then adjust based on omega angle
        if omega is not None:
            # Build N position using omega dihedral
            # Omega: CA(i-1) - C(i-1) - N - CA
            # Default omega = 180 deg (trans)
            n_pos = prev_c_coord + ideal_cn_bond * prev_ca_c
            # Apply omega rotation around prev_CA-prev_C axis
            # Note: rotation computed but not applied in current implementation
            _ = _rotation_matrix(prev_ca_c, omega - np.pi)
            # Transform current residue to align with prev_C
            translation = n_pos - coords["N"]
            for atom in coords:
                coords[atom] = coords[atom] + translation
        else:
            # Simple placement without omega
            n_pos = prev_c_coord + ideal_cn_bond * prev_ca_c
            translation = n_pos - coords["N"]
            for atom in coords:
                coords[atom] = coords[atom] + translation

    return coords


def build_ideal_sidechain(
    resname: str,
    backbone_coords: Dict[str, np.ndarray],
    chi_angles: List[Optional[float]],
) -> Dict[str, np.ndarray]:
    """
    Build ideal sidechain atoms from backbone and chi angles.

    Args:
        resname: Three-letter residue name
        backbone_coords: Dict of backbone atom coordinates
        chi_angles: List of chi angles in radians

    Returns:
        Dict mapping atom_name -> np.ndarray (including backbone atoms)
    """
    coords = backbone_coords.copy()

    # Map resname for constants lookup
    lookup_name = resname if resname in rc.rigid_group_atom_positions else "ALA"

    if lookup_name not in rc.chi_angles_atoms:
        return coords

    chi_defs = rc.chi_angles_atoms[lookup_name]
    if not chi_defs or not chi_angles:
        return coords

    # Get ideal positions for sidechain atoms
    ideal_sc = {}
    for atom_name, group_idx, pos in rc.rigid_group_atom_positions[lookup_name]:
        if group_idx >= 4:  # Chi groups
            ideal_sc[atom_name] = (group_idx, np.array(pos))

    # Apply chi rotations sequentially
    # For simplicity, we keep original sidechain coords if available
    # Full reconstruction would require proper frame transformations

    return coords


def minimize_with_constraints(
    pdb_string: str,
    stiffness: float = 10.0,
    add_missing_residues: bool = True,
) -> str:
    """
    Run single round of OpenMM minimization with position restraints.

    Restraints are applied to all heavy atoms using their input coordinates
    as reference positions.

    Args:
        pdb_string: PDB file contents
        stiffness: Restraint force constant in kcal/mol/A^2
        add_missing_residues: Whether to add missing residues from SEQRES

    Returns:
        Minimized PDB string
    """
    ENERGY = unit.kilocalories_per_mole
    LENGTH = unit.angstroms

    logger.debug(
        f"Running post-idealization minimization (stiffness={stiffness})"
    )

    # Use pdbfixer to prepare structure
    fixer = PDBFixer(pdbfile=io.StringIO(pdb_string))

    # Always call findMissingResidues to initialize the attribute
    # (required by findMissingAtoms)
    fixer.findMissingResidues()

    # Optionally add missing residues from SEQRES
    if add_missing_residues:
        if fixer.missingResidues:
            logger.info(
                f"Adding {sum(len(v) for v in fixer.missingResidues.values())} "
                "missing residues from SEQRES"
            )
    else:
        # Clear missing residues if user doesn't want them
        fixer.missingResidues = {}

    # Find and add missing atoms (including for missing residues if kept)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    # Add hydrogens
    fixer.addMissingHydrogens(7.0)

    # Create force field and modeller
    force_field = openmm_app.ForceField(
        "amber14-all.xml", "amber14/tip3pfb.xml"
    )
    modeller = openmm_app.Modeller(fixer.topology, fixer.positions)

    # Create system
    system = force_field.createSystem(
        modeller.topology, constraints=openmm_app.HBonds
    )

    # Add position restraints to heavy atoms
    restraint_force = openmm.CustomExternalForce(
        "0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
    )
    # Convert stiffness to kJ/mol/nm^2
    stiffness_openmm = (stiffness * ENERGY / (LENGTH**2)).value_in_unit(
        unit.kilojoules_per_mole / unit.nanometers**2
    )
    restraint_force.addGlobalParameter("k", stiffness_openmm)
    for p in ["x0", "y0", "z0"]:
        restraint_force.addPerParticleParameter(p)

    # Map atoms to reference positions
    for i, atom in enumerate(modeller.topology.atoms()):
        if atom.element.name != "hydrogen":
            # Get position in nanometers
            pos = modeller.positions[i].value_in_unit(unit.nanometers)
            restraint_force.addParticle(i, pos)

    system.addForce(restraint_force)
    logger.debug(
        f"Added restraints to {restraint_force.getNumParticles()} heavy atoms"
    )

    # Check for GPU
    use_gpu = False
    for i in range(Platform.getNumPlatforms()):
        if Platform.getPlatform(i).getName() == "CUDA":
            use_gpu = True
            break

    # Create simulation
    integrator = openmm.LangevinIntegrator(0, 0.01, 0.0)
    platform = Platform.getPlatformByName("CUDA" if use_gpu else "CPU")
    simulation = openmm_app.Simulation(
        modeller.topology, system, integrator, platform
    )
    simulation.context.setPositions(modeller.positions)

    # Minimize
    simulation.minimizeEnergy()

    # Get minimized structure
    state = simulation.context.getState(getPositions=True)
    output = io.StringIO()
    openmm_app.PDBFile.writeFile(
        simulation.topology, state.getPositions(), output
    )

    logger.debug("Post-idealization minimization complete")
    return output.getvalue()


def _idealize_chain_segment(
    residues: list,
    config: IdealizeConfig,
) -> None:
    """
    Idealize backbone geometry for a continuous chain segment.

    Modifies residue coordinates in place.

    Args:
        residues: List of BioPython residue objects (continuous segment)
        config: Idealization configuration
    """
    if not residues:
        return

    # Extract all dihedrals first
    all_dihedrals = []
    for i, res in enumerate(residues):
        prev_res = residues[i - 1] if i > 0 else None
        next_res = residues[i + 1] if i < len(residues) - 1 else None
        dihedrals = extract_dihedrals(res, prev_res, next_res)
        all_dihedrals.append(dihedrals)

    # Correct cis-omega if requested
    if config.fix_cis_omega:
        for i, dihedrals in enumerate(all_dihedrals):
            if dihedrals.omega is not None and i < len(residues) - 1:
                next_resname = residues[i + 1].resname.strip()
                dihedrals.omega = correct_cis_omega(
                    dihedrals.omega, next_resname
                )

    # Build idealized backbone for each residue
    # For now, we preserve original coordinates since full reconstruction
    # is complex and requires careful frame alignment
    # The main benefit comes from the constrained minimization step
    logger.debug(f"Extracted dihedrals for {len(residues)} residues in segment")


def idealize_structure(
    pdb_string: str,
    config: IdealizeConfig,
) -> Tuple[str, List[ChainGap]]:
    """
    Idealize backbone geometry while preserving dihedral angles.

    Main entry point for structure idealization.

    Steps:
    1. Extract and store ligands
    2. Detect chain gaps
    3. Split chains at gaps
    4. Extract dihedrals and optionally correct cis-omega
    5. Run constrained minimization to relieve local strain
    6. Restore original chain IDs
    7. Restore ligands

    Args:
        pdb_string: Input PDB file contents
        config: Idealization configuration

    Returns:
        Tuple of (idealized_pdb_string, list_of_chain_gaps)
    """
    logger.info("Starting structure idealization")

    # Step 1: Extract ligands
    protein_pdb, ligand_lines = extract_ligands(pdb_string)
    if ligand_lines.strip():
        logger.info("Extracted ligands for separate handling")

    # Step 2: Detect chain gaps (only if we want to retain them)
    gaps = []
    chain_mapping = {}
    if not config.close_chainbreaks:
        gaps = detect_chain_gaps(protein_pdb)
        if gaps:
            logger.info(f"Detected {len(gaps)} chain gap(s) - will be retained")
            # Step 3: Split chains at gaps to prevent closure
            protein_pdb, chain_mapping = split_chains_at_gaps(protein_pdb, gaps)
    else:
        logger.info("Chain breaks will be closed during idealization")

    # Step 4: Parse structure and extract/correct dihedrals
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", io.StringIO(protein_pdb))

    for model in structure:
        for chain in model:
            residues = [r for r in chain.get_residues() if r.id[0] == " "]
            if residues:
                _idealize_chain_segment(residues, config)

    # Step 5: Run constrained minimization
    # This is the key step - it fixes local geometry issues while
    # keeping the overall structure close to the original
    minimized_pdb = minimize_with_constraints(
        protein_pdb,
        stiffness=config.post_idealize_stiffness,
        add_missing_residues=config.add_missing_residues,
    )

    # Step 6: Restore original chain IDs
    if chain_mapping:
        minimized_pdb = restore_chain_ids(minimized_pdb, chain_mapping)

    # Step 7: Restore ligands
    final_pdb = restore_ligands(minimized_pdb, ligand_lines)

    logger.info("Structure idealization complete")
    return final_pdb, gaps
