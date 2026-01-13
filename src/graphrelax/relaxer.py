"""OpenMM AMBER relaxation wrapper."""

import io
import logging
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
from openmm import app as openmm_app
from openmm import openmm, unit
from pdbfixer import PDBFixer

from graphrelax.artifacts import WATER_RESIDUES
from graphrelax.chain_gaps import (
    detect_chain_gaps,
    get_gap_summary,
    restore_chain_ids,
    split_chains_at_gaps,
)
from graphrelax.config import RelaxConfig
from graphrelax.idealize import extract_ligands, restore_ligands
from graphrelax.ligand_utils import (
    create_openff_molecule,
    extract_ligands_from_pdb,
    is_unparameterizable_cofactor,
    ligand_pdb_to_topology,
)
from graphrelax.utils import check_gpu_available

# openmmforcefields is only available via conda-forge
# Install with: conda install -c conda-forge openmmforcefields=0.13.0
try:
    from openmmforcefields.generators import SystemGenerator

    OPENMMFF_AVAILABLE = True
except ImportError:
    OPENMMFF_AVAILABLE = False
    SystemGenerator = None


def _check_openmmff():
    """Raise ImportError if openmmforcefields is not available."""
    if not OPENMMFF_AVAILABLE:
        raise ImportError(
            "Ligand support requires openmmforcefields.\n"
            "This package is only available via conda-forge:\n\n"
            "  conda install -c conda-forge openmmforcefields=0.13.0\n\n"
            "See the README for full installation instructions."
        )


# Add vendored LigandMPNN to path for OpenFold imports
# Must happen before importing from openfold
LIGANDMPNN_PATH = Path(__file__).parent / "LigandMPNN"
if str(LIGANDMPNN_PATH) not in sys.path:
    sys.path.insert(0, str(LIGANDMPNN_PATH))

from openfold.np import protein  # noqa: E402
from openfold.np.relax.relax import AmberRelaxation  # noqa: E402

logger = logging.getLogger(__name__)


class Relaxer:
    """Wrapper for OpenMM AMBER relaxation."""

    def __init__(self, config: RelaxConfig):
        self.config = config

    def relax(self, pdb_string: str) -> Tuple[str, dict, np.ndarray]:
        """
        Relax a structure from PDB string.

        Uses unconstrained minimization by default, or constrained
        AmberRelaxation if config.constrained is True.

        If split_chains_at_gaps is enabled, chains will be split at detected
        gaps before minimization to prevent artificial gap closure.

        For unconstrained minimization with ligands present, ligands are
        parameterized using openmmforcefields and included in the minimization.

        For constrained minimization, ligands are extracted before relaxation
        and restored afterward (unchanged) since AmberRelaxation cannot
        handle arbitrary ligands.

        Args:
            pdb_string: PDB file contents as string

        Returns:
            Tuple of (relaxed_pdb_string, debug_info, violations)
        """
        # Check if ligands are present (non-water HETATM records)
        has_ligands = any(
            line.startswith("HETATM")
            and line[17:20].strip() not in WATER_RESIDUES
            for line in pdb_string.split("\n")
        )

        # For unconstrained minimization with ligands, use the ligand-aware path
        # which includes ligands in the minimization via openmmforcefields
        if not self.config.constrained and has_ligands:
            if not self.config.ignore_ligands:
                # Detect and handle chain gaps on protein portion only
                chain_mapping = {}
                if self.config.split_chains_at_gaps:
                    protein_pdb, _ = extract_ligands(pdb_string)
                    gaps = detect_chain_gaps(protein_pdb)
                    if gaps:
                        logger.info(get_gap_summary(gaps))
                        # Split the full PDB (with ligands) at gaps
                        pdb_string, chain_mapping = split_chains_at_gaps(
                            pdb_string, gaps
                        )

                # Run unconstrained minimization with ligands included
                relaxed_pdb, debug_info, violations = self._relax_unconstrained(
                    pdb_string
                )

                # Restore original chain IDs if chains were split
                if chain_mapping:
                    relaxed_pdb = restore_chain_ids(relaxed_pdb, chain_mapping)
                    debug_info["chains_split"] = True
                    debug_info["gaps_detected"] = len(
                        [k for k, v in chain_mapping.items() if k != v]
                    )

                return relaxed_pdb, debug_info, violations

        # For constrained minimization or when ignoring ligands:
        # Extract ligands, relax protein-only, restore ligands
        protein_pdb, ligand_lines = extract_ligands(pdb_string)
        if ligand_lines.strip():
            logger.debug(
                "Extracted ligands for separate handling during relaxation"
            )

        # Detect and handle chain gaps if configured
        chain_mapping = {}
        if self.config.split_chains_at_gaps:
            gaps = detect_chain_gaps(protein_pdb)
            if gaps:
                logger.info(get_gap_summary(gaps))
                protein_pdb, chain_mapping = split_chains_at_gaps(
                    protein_pdb, gaps
                )

        if self.config.constrained:
            prot = protein.from_pdb_string(protein_pdb)
            relaxed_pdb, debug_info, violations = self.relax_protein(prot)
        else:
            relaxed_pdb, debug_info, violations = self._relax_unconstrained(
                protein_pdb
            )

        # Restore original chain IDs if chains were split
        if chain_mapping:
            relaxed_pdb = restore_chain_ids(relaxed_pdb, chain_mapping)
            debug_info["chains_split"] = True
            debug_info["gaps_detected"] = len(
                [k for k, v in chain_mapping.items() if k != v]
            )

        # Restore ligands after relaxation
        relaxed_pdb = restore_ligands(relaxed_pdb, ligand_lines)

        return relaxed_pdb, debug_info, violations

    def relax_pdb_file(self, pdb_path: Path) -> Tuple[str, dict, np.ndarray]:
        """
        Relax a PDB file.

        Args:
            pdb_path: Path to input PDB file

        Returns:
            Tuple of (relaxed_pdb_string, debug_info, violations)
        """
        with open(pdb_path) as f:
            pdb_string = f.read()
        return self.relax(pdb_string)

    def relax_protein(self, prot) -> Tuple[str, dict, np.ndarray]:
        """
        Relax a Protein object using OpenFold's AmberRelaxation.

        Args:
            prot: OpenFold Protein object

        Returns:
            Tuple of (relaxed_pdb_string, debug_info, violations)
        """
        use_gpu = check_gpu_available()

        relaxer = AmberRelaxation(
            max_iterations=self.config.max_iterations,
            tolerance=self.config.tolerance,
            stiffness=self.config.stiffness,
            exclude_residues=[],
            max_outer_iterations=self.config.max_outer_iterations,
            use_gpu=use_gpu,
        )

        logger.info(
            f"Running AMBER relaxation (max_iter={self.config.max_iterations}, "
            f"stiffness={self.config.stiffness}, gpu={use_gpu})"
        )

        relaxed_pdb, debug_data, violations = relaxer.process(prot=prot)

        logger.info(
            f"Relaxation complete: E_init={debug_data['initial_energy']:.2f}, "
            f"E_final={debug_data['final_energy']:.2f}, "
            f"RMSD={debug_data['rmsd']:.3f} A"
        )

        return relaxed_pdb, debug_data, violations

    def _relax_unconstrained(
        self, pdb_string: str
    ) -> Tuple[str, dict, np.ndarray]:
        """
        Bare-bones unconstrained OpenMM minimization.

        No position restraints, no violation checking, uses OpenMM defaults.
        This is the default minimization mode.

        Ligands are auto-detected. If present and ignore_ligands is False,
        uses openmmforcefields for ligand parameterization.

        Args:
            pdb_string: PDB file contents as string (protein-only)

        Returns:
            Tuple of (relaxed_pdb_string, debug_info, violations)
        """
        # Check if ligands are present (non-water HETATM records)
        has_ligands = any(
            line.startswith("HETATM")
            and line[17:20].strip() not in WATER_RESIDUES
            for line in pdb_string.split("\n")
        )

        # Auto-detect ligands and use openmmforcefields unless ignore_ligands
        if has_ligands and not self.config.ignore_ligands:
            return self._relax_unconstrained_with_ligands(pdb_string)

        ENERGY = unit.kilocalories_per_mole
        LENGTH = unit.angstroms

        use_gpu = check_gpu_available()

        logger.info(
            f"Running unconstrained OpenMM minimization "
            f"(max_iter={self.config.max_iterations}, gpu={use_gpu})"
        )

        # Use pdbfixer to add missing atoms and terminal groups
        fixer = PDBFixer(pdbfile=io.StringIO(pdb_string))
        fixer.findMissingResidues()
        if not self.config.add_missing_residues:
            fixer.missingResidues = {}  # Clear to preserve original numbering
        elif fixer.missingResidues:
            n_missing = sum(len(v) for v in fixer.missingResidues.values())
            logger.info(f"Adding {n_missing} missing residues from SEQRES")
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()

        # Create force field and system
        force_field = openmm_app.ForceField(
            "amber14-all.xml", "amber14/tip3pfb.xml"
        )

        # Use Modeller to add hydrogens
        modeller = openmm_app.Modeller(fixer.topology, fixer.positions)
        modeller.addHydrogens(force_field)

        # Create system with HBonds constraints (standard for minimization)
        system = force_field.createSystem(
            modeller.topology, constraints=openmm_app.HBonds
        )

        # Create integrator and simulation
        integrator = openmm.LangevinIntegrator(0, 0.01, 0.0)
        platform = openmm.Platform.getPlatformByName(
            "CUDA" if use_gpu else "CPU"
        )
        simulation = openmm_app.Simulation(
            modeller.topology, system, integrator, platform
        )
        simulation.context.setPositions(modeller.positions)

        # Get initial energy
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        einit = state.getPotentialEnergy().value_in_unit(ENERGY)
        posinit = state.getPositions(asNumpy=True).value_in_unit(LENGTH)

        # Minimize with default tolerance
        if self.config.max_iterations > 0:
            simulation.minimizeEnergy(maxIterations=self.config.max_iterations)
        else:
            simulation.minimizeEnergy()

        # Get final state
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        efinal = state.getPotentialEnergy().value_in_unit(ENERGY)
        pos = state.getPositions(asNumpy=True).value_in_unit(LENGTH)

        # Calculate RMSD
        rmsd = np.sqrt(np.sum((posinit - pos) ** 2) / len(posinit))

        # Write output PDB (keepIds=True preserves original residue numbering)
        output = io.StringIO()
        openmm_app.PDBFile.writeFile(
            simulation.topology, state.getPositions(), output, keepIds=True
        )
        relaxed_pdb = output.getvalue()

        debug_data = {
            "initial_energy": einit,
            "final_energy": efinal,
            "rmsd": rmsd,
            "attempts": 1,
        }

        logger.info(
            f"Minimization complete: E_init={einit:.2f}, "
            f"E_final={efinal:.2f}, RMSD={rmsd:.3f} A"
        )

        # No violations tracking in unconstrained mode
        violations = np.zeros(0)

        return relaxed_pdb, debug_data, violations

    def _relax_unconstrained_with_ligands(
        self, pdb_string: str
    ) -> Tuple[str, dict, np.ndarray]:
        """
        Unconstrained OpenMM minimization with ligand support.

        Uses openmmforcefields to parameterize ligands with GAFF2/OpenFF.
        The protein is processed separately with pdbfixer, then combined
        with parameterized ligands for minimization.

        Requires openmmforcefields (install via conda-forge).

        Args:
            pdb_string: PDB file contents as string

        Returns:
            Tuple of (relaxed_pdb_string, debug_info, violations)

        Raises:
            ImportError: If openmmforcefields is not installed.
        """
        _check_openmmff()

        ENERGY = unit.kilocalories_per_mole
        LENGTH = unit.angstroms

        use_gpu = check_gpu_available()

        logger.info(
            f"Running unconstrained minimization with ligands "
            f"(forcefield={self.config.ligand_forcefield}, gpu={use_gpu})"
        )

        # Step 1: Separate protein and ligands
        protein_pdb, ligands = extract_ligands_from_pdb(pdb_string)

        # Check for unparameterizable cofactors and fail early
        for lig in ligands:
            if is_unparameterizable_cofactor(lig.resname):
                raise ValueError(
                    f"Cannot parameterize cofactor '{lig.resname}' "
                    f"(chain {lig.chain_id}, res {lig.resnum}). "
                    f"Metallocofactors like heme, Fe-S clusters, and "
                    f"chlorophylls cannot be parameterized.\n\n"
                    f"Options:\n"
                    f"  1. Remove the cofactor from the input PDB\n"
                    f"  2. Use --ignore-ligands to exclude all ligands\n"
                    f"  3. Use --constrained-minimization (protein-only)"
                )

        ligand_names = [lig.resname for lig in ligands]
        logger.info(f"Found {len(ligands)} ligand(s): {ligand_names}")

        # Step 2: Fix protein with pdbfixer (without ligands)
        fixer = PDBFixer(pdbfile=io.StringIO(protein_pdb))
        fixer.findMissingResidues()
        if not self.config.add_missing_residues:
            fixer.missingResidues = {}  # Clear to preserve original numbering
        elif fixer.missingResidues:
            n_missing = sum(len(v) for v in fixer.missingResidues.values())
            logger.info(f"Adding {n_missing} missing residues from SEQRES")
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()

        # Step 3: Create OpenFF molecules for ligands
        # User-provided SMILES override automatic detection
        user_smiles = self.config.ligand_smiles or {}

        openff_molecules = []
        for ligand in ligands:
            smiles = user_smiles.get(ligand.resname)
            try:
                mol = create_openff_molecule(ligand, smiles=smiles)
                openff_molecules.append(mol)
                logger.debug(f"Created OpenFF molecule for {ligand.resname}")
            except Exception as e:
                resname = ligand.resname
                raise ValueError(
                    f"Could not parameterize ligand '{resname}' "
                    f"(chain {ligand.chain_id}, res {ligand.resnum}): {e}\n\n"
                    f"Options to resolve:\n"
                    f"  1. Provide SMILES via config: "
                    f"ligand_smiles={{'{resname}': 'SMILES_STRING'}}\n"
                    f"  2. Use constrained minimization: constrained=True\n"
                    f"  3. Remove the ligand from the input PDB\n\n"
                    f"For common ligands, see: "
                    f"https://www.ebi.ac.uk/pdbe-srv/pdbechem/"
                )

        # Step 4: Create combined topology using Modeller
        modeller = openmm_app.Modeller(fixer.topology, fixer.positions)

        # Add ligands back to modeller
        for ligand in ligands:
            ligand_topology, ligand_positions = ligand_pdb_to_topology(ligand)
            modeller.add(ligand_topology, ligand_positions)

        # Step 5: Create SystemGenerator with ligand support
        system_generator = SystemGenerator(
            forcefields=["amber/ff14SB.xml"],
            small_molecule_forcefield=self.config.ligand_forcefield,
            molecules=openff_molecules,
            forcefield_kwargs={
                "constraints": openmm_app.HBonds,
                "removeCMMotion": True,
            },
        )

        # Add hydrogens
        modeller.addHydrogens(system_generator.forcefield)

        # Create system
        system = system_generator.create_system(modeller.topology)

        # Step 6: Run minimization
        integrator = openmm.LangevinIntegrator(0, 0.01, 0.0)
        platform = openmm.Platform.getPlatformByName(
            "CUDA" if use_gpu else "CPU"
        )
        simulation = openmm_app.Simulation(
            modeller.topology, system, integrator, platform
        )
        simulation.context.setPositions(modeller.positions)

        # Get initial energy
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        einit = state.getPotentialEnergy().value_in_unit(ENERGY)
        posinit = state.getPositions(asNumpy=True).value_in_unit(LENGTH)

        # Minimize
        if self.config.max_iterations > 0:
            simulation.minimizeEnergy(maxIterations=self.config.max_iterations)
        else:
            simulation.minimizeEnergy()

        # Get final state
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        efinal = state.getPotentialEnergy().value_in_unit(ENERGY)
        pos = state.getPositions(asNumpy=True).value_in_unit(LENGTH)

        # Calculate RMSD
        rmsd = np.sqrt(np.sum((posinit - pos) ** 2) / len(posinit))

        # Write output PDB (keepIds=True preserves original residue numbering)
        output = io.StringIO()
        openmm_app.PDBFile.writeFile(
            simulation.topology, state.getPositions(), output, keepIds=True
        )
        relaxed_pdb = output.getvalue()

        debug_data = {
            "initial_energy": einit,
            "final_energy": efinal,
            "rmsd": rmsd,
            "attempts": 1,
            "ligands_included": [lig.resname for lig in ligands],
            "ligand_forcefield": self.config.ligand_forcefield,
        }

        logger.info(
            f"Minimization complete: E_init={einit:.2f}, "
            f"E_final={efinal:.2f}, RMSD={rmsd:.3f} A"
        )

        violations = np.zeros(0)
        return relaxed_pdb, debug_data, violations

    def _add_restraints(self, system, modeller, stiffness):
        """Add harmonic position restraints to heavy atoms."""
        force = openmm.CustomExternalForce(
            "0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)"
        )
        # Convert stiffness to OpenMM internal units (kJ/mol/nm^2)
        stiffness_value = stiffness.value_in_unit(
            unit.kilojoules_per_mole / unit.nanometers**2
        )
        force.addGlobalParameter("k", stiffness_value)
        for p in ["x0", "y0", "z0"]:
            force.addPerParticleParameter(p)

        for i, atom in enumerate(modeller.topology.atoms()):
            if atom.element.name != "hydrogen":
                # Convert positions to nanometers (OpenMM internal units)
                pos = modeller.positions[i].value_in_unit(unit.nanometers)
                force.addParticle(i, pos)

        logger.debug(
            f"Added restraints to {force.getNumParticles()} / "
            f"{system.getNumParticles()} atoms"
        )
        system.addForce(force)

    def get_energy_breakdown(self, pdb_string: str) -> dict:
        """
        Get individual force field energy terms for a structure.

        Args:
            pdb_string: PDB file contents as string

        Returns:
            Dictionary with energy breakdown by force type
        """
        try:
            ENERGY = unit.kilocalories_per_mole

            # Parse PDB
            pdb_file = io.StringIO(pdb_string)
            pdb = openmm_app.PDBFile(pdb_file)

            # Create force field and system
            force_field = openmm_app.ForceField("amber99sb.xml")
            system = force_field.createSystem(
                pdb.topology, constraints=openmm_app.HBonds
            )

            # Create simulation
            use_gpu = check_gpu_available()
            platform = openmm.Platform.getPlatformByName(
                "CUDA" if use_gpu else "CPU"
            )
            integrator = openmm.LangevinIntegrator(0, 0.01, 0.0)
            simulation = openmm_app.Simulation(
                pdb.topology, system, integrator, platform
            )
            simulation.context.setPositions(pdb.positions)

            # Get total energy
            state = simulation.context.getState(getEnergy=True)
            total_energy = state.getPotentialEnergy().value_in_unit(ENERGY)

            # Get energy by force group
            energy_breakdown = {"total_energy": total_energy}

            # Map force types to names
            force_names = {
                "HarmonicBondForce": "bond_energy",
                "HarmonicAngleForce": "angle_energy",
                "PeriodicTorsionForce": "dihedral_energy",
                "NonbondedForce": "nonbonded_energy",
            }

            for i in range(system.getNumForces()):
                force = system.getForce(i)
                force_type = force.__class__.__name__

                # Set this force to group i
                force.setForceGroup(i)

            # Recreate simulation with force groups
            simulation = openmm_app.Simulation(
                pdb.topology, system, integrator, platform
            )
            simulation.context.setPositions(pdb.positions)

            # Get energy for each force group
            for i in range(system.getNumForces()):
                force = system.getForce(i)
                force_type = force.__class__.__name__

                state = simulation.context.getState(getEnergy=True, groups={i})
                energy = state.getPotentialEnergy().value_in_unit(ENERGY)

                name = force_names.get(force_type, force_type.lower())
                energy_breakdown[name] = energy

            return energy_breakdown

        except Exception as e:
            logger.warning(f"Could not compute energy breakdown: {e}")
            return {"total_energy": 0.0}
