"""OpenMM AMBER relaxation wrapper."""

import io
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from openmm import Platform
from openmm import app as openmm_app
from openmm import openmm, unit

from graphrelax.chain_gaps import (
    detect_chain_gaps,
    get_gap_summary,
    restore_chain_ids,
    split_chains_at_gaps,
)
from graphrelax.config import RelaxConfig
from graphrelax.idealize import extract_ligands, restore_ligands

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
        self._use_gpu: Optional[bool] = None

    def _check_gpu_available(self) -> bool:
        """Check if CUDA is available for OpenMM."""
        if self._use_gpu is not None:
            return self._use_gpu

        for i in range(Platform.getNumPlatforms()):
            if Platform.getPlatform(i).getName() == "CUDA":
                self._use_gpu = True
                logger.info("OpenMM CUDA platform detected, using GPU")
                return True

        self._use_gpu = False
        logger.info("OpenMM CUDA not available, using CPU")
        return False

    def relax(self, pdb_string: str) -> Tuple[str, dict, np.ndarray]:
        """
        Relax a structure from PDB string.

        Uses unconstrained minimization by default, or constrained
        AmberRelaxation if config.constrained is True.

        If split_chains_at_gaps is enabled, chains will be split at detected
        gaps before minimization to prevent artificial gap closure.

        Ligands (non-water HETATM records) are extracted before relaxation
        and restored afterward. Protein atoms near ligands are restrained
        to prevent clashes when ligands are restored.

        Args:
            pdb_string: PDB file contents as string

        Returns:
            Tuple of (relaxed_pdb_string, debug_info, violations)
        """
        # Extract ligands before relaxation (AMBER can't parameterize them)
        protein_pdb, ligand_lines = extract_ligands(pdb_string)
        ligand_coords = None
        if ligand_lines.strip():
            logger.debug(
                "Extracted ligands for separate handling during relaxation"
            )
            # Parse ligand coordinates to restrain nearby protein atoms
            ligand_coords = self._parse_ligand_coords(ligand_lines)

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
                protein_pdb, ligand_coords=ligand_coords
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

    def _parse_ligand_coords(self, ligand_lines: str) -> np.ndarray:
        """Parse ligand atom coordinates from HETATM lines."""
        coords = []
        for line in ligand_lines.split("\n"):
            if line.startswith("HETATM"):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    coords.append([x, y, z])
                except (ValueError, IndexError):
                    continue
        return np.array(coords) if coords else None

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
        use_gpu = self._check_gpu_available()

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
        self, pdb_string: str, ligand_coords: np.ndarray = None
    ) -> Tuple[str, dict, np.ndarray]:
        """
        Bare-bones unconstrained OpenMM minimization.

        No position restraints on protein, uses OpenMM defaults.
        If ligand_coords is provided, adds fixed "dummy" particles at those
        positions with LJ repulsion to prevent protein from clashing with
        ligand positions.

        Args:
            pdb_string: PDB file contents as string (protein-only)
            ligand_coords: Optional array of ligand atom positions (Angstroms)

        Returns:
            Tuple of (relaxed_pdb_string, debug_info, violations)
        """
        ENERGY = unit.kilocalories_per_mole
        LENGTH = unit.angstroms

        use_gpu = self._check_gpu_available()

        has_ligand = ligand_coords is not None and len(ligand_coords) > 0
        logger.info(
            f"Running unconstrained OpenMM minimization "
            f"(max_iter={self.config.max_iterations}, gpu={use_gpu}"
            f"{', with ligand exclusion zone' if has_ligand else ''})"
        )

        # Use pdbfixer to add missing atoms and terminal groups
        from pdbfixer import PDBFixer

        fixer = PDBFixer(pdbfile=io.StringIO(pdb_string))
        fixer.findMissingResidues()
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

        n_protein_atoms = system.getNumParticles()

        # Add ligand atoms as fixed dummy particles with LJ repulsion
        ligand_particle_indices = []
        if has_ligand:
            # Add a custom nonbonded force for ligand-protein repulsion
            # Using soft-core LJ potential to prevent singularities
            ligand_repulsion = openmm.CustomNonbondedForce(
                "epsilon * (sigma/r)^12; "
                "sigma=0.3; epsilon=4.0"  # 3 Angstrom radius, 4 kJ/mol
            )
            ligand_repulsion.setNonbondedMethod(
                openmm.CustomNonbondedForce.CutoffNonPeriodic
            )
            ligand_repulsion.setCutoffDistance(1.2 * unit.nanometers)

            # Add all protein atoms to the force
            for _ in range(n_protein_atoms):
                ligand_repulsion.addParticle([])

            # Add ligand dummy particles to the system
            for _ in ligand_coords:
                # Add massless particle (won't move)
                idx = system.addParticle(0.0)
                ligand_particle_indices.append(idx)
                ligand_repulsion.addParticle([])

            # Set interaction groups: protein interacts with ligand dummies
            protein_set = set(range(n_protein_atoms))
            ligand_set = set(ligand_particle_indices)
            ligand_repulsion.addInteractionGroup(protein_set, ligand_set)

            system.addForce(ligand_repulsion)
            logger.debug(
                f"Added {len(ligand_coords)} ligand exclusion particles"
            )

        # Create integrator and simulation
        integrator = openmm.LangevinIntegrator(0, 0.01, 0.0)
        platform = openmm.Platform.getPlatformByName(
            "CUDA" if use_gpu else "CPU"
        )
        simulation = openmm_app.Simulation(
            modeller.topology, system, integrator, platform
        )

        # Set positions: protein from modeller, ligand dummies from coords
        positions = list(modeller.positions)
        if has_ligand:
            for coord in ligand_coords:
                # Convert Angstroms to nanometers
                positions.append(
                    openmm.Vec3(coord[0], coord[1], coord[2])
                    * 0.1
                    * unit.nanometers
                )
        simulation.context.setPositions(positions)

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

        # Calculate RMSD (protein atoms only)
        rmsd = np.sqrt(
            np.sum((posinit[:n_protein_atoms] - pos[:n_protein_atoms]) ** 2)
            / n_protein_atoms
        )

        # Write output PDB (protein only - exclude dummy ligand particles)
        output = io.StringIO()
        # Get only protein positions
        protein_positions = state.getPositions()[:n_protein_atoms]
        openmm_app.PDBFile.writeFile(
            modeller.topology, protein_positions, output
        )
        relaxed_pdb = output.getvalue()

        debug_data = {
            "initial_energy": einit,
            "final_energy": efinal,
            "rmsd": rmsd,
            "attempts": 1,
        }
        if has_ligand:
            debug_data["ligand_exclusion_atoms"] = len(ligand_coords)

        logger.info(
            f"Minimization complete: E_init={einit:.2f}, "
            f"E_final={efinal:.2f}, RMSD={rmsd:.3f} A"
        )

        # No violations tracking in unconstrained mode
        violations = np.zeros(0)

        return relaxed_pdb, debug_data, violations

    def _relax_direct(self, pdb_string: str) -> Tuple[str, dict, np.ndarray]:
        """
        Direct OpenMM minimization without pdbfixer.

        This is a simpler approach that works for already-complete structures
        (like those from LigandMPNN with packed side chains).

        Args:
            pdb_string: PDB file contents as string

        Returns:
            Tuple of (relaxed_pdb_string, debug_info, violations)
        """
        ENERGY = unit.kilocalories_per_mole
        LENGTH = unit.angstroms

        use_gpu = self._check_gpu_available()

        logger.info(
            f"Running direct OpenMM minimization "
            f"(max_iter={self.config.max_iterations}, "
            f"stiffness={self.config.stiffness}, gpu={use_gpu})"
        )

        # Parse PDB
        pdb_file = io.StringIO(pdb_string)
        pdb = openmm_app.PDBFile(pdb_file)

        # Create force field and system
        force_field = openmm_app.ForceField(
            "amber14-all.xml", "amber14/tip3pfb.xml"
        )

        # Use Modeller to add hydrogens (doesn't require pdbfixer)
        modeller = openmm_app.Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(force_field)

        # Create system with constraints on hydrogen bonds
        system = force_field.createSystem(
            modeller.topology, constraints=openmm_app.HBonds
        )

        # Add position restraints if stiffness > 0
        if self.config.stiffness > 0:
            self._add_restraints(
                system, modeller, self.config.stiffness * ENERGY / (LENGTH**2)
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

        # Minimize
        # OpenMM minimizeEnergy tolerance is in kJ/mol/nm (gradient threshold)
        tolerance = (
            self.config.tolerance * unit.kilojoules_per_mole / unit.nanometer
        )
        simulation.minimizeEnergy(
            maxIterations=self.config.max_iterations, tolerance=tolerance
        )

        # Get final state
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        efinal = state.getPotentialEnergy().value_in_unit(ENERGY)
        pos = state.getPositions(asNumpy=True).value_in_unit(LENGTH)

        # Calculate RMSD
        rmsd = np.sqrt(np.sum((posinit - pos) ** 2) / len(posinit))

        # Write output PDB
        output = io.StringIO()
        openmm_app.PDBFile.writeFile(
            simulation.topology, state.getPositions(), output
        )
        relaxed_pdb = output.getvalue()

        debug_data = {
            "initial_energy": einit,
            "final_energy": efinal,
            "rmsd": rmsd,
            "attempts": 1,
        }

        logger.info(
            f"Relaxation complete: E_init={einit:.2f}, "
            f"E_final={efinal:.2f}, RMSD={rmsd:.3f} A"
        )

        # No violations tracking in direct mode
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
            use_gpu = self._check_gpu_available()
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
