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
from graphrelax.ligand_utils import (
    WATER_RESIDUES,
    create_openff_molecule,
    extract_ligands_from_pdb,
    ligand_pdb_to_topology,
)

# Check if openmmforcefields is available for ligand support
try:
    from openmmforcefields.generators import SystemGenerator

    OPENMMFF_AVAILABLE = True
except ImportError:
    OPENMMFF_AVAILABLE = False

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

        Args:
            pdb_string: PDB file contents as string

        Returns:
            Tuple of (relaxed_pdb_string, debug_info, violations)
        """
        # Detect and handle chain gaps if configured
        chain_mapping = {}
        if self.config.split_chains_at_gaps:
            gaps = detect_chain_gaps(pdb_string)
            if gaps:
                logger.info(get_gap_summary(gaps))
                pdb_string, chain_mapping = split_chains_at_gaps(
                    pdb_string, gaps
                )

        if self.config.constrained:
            prot = protein.from_pdb_string(pdb_string)
            relaxed_pdb, debug_info, violations = self.relax_protein(prot)
        else:
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
        self, pdb_string: str
    ) -> Tuple[str, dict, np.ndarray]:
        """
        Bare-bones unconstrained OpenMM minimization.

        No position restraints, no violation checking, uses OpenMM defaults.
        This is the default minimization mode.

        If config.include_ligands is True and ligands are present, uses
        openmmforcefields for ligand parameterization.

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

        if has_ligands and self.config.include_ligands:
            if not OPENMMFF_AVAILABLE:
                raise ImportError(
                    "openmmforcefields is required for ligand support. "
                    "Install with: pip install openmmforcefields openff-toolkit"
                )
            return self._relax_unconstrained_with_ligands(pdb_string)

        ENERGY = unit.kilocalories_per_mole
        LENGTH = unit.angstroms

        use_gpu = self._check_gpu_available()

        logger.info(
            f"Running unconstrained OpenMM minimization "
            f"(max_iter={self.config.max_iterations}, gpu={use_gpu})"
        )

        # Separate protein (ATOM) from ligands (HETATM) for proper processing
        # Ligands in the same chain confuse pdbfixer's terminal detection
        atom_lines = []
        hetatm_lines = []
        for line in pdb_string.split("\n"):
            if line.startswith("HETATM"):
                hetatm_lines.append(line)
            elif line.startswith("END"):
                pass  # Skip END, we'll add it back
            else:
                atom_lines.append(line)

        protein_pdb = "\n".join(atom_lines) + "\nEND\n"

        # Use pdbfixer to add missing atoms and terminal groups
        from pdbfixer import PDBFixer

        fixer = PDBFixer(pdbfile=io.StringIO(protein_pdb))
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

        Args:
            pdb_string: PDB file contents as string

        Returns:
            Tuple of (relaxed_pdb_string, debug_info, violations)
        """
        from pdbfixer import PDBFixer

        ENERGY = unit.kilocalories_per_mole
        LENGTH = unit.angstroms

        use_gpu = self._check_gpu_available()

        logger.info(
            f"Running unconstrained minimization with ligands "
            f"(forcefield={self.config.ligand_forcefield}, gpu={use_gpu})"
        )

        # Step 1: Separate protein and ligands
        protein_pdb, ligands = extract_ligands_from_pdb(pdb_string)
        ligand_names = [lig.resname for lig in ligands]
        logger.info(f"Found {len(ligands)} ligand(s): {ligand_names}")

        # Step 2: Fix protein with pdbfixer (without ligands)
        fixer = PDBFixer(pdbfile=io.StringIO(protein_pdb))
        fixer.findMissingResidues()
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
            "ligands_included": [lig.resname for lig in ligands],
            "ligand_forcefield": self.config.ligand_forcefield,
        }

        logger.info(
            f"Minimization complete: E_init={einit:.2f}, "
            f"E_final={efinal:.2f}, RMSD={rmsd:.3f} A"
        )

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
