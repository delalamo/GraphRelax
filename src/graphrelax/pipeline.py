"""Main pipeline orchestration for GraphRelax."""

import logging
import tempfile
from pathlib import Path
from typing import Optional

from graphrelax.config import PipelineConfig, PipelineMode
from graphrelax.designer import Designer
from graphrelax.idealize import idealize_structure
from graphrelax.relaxer import Relaxer
from graphrelax.resfile import DesignSpec, ResfileParser
from graphrelax.structure_io import (
    StructureFormat,
    convert_to_format,
    detect_format,
    ensure_pdb_format,
    get_output_format,
)
from graphrelax.utils import (
    compute_ligandmpnn_score,
    compute_sequence_recovery,
    format_output_path,
    format_sequence_alignment,
    remove_waters,
    save_pdb_string,
    write_scorefile,
)

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Main orchestration class for alternating design/repack and relaxation.

    Implements the core logic of GraphRelax:
    1. Parse input PDB and resfile
    2. For each output:
        a. For each iteration:
            i. Run design/repack with LigandMPNN (if applicable)
            ii. Run AMBER minimization (if applicable)
    3. Output final structure(s) and scorefile
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.designer = Designer(config.design)
        self.relaxer = Relaxer(config.relax)
        self.resfile_parser = ResfileParser()

    def run(
        self,
        input_pdb: Path,
        output_pdb: Path,
        resfile: Optional[Path] = None,
    ) -> dict:
        """
        Run the full pipeline.

        Args:
            input_pdb: Input structure file path (PDB or CIF)
            output_pdb: Output structure file path (or prefix if n_outputs > 1)
            resfile: Optional Rosetta resfile for residue control

        Returns:
            Dictionary with pipeline results and metrics
        """
        # Detect input format and determine output format from file extensions
        input_format = detect_format(input_pdb)
        target_format = get_output_format(input_pdb, output_pdb)
        logger.info(
            f"Input format: {input_format.value}, "
            f"Output format: {target_format.value}"
        )

        # Parse resfile if provided
        design_spec = None
        if resfile:
            logger.info(f"Parsing resfile: {resfile}")
            design_spec = self.resfile_parser.parse(resfile)

        # Determine if we're doing full design
        design_all = self.config.mode in (
            PipelineMode.DESIGN,
            PipelineMode.DESIGN_ONLY,
        )
        if design_all and resfile:
            # Resfile overrides full design - use resfile specification
            design_all = False

        all_results = []
        all_scores = []

        for output_idx in range(1, self.config.n_outputs + 1):
            logger.info(
                f"Generating output {output_idx}/{self.config.n_outputs}"
            )

            result = self._run_single_output(
                input_pdb=input_pdb,
                design_spec=design_spec,
                design_all=design_all,
                input_format=input_format,
            )

            # Format output path
            out_path = format_output_path(
                output_pdb, output_idx, self.config.n_outputs
            )

            # Convert to target format if needed and save
            final_structure = result["final_pdb"]
            if target_format != StructureFormat.PDB:
                final_structure = convert_to_format(
                    final_structure, target_format
                )
            save_pdb_string(final_structure, out_path)

            result["output_path"] = out_path
            all_results.append(result)

            # Collect scores
            score_dict = {
                "total_score": result.get("final_energy", 0.0),
                "openmm_energy": result.get("final_energy", 0.0),
            }

            # Add energy breakdown if available
            if "energy_breakdown" in result:
                for key, val in result["energy_breakdown"].items():
                    if key != "total_energy":
                        score_dict[key] = val

            # Add LigandMPNN score
            if "ligandmpnn_loss" in result:
                score_dict["ligandmpnn_score"] = compute_ligandmpnn_score(
                    result["ligandmpnn_loss"]
                )

            # Add sequence recovery
            if result.get("sequence") and result.get("native_sequence"):
                score_dict["seq_recovery"] = compute_sequence_recovery(
                    result["sequence"], result["native_sequence"]
                )

            score_dict["description"] = out_path.name
            all_scores.append(score_dict)

        # Write scorefile if requested
        if self.config.scorefile and all_scores:
            write_scorefile(self.config.scorefile, all_scores)

        return {
            "outputs": all_results,
            "scores": all_scores,
        }

    def _run_single_output(
        self,
        input_pdb: Path,
        design_spec: Optional[DesignSpec],
        design_all: bool,
        input_format: StructureFormat = StructureFormat.PDB,
    ) -> dict:
        """Run pipeline for a single output."""
        result = {
            "iterations": [],
            "native_sequence": None,
            "sequence": None,
        }

        # Read input structure
        with open(input_pdb) as f:
            current_structure = f.read()

        # Remove waters if requested
        if self.config.remove_waters:
            original_lines = len(current_structure.splitlines())
            current_structure = remove_waters(current_structure, input_format)
            new_lines = len(current_structure.splitlines())
            removed = original_lines - new_lines
            if removed > 0:
                logger.info(f"Removed {removed} water-related lines from input")

        # Convert to PDB format for internal processing if needed
        current_pdb = ensure_pdb_format(current_structure, input_pdb)

        # Apply pre-idealization if enabled
        if self.config.idealize.enabled:
            logger.info("Running pre-idealization...")
            current_pdb, gaps = idealize_structure(
                current_pdb, self.config.idealize
            )
            if gaps:
                logger.info(
                    f"Idealized with {len(gaps)} chain gap(s) preserved"
                )
            else:
                logger.info("Structure idealized (no chain gaps detected)")

        # Store input as temp file for processing
        with tempfile.NamedTemporaryFile(
            suffix=".pdb", delete=False, mode="w"
        ) as tmp:
            tmp.write(current_pdb)
            current_pdb_path = Path(tmp.name)

        # Track the original native sequence (from input PDB)
        original_native_sequence = None

        try:
            for iteration in range(1, self.config.n_iterations + 1):
                logger.info(
                    f"  Iteration {iteration}/{self.config.n_iterations}"
                )

                iter_result = self._run_iteration(
                    pdb_path=current_pdb_path,
                    design_spec=design_spec,
                    design_all=design_all,
                    iteration=iteration,
                    original_native_sequence=original_native_sequence,
                )
                result["iterations"].append(iter_result)

                # Update current PDB for next iteration
                current_pdb = iter_result["output_pdb"]
                current_pdb_path.write_text(current_pdb)

                # Track sequence - capture original native on first iteration
                if "sequence" in iter_result:
                    result["sequence"] = iter_result["sequence"]
                if (
                    "native_sequence" in iter_result
                    and result["native_sequence"] is None
                ):
                    result["native_sequence"] = iter_result["native_sequence"]
                    original_native_sequence = iter_result["native_sequence"]

                # Log progress
                if "relax_info" in iter_result:
                    info = iter_result["relax_info"]
                    logger.info(
                        f"    E_init={info['initial_energy']:.2f}, "
                        f"E_final={info['final_energy']:.2f}, "
                        f"RMSD={info['rmsd']:.3f}"
                    )

            # Store final results
            result["final_pdb"] = current_pdb
            if result["iterations"]:
                last_iter = result["iterations"][-1]
                if "relax_info" in last_iter:
                    result["final_energy"] = last_iter["relax_info"][
                        "final_energy"
                    ]
                if "energy_breakdown" in last_iter:
                    result["energy_breakdown"] = last_iter["energy_breakdown"]
                if "ligandmpnn_loss" in last_iter:
                    result["ligandmpnn_loss"] = last_iter["ligandmpnn_loss"]

        finally:
            # Clean up temp file
            current_pdb_path.unlink(missing_ok=True)

        return result

    def _run_iteration(
        self,
        pdb_path: Path,
        design_spec: Optional[DesignSpec],
        design_all: bool,
        iteration: int,
        original_native_sequence: Optional[str] = None,
    ) -> dict:
        """Run a single iteration of design/repack + relax."""
        iter_result = {}

        # Phase 1: Design/Repack (if applicable)
        if self.config.mode in (PipelineMode.DESIGN, PipelineMode.DESIGN_ONLY):
            # Design mode
            logger.debug("    Running design...")
            design_result = self.designer.design(
                pdb_path=pdb_path,
                design_spec=design_spec,
                design_all=design_all,
            )
            pdb_string = self.designer.result_to_pdb_string(design_result)

            iter_result["sequence"] = design_result["sequence"]
            iter_result["native_sequence"] = design_result["native_sequence"]
            iter_result["ligandmpnn_loss"] = float(design_result["loss"][0])

            # Log sequence alignment vs original native sequence
            compare_to = (
                original_native_sequence or design_result["native_sequence"]
            )
            alignment = format_sequence_alignment(
                compare_to,
                design_result["sequence"],
            )
            logger.info(f"    Sequence design result:\n{alignment}")

        elif self.config.mode in (PipelineMode.RELAX, PipelineMode.REPACK_ONLY):
            # Repack mode
            logger.debug("    Running repack...")
            repack_result = self.designer.repack(
                pdb_path=pdb_path,
                design_spec=design_spec,
            )
            pdb_string = self.designer.result_to_pdb_string(repack_result)

            iter_result["sequence"] = repack_result["sequence"]
            iter_result["native_sequence"] = repack_result["native_sequence"]

        else:
            # No design/repack (NO_REPACK mode)
            with open(pdb_path) as f:
                pdb_string = f.read()

        iter_result["designed_pdb"] = pdb_string

        # Phase 2: Relax (if applicable)
        if self.config.mode in (
            PipelineMode.RELAX,
            PipelineMode.NO_REPACK,
            PipelineMode.DESIGN,
        ):
            logger.debug("    Running relaxation...")
            relaxed_pdb, relax_info, violations = self.relaxer.relax(pdb_string)

            iter_result["output_pdb"] = relaxed_pdb
            iter_result["relax_info"] = relax_info
            iter_result["violations"] = violations

            # Get energy breakdown for final iteration
            if iteration == self.config.n_iterations:
                iter_result["energy_breakdown"] = (
                    self.relaxer.get_energy_breakdown(relaxed_pdb)
                )
        else:
            # No relaxation
            iter_result["output_pdb"] = pdb_string

        return iter_result
