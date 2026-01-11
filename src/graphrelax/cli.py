#!/usr/bin/env python
"""GraphRelax CLI - Combine LigandMPNN design with AMBER relaxation."""

import argparse
import logging
import sys
from pathlib import Path


def setup_logging(verbose: bool):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="graphrelax",
        description=(
            "GraphRelax: Combine LigandMPNN design with AMBER relaxation.\n\n"
            "This tool alternates between neural network-based design/repacking"
            " (LigandMPNN) and physics-based energy minimization (OpenMM AMBER)"
            ", similar to Rosetta FastRelax and Design protocols."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default: repack + minimize for 5 cycles
  graphrelax -i input.pdb -o relaxed.pdb

  # Only minimize (no repacking)
  graphrelax -i input.pdb -o minimized.pdb --no-repack

  # Full redesign + minimize
  graphrelax -i input.pdb -o designed.pdb --design

  # Design with resfile specification
  graphrelax -i input.pdb -o designed.pdb --design --resfile design.resfile

  # Generate 10 different designs
  graphrelax -i input.pdb -o designed.pdb --design -n 10

  # With scorefile output
  graphrelax -i input.pdb -o relaxed.pdb --scorefile scores.sc
""",
    )

    # Required arguments
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        metavar="PDB",
        help="Input PDB file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        metavar="PDB",
        help="Output PDB file (or prefix if -n > 1)",
    )

    # Mode selection (mutually exclusive)
    mode_group = parser.add_argument_group(
        "Mode selection (mutually exclusive)"
    )
    mode = mode_group.add_mutually_exclusive_group()
    mode.add_argument(
        "--relax",
        action="store_true",
        help="Repack + minimize cycles (default mode)",
    )
    mode.add_argument(
        "--repack-only",
        action="store_true",
        help="Only repack side chains, no minimization",
    )
    mode.add_argument(
        "--no-repack",
        action="store_true",
        help="Only minimize, no repacking",
    )
    mode.add_argument(
        "--design",
        action="store_true",
        help="Design + minimize (full redesign or per --resfile)",
    )
    mode.add_argument(
        "--design-only",
        action="store_true",
        help="Only design, no minimization",
    )

    # Iteration and output control
    iter_group = parser.add_argument_group("Iteration and output control")
    iter_group.add_argument(
        "--n-iter",
        type=int,
        default=5,
        metavar="N",
        help="Number of repack/design + minimize cycles (default: 5)",
    )
    iter_group.add_argument(
        "-n",
        "--n-outputs",
        type=int,
        default=1,
        metavar="N",
        help="Number of output models to generate (default: 1)",
    )

    # Design options
    design_group = parser.add_argument_group("Design options")
    design_group.add_argument(
        "--resfile",
        type=Path,
        metavar="FILE",
        help="Rosetta-style resfile for residue-specific design control",
    )
    design_group.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        metavar="T",
        help="Sampling temperature for LigandMPNN (default: 0.1)",
    )
    design_group.add_argument(
        "--model-type",
        choices=["protein_mpnn", "ligand_mpnn", "soluble_mpnn"],
        default="ligand_mpnn",
        help="LigandMPNN model variant (default: ligand_mpnn)",
    )

    # Relaxation options
    relax_group = parser.add_argument_group("Relaxation options")
    relax_group.add_argument(
        "--constrained-minimization",
        action="store_true",
        help=(
            "Use constrained minimization with position restraints and "
            "violation checking (AlphaFold-style). Default is unconstrained."
        ),
    )
    relax_group.add_argument(
        "--stiffness",
        type=float,
        default=10.0,
        metavar="K",
        help=(
            "Restraint stiffness in kcal/mol/A^2 for constrained mode "
            "(default: 10.0)"
        ),
    )
    relax_group.add_argument(
        "--max-iterations",
        type=int,
        default=0,
        metavar="N",
        help="Max L-BFGS iterations, 0=unlimited (default: 0)",
    )

    # Scoring options
    score_group = parser.add_argument_group("Scoring options")
    score_group.add_argument(
        "--scorefile",
        type=Path,
        metavar="FILE",
        help="Output scorefile with OpenMM energy terms and LigandMPNN scores",
    )

    # Input preprocessing options
    preprocess_group = parser.add_argument_group("Input preprocessing options")
    preprocess_group.add_argument(
        "--keep-waters",
        action="store_true",
        help="Keep water molecules in input (default: waters are removed)",
    )

    # General options
    general_group = parser.add_argument_group("General options")
    general_group.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    general_group.add_argument(
        "--seed",
        type=int,
        metavar="N",
        help="Random seed for reproducibility",
    )

    return parser


def main(args=None) -> int:
    """Main entry point."""
    parser = create_parser()
    opts = parser.parse_args(args)

    setup_logging(opts.verbose)
    logger = logging.getLogger(__name__)

    # Validate inputs
    if not opts.input.exists():
        logger.error(f"Input file not found: {opts.input}")
        return 1

    if opts.resfile and not opts.resfile.exists():
        logger.error(f"Resfile not found: {opts.resfile}")
        return 1

    # Import here to avoid slow startup from heavy dependencies
    from graphrelax.config import (
        DesignConfig,
        PipelineConfig,
        PipelineMode,
        RelaxConfig,
    )
    from graphrelax.pipeline import Pipeline

    # Determine mode
    if opts.repack_only:
        mode = PipelineMode.REPACK_ONLY
    elif opts.no_repack:
        mode = PipelineMode.NO_REPACK
    elif opts.design:
        mode = PipelineMode.DESIGN
    elif opts.design_only:
        mode = PipelineMode.DESIGN_ONLY
    else:
        mode = PipelineMode.RELAX  # default

    logger.info(f"Running GraphRelax in {mode.value} mode")
    logger.info(f"Input: {opts.input}")
    logger.info(f"Output: {opts.output}")
    logger.info(f"Iterations: {opts.n_iter}, Outputs: {opts.n_outputs}")

    # Build configuration
    design_config = DesignConfig(
        model_type=opts.model_type,
        temperature=opts.temperature,
        seed=opts.seed,
    )

    relax_config = RelaxConfig(
        stiffness=opts.stiffness,
        max_iterations=opts.max_iterations,
        constrained=opts.constrained_minimization,
    )

    pipeline_config = PipelineConfig(
        mode=mode,
        n_iterations=opts.n_iter,
        n_outputs=opts.n_outputs,
        scorefile=opts.scorefile,
        verbose=opts.verbose,
        remove_waters=not opts.keep_waters,
        design=design_config,
        relax=relax_config,
    )

    # Run pipeline
    try:
        pipeline = Pipeline(pipeline_config)
        results = pipeline.run(
            input_pdb=opts.input,
            output_pdb=opts.output,
            resfile=opts.resfile,
        )

        # Summary
        logger.info("=" * 50)
        logger.info("GraphRelax completed successfully!")
        logger.info(f"Generated {len(results['outputs'])} output(s)")

        for output_result in results["outputs"]:
            logger.info(f"  {output_result['output_path']}")
            if "final_energy" in output_result:
                energy = output_result["final_energy"]
                logger.info(f"    Final energy: {energy:.2f} kcal/mol")

        if opts.scorefile:
            logger.info(f"Scorefile: {opts.scorefile}")

        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Pipeline failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
