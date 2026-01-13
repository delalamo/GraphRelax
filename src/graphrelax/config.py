"""Configuration dataclasses for GraphRelax pipeline."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal, Optional


class PipelineMode(Enum):
    """Operating mode for the pipeline."""

    RELAX = "relax"  # Repack + minimize (default)
    REPACK_ONLY = "repack_only"  # Only repack, no minimize
    NO_REPACK = "no_repack"  # Only minimize, no repack
    DESIGN = "design"  # Design + minimize
    DESIGN_ONLY = "design_only"  # Only design, no minimize


ModelType = Literal["protein_mpnn", "ligand_mpnn", "soluble_mpnn"]


@dataclass
class DesignConfig:
    """Configuration for LigandMPNN design/repacking."""

    model_type: ModelType = "ligand_mpnn"
    temperature: float = 0.1
    pack_side_chains: bool = True
    seed: Optional[int] = None
    use_ligand_context: bool = True
    sc_num_denoising_steps: int = 3
    sc_num_samples: int = 16


@dataclass
class RelaxConfig:
    """Configuration for AMBER relaxation."""

    max_iterations: int = 0  # 0 = no limit
    tolerance: float = 2.39  # kcal/mol (OpenMM default)
    stiffness: float = 10.0  # kcal/mol/A^2
    max_outer_iterations: int = 3  # Violation-fixing iterations
    constrained: bool = False  # Use constrained (AmberRelaxation) minimization
    split_chains_at_gaps: bool = True  # Split chains at gaps to prevent closure
    # GPU is auto-detected and used when available


@dataclass
class IdealizeConfig:
    """Configuration for structure idealization preprocessing."""

    enabled: bool = False  # Idealization disabled by default
    fix_cis_omega: bool = True  # Correct non-trans peptide bonds (except Pro)
    post_idealize_stiffness: float = 10.0  # kcal/mol/A^2 for restraint
    add_missing_residues: bool = True  # Add missing residues from SEQRES
    close_chainbreaks: bool = True  # Close chain breaks during idealization


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline."""

    mode: PipelineMode = PipelineMode.RELAX
    n_iterations: int = 5  # Number of repack/design + minimize cycles
    n_outputs: int = 1  # Number of output models to generate
    scorefile: Optional[Path] = None  # If set, write scores to this file
    verbose: bool = False
    remove_waters: bool = True  # Remove water molecules from input
    design: DesignConfig = field(default_factory=DesignConfig)
    relax: RelaxConfig = field(default_factory=RelaxConfig)
    idealize: IdealizeConfig = field(default_factory=IdealizeConfig)
