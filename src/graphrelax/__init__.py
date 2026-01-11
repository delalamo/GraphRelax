"""
GraphRelax: Combine LigandMPNN sequence design with AMBER relaxation.

This package provides a CLI tool that alternates between neural network-based
sequence design/repacking (LigandMPNN) and physics-based energy minimization
(OpenMM AMBER), similar to Rosetta FastRelax and Design protocols.
"""

try:
    from graphrelax._version import __version__
except ImportError:
    # Package not installed (running from source without build)
    __version__ = "0.0.0.dev0"


def __getattr__(name):
    """Lazy import heavy modules only when accessed."""
    if name == "Pipeline":
        from graphrelax.pipeline import Pipeline

        return Pipeline
    elif name == "PipelineMode":
        from graphrelax.config import PipelineMode

        return PipelineMode
    elif name == "PipelineConfig":
        from graphrelax.config import PipelineConfig

        return PipelineConfig
    elif name == "DesignConfig":
        from graphrelax.config import DesignConfig

        return DesignConfig
    elif name == "RelaxConfig":
        from graphrelax.config import RelaxConfig

        return RelaxConfig
    elif name == "ResidueMode":
        from graphrelax.resfile import ResidueMode

        return ResidueMode
    elif name == "ResidueSpec":
        from graphrelax.resfile import ResidueSpec

        return ResidueSpec
    elif name == "DesignSpec":
        from graphrelax.resfile import DesignSpec

        return DesignSpec
    elif name == "ResfileParser":
        from graphrelax.resfile import ResfileParser

        return ResfileParser
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "PipelineMode",
    "PipelineConfig",
    "DesignConfig",
    "RelaxConfig",
    "ResidueMode",
    "ResidueSpec",
    "DesignSpec",
    "ResfileParser",
    "Pipeline",
]
