"""RFD4-Proteina sequence design and repacking interface."""

from pathlib import Path
from typing import Optional

from graphrelax.config import DesignConfig
from graphrelax.resfile import DesignSpec
from graphrelax.rfd4 import RFD4Engine


class Designer:
    """Generate all-atom structures with fixed-backbone RFD4-Proteina."""

    def __init__(self, config: DesignConfig):
        self.config = config
        self.engine = RFD4Engine(
            seed=config.seed,
            tsr_k=config.tsr_k,
            tsr_sigma=config.tsr_sigma,
            recycle_strategy=config.recycle_strategy,
        )

    def design(
        self,
        pdb_path: Path,
        design_spec: Optional[DesignSpec] = None,
        design_all: bool = False,
    ) -> dict:
        return self.engine.design(pdb_path, design_spec, design_all)

    def repack(
        self, pdb_path: Path, design_spec: Optional[DesignSpec] = None
    ) -> dict:
        return self.engine.repack(pdb_path, design_spec)

    def result_to_pdb_string(self, result: dict) -> str:
        return result["pdb_string"]
