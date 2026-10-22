"""LigandMPNN sequence design with RFD4-Proteina side-chain packing."""

import logging
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from prody import writePDB

from graphrelax.config import DesignConfig
from graphrelax.repacker import Repacker
from graphrelax.resfile import ALL_AAS, DesignSpec, ResidueMode
from graphrelax.weights import find_weights_dir

logger = logging.getLogger(__name__)

# Add vendored LigandMPNN to path
LIGANDMPNN_PATH = Path(__file__).parent / "LigandMPNN"
if str(LIGANDMPNN_PATH) not in sys.path:
    sys.path.insert(0, str(LIGANDMPNN_PATH))

# LigandMPNN imports (after sys.path manipulation)
from data_utils import (  # noqa: E402
    featurize,
    get_score,
    parse_PDB,
    restype_1to3,
    restype_int_to_str,
    restype_str_to_int,
    write_full_PDB,
)
from model_utils import ProteinMPNN  # noqa: E402


class Designer:
    """LigandMPNN sequence design and RFD4-Proteina packing."""

    CHECKPOINT_PATHS = {
        "protein_mpnn": "proteinmpnn_v_48_020.pt",
        "ligand_mpnn": "ligandmpnn_v_32_010_25.pt",
        "soluble_mpnn": "solublempnn_v_48_020.pt",
    }

    def __init__(self, config: DesignConfig):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model: Optional[ProteinMPNN] = None
        self._packer = Repacker(config.seed)
        self._setup_seed()

        logger.info(f"Designer initialized with device: {self.device}")

    def _setup_seed(self):
        """Set random seeds for reproducibility."""
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            np.random.seed(self.config.seed)

    def _load_models(self):
        """Lazy load models on first use."""
        if self._model is not None:
            return

        weights_dir = find_weights_dir()
        checkpoint_path = (
            weights_dir / self.CHECKPOINT_PATHS[self.config.model_type]
        )
        logger.info(f"Loading model from {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        k_neighbors = checkpoint["num_edges"]
        atom_context_num = checkpoint.get("atom_context_num", 1)

        self._model = ProteinMPNN(
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            k_neighbors=k_neighbors,
            device=self.device,
            atom_context_num=atom_context_num,
            model_type=self.config.model_type,
            ligand_mpnn_use_side_chain_context=0,
        )
        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._model.to(self.device)
        self._model.eval()

    def design(
        self,
        pdb_path: Path,
        design_spec: Optional[DesignSpec] = None,
        design_all: bool = False,
    ) -> dict:
        """
        Run sequence design on a structure.

        Args:
            pdb_path: Path to input PDB
            design_spec: Specification of which residues to design/fix
            design_all: If True, design all residues (full redesign)

        Returns:
            Dictionary with designed sequence, structure, and scores
        """
        self._load_models()

        # Parse PDB
        parse_all_atoms = (
            self.config.use_ligand_context or self.config.pack_side_chains
        )
        protein_dict, backbone, other_atoms, icodes, _ = parse_PDB(
            str(pdb_path),
            device=self.device,
            parse_all_atoms=parse_all_atoms,
        )

        # Build residue key mapping
        encoded_residues = self._build_residue_mapping(protein_dict, icodes)

        # Create chain_mask based on design_spec
        chain_mask = self._build_chain_mask(
            encoded_residues, design_spec, design_all
        )
        protein_dict["chain_mask"] = chain_mask

        # Build bias tensor for PIKAA/NOTAA constraints
        bias = self._build_aa_bias(encoded_residues, design_spec)

        # Featurize
        atom_context_num = 16 if self.config.model_type == "ligand_mpnn" else 1
        feature_dict = featurize(
            protein_dict,
            cutoff_for_score=8.0,
            use_atom_context=self.config.use_ligand_context,
            number_of_ligand_atoms=atom_context_num,
            model_type=self.config.model_type,
        )

        # Set sampling parameters
        B = 1  # batch size
        L = feature_dict["mask"].shape[1]
        feature_dict["batch_size"] = B
        feature_dict["temperature"] = self.config.temperature
        feature_dict["bias"] = bias
        feature_dict["symmetry_residues"] = [[]]
        feature_dict["symmetry_weights"] = [[]]
        feature_dict["randn"] = torch.randn([B, L], device=self.device)

        # Run design
        with torch.no_grad():
            output_dict = self._model.sample(feature_dict)

            # Compute scores
            loss, loss_per_residue = get_score(
                output_dict["S"],
                output_dict["log_probs"],
                feature_dict["mask"] * feature_dict["chain_mask"],
            )

        # Extract sequence
        seq_ints = output_dict["S"][0].cpu().numpy()
        sequence = "".join([restype_int_to_str[aa] for aa in seq_ints])

        # Get native sequence for recovery calculation
        native_seq_ints = feature_dict["S"][0].cpu().numpy()
        native_sequence = "".join(
            [restype_int_to_str[aa] for aa in native_seq_ints]
        )

        result = {
            "sequence": sequence,
            "native_sequence": native_sequence,
            "S": output_dict["S"],
            "log_probs": output_dict["log_probs"],
            "loss": loss.cpu().numpy(),
            "loss_per_residue": loss_per_residue.cpu().numpy(),
            "coordinates": output_dict.get("X"),
            "coord_mask": output_dict.get("X_m"),
            "b_factors": output_dict.get("b_factors"),
            "backbone": backbone,
            "other_atoms": other_atoms,
            "icodes": icodes,
            "protein_dict": protein_dict,
            "feature_dict": feature_dict,
            "encoded_residues": encoded_residues,
        }
        if self.config.pack_side_chains:
            with tempfile.TemporaryDirectory() as tmp:
                designed_path = Path(tmp) / "designed.pdb"
                designed_path.write_text(self.result_to_pdb_string(result))
                result["pdb_string"] = self._packer.repack(
                    designed_path, design_spec, original_path=pdb_path
                )["pdb_string"]
        return result

    def repack(
        self,
        pdb_path: Path,
        design_spec: Optional[DesignSpec] = None,
    ) -> dict:
        """
        Repack side chains without changing sequence.

        Args:
            pdb_path: Path to input PDB
            design_spec: Specification (NATRO residues excluded from repacking)

        Returns:
            Dictionary with repacked structure
        """
        return self._packer.repack(pdb_path, design_spec)

    def result_to_pdb_string(self, result: dict) -> str:
        """Convert design/repack result to PDB string."""
        if "pdb_string" in result:
            return result["pdb_string"]
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            temp_path = f.name

        try:
            if result.get("coordinates") is not None:
                # Full atom structure from packer
                write_full_PDB(
                    temp_path,
                    result["coordinates"][0].cpu().numpy(),
                    result["coord_mask"][0].cpu().numpy(),
                    result.get(
                        "b_factors", torch.zeros_like(result["coord_mask"])
                    )[0]
                    .cpu()
                    .numpy(),
                    result["feature_dict"]["R_idx_original"][0].cpu().numpy(),
                    result["protein_dict"]["chain_letters"],
                    result["S"][0].cpu().numpy(),
                    other_atoms=result["other_atoms"],
                    icodes=result["icodes"],
                )
            else:
                # Backbone only - use prody
                backbone = result["backbone"].copy()
                residue_indices = backbone.getResindices()
                names = dict(
                    zip(
                        np.unique(residue_indices),
                        [restype_1to3[aa] for aa in result["sequence"]],
                    )
                )
                backbone.setResnames(
                    [names[index] for index in residue_indices]
                )
                if result["other_atoms"]:
                    writePDB(temp_path, backbone + result["other_atoms"])
                else:
                    writePDB(temp_path, backbone)

            with open(temp_path) as f:
                pdb_string = f.read()

        finally:
            Path(temp_path).unlink(missing_ok=True)

        return pdb_string

    def _build_residue_mapping(self, protein_dict: dict, icodes: list) -> list:
        """Build list of residue keys matching LigandMPNN format."""
        R_idx_list = protein_dict["R_idx"].cpu().numpy()
        chain_letters = protein_dict["chain_letters"]
        encoded = []
        for i, r_idx in enumerate(R_idx_list):
            key = f"{chain_letters[i]}{int(r_idx)}{icodes[i]}"
            encoded.append(key)
        return encoded

    def _build_chain_mask(
        self,
        encoded_residues: list,
        design_spec: Optional[DesignSpec],
        design_all: bool = False,
    ) -> torch.Tensor:
        """Build chain_mask tensor: 1=design, 0=fixed."""
        if design_all:
            # Design everything
            return torch.ones(len(encoded_residues), device=self.device)

        if design_spec is None:
            # Default: fix all (repack only)
            return torch.zeros(len(encoded_residues), device=self.device)

        mask = []
        for key in encoded_residues:
            if key in design_spec.residue_specs:
                spec = design_spec.residue_specs[key]
                mask.append(1.0 if spec.is_designable() else 0.0)
            else:
                # Use default mode
                default_designable = design_spec.default_mode in (
                    ResidueMode.ALLAA,
                    ResidueMode.PIKAA,
                    ResidueMode.NOTAA,
                    ResidueMode.POLAR,
                    ResidueMode.APOLAR,
                )
                mask.append(1.0 if default_designable else 0.0)

        return torch.tensor(mask, device=self.device, dtype=torch.float32)

    def _build_aa_bias(
        self,
        encoded_residues: list,
        design_spec: Optional[DesignSpec],
    ) -> torch.Tensor:
        """Build amino acid bias tensor for PIKAA/NOTAA/POLAR/APOLAR."""
        L = len(encoded_residues)
        bias = torch.zeros([1, L, 21], device=self.device, dtype=torch.float32)

        if design_spec is None:
            return bias

        for i, key in enumerate(encoded_residues):
            if key in design_spec.residue_specs:
                spec = design_spec.residue_specs[key]
                if spec.is_designable():
                    allowed = spec.get_allowed_aas()
                    if allowed and allowed != ALL_AAS:
                        # Apply large negative bias to excluded amino acids
                        for aa in ALL_AAS:
                            if aa not in allowed:
                                aa_idx = restype_str_to_int.get(aa)
                                if aa_idx is not None:
                                    bias[0, i, aa_idx] = -1e8

        return bias
