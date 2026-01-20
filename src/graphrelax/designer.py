"""LigandMPNN wrapper for sequence design and side-chain packing."""

import logging
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from graphrelax.config import DesignConfig
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
    restype_int_to_str,
    restype_str_to_int,
    write_full_PDB,
)
from model_utils import ProteinMPNN  # noqa: E402
from sc_utils import Packer, pack_side_chains  # noqa: E402


class Designer:
    """Wrapper for LigandMPNN design and side-chain packing."""

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
        self._packer: Optional[Packer] = None
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

        if self.config.pack_side_chains:
            self._load_packer()

    def _load_packer(self):
        """Load side chain packer model."""
        weights_dir = find_weights_dir()
        sc_checkpoint_path = weights_dir / "ligandmpnn_sc_v_32_002_16.pt"
        logger.info(f"Loading packer from {sc_checkpoint_path}")

        checkpoint_sc = torch.load(sc_checkpoint_path, map_location=self.device)

        self._packer = Packer(
            node_features=128,
            edge_features=128,
            num_positional_embeddings=16,
            num_chain_embeddings=16,
            num_rbf=16,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            atom_context_num=16,
            lower_bound=0.0,
            upper_bound=20.0,
            top_k=32,
            dropout=0.0,
            augment_eps=0.0,
            atom37_order=False,
            device=self.device,
            num_mix=3,
        )
        self._packer.load_state_dict(checkpoint_sc["model_state_dict"])
        self._packer.to(self.device)
        self._packer.eval()

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

            # Pack side chains if enabled
            if self.config.pack_side_chains:
                sc_feature_dict = featurize(
                    protein_dict,
                    cutoff_for_score=8.0,
                    use_atom_context=True,
                    number_of_ligand_atoms=16,
                    model_type="ligand_mpnn",
                )
                sc_feature_dict["S"] = output_dict["S"]

                sc_dict = pack_side_chains(
                    sc_feature_dict,
                    self._packer,
                    self.config.sc_num_denoising_steps,
                    self.config.sc_num_samples,
                    repack_everything=False,
                )
                output_dict.update(sc_dict)

        # Extract sequence
        seq_ints = output_dict["S"][0].cpu().numpy()
        sequence = "".join([restype_int_to_str[aa] for aa in seq_ints])

        # Get native sequence for recovery calculation
        native_seq_ints = feature_dict["S"][0].cpu().numpy()
        native_sequence = "".join(
            [restype_int_to_str[aa] for aa in native_seq_ints]
        )

        return {
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
        self._load_models()

        # Ensure packer is loaded (repack always needs it)
        if self._packer is None:
            self._load_packer()

        # Parse PDB
        protein_dict, backbone, other_atoms, icodes, _ = parse_PDB(
            str(pdb_path),
            device=self.device,
            parse_all_atoms=True,
        )

        # Build residue mapping
        encoded_residues = self._build_residue_mapping(protein_dict, icodes)

        # For repacking, chain_mask = 0 (don't change sequence)
        chain_mask = torch.zeros(len(encoded_residues), device=self.device)
        protein_dict["chain_mask"] = chain_mask

        # Featurize for packer
        feature_dict = featurize(
            protein_dict,
            cutoff_for_score=8.0,
            use_atom_context=True,
            number_of_ligand_atoms=16,
            model_type="ligand_mpnn",
        )
        # Convert S to int64 (required by one_hot in pack_side_chains)
        feature_dict["S"] = protein_dict["S"].unsqueeze(0).long()

        # Pack side chains
        with torch.no_grad():
            sc_dict = pack_side_chains(
                feature_dict,
                self._packer,
                self.config.sc_num_denoising_steps,
                self.config.sc_num_samples,
                repack_everything=True,
            )

        # Get sequence
        seq_ints = protein_dict["S"].cpu().numpy()
        sequence = "".join([restype_int_to_str[aa] for aa in seq_ints])

        # Compute score (loss) for the repacked structure
        # For repack, we compute the score of the native sequence
        with torch.no_grad():
            # Get log probs from the model
            B = 1
            L = feature_dict["mask"].shape[1]
            feature_dict["batch_size"] = B
            feature_dict["temperature"] = self.config.temperature
            feature_dict["bias"] = torch.zeros(
                [1, L, 21], device=self.device, dtype=torch.float32
            )
            feature_dict["symmetry_residues"] = [[]]
            feature_dict["symmetry_weights"] = [[]]
            feature_dict["randn"] = torch.randn([B, L], device=self.device)

            output_dict = self._model.sample(feature_dict)
            loss, loss_per_residue = get_score(
                feature_dict["S"],
                output_dict["log_probs"],
                feature_dict["mask"],
            )

        return {
            "sequence": sequence,
            "native_sequence": sequence,
            "S": protein_dict["S"].unsqueeze(0),
            "loss": loss.cpu().numpy(),
            "loss_per_residue": loss_per_residue.cpu().numpy(),
            "log_probs": output_dict["log_probs"],
            "coordinates": sc_dict["X"],
            "coord_mask": sc_dict["X_m"],
            "b_factors": sc_dict["b_factors"],
            "backbone": backbone,
            "other_atoms": other_atoms,
            "icodes": icodes,
            "protein_dict": protein_dict,
            "feature_dict": feature_dict,
            "encoded_residues": encoded_residues,
        }

    def result_to_pdb_string(self, result: dict) -> str:
        """Convert design/repack result to PDB string."""
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
                from data_utils import restype_1to3
                from prody import writePDB

                seq = result["sequence"]
                seq_prody = np.array([restype_1to3[aa] for aa in seq])[
                    None, :
                ].repeat(4, 0)
                result["backbone"].setResnames(seq_prody)
                if result["other_atoms"]:
                    writePDB(
                        temp_path, result["backbone"] + result["other_atoms"]
                    )
                else:
                    writePDB(temp_path, result["backbone"])

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
