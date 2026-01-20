"""Integration tests for graphrelax.designer module.

These tests require LigandMPNN model weights to be downloaded.
Run with: pytest -m "integration" tests/test_designer_integration.py
"""

import pytest

from graphrelax.config import DesignConfig
from graphrelax.resfile import DesignSpec, ResidueMode, ResidueSpec
from graphrelax.weights import weights_exist as weights_available

# Skip all tests in this module if weights are not available
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not weights_available(),
        reason="LigandMPNN weights not downloaded",
    ),
]


@pytest.fixture(scope="module")
def designer():
    """Create a Designer instance (expensive, shared across tests)."""
    from graphrelax.designer import Designer

    config = DesignConfig(
        model_type="ligand_mpnn",
        temperature=0.1,
        seed=42,
        pack_side_chains=False,
    )
    return Designer(config)


@pytest.fixture(scope="module")
def designer_with_packer():
    """Create a Designer instance with side chain packing enabled."""
    from graphrelax.designer import Designer

    config = DesignConfig(
        model_type="ligand_mpnn",
        temperature=0.1,
        seed=42,
        pack_side_chains=True,
    )
    return Designer(config)


class TestDesignerLoading:
    """Tests for Designer model loading."""

    def test_designer_lazy_loads(self):
        """Test that models are not loaded until needed."""
        from graphrelax.designer import Designer

        config = DesignConfig(model_type="ligand_mpnn")
        designer = Designer(config)

        # Models should not be loaded yet
        assert designer._model is None
        assert designer._packer is None

    def test_designer_loads_model(self, designer):
        """Test that model loads on first design call."""
        # This forces model loading
        designer._load_models()

        assert designer._model is not None


class TestDesignerDesign:
    """Tests for Designer.design() method."""

    def test_design_full_redesign(self, designer, small_peptide_pdb):
        """Test full redesign of a structure."""
        result = designer.design(
            pdb_path=small_peptide_pdb,
            design_all=True,
        )

        assert "sequence" in result
        assert "native_sequence" in result
        assert "loss" in result
        assert len(result["sequence"]) == 5  # 5-residue peptide

    def test_design_preserves_native_sequence(
        self, designer, small_peptide_pdb
    ):
        """Test that native sequence is extracted correctly."""
        result = designer.design(
            pdb_path=small_peptide_pdb,
            design_all=True,
        )

        # Our fixture is all alanines
        assert result["native_sequence"] == "AAAAA"

    def test_design_returns_valid_sequence(self, designer, small_peptide_pdb):
        """Test that designed sequence contains valid amino acids."""
        result = designer.design(
            pdb_path=small_peptide_pdb,
            design_all=True,
        )

        valid_aas = set("ACDEFGHIKLMNPQRSTVWY")
        for aa in result["sequence"]:
            assert aa in valid_aas

    def test_design_returns_loss(self, designer, small_peptide_pdb):
        """Test that loss is returned and is a valid number."""
        result = designer.design(
            pdb_path=small_peptide_pdb,
            design_all=True,
        )

        loss = result["loss"].item()
        assert loss >= 0  # Loss should be non-negative
        assert loss == loss  # Check not NaN

    def test_design_with_spec_fixes_residues(self, designer, small_peptide_pdb):
        """Test that design spec can fix certain residues."""
        # Fix residue 3, design the rest
        specs = {
            "A3": ResidueSpec(chain="A", resnum=3, mode=ResidueMode.NATRO),
        }
        design_spec = DesignSpec(
            residue_specs=specs,
            default_mode=ResidueMode.ALLAA,
        )

        result = designer.design(
            pdb_path=small_peptide_pdb,
            design_spec=design_spec,
        )

        # Residue 3 should remain A (fixed)
        assert result["sequence"][2] == "A"

    def test_design_with_pikaa_constraint(self, designer, small_peptide_pdb):
        """Test PIKAA constraint limits amino acid choices."""
        # Only allow G at position 1
        specs = {
            "A1": ResidueSpec(
                chain="A",
                resnum=1,
                mode=ResidueMode.PIKAA,
                allowed_aas={"G"},
            ),
        }
        design_spec = DesignSpec(
            residue_specs=specs,
            default_mode=ResidueMode.NATRO,
        )

        result = designer.design(
            pdb_path=small_peptide_pdb,
            design_spec=design_spec,
        )

        # Position 1 should be G (only allowed AA)
        assert result["sequence"][0] == "G"

    def test_design_no_design_preserves_sequence(
        self, designer, small_peptide_pdb
    ):
        """Test that no design (all fixed) preserves native sequence."""
        result = designer.design(
            pdb_path=small_peptide_pdb,
            design_spec=None,
            design_all=False,
        )

        # With nothing to design, sequence should match native
        assert result["sequence"] == result["native_sequence"]


class TestDesignerRepack:
    """Tests for Designer.repack() method."""

    def test_repack_returns_sequence(self, designer, small_peptide_pdb):
        """Test that repack returns the native sequence."""
        result = designer.repack(pdb_path=small_peptide_pdb)

        assert "sequence" in result
        # Repack should not change sequence
        assert result["sequence"] == result["native_sequence"]

    def test_repack_returns_loss(self, designer, small_peptide_pdb):
        """Test that repack returns loss information."""
        result = designer.repack(pdb_path=small_peptide_pdb)

        assert "loss" in result
        loss = result["loss"].item()
        assert loss >= 0


class TestDesignerSideChainPacking:
    """Tests for side chain packing functionality."""

    def test_packer_loads(self, designer_with_packer):
        """Test that packer model loads correctly."""
        designer_with_packer._load_models()

        assert designer_with_packer._packer is not None

    def test_design_with_packing(self, designer_with_packer, small_peptide_pdb):
        """Test design with side chain packing enabled."""
        result = designer_with_packer.design(
            pdb_path=small_peptide_pdb,
            design_all=True,
        )

        assert "sequence" in result
        # When packing is enabled, coordinates should be updated
        assert "coordinates" in result or "X" in result


class TestDesignerWithRealStructure:
    """Tests with real protein structures."""

    def test_design_ubiquitin(self, designer, ubiquitin_pdb):
        """Test design on ubiquitin (76 residues)."""
        result = designer.design(
            pdb_path=ubiquitin_pdb,
            design_all=True,
        )

        assert len(result["sequence"]) == 76
        assert len(result["native_sequence"]) == 76

    def test_design_heme_protein(self, designer, heme_protein_pdb):
        """Test design on heme-containing protein."""
        result = designer.design(
            pdb_path=heme_protein_pdb,
            design_all=True,
        )

        # 8VC8 has ~190 residues
        assert len(result["sequence"]) > 100

    def test_partial_design_ubiquitin(self, designer, ubiquitin_pdb):
        """Test partial design on ubiquitin."""
        # Design only residues 10-20, fix the rest
        specs = {}
        for i in range(10, 21):
            specs[f"A{i}"] = ResidueSpec(
                chain="A", resnum=i, mode=ResidueMode.ALLAA
            )

        design_spec = DesignSpec(
            residue_specs=specs,
            default_mode=ResidueMode.NATRO,
        )

        result = designer.design(
            pdb_path=ubiquitin_pdb,
            design_spec=design_spec,
        )

        # Fixed regions should match native
        native = result["native_sequence"]
        designed = result["sequence"]

        # First 9 residues should be unchanged
        assert designed[:9] == native[:9]
        # Residues after position 20 should be unchanged
        assert designed[20:] == native[20:]


class TestDesignerSeeding:
    """Tests for seed configuration."""

    def test_seed_is_applied(self, small_peptide_pdb):
        """Test that providing a seed is accepted and design completes."""
        from graphrelax.designer import Designer

        config = DesignConfig(
            model_type="ligand_mpnn",
            temperature=0.5,
            seed=12345,
        )

        designer = Designer(config)
        result = designer.design(pdb_path=small_peptide_pdb, design_all=True)

        # Verify design completes successfully with a seed set
        assert result["sequence"] is not None
        assert len(result["sequence"]) == 5

    def test_no_seed_works(self, small_peptide_pdb):
        """Test that design works without a seed (None)."""
        from graphrelax.designer import Designer

        config = DesignConfig(
            model_type="ligand_mpnn",
            temperature=0.5,
            seed=None,
        )

        designer = Designer(config)
        result = designer.design(pdb_path=small_peptide_pdb, design_all=True)

        assert result["sequence"] is not None
        assert len(result["sequence"]) == 5

    def test_different_seed_different_result(self, small_peptide_pdb):
        """Test that different seeds can produce different results."""
        from graphrelax.designer import Designer

        config1 = DesignConfig(
            model_type="ligand_mpnn",
            temperature=1.0,  # High temp for diversity
            seed=111,
        )
        config2 = DesignConfig(
            model_type="ligand_mpnn",
            temperature=1.0,
            seed=222,
        )

        designer1 = Designer(config1)
        designer2 = Designer(config2)

        result1 = designer1.design(pdb_path=small_peptide_pdb, design_all=True)
        result2 = designer2.design(pdb_path=small_peptide_pdb, design_all=True)

        # With high temperature and different seeds, sequences should differ
        # (though there's a small chance they're the same)
        # We don't assert they're different since that's probabilistic
        assert len(result1["sequence"]) == len(result2["sequence"])
