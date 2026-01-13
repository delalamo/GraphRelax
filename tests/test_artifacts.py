"""Tests for crystallography artifact detection and removal."""

from graphrelax.artifacts import (
    BIOLOGICALLY_RELEVANT_IONS,
    BUFFER_ARTIFACTS,
    CRYOPROTECTANT_ARTIFACTS,
    CRYSTALLOGRAPHY_ARTIFACTS,
    DETERGENT_ARTIFACTS,
    HALIDE_ARTIFACTS,
    LIPID_ARTIFACTS,
    REDUCING_AGENT_ARTIFACTS,
    WATER_RESIDUES,
    is_artifact,
    is_biologically_relevant_ion,
    remove_artifacts,
)


class TestArtifactConstants:
    """Tests for artifact constant definitions."""

    def test_common_buffers_included(self):
        """Test that common buffer artifacts are in the set."""
        buffers = ["SO4", "PO4", "CIT", "ACT", "MES", "EPE", "TRS"]
        for buf in buffers:
            assert (
                buf in BUFFER_ARTIFACTS
            ), f"{buf} should be in BUFFER_ARTIFACTS"

    def test_common_cryoprotectants_included(self):
        """Test that common cryoprotectants are in the set."""
        cryo = ["GOL", "EDO", "MPD", "PEG", "DMS"]
        for c in cryo:
            assert (
                c in CRYOPROTECTANT_ARTIFACTS
            ), f"{c} should be in CRYOPROTECTANT_ARTIFACTS"

    def test_common_detergents_included(self):
        """Test that common detergents are in the set."""
        detergents = ["BOG", "LDA", "SDS"]
        for d in detergents:
            assert (
                d in DETERGENT_ARTIFACTS
            ), f"{d} should be in DETERGENT_ARTIFACTS"

    def test_common_lipids_included(self):
        """Test that common lipids/fatty acids are in the set."""
        lipids = ["PLM", "MYR", "OLA", "STE"]
        for lip in lipids:
            assert lip in LIPID_ARTIFACTS, f"{lip} should be in LIPID_ARTIFACTS"

    def test_reducing_agents_included(self):
        """Test that common reducing agents are in the set."""
        agents = ["BME", "DTT"]
        for a in agents:
            assert (
                a in REDUCING_AGENT_ARTIFACTS
            ), f"{a} should be in REDUCING_AGENT_ARTIFACTS"

    def test_halides_included(self):
        """Test that halide ions are in the set."""
        halides = ["CL", "BR", "IOD", "F"]
        for h in halides:
            assert h in HALIDE_ARTIFACTS, f"{h} should be in HALIDE_ARTIFACTS"

    def test_metal_ions_not_in_artifacts(self):
        """Test that biologically relevant metal ions are NOT artifacts."""
        metals = ["ZN", "MG", "CA", "FE", "MN", "CU"]
        for m in metals:
            assert (
                m not in CRYSTALLOGRAPHY_ARTIFACTS
            ), f"{m} should NOT be in CRYSTALLOGRAPHY_ARTIFACTS"

    def test_biologically_relevant_ions_defined(self):
        """Test that biologically relevant ions are in their own set."""
        ions = ["ZN", "MG", "CA", "FE", "MN", "CU", "CO", "NI"]
        for ion in ions:
            assert (
                ion in BIOLOGICALLY_RELEVANT_IONS
            ), f"{ion} should be in BIOLOGICALLY_RELEVANT_IONS"

    def test_master_set_contains_all_categories(self):
        """Test that CRYSTALLOGRAPHY_ARTIFACTS combines all category sets."""
        all_categories = (
            BUFFER_ARTIFACTS
            | CRYOPROTECTANT_ARTIFACTS
            | DETERGENT_ARTIFACTS
            | LIPID_ARTIFACTS
            | REDUCING_AGENT_ARTIFACTS
            | HALIDE_ARTIFACTS
        )
        for artifact in all_categories:
            assert (
                artifact in CRYSTALLOGRAPHY_ARTIFACTS
            ), f"{artifact} should be in CRYSTALLOGRAPHY_ARTIFACTS"

    def test_water_residues_separate(self):
        """Test that water residues are in their own set."""
        waters = ["HOH", "WAT", "SOL"]
        for w in waters:
            assert w in WATER_RESIDUES, f"{w} should be in WATER_RESIDUES"
            # Waters should NOT be in crystallography artifacts
            assert (
                w not in CRYSTALLOGRAPHY_ARTIFACTS
            ), f"{w} should NOT be in CRYSTALLOGRAPHY_ARTIFACTS"


class TestIsArtifact:
    """Tests for is_artifact function."""

    def test_glycerol_is_artifact(self):
        """Test that glycerol is detected as artifact."""
        assert is_artifact("GOL")
        assert is_artifact("gol")  # case insensitive

    def test_sulfate_is_artifact(self):
        """Test that sulfate is detected as artifact."""
        assert is_artifact("SO4")

    def test_zinc_is_not_artifact(self):
        """Test that zinc is not detected as artifact."""
        assert not is_artifact("ZN")

    def test_heme_is_not_artifact(self):
        """Test that heme is not detected as artifact."""
        assert not is_artifact("HEM")


class TestIsBiologicallyRelevantIon:
    """Tests for is_biologically_relevant_ion function."""

    def test_zinc_is_relevant(self):
        """Test that zinc is detected as relevant."""
        assert is_biologically_relevant_ion("ZN")
        assert is_biologically_relevant_ion("zn")  # case insensitive

    def test_magnesium_is_relevant(self):
        """Test that magnesium is detected as relevant."""
        assert is_biologically_relevant_ion("MG")

    def test_glycerol_is_not_relevant_ion(self):
        """Test that glycerol is not a relevant ion."""
        assert not is_biologically_relevant_ion("GOL")


class TestRemoveArtifacts:
    """Tests for remove_artifacts function."""

    def test_removes_glycerol(self):
        """Test that glycerol atoms are removed."""
        # fmt: off
        pdb = (
            "ATOM      1  N   ALA A   1       0.0   0.0   0.0  1.00  0.00\n"
            "ATOM      2  CA  ALA A   1       1.5   0.0   0.0  1.00  0.00\n"
            "HETATM    3  O1  GOL A 100       5.0   5.0   5.0  1.00  0.00\n"
            "HETATM    4  O2  GOL A 100       6.0   5.0   5.0  1.00  0.00\n"
            "END\n"
        )
        # fmt: on
        result, removed = remove_artifacts(pdb)

        assert "GOL" not in result
        assert "GOL" in removed
        assert removed["GOL"] == 2
        assert "ATOM" in result  # Protein preserved

    def test_removes_sulfate(self):
        """Test that sulfate atoms are removed."""
        # fmt: off
        pdb = (
            "ATOM      1  N   ALA A   1       0.0   0.0   0.0  1.00  0.00\n"
            "HETATM    2  S   SO4 A 200       5.0   5.0   5.0  1.00  0.00\n"
            "HETATM    3  O1  SO4 A 200       6.0   5.0   5.0  1.00  0.00\n"
            "HETATM    4  O2  SO4 A 200       7.0   5.0   5.0  1.00  0.00\n"
            "HETATM    5  O3  SO4 A 200       8.0   5.0   5.0  1.00  0.00\n"
            "HETATM    6  O4  SO4 A 200       9.0   5.0   5.0  1.00  0.00\n"
            "END\n"
        )
        # fmt: on
        result, removed = remove_artifacts(pdb)

        assert "SO4" not in result
        assert removed["SO4"] == 5

    def test_keeps_zinc(self):
        """Test that zinc ions are preserved."""
        # fmt: off
        pdb = (
            "ATOM      1  N   ALA A   1       0.0   0.0   0.0  1.00  0.00\n"
            "HETATM    2 ZN   ZN  A 200       5.0   5.0   5.0  1.00  0.00\n"
            "END\n"
        )
        # fmt: on
        result, removed = remove_artifacts(pdb)

        assert "ZN" in result
        assert "ZN" not in removed

    def test_keeps_heme(self):
        """Test that heme is preserved (not an artifact)."""
        # fmt: off
        pdb = (
            "ATOM      1  N   ALA A   1       0.0   0.0   0.0  1.00  0.00\n"
            "HETATM    2 FE   HEM A 200       5.0   5.0   5.0  1.00  0.00\n"
            "HETATM    3  NA  HEM A 200       6.0   5.0   5.0  1.00  0.00\n"
            "END\n"
        )
        # fmt: on
        result, removed = remove_artifacts(pdb)

        assert "HEM" in result
        assert "HEM" not in removed

    def test_whitelist_preserves_artifact(self):
        """Test that whitelisted artifacts are preserved."""
        # fmt: off
        pdb = (
            "ATOM      1  N   ALA A   1       0.0   0.0   0.0  1.00  0.00\n"
            "HETATM    2  O1  GOL A 100       5.0   5.0   5.0  1.00  0.00\n"
            "END\n"
        )
        # fmt: on
        result, removed = remove_artifacts(pdb, keep_residues={"GOL"})

        assert "GOL" in result
        assert "GOL" not in removed

    def test_whitelist_case_insensitive(self):
        """Test that whitelist is case insensitive."""
        # fmt: off
        pdb = (
            "ATOM      1  N   ALA A   1       0.0   0.0   0.0  1.00  0.00\n"
            "HETATM    2  O1  GOL A 100       5.0   5.0   5.0  1.00  0.00\n"
            "END\n"
        )
        # fmt: on
        result, removed = remove_artifacts(pdb, keep_residues={"gol"})

        assert "GOL" in result

    def test_multiple_artifact_types(self):
        """Test removal of multiple artifact types."""
        # fmt: off
        pdb = (
            "ATOM      1  N   ALA A   1       0.0   0.0   0.0  1.00  0.00\n"
            "HETATM    2  O1  GOL A 100       5.0   5.0   5.0  1.00  0.00\n"
            "HETATM    3  S   SO4 A 200       6.0   6.0   6.0  1.00  0.00\n"
            "HETATM    4  CL  CL  A 300       7.0   7.0   7.0  1.00  0.00\n"
            "END\n"
        )
        # fmt: on
        result, removed = remove_artifacts(pdb)

        assert "GOL" not in result
        assert "SO4" not in result
        assert "CL" not in result
        assert removed["GOL"] == 1
        assert removed["SO4"] == 1
        assert removed["CL"] == 1

    def test_empty_pdb(self):
        """Test handling of empty PDB."""
        pdb = ""
        result, removed = remove_artifacts(pdb)
        assert result == ""
        assert len(removed) == 0

    def test_protein_only_pdb(self):
        """Test handling of PDB with no artifacts."""
        # fmt: off
        pdb = (
            "ATOM      1  N   ALA A   1       0.0   0.0   0.0  1.00  0.00\n"
            "ATOM      2  CA  ALA A   1       1.5   0.0   0.0  1.00  0.00\n"
            "END\n"
        )
        # fmt: on
        result, removed = remove_artifacts(pdb)

        assert result == pdb
        assert len(removed) == 0


class TestRemoveArtifactsDetergentsAndLipids:
    """Tests specifically for detergent and lipid removal."""

    def test_removes_palmitic_acid(self):
        """Test that palmitic acid is removed."""
        # fmt: off
        pdb = (
            "ATOM      1  N   ALA A   1       0.0   0.0   0.0  1.00  0.00\n"
            "HETATM    2  C1  PLM A 100       5.0   5.0   5.0  1.00  0.00\n"
            "END\n"
        )
        # fmt: on
        result, removed = remove_artifacts(pdb)

        assert "PLM" not in result
        assert removed["PLM"] == 1

    def test_removes_octyl_glucoside(self):
        """Test that octyl glucoside detergent is removed."""
        # fmt: off
        pdb = (
            "ATOM      1  N   ALA A   1       0.0   0.0   0.0  1.00  0.00\n"
            "HETATM    2  C1  BOG A 100       5.0   5.0   5.0  1.00  0.00\n"
            "END\n"
        )
        # fmt: on
        result, removed = remove_artifacts(pdb)

        assert "BOG" not in result
        assert removed["BOG"] == 1

    def test_removes_ldao(self):
        """Test that LDAO detergent is removed."""
        # fmt: off
        pdb = (
            "ATOM      1  N   ALA A   1       0.0   0.0   0.0  1.00  0.00\n"
            "HETATM    2  N   LDA A 100       5.0   5.0   5.0  1.00  0.00\n"
            "END\n"
        )
        # fmt: on
        result, removed = remove_artifacts(pdb)

        assert "LDA" not in result
        assert removed["LDA"] == 1
