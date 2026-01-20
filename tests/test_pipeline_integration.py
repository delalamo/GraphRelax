"""Integration tests for graphrelax.pipeline module.

These tests require both LigandMPNN and OpenMM to be installed.
They test the full pipeline workflow.
"""

import pytest

from graphrelax.config import (
    DesignConfig,
    PipelineConfig,
    PipelineMode,
    RelaxConfig,
)
from graphrelax.weights import weights_exist as weights_available

# Skip entire module if OpenMM not available
pytest.importorskip("openmm")


@pytest.mark.integration
@pytest.mark.slow
class TestPipelineNoRepackMode:
    """Tests for NO_REPACK mode (minimize only)."""

    def test_no_repack_mode_completes(self, small_peptide_pdb, tmp_path):
        """Test that NO_REPACK mode completes successfully."""
        from graphrelax.pipeline import Pipeline

        config = PipelineConfig(
            mode=PipelineMode.NO_REPACK,
            n_iterations=1,
            n_outputs=1,
            relax=RelaxConfig(max_iterations=50, stiffness=10.0),
        )
        pipeline = Pipeline(config)

        output_pdb = tmp_path / "output.pdb"
        result = pipeline.run(
            input_pdb=small_peptide_pdb,
            output_pdb=output_pdb,
        )

        assert output_pdb.exists()
        assert len(result["outputs"]) == 1

    def test_no_repack_preserves_sequence(self, small_peptide_pdb, tmp_path):
        """Test that NO_REPACK mode doesn't change sequence."""
        from graphrelax.pipeline import Pipeline

        config = PipelineConfig(
            mode=PipelineMode.NO_REPACK,
            n_iterations=1,
            n_outputs=1,
            relax=RelaxConfig(max_iterations=50),
        )
        pipeline = Pipeline(config)

        result = pipeline.run(
            input_pdb=small_peptide_pdb,
            output_pdb=tmp_path / "output.pdb",
        )

        # Read original sequence
        original_content = small_peptide_pdb.read_text()
        # All residues should still be ALA
        assert "ALA" in original_content

        output = result["outputs"][0]
        # Final PDB should still have ALA residues
        assert "ALA" in output["final_pdb"]

    def test_no_repack_returns_energy(self, small_peptide_pdb, tmp_path):
        """Test that final energy is returned."""
        from graphrelax.pipeline import Pipeline

        config = PipelineConfig(
            mode=PipelineMode.NO_REPACK,
            n_iterations=1,
            n_outputs=1,
            relax=RelaxConfig(max_iterations=50),
        )
        pipeline = Pipeline(config)

        result = pipeline.run(
            input_pdb=small_peptide_pdb,
            output_pdb=tmp_path / "output.pdb",
        )

        assert "final_energy" in result["outputs"][0]


@pytest.mark.integration
@pytest.mark.slow
class TestPipelineMultipleOutputs:
    """Tests for generating multiple outputs."""

    def test_multiple_outputs_created(self, small_peptide_pdb, tmp_path):
        """Test that multiple output files are created."""
        from graphrelax.pipeline import Pipeline

        config = PipelineConfig(
            mode=PipelineMode.NO_REPACK,
            n_iterations=1,
            n_outputs=3,
            relax=RelaxConfig(max_iterations=50),
        )
        pipeline = Pipeline(config)

        result = pipeline.run(
            input_pdb=small_peptide_pdb,
            output_pdb=tmp_path / "output.pdb",
        )

        # Should create output_1.pdb, output_2.pdb, output_3.pdb
        assert (tmp_path / "output_1.pdb").exists()
        assert (tmp_path / "output_2.pdb").exists()
        assert (tmp_path / "output_3.pdb").exists()

        assert len(result["outputs"]) == 3

    def test_multiple_outputs_have_paths(self, small_peptide_pdb, tmp_path):
        """Test that each output has its path recorded."""
        from graphrelax.pipeline import Pipeline

        config = PipelineConfig(
            mode=PipelineMode.NO_REPACK,
            n_iterations=1,
            n_outputs=2,
            relax=RelaxConfig(max_iterations=50),
        )
        pipeline = Pipeline(config)

        result = pipeline.run(
            input_pdb=small_peptide_pdb,
            output_pdb=tmp_path / "output.pdb",
        )

        for output in result["outputs"]:
            assert "output_path" in output
            assert output["output_path"].exists()


@pytest.mark.integration
@pytest.mark.slow
class TestPipelineScorefile:
    """Tests for scorefile output."""

    def test_scorefile_created(self, small_peptide_pdb, tmp_path):
        """Test that scorefile is created when specified."""
        from graphrelax.pipeline import Pipeline

        scorefile_path = tmp_path / "scores.sc"
        config = PipelineConfig(
            mode=PipelineMode.NO_REPACK,
            n_iterations=1,
            n_outputs=1,
            scorefile=scorefile_path,
            relax=RelaxConfig(max_iterations=50),
        )
        pipeline = Pipeline(config)

        pipeline.run(
            input_pdb=small_peptide_pdb,
            output_pdb=tmp_path / "output.pdb",
        )

        assert scorefile_path.exists()

    def test_scorefile_format(self, small_peptide_pdb, tmp_path):
        """Test that scorefile has correct format."""
        from graphrelax.pipeline import Pipeline

        scorefile_path = tmp_path / "scores.sc"
        config = PipelineConfig(
            mode=PipelineMode.NO_REPACK,
            n_iterations=1,
            n_outputs=1,
            scorefile=scorefile_path,
            relax=RelaxConfig(max_iterations=50),
        )
        pipeline = Pipeline(config)

        pipeline.run(
            input_pdb=small_peptide_pdb,
            output_pdb=tmp_path / "output.pdb",
        )

        content = scorefile_path.read_text()
        lines = content.strip().split("\n")

        # Should have header line and data line
        assert len(lines) >= 2
        assert lines[0].startswith("SCORE:")
        assert lines[1].startswith("SCORE:")

    def test_scorefile_multiple_entries(self, small_peptide_pdb, tmp_path):
        """Test scorefile with multiple outputs."""
        from graphrelax.pipeline import Pipeline

        scorefile_path = tmp_path / "scores.sc"
        config = PipelineConfig(
            mode=PipelineMode.NO_REPACK,
            n_iterations=1,
            n_outputs=2,
            scorefile=scorefile_path,
            relax=RelaxConfig(max_iterations=50),
        )
        pipeline = Pipeline(config)

        pipeline.run(
            input_pdb=small_peptide_pdb,
            output_pdb=tmp_path / "output.pdb",
        )

        content = scorefile_path.read_text()
        lines = content.strip().split("\n")

        # Header + 2 data lines
        assert len(lines) == 3


@pytest.mark.integration
@pytest.mark.slow
class TestPipelineMultipleIterations:
    """Tests for multiple iteration cycles."""

    def test_multiple_iterations(self, small_peptide_pdb, tmp_path):
        """Test pipeline with multiple iterations."""
        from graphrelax.pipeline import Pipeline

        config = PipelineConfig(
            mode=PipelineMode.NO_REPACK,
            n_iterations=3,
            n_outputs=1,
            relax=RelaxConfig(max_iterations=50),
        )
        pipeline = Pipeline(config)

        result = pipeline.run(
            input_pdb=small_peptide_pdb,
            output_pdb=tmp_path / "output.pdb",
        )

        output = result["outputs"][0]
        assert len(output["iterations"]) == 3

    def test_iterations_have_relax_info(self, small_peptide_pdb, tmp_path):
        """Test that each iteration has relaxation info."""
        from graphrelax.pipeline import Pipeline

        config = PipelineConfig(
            mode=PipelineMode.NO_REPACK,
            n_iterations=2,
            n_outputs=1,
            relax=RelaxConfig(max_iterations=50),
        )
        pipeline = Pipeline(config)

        result = pipeline.run(
            input_pdb=small_peptide_pdb,
            output_pdb=tmp_path / "output.pdb",
        )

        for iteration in result["outputs"][0]["iterations"]:
            assert "relax_info" in iteration
            assert "initial_energy" in iteration["relax_info"]
            assert "final_energy" in iteration["relax_info"]


@pytest.mark.integration
@pytest.mark.slow
class TestPipelineResult:
    """Tests for pipeline result structure."""

    def test_result_structure(self, small_peptide_pdb, tmp_path):
        """Test that result has expected structure."""
        from graphrelax.pipeline import Pipeline

        config = PipelineConfig(
            mode=PipelineMode.NO_REPACK,
            n_iterations=1,
            n_outputs=1,
            relax=RelaxConfig(max_iterations=50),
        )
        pipeline = Pipeline(config)

        result = pipeline.run(
            input_pdb=small_peptide_pdb,
            output_pdb=tmp_path / "output.pdb",
        )

        assert "outputs" in result
        assert "scores" in result
        assert isinstance(result["outputs"], list)
        assert isinstance(result["scores"], list)

    def test_output_structure(self, small_peptide_pdb, tmp_path):
        """Test that each output has expected keys."""
        from graphrelax.pipeline import Pipeline

        config = PipelineConfig(
            mode=PipelineMode.NO_REPACK,
            n_iterations=1,
            n_outputs=1,
            relax=RelaxConfig(max_iterations=50),
        )
        pipeline = Pipeline(config)

        result = pipeline.run(
            input_pdb=small_peptide_pdb,
            output_pdb=tmp_path / "output.pdb",
        )

        output = result["outputs"][0]
        assert "final_pdb" in output
        assert "output_path" in output
        assert "iterations" in output

    def test_scores_structure(self, small_peptide_pdb, tmp_path):
        """Test that scores list has expected structure."""
        from graphrelax.pipeline import Pipeline

        config = PipelineConfig(
            mode=PipelineMode.NO_REPACK,
            n_iterations=1,
            n_outputs=1,
            relax=RelaxConfig(max_iterations=50),
        )
        pipeline = Pipeline(config)

        result = pipeline.run(
            input_pdb=small_peptide_pdb,
            output_pdb=tmp_path / "output.pdb",
        )

        score = result["scores"][0]
        assert "total_score" in score
        assert "description" in score


@pytest.mark.integration
@pytest.mark.slow
class TestPipelineWithUbiquitin:
    """Integration tests using 1UBQ (ubiquitin) as a realistic test case.

    Ubiquitin (PDB: 1UBQ) is a 76-residue protein commonly used as a
    benchmark for protein structure prediction and design methods.
    These tests verify the pipeline works with a real protein structure.
    """

    def test_relax_ubiquitin(self, ubiquitin_pdb, tmp_path):
        """Test relaxation of ubiquitin structure."""
        from graphrelax.pipeline import Pipeline

        config = PipelineConfig(
            mode=PipelineMode.NO_REPACK,
            n_iterations=1,
            n_outputs=1,
            relax=RelaxConfig(max_iterations=100, stiffness=10.0),
        )
        pipeline = Pipeline(config)

        output_pdb = tmp_path / "1ubq_relaxed.pdb"
        result = pipeline.run(
            input_pdb=ubiquitin_pdb,
            output_pdb=output_pdb,
        )

        assert output_pdb.exists()
        assert "final_energy" in result["outputs"][0]
        # Ubiquitin should have reasonable energy
        energy = result["outputs"][0]["final_energy"]
        assert isinstance(energy, (int, float))

    def test_relax_ubiquitin_energy_decreases(self, ubiquitin_pdb, tmp_path):
        """Test that relaxation decreases energy for ubiquitin."""
        from graphrelax.pipeline import Pipeline

        config = PipelineConfig(
            mode=PipelineMode.NO_REPACK,
            n_iterations=1,
            n_outputs=1,
            relax=RelaxConfig(max_iterations=100, stiffness=5.0),
        )
        pipeline = Pipeline(config)

        result = pipeline.run(
            input_pdb=ubiquitin_pdb,
            output_pdb=tmp_path / "output.pdb",
        )

        relax_info = result["outputs"][0]["iterations"][0]["relax_info"]
        initial_energy = relax_info["initial_energy"]
        final_energy = relax_info["final_energy"]

        # Energy should decrease or stay similar after relaxation
        assert final_energy <= initial_energy + 10.0  # Allow small increase

    def test_relax_ubiquitin_rmsd_reasonable(self, ubiquitin_pdb, tmp_path):
        """Test that RMSD after relaxation is reasonable."""
        from graphrelax.pipeline import Pipeline

        config = PipelineConfig(
            mode=PipelineMode.NO_REPACK,
            n_iterations=1,
            n_outputs=1,
            relax=RelaxConfig(max_iterations=100, stiffness=10.0),
        )
        pipeline = Pipeline(config)

        result = pipeline.run(
            input_pdb=ubiquitin_pdb,
            output_pdb=tmp_path / "output.pdb",
        )

        rmsd = result["outputs"][0]["iterations"][0]["relax_info"]["rmsd"]
        # RMSD should be small with restraints
        assert rmsd < 2.0  # Less than 2 Angstrom


@pytest.mark.integration
class TestPipelineWithLigand:
    """
    Integration tests for pipeline with ligand-containing structures.

    These tests use PDB 8VC8, a designed protein with a bound heme group,
    to verify the pipeline handles non-protein molecules correctly.
    """

    def test_relax_heme_protein(self, heme_protein_pdb, tmp_path):
        """Test relaxation of protein with heme ligand."""
        from graphrelax.pipeline import Pipeline

        # Use constrained mode for ligand handling - unconstrained mode
        # doesn't have force field parameters for non-standard residues
        config = PipelineConfig(
            mode=PipelineMode.NO_REPACK,
            n_iterations=1,
            n_outputs=1,
            relax=RelaxConfig(
                max_iterations=100, stiffness=10.0, constrained=True
            ),
        )
        pipeline = Pipeline(config)

        output_pdb = tmp_path / "8vc8_relaxed.pdb"
        result = pipeline.run(
            input_pdb=heme_protein_pdb,
            output_pdb=output_pdb,
        )

        # Check relaxation completed
        assert result is not None
        assert len(result["outputs"]) == 1
        assert output_pdb.exists()

    def test_heme_protein_energy_reasonable(self, heme_protein_pdb, tmp_path):
        """Test that energy values are reasonable for heme-bound protein."""
        from graphrelax.pipeline import Pipeline

        # Use constrained mode for ligand handling
        config = PipelineConfig(
            mode=PipelineMode.NO_REPACK,
            n_iterations=1,
            n_outputs=1,
            relax=RelaxConfig(
                max_iterations=100, stiffness=10.0, constrained=True
            ),
        )
        pipeline = Pipeline(config)

        result = pipeline.run(
            input_pdb=heme_protein_pdb,
            output_pdb=tmp_path / "output.pdb",
        )

        relax_info = result["outputs"][0]["iterations"][0]["relax_info"]
        final_energy = relax_info["final_energy"]

        # Energy should be a reasonable number (not nan or inf)
        assert final_energy is not None
        assert not (final_energy != final_energy)  # Check not NaN
        assert abs(final_energy) < 1e10  # Not unreasonably large

    def test_heme_protein_rmsd_reasonable(self, heme_protein_pdb, tmp_path):
        """Test that RMSD after relaxation is reasonable for heme protein."""
        from graphrelax.pipeline import Pipeline

        # Use constrained mode for ligand handling
        config = PipelineConfig(
            mode=PipelineMode.NO_REPACK,
            n_iterations=1,
            n_outputs=1,
            relax=RelaxConfig(
                max_iterations=100, stiffness=10.0, constrained=True
            ),
        )
        pipeline = Pipeline(config)

        result = pipeline.run(
            input_pdb=heme_protein_pdb,
            output_pdb=tmp_path / "output.pdb",
        )

        rmsd = result["outputs"][0]["iterations"][0]["relax_info"]["rmsd"]
        # RMSD should be small with restraints
        assert rmsd < 2.0  # Less than 2 Angstrom


@pytest.mark.integration
@pytest.mark.skipif(
    not weights_available(),
    reason="LigandMPNN weights not downloaded",
)
class TestPipelineDesignMode:
    """
    Tests for DESIGN mode (full sequence redesign + relaxation).

    These tests require LigandMPNN weights to be available.
    """

    def test_design_mode_completes(self, small_peptide_pdb, tmp_path):
        """Test that DESIGN mode completes successfully."""
        from graphrelax.pipeline import Pipeline

        config = PipelineConfig(
            mode=PipelineMode.DESIGN,
            n_iterations=1,
            n_outputs=1,
            design=DesignConfig(
                model_type="ligand_mpnn",
                temperature=0.1,
                seed=42,
            ),
            relax=RelaxConfig(max_iterations=50, stiffness=10.0),
        )
        pipeline = Pipeline(config)

        output_pdb = tmp_path / "designed.pdb"
        result = pipeline.run(
            input_pdb=small_peptide_pdb,
            output_pdb=output_pdb,
        )

        assert result is not None
        assert len(result["outputs"]) == 1
        assert output_pdb.exists()

    def test_design_mode_changes_sequence(self, small_peptide_pdb, tmp_path):
        """Test that DESIGN mode can change the sequence."""
        from graphrelax.pipeline import Pipeline

        config = PipelineConfig(
            mode=PipelineMode.DESIGN,
            n_iterations=1,
            n_outputs=1,
            design=DesignConfig(
                model_type="ligand_mpnn",
                temperature=0.5,  # Higher temp for more diversity
                seed=42,
            ),
            relax=RelaxConfig(max_iterations=50, stiffness=10.0),
        )
        pipeline = Pipeline(config)

        result = pipeline.run(
            input_pdb=small_peptide_pdb,
            output_pdb=tmp_path / "designed.pdb",
        )

        # Check that sequence info is returned
        output = result["outputs"][0]
        assert "sequence" in output
        assert "native_sequence" in output

        # Native sequence is all alanines
        assert output["native_sequence"] == "AAAAA"

    def test_design_mode_returns_scores(self, small_peptide_pdb, tmp_path):
        """Test that DESIGN mode returns design and energy scores."""
        from graphrelax.pipeline import Pipeline

        config = PipelineConfig(
            mode=PipelineMode.DESIGN,
            n_iterations=1,
            n_outputs=1,
            design=DesignConfig(model_type="ligand_mpnn", seed=42),
            relax=RelaxConfig(max_iterations=50, stiffness=10.0),
        )
        pipeline = Pipeline(config)

        result = pipeline.run(
            input_pdb=small_peptide_pdb,
            output_pdb=tmp_path / "designed.pdb",
        )

        scores = result["scores"][0]
        assert "total_score" in scores
        assert "openmm_energy" in scores

    def test_design_with_resfile(self, small_peptide_pdb, tmp_path):
        """Test DESIGN mode with resfile constraints."""
        from graphrelax.pipeline import Pipeline

        # Create a resfile that fixes position 3
        resfile_path = tmp_path / "design.resfile"
        resfile_path.write_text(
            """NATAA
START
1 A ALLAA
2 A ALLAA
3 A NATRO
4 A ALLAA
5 A ALLAA
"""
        )

        config = PipelineConfig(
            mode=PipelineMode.DESIGN,
            n_iterations=1,
            n_outputs=1,
            design=DesignConfig(model_type="ligand_mpnn", seed=42),
            relax=RelaxConfig(max_iterations=50, stiffness=10.0),
        )
        pipeline = Pipeline(config)

        result = pipeline.run(
            input_pdb=small_peptide_pdb,
            output_pdb=tmp_path / "designed.pdb",
            resfile=resfile_path,
        )

        output = result["outputs"][0]
        # Position 3 should remain A (fixed by NATRO)
        assert output["sequence"][2] == "A"


@pytest.mark.integration
@pytest.mark.skipif(
    not weights_available(),
    reason="LigandMPNN weights not downloaded",
)
class TestPipelineRepackMode:
    """
    Tests for RELAX mode (repack + relaxation cycles).

    These tests require LigandMPNN weights to be available.
    """

    def test_relax_mode_completes(self, small_peptide_pdb, tmp_path):
        """Test that RELAX mode (repack+minimize) completes."""
        from graphrelax.pipeline import Pipeline

        config = PipelineConfig(
            mode=PipelineMode.RELAX,
            n_iterations=2,
            n_outputs=1,
            design=DesignConfig(model_type="ligand_mpnn", seed=42),
            relax=RelaxConfig(max_iterations=50, stiffness=10.0),
        )
        pipeline = Pipeline(config)

        output_pdb = tmp_path / "relaxed.pdb"
        result = pipeline.run(
            input_pdb=small_peptide_pdb,
            output_pdb=output_pdb,
        )

        assert result is not None
        assert output_pdb.exists()

    def test_relax_mode_preserves_sequence(self, small_peptide_pdb, tmp_path):
        """Test that RELAX mode preserves the native sequence."""
        from graphrelax.pipeline import Pipeline

        config = PipelineConfig(
            mode=PipelineMode.RELAX,
            n_iterations=1,
            n_outputs=1,
            design=DesignConfig(model_type="ligand_mpnn", seed=42),
            relax=RelaxConfig(max_iterations=50, stiffness=10.0),
        )
        pipeline = Pipeline(config)

        result = pipeline.run(
            input_pdb=small_peptide_pdb,
            output_pdb=tmp_path / "relaxed.pdb",
        )

        output = result["outputs"][0]
        # Repack should not change sequence
        assert output["sequence"] == output["native_sequence"]

    def test_relax_mode_multiple_iterations(self, small_peptide_pdb, tmp_path):
        """Test RELAX mode with multiple iterations."""
        from graphrelax.pipeline import Pipeline

        config = PipelineConfig(
            mode=PipelineMode.RELAX,
            n_iterations=3,
            n_outputs=1,
            design=DesignConfig(model_type="ligand_mpnn", seed=42),
            relax=RelaxConfig(max_iterations=50, stiffness=10.0),
        )
        pipeline = Pipeline(config)

        result = pipeline.run(
            input_pdb=small_peptide_pdb,
            output_pdb=tmp_path / "relaxed.pdb",
        )

        output = result["outputs"][0]
        assert len(output["iterations"]) == 3


@pytest.mark.integration
@pytest.mark.skipif(
    not weights_available(),
    reason="LigandMPNN weights not downloaded",
)
class TestPipelineDesignOnlyMode:
    """Tests for DESIGN_ONLY mode (design without relaxation)."""

    def test_design_only_mode_completes(self, small_peptide_pdb, tmp_path):
        """Test that DESIGN_ONLY mode completes without relaxation."""
        from graphrelax.pipeline import Pipeline

        config = PipelineConfig(
            mode=PipelineMode.DESIGN_ONLY,
            n_iterations=1,
            n_outputs=1,
            design=DesignConfig(model_type="ligand_mpnn", seed=42),
        )
        pipeline = Pipeline(config)

        output_pdb = tmp_path / "designed.pdb"
        result = pipeline.run(
            input_pdb=small_peptide_pdb,
            output_pdb=output_pdb,
        )

        assert result is not None
        assert output_pdb.exists()

    def test_design_only_no_energy(self, small_peptide_pdb, tmp_path):
        """Test that DESIGN_ONLY mode doesn't compute relaxation energy."""
        from graphrelax.pipeline import Pipeline

        config = PipelineConfig(
            mode=PipelineMode.DESIGN_ONLY,
            n_iterations=1,
            n_outputs=1,
            design=DesignConfig(model_type="ligand_mpnn", seed=42),
        )
        pipeline = Pipeline(config)

        result = pipeline.run(
            input_pdb=small_peptide_pdb,
            output_pdb=tmp_path / "designed.pdb",
        )

        output = result["outputs"][0]
        iterations = output["iterations"]
        # DESIGN_ONLY should not have relax_info
        assert (
            "relax_info" not in iterations[0]
            or iterations[0].get("relax_info") is None
        )


@pytest.mark.integration
@pytest.mark.skipif(
    not weights_available(),
    reason="LigandMPNN weights not downloaded",
)
class TestPipelineFullWorkflow:
    """End-to-end tests for the full design+relax workflow."""

    def test_full_workflow_ubiquitin(self, ubiquitin_pdb, tmp_path):
        """Test full design+relax workflow on ubiquitin."""
        from graphrelax.pipeline import Pipeline

        config = PipelineConfig(
            mode=PipelineMode.DESIGN,
            n_iterations=1,
            n_outputs=1,
            design=DesignConfig(
                model_type="ligand_mpnn",
                temperature=0.1,
                seed=42,
            ),
            relax=RelaxConfig(max_iterations=100, stiffness=10.0),
        )
        pipeline = Pipeline(config)

        output_pdb = tmp_path / "1ubq_designed.pdb"
        result = pipeline.run(
            input_pdb=ubiquitin_pdb,
            output_pdb=output_pdb,
        )

        assert output_pdb.exists()
        output = result["outputs"][0]

        # Should have designed sequence
        assert "sequence" in output
        assert len(output["sequence"]) == 76

        # Should have energy information
        assert "final_energy" in output

    def test_multiple_designs(self, small_peptide_pdb, tmp_path):
        """Test generating multiple designs."""
        from graphrelax.pipeline import Pipeline

        config = PipelineConfig(
            mode=PipelineMode.DESIGN,
            n_iterations=1,
            n_outputs=3,
            design=DesignConfig(
                model_type="ligand_mpnn",
                temperature=0.5,  # Higher temp for diversity
                seed=42,
            ),
            relax=RelaxConfig(max_iterations=50, stiffness=10.0),
        )
        pipeline = Pipeline(config)

        result = pipeline.run(
            input_pdb=small_peptide_pdb,
            output_pdb=tmp_path / "designed.pdb",
        )

        assert len(result["outputs"]) == 3

        # All output files should exist
        assert (tmp_path / "designed_1.pdb").exists()
        assert (tmp_path / "designed_2.pdb").exists()
        assert (tmp_path / "designed_3.pdb").exists()

    def test_scorefile_with_design(self, small_peptide_pdb, tmp_path):
        """Test scorefile generation with design mode."""
        from graphrelax.pipeline import Pipeline

        scorefile = tmp_path / "scores.sc"
        config = PipelineConfig(
            mode=PipelineMode.DESIGN,
            n_iterations=1,
            n_outputs=2,
            scorefile=scorefile,
            design=DesignConfig(model_type="ligand_mpnn", seed=42),
            relax=RelaxConfig(max_iterations=50, stiffness=10.0),
        )
        pipeline = Pipeline(config)

        pipeline.run(
            input_pdb=small_peptide_pdb,
            output_pdb=tmp_path / "designed.pdb",
        )

        assert scorefile.exists()
        content = scorefile.read_text()
        # Should have header and 2 data lines
        lines = [ln for ln in content.strip().split("\n") if ln]
        assert len(lines) >= 3  # header + 2 entries


@pytest.mark.integration
@pytest.mark.slow
class TestCIFFormatSupport:
    """Tests for CIF file format support."""

    def test_cif_input_pdb_output(self, small_peptide_cif, tmp_path):
        """Test CIF input with PDB output."""
        from graphrelax.pipeline import Pipeline

        config = PipelineConfig(
            mode=PipelineMode.NO_REPACK,
            n_iterations=1,
            n_outputs=1,
            relax=RelaxConfig(max_iterations=50),
        )
        pipeline = Pipeline(config)

        output_pdb = tmp_path / "output.pdb"
        pipeline.run(
            input_pdb=small_peptide_cif,
            output_pdb=output_pdb,
        )

        assert output_pdb.exists()
        content = output_pdb.read_text()
        # Should be PDB format
        assert "ATOM" in content
        assert not content.startswith("data_")

    def test_cif_input_cif_output(self, small_peptide_cif, tmp_path):
        """Test CIF input with CIF output (format preserved)."""
        from graphrelax.pipeline import Pipeline

        config = PipelineConfig(
            mode=PipelineMode.NO_REPACK,
            n_iterations=1,
            n_outputs=1,
            relax=RelaxConfig(max_iterations=50),
        )
        pipeline = Pipeline(config)

        output_cif = tmp_path / "output.cif"
        pipeline.run(
            input_pdb=small_peptide_cif,
            output_pdb=output_cif,
        )

        assert output_cif.exists()
        content = output_cif.read_text()
        # Should be CIF format
        assert content.startswith("data_")
        assert "_atom_site" in content

    def test_pdb_input_cif_output(self, small_peptide_pdb, tmp_path):
        """Test PDB input with CIF output (format conversion)."""
        from graphrelax.pipeline import Pipeline

        config = PipelineConfig(
            mode=PipelineMode.NO_REPACK,
            n_iterations=1,
            n_outputs=1,
            relax=RelaxConfig(max_iterations=50),
        )
        pipeline = Pipeline(config)

        output_cif = tmp_path / "output.cif"
        pipeline.run(
            input_pdb=small_peptide_pdb,
            output_pdb=output_cif,
        )

        assert output_cif.exists()
        content = output_cif.read_text()
        # Should be CIF format
        assert content.startswith("data_")

    def test_cif_multiple_outputs(self, small_peptide_cif, tmp_path):
        """Test CIF input with multiple outputs."""
        from graphrelax.pipeline import Pipeline

        config = PipelineConfig(
            mode=PipelineMode.NO_REPACK,
            n_iterations=1,
            n_outputs=2,
            relax=RelaxConfig(max_iterations=50),
        )
        pipeline = Pipeline(config)

        output_cif = tmp_path / "output.cif"
        result = pipeline.run(
            input_pdb=small_peptide_cif,
            output_pdb=output_cif,
        )

        # Should create output_1.cif and output_2.cif
        assert (tmp_path / "output_1.cif").exists()
        assert (tmp_path / "output_2.cif").exists()
        assert len(result["outputs"]) == 2
