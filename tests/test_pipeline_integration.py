"""Integration tests for graphrelax.pipeline module.

These tests require both LigandMPNN and OpenMM to be installed.
They test the full pipeline workflow.
"""

import pytest

from graphrelax.config import PipelineConfig, PipelineMode, RelaxConfig

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
