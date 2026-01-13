"""Integration tests for graphrelax.relaxer module.

These tests require OpenMM to be installed and will be skipped if not available.
"""

import pytest

from graphrelax.config import RelaxConfig

# Skip entire module if OpenMM not available
pytest.importorskip("openmm")


@pytest.fixture
def relaxer():
    """Create a Relaxer instance for testing."""
    from graphrelax.relaxer import Relaxer

    config = RelaxConfig(max_iterations=50, stiffness=10.0)
    return Relaxer(config)


@pytest.fixture
def unconstrained_relaxer():
    """Create a Relaxer instance with unconstrained minimization."""
    from graphrelax.relaxer import Relaxer

    config = RelaxConfig(max_iterations=50, stiffness=10.0, constrained=False)
    return Relaxer(config)


@pytest.mark.integration
class TestRelaxerGPUDetection:
    """Tests for GPU detection logic."""

    def test_check_gpu_available_returns_bool(self):
        """Test that GPU check returns a boolean."""
        from graphrelax.utils import check_gpu_available

        result = check_gpu_available()
        assert isinstance(result, bool)

    def test_check_gpu_available_cached(self):
        """Test that GPU check result is cached."""
        from graphrelax.utils import check_gpu_available

        result1 = check_gpu_available()
        result2 = check_gpu_available()
        assert result1 == result2


@pytest.mark.integration
class TestRelaxUnconstrained:
    """Tests for unconstrained OpenMM minimization."""

    def test_relax_unconstrained_runs(
        self, unconstrained_relaxer, small_peptide_pdb_string
    ):
        """Test that unconstrained relaxation completes without error."""
        relaxed_pdb, debug_info, violations = unconstrained_relaxer.relax(
            small_peptide_pdb_string
        )

        assert relaxed_pdb is not None
        assert isinstance(relaxed_pdb, str)
        assert len(relaxed_pdb) > 0

    def test_relax_unconstrained_returns_debug_info(
        self, unconstrained_relaxer, small_peptide_pdb_string
    ):
        """Test that debug info contains expected keys."""
        _, debug_info, _ = unconstrained_relaxer.relax(small_peptide_pdb_string)

        assert "initial_energy" in debug_info
        assert "final_energy" in debug_info
        assert "rmsd" in debug_info

    def test_relax_unconstrained_energy_types(
        self, unconstrained_relaxer, small_peptide_pdb_string
    ):
        """Test that energy values are numeric."""
        _, debug_info, _ = unconstrained_relaxer.relax(small_peptide_pdb_string)

        assert isinstance(debug_info["initial_energy"], (int, float))
        assert isinstance(debug_info["final_energy"], (int, float))
        assert isinstance(debug_info["rmsd"], (int, float))

    def test_relax_unconstrained_pdb_format(
        self, unconstrained_relaxer, small_peptide_pdb_string
    ):
        """Test that output is valid PDB format."""
        relaxed_pdb, _, _ = unconstrained_relaxer.relax(
            small_peptide_pdb_string
        )

        # Should contain ATOM records
        assert "ATOM" in relaxed_pdb or "HETATM" in relaxed_pdb

    def test_relax_unconstrained_violations_array(
        self, unconstrained_relaxer, small_peptide_pdb_string
    ):
        """Test that violations is a numpy array."""
        import numpy as np

        _, _, violations = unconstrained_relaxer.relax(small_peptide_pdb_string)

        assert isinstance(violations, np.ndarray)


@pytest.mark.integration
class TestRelaxMethod:
    """Tests for main relax() method."""

    def test_relax_from_pdb_string(self, relaxer, small_peptide_pdb_string):
        """Test relaxing from PDB string."""
        relaxed_pdb, debug_info, violations = relaxer.relax(
            small_peptide_pdb_string
        )

        assert relaxed_pdb is not None
        assert "final_energy" in debug_info

    def test_relax_pdb_file(self, relaxer, small_peptide_pdb):
        """Test relaxing from PDB file path."""
        relaxed_pdb, debug_info, violations = relaxer.relax_pdb_file(
            small_peptide_pdb
        )

        assert relaxed_pdb is not None
        assert "final_energy" in debug_info


@pytest.mark.integration
class TestEnergyBreakdown:
    """Tests for energy breakdown functionality."""

    def test_get_energy_breakdown_returns_dict(
        self, relaxer, small_peptide_pdb_string
    ):
        """Test that energy breakdown returns a dictionary."""
        result = relaxer.get_energy_breakdown(small_peptide_pdb_string)

        assert isinstance(result, dict)

    def test_get_energy_breakdown_has_total(
        self, relaxer, small_peptide_pdb_string
    ):
        """Test that energy breakdown contains total energy."""
        result = relaxer.get_energy_breakdown(small_peptide_pdb_string)

        assert "total_energy" in result


@pytest.mark.integration
class TestRelaxerConfig:
    """Tests for Relaxer with different configurations."""

    def test_high_stiffness_constrained(self, small_peptide_pdb_string):
        """Test constrained relaxation with high stiffness (more restrained)."""
        from graphrelax.relaxer import Relaxer

        config = RelaxConfig(
            max_iterations=50, stiffness=100.0, constrained=True
        )
        relaxer = Relaxer(config)

        relaxed_pdb, debug_info, _ = relaxer.relax(small_peptide_pdb_string)

        # High stiffness should result in small RMSD
        assert debug_info["rmsd"] < 1.0  # Less than 1 Angstrom

    def test_unconstrained_minimization(self, small_peptide_pdb_string):
        """Test unconstrained minimization (no restraints)."""
        from graphrelax.relaxer import Relaxer

        config = RelaxConfig(max_iterations=50, constrained=False)
        relaxer = Relaxer(config)

        relaxed_pdb, debug_info, _ = relaxer.relax(small_peptide_pdb_string)

        assert relaxed_pdb is not None

    def test_limited_iterations(self, small_peptide_pdb_string):
        """Test relaxation with limited iterations."""
        from graphrelax.relaxer import Relaxer

        config = RelaxConfig(
            max_iterations=10, stiffness=10.0, constrained=False
        )
        relaxer = Relaxer(config)

        relaxed_pdb, debug_info, _ = relaxer.relax(small_peptide_pdb_string)

        assert relaxed_pdb is not None
