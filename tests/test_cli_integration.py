"""CLI integration tests for graphrelax.

These tests verify the CLI interface works correctly end-to-end.
"""

import subprocess
import sys

import pytest

from graphrelax.weights import weights_exist as weights_available

# Skip entire module if OpenMM not available
pytest.importorskip("openmm")


class TestCLIHelp:
    """Tests for CLI help and version output."""

    def test_help_output(self):
        """Test that --help produces output and exits successfully."""
        result = subprocess.run(
            [sys.executable, "-m", "graphrelax.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "graphrelax" in result.stdout.lower()
        assert "usage" in result.stdout.lower()

    def test_help_shows_modes(self):
        """Test that help shows all available modes."""
        result = subprocess.run(
            [sys.executable, "-m", "graphrelax.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert "--relax" in result.stdout
        assert "--design" in result.stdout
        assert "--no-repack" in result.stdout
        assert "--repack-only" in result.stdout
        assert "--design-only" in result.stdout

    def test_help_shows_required_args(self):
        """Test that help shows required arguments."""
        result = subprocess.run(
            [sys.executable, "-m", "graphrelax.cli", "--help"],
            capture_output=True,
            text=True,
        )
        assert "-i" in result.stdout or "--input" in result.stdout
        assert "-o" in result.stdout or "--output" in result.stdout


class TestCLIArgumentValidation:
    """Tests for CLI argument validation."""

    def test_missing_input_fails(self, tmp_path):
        """Test that missing input file fails with error."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "graphrelax.cli",
                "-o",
                str(tmp_path / "output.pdb"),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_missing_output_fails(self, small_peptide_pdb):
        """Test that missing output file fails with error."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "graphrelax.cli",
                "-i",
                str(small_peptide_pdb),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_nonexistent_input_fails(self, tmp_path):
        """Test that nonexistent input file fails with error."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "graphrelax.cli",
                "-i",
                str(tmp_path / "nonexistent.pdb"),
                "-o",
                str(tmp_path / "output.pdb"),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert (
            "not found" in result.stderr.lower()
            or "error" in result.stderr.lower()
        )

    def test_nonexistent_resfile_fails(self, small_peptide_pdb, tmp_path):
        """Test that nonexistent resfile fails with error."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "graphrelax.cli",
                "-i",
                str(small_peptide_pdb),
                "-o",
                str(tmp_path / "output.pdb"),
                "--resfile",
                str(tmp_path / "nonexistent.resfile"),
                "--no-repack",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0


class TestCLIMutuallyExclusiveModes:
    """Tests for mutually exclusive mode arguments."""

    def test_relax_and_design_exclusive(self, small_peptide_pdb, tmp_path):
        """Test that --relax and --design are mutually exclusive."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "graphrelax.cli",
                "-i",
                str(small_peptide_pdb),
                "-o",
                str(tmp_path / "output.pdb"),
                "--relax",
                "--design",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "not allowed" in result.stderr.lower()

    def test_no_repack_and_repack_only_exclusive(
        self, small_peptide_pdb, tmp_path
    ):
        """Test that --no-repack and --repack-only are mutually exclusive."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "graphrelax.cli",
                "-i",
                str(small_peptide_pdb),
                "-o",
                str(tmp_path / "output.pdb"),
                "--no-repack",
                "--repack-only",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0


@pytest.mark.integration
class TestCLINoRepackMode:
    """Integration tests for CLI --no-repack mode."""

    def test_no_repack_completes(
        self, small_peptide_pdb, tmp_path, weights_available
    ):
        """Test that --no-repack mode completes successfully."""
        output_pdb = tmp_path / "minimized.pdb"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "graphrelax.cli",
                "-i",
                str(small_peptide_pdb),
                "-o",
                str(output_pdb),
                "--no-repack",
                "--n-iter",
                "1",
                "--max-iterations",
                "50",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"CLI failed with: {result.stderr}"
        assert output_pdb.exists()

    def test_no_repack_creates_output(
        self, small_peptide_pdb, tmp_path, weights_available
    ):
        """Test that --no-repack creates valid PDB output."""
        output_pdb = tmp_path / "minimized.pdb"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "graphrelax.cli",
                "-i",
                str(small_peptide_pdb),
                "-o",
                str(output_pdb),
                "--no-repack",
                "--n-iter",
                "1",
                "--max-iterations",
                "50",
            ],
            capture_output=True,
            text=True,
        )

        content = output_pdb.read_text()
        assert "ATOM" in content

    def test_no_repack_multiple_outputs(
        self, small_peptide_pdb, tmp_path, weights_available
    ):
        """Test --no-repack with multiple outputs."""
        output_pdb = tmp_path / "minimized.pdb"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "graphrelax.cli",
                "-i",
                str(small_peptide_pdb),
                "-o",
                str(output_pdb),
                "--no-repack",
                "-n",
                "2",
                "--n-iter",
                "1",
                "--max-iterations",
                "50",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert (tmp_path / "minimized_1.pdb").exists()
        assert (tmp_path / "minimized_2.pdb").exists()

    def test_no_repack_with_scorefile(
        self, small_peptide_pdb, tmp_path, weights_available
    ):
        """Test --no-repack with scorefile output."""
        output_pdb = tmp_path / "minimized.pdb"
        scorefile = tmp_path / "scores.sc"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "graphrelax.cli",
                "-i",
                str(small_peptide_pdb),
                "-o",
                str(output_pdb),
                "--no-repack",
                "--n-iter",
                "1",
                "--max-iterations",
                "50",
                "--scorefile",
                str(scorefile),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert scorefile.exists()
        content = scorefile.read_text()
        assert "SCORE:" in content


@pytest.mark.integration
@pytest.mark.skipif(
    not weights_available(),
    reason="LigandMPNN weights not downloaded",
)
class TestCLIDesignMode:
    """Integration tests for CLI --design mode (requires weights)."""

    def test_design_mode_completes(self, small_peptide_pdb, tmp_path):
        """Test that --design mode completes successfully."""
        output_pdb = tmp_path / "designed.pdb"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "graphrelax.cli",
                "-i",
                str(small_peptide_pdb),
                "-o",
                str(output_pdb),
                "--design",
                "--n-iter",
                "1",
                "--max-iterations",
                "50",
                "--seed",
                "42",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0, f"CLI failed with: {result.stderr}"
        assert output_pdb.exists()

    def test_design_with_resfile(self, small_peptide_pdb, tmp_path):
        """Test --design with resfile constraints."""
        # Create resfile
        resfile = tmp_path / "design.resfile"
        resfile.write_text(
            """NATAA
START
1 A ALLAA
2 A NATRO
3 A ALLAA
4 A NATRO
5 A ALLAA
"""
        )

        output_pdb = tmp_path / "designed.pdb"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "graphrelax.cli",
                "-i",
                str(small_peptide_pdb),
                "-o",
                str(output_pdb),
                "--design",
                "--resfile",
                str(resfile),
                "--n-iter",
                "1",
                "--max-iterations",
                "50",
                "--seed",
                "42",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0
        assert output_pdb.exists()


@pytest.mark.integration
@pytest.mark.skipif(
    not weights_available(),
    reason="LigandMPNN weights not downloaded",
)
class TestCLIRelaxMode:
    """Integration tests for CLI default relax mode (requires weights)."""

    def test_default_mode_completes(self, small_peptide_pdb, tmp_path):
        """Test that default mode (relax) completes successfully."""
        output_pdb = tmp_path / "relaxed.pdb"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "graphrelax.cli",
                "-i",
                str(small_peptide_pdb),
                "-o",
                str(output_pdb),
                "--n-iter",
                "1",
                "--max-iterations",
                "50",
                "--seed",
                "42",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert result.returncode == 0, f"CLI failed with: {result.stderr}"
        assert output_pdb.exists()


class TestCLIVerbosity:
    """Tests for CLI verbosity options."""

    @pytest.mark.integration
    def test_verbose_output(
        self, small_peptide_pdb, tmp_path, weights_available
    ):
        """Test that -v produces more output."""
        output_pdb = tmp_path / "minimized.pdb"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "graphrelax.cli",
                "-i",
                str(small_peptide_pdb),
                "-o",
                str(output_pdb),
                "--no-repack",
                "--n-iter",
                "1",
                "-v",
            ],
            capture_output=True,
            text=True,
        )

        # Verbose mode should produce more detailed logging
        assert result.returncode == 0
        # Check for some logging output (stderr or stdout depending on config)
        output = result.stderr + result.stdout
        assert len(output) > 0
