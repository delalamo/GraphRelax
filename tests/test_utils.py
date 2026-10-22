"""Tests for graphrelax.utils module."""

from pathlib import Path

import pytest

from graphrelax.utils import (
    compute_sequence_recovery,
    format_output_path,
    save_pdb_string,
    write_scorefile,
)


class TestComputeSequenceRecovery:
    """Tests for compute_sequence_recovery function."""

    def test_identical_sequences(self):
        """Test recovery of identical sequences."""
        result = compute_sequence_recovery("ACDEFG", "ACDEFG")
        assert result == 1.0

    def test_completely_different_sequences(self):
        """Test recovery of completely different sequences."""
        result = compute_sequence_recovery("AAAAAA", "GGGGGG")
        assert result == 0.0

    def test_partial_recovery(self):
        """Test partial sequence recovery."""
        result = compute_sequence_recovery("ACDEFG", "ACDXYZ")
        assert result == pytest.approx(0.5)

    def test_single_mismatch(self):
        """Test with single mismatch."""
        result = compute_sequence_recovery("ACDEFG", "ACDEXX")
        assert result == pytest.approx(4 / 6)

    def test_different_lengths_truncates(self):
        """Test that different lengths are handled by truncating."""
        result = compute_sequence_recovery("ACDEFG", "ACD")
        # Should compare first 3 characters
        assert result == 1.0

    def test_empty_sequences(self):
        """Test with empty sequences."""
        result = compute_sequence_recovery("", "")
        assert result == 0.0

    def test_one_empty_sequence(self):
        """Test with one empty sequence."""
        result = compute_sequence_recovery("ACDEFG", "")
        assert result == 0.0


class TestFormatOutputPath:
    """Tests for format_output_path function."""

    def test_single_output(self):
        """Test with single output (no index needed)."""
        base = Path("/tmp/output.pdb")
        result = format_output_path(base, 1, 1)
        assert result == base

    def test_multiple_outputs_first(self):
        """Test first output with multiple outputs."""
        base = Path("/tmp/output.pdb")
        result = format_output_path(base, 1, 5)
        assert result == Path("/tmp/output_1.pdb")

    def test_multiple_outputs_middle(self):
        """Test middle output with multiple outputs."""
        base = Path("/tmp/output.pdb")
        result = format_output_path(base, 3, 5)
        assert result == Path("/tmp/output_3.pdb")

    def test_multiple_outputs_last(self):
        """Test last output with multiple outputs."""
        base = Path("/tmp/output.pdb")
        result = format_output_path(base, 5, 5)
        assert result == Path("/tmp/output_5.pdb")

    def test_preserves_directory(self):
        """Test that directory is preserved."""
        base = Path("/path/to/dir/output.pdb")
        result = format_output_path(base, 2, 3)
        assert result.parent == Path("/path/to/dir")

    def test_preserves_suffix(self):
        """Test that suffix is preserved."""
        base = Path("/tmp/output.cif")
        result = format_output_path(base, 1, 2)
        assert result.suffix == ".cif"


class TestWriteScorefile:
    """Tests for write_scorefile function."""

    def test_basic_scorefile(self, tmp_path):
        """Test writing a basic scorefile."""
        scores = [
            {"total_score": -100.0, "description": "test1.pdb"},
            {"total_score": -90.0, "description": "test2.pdb"},
        ]
        path = tmp_path / "scores.sc"
        write_scorefile(path, scores)

        content = path.read_text()
        lines = content.strip().split("\n")

        # Check header
        assert lines[0].startswith("SCORE:")
        assert "total_score" in lines[0]
        assert "description" in lines[0]

        # Check data lines
        assert lines[1].startswith("SCORE:")
        assert "-100.0" in lines[1] or "-100" in lines[1]
        assert "test1.pdb" in lines[1]

    def test_scorefile_multiple_columns(self, tmp_path):
        """Test scorefile with multiple columns."""
        scores = [
            {
                "total_score": -100.0,
                "bond_energy": 10.0,
                "description": "test.pdb",
            }
        ]
        path = tmp_path / "scores.sc"
        write_scorefile(path, scores)

        content = path.read_text()
        assert "total_score" in content
        assert "bond_energy" in content

    def test_scorefile_empty_scores(self, tmp_path):
        """Test that empty scores list writes nothing."""
        path = tmp_path / "scores.sc"
        write_scorefile(path, [])

        # File should not be created or be empty
        assert not path.exists() or path.read_text() == ""

    def test_scorefile_custom_header(self, tmp_path):
        """Test scorefile with custom header order."""
        scores = [{"b": 2.0, "a": 1.0, "c": 3.0}]
        header = ["a", "b", "c"]
        path = tmp_path / "scores.sc"
        write_scorefile(path, scores, header=header)

        content = path.read_text()
        lines = content.strip().split("\n")

        # Check header order
        header_line = lines[0]
        a_pos = header_line.index("a")
        b_pos = header_line.index("b")
        c_pos = header_line.index("c")
        assert a_pos < b_pos < c_pos


class TestSavePdbString:
    """Tests for save_pdb_string function."""

    def test_basic_save(self, tmp_path):
        """Test basic PDB string saving."""
        pdb_string = "ATOM      1  N   ALA A   1"
        path = tmp_path / "test.pdb"

        save_pdb_string(pdb_string, path)

        assert path.exists()
        assert path.read_text() == pdb_string

    def test_creates_parent_dirs(self, tmp_path):
        """Test that parent directories are created."""
        pdb_string = "ATOM      1  N   ALA A   1"
        path = tmp_path / "subdir" / "nested" / "test.pdb"

        save_pdb_string(pdb_string, path)

        assert path.exists()
        assert path.read_text() == pdb_string

    def test_overwrites_existing(self, tmp_path):
        """Test that existing files are overwritten."""
        path = tmp_path / "test.pdb"
        path.write_text("old content")

        save_pdb_string("new content", path)

        assert path.read_text() == "new content"

    def test_multiline_pdb(self, tmp_path):
        """Test saving multiline PDB string."""
        pdb_string = (
            "ATOM      1  N   ALA A   1\nATOM      2  CA  ALA A   1\nEND\n"
        )
        path = tmp_path / "test.pdb"

        save_pdb_string(pdb_string, path)

        assert path.read_text() == pdb_string
