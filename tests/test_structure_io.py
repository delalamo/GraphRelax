"""Tests for graphrelax.structure_io module."""

import pytest

from graphrelax.structure_io import (
    StructureFormat,
    convert_cif_to_pdb,
    convert_pdb_to_cif,
    convert_to_format,
    detect_format,
    ensure_pdb_format,
    get_output_format,
    read_structure,
    write_structure,
)


class TestDetectFormat:
    """Tests for detect_format function."""

    def test_detect_pdb_format(self, tmp_path):
        """Test detection of PDB format."""
        path = tmp_path / "test.pdb"
        assert detect_format(path) == StructureFormat.PDB

    def test_detect_cif_format(self, tmp_path):
        """Test detection of CIF format."""
        path = tmp_path / "test.cif"
        assert detect_format(path) == StructureFormat.CIF

    def test_detect_mmcif_format(self, tmp_path):
        """Test detection of mmCIF format."""
        path = tmp_path / "test.mmcif"
        assert detect_format(path) == StructureFormat.CIF

    def test_detect_uppercase_extension(self, tmp_path):
        """Test detection handles uppercase extensions."""
        path = tmp_path / "test.PDB"
        assert detect_format(path) == StructureFormat.PDB

        path = tmp_path / "test.CIF"
        assert detect_format(path) == StructureFormat.CIF

    def test_unknown_format_raises(self, tmp_path):
        """Test that unknown extension raises ValueError."""
        path = tmp_path / "test.xyz"
        with pytest.raises(ValueError, match="Unknown structure format"):
            detect_format(path)

    def test_no_extension_raises(self, tmp_path):
        """Test that missing extension raises ValueError."""
        path = tmp_path / "testfile"
        with pytest.raises(ValueError, match="Unknown structure format"):
            detect_format(path)


class TestReadStructure:
    """Tests for read_structure function."""

    def test_read_pdb_file(self, tmp_path, small_peptide_pdb_string):
        """Test reading PDB file."""
        path = tmp_path / "test.pdb"
        path.write_text(small_peptide_pdb_string)

        content = read_structure(path)
        assert content == small_peptide_pdb_string

    def test_read_preserves_newlines(self, tmp_path):
        """Test that newlines are preserved."""
        content = "ATOM      1\nATOM      2\n"
        path = tmp_path / "test.pdb"
        path.write_text(content)

        result = read_structure(path)
        assert result == content


class TestWriteStructure:
    """Tests for write_structure function."""

    def test_write_pdb_format(self, tmp_path):
        """Test writing PDB format."""
        content = "ATOM      1  N   ALA A   1"
        path = tmp_path / "output.pdb"

        write_structure(content, path, StructureFormat.PDB)

        assert path.exists()
        assert path.read_text() == content

    def test_write_creates_parent_dirs(self, tmp_path):
        """Test that parent directories are created."""
        content = "ATOM      1  N   ALA A   1"
        path = tmp_path / "subdir" / "nested" / "output.pdb"

        write_structure(content, path)

        assert path.exists()
        assert path.read_text() == content

    def test_write_auto_detect_format(self, tmp_path):
        """Test format auto-detection from path."""
        content = "ATOM      1  N   ALA A   1"
        path = tmp_path / "output.pdb"

        # Should not raise even without explicit format
        write_structure(content, path)
        assert path.exists()


class TestConversion:
    """Tests for format conversion functions."""

    def test_pdb_to_cif_conversion(self, small_peptide_pdb_string):
        """Test converting PDB to CIF format."""
        cif_string = convert_pdb_to_cif(small_peptide_pdb_string)

        # CIF files start with data_ block
        assert cif_string.startswith("data_")
        # Should contain atom_site loop
        assert "_atom_site" in cif_string

    def test_cif_to_pdb_conversion(self, small_peptide_pdb_string):
        """Test converting CIF to PDB format."""
        # First convert to CIF
        cif_string = convert_pdb_to_cif(small_peptide_pdb_string)

        # Then convert back to PDB
        pdb_string = convert_cif_to_pdb(cif_string)

        # Should have ATOM records
        assert "ATOM" in pdb_string

    def test_roundtrip_preserves_atoms(self, small_peptide_pdb_string):
        """Test that roundtrip conversion preserves atom count."""
        # Count atoms in original
        original_atoms = len(
            [
                line
                for line in small_peptide_pdb_string.splitlines()
                if line.startswith("ATOM")
            ]
        )

        # Roundtrip: PDB -> CIF -> PDB
        cif_string = convert_pdb_to_cif(small_peptide_pdb_string)
        pdb_string = convert_cif_to_pdb(cif_string)

        # Count atoms after roundtrip
        final_atoms = len(
            [
                line
                for line in pdb_string.splitlines()
                if line.startswith("ATOM")
            ]
        )

        assert final_atoms == original_atoms

    def test_convert_to_format_pdb(self, small_peptide_pdb_string):
        """Test convert_to_format with PDB target."""
        result = convert_to_format(
            small_peptide_pdb_string, StructureFormat.PDB
        )
        # Should return unchanged
        assert result == small_peptide_pdb_string

    def test_convert_to_format_cif(self, small_peptide_pdb_string):
        """Test convert_to_format with CIF target."""
        result = convert_to_format(
            small_peptide_pdb_string, StructureFormat.CIF
        )
        assert result.startswith("data_")


class TestEnsurePdbFormat:
    """Tests for ensure_pdb_format function."""

    def test_pdb_input_unchanged(self, tmp_path, small_peptide_pdb_string):
        """Test that PDB input is returned unchanged."""
        path = tmp_path / "test.pdb"
        path.write_text(small_peptide_pdb_string)

        result = ensure_pdb_format(small_peptide_pdb_string, path)
        assert result == small_peptide_pdb_string

    def test_cif_input_converted(self, tmp_path, small_peptide_pdb_string):
        """Test that CIF input is converted to PDB."""
        # Create CIF content
        cif_string = convert_pdb_to_cif(small_peptide_pdb_string)
        path = tmp_path / "test.cif"
        path.write_text(cif_string)

        result = ensure_pdb_format(cif_string, path)
        # Should be PDB format now
        assert "ATOM" in result
        assert not result.startswith("data_")


class TestGetOutputFormat:
    """Tests for get_output_format function."""

    def test_uses_output_path_extension(self, tmp_path):
        """Test that output path extension determines format."""
        input_path = tmp_path / "input.pdb"
        output_path = tmp_path / "output.cif"

        result = get_output_format(input_path, output_path)
        assert result == StructureFormat.CIF

    def test_pdb_output_extension(self, tmp_path):
        """Test PDB output extension."""
        input_path = tmp_path / "input.cif"
        output_path = tmp_path / "output.pdb"

        result = get_output_format(input_path, output_path)
        assert result == StructureFormat.PDB

    def test_falls_back_to_input_format(self, tmp_path):
        """Test fallback to input format when output has no extension."""
        input_path = tmp_path / "input.cif"
        output_path = tmp_path / "output"  # No extension

        result = get_output_format(input_path, output_path)
        assert result == StructureFormat.CIF

    def test_same_format_preserved(self, tmp_path):
        """Test same format is preserved."""
        input_path = tmp_path / "input.pdb"
        output_path = tmp_path / "output.pdb"

        result = get_output_format(input_path, output_path)
        assert result == StructureFormat.PDB
