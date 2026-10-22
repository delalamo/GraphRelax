"""Tests for graphrelax.resfile module."""

from io import StringIO

import pytest

from graphrelax.resfile import (
    ALL_AAS,
    APOLAR_AAS,
    POLAR_AAS,
    DesignSpec,
    ResfileParser,
    ResidueMode,
    ResidueSpec,
)


class TestConstants:
    """Tests for amino acid constant sets."""

    def test_polar_aas(self):
        """Test polar amino acid set."""
        expected = {"D", "E", "H", "K", "N", "Q", "R", "S", "T"}
        assert POLAR_AAS == expected

    def test_apolar_aas(self):
        """Test apolar amino acid set."""
        expected = {"A", "C", "F", "G", "I", "L", "M", "P", "V", "W", "Y"}
        assert APOLAR_AAS == expected

    def test_all_aas(self):
        """Test all amino acids set."""
        expected = set("ACDEFGHIKLMNPQRSTVWY")
        assert ALL_AAS == expected
        assert len(ALL_AAS) == 20

    def test_polar_apolar_disjoint(self):
        """Test that polar and apolar sets don't overlap."""
        assert POLAR_AAS & APOLAR_AAS == set()

    def test_polar_apolar_cover_all(self):
        """Test that polar + apolar = all amino acids."""
        assert POLAR_AAS | APOLAR_AAS == ALL_AAS


class TestResidueMode:
    """Tests for ResidueMode enum."""

    def test_all_modes_exist(self):
        """Test that all 7 modes are defined."""
        assert len(ResidueMode) == 7

    def test_mode_values(self):
        """Test mode values match their names."""
        assert ResidueMode.NATRO.value == "NATRO"
        assert ResidueMode.NATAA.value == "NATAA"
        assert ResidueMode.ALLAA.value == "ALLAA"
        assert ResidueMode.PIKAA.value == "PIKAA"
        assert ResidueMode.NOTAA.value == "NOTAA"
        assert ResidueMode.POLAR.value == "POLAR"
        assert ResidueMode.APOLAR.value == "APOLAR"


class TestResidueSpec:
    """Tests for ResidueSpec dataclass."""

    def test_default_values(self):
        """Test default values."""
        spec = ResidueSpec(chain="A", resnum=10)
        assert spec.chain == "A"
        assert spec.resnum == 10
        assert spec.icode == ""
        assert spec.mode == ResidueMode.NATAA
        assert spec.allowed_aas is None

    def test_key_property(self):
        """Test key property formatting."""
        spec = ResidueSpec(chain="A", resnum=10)
        assert spec.key == "A10"

    def test_key_with_icode(self):
        """Test key property with insertion code."""
        spec = ResidueSpec(chain="B", resnum=25, icode="A")
        assert spec.key == "B25A"

    def test_get_allowed_aas_allaa(self):
        """Test get_allowed_aas for ALLAA mode."""
        spec = ResidueSpec(chain="A", resnum=1, mode=ResidueMode.ALLAA)
        assert spec.get_allowed_aas() == ALL_AAS

    def test_get_allowed_aas_pikaa(self):
        """Test get_allowed_aas for PIKAA mode."""
        spec = ResidueSpec(
            chain="A",
            resnum=1,
            mode=ResidueMode.PIKAA,
            allowed_aas={"H", "Y", "W"},
        )
        assert spec.get_allowed_aas() == {"H", "Y", "W"}

    def test_get_allowed_aas_pikaa_empty(self):
        """Test get_allowed_aas for PIKAA mode without allowed_aas set."""
        spec = ResidueSpec(chain="A", resnum=1, mode=ResidueMode.PIKAA)
        # Should return all amino acids if none specified
        assert spec.get_allowed_aas() == ALL_AAS

    def test_get_allowed_aas_notaa(self):
        """Test get_allowed_aas for NOTAA mode."""
        spec = ResidueSpec(
            chain="A",
            resnum=1,
            mode=ResidueMode.NOTAA,
            allowed_aas={"C", "P"},
        )
        expected = ALL_AAS - {"C", "P"}
        assert spec.get_allowed_aas() == expected

    def test_get_allowed_aas_notaa_empty(self):
        """Test get_allowed_aas for NOTAA with no exclusions."""
        spec = ResidueSpec(chain="A", resnum=1, mode=ResidueMode.NOTAA)
        assert spec.get_allowed_aas() == ALL_AAS

    def test_get_allowed_aas_polar(self):
        """Test get_allowed_aas for POLAR mode."""
        spec = ResidueSpec(chain="A", resnum=1, mode=ResidueMode.POLAR)
        assert spec.get_allowed_aas() == POLAR_AAS

    def test_get_allowed_aas_apolar(self):
        """Test get_allowed_aas for APOLAR mode."""
        spec = ResidueSpec(chain="A", resnum=1, mode=ResidueMode.APOLAR)
        assert spec.get_allowed_aas() == APOLAR_AAS

    def test_get_allowed_aas_natro(self):
        """Test get_allowed_aas for NATRO mode."""
        spec = ResidueSpec(chain="A", resnum=1, mode=ResidueMode.NATRO)
        assert spec.get_allowed_aas() == set()

    def test_get_allowed_aas_nataa(self):
        """Test get_allowed_aas for NATAA mode."""
        spec = ResidueSpec(chain="A", resnum=1, mode=ResidueMode.NATAA)
        assert spec.get_allowed_aas() == set()

    def test_is_designable_true(self):
        """Test is_designable returns True for design modes."""
        for mode in [
            ResidueMode.ALLAA,
            ResidueMode.PIKAA,
            ResidueMode.NOTAA,
            ResidueMode.POLAR,
            ResidueMode.APOLAR,
        ]:
            spec = ResidueSpec(chain="A", resnum=1, mode=mode)
            assert spec.is_designable() is True

    def test_is_designable_false(self):
        """Test is_designable returns False for non-design modes."""
        for mode in [ResidueMode.NATRO, ResidueMode.NATAA]:
            spec = ResidueSpec(chain="A", resnum=1, mode=mode)
            assert spec.is_designable() is False

    def test_is_repackable_true(self):
        """Test is_repackable returns True for all modes except NATRO."""
        for mode in ResidueMode:
            spec = ResidueSpec(chain="A", resnum=1, mode=mode)
            if mode == ResidueMode.NATRO:
                assert spec.is_repackable() is False
            else:
                assert spec.is_repackable() is True


class TestDesignSpec:
    """Tests for DesignSpec dataclass."""

    def test_get_spec_explicit(self):
        """Test get_spec returns explicit spec when defined."""
        spec = ResidueSpec(chain="A", resnum=10, mode=ResidueMode.ALLAA)
        design_spec = DesignSpec(residue_specs={"A10": spec})

        result = design_spec.get_spec("A", 10)
        assert result.mode == ResidueMode.ALLAA

    def test_get_spec_default(self):
        """Test get_spec returns default spec when not defined."""
        design_spec = DesignSpec(
            residue_specs={}, default_mode=ResidueMode.NATAA
        )

        result = design_spec.get_spec("A", 10)
        assert result.chain == "A"
        assert result.resnum == 10
        assert result.mode == ResidueMode.NATAA

    def test_get_spec_with_icode(self):
        """Test get_spec with insertion code."""
        spec = ResidueSpec(
            chain="A", resnum=10, icode="A", mode=ResidueMode.NATRO
        )
        design_spec = DesignSpec(residue_specs={"A10A": spec})

        result = design_spec.get_spec("A", 10, "A")
        assert result.mode == ResidueMode.NATRO

    def test_get_designable_keys(self):
        """Test get_designable_keys returns correct keys."""
        specs = {
            "A10": ResidueSpec(chain="A", resnum=10, mode=ResidueMode.ALLAA),
            "A15": ResidueSpec(chain="A", resnum=15, mode=ResidueMode.NATAA),
            "A20": ResidueSpec(chain="A", resnum=20, mode=ResidueMode.POLAR),
            "A25": ResidueSpec(chain="A", resnum=25, mode=ResidueMode.NATRO),
        }
        design_spec = DesignSpec(residue_specs=specs)

        designable = design_spec.get_designable_keys()
        assert set(designable) == {"A10", "A20"}

    def test_get_fixed_keys(self):
        """Test get_fixed_keys returns NATRO residues."""
        specs = {
            "A10": ResidueSpec(chain="A", resnum=10, mode=ResidueMode.ALLAA),
            "A15": ResidueSpec(chain="A", resnum=15, mode=ResidueMode.NATAA),
            "A20": ResidueSpec(chain="A", resnum=20, mode=ResidueMode.NATRO),
            "A25": ResidueSpec(chain="A", resnum=25, mode=ResidueMode.NATRO),
        }
        design_spec = DesignSpec(residue_specs=specs)

        fixed = design_spec.get_fixed_keys()
        assert set(fixed) == {"A20", "A25"}


class TestResfileParser:
    """Tests for ResfileParser class."""

    def test_parse_basic_resfile(self, sample_resfile):
        """Test parsing a basic resfile."""
        parser = ResfileParser()
        result = parser.parse(sample_resfile)

        assert result.default_mode == ResidueMode.NATAA
        assert "A10" in result.residue_specs
        assert "A15" in result.residue_specs
        assert "A20" in result.residue_specs

    def test_parse_from_string_io(self, sample_resfile_content):
        """Test parsing from StringIO."""
        parser = ResfileParser()
        result = parser.parse(StringIO(sample_resfile_content))

        assert result.default_mode == ResidueMode.NATAA
        assert "A10" in result.residue_specs

    def test_parse_allaa(self):
        """Test parsing ALLAA command."""
        content = "NATAA\nSTART\n10 A ALLAA\n"
        parser = ResfileParser()
        result = parser.parse(StringIO(content))

        assert result.residue_specs["A10"].mode == ResidueMode.ALLAA

    def test_parse_pikaa(self):
        """Test parsing PIKAA command with amino acids."""
        content = "NATAA\nSTART\n10 A PIKAA HYW\n"
        parser = ResfileParser()
        result = parser.parse(StringIO(content))

        spec = result.residue_specs["A10"]
        assert spec.mode == ResidueMode.PIKAA
        assert spec.allowed_aas == {"H", "Y", "W"}

    def test_parse_notaa(self):
        """Test parsing NOTAA command."""
        content = "NATAA\nSTART\n10 A NOTAA CP\n"
        parser = ResfileParser()
        result = parser.parse(StringIO(content))

        spec = result.residue_specs["A10"]
        assert spec.mode == ResidueMode.NOTAA
        assert spec.allowed_aas == {"C", "P"}

    def test_parse_polar(self):
        """Test parsing POLAR command."""
        content = "NATAA\nSTART\n10 A POLAR\n"
        parser = ResfileParser()
        result = parser.parse(StringIO(content))

        assert result.residue_specs["A10"].mode == ResidueMode.POLAR

    def test_parse_apolar(self):
        """Test parsing APOLAR command."""
        content = "NATAA\nSTART\n10 A APOLAR\n"
        parser = ResfileParser()
        result = parser.parse(StringIO(content))

        assert result.residue_specs["A10"].mode == ResidueMode.APOLAR

    def test_parse_natro(self):
        """Test parsing NATRO command."""
        content = "NATAA\nSTART\n10 A NATRO\n"
        parser = ResfileParser()
        result = parser.parse(StringIO(content))

        assert result.residue_specs["A10"].mode == ResidueMode.NATRO

    def test_parse_default_mode_allaa(self):
        """Test setting default mode to ALLAA."""
        content = "ALLAA\nSTART\n"
        parser = ResfileParser()
        result = parser.parse(StringIO(content))

        assert result.default_mode == ResidueMode.ALLAA

    def test_parse_comments(self):
        """Test that comments are ignored."""
        content = (
            "# This is a comment\nNATAA\nSTART\n# Another comment\n10 A ALLAA\n"
        )
        parser = ResfileParser()
        result = parser.parse(StringIO(content))

        assert result.default_mode == ResidueMode.NATAA
        assert "A10" in result.residue_specs

    def test_parse_inline_comments(self):
        """Test that inline comments are handled."""
        content = "NATAA\nSTART\n10 A ALLAA # design this position\n"
        parser = ResfileParser()
        result = parser.parse(StringIO(content))

        assert result.residue_specs["A10"].mode == ResidueMode.ALLAA

    def test_parse_empty_lines(self):
        """Test that empty lines are handled."""
        content = "NATAA\n\nSTART\n\n10 A ALLAA\n\n"
        parser = ResfileParser()
        result = parser.parse(StringIO(content))

        assert "A10" in result.residue_specs

    def test_parse_no_start_line(self):
        """Test parsing without START line (header only)."""
        content = "ALLAA\n"
        parser = ResfileParser()
        result = parser.parse(StringIO(content))

        assert result.default_mode == ResidueMode.ALLAA
        assert len(result.residue_specs) == 0

    def test_parse_malformed_line_skipped(self):
        """Test that malformed lines are skipped."""
        content = "NATAA\nSTART\ninvalid line\n10 A ALLAA\n"
        parser = ResfileParser()
        result = parser.parse(StringIO(content))

        # Only valid line should be parsed
        assert len(result.residue_specs) == 1
        assert "A10" in result.residue_specs

    def test_parse_unknown_command_skipped(self):
        """Test that unknown commands are skipped."""
        content = "NATAA\nSTART\n10 A UNKNOWN\n15 A ALLAA\n"
        parser = ResfileParser()
        result = parser.parse(StringIO(content))

        assert "A10" not in result.residue_specs
        assert "A15" in result.residue_specs

    def test_parse_case_insensitive(self):
        """Test that commands are case insensitive."""
        content = "nataa\nstart\n10 A allaa\n"
        parser = ResfileParser()
        result = parser.parse(StringIO(content))

        assert result.default_mode == ResidueMode.NATAA
        assert result.residue_specs["A10"].mode == ResidueMode.ALLAA

    def test_parse_multiple_chains(self):
        """Test parsing residues from multiple chains."""
        content = "NATAA\nSTART\n10 A ALLAA\n20 B POLAR\n"
        parser = ResfileParser()
        result = parser.parse(StringIO(content))

        assert result.residue_specs["A10"].chain == "A"
        assert result.residue_specs["B20"].chain == "B"

    def test_parse_file_path(self, sample_resfile):
        """Test parsing from file path."""
        parser = ResfileParser()
        result = parser.parse(sample_resfile)

        assert isinstance(result, DesignSpec)
        assert result.default_mode == ResidueMode.NATAA


@pytest.mark.parametrize(
    "command,allowed",
    [("PIKAA AC", {"A", "C"}), ("NOTAA CP", ALL_AAS - {"C", "P"})],
)
def test_header_amino_acid_restrictions(command, allowed):
    parsed = ResfileParser().parse(StringIO(command + "\nSTART\n"))
    assert parsed.get_spec("A", 1).get_allowed_aas() == allowed
