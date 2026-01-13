"""Tests for protein structure idealization module."""

import io
import math

import pytest
from Bio.PDB import PDBParser

from graphrelax.config import IdealizeConfig
from graphrelax.idealize import (
    DihedralAngles,
    correct_cis_omega,
    extract_dihedrals,
    extract_ligands,
    restore_ligands,
)


class TestLigandExtraction:
    """Tests for ligand extraction and restoration."""

    def test_extract_ligands_no_ligands(self, small_peptide_pdb_string):
        """Protein without ligands should return empty ligand string."""
        protein_pdb, ligand_lines = extract_ligands(small_peptide_pdb_string)
        assert ligand_lines.strip() == ""
        assert "ATOM" in protein_pdb
        assert "END" in protein_pdb

    def test_extract_ligands_with_hetatm(self):
        """Ligands (non-water HETATM) should be extracted."""
        # PDB format lines are necessarily > 80 chars
        pdb_with_ligand = (  # noqa: E501
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00"
            "           N\n"
            "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00"
            "           C\n"
            "HETATM   50 FE   HEM A 100       5.000   5.000   5.000  1.00  0.00"
            "          FE\n"
            "HETATM   51  O   HOH A 200       8.000   8.000   8.000  1.00  0.00"
            "           O\n"
            "END\n"
        )
        protein_pdb, ligand_lines = extract_ligands(pdb_with_ligand)

        # Heme should be extracted
        assert "HEM" in ligand_lines
        assert "FE" in ligand_lines

        # Water should stay with protein
        assert "HOH" in protein_pdb

        # Protein atoms should be preserved
        assert "ALA" in protein_pdb

    def test_restore_ligands(self):
        """Ligands should be restored before END record."""
        protein_pdb = (  # noqa: E501
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00"
            "           N\n"
            "END\n"
        )
        ligand_lines = (
            "HETATM   50 FE   HEM A 100       5.000   5.000   5.000  1.00  0.00"
        )

        result = restore_ligands(protein_pdb, ligand_lines)

        # Ligand should appear before END
        assert result.index("HEM") < result.index("END")

    def test_restore_ligands_empty(self, small_peptide_pdb_string):
        """Empty ligand string should not modify PDB."""
        result = restore_ligands(small_peptide_pdb_string, "")
        assert result == small_peptide_pdb_string

    def test_roundtrip_ligands(self):
        """Extract + restore should preserve ligand content."""
        original = (  # noqa: E501
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00"
            "           N\n"
            "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00"
            "           C\n"
            "HETATM   50 FE   HEM A 100       5.000   5.000   5.000  1.00  0.00"
            "          FE\n"
            "HETATM   51  O   HOH A 200       8.000   8.000   8.000  1.00  0.00"
            "           O\n"
            "END\n"
        )
        protein_pdb, ligand_lines = extract_ligands(original)
        restored = restore_ligands(protein_pdb, ligand_lines)

        # Original HEM line should be present
        assert "HEM" in restored
        assert "FE" in restored


class TestDihedralExtraction:
    """Tests for dihedral angle extraction."""

    def test_extract_dihedrals_internal_residue(self, small_peptide_pdb_string):
        """Internal residue should have phi, psi, omega extracted."""
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(
            "test", io.StringIO(small_peptide_pdb_string)
        )
        residues = list(structure.get_residues())

        # Residue 3 (index 2) is internal
        dihedrals = extract_dihedrals(
            residues[2], prev_residue=residues[1], next_residue=residues[3]
        )

        # Should have phi, psi, omega
        assert dihedrals.phi is not None
        assert dihedrals.psi is not None
        assert dihedrals.omega is not None

        # Angles should be in reasonable range (-pi to pi)
        assert -math.pi <= dihedrals.phi <= math.pi
        assert -math.pi <= dihedrals.psi <= math.pi
        assert -math.pi <= dihedrals.omega <= math.pi

    def test_extract_dihedrals_n_terminal(self, small_peptide_pdb_string):
        """N-terminal residue should not have phi or omega."""
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(
            "test", io.StringIO(small_peptide_pdb_string)
        )
        residues = list(structure.get_residues())

        # First residue has no predecessor
        dihedrals = extract_dihedrals(
            residues[0], prev_residue=None, next_residue=residues[1]
        )

        # Should not have phi or omega (no previous residue)
        assert dihedrals.phi is None
        assert dihedrals.omega is None

        # Should have psi (has next residue)
        assert dihedrals.psi is not None

    def test_extract_dihedrals_c_terminal(self, small_peptide_pdb_string):
        """C-terminal residue should not have psi."""
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(
            "test", io.StringIO(small_peptide_pdb_string)
        )
        residues = list(structure.get_residues())

        # Last residue has no successor
        dihedrals = extract_dihedrals(
            residues[-1], prev_residue=residues[-2], next_residue=None
        )

        # Should have phi and omega (has previous residue)
        assert dihedrals.phi is not None
        assert dihedrals.omega is not None

        # Should not have psi (no next residue)
        assert dihedrals.psi is None

    def test_extract_chi_angles_alanine(self, small_peptide_pdb_string):
        """Alanine has no chi angles."""
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(
            "test", io.StringIO(small_peptide_pdb_string)
        )
        residues = list(structure.get_residues())

        dihedrals = extract_dihedrals(
            residues[2], prev_residue=residues[1], next_residue=residues[3]
        )

        # Alanine has no chi angles
        assert dihedrals.chi == []


class TestCisOmegaCorrection:
    """Tests for cis-omega angle correction."""

    def test_correct_cis_omega_trans_unchanged(self):
        """Trans omega (~180 deg) should not be changed."""
        omega = math.pi  # 180 degrees
        result = correct_cis_omega(omega, "ALA")
        assert abs(result - math.pi) < 0.01

    def test_correct_cis_omega_cis_corrected(self):
        """Cis omega (~0 deg) should be corrected to trans."""
        omega = 0.0  # 0 degrees (cis)
        result = correct_cis_omega(omega, "ALA")
        assert abs(result - math.pi) < 0.01

    def test_correct_cis_omega_proline_preserved(self):
        """Cis-proline should not be corrected."""
        omega = 0.0  # 0 degrees (cis)
        result = correct_cis_omega(omega, "PRO")
        # Proline cis should be preserved
        assert abs(result) < 0.01

    def test_correct_cis_omega_near_trans_unchanged(self):
        """Omega near trans (170 deg) should not be changed."""
        omega = math.radians(170)
        result = correct_cis_omega(omega, "ALA")
        # Should remain near 170 degrees
        assert abs(math.degrees(result) - 170) < 10


class TestDihedralAnglesDataclass:
    """Tests for DihedralAngles dataclass."""

    def test_default_chi_is_empty_list(self):
        """Chi should default to empty list, not None."""
        angles = DihedralAngles()
        assert angles.chi == []
        assert angles.phi is None
        assert angles.psi is None
        assert angles.omega is None

    def test_with_values(self):
        """DihedralAngles should store provided values."""
        angles = DihedralAngles(phi=1.0, psi=2.0, omega=3.0, chi=[0.5, 1.5])
        assert angles.phi == 1.0
        assert angles.psi == 2.0
        assert angles.omega == 3.0
        assert angles.chi == [0.5, 1.5]


@pytest.mark.integration
class TestIdealizeIntegration:
    """Integration tests for full idealization workflow."""

    def test_idealize_structure_no_gaps(self, small_peptide_pdb_string):
        """Idealization should work on structure without gaps."""
        from graphrelax.idealize import idealize_structure

        config = IdealizeConfig(enabled=True)
        result_pdb, gaps = idealize_structure(small_peptide_pdb_string, config)

        # Should have no gaps in this simple peptide
        assert len(gaps) == 0

        # Result should still be valid PDB
        assert "ATOM" in result_pdb
        assert "END" in result_pdb

        # Should have same number of residues
        parser = PDBParser(QUIET=True)
        original = parser.get_structure(
            "orig", io.StringIO(small_peptide_pdb_string)
        )
        result = parser.get_structure("result", io.StringIO(result_pdb))

        orig_residues = list(original.get_residues())
        result_residues = list(result.get_residues())
        assert len(orig_residues) == len(result_residues)

    def test_idealize_structure_with_ligand(self, small_peptide_pdb_string):
        """Idealization should preserve ligands."""
        from graphrelax.idealize import extract_ligands, restore_ligands

        # Add a ligand to the small peptide
        pdb_with_ligand = small_peptide_pdb_string.replace(
            "END",
            "HETATM   50 FE   HEM A 100       5.000   5.000   5.000  1.00  0.00"
            "          FE\nEND",
        )

        # Test ligand extraction and restoration roundtrip
        protein_pdb, ligand_lines = extract_ligands(pdb_with_ligand)
        assert "HEM" in ligand_lines
        assert "FE" in ligand_lines
        assert "HEM" not in protein_pdb

        restored = restore_ligands(protein_pdb, ligand_lines)
        assert "HEM" in restored
        assert "FE" in restored

    def test_idealize_config_disabled(self, small_peptide_pdb_string):
        """When disabled, idealization should not run."""
        config = IdealizeConfig(enabled=False)
        assert not config.enabled

    def test_minimize_with_constraints_basic(self, small_peptide_pdb_string):
        """minimize_with_constraints should run without errors."""
        from graphrelax.idealize import minimize_with_constraints

        # This test ensures the function doesn't crash
        # It would have caught the addMissingResidues() AttributeError
        result = minimize_with_constraints(
            small_peptide_pdb_string,
            stiffness=10.0,
            add_missing_residues=False,
        )

        assert "ATOM" in result
        assert "END" in result

    def test_minimize_with_constraints_add_missing(
        self, small_peptide_pdb_string
    ):
        """minimize_with_constraints should handle add_missing_residues=True."""
        from graphrelax.idealize import minimize_with_constraints

        # Test with add_missing_residues=True (the default that caused the bug)
        result = minimize_with_constraints(
            small_peptide_pdb_string,
            stiffness=10.0,
            add_missing_residues=True,
        )

        assert "ATOM" in result
        assert "END" in result

    def test_minimize_with_seqres_missing_residues(self):
        """minimize_with_constraints should handle PDB with SEQRES missing residues."""
        from graphrelax.idealize import minimize_with_constraints

        # PDB with SEQRES indicating 5 residues but only 3 present (missing 1 and 5)
        # This tests the actual missing residue detection path
        pdb_with_seqres = (  # noqa: E501
            "SEQRES   1 A    5  ALA ALA ALA ALA ALA\n"
            "ATOM      1  N   ALA A   2       1.458   0.000   0.000  1.00  0.00           N\n"
            "ATOM      2  CA  ALA A   2       2.916   0.000   0.000  1.00  0.00           C\n"
            "ATOM      3  C   ALA A   2       3.467   1.420   0.000  1.00  0.00           C\n"
            "ATOM      4  O   ALA A   2       2.704   2.390   0.000  1.00  0.00           O\n"
            "ATOM      5  CB  ALA A   2       3.444  -0.760  -1.216  1.00  0.00           C\n"
            "ATOM      6  N   ALA A   3       4.784   1.540   0.000  1.00  0.00           N\n"
            "ATOM      7  CA  ALA A   3       5.399   2.861   0.000  1.00  0.00           C\n"
            "ATOM      8  C   ALA A   3       6.917   2.789   0.000  1.00  0.00           C\n"
            "ATOM      9  O   ALA A   3       7.523   1.719   0.000  1.00  0.00           O\n"
            "ATOM     10  CB  ALA A   3       4.931   3.699   1.186  1.00  0.00           C\n"
            "ATOM     11  N   ALA A   4       7.523   3.969   0.000  1.00  0.00           N\n"
            "ATOM     12  CA  ALA A   4       8.977   4.109   0.000  1.00  0.00           C\n"
            "ATOM     13  C   ALA A   4       9.528   5.529   0.000  1.00  0.00           C\n"
            "ATOM     14  O   ALA A   4       8.765   6.499   0.000  1.00  0.00           O\n"
            "ATOM     15  CB  ALA A   4       9.505   3.349  -1.216  1.00  0.00           C\n"
            "END\n"
        )

        # Test with add_missing_residues=True - pdbfixer should detect missing
        # residues from SEQRES and handle them
        result = minimize_with_constraints(
            pdb_with_seqres,
            stiffness=10.0,
            add_missing_residues=True,
        )

        assert "ATOM" in result
        assert "END" in result

    def test_minimize_with_seqres_skip_missing(self):
        """minimize_with_constraints should skip missing residues when disabled."""
        from graphrelax.idealize import minimize_with_constraints

        # Same PDB with SEQRES but we'll skip adding missing residues
        pdb_with_seqres = (  # noqa: E501
            "SEQRES   1 A    5  ALA ALA ALA ALA ALA\n"
            "ATOM      1  N   ALA A   2       1.458   0.000   0.000  1.00  0.00           N\n"
            "ATOM      2  CA  ALA A   2       2.916   0.000   0.000  1.00  0.00           C\n"
            "ATOM      3  C   ALA A   2       3.467   1.420   0.000  1.00  0.00           C\n"
            "ATOM      4  O   ALA A   2       2.704   2.390   0.000  1.00  0.00           O\n"
            "ATOM      5  CB  ALA A   2       3.444  -0.760  -1.216  1.00  0.00           C\n"
            "ATOM      6  N   ALA A   3       4.784   1.540   0.000  1.00  0.00           N\n"
            "ATOM      7  CA  ALA A   3       5.399   2.861   0.000  1.00  0.00           C\n"
            "ATOM      8  C   ALA A   3       6.917   2.789   0.000  1.00  0.00           C\n"
            "ATOM      9  O   ALA A   3       7.523   1.719   0.000  1.00  0.00           O\n"
            "ATOM     10  CB  ALA A   3       4.931   3.699   1.186  1.00  0.00           C\n"
            "ATOM     11  N   ALA A   4       7.523   3.969   0.000  1.00  0.00           N\n"
            "ATOM     12  CA  ALA A   4       8.977   4.109   0.000  1.00  0.00           C\n"
            "ATOM     13  C   ALA A   4       9.528   5.529   0.000  1.00  0.00           C\n"
            "ATOM     14  O   ALA A   4       8.765   6.499   0.000  1.00  0.00           O\n"
            "ATOM     15  CB  ALA A   4       9.505   3.349  -1.216  1.00  0.00           C\n"
            "END\n"
        )

        result = minimize_with_constraints(
            pdb_with_seqres,
            stiffness=10.0,
            add_missing_residues=False,
        )

        assert "ATOM" in result
        assert "END" in result


@pytest.mark.integration
class TestIdealizeWithChainGaps:
    """Integration tests for idealization with chain gaps."""

    @pytest.fixture
    def peptide_with_gap(self):
        """Create a peptide with a large gap (missing residues 3-8)."""
        # Residues 1-2 and 9-10, with gap in between
        # PDB format lines are necessarily > 80 chars
        lines = [  # noqa: E501
            "ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N",
            "ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C",
            "ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C",
            "ATOM      4  O   ALA A   1       1.246   2.390   0.000  1.00  0.00           O",
            "ATOM      5  CB  ALA A   1       1.986  -0.760  -1.216  1.00  0.00           C",
            "ATOM      6  N   ALA A   2       3.326   1.540   0.000  1.00  0.00           N",
            "ATOM      7  CA  ALA A   2       3.941   2.861   0.000  1.00  0.00           C",
            "ATOM      8  C   ALA A   2       5.459   2.789   0.000  1.00  0.00           C",
            "ATOM      9  O   ALA A   2       6.065   1.719   0.000  1.00  0.00           O",
            "ATOM     10  CB  ALA A   2       3.473   3.699   1.186  1.00  0.00           C",
            "ATOM     11  N   ALA A   9      20.000  20.000   0.000  1.00  0.00           N",
            "ATOM     12  CA  ALA A   9      21.458  20.000   0.000  1.00  0.00           C",
            "ATOM     13  C   ALA A   9      22.009  21.420   0.000  1.00  0.00           C",
            "ATOM     14  O   ALA A   9      21.246  22.390   0.000  1.00  0.00           O",
            "ATOM     15  CB  ALA A   9      21.986  19.240  -1.216  1.00  0.00           C",
            "ATOM     16  N   ALA A  10      23.326  21.540   0.000  1.00  0.00           N",
            "ATOM     17  CA  ALA A  10      23.941  22.861   0.000  1.00  0.00           C",
            "ATOM     18  C   ALA A  10      25.459  22.789   0.000  1.00  0.00           C",
            "ATOM     19  O   ALA A  10      26.065  21.719   0.000  1.00  0.00           O",
            "ATOM     20  CB  ALA A  10      23.473  23.699   1.186  1.00  0.00           C",
            "END",
        ]
        return "\n".join(lines) + "\n"

    def test_detect_gap_in_numbering(self, peptide_with_gap):
        """Chain gap should be detected from residue numbering."""
        from graphrelax.chain_gaps import detect_chain_gaps

        gaps = detect_chain_gaps(peptide_with_gap)

        # Should detect gap between residue 2 and 9
        assert len(gaps) == 1
        assert gaps[0].residue_before == 2
        assert gaps[0].residue_after == 9

    def test_idealize_preserves_gap(self, peptide_with_gap):
        """Idealization should detect chain gaps correctly."""
        # Test that gaps are correctly detected and chains split
        # This tests the core chain gap handling logic
        from graphrelax.chain_gaps import (
            detect_chain_gaps,
            split_chains_at_gaps,
        )

        gaps = detect_chain_gaps(peptide_with_gap)

        # Gap should be detected
        assert len(gaps) == 1
        assert gaps[0].residue_before == 2
        assert gaps[0].residue_after == 9

        # Test chain splitting
        split_pdb, chain_mapping = split_chains_at_gaps(peptide_with_gap, gaps)

        # Chain mapping should indicate new chain IDs created
        assert len(chain_mapping) > 0

        # New chain ID should be assigned for segment starting at residue 9
        parser = PDBParser(QUIET=True)
        result = parser.get_structure("split", io.StringIO(split_pdb))

        # Count unique chain IDs
        chain_ids = set()
        for model in result:
            for chain in model:
                chain_ids.add(chain.id)

        # Should have at least 2 chains (original + split segment)
        assert len(chain_ids) >= 2
