"""Tests for graphrelax.chain_gaps module."""

import pytest

from graphrelax.chain_gaps import (
    ChainGap,
    add_ter_records_at_gaps,
    detect_chain_gaps,
    get_gap_summary,
    restore_chain_ids,
    split_chains_at_gaps,
)


@pytest.fixture
def continuous_peptide_pdb():
    """A continuous peptide with no gaps (residues 1-5)."""
    # fmt: off
    return """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.246   2.390   0.000  1.00  0.00           O
ATOM      5  N   ALA A   2       3.326   1.540   0.000  1.00  0.00           N
ATOM      6  CA  ALA A   2       3.941   2.861   0.000  1.00  0.00           C
ATOM      7  C   ALA A   2       5.459   2.789   0.000  1.00  0.00           C
ATOM      8  O   ALA A   2       6.065   1.719   0.000  1.00  0.00           O
ATOM      9  N   ALA A   3       6.063   3.970   0.000  1.00  0.00           N
ATOM     10  CA  ALA A   3       7.510   4.096   0.000  1.00  0.00           C
ATOM     11  C   ALA A   3       8.061   5.516   0.000  1.00  0.00           C
ATOM     12  O   ALA A   3       7.298   6.486   0.000  1.00  0.00           O
ATOM     13  N   ALA A   4       9.378   5.636   0.000  1.00  0.00           N
ATOM     14  CA  ALA A   4       9.993   6.957   0.000  1.00  0.00           C
ATOM     15  C   ALA A   4      11.511   6.885   0.000  1.00  0.00           C
ATOM     16  O   ALA A   4      12.117   5.815   0.000  1.00  0.00           O
ATOM     17  N   ALA A   5      12.115   8.066   0.000  1.00  0.00           N
ATOM     18  CA  ALA A   5      13.562   8.192   0.000  1.00  0.00           C
ATOM     19  C   ALA A   5      14.113   9.612   0.000  1.00  0.00           C
ATOM     20  O   ALA A   5      13.350  10.582   0.000  1.00  0.00           O
END
"""  # noqa: E501
    # fmt: on


@pytest.fixture
def gapped_peptide_pdb():
    """A peptide with a gap - residues 1, 2, 5, 6, 7 (missing 3-4)."""
    # fmt: off
    return """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.246   2.390   0.000  1.00  0.00           O
ATOM      5  N   ALA A   2       3.326   1.540   0.000  1.00  0.00           N
ATOM      6  CA  ALA A   2       3.941   2.861   0.000  1.00  0.00           C
ATOM      7  C   ALA A   2       5.459   2.789   0.000  1.00  0.00           C
ATOM      8  O   ALA A   2       6.065   1.719   0.000  1.00  0.00           O
ATOM      9  N   ALA A   5      12.115   8.066   0.000  1.00  0.00           N
ATOM     10  CA  ALA A   5      13.562   8.192   0.000  1.00  0.00           C
ATOM     11  C   ALA A   5      14.113   9.612   0.000  1.00  0.00           C
ATOM     12  O   ALA A   5      13.350  10.582   0.000  1.00  0.00           O
ATOM     13  N   ALA A   6      15.115  10.066   0.000  1.00  0.00           N
ATOM     14  CA  ALA A   6      16.562  10.192   0.000  1.00  0.00           C
ATOM     15  C   ALA A   6      17.113  11.612   0.000  1.00  0.00           C
ATOM     16  O   ALA A   6      16.350  12.582   0.000  1.00  0.00           O
ATOM     17  N   ALA A   7      18.115  12.066   0.000  1.00  0.00           N
ATOM     18  CA  ALA A   7      19.562  12.192   0.000  1.00  0.00           C
ATOM     19  C   ALA A   7      20.113  13.612   0.000  1.00  0.00           C
ATOM     20  O   ALA A   7      19.350  14.582   0.000  1.00  0.00           O
END
"""  # noqa: E501
    # fmt: on


@pytest.fixture
def multi_gap_pdb():
    """A peptide with multiple gaps - residues 1, 5, 10."""
    # fmt: off
    return """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.246   2.390   0.000  1.00  0.00           O
ATOM      5  N   ALA A   5      12.115   8.066   0.000  1.00  0.00           N
ATOM      6  CA  ALA A   5      13.562   8.192   0.000  1.00  0.00           C
ATOM      7  C   ALA A   5      14.113   9.612   0.000  1.00  0.00           C
ATOM      8  O   ALA A   5      13.350  10.582   0.000  1.00  0.00           O
ATOM      9  N   ALA A  10      22.115  18.066   0.000  1.00  0.00           N
ATOM     10  CA  ALA A  10      23.562  18.192   0.000  1.00  0.00           C
ATOM     11  C   ALA A  10      24.113  19.612   0.000  1.00  0.00           C
ATOM     12  O   ALA A  10      23.350  20.582   0.000  1.00  0.00           O
END
"""  # noqa: E501
    # fmt: on


@pytest.fixture
def multi_chain_gapped_pdb():
    """A multi-chain structure with a gap in chain A and continuous chain B."""
    # fmt: off
    return """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.246   2.390   0.000  1.00  0.00           O
ATOM      5  N   ALA A   5      12.115   8.066   0.000  1.00  0.00           N
ATOM      6  CA  ALA A   5      13.562   8.192   0.000  1.00  0.00           C
ATOM      7  C   ALA A   5      14.113   9.612   0.000  1.00  0.00           C
ATOM      8  O   ALA A   5      13.350  10.582   0.000  1.00  0.00           O
TER
ATOM      9  N   ALA B   1      30.000   0.000   0.000  1.00  0.00           N
ATOM     10  CA  ALA B   1      31.458   0.000   0.000  1.00  0.00           C
ATOM     11  C   ALA B   1      32.009   1.420   0.000  1.00  0.00           C
ATOM     12  O   ALA B   1      31.246   2.390   0.000  1.00  0.00           O
ATOM     13  N   ALA B   2      33.326   1.540   0.000  1.00  0.00           N
ATOM     14  CA  ALA B   2      33.941   2.861   0.000  1.00  0.00           C
ATOM     15  C   ALA B   2      35.459   2.789   0.000  1.00  0.00           C
ATOM     16  O   ALA B   2      36.065   1.719   0.000  1.00  0.00           O
END
"""  # noqa: E501
    # fmt: on


class TestDetectChainGaps:
    """Tests for detect_chain_gaps function."""

    def test_no_gaps_in_continuous_peptide(self, continuous_peptide_pdb):
        """Test that no gaps are detected in a continuous peptide."""
        gaps = detect_chain_gaps(continuous_peptide_pdb, check_distance=False)
        assert len(gaps) == 0

    def test_detects_single_gap(self, gapped_peptide_pdb):
        """Test detection of a single gap."""
        gaps = detect_chain_gaps(gapped_peptide_pdb, check_distance=False)
        assert len(gaps) == 1
        assert gaps[0].chain_id == "A"
        assert gaps[0].residue_before == 2
        assert gaps[0].residue_after == 5

    def test_detects_multiple_gaps(self, multi_gap_pdb):
        """Test detection of multiple gaps."""
        gaps = detect_chain_gaps(multi_gap_pdb, check_distance=False)
        assert len(gaps) == 2

        # First gap: 1 -> 5
        assert gaps[0].chain_id == "A"
        assert gaps[0].residue_before == 1
        assert gaps[0].residue_after == 5

        # Second gap: 5 -> 10
        assert gaps[1].chain_id == "A"
        assert gaps[1].residue_before == 5
        assert gaps[1].residue_after == 10

    def test_detects_gap_in_specific_chain(self, multi_chain_gapped_pdb):
        """Test that gaps are correctly attributed to their chains."""
        gaps = detect_chain_gaps(multi_chain_gapped_pdb, check_distance=False)

        # Only chain A has a gap
        assert len(gaps) == 1
        assert gaps[0].chain_id == "A"

    def test_distance_check_finds_large_cn_distance(self, gapped_peptide_pdb):
        """Test that distance checking finds large C-N distances."""
        gaps = detect_chain_gaps(gapped_peptide_pdb, check_distance=True)

        # Should find the gap
        assert len(gaps) >= 1
        # The gap should have a distance measurement
        gap = gaps[0]
        assert gap.distance is None or gap.distance > 2.0


class TestChainGap:
    """Tests for ChainGap dataclass."""

    def test_str_representation(self):
        """Test string representation of ChainGap."""
        gap = ChainGap(
            chain_id="A",
            residue_before=10,
            residue_after=15,
            distance=5.5,
        )
        s = str(gap)
        assert "A" in s
        assert "4 residues" in s  # 15 - 10 - 1 = 4 missing
        assert "10" in s
        assert "15" in s
        assert "5.50" in s

    def test_str_without_distance(self):
        """Test string representation without distance."""
        gap = ChainGap(
            chain_id="B",
            residue_before=1,
            residue_after=5,
        )
        s = str(gap)
        assert "B" in s
        assert "3 residues" in s
        assert "dist=" not in s


class TestSplitChainsAtGaps:
    """Tests for split_chains_at_gaps function."""

    def test_no_split_without_gaps(self, continuous_peptide_pdb):
        """Test that continuous peptide is not split."""
        result, mapping = split_chains_at_gaps(continuous_peptide_pdb)
        assert result == continuous_peptide_pdb
        assert mapping == {}

    def test_splits_at_single_gap(self, gapped_peptide_pdb):
        """Test splitting at a single gap."""
        result, mapping = split_chains_at_gaps(gapped_peptide_pdb)

        # Should have new chain ID for segment after gap
        assert len(mapping) > 0

        # Check that atoms after gap have new chain ID
        lines = result.split("\n")
        chains_found = set()
        for line in lines:
            if line.startswith("ATOM") and len(line) > 21:
                chains_found.add(line[21])

        # Should have at least 2 chains (original A + new segment)
        assert len(chains_found) >= 2

    def test_splits_at_multiple_gaps(self, multi_gap_pdb):
        """Test splitting at multiple gaps."""
        result, mapping = split_chains_at_gaps(multi_gap_pdb)

        # Count unique chains in result
        lines = result.split("\n")
        chains_found = set()
        for line in lines:
            if line.startswith("ATOM") and len(line) > 21:
                chains_found.add(line[21])

        # Should have 3 segments (original + 2 new)
        assert len(chains_found) == 3

    def test_preserves_atom_coordinates(self, gapped_peptide_pdb):
        """Test that splitting preserves atom coordinates."""
        result, _ = split_chains_at_gaps(gapped_peptide_pdb)

        # Count atoms in original and result
        orig_atoms = sum(
            1
            for line in gapped_peptide_pdb.split("\n")
            if line.startswith("ATOM")
        )
        result_atoms = sum(
            1 for line in result.split("\n") if line.startswith("ATOM")
        )

        assert orig_atoms == result_atoms

    def test_mapping_tracks_original_chain(self, gapped_peptide_pdb):
        """Test that mapping correctly tracks original chain IDs."""
        _, mapping = split_chains_at_gaps(gapped_peptide_pdb)

        # All mapped chains should map back to A
        for orig_chain in mapping.values():
            assert orig_chain == "A"


class TestRestoreChainIds:
    """Tests for restore_chain_ids function."""

    def test_no_change_without_mapping(self, gapped_peptide_pdb):
        """Test that PDB is unchanged without mapping."""
        result = restore_chain_ids(gapped_peptide_pdb, {})
        assert result == gapped_peptide_pdb

    def test_restores_original_chain_ids(self, gapped_peptide_pdb):
        """Test that original chain IDs are restored after splitting."""
        # Split chains
        split_pdb, mapping = split_chains_at_gaps(gapped_peptide_pdb)

        # Restore original chain IDs
        restored = restore_chain_ids(split_pdb, mapping)

        # All atoms should be in chain A again
        for line in restored.split("\n"):
            if line.startswith("ATOM") and len(line) > 21:
                assert line[21] == "A"

    def test_roundtrip_preserves_atoms(self, gapped_peptide_pdb):
        """Test split -> restore roundtrip preserves atom count."""
        orig_atoms = sum(
            1
            for line in gapped_peptide_pdb.split("\n")
            if line.startswith("ATOM")
        )

        split_pdb, mapping = split_chains_at_gaps(gapped_peptide_pdb)
        restored = restore_chain_ids(split_pdb, mapping)

        restored_atoms = sum(
            1 for line in restored.split("\n") if line.startswith("ATOM")
        )

        assert orig_atoms == restored_atoms


class TestAddTerRecordsAtGaps:
    """Tests for add_ter_records_at_gaps function."""

    def test_no_ter_without_gaps(self, continuous_peptide_pdb):
        """Test no TER records added to continuous peptide."""
        gaps = detect_chain_gaps(continuous_peptide_pdb, check_distance=False)
        result = add_ter_records_at_gaps(continuous_peptide_pdb, gaps)
        assert result == continuous_peptide_pdb

    def test_adds_ter_at_gap(self, gapped_peptide_pdb):
        """Test TER record is added at gap location."""
        gaps = detect_chain_gaps(gapped_peptide_pdb, check_distance=False)
        result = add_ter_records_at_gaps(gapped_peptide_pdb, gaps)

        # Count TER records
        ter_count = sum(
            1 for line in result.split("\n") if line.startswith("TER")
        )
        assert ter_count >= 1


class TestGetGapSummary:
    """Tests for get_gap_summary function."""

    def test_no_gaps_message(self):
        """Test message when no gaps detected."""
        summary = get_gap_summary([])
        assert "No chain gaps detected" in summary

    def test_summary_with_gaps(self):
        """Test summary includes gap information."""
        gaps = [
            ChainGap(chain_id="A", residue_before=10, residue_after=15),
            ChainGap(chain_id="B", residue_before=5, residue_after=20),
        ]
        summary = get_gap_summary(gaps)

        assert "2 chain gap(s)" in summary
        assert "Chain A" in summary
        assert "Chain B" in summary
