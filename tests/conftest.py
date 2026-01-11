"""Shared fixtures for GraphRelax tests."""

from pathlib import Path

import pytest


@pytest.fixture
def test_data_dir():
    """Return the path to the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_resfile_content():
    """Sample resfile content for testing."""
    return """# Test resfile
NATAA
START
10 A ALLAA
15 A PIKAA HYW
20 A NOTAA CP
25 A POLAR
30 A APOLAR
40 A NATRO
"""


@pytest.fixture
def sample_resfile(tmp_path, sample_resfile_content):
    """Create a sample resfile for testing."""
    path = tmp_path / "test.resfile"
    path.write_text(sample_resfile_content)
    return path


@pytest.fixture
def small_peptide_pdb_string():
    """A minimal 5-residue alanine peptide PDB string for testing."""
    # fmt: off
    return """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.458   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.009   1.420   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       1.246   2.390   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.986  -0.760  -1.216  1.00  0.00           C
ATOM      6  N   ALA A   2       3.326   1.540   0.000  1.00  0.00           N
ATOM      7  CA  ALA A   2       3.941   2.861   0.000  1.00  0.00           C
ATOM      8  C   ALA A   2       5.459   2.789   0.000  1.00  0.00           C
ATOM      9  O   ALA A   2       6.065   1.719   0.000  1.00  0.00           O
ATOM     10  CB  ALA A   2       3.473   3.699   1.186  1.00  0.00           C
ATOM     11  N   ALA A   3       6.063   3.970   0.000  1.00  0.00           N
ATOM     12  CA  ALA A   3       7.510   4.096   0.000  1.00  0.00           C
ATOM     13  C   ALA A   3       8.061   5.516   0.000  1.00  0.00           C
ATOM     14  O   ALA A   3       7.298   6.486   0.000  1.00  0.00           O
ATOM     15  CB  ALA A   3       8.038   3.336  -1.216  1.00  0.00           C
ATOM     16  N   ALA A   4       9.378   5.636   0.000  1.00  0.00           N
ATOM     17  CA  ALA A   4       9.993   6.957   0.000  1.00  0.00           C
ATOM     18  C   ALA A   4      11.511   6.885   0.000  1.00  0.00           C
ATOM     19  O   ALA A   4      12.117   5.815   0.000  1.00  0.00           O
ATOM     20  CB  ALA A   4       9.525   7.795   1.186  1.00  0.00           C
ATOM     21  N   ALA A   5      12.115   8.066   0.000  1.00  0.00           N
ATOM     22  CA  ALA A   5      13.562   8.192   0.000  1.00  0.00           C
ATOM     23  C   ALA A   5      14.113   9.612   0.000  1.00  0.00           C
ATOM     24  O   ALA A   5      13.350  10.582   0.000  1.00  0.00           O
ATOM     25  CB  ALA A   5      14.090   7.432  -1.216  1.00  0.00           C
ATOM     26  OXT ALA A   5      15.350   9.732   0.000  1.00  0.00           O
END
"""  # noqa: E501
    # fmt: on


@pytest.fixture
def small_peptide_pdb(tmp_path, small_peptide_pdb_string):
    """Create a small peptide PDB file for testing."""
    path = tmp_path / "small_peptide.pdb"
    path.write_text(small_peptide_pdb_string)
    return path


@pytest.fixture(scope="session")
def ubiquitin_pdb(tmp_path_factory):
    """
    Download 1UBQ (ubiquitin, 76 residues) for integration testing.

    This is a realistic test case with a real protein structure.
    The file is cached for the session to avoid repeated downloads.
    """
    import urllib.request

    cache_dir = tmp_path_factory.mktemp("pdb_cache")
    pdb_path = cache_dir / "1ubq.pdb"

    url = "https://files.rcsb.org/download/1UBQ.pdb"
    urllib.request.urlretrieve(url, pdb_path)

    return pdb_path
