"""Shared fixtures for GraphRelax tests."""

import urllib.error
import urllib.request
from pathlib import Path

import pytest

from graphrelax.structure_io import convert_pdb_to_cif


@pytest.fixture(scope="session")
def weights_available():
    """
    Check if LigandMPNN weights are available or can be downloaded.

    Skips tests if weights don't exist and cannot be downloaded due to
    network restrictions.
    """
    from graphrelax.weights import weights_exist

    if weights_exist():
        return True

    # Try a quick connectivity check to the weights server
    test_url = "https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_020.pt"
    try:
        # Just check if we can connect (don't download)
        urllib.request.urlopen(test_url, timeout=5)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        pytest.skip(
            "LigandMPNN weights not available and cannot be downloaded. "
            "Tests requiring weights will be skipped."
        )


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


@pytest.fixture
def small_peptide_cif_string(small_peptide_pdb_string):
    """A minimal 5-residue alanine peptide CIF string for testing."""
    return convert_pdb_to_cif(small_peptide_pdb_string)


@pytest.fixture
def small_peptide_cif(tmp_path, small_peptide_cif_string):
    """Create a small peptide CIF file for testing."""
    path = tmp_path / "small_peptide.cif"
    path.write_text(small_peptide_cif_string)
    return path


@pytest.fixture(scope="session")
def ubiquitin_pdb(tmp_path_factory):
    """
    Download 1UBQ (ubiquitin, 76 residues) for integration testing.

    This is a realistic test case with a real protein structure.
    The file is cached for the session to avoid repeated downloads.

    The PDB is cleaned to remove:
    - HETATM records (water molecules, ligands)
    - Alternate conformations (keep only 'A' or first conformer)
    - Non-protein records

    This makes it compatible with the OpenFold relaxation pipeline.
    """
    import urllib.error
    import urllib.request

    cache_dir = tmp_path_factory.mktemp("pdb_cache")
    raw_pdb_path = cache_dir / "1ubq_raw.pdb"
    pdb_path = cache_dir / "1ubq.pdb"

    url = "https://files.rcsb.org/download/1UBQ.pdb"
    try:
        urllib.request.urlretrieve(url, raw_pdb_path)
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        pytest.skip(f"Cannot download PDB file from RCSB: {e}")

    # Clean the PDB file for compatibility with OpenFold relaxation
    clean_lines = []
    with open(raw_pdb_path) as f:
        for line in f:
            # Skip HETATM records (water, ligands)
            if line.startswith("HETATM"):
                continue
            # Keep ATOM records
            if line.startswith("ATOM"):
                # Check alternate location indicator (column 17)
                alt_loc = line[16] if len(line) > 16 else " "
                # Keep only first conformer (blank or 'A')
                if alt_loc not in (" ", "A"):
                    continue
                # Remove alt loc indicator for consistency
                if alt_loc == "A":
                    line = line[:16] + " " + line[17:]
                clean_lines.append(line)
            # Keep essential records
            elif line.startswith(("TER", "END", "MODEL", "ENDMDL")):
                clean_lines.append(line)

    with open(pdb_path, "w") as f:
        f.writelines(clean_lines)

    return pdb_path


@pytest.fixture(scope="session")
def ubiquitin_cif(tmp_path_factory):
    """
    Download 1UBQ (ubiquitin) in CIF format for integration testing.

    Same cleaning as ubiquitin_pdb but in CIF format.
    """
    import urllib.error
    import urllib.request

    from Bio.PDB import MMCIFIO, MMCIFParser, Select

    cache_dir = tmp_path_factory.mktemp("cif_cache")
    raw_cif_path = cache_dir / "1ubq_raw.cif"
    cif_path = cache_dir / "1ubq.cif"

    url = "https://files.rcsb.org/download/1UBQ.cif"
    try:
        urllib.request.urlretrieve(url, raw_cif_path)
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        pytest.skip(f"Cannot download CIF file from RCSB: {e}")

    # Clean the CIF file using BioPython
    class CleanSelect(Select):
        def accept_residue(self, residue):
            # Skip water and other heteroatoms
            hetflag = residue.id[0]
            if hetflag.startswith("H_") or hetflag == "W":
                return False
            return True

        def accept_atom(self, atom):
            # Keep only first conformer
            if atom.altloc not in (" ", "A"):
                return False
            return True

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("1ubq", str(raw_cif_path))

    cif_io = MMCIFIO()
    cif_io.set_structure(structure)
    cif_io.save(str(cif_path), CleanSelect())

    return cif_path


@pytest.fixture(scope="session")
def heme_protein_pdb(tmp_path_factory):
    """
    Download 8VC8 (heme-loaded designed protein) for integration testing.

    This structure contains:
    - A designed protein (single chain A)
    - A heme group (HEM)
    - Sulfate ions (SO4)
    - Crystal waters (HOH)

    This tests that the pipeline handles non-protein molecules correctly.
    The PDB is cleaned to remove waters and ions but preserves the heme.
    """
    import urllib.error
    import urllib.request

    cache_dir = tmp_path_factory.mktemp("pdb_cache")
    raw_pdb_path = cache_dir / "8vc8_raw.pdb"
    pdb_path = cache_dir / "8vc8.pdb"

    url = "https://files.rcsb.org/download/8VC8.pdb"
    try:
        urllib.request.urlretrieve(url, raw_pdb_path)
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        pytest.skip(f"Cannot download PDB file from RCSB: {e}")

    # Clean the PDB file:
    # - Keep ATOM records (protein)
    # - Keep HETATM for HEM (heme group) - this is the key ligand
    # - Remove water (HOH), sulfate (SO4), and other small molecules
    clean_lines = []
    with open(raw_pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                # Check alternate location indicator (column 17)
                alt_loc = line[16] if len(line) > 16 else " "
                if alt_loc not in (" ", "A"):
                    continue
                if alt_loc == "A":
                    line = line[:16] + " " + line[17:]
                clean_lines.append(line)
            elif line.startswith("HETATM"):
                # Keep heme group, skip water and ions
                res_name = line[17:20].strip()
                if res_name == "HEM":
                    clean_lines.append(line)
            elif line.startswith(("TER", "END", "MODEL", "ENDMDL")):
                clean_lines.append(line)

    with open(pdb_path, "w") as f:
        f.writelines(clean_lines)

    return pdb_path
