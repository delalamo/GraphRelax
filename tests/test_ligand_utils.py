"""Unit tests for ligand utilities."""

from graphrelax.ligand_utils import (
    WATER_RESIDUES,
    LigandInfo,
    extract_ligands_from_pdb,
    get_ion_smiles,
    is_single_atom_ligand,
)


class TestExtractLigands:
    """Tests for ligand extraction from PDB."""

    def test_extract_no_ligands(self, small_peptide_pdb_string):
        """Test extraction from protein-only PDB."""
        protein_pdb, ligands = extract_ligands_from_pdb(
            small_peptide_pdb_string
        )

        assert len(ligands) == 0
        assert "ATOM" in protein_pdb
        assert "END" in protein_pdb

    def test_extract_with_ligand(self):
        """Test extraction of a ligand from PDB."""
        # fmt: off
        pdb_with_ligand = (
            "ATOM      1  N   ALA A   1       0.0   0.0   0.0  1.00  0.00\n"
            "ATOM      2  CA  ALA A   1       1.5   0.0   0.0  1.00  0.00\n"
            "ATOM      3  C   ALA A   1       2.0   1.4   0.0  1.00  0.00\n"
            "ATOM      4  O   ALA A   1       1.2   2.4   0.0  1.00  0.00\n"
            "HETATM    5 FE   HEM A 200       5.0   5.0   5.0  1.00  0.00\n"
            "HETATM    6  NA  HEM A 200       3.5   5.0   5.0  1.00  0.00\n"
            "HETATM    7  NB  HEM A 200       5.0   3.5   5.0  1.00  0.00\n"
            "END\n"
        )
        # fmt: on
        protein_pdb, ligands = extract_ligands_from_pdb(pdb_with_ligand)

        assert len(ligands) == 1
        assert ligands[0].resname == "HEM"
        assert ligands[0].chain_id == "A"
        assert ligands[0].resnum == 200
        assert len(ligands[0].pdb_lines) == 3  # FE, NA, NB

        # Check protein PDB doesn't have HETATM
        assert "HETATM" not in protein_pdb
        assert "ATOM" in protein_pdb

    def test_water_excluded(self):
        """Test that water is not extracted as ligand."""
        # fmt: off
        pdb_with_water = (
            "ATOM      1  N   ALA A   1       0.0   0.0   0.0  1.00  0.00\n"
            "ATOM      2  CA  ALA A   1       1.5   0.0   0.0  1.00  0.00\n"
            "HETATM    5  O   HOH A 301      10.0  10.0  10.0  1.00  0.00\n"
            "HETATM    6  O   WAT A 302      11.0  10.0  10.0  1.00  0.00\n"
            "HETATM    7  O   SOL A 303      12.0  10.0  10.0  1.00  0.00\n"
            "END\n"
        )
        # fmt: on
        protein_pdb, ligands = extract_ligands_from_pdb(pdb_with_water)

        # Waters should not be in ligand list
        assert len(ligands) == 0
        assert not any(lig.resname in WATER_RESIDUES for lig in ligands)

    def test_multiple_ligands(self):
        """Test extraction of multiple different ligands."""
        # fmt: off
        pdb_with_multiple = (
            "ATOM      1  N   ALA A   1       0.0   0.0   0.0  1.00  0.00\n"
            "HETATM    5 FE   HEM A 200       5.0   5.0   5.0  1.00  0.00\n"
            "HETATM    6  N1  NAD B 301       8.0   8.0   8.0  1.00  0.00\n"
            "END\n"
        )
        # fmt: on
        protein_pdb, ligands = extract_ligands_from_pdb(pdb_with_multiple)

        assert len(ligands) == 2
        resnames = {lig.resname for lig in ligands}
        assert resnames == {"HEM", "NAD"}


class TestIonSmiles:
    """Tests for ion SMILES lookup."""

    def test_common_ions(self):
        """Test that common metal ions are defined."""
        smiles = get_ion_smiles()
        ions = ["ZN", "MG", "CA", "FE", "MN", "CU"]
        for ion in ions:
            assert ion in smiles, f"{ion} should be in ion SMILES"

    def test_ion_smiles_format(self):
        """Test that ion SMILES have proper format with charge."""
        smiles = get_ion_smiles()
        # All ions should have brackets indicating charged species
        for ion, smi in smiles.items():
            assert smi.startswith("["), f"{ion} SMILES should start with ["
            assert smi.endswith("]"), f"{ion} SMILES should end with ]"

    def test_halide_ions(self):
        """Test that halide ions are defined."""
        smiles = get_ion_smiles()
        assert "CL" in smiles
        assert smiles["CL"] == "[Cl-]"


class TestSingleAtomLigand:
    """Tests for single atom ligand detection."""

    def test_ion_is_single_atom(self):
        """Test that ions are detected as single atoms."""
        ligand = LigandInfo(
            resname="ZN",
            chain_id="A",
            resnum=1,
            pdb_lines=["HETATM    1 ZN   ZN  A   1       0.0   0.0   0.0"],
        )
        assert is_single_atom_ligand(ligand)

    def test_heme_is_not_single_atom(self):
        """Test that multi-atom ligands are not single atoms."""
        ligand = LigandInfo(
            resname="HEM",
            chain_id="A",
            resnum=1,
            pdb_lines=[
                "HETATM    1 FE   HEM A   1       0.0   0.0   0.0",
                "HETATM    2  NA  HEM A   1       1.0   0.0   0.0",
            ],
        )
        assert not is_single_atom_ligand(ligand)


class TestLigandInfo:
    """Tests for LigandInfo dataclass."""

    def test_ligand_info_creation(self):
        """Test creating a LigandInfo object."""
        ligand = LigandInfo(
            resname="HEM",
            chain_id="A",
            resnum=200,
            pdb_lines=[
                "HETATM    5 FE   HEM A 200       5.000   5.000   5.000"
            ],
        )
        assert ligand.resname == "HEM"
        assert ligand.chain_id == "A"
        assert ligand.resnum == 200
        assert ligand.smiles is None

    def test_ligand_info_with_smiles(self):
        """Test LigandInfo with optional SMILES."""
        ligand = LigandInfo(
            resname="BEN",
            chain_id="B",
            resnum=1,
            pdb_lines=[
                "HETATM    1  C1  BEN B   1       0.000   0.000   0.000"
            ],
            smiles="c1ccccc1",
        )
        assert ligand.smiles == "c1ccccc1"


class TestWaterResidues:
    """Tests for water residue constants."""

    def test_common_water_names(self):
        """Test that common water names are included."""
        assert "HOH" in WATER_RESIDUES
        assert "WAT" in WATER_RESIDUES
        assert "SOL" in WATER_RESIDUES

    def test_tip_models(self):
        """Test that TIP water models are included."""
        assert "TIP3" in WATER_RESIDUES
        assert "TIP4" in WATER_RESIDUES
        assert "SPC" in WATER_RESIDUES
