"""Contract tests for the external RFD4 adapter (no GPU or weights needed)."""

import io
import json
from pathlib import Path

import numpy as np
import pytest
from Bio.PDB import MMCIFIO, MMCIFParser, PDBParser
from numpy.testing import assert_allclose, assert_array_equal

from graphrelax import repacker
from graphrelax.repacker import Repacker
from graphrelax.resfile import DesignSpec, ResidueMode, ResidueSpec


@pytest.fixture
def fake_rfd4(monkeypatch):
    calls = []

    def run(command, check, cwd):
        args = dict(arg.split("=", 1) for arg in command[1:])
        task = json.loads(Path(args["inputs"]).read_text())["repack"]
        calls.append((args, task))
        structure = MMCIFParser(QUIET=True).get_structure(
            "sample", task["input"]
        )
        for atom in structure.get_atoms():
            atom.coord += np.array([10, 20, 30])
            if atom.name == "CB":
                atom.coord += np.array([0.2, 0.3, 0.4])
        output = Path(args["out_dir"]) / "inference_samples" / "repack"
        output.mkdir(parents=True)
        writer = MMCIFIO()
        writer.set_structure(structure)
        writer.save(str(output / "repack_generated.cif"))

    monkeypatch.setattr("graphrelax.repacker.subprocess.run", run)
    return calls


def test_repacking_preserves_backbone_and_natro(small_peptide_pdb, fake_rfd4):
    spec = DesignSpec({"A2": ResidueSpec("A", 2, mode=ResidueMode.NATRO)})
    result = Repacker(seed=42).repack(small_peptide_pdb, spec)
    before = PDBParser(QUIET=True).get_structure(
        "before", str(small_peptide_pdb)
    )
    after = PDBParser(QUIET=True).get_structure(
        "after", io.StringIO(result["pdb_string"])
    )
    assert result["sequence"] == result["native_sequence"] == "AAAAA"
    for original in before.get_atoms():
        packed = after[0][original.parent.parent.id][original.parent.id][
            original.id
        ]
        if original.name != "CB" or original.parent.id[1] == 2:
            assert_array_equal(original.coord, packed.coord)
        else:
            assert_allclose(
                packed.coord - original.coord, [0.2, 0.3, 0.4], atol=0.001
            )
    args, task = fake_rfd4[0]
    assert args["generation"] == "inverse_folding"
    assert args["seed"] == "42"
    assert task["conditions"]["C_SEQ"]
    assert {"select": "chain_id == 'A' and res_id == 2"} in task["conditions"][
        "C_CRD"
    ]["true"]


def test_default_natro_needs_no_model(small_peptide_pdb, monkeypatch):
    monkeypatch.setattr(
        "graphrelax.repacker.subprocess.run",
        lambda *a, **kw: pytest.fail("Must not call RFD4"),
    )
    result = Repacker().repack(
        small_peptide_pdb, DesignSpec({}, ResidueMode.NATRO)
    )
    assert result["pdb_string"] == small_peptide_pdb.read_text()


def test_seed_progresses(small_peptide_pdb, fake_rfd4):
    packer = Repacker(seed=7)
    packer.repack(small_peptide_pdb)
    packer.repack(small_peptide_pdb)
    assert [args["seed"] for args, _ in fake_rfd4] == ["7", "8"]


def test_missing_executable(small_peptide_pdb, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr("graphrelax.repacker.subprocess.run", missing)
    with pytest.raises(RuntimeError, match="GRAPHRELAX_RFD4_EXECUTABLE"):
        Repacker().repack(small_peptide_pdb)


def test_missing_output(small_peptide_pdb, monkeypatch):
    monkeypatch.setattr(
        "graphrelax.repacker.subprocess.run", lambda *a, **kw: None
    )
    with pytest.raises(RuntimeError, match="Expected one RFD4 structure"):
        Repacker().repack(small_peptide_pdb)


@pytest.mark.parametrize("change", ["sequence", "backbone", "sidechain"])
def test_invalid_output_rejected(
    small_peptide_pdb, monkeypatch, fake_rfd4, change
):
    run = repacker.subprocess.run

    def corrupt(command, **kwargs):
        run(command, **kwargs)
        root = Path(kwargs["cwd"]) / "output" / "inference_samples"
        output = next(root.rglob("*_generated.cif"))
        structure = MMCIFParser(QUIET=True).get_structure("bad", str(output))
        residue = next(structure.get_residues())
        if change == "sequence":
            residue.resname = "GLY"
        elif change == "backbone":
            residue["CA"].coord += 5
        else:
            residue.detach_child("CB")
        writer = MMCIFIO()
        writer.set_structure(structure)
        writer.save(str(output))

    monkeypatch.setattr("graphrelax.repacker.subprocess.run", corrupt)
    with pytest.raises(RuntimeError):
        Repacker().repack(small_peptide_pdb)


def test_author_ids_and_ligand_preserved(small_peptide_pdb, fake_rfd4):
    text = small_peptide_pdb.read_text()
    lines = []
    for line in text.splitlines():
        if line.startswith("ATOM"):
            number = int(line[22:26])
            line = line[:21] + "Z" + f"{number + 9:4d}" + "A" + line[27:]
        if line == "END":
            lines.append(
                "HETATM   27 ZN    ZN X 100      20.000  21.000  22.000"
                "  1.00  0.00          ZN"
            )
        lines.append(line)
    small_peptide_pdb.write_text("\n".join(lines) + "\n")
    spec = DesignSpec(
        {"Z11A": ResidueSpec("Z", 11, "A", ResidueMode.NATAA)},
        ResidueMode.NATRO,
    )
    result = Repacker().repack(small_peptide_pdb, spec)
    structure = PDBParser(QUIET=True).get_structure(
        "packed", io.StringIO(result["pdb_string"])
    )
    assert structure[0]["Z"][(" ", 10, "A")].resname == "ALA"
    zinc = next(structure[0]["X"].get_atoms())
    assert_array_equal(zinc.coord, [20, 21, 22])
    selections = fake_rfd4[0][1]["conditions"]["C_CRD"]["true"]
    assert {"select": "chain_id == 'A' and res_id == 1"} in selections
    assert {"select": "chain_id == 'A' and res_id == 2"} not in selections


def test_designer_repack_does_not_load_ligandmpnn(
    small_peptide_pdb, fake_rfd4, monkeypatch
):
    from graphrelax.config import DesignConfig
    from graphrelax.designer import Designer

    designer = Designer(DesignConfig())
    monkeypatch.setattr(
        designer, "_load_models", lambda: pytest.fail("Loaded LigandMPNN")
    )
    result = designer.repack(small_peptide_pdb)
    assert designer._model is None
    assert "ATOM" in designer.result_to_pdb_string(result)


def test_design_backbone_serialization(small_peptide_pdb):
    from prody import parsePDB

    from graphrelax.config import DesignConfig
    from graphrelax.designer import Designer

    backbone = parsePDB(str(small_peptide_pdb)).select("name N CA C O")
    designer = Designer(DesignConfig(pack_side_chains=False))
    output = designer.result_to_pdb_string(
        {
            "sequence": "ACDEF",
            "backbone": backbone,
            "other_atoms": None,
        }
    )
    parsed = PDBParser(QUIET=True).get_structure(
        "designed", io.StringIO(output)
    )
    assert [r.resname for r in parsed.get_residues()] == [
        "ALA",
        "CYS",
        "ASP",
        "GLU",
        "PHE",
    ]


def test_design_natro_restored_before_packing(
    small_peptide_pdb, tmp_path, fake_rfd4
):
    backbone_path = tmp_path / "backbone.pdb"
    backbone_path.write_text(
        "\n".join(
            line
            for line in small_peptide_pdb.read_text().splitlines()
            if line[12:16].strip() != "CB"
        )
        + "\n"
    )
    spec = DesignSpec({"A2": ResidueSpec("A", 2, mode=ResidueMode.NATRO)})
    # Fake generation needs to supply the missing CB atoms of free residues.
    # Keep all residues fixed here to isolate native-context restoration.
    spec.default_mode = ResidueMode.NATRO
    result = Repacker().repack(
        backbone_path, spec, original_path=small_peptide_pdb
    )
    parsed = PDBParser(QUIET=True).get_structure(
        "packed", io.StringIO(result["pdb_string"])
    )
    assert all("CB" in residue for residue in parsed.get_residues())


@pytest.mark.integration
def test_real_rfd4_repacking(small_peptide_pdb, rfd4_available):
    """Run explicitly on a host with the RFD4 GPU environment and weights."""
    spec = DesignSpec({"A2": ResidueSpec("A", 2, mode=ResidueMode.NATRO)})
    result = Repacker(seed=42).repack(small_peptide_pdb, spec)
    assert result["sequence"] == result["native_sequence"] == "AAAAA"
    before = PDBParser(QUIET=True).get_structure(
        "input", str(small_peptide_pdb)
    )
    after = PDBParser(QUIET=True).get_structure(
        "output", io.StringIO(result["pdb_string"])
    )
    for original in before.get_atoms():
        if (
            original.name in {"N", "CA", "C", "O", "OXT"}
            or original.parent.id[1] == 2
        ):
            packed = after[0][original.parent.parent.id][original.parent.id][
                original.id
            ]
            assert_array_equal(original.coord, packed.coord)
