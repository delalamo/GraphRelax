"""Contract tests for the external RFD4 adapter (no GPU or weights needed)."""

import io
import json
from pathlib import Path

import numpy as np
import pytest
from Bio.PDB import MMCIFIO, MMCIFParser, PDBParser
from numpy.testing import assert_allclose, assert_array_equal

from graphrelax import rfd4
from graphrelax.resfile import DesignSpec, ResidueMode, ResidueSpec
from graphrelax.rfd4 import BACKBONE, SIDECHAIN_ATOMS, RFD4Engine


@pytest.fixture
def mutations():
    return {}


@pytest.fixture
def fake_rfd4(monkeypatch, mutations):
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
        for i, residue in enumerate(structure.get_residues(), 1):
            if i not in mutations:
                continue
            residue.resname = mutations[i]
            for atom in list(residue):
                if atom.name not in BACKBONE:
                    residue.detach_child(atom.id)
            for name in SIDECHAIN_ATOMS[residue.resname].split():
                atom = residue["CA"].copy()
                atom.id = atom.name = name
                atom.fullname = name.rjust(4)
                atom.element = name[0]
                atom.coord = atom.coord + np.array([1.0, 1.0, 1.0])
                residue.add(atom)
        output = Path(args["out_dir"]) / "inference_samples" / "repack"
        output.mkdir(parents=True)
        writer = MMCIFIO()
        writer.set_structure(structure)
        writer.save(str(output / "repack_generated.cif"))

    monkeypatch.setattr("graphrelax.rfd4.subprocess.run", run)
    return calls


def test_repacking_preserves_backbone_and_natro(small_peptide_pdb, fake_rfd4):
    spec = DesignSpec({"A2": ResidueSpec("A", 2, mode=ResidueMode.NATRO)})
    result = RFD4Engine(seed=42).repack(small_peptide_pdb, spec)
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
    assert not any(
        "res_id == 2 " in entry["select"]
        for entry in task["conditions"]["C_CRD"]["false"]
    )


def test_default_natro_needs_no_model(small_peptide_pdb, monkeypatch):
    monkeypatch.setattr(
        "graphrelax.rfd4.subprocess.run",
        lambda *a, **kw: pytest.fail("Must not call RFD4"),
    )
    result = RFD4Engine().repack(
        small_peptide_pdb, DesignSpec({}, ResidueMode.NATRO)
    )
    assert result["pdb_string"] == small_peptide_pdb.read_text()


def test_seed_progresses(small_peptide_pdb, fake_rfd4):
    packer = RFD4Engine(seed=7)
    packer.repack(small_peptide_pdb)
    packer.repack(small_peptide_pdb)
    assert [args["seed"] for args, _ in fake_rfd4] == ["7", "8"]


def test_missing_executable(small_peptide_pdb, monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError()

    monkeypatch.setattr("graphrelax.rfd4.subprocess.run", missing)
    with pytest.raises(RuntimeError, match="GRAPHRELAX_RFD4_EXECUTABLE"):
        RFD4Engine().repack(small_peptide_pdb)


def test_missing_output(small_peptide_pdb, monkeypatch):
    monkeypatch.setattr("graphrelax.rfd4.subprocess.run", lambda *a, **kw: None)
    with pytest.raises(RuntimeError, match="Expected one RFD4 structure"):
        RFD4Engine().repack(small_peptide_pdb)


@pytest.mark.parametrize("change", ["sequence", "backbone", "sidechain"])
def test_invalid_output_rejected(
    small_peptide_pdb, monkeypatch, fake_rfd4, change
):
    run = rfd4.subprocess.run

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

    monkeypatch.setattr("graphrelax.rfd4.subprocess.run", corrupt)
    with pytest.raises(RuntimeError):
        RFD4Engine().repack(small_peptide_pdb)


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
    result = RFD4Engine().repack(small_peptide_pdb, spec)
    structure = PDBParser(QUIET=True).get_structure(
        "packed", io.StringIO(result["pdb_string"])
    )
    assert structure[0]["Z"][(" ", 10, "A")].resname == "ALA"
    zinc = next(structure[0]["X"].get_atoms())
    assert_array_equal(zinc.coord, [20, 21, 22])
    selections = fake_rfd4[0][1]["conditions"]["C_CRD"]["false"]
    assert selections == [
        {
            "select": "chain_id == 'A' and res_id == 2 "
            "and not is_protein_backbone"
        }
    ]


@pytest.mark.integration
def test_real_rfd4_repacking(small_peptide_pdb, rfd4_available):
    """Run explicitly on a host with the RFD4 GPU environment and weights."""
    spec = DesignSpec({"A2": ResidueSpec("A", 2, mode=ResidueMode.NATRO)})
    result = RFD4Engine(seed=42).repack(small_peptide_pdb, spec)
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


def test_full_design_generates_sequence_and_atoms(
    small_peptide_pdb, fake_rfd4, mutations
):
    mutations.update({1: "CYS", 3: "GLY"})
    engine = RFD4Engine(seed=7, tsr_k=2, tsr_sigma=8)
    result = engine.design(small_peptide_pdb, design_all=True)
    assert result["sequence"] == "CAGAA"
    assert result["native_sequence"] == "AAAAA"
    parsed = PDBParser(QUIET=True).get_structure(
        "design", io.StringIO(result["pdb_string"])
    )
    assert "SG" in parsed[0]["A"][1]
    assert "CB" not in parsed[0]["A"][3]
    original = PDBParser(QUIET=True).get_structure(
        "input", str(small_peptide_pdb)
    )
    for residue in original.get_residues():
        for atom in residue:
            if atom.name in BACKBONE:
                assert_array_equal(
                    atom.coord, parsed[0]["A"][residue.id][atom.name].coord
                )
    args, task = fake_rfd4[0]
    assert (
        args["generation.model.local_latents.simulation_step_params.tsr_k"]
        == "2"
    )
    assert (
        args["generation.model.local_latents.simulation_step_params.tsr_sigma"]
        == "8"
    )
    assert len(task["conditions"]["C_SEQ"]["false"]) == 5
    assert task["conditions"]["C_SEQ"]["true"] == [
        {"select": "atom_name != ''"}
    ]


@pytest.mark.parametrize(
    "mode,choices,target",
    [
        (ResidueMode.PIKAA, {"C", "S"}, "CYS"),
        (ResidueMode.NOTAA, {"A", "G"}, "CYS"),
        (ResidueMode.POLAR, None, "SER"),
        (ResidueMode.APOLAR, None, "VAL"),
    ],
)
def test_resfile_design_constraints(
    small_peptide_pdb, fake_rfd4, mutations, mode, choices, target
):
    specification = ResidueSpec("A", 1, mode=mode, allowed_aas=choices)
    spec = DesignSpec(
        {"A1": specification, "A2": ResidueSpec("A", 2, mode=ResidueMode.NATRO)}
    )
    mutations[1] = target
    result = RFD4Engine().design(small_peptide_pdb, spec, design_all=True)
    assert result["sequence"][0] in specification.get_allowed_aas()
    assert result["sequence"][1:] == "AAAA"
    conditions = fake_rfd4[0][1]["conditions"]
    selection = {"select": "chain_id == 'A' and res_id == 1"}
    assert selection in conditions["C_SEQ"]["false"]
    assert conditions["C_SEQ"]["false"] == [selection]
    assert not any(
        "res_id == 2 " in item["select"]
        for item in conditions["C_CRD"]["false"]
    )
    excluded = "ASP" if mode == ResidueMode.APOLAR else "ALA"
    assert selection in conditions[f"C_SEQ_PROB_{excluded}"]["0.0"]


@pytest.mark.parametrize(
    "mode,choices",
    [
        (ResidueMode.NATAA, None),
        (ResidueMode.NATRO, None),
        (ResidueMode.PIKAA, {"A"}),
        (ResidueMode.NOTAA, {"C"}),
        (ResidueMode.POLAR, None),
    ],
)
def test_design_rejects_disallowed_identity(
    small_peptide_pdb, fake_rfd4, mutations, mode, choices
):
    mutations[1] = "CYS"
    spec = DesignSpec(
        {"A1": ResidueSpec("A", 1, mode=mode, allowed_aas=choices)}
    )
    with pytest.raises(RuntimeError, match="sequence constraints"):
        RFD4Engine().design(small_peptide_pdb, spec)


def test_repack_ignores_design_permissions(
    small_peptide_pdb, fake_rfd4, mutations
):
    mutations[1] = "CYS"
    with pytest.raises(RuntimeError, match="sequence constraints"):
        RFD4Engine().repack(
            small_peptide_pdb, DesignSpec({}, ResidueMode.ALLAA)
        )


@pytest.mark.integration
def test_real_rfd4_design(small_peptide_pdb, rfd4_available):
    spec = DesignSpec(
        {
            "A1": ResidueSpec(
                "A", 1, mode=ResidueMode.PIKAA, allowed_aas={"C"}
            ),
            "A2": ResidueSpec("A", 2, mode=ResidueMode.NATRO),
        }
    )
    result = RFD4Engine(seed=42).design(small_peptide_pdb, spec)
    assert result["sequence"] == "CAAAA"
    assert result["native_sequence"] == "AAAAA"
    assert "SG" in result["pdb_string"]


@pytest.mark.parametrize("design", [True, False])
def test_pipeline_uses_rfd4_for_both_modes(
    small_peptide_pdb, tmp_path, fake_rfd4, mutations, design
):
    from graphrelax.config import PipelineConfig, PipelineMode
    from graphrelax.pipeline import Pipeline

    if design:
        mutations[1] = "CYS"
    mode = PipelineMode.DESIGN_ONLY if design else PipelineMode.REPACK_ONLY
    output = tmp_path / "output.pdb"
    scorefile = tmp_path / "scores.sc"
    result = Pipeline(
        PipelineConfig(mode=mode, n_iterations=2, scorefile=scorefile)
    ).run(small_peptide_pdb, output)
    assert len(fake_rfd4) == 2
    assert result["outputs"][0]["sequence"] == ("CAAAA" if design else "AAAAA")
    assert result["outputs"][0]["native_sequence"] == "AAAAA"
    assert result["scores"][0]["seq_recovery"] == (0.8 if design else 1.0)
    assert set(result["scores"][0]) == {
        "total_score",
        "openmm_energy",
        "seq_recovery",
        "description",
    }
    assert output.exists() and scorefile.exists()


def test_designer_interface(small_peptide_pdb, fake_rfd4, mutations):
    from graphrelax.config import DesignConfig
    from graphrelax.designer import Designer

    designer = Designer(DesignConfig(seed=5))
    mutations[1] = "CYS"
    result = designer.design(small_peptide_pdb, design_all=True)
    assert result["sequence"] == "CAAAA"
    assert designer.result_to_pdb_string(result) == result["pdb_string"]
    assert set(result) == {"pdb_string", "sequence", "native_sequence"}
