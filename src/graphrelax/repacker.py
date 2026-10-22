"""RFD4-Proteina side-chain packing through its public generation CLI."""

import io
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from Bio.PDB import MMCIFIO, PDBIO, MMCIFParser, PDBParser, Superimposer
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.SeqUtils import seq1

from graphrelax.resfile import DesignSpec

BACKBONE = {"N", "CA", "C", "O", "OXT"}
AMINO_ACIDS = set(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET "
    "PHE PRO SER THR TRP TYR VAL".split()
)

SIDECHAIN_ATOMS = {
    "ALA": "CB",
    "ARG": "CB CG CD NE CZ NH1 NH2",
    "ASN": "CB CG OD1 ND2",
    "ASP": "CB CG OD1 OD2",
    "CYS": "CB SG",
    "GLN": "CB CG CD OE1 NE2",
    "GLU": "CB CG CD OE1 OE2",
    "GLY": "",
    "HIS": "CB CG ND1 CD2 CE1 NE2",
    "ILE": "CB CG1 CG2 CD1",
    "LEU": "CB CG CD1 CD2",
    "LYS": "CB CG CD CE NZ",
    "MET": "CB CG SD CE",
    "PHE": "CB CG CD1 CD2 CE1 CE2 CZ",
    "PRO": "CB CG CD",
    "SER": "CB OG",
    "THR": "CB OG1 CG2",
    "TRP": "CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2",
    "TYR": "CB CG CD1 CD2 CE1 CE2 CZ OH",
    "VAL": "CB CG1 CG2",
}


class Repacker:
    """Keep sequence and backbone fixed while sampling side chains.

    The executable can live in a separate Python 3.12/CUDA environment via
    GRAPHRELAX_RFD4_EXECUTABLE. No RFD4 imports are needed in GraphRelax.
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        self._calls = 0

    def repack(
        self,
        pdb_path: Path,
        design_spec: Optional[DesignSpec] = None,
        original_path: Optional[Path] = None,
    ) -> dict:
        structure = PDBParser(QUIET=True).get_structure("input", str(pdb_path))
        if len(structure) != 1:
            raise ValueError(
                "RFD4 repacking requires a single structural model"
            )
        residues = [
            r
            for r in structure.get_residues()
            if r.id[0] == " " and r.resname in AMINO_ACIDS
        ]
        if not residues:
            raise ValueError("No standard protein residues to repack")
        fixed = [
            design_spec is not None
            and not design_spec.get_spec(
                r.parent.id, r.id[1], r.id[2].strip()
            ).is_repackable()
            for r in residues
        ]
        if original_path is not None:
            native = PDBParser(QUIET=True).get_structure(
                "native", str(original_path)
            )
            for residue, keep in zip(residues, fixed):
                if keep:
                    source = native[0][residue.parent.id][residue.id]
                    if source.resname != residue.resname:
                        raise ValueError(
                            "Sequence design changed a NATRO residue"
                        )
                    for atom in list(residue):
                        residue.detach_child(atom.id)
                    for atom in source:
                        residue.add(atom.copy())
        sequence = "".join(seq1(r.resname) for r in residues)
        if all(fixed) and original_path is None:
            return {
                "pdb_string": pdb_path.read_text(),
                "sequence": sequence,
                "native_sequence": sequence,
            }
        for residue in residues:
            if not {"N", "CA", "C", "O"}.issubset(residue.child_dict):
                raise ValueError(
                    "RFD4 repacking requires complete N/CA/C/O backbones"
                )
            if residue.is_disordered():
                raise ValueError(
                    "Resolve alternate locations before RFD4 repacking"
                )

        with tempfile.TemporaryDirectory(prefix="graphrelax-rfd4-") as tmp:
            directory = Path(tmp)
            cif_path = directory / "input.cif"
            writer = MMCIFIO()
            writer.set_structure(structure)
            writer.save(str(cif_path))
            # BioPython writes sequential label_seq_id per protein chain;
            # conditions use those IDs, not author numbering/insertion codes.
            cif = MMCIF2Dict(str(cif_path))
            addresses = {}
            for chain, number, code, label_chain, label_number in zip(
                cif["_atom_site.auth_asym_id"],
                cif["_atom_site.auth_seq_id"],
                cif["_atom_site.pdbx_PDB_ins_code"],
                cif["_atom_site.label_asym_id"],
                cif["_atom_site.label_seq_id"],
            ):
                addresses[
                    (chain, int(number), code if code not in ("?", ".") else "")
                ] = (label_chain, label_number)
            selections = []
            for residue, keep in zip(residues, fixed):
                if keep:
                    chain, number = addresses[
                        (
                            residue.parent.id,
                            residue.id[1],
                            residue.id[2].strip(),
                        )
                    ]
                    selections.append(
                        {
                            "select": (
                                f"chain_id == {chain!r} and res_id == {number}"
                            )
                        }
                    )
            conditions = {
                "C_SEQ": {"true": [{"select": "atom_name != ''"}]},
                "C_CRD": {
                    "true": [
                        {"select": "is_protein_backbone or not is_protein"},
                        *selections,
                    ]
                },
            }
            spec_path = directory / "repack.json"
            spec_path.write_text(
                json.dumps(
                    {
                        "repack": {
                            "input": str(cif_path),
                            "input_as_annotated_cif": True,
                            "conditions": conditions,
                        }
                    }
                )
            )
            executable = os.environ.get(
                "GRAPHRELAX_RFD4_EXECUTABLE", "rfd4-generate"
            )
            command = [
                executable,
                f"inputs={spec_path}",
                "generation=inverse_folding",
                "recycle_strategy=medium/flat",
                "num_replicates=1",
                f"out_dir={directory / 'output'}",
                "compress_outputs=false",
                "metrics_tags=[]",
                f"hydra.run.dir={directory / 'hydra'}",
            ]
            if self.seed is not None:
                command.append(f"seed={self.seed + self._calls}")
            self._calls += 1
            try:
                subprocess.run(command, check=True, cwd=directory)
            except FileNotFoundError as exc:
                raise RuntimeError(
                    "Install RFD4-Proteina in its supported GPU environment "
                    "and set GRAPHRELAX_RFD4_EXECUTABLE to its "
                    "rfd4-generate executable"
                ) from exc
            outputs = list(
                (directory / "output" / "inference_samples").rglob(
                    "*_generated.cif"
                )
            )
            if len(outputs) != 1:
                raise RuntimeError(
                    f"Expected one RFD4 structure, found {len(outputs)}"
                )
            generated = MMCIFParser(QUIET=True).get_structure(
                "packed", str(outputs[0])
            )
            packed = [
                r
                for r in generated.get_residues()
                if r.id[0] == " " and r.resname in AMINO_ACIDS
            ]
            if len(packed) != len(residues) or any(
                a.resname != b.resname for a, b in zip(residues, packed)
            ):
                raise RuntimeError("RFD4 output changed the protein sequence")
            # Sampling may apply a global rigid transform. Align before copying
            # side chains into the original structure, which retains all IDs,
            # backbone atoms, fixed residues and non-protein context exactly.
            reference_atoms = []
            moving_atoms = []
            for original, sampled in zip(residues, packed):
                for name in ("N", "CA", "C", "O"):
                    if name not in sampled:
                        raise RuntimeError(
                            "RFD4 output has an incomplete backbone"
                        )
                    reference_atoms.append(original[name])
                    moving_atoms.append(sampled[name])
            alignment = Superimposer()
            alignment.set_atoms(reference_atoms, moving_atoms)
            alignment.apply(list(generated.get_atoms()))
            if any(
                not np.isfinite(b.coord).all()
                or np.linalg.norm(a.coord - b.coord) > 0.1
                for a, b in zip(reference_atoms, moving_atoms)
            ):
                raise RuntimeError("RFD4 output changed the fixed backbone")
            for original, sampled, keep in zip(residues, packed, fixed):
                if keep:
                    continue
                sidechains = [
                    a.copy()
                    for a in sampled
                    if a.name not in BACKBONE and a.element != "H"
                ]
                if {a.name for a in sidechains} != set(
                    SIDECHAIN_ATOMS[original.resname].split()
                ):
                    raise RuntimeError(
                        "RFD4 output is missing side-chain atoms"
                    )
                for atom in list(original):
                    if atom.name not in BACKBONE:
                        original.detach_child(atom.id)
                for atom in sidechains:
                    if not np.isfinite(atom.coord).all():
                        raise RuntimeError(
                            "RFD4 output contains non-finite coordinates"
                        )
                    original.add(atom)
            output = io.StringIO()
            writer = PDBIO()
            writer.set_structure(structure)
            writer.save(output)
            return {
                "pdb_string": output.getvalue(),
                "sequence": sequence,
                "native_sequence": sequence,
            }
