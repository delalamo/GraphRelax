# GraphRelax

GraphRelax alternates **RFD4-Proteina fixed-backbone sequence design and side-chain
packing** with **OpenMM AMBER minimization**, providing relax and design protocols
similar to Rosetta FastRelax and Design.

## Installation

Install GraphRelax and its relaxation dependencies:

```bash
conda install -c conda-forge pdbfixer
git clone https://github.com/delalamo/GraphRelax.git
cd GraphRelax
pip install -e .
```

Design and repacking require a separate
[RFD4-Proteina installation](https://github.com/RosettaCommons/RFD4-Proteina#installation)
on a supported Linux NVIDIA GPU host with Python 3.12. Follow upstream installation
and model-weight instructions, then point GraphRelax at that environment:

```bash
export GRAPHRELAX_RFD4_EXECUTABLE=/path/to/rfd4-env/bin/rfd4-generate
```

If `rfd4-generate` is on PATH, the variable is unnecessary. The executable must be
available on the same host as GraphRelax. RFD4 manages its own model weights and
cache; GraphRelax does not download design-model weights. `--no-repack` requires
only the relaxation environment.

GraphRelax's Python dependencies include NumPy, BioPython, OpenMM, and the
PyTorch/absl-py/ml-collections/dm-tree utilities used by the independently licensed
vendored OpenFold relaxation code. ProDy is not required. OpenFold resides under
`graphrelax._vendor.openfold`; its original license headers are retained.

### Docker

The base image provides the relaxation environment. It does not bundle RFD4's
GPU runtime or weights. For design or repacking, provision RFD4 inside a derived
GPU image and expose its executable using `GRAPHRELAX_RFD4_EXECUTABLE`.

```bash
docker build -t graphrelax .
docker run --rm -v "$(pwd):/data" graphrelax \
    -i /data/input.pdb -o /data/minimized.pdb --no-repack
```

## Usage

Inputs and outputs support PDB and mmCIF formats.

```bash
# Repack and minimize for five cycles (default)
graphrelax -i input.pdb -o relaxed.pdb

# Repack only; keep sequence and backbone fixed
graphrelax -i input.pdb -o packed.pdb --repack-only

# Minimize only; no design model needed
graphrelax -i input.pdb -o minimized.pdb --no-repack

# Design sequence and side chains, then minimize
graphrelax -i input.pdb -o designed.pdb --design

# Generate ten all-atom designs without minimization
graphrelax -i input.pdb -o designed.pdb --design-only -n 10

# Restrict design by residue
graphrelax -i input.pdb -o designed.pdb --design --resfile design.resfile

# Design with ligand context
graphrelax -i complex.pdb -o designed.pdb --design --constrained-minimization
```

| Mode                | RFD4 operation                 | Minimize |
| ------------------- | ------------------------------ | -------- |
| `--relax` (default) | Fixed-sequence repacking       | Yes      |
| `--repack-only`     | Fixed-sequence repacking       | No       |
| `--no-repack`       | None                           | Yes      |
| `--design`          | Sequence and side-chain design | Yes      |
| `--design-only`     | Sequence and side-chain design | No       |

### Sampling controls

All model calls use `generation=inverse_folding` to hold the input backbone fixed.
Design generates sequence and side chains together in one call. Repacking also
conditions on the entire input sequence. The adapter uses the public CLI and
condition-spec interface at upstream commit
`6fc44c060ef96d4c759bafeaa1eae1a66a5be67c`.

- `--tsr-k K`: sequence sharpening strength (default `4`). Lower values raise
  sequence sampling temperature.
- `--tsr-sigma S`: sequence sharpening schedule parameter (default `12`).
- `--recycle-strategy NAME`: upstream recycle preset (default `medium/flat`).
- `--seed N`: seed the first call with N and increment for each subsequent call.
- `--n-iter N`: number of cycles (default `5`).
- `-n N`: number of outputs (default `1`).

The old `--model-type` and `--temperature` options and their Python configuration
fields are removed. There is no separate side-chain packing switch for design:
RFD4 generates an all-atom structure. Each call starts a new model process, which
adds startup cost per cycle.

The adapter checks sequence restrictions, backbone geometry, and complete standard
side-chain heavy atoms before accepting output. It aligns generated coordinates
into the input frame, copies the generated sequence and side chains, and retains
original residue IDs, insertion codes, backbone coordinates, fixed residues, and
non-protein atoms. Supply a single model with complete N/CA/C/O backbones and
resolved alternate locations.

### Resfiles

Rosetta-style resfiles control both design and repacking:

```text
NATAA
START
10 A ALLAA
11 A PIKAA AVIL
12 A NOTAA CP
13 A POLAR
14 A APOLAR
20 A NATRO
```

| Command     | Behavior                                         |
| ----------- | ------------------------------------------------ |
| `NATRO`     | Preserve identity and all atoms                  |
| `NATAA`     | Preserve identity; repack side chains            |
| `ALLAA`     | Allow all 20 standard amino acids in design mode |
| `PIKAA XYZ` | Allow only the listed amino acids                |
| `NOTAA XYZ` | Exclude the listed amino acids                   |
| `POLAR`     | Allow DEHKNQRST                                  |
| `APOLAR`    | Allow ACFGILMPVWY                                |

In repack/relax modes, all identities remain fixed regardless of design commands.
In design modes, a resfile overrides full redesign. Fixed identities use `C_SEQ`;
allowed-amino-acid restrictions use `C_SEQ_PROB_<TOKEN>` exclusions. `NATRO`
coordinates are conditioned and retained. These restrictions apply to the model
step; subsequent AMBER minimization may move fixed residues.

### Relaxation and preprocessing

Default minimization is unconstrained. `--constrained-minimization` uses OpenFold's
AMBER relaxation with position restraints and violation checks. `--stiffness K`
sets restraint stiffness (default `10` kcal/mol/Å²), and `--max-iterations N` limits
minimization iterations (`0` means unlimited).

Inputs containing ligands require `--constrained-minimization` when minimization
is enabled. Ligands provide RFD4 context; relaxation extracts and restores them
because the AMBER setup does not parameterize arbitrary ligands.

- `--pre-idealize`: idealize backbone geometry before processing.
- `--ignore-missing-residues`: do not add SEQRES missing residues during idealization.
- `--retain-chainbreaks`: retain chain breaks during idealization.
- `--no-split-gaps`: disable chain splitting at gaps during minimization.
- `--keep-waters`: preserve input waters (removed by default).
- `--scorefile FILE`: write OpenMM energy terms and sequence recovery.
- `-v`: verbose logging.

Scores contain physical energies when minimization runs and sequence recovery when
design/repacking runs. The adapter does not invent a model confidence score from
RFD4 output.

## Python API

```python
from pathlib import Path
from graphrelax import Pipeline, PipelineConfig, PipelineMode
from graphrelax.config import DesignConfig, RelaxConfig

pipeline = Pipeline(PipelineConfig(
    mode=PipelineMode.DESIGN,
    n_iterations=5,
    design=DesignConfig(seed=42, tsr_k=4.0, tsr_sigma=12.0,
                        recycle_strategy="medium/flat"),
    relax=RelaxConfig(max_iterations=100),
))
results = pipeline.run(Path("input.pdb"), Path("output.pdb"),
                       resfile=Path("design.resfile"))
```

`Designer.design()` and `Designer.repack()` return `pdb_string`, `sequence`, and
`native_sequence`. Use `Designer.result_to_pdb_string()` to obtain the PDB text.
No sequence log-probability arrays or design-model loss are returned.

## Validation

```bash
pytest
pre-commit run --all-files

# Real model smoke tests on a provisioned GPU host
pytest -m integration tests/test_rfd4.py
```

Unit tests simulate RFD4 output to verify conditioning, restrictions, output
validation, and pipeline behavior. The GPU smoke tests skip when the RFD4 CLI is
absent; unit tests alone do not establish model inference or design quality.

## License and references

GraphRelax is MIT licensed. Vendored OpenFold code retains its Apache 2.0 notices.
Please cite the [RFD4-Proteina project](https://github.com/RosettaCommons/RFD4-Proteina),
OpenMM (Eastman et al., 2017), and the AlphaFold relaxation protocol (Jumper et al.,
2021), as appropriate.
