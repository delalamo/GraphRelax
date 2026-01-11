# GraphRelax

A drop-in replacement for Rosetta Relax that replaces force field-guided residue repacking and design with equivalent functions from graph neural networks.

GraphRelax combines **LigandMPNN** (for sequence design and side-chain packing) with **OpenMM AMBER minimization** to reproduce Rosetta FastRelax and Design protocols.

## Features

- **FastRelax-like protocol**: Alternate between side-chain repacking and energy minimization
- **Sequence design**: Full redesign or residue-specific control via Rosetta-style resfiles
- **Multiple output modes**: Relax-only, repack-only, design-only, or combinations
- **GPU acceleration**: Automatic GPU detection for both LigandMPNN and OpenMM
- **Scorefile output**: Rosetta-compatible scorefiles with energy terms and sequence metrics

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/GraphRelax.git
cd GraphRelax

# Install pdbfixer (not available on PyPI)
conda install -c conda-forge pdbfixer

# Install GraphRelax and dependencies
pip install -e .

# Download LigandMPNN model weights
./scripts/download_weights.sh
```

### Platform-specific Installation

```bash
# CPU-only (smaller install, no GPU dependencies)
pip install -e ".[cpu]"

# With CUDA 11 GPU support
pip install -e ".[cuda11]"

# With CUDA 12 GPU support
pip install -e ".[cuda12]"
```

### Dependencies

- Python >= 3.9
- PyTorch
- OpenMM (pip-installable)
- BioPython
- ProDy
- pdbfixer (conda-forge only)
- dm-tree

## Usage

### Basic Commands

```bash
# Default: repack + minimize for 5 cycles
graphrelax -i input.pdb -o relaxed.pdb

# Repack + minimize with 10 cycles
graphrelax -i input.pdb -o relaxed.pdb --n-iter 10

# Only minimize (no repacking)
graphrelax -i input.pdb -o minimized.pdb --no-repack

# Only repack side chains (no minimization)
graphrelax -i input.pdb -o repacked.pdb --repack-only

# Full redesign + minimize
graphrelax -i input.pdb -o designed.pdb --design

# Design with resfile specification
graphrelax -i input.pdb -o designed.pdb --design --resfile design.resfile

# Generate 10 different designs
graphrelax -i input.pdb -o designed.pdb --design -n 10

# Design only (no minimization) - fast sampling
graphrelax -i input.pdb -o designed.pdb --design-only -n 100

# With scorefile output
graphrelax -i input.pdb -o relaxed.pdb --scorefile scores.sc

# Design with ligand context
graphrelax -i complex.pdb -o designed.pdb --design --model-type ligand_mpnn
```

### Operating Modes

| Flag                | Repack | Design | Minimize |
| ------------------- | ------ | ------ | -------- |
| `--relax` (default) | Yes    | No     | Yes      |
| `--repack-only`     | Yes    | No     | No       |
| `--no-repack`       | No     | No     | Yes      |
| `--design`          | No     | Yes    | Yes      |
| `--design-only`     | No     | Yes    | No       |

### Resfile Format

GraphRelax supports Rosetta-style resfiles for residue-specific control:

```
# Default behavior for all residues
NATAA
START
# Design positions 10-15 on chain A
10 A ALLAA
11 A ALLAA
12 A ALLAA
13 A ALLAA
14 A ALLAA
15 A ALLAA
# Position 20: only allow hydrophobics
20 A PIKAA AVILMFYW
# Position 25: exclude cysteine and proline
25 A NOTAA CP
# Position 30: only polar residues
30 A POLAR
# Keep position 40 completely fixed
40 A NATRO
```

#### Supported Commands

| Command     | Description                                      |
| ----------- | ------------------------------------------------ |
| `NATRO`     | Fixed completely (no design, no repacking)       |
| `NATAA`     | Repack only (same amino acid, optimize rotamers) |
| `ALLAA`     | Design with all 20 amino acids                   |
| `PIKAA XYZ` | Design with only specified amino acids           |
| `NOTAA XYZ` | Design excluding specified amino acids           |
| `POLAR`     | Design with polar residues only (DEHKNQRST)      |
| `APOLAR`    | Design with nonpolar residues only (ACFGILMPVWY) |

### Command-Line Options

```
Required:
  -i, --input PDB       Input PDB file
  -o, --output PDB      Output PDB file (or prefix if -n > 1)

Mode selection:
  --relax               Repack + minimize cycles (default)
  --repack-only         Only repack side chains
  --no-repack           Only minimize
  --design              Design + minimize
  --design-only         Only design

Iteration and output:
  --n-iter N            Number of cycles (default: 5)
  -n, --n-outputs N     Number of outputs to generate (default: 1)

Design options:
  --resfile FILE        Rosetta-style resfile
  --temperature T       LigandMPNN sampling temperature (default: 0.1)
  --model-type TYPE     protein_mpnn, ligand_mpnn, or soluble_mpnn

Relaxation options:
  --stiffness K         Restraint stiffness in kcal/mol/A^2 (default: 10.0)
  --max-iterations N    Max L-BFGS iterations, 0=unlimited (default: 0)

Scoring:
  --scorefile FILE      Output scorefile with energy terms

General:
  -v, --verbose         Verbose output
  --seed N              Random seed for reproducibility
```

### Scorefile Output

When `--scorefile` is specified, outputs a Rosetta-style scorefile:

```
SCORE:  total_score  openmm_energy  bond_energy  angle_energy  dihedral_energy  nonbonded_energy  ligandmpnn_score  seq_recovery  description
SCORE:     -234.56       -234.56        12.3         45.6             23.1              -315.6             0.847          0.92   output_1.pdb
SCORE:     -228.12       -228.12        11.8         44.2             22.8              -307.0             0.823          0.89   output_2.pdb
```

## Python API

```python
from graphrelax import Pipeline, PipelineConfig, PipelineMode
from graphrelax.config import DesignConfig, RelaxConfig
from pathlib import Path

# Configure pipeline
config = PipelineConfig(
    mode=PipelineMode.DESIGN,
    n_iterations=5,
    n_outputs=10,
    design=DesignConfig(
        model_type="ligand_mpnn",
        temperature=0.1,
    ),
    relax=RelaxConfig(
        stiffness=10.0,
    ),
)

# Run pipeline
pipeline = Pipeline(config)
results = pipeline.run(
    input_pdb=Path("input.pdb"),
    output_pdb=Path("output.pdb"),
    resfile=Path("design.resfile"),  # optional
)

# Access results
for output in results["outputs"]:
    print(f"Output: {output['output_path']}")
    print(f"Sequence: {output['sequence']}")
    print(f"Final energy: {output.get('final_energy', 'N/A')}")
```

## How It Works

GraphRelax implements an alternating optimization protocol similar to Rosetta FastRelax:

1. **Parse Input**: Read PDB structure and optional resfile
2. **For each iteration**:
   - **Design/Repack Phase**: Use LigandMPNN to generate sequences or repack side chains
   - **Minimize Phase**: Use OpenMM with AMBER99SB force field for energy minimization
3. **Output**: Write final structure(s) and optional scorefile

### Key Differences from Rosetta

| Aspect            | Rosetta                      | GraphRelax                     |
| ----------------- | ---------------------------- | ------------------------------ |
| Sequence sampling | Monte Carlo with force field | LigandMPNN neural network      |
| Rotamer packing   | Discrete rotamer library     | LigandMPNN continuous sampling |
| Energy function   | Rosetta energy function      | AMBER99SB force field          |
| Speed             | Slower                       | Faster (GPU acceleration)      |

## License

MIT License

## Citation

If you use GraphRelax in your research, please cite:

- LigandMPNN: Dauparas et al. (2023)
- OpenMM: Eastman et al. (2017)
- AlphaFold relaxation protocol: Jumper et al. (2021)
