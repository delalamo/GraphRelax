# GraphRelax

A drop-in replacement for Rosetta Relax that replaces force field-guided residue repacking and design with equivalent functions from graph neural networks.

GraphRelax combines **LigandMPNN** (for sequence design and side-chain packing) with **OpenMM AMBER minimization** to reproduce Rosetta FastRelax and Design protocols.

## Installation

GraphRelax requires pdbfixer, which is only available via conda-forge. We recommend using mamba for faster installation.

### From PyPI (Latest Release)

```bash
# First, install conda-forge dependencies (not available on PyPI)
mamba install -c conda-forge pdbfixer

# Then install graphrelax from PyPI
pip install graphrelax
```

This installs the latest stable release with basic functionality.

**For ligand support** (auto-parameterization of small molecules), also install:

```bash
mamba install -c conda-forge openmmforcefields openff-toolkit rdkit
```

### From Source (Latest Development Version)

```bash
# First, install pdbfixer via mamba/conda (required)
mamba install -c conda-forge pdbfixer

# Clone the repository
git clone https://github.com/delalamo/GraphRelax.git
cd GraphRelax

# Option A: Use environment.yml with mamba (recommended)
mamba env create -f environment.yml
conda activate graphrelax
pip install -e .

# Option B: Manual installation
mamba install -c conda-forge pdbfixer openmmforcefields openff-toolkit rdkit
pip install -e .
```

This installs the latest development version with all recent changes.

LigandMPNN model weights (~40MB) are downloaded automatically on first run.

> **Note:** `mamba` is a drop-in replacement for `conda` that's 10-100x faster. Install it with `conda install -n base -c conda-forge mamba`, or use `micromamba` as a standalone tool.

### Dependencies

Core dependencies (installed automatically via pip):

- Python >= 3.9
- PyTorch >= 2.0
- NumPy < 2.0 (PyTorch <2.5 is incompatible with NumPy 2.x)
- OpenMM
- BioPython
- ProDy
- dm-tree
- absl-py
- ml-collections

Conda-forge only (must be installed via mamba/conda, not pip):

| Package           | Required | Purpose                          |
| ----------------- | -------- | -------------------------------- |
| pdbfixer          | Yes      | Structure preparation            |
| openmmforcefields | Optional | Small molecule force fields      |
| openff-toolkit    | Optional | OpenFF Molecule parameterization |
| rdkit             | Optional | Bond perception from PDB         |

The optional packages are needed for **ligand support** (auto-parameterization of small molecules during minimization). Install all conda-forge dependencies with:

```bash
mamba install -c conda-forge pdbfixer openmmforcefields openff-toolkit rdkit
```

### Docker

Run GraphRelax using Docker without installing dependencies:

```bash
# Pull the image
docker pull ghcr.io/delalamo/graphrelax:latest

# Run with input/output files
docker run --rm -v $(pwd):/data ghcr.io/delalamo/graphrelax:latest \
    -i /data/input.pdb -o /data/output.pdb

# Design mode with 5 outputs
docker run --rm -v $(pwd):/data ghcr.io/delalamo/graphrelax:latest \
    -i /data/input.pdb -o /data/designed.pdb --design -n 5
```

Build locally:

```bash
docker build -t graphrelax .
docker run --rm graphrelax --help
```

## Features

- **FastRelax-like protocol**: Alternate between side-chain repacking and energy minimization
- **Sequence design**: Full redesign or residue-specific control via Rosetta-style resfiles
- **Multiple output modes**: Relax-only, repack-only, design-only, or combinations
- **GPU acceleration**: Automatic GPU detection for both LigandMPNN and OpenMM
- **Scorefile output**: Rosetta-compatible scorefiles with energy terms and sequence metrics

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

# Design with ligand context (requires --constrained-minimization for ligands)
graphrelax -i complex.pdb -o designed.pdb --design --model-type ligand_mpnn --constrained-minimization
```

### Operating Modes

| Flag                | Repack | Design | Minimize |
| ------------------- | ------ | ------ | -------- |
| `--relax` (default) | Yes    | No     | Yes      |
| `--repack-only`     | Yes    | No     | No       |
| `--no-repack`       | No     | No     | Yes      |
| `--design`          | No     | Yes    | Yes      |
| `--design-only`     | No     | Yes    | No       |

### Minimization Modes

By default, GraphRelax uses **unconstrained minimization** - a simple, bare-bones OpenMM energy minimization with no position restraints and default tolerance parameters. This is fast and works well for most use cases.

For more controlled minimization (AlphaFold-style), use `--constrained-minimization`:

```bash
# Default: unconstrained minimization (fast, no restraints)
graphrelax -i input.pdb -o relaxed.pdb --no-repack

# Constrained minimization with position restraints and violation checking
# Note: requires pdbfixer (conda install -c conda-forge pdbfixer)
graphrelax -i input.pdb -o relaxed.pdb --no-repack --constrained-minimization

# Constrained with custom restraint stiffness
graphrelax -i input.pdb -o relaxed.pdb --constrained-minimization --stiffness 5.0
```

| Mode                         | Position Restraints | Violation Checking | Speed  | Requires pdbfixer |
| ---------------------------- | ------------------- | ------------------ | ------ | ----------------- |
| Default (unconstrained)      | No                  | No                 | Fast   | No                |
| `--constrained-minimization` | Yes (harmonic)      | Yes                | Slower | Yes               |

### Crystallography Artifact Removal

By default, GraphRelax **automatically removes** common crystallography and cryo-EM artifacts from input structures:

- **Buffers**: SO4, PO4, CIT, ACT, MES, HEPES, Tris, etc.
- **Cryoprotectants**: GOL (glycerol), EDO (ethylene glycol), MPD, PEG variants, DMSO
- **Detergents**: DDM, OG, LDAO, CHAPS, SDS, etc.
- **Lipids/Fatty acids**: PLM (palmitic), MYR (myristic), OLA (oleic), etc.
- **Reducing agents**: BME, DTT
- **Halide ions**: CL, BR, IOD

**Biologically relevant metal ions** (ZN, MG, CA, FE, MN, CU, etc.) are **preserved**.

```bash
# Artifacts are removed by default
graphrelax -i crystal_structure.pdb -o relaxed.pdb

# Keep all HETATM records including artifacts
graphrelax -i crystal_structure.pdb -o relaxed.pdb --keep-all-ligands

# Keep specific artifacts that you need (comma-separated)
graphrelax -i crystal_structure.pdb -o relaxed.pdb --keep-ligand GOL,SO4
```

### Working with Ligands

Biologically relevant ligands (cofactors, substrates, inhibitors) are **auto-detected** and parameterized using openmmforcefields. No special flags are needed.

```bash
# Ligands are automatically included and parameterized
graphrelax -i protein_with_ligand.pdb -o designed.pdb --design

# Choose a specific force field for ligands
graphrelax -i protein_with_ligand.pdb -o designed.pdb \
    --ligand-forcefield gaff-2.11

# Provide SMILES for unknown ligands
graphrelax -i protein_with_ligand.pdb -o designed.pdb \
    --ligand-smiles "LIG:c1ccccc1"

# Strip all ligands (protein-only processing)
graphrelax -i protein_with_ligand.pdb -o designed.pdb --ignore-ligands
```

**Requires:** `mamba install -c conda-forge openmmforcefields openff-toolkit rdkit`

For design around ligands, use `ligand_mpnn` model type:

```bash
graphrelax -i protein_with_ligand.pdb -o designed.pdb \
    --design --model-type ligand_mpnn
```

#### Ligand Force Field Options

| Force Field     | Flag                                 | Description                |
| --------------- | ------------------------------------ | -------------------------- |
| OpenFF Sage 2.0 | `--ligand-forcefield openff-2.0.0`   | Modern, accurate (default) |
| GAFF 2.11       | `--ligand-forcefield gaff-2.11`      | Well-tested, robust        |
| Espaloma 0.3    | `--ligand-forcefield espaloma-0.3.0` | ML-based, fast             |

#### Alternative: Constrained Minimization

Use `--constrained-minimization` for position-restrained minimization (does not require openmmforcefields):

```bash
graphrelax -i protein_with_ligand.pdb -o designed.pdb \
    --design --constrained-minimization
```

**Requires:** `mamba install -c conda-forge pdbfixer`

### Pre-Idealization

GraphRelax can optionally idealize backbone geometry before processing. This is useful for structures with distorted bond lengths or angles (e.g., from homology modeling or low-resolution experimental data). The idealization step:

1. Corrects backbone bond lengths and angles to ideal values
2. Preserves phi/psi/omega dihedral angles
3. Adds missing atoms and optionally missing residues from SEQRES
4. Runs constrained minimization to relieve local strain
5. By default, closes chain breaks (gaps) in the structure

```bash
# Idealize before relaxation
graphrelax -i input.pdb -o relaxed.pdb --pre-idealize

# Idealize but don't add missing residues from SEQRES
graphrelax -i input.pdb -o relaxed.pdb --pre-idealize --ignore-missing-residues

# Idealize but keep chain breaks as separate chains (don't close gaps)
graphrelax -i input.pdb -o relaxed.pdb --pre-idealize --retain-chainbreaks

# Combine with design
graphrelax -i input.pdb -o designed.pdb --pre-idealize --design
```

**Note:** Pre-idealization requires pdbfixer (`mamba install -c conda-forge pdbfixer`).

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
  --constrained-minimization  Use constrained minimization with position
                              restraints (AlphaFold-style). Default is
                              unconstrained. Requires pdbfixer.
  --stiffness K         Restraint stiffness in kcal/mol/A^2 (default: 10.0)
                        Only applies to constrained minimization.
  --max-iterations N    Max L-BFGS iterations, 0=unlimited (default: 0)
  --ignore-ligands      Strip all ligands before processing. By default,
                        ligands are auto-detected and parameterized.
  --ligand-forcefield FF  Force field for ligands: openff-2.0.0 (default),
                          gaff-2.11, or espaloma-0.3.0
  --ligand-smiles RES:SMILES  Provide SMILES for a ligand residue. Can be
                              used multiple times for multiple ligands.

Input preprocessing:
  --keep-waters         Keep water molecules in input (default: removed)
  --keep-all-ligands    Keep all HETATM records including crystallography
                        artifacts. By default, common artifacts are removed.
  --keep-ligand RES1,RES2,...
                        Keep specific ligand residues (comma-separated).
                        Example: --keep-ligand GOL,SO4
  --pre-idealize        Idealize backbone geometry before processing.
                        Corrects bond lengths/angles while preserving
                        dihedral angles. By default, chain breaks are closed.
                        Requires pdbfixer.
  --ignore-missing-residues
                        Do not add missing residues from SEQRES during
                        pre-idealization. By default, missing terminal and
                        internal loop residues are added.
  --retain-chainbreaks  Do not close chain breaks during pre-idealization.
                        By default, chain breaks are closed by treating all
                        segments as a single chain.

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
from graphrelax.config import DesignConfig, RelaxConfig, IdealizeConfig
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
        constrained=False,  # Default: unconstrained minimization
    ),
    idealize=IdealizeConfig(
        enabled=True,  # Enable pre-idealization
        add_missing_residues=True,  # Add missing residues from SEQRES
        close_chainbreaks=True,  # Close chain breaks (default)
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
   - **Minimize Phase**: Use OpenMM with AMBER force field for energy minimization
3. **Output**: Write final structure(s) and optional scorefile

### Key Differences from Rosetta

| Aspect            | Rosetta                      | GraphRelax                     |
| ----------------- | ---------------------------- | ------------------------------ |
| Sequence sampling | Monte Carlo with force field | LigandMPNN neural network      |
| Rotamer packing   | Discrete rotamer library     | LigandMPNN continuous sampling |
| Energy function   | Rosetta energy function      | AMBER force field              |
| Speed             | Slower                       | Faster (GPU acceleration)      |

## License

MIT License

## Citation

If you use GraphRelax in your research, please cite:

- LigandMPNN: Dauparas et al. (2023)
- OpenMM: Eastman et al. (2017)
- AlphaFold relaxation protocol: Jumper et al. (2021)
