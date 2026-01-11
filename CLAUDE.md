# CLAUDE.md

Guidelines for AI assistants working on this codebase.

## Pre-commit Hooks

Always run pre-commit hooks before pushing code:

```bash
pre-commit run --all-files
```

If pre-commit is not installed, install it first:

```bash
pip install pre-commit
pre-commit install
```

## Imports

All imports must be placed at the top of the file. Avoid lazy imports (imports inside functions, methods, or conditionals). This keeps dependencies explicit and improves code readability.

**Do this:**

```python
import numpy as np
from Bio.PDB import PDBParser

def process_structure(pdb_file: str) -> dict:
    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_file)
    return {"atoms": len(list(structure.get_atoms()))}
```

**Don't do this:**

```python
def process_structure(pdb_file: str) -> dict:
    import numpy as np  # Bad: lazy import
    from Bio.PDB import PDBParser  # Bad: lazy import

    parser = PDBParser()
    structure = parser.get_structure("protein", pdb_file)
    return {"atoms": len(list(structure.get_atoms()))}
```

## Type Hints

Avoid using type hints outside of function definitions. Type hints should only be used in function signatures (parameters and return types), not for variable assignments or class attributes.

**Do this:**

```python
def process_sequence(sequence: str, threshold: float = 0.5) -> list:
    result = []
    count = 0
    # ...
    return result
```

**Don't do this:**

```python
def process_sequence(sequence: str, threshold: float = 0.5) -> list:
    result: list = []
    count: int = 0
    # ...
    return result
```
