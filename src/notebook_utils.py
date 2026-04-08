"""
Repair notebook JSON so ``jupyter nbconvert`` (nbformat ≥5) accepts saved outputs.
Some Jupyter front ends omit ``metadata`` / ``execution_count`` / stream ``name`` on outputs.
"""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat import NotebookNode


def repair_notebook_outputs(nb: NotebookNode) -> None:
    """Mutate ``nb`` in place: fill required fields on code cell outputs."""
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        ec = cell.get("execution_count")
        for out in cell.get("outputs", []):
            ot = out.get("output_type")
            if ot == "stream":
                out.setdefault("name", "stdout")
            elif ot in ("execute_result", "display_data"):
                out.setdefault("metadata", {})
            if ot == "execute_result":
                out.setdefault("execution_count", ec)


def repair_notebook_file(path: Path | str) -> Path:
    """Read notebook, repair outputs, write back. Returns ``path``."""
    path = Path(path)
    nb = nbformat.read(path, as_version=4)
    repair_notebook_outputs(nb)
    nbformat.write(nb, path)
    return path
