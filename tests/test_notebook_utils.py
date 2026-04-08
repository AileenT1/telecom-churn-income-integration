"""notebook_utils: repair outputs for nbformat validation."""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.notebook_utils import repair_notebook_outputs


def test_repair_minimal_invalid_output():
    nb = nbformat.v4.new_notebook()
    cell = nbformat.v4.new_code_cell("1+1")
    cell["outputs"] = [
        {
            "output_type": "execute_result",
            "data": {"text/plain": "2"},
        }
    ]
    nb["cells"] = [cell]
    repair_notebook_outputs(nb)
    out = nb["cells"][0]["outputs"][0]
    assert out["metadata"] == {}
    assert out["execution_count"] is None
    nbformat.validator.validate(nb)


def test_repair_stream_name():
    nb = nbformat.v4.new_notebook()
    cell = nbformat.v4.new_code_cell("print(1)")
    cell["outputs"] = [{"output_type": "stream", "text": "1\n"}]
    nb["cells"] = [cell]
    repair_notebook_outputs(nb)
    assert nb["cells"][0]["outputs"][0]["name"] == "stdout"
    nbformat.validator.validate(nb)
