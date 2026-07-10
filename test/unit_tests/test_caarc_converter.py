"""Tests for the streaming CAARC-to-MFEM mesh converter."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _converter_module():
    path = Path(__file__).parents[2] / "mfem" / "tools" / "convert_caarc_mesh.py"
    spec = spec_from_file_location("convert_caarc_mesh", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_convert_small_mixed_mesh(tmp_path):
    """Tetrahedra and triangles become MFEM elements and boundary elements."""
    source = tmp_path / "tiny.msh"
    source.write_text(
        """MESH dimension 3 ElemType Tetrahedra Nnode 4
Coordinates
1 0 0 0
2 1 0 0
3 0 1 0
4 0 0 1
End Coordinates
Elements
1 1 2 3 4
End Elements
MESH dimension 3 ElemType Triangle Nnode 3
Coordinates
End Coordinates
Elements
1 1 3 2
2 1 2 4
3 2 3 4
4 3 1 4
End Elements
"""
    )
    output = tmp_path / "tiny.mesh"

    _converter_module().convert(source, output, verbose=False)

    text = output.read_text()
    assert "elements\n1\n1 4 0 1 2 3" in text
    assert "boundary\n4\n1 2 0 2 1" in text
    assert "vertices\n4\n3\n0 0 0" in text
