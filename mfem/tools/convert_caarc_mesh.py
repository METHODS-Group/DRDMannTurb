#!/usr/bin/env python3
"""Convert the mixed CAARC ASCII mesh to MFEM with explicit boundary triangles.

MFEM's VTK readers do not import lower-dimensional triangle cells alongside
tetrahedra as boundary elements. The original CAARC ``.msh`` stores a shared
vertex list followed by tetrahedron, triangle, and line blocks; this streaming
converter maps the tetrahedra to MFEM volume elements and triangles to MFEM
boundary elements without holding the large mesh in memory.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from pathlib import Path
from typing import TextIO

CHUNK_SIZE = 8 * 1024 * 1024


def _copy(source: TextIO, destination: TextIO) -> None:
    while chunk := source.read(CHUNK_SIZE):
        destination.write(chunk)


def convert(
    input_path: Path,
    output_path: Path,
    *,
    volume_attribute: int = 1,
    boundary_attribute: int = 1,
    verbose: bool = True,
) -> None:
    """Convert a CAARC mesh, streaming staged sections beside the output."""
    if volume_attribute < 1 or boundary_attribute < 1:
        raise ValueError("MFEM attributes must be positive")

    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary = {
        "vertices": output_path.with_suffix(output_path.suffix + ".tmp.vertices"),
        "elements": output_path.with_suffix(output_path.suffix + ".tmp.elements"),
        "boundary": output_path.with_suffix(output_path.suffix + ".tmp.boundary"),
    }
    vertex_count = 0
    element_count = 0
    boundary_count = 0
    maximum_node = -1
    expected_vertex_id = 1
    coordinates_written = False
    element_type: str | None = None
    mode: str | None = None

    if verbose:
        print(f"Reading  {input_path}")
        print(f"Writing  {output_path}")

    try:
        with ExitStack() as stack:
            source = stack.enter_context(input_path.open(encoding="utf-8", errors="replace"))
            vertices = stack.enter_context(temporary["vertices"].open("w", encoding="utf-8", newline="\n"))
            elements = stack.enter_context(temporary["elements"].open("w", encoding="utf-8", newline="\n"))
            boundary = stack.enter_context(temporary["boundary"].open("w", encoding="utf-8", newline="\n"))

            for raw_line in source:
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("MESH "):
                    words = line.split()
                    element_type = words[words.index("ElemType") + 1] if "ElemType" in words else None
                    mode = None
                    continue
                if line == "Coordinates":
                    mode = "coordinates"
                    continue
                if line == "End Coordinates":
                    coordinates_written = coordinates_written or vertex_count > 0
                    mode = None
                    continue
                if line == "Elements":
                    mode = "elements"
                    continue
                if line == "End Elements":
                    mode = None
                    continue

                columns = line.split()
                if mode == "coordinates" and not coordinates_written:
                    if len(columns) < 4:
                        raise ValueError(f"Malformed coordinate line: {line!r}")
                    vertex_id = int(columns[0])
                    if vertex_id != expected_vertex_id:
                        raise ValueError(f"Expected vertex {expected_vertex_id}, found {vertex_id}")
                    expected_vertex_id += 1
                    vertices.write(f"{columns[1]} {columns[2]} {columns[3]}\n")
                    vertex_count += 1
                elif mode == "elements" and element_type == "Tetrahedra":
                    if len(columns) < 5:
                        raise ValueError(f"Malformed tetrahedron line: {line!r}")
                    nodes = [int(value) - 1 for value in columns[1:5]]
                    maximum_node = max(maximum_node, *nodes)
                    elements.write(f"{volume_attribute} 4 {' '.join(map(str, nodes))}\n")
                    element_count += 1
                    if verbose and element_count % 500_000 == 0:
                        print(f"  staged {element_count:,} tetrahedra")
                elif mode == "elements" and element_type == "Triangle":
                    if len(columns) < 4:
                        raise ValueError(f"Malformed triangle line: {line!r}")
                    nodes = [int(value) - 1 for value in columns[1:4]]
                    maximum_node = max(maximum_node, *nodes)
                    boundary.write(f"{boundary_attribute} 2 {' '.join(map(str, nodes))}\n")
                    boundary_count += 1

        if not vertex_count or not element_count or not boundary_count:
            raise ValueError(
                "Input must contain non-empty Coordinates, Tetrahedra Elements, and Triangle Elements blocks"
            )
        if maximum_node >= vertex_count:
            raise ValueError(f"Connectivity references vertex {maximum_node}, but only {vertex_count} exist")

        with output_path.open("w", encoding="utf-8", newline="\n") as output:
            output.write("MFEM mesh v1.0\n\ndimension\n3\n\nelements\n")
            output.write(f"{element_count}\n")
            with temporary["elements"].open(encoding="utf-8") as section:
                _copy(section, output)
            output.write("\nboundary\n")
            output.write(f"{boundary_count}\n")
            with temporary["boundary"].open(encoding="utf-8") as section:
                _copy(section, output)
            output.write("\nvertices\n")
            output.write(f"{vertex_count}\n3\n")
            with temporary["vertices"].open(encoding="utf-8") as section:
                _copy(section, output)

        if verbose:
            print(f"Converted {vertex_count:,} vertices, {element_count:,} tetrahedra, and {boundary_count:,} faces")
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def main() -> None:
    """Parse command-line arguments and convert the requested mesh."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True, type=Path)
    parser.add_argument("-o", "--output", required=True, type=Path)
    parser.add_argument("--volume-attribute", type=int, default=1)
    parser.add_argument("--boundary-attribute", type=int, default=1)
    parser.add_argument("-q", "--quiet", action="store_true")
    args = parser.parse_args()
    convert(
        args.input,
        args.output,
        volume_attribute=args.volume_attribute,
        boundary_attribute=args.boundary_attribute,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
