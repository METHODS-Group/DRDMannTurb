#!/usr/bin/env pvpython
"""Render a poster-ready streamline view from an MFEM ParaView collection."""

from __future__ import annotations

import argparse
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True, type=Path, help="MFEM .pvd collection")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Output .png file")
    parser.add_argument("--width", type=int, default=2400)
    parser.add_argument("--height", type=int, default=1600)
    parser.add_argument("--seed-resolution", type=int, default=24)
    parser.add_argument("--tube-radius", type=float, default=0.0, help="0 chooses a radius from the data bounds")
    parser.add_argument("--camera-position", type=float, nargs=3)
    parser.add_argument("--camera-focal-point", type=float, nargs=3)
    parser.add_argument("--time", type=float, help="Render the nearest available simulation time (default: last)")
    return parser


def main() -> None:
    """Build the ParaView pipeline and save a PNG."""
    args = _parser().parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from paraview.simple import (  # type: ignore[import-not-found]
        Calculator,
        CellDatatoPointData,
        CleantoGrid,
        ColorBy,
        ExtractSurface,
        GetActiveViewOrCreate,
        GetAnimationScene,
        GetColorTransferFunction,
        GetScalarBar,
        Hide,
        HideAll,
        MergeBlocks,
        OpenDataFile,
        Plane,
        ProgrammableFilter,
        Render,
        SaveScreenshot,
        Show,
        StreamTracerWithCustomSource,
        Threshold,
        Tube,
        _DisableFirstRenderCameraReset,
    )

    _DisableFirstRenderCameraReset()
    source = OpenDataFile(str(input_path))
    if source is None:
        raise RuntimeError(f"ParaView could not open {input_path}")

    scene = GetAnimationScene()
    scene.UpdateAnimationUsingDataTimeSteps()
    times = list(getattr(scene, "TimeKeeper").TimestepValues or [])
    if times:
        scene.AnimationTime = min(times, key=lambda value: abs(value - args.time)) if args.time else times[-1]
    source.UpdatePipeline(scene.AnimationTime)

    point_data = CellDatatoPointData(Input=source)
    point_data.ProcessAllArrays = 1
    magnitude = Calculator(Input=point_data)
    magnitude.ResultArrayName = "VelocityMagnitude"
    magnitude.Function = "mag(velocity)"
    magnitude.UpdatePipeline(scene.AnimationTime)

    merged = MergeBlocks(Input=magnitude)
    merged.MergePoints = 1
    merged.ToleranceIsAbsolute = 1
    merged.Tolerance = 0.0
    merged.UpdatePipeline(scene.AnimationTime)

    clean_grid = CleantoGrid(Input=merged)
    clean_grid.ToleranceIsAbsolute = 1
    clean_grid.AbsoluteTolerance = 0.0
    clean_grid.UpdatePipeline(scene.AnimationTime)

    bounds = clean_grid.GetDataInformation().GetBounds()
    if not bounds or bounds[1] <= bounds[0]:
        raise RuntimeError("The input collection has invalid or empty bounds")
    lengths = [bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]]
    scale = max(lengths)
    center = [(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2, (bounds[4] + bounds[5]) / 2]

    seed = Plane()
    seed.Origin = [bounds[0] + 0.01 * lengths[0], bounds[2] + 0.08 * lengths[1], bounds[4] + 0.05 * lengths[2]]
    seed.Point1 = [seed.Origin[0], bounds[3] - 0.08 * lengths[1], seed.Origin[2]]
    seed.Point2 = [seed.Origin[0], seed.Origin[1], bounds[5] - 0.05 * lengths[2]]
    seed.XResolution = args.seed_resolution
    seed.YResolution = args.seed_resolution

    streamlines = StreamTracerWithCustomSource(Input=clean_grid, SeedSource=seed)
    streamlines.Vectors = ["POINTS", "velocity"]
    streamlines.MaximumStreamlineLength = 1.25 * lengths[0]
    streamlines.InitialStepLength = 0.01
    streamlines.IntegrationDirection = "FORWARD"
    tubes = Tube(Input=streamlines)
    tubes.Radius = args.tube_radius if args.tube_radius > 0 else 0.00035 * scale
    tubes.NumberofSides = 8

    surface = ExtractSurface(Input=clean_grid)
    classified_surface = ProgrammableFilter(Input=surface)
    tolerance = max(scale * 1.0e-7, 1.0e-10)
    classified_surface.Script = f"""
import vtk

input_surface = inputs[0].VTKObject
output.ShallowCopy(input_surface)
domain_min = ({bounds[0]!r}, {bounds[2]!r}, {bounds[4]!r})
domain_max = ({bounds[1]!r}, {bounds[3]!r}, {bounds[5]!r})
tolerance = {tolerance!r}
is_building = vtk.vtkUnsignedCharArray()
is_building.SetName("IsBuilding")
is_building.SetNumberOfTuples(input_surface.GetNumberOfCells())
for cell_index in range(input_surface.GetNumberOfCells()):
    cell = input_surface.GetCell(cell_index)
    on_outer_plane = False
    for axis in range(3):
        for plane in (domain_min[axis], domain_max[axis]):
            on_plane = True
            for point_index in range(cell.GetNumberOfPoints()):
                point = input_surface.GetPoint(cell.GetPointId(point_index))
                if abs(point[axis] - plane) > tolerance:
                    on_plane = False
                    break
            if on_plane:
                on_outer_plane = True
                break
        if on_outer_plane:
            break
    is_building.SetValue(cell_index, 0 if on_outer_plane else 1)
output.GetCellData().AddArray(is_building)
"""
    classified_surface.UpdatePipeline(scene.AnimationTime)

    building = Threshold(Input=classified_surface)
    building.Scalars = ["CELLS", "IsBuilding"]
    building.ThresholdMethod = "Between"
    building.LowerThreshold = 0.5
    building.UpperThreshold = 1.5
    building.UpdatePipeline(scene.AnimationTime)
    building_cells = building.GetDataInformation().GetNumberOfCells()
    print(f"Detected {building_cells} internal building surface cells")

    view = GetActiveViewOrCreate("RenderView")
    view.ViewSize = [args.width, args.height]
    view.BackgroundColorMode = "Single Color"
    view.Background = [1.0, 1.0, 1.0]
    view.UseColorPaletteForBackground = 0
    view.OrientationAxesVisibility = 0
    HideAll(view)

    building_display = None
    if building_cells > 0:
        building_display = Show(building, view, "GeometryRepresentation")
        building_display.Representation = "Surface"
        building_display.Opacity = 0.85
        ColorBy(building_display, ("POINTS", "VelocityMagnitude"))

    tube_display = Show(tubes, view, "GeometryRepresentation")
    tube_display.Representation = "Surface"
    ColorBy(tube_display, ("POINTS", "VelocityMagnitude"))
    Hide(source, view)
    Hide(point_data, view)
    Hide(magnitude, view)
    Hide(merged, view)
    Hide(clean_grid, view)
    Hide(seed, view)
    Hide(streamlines, view)
    Hide(surface, view)
    Hide(classified_surface, view)

    colors = GetColorTransferFunction("VelocityMagnitude")
    colors.ApplyPreset("Turbo", True)
    tube_display.RescaleTransferFunctionToDataRange(False, True)
    if building_display is not None:
        building_display.RescaleTransferFunctionToDataRange(False, True)
    scalar_bar = GetScalarBar(colors, view)
    tube_display.SetScalarBarVisibility(view, True)
    scalar_bar.Title = "Velocity magnitude"
    scalar_bar.ComponentTitle = "m/s"
    scalar_bar.Orientation = "Horizontal"
    scalar_bar.WindowLocation = "Upper Center"
    scalar_bar.ScalarBarLength = 0.42
    scalar_bar.TitleColor = [0.1, 0.1, 0.1]
    scalar_bar.LabelColor = [0.1, 0.1, 0.1]

    view.ResetCamera()
    view.CameraFocalPoint = args.camera_focal_point or center
    view.CameraPosition = args.camera_position or [
        bounds[1] + 0.32 * scale,
        bounds[2] - 0.72 * scale,
        bounds[5] + 0.38 * scale,
    ]
    view.CameraViewUp = [0.0, 0.0, 1.0]
    view.CameraParallelProjection = 1
    view.CameraParallelScale *= 0.88

    Render(view)
    SaveScreenshot(
        str(output_path),
        view,
        ImageResolution=[args.width, args.height],
        OverrideColorPalette="WhiteBackground",
    )
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
