# MFEM wind-tunnel integration

This directory builds the DRDMannTurb wind-tunnel application **outside** the
MFEM source tree. It links to an existing `libmfem` and compiles MFEM's
`miniapps/navier/navier_solver.cpp` as application support code. No files in the
MFEM checkout need to be edited.

## Build

MFEM must be built with MPI and Hypre. This integration is tested against MFEM
4.9 with HYPRE 3.1 and METIS 5. When using a modern METIS installation, make
sure MFEM is configured with its METIS 5 API rather than the legacy default:

```bash
make config \
  MFEM_USE_MPI=YES MFEM_USE_METIS_5=YES \
  HYPRE_DIR=/path/to/hypre \
  METIS_DIR=/path/to/metis
make -j4
```

Then build the consumer against an in-source MFEM build:

```bash
make -C mfem MFEM_DIR=/absolute/path/to/mfem
```

For an out-of-source MFEM build, also pass its build directory:

```bash
make -C mfem \
  MFEM_DIR=/absolute/path/to/mfem/source \
  MFEM_BUILD_DIR=/absolute/path/to/mfem/build
```

The Makefile reads MFEM's generated `config/config.mk`, so it inherits the same
compiler, MPI wrapper, include directories, link libraries, and feature flags as
the selected MFEM build.

## Check a DRDMannTurb VTI field

`FluctuationFieldGenerator.save_to_vtk()` writes a cell-centred vector named
`wind`. Check that C++ can read an exported field without running a flow solve:

```bash
mpirun -np 1 mfem/navier_windtunnel \
  --turbulence-field outputs/fluctuation_drd.vti \
  --check-field-only
```

The reader validates the VTI extents, spacing, component count, scalar type,
appended-data offset, byte count, and finite values. Both Float32 and Float64
fields written with UInt32 or UInt64 block headers are accepted.

## Small generated-domain run

```bash
mpirun -np 2 mfem/navier_windtunnel \
  --turbulence-field outputs/fluctuation_drd.vti \
  --order 1 --refine-serial 0 \
  --time-step 0.01 --final-time 0.10 \
  --paraview --paraview-name mfem/output/windtunnel
```

The VTI `wind` array is treated as the complete inlet velocity, including the
mean profile produced by `FluctuationFieldGenerator.generate()`. Taylor's frozen
turbulence hypothesis advances through its x-axis at
`--advection-velocity` (default: `--inlet-velocity`). The field wraps
periodically in all three axes.

Without `--turbulence-field`, the inlet uses the analytical profile selected by
`--inlet-profile`: 0 is constant, 1 is a power law, and 2 is logarithmic. Run
`mfem/navier_windtunnel --help` for all mesh, physics, output, and visualization
options.

## CAARC building run

The poster calculation uses the large CAARC volume mesh, which is intentionally
not duplicated in this repository. If only the original mixed CAARC `.msh` is
available, convert it once:

```bash
python mfem/tools/convert_caarc_mesh.py \
  --input /path/to/caarc.msh \
  --output mfem/meshes/caarc-mfem.mesh
```

Then run the tetrahedral mesh with partial assembly disabled:

```bash
mpirun -np 4 mfem/navier_windtunnel \
  --external-mesh --mesh mfem/meshes/caarc-mfem.mesh \
  --turbulence-field outputs/fluctuation_drd.vti \
  --order 1 --disable-pa --refine-serial 0 \
  --time-step 0.01 --final-time 2.0 \
  --paraview --paraview-name mfem/output/caarc --paraview-steps 20
```

Before committing to a large solve, the converted mesh and all seven boundary
classes can be checked without constructing the flow solver:

```bash
mpirun -np 1 mfem/navier_windtunnel \
  --external-mesh --mesh mfem/meshes/caarc-mfem.mesh \
  --check-mesh-only
```

External boundary faces are relabelled from the mesh bounding box: ground=1,
front=2, outlet=3, back=4, inlet=5, top=6, and the CAARC surface=7. All walls
and the building are currently no-slip; the outlet is natural.

## Poster rendering

After a ParaView run, use the reproducible render helper:

```bash
/path/to/pvpython mfem/render_paraview.py \
  --input mfem/output/caarc/caarc.pvd \
  --output mfem/output/caarc-poster.png
```

The helper creates a white-background, high-resolution velocity-magnitude view
with a horizontal colour legend and streamline tubes. Its camera and seed plane
can be overridden from the command line so the final frame can match the poster
layout without editing a ParaView state file.
