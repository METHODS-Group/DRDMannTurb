// Copyright (c) 2010-2026, Lawrence Livermore National Security, LLC.
// Copyright (c) 2026, DRDMannTurb contributors.
//
// This application is derived from MFEM's Navier wind-tunnel miniapp and is
// distributed under the BSD-3-Clause terms used by MFEM. It is kept here so
// DRDMannTurb-to-MFEM compatibility can be built and checked out of tree.

#include "navier_solver.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace mfem;
using namespace mfem::navier;

namespace
{

struct Context
{
   int serial_refinements = 0;
   int order = 1;
   int nx = 6;
   int ny = 2;
   int nz = 2;
   int inlet_profile = 0;
   int paraview_steps = 1;
   int vis_port = 19916;

   real_t length = 300.0;
   real_t width = 100.0;
   real_t height = 100.0;
   real_t viscosity = 0.1;
   real_t final_time = 0.02;
   real_t time_step = 0.01;
   real_t inlet_velocity = 1.0;
   real_t inlet_ramp_time = 0.0;
   real_t reference_height = 90.0;
   real_t power_alpha = 0.2;
   real_t roughness_length = 0.02;
   real_t friction_velocity = 0.556;
   real_t von_karman = 0.4;
   real_t advection_velocity = -1.0;
   real_t boundary_tolerance = 1.0e-8;

   bool partial_assembly = true;
   bool numerical_integration = false;
   bool visualization = false;
   bool paraview = false;
   bool external_mesh = false;
   bool relabel_boundaries = true;
   bool check_field_only = false;
   bool check_mesh_only = false;

   std::string mesh_file;
   std::string turbulence_file;
   std::string paraview_name = "mfem/output/navier_windtunnel";
} ctx;

std::string Attribute(const std::string &text, const std::string &name)
{
   const std::string prefix = name + "=\"";
   const std::size_t begin = text.find(prefix);
   if (begin == std::string::npos)
   {
      throw std::runtime_error("VTI attribute not found: " + name);
   }
   const std::size_t value_begin = begin + prefix.size();
   const std::size_t end = text.find('"', value_begin);
   if (end == std::string::npos)
   {
      throw std::runtime_error("Unterminated VTI attribute: " + name);
   }
   return text.substr(value_begin, end - value_begin);
}

std::string NormalizeLegacyNumbers(std::string value)
{
   const std::string wrapper = "np.float64(";
   std::size_t position = 0;
   while ((position = value.find(wrapper, position)) != std::string::npos)
   {
      value.erase(position, wrapper.size());
   }
   std::replace(value.begin(), value.end(), ')', ' ');
   return value;
}

template <typename T, std::size_t N>
std::array<T, N> ParseArray(const std::string &value, const std::string &name)
{
   std::istringstream stream(NormalizeLegacyNumbers(value));
   std::array<T, N> result{};
   for (std::size_t i = 0; i < N; ++i)
   {
      if (!(stream >> result[i]))
      {
         throw std::runtime_error("Could not parse VTI attribute " + name + ": " + value);
      }
   }
   return result;
}

class VtiVectorField
{
public:
   explicit VtiVectorField(const std::string &path) { Load(path); }

   void Sample(real_t x, real_t y, real_t z, Vector &value) const
   {
      const std::array<real_t, 3> point{x, y, z};
      std::array<int, 3> lower{};
      std::array<int, 3> upper{};
      std::array<real_t, 3> fraction{};

      for (int axis = 0; axis < 3; ++axis)
      {
         // The VTI array is cell-centred, so cell i is located at i + 1/2.
         const real_t coordinate = (point[axis] - origin_[axis]) / spacing_[axis] - 0.5;
         const auto base = static_cast<long long>(std::floor(coordinate));
         fraction[axis] = coordinate - static_cast<real_t>(base);
         lower[axis] = Wrap(base, dimensions_[axis]);
         upper[axis] = Wrap(base + 1, dimensions_[axis]);
      }

      value.SetSize(3);
      value = 0.0;
      for (int component = 0; component < 3; ++component)
      {
         for (int corner = 0; corner < 8; ++corner)
         {
            const bool high_x = corner & 1;
            const bool high_y = corner & 2;
            const bool high_z = corner & 4;
            const int ix = high_x ? upper[0] : lower[0];
            const int iy = high_y ? upper[1] : lower[1];
            const int iz = high_z ? upper[2] : lower[2];
            const real_t weight =
               (high_x ? fraction[0] : 1.0 - fraction[0]) *
               (high_y ? fraction[1] : 1.0 - fraction[1]) *
               (high_z ? fraction[2] : 1.0 - fraction[2]);
            value(component) += weight * data_[Index(ix, iy, iz, component)];
         }
      }
   }

   real_t XCenter() const { return origin_[0] + 0.5 * spacing_[0]; }

   void PrintSummary(std::ostream &out) const
   {
      std::array<real_t, 3> minimum{
         std::numeric_limits<real_t>::infinity(),
         std::numeric_limits<real_t>::infinity(),
         std::numeric_limits<real_t>::infinity(),
      };
      std::array<real_t, 3> maximum{
         -std::numeric_limits<real_t>::infinity(),
         -std::numeric_limits<real_t>::infinity(),
         -std::numeric_limits<real_t>::infinity(),
      };
      for (std::size_t i = 0; i < data_.size(); i += 3)
      {
         for (int component = 0; component < 3; ++component)
         {
            minimum[component] = std::min(minimum[component], data_[i + component]);
            maximum[component] = std::max(maximum[component], data_[i + component]);
         }
      }

      out << "Loaded DRDMannTurb VTI field\n"
          << "  cells:   " << dimensions_[0] << " x " << dimensions_[1] << " x " << dimensions_[2] << '\n'
          << "  origin:  " << origin_[0] << ' ' << origin_[1] << ' ' << origin_[2] << '\n'
          << "  spacing: " << spacing_[0] << ' ' << spacing_[1] << ' ' << spacing_[2] << '\n';
      for (int component = 0; component < 3; ++component)
      {
         out << "  component " << component << " range: [" << minimum[component] << ", " << maximum[component]
             << "]\n";
      }
   }

private:
   std::array<int, 3> dimensions_{};
   std::array<real_t, 3> origin_{};
   std::array<real_t, 3> spacing_{};
   std::vector<real_t> data_;

   static int Wrap(long long index, int size)
   {
      const long long remainder = index % size;
      return static_cast<int>(remainder < 0 ? remainder + size : remainder);
   }

   std::size_t Index(int ix, int iy, int iz, int component) const
   {
      // VTK image arrays use x-fastest (Fortran-flattened) cell ordering and
      // pyevtk interleaves the three vector components per cell.
      return static_cast<std::size_t>(
         (ix + dimensions_[0] * (iy + dimensions_[1] * iz)) * 3 + component);
   }

   template <typename T>
   static T ReadScalar(const std::vector<char> &buffer, std::size_t position)
   {
      if (position + sizeof(T) > buffer.size())
      {
         throw std::runtime_error("VTI appended-data header extends beyond end of file");
      }
      T value;
      std::memcpy(&value, buffer.data() + position, sizeof(T));
      return value;
   }

   void Load(const std::string &path)
   {
      std::ifstream input(path, std::ios::binary | std::ios::ate);
      if (!input)
      {
         throw std::runtime_error("Could not open turbulence field: " + path);
      }
      const std::streamsize stream_size = input.tellg();
      if (stream_size <= 0)
      {
         throw std::runtime_error("Turbulence field is empty: " + path);
      }
      input.seekg(0, std::ios::beg);
      std::vector<char> buffer(static_cast<std::size_t>(stream_size));
      if (!input.read(buffer.data(), stream_size))
      {
         throw std::runtime_error("Could not read turbulence field: " + path);
      }

      const std::string contents(buffer.data(), buffer.size());
      const std::size_t appended_tag = contents.find("<AppendedData");
      if (appended_tag == std::string::npos)
      {
         throw std::runtime_error("VTI has no AppendedData section: " + path);
      }
      const std::size_t raw_marker = contents.find('_', appended_tag);
      if (raw_marker == std::string::npos)
      {
         throw std::runtime_error("VTI AppendedData section has no raw-data marker");
      }
      const std::string header = contents.substr(0, appended_tag);

      if (Attribute(header, "byte_order") != "LittleEndian")
      {
         throw std::runtime_error("Only LittleEndian VTI fields are currently supported");
      }

      const auto extent = ParseArray<int, 6>(Attribute(header, "WholeExtent"), "WholeExtent");
      for (int axis = 0; axis < 3; ++axis)
      {
         dimensions_[axis] = extent[2 * axis + 1] - extent[2 * axis];
         if (dimensions_[axis] < 2)
         {
            throw std::runtime_error("VTI wind field needs at least two cells in every dimension");
         }
      }
      origin_ = ParseArray<real_t, 3>(Attribute(header, "Origin"), "Origin");
      spacing_ = ParseArray<real_t, 3>(Attribute(header, "Spacing"), "Spacing");
      for (real_t spacing : spacing_)
      {
         if (!(spacing > 0.0) || !std::isfinite(spacing))
         {
            throw std::runtime_error("VTI Spacing values must be positive and finite");
         }
      }

      const std::size_t name_position = header.find("Name=\"wind\"");
      if (name_position == std::string::npos)
      {
         throw std::runtime_error("VTI CellData has no vector named 'wind'");
      }
      const std::size_t tag_begin = header.rfind("<DataArray", name_position);
      const std::size_t tag_end = header.find("/>", name_position);
      if (tag_begin == std::string::npos || tag_end == std::string::npos)
      {
         throw std::runtime_error("Malformed VTI DataArray tag for 'wind'");
      }
      const std::string wind_tag = header.substr(tag_begin, tag_end + 2 - tag_begin);
      if (Attribute(wind_tag, "NumberOfComponents") != "3")
      {
         throw std::runtime_error("VTI 'wind' array must have three components");
      }
      if (Attribute(wind_tag, "format") != "appended")
      {
         throw std::runtime_error("VTI 'wind' array must use appended format");
      }

      const std::size_t offset = std::stoull(Attribute(wind_tag, "offset"));
      const std::size_t block_position = raw_marker + 1 + offset;
      const std::string header_type = Attribute(header, "header_type");
      std::uint64_t block_bytes = 0;
      std::size_t scalar_position = block_position;
      if (header_type == "UInt64")
      {
         block_bytes = ReadScalar<std::uint64_t>(buffer, block_position);
         scalar_position += sizeof(std::uint64_t);
      }
      else if (header_type == "UInt32")
      {
         block_bytes = ReadScalar<std::uint32_t>(buffer, block_position);
         scalar_position += sizeof(std::uint32_t);
      }
      else
      {
         throw std::runtime_error("Unsupported VTI header_type: " + header_type);
      }

      const std::size_t values = static_cast<std::size_t>(dimensions_[0]) * dimensions_[1] * dimensions_[2] * 3;
      const std::string scalar_type = Attribute(wind_tag, "type");
      const std::size_t scalar_bytes = scalar_type == "Float64" ? sizeof(double) :
                                       scalar_type == "Float32" ? sizeof(float) : 0;
      if (scalar_bytes == 0)
      {
         throw std::runtime_error("Unsupported VTI wind scalar type: " + scalar_type);
      }
      const std::size_t expected_bytes = values * scalar_bytes;
      if (block_bytes != expected_bytes || scalar_position + expected_bytes > buffer.size())
      {
         throw std::runtime_error(
            "VTI wind byte count mismatch: expected " + std::to_string(expected_bytes) +
            ", found " + std::to_string(block_bytes));
      }

      data_.resize(values);
      if (scalar_type == "Float64")
      {
         std::vector<double> source(values);
         std::memcpy(source.data(), buffer.data() + scalar_position, expected_bytes);
         std::copy(source.begin(), source.end(), data_.begin());
      }
      else
      {
         std::vector<float> source(values);
         std::memcpy(source.data(), buffer.data() + scalar_position, expected_bytes);
         std::copy(source.begin(), source.end(), data_.begin());
      }
      if (!std::all_of(data_.begin(), data_.end(), [](real_t value) { return std::isfinite(value); }))
      {
         throw std::runtime_error("VTI wind array contains non-finite values");
      }
   }
};

std::unique_ptr<VtiVectorField> turbulence_field;

real_t Ramp(real_t time)
{
   if (ctx.inlet_ramp_time <= 0.0)
   {
      return 1.0;
   }
   const real_t s = std::clamp(time / ctx.inlet_ramp_time, real_t(0.0), real_t(1.0));
   return s * s * (3.0 - 2.0 * s);
}

void InletVelocity(const Vector &position, real_t time, Vector &velocity)
{
   velocity.SetSize(3);
   velocity = 0.0;

   if (turbulence_field)
   {
      const real_t x = turbulence_field->XCenter() + ctx.advection_velocity * time;
      turbulence_field->Sample(x, position(1), position(2), velocity);
   }
   else
   {
      const real_t height = std::max(position(2), real_t(0.0));
      if (ctx.inlet_profile == 0)
      {
         velocity(0) = ctx.inlet_velocity;
      }
      else if (ctx.inlet_profile == 1)
      {
         velocity(0) = ctx.inlet_velocity *
                       std::pow(height / ctx.reference_height, ctx.power_alpha);
      }
      else
      {
         velocity(0) = (ctx.friction_velocity / ctx.von_karman) *
                       std::log1p(height / ctx.roughness_length);
      }
   }
   velocity *= Ramp(time);
}

void RelabelWindTunnelBoundaries(Mesh &mesh, real_t relative_tolerance)
{
   MFEM_VERIFY(relative_tolerance > 0.0, "Boundary relabel tolerance must be positive.");
   Vector minimum;
   Vector maximum;
   mesh.GetBoundingBox(minimum, maximum);
   real_t scale = 0.0;
   for (int axis = 0; axis < 3; ++axis)
   {
      scale = std::max(scale, maximum(axis) - minimum(axis));
   }
   const real_t tolerance = relative_tolerance * scale;
   Array<int> vertices;

   for (int boundary = 0; boundary < mesh.GetNBE(); ++boundary)
   {
      mesh.GetBdrElementVertices(boundary, vertices);
      auto on_plane = [&](int axis, real_t coordinate)
      {
         for (int i = 0; i < vertices.Size(); ++i)
         {
            if (std::abs(mesh.GetVertex(vertices[i])[axis] - coordinate) > tolerance)
            {
               return false;
            }
         }
         return true;
      };

      int attribute = 7; // Any non-box boundary is the building/obstacle.
      if (on_plane(0, minimum(0))) { attribute = 5; }
      else if (on_plane(0, maximum(0))) { attribute = 3; }
      else if (on_plane(2, minimum(2))) { attribute = 1; }
      else if (on_plane(1, minimum(1))) { attribute = 2; }
      else if (on_plane(1, maximum(1))) { attribute = 4; }
      else if (on_plane(2, maximum(2))) { attribute = 6; }
      mesh.GetBdrElement(boundary)->SetAttribute(attribute);
   }
   mesh.SetAttributes();
}

std::vector<int> BoundaryCounts(const Mesh &mesh)
{
   std::vector<int> counts(static_cast<std::size_t>(mesh.bdr_attributes.Max() + 1), 0);
   for (int boundary = 0; boundary < mesh.GetNBE(); ++boundary)
   {
      ++counts[mesh.GetBdrAttribute(boundary)];
   }
   return counts;
}

void PrintMeshSummary(const Mesh &mesh)
{
   mfem::out << "Mesh dimension: " << mesh.Dimension() << '\n'
             << "Elements: " << mesh.GetNE() << '\n'
             << "Boundary elements: " << mesh.GetNBE() << '\n'
             << "Boundary attributes:\n";
   const auto counts = BoundaryCounts(mesh);
   for (std::size_t attribute = 1; attribute < counts.size(); ++attribute)
   {
      if (counts[attribute] > 0)
      {
         mfem::out << "  " << attribute << ": " << counts[attribute] << '\n';
      }
   }
}

void Mark(Array<int> &marker, std::initializer_list<int> attributes)
{
   marker = 0;
   for (int attribute : attributes)
   {
      if (attribute <= marker.Size())
      {
         marker[attribute - 1] = 1;
      }
   }
}

} // namespace

int main(int argc, char *argv[])
{
   Mpi::Init(argc, argv);
   Hypre::Init();
   HYPRE_SetPrintErrorMode(0);
   HYPRE_SetPrintErrorVerbosity(-1, 1);

   OptionsParser args(argc, argv);
   args.AddOption(&ctx.mesh_file, "-m", "--mesh", "External MFEM mesh file.");
   args.AddOption(&ctx.external_mesh, "-em", "--external-mesh", "-no-em", "--no-external-mesh",
                  "Load the mesh passed with --mesh instead of generating a box.");
   args.AddOption(&ctx.relabel_boundaries, "-rb", "--relabel-boundaries", "-no-rb", "--no-relabel-boundaries",
                  "Relabel external boundaries from the mesh bounding box.");
   args.AddOption(&ctx.boundary_tolerance, "-bdrtol", "--boundary-relabel-tol",
                  "Relative tolerance for boundary relabelling.");
   args.AddOption(&ctx.serial_refinements, "-rs", "--refine-serial", "Serial uniform refinements.");
   args.AddOption(&ctx.order, "-o", "--order", "Finite-element degree.");
   args.AddOption(&ctx.nx, "-nx", "--elements-x", "Generated-box elements in x.");
   args.AddOption(&ctx.ny, "-ny", "--elements-y", "Generated-box elements in y.");
   args.AddOption(&ctx.nz, "-nz", "--elements-z", "Generated-box elements in z.");
   args.AddOption(&ctx.length, "-lx", "--length", "Generated-box length.");
   args.AddOption(&ctx.width, "-ly", "--width", "Generated-box width.");
   args.AddOption(&ctx.height, "-lz", "--height", "Generated-box height.");
   args.AddOption(&ctx.viscosity, "-nu", "--kinematic-viscosity", "Kinematic viscosity.");
   args.AddOption(&ctx.time_step, "-dt", "--time-step", "Time step.");
   args.AddOption(&ctx.final_time, "-tf", "--final-time", "Final time.");
   args.AddOption(&ctx.inlet_ramp_time, "-tr", "--inlet-ramp-time", "Smoothstep inlet ramp duration.");
   args.AddOption(&ctx.inlet_profile, "-profile", "--inlet-profile",
                  "Analytical inlet: 0=constant, 1=power, 2=logarithmic.");
   args.AddOption(&ctx.inlet_velocity, "-u0", "--inlet-velocity", "Constant/reference inlet velocity.");
   args.AddOption(&ctx.reference_height, "-href", "--reference-height", "Power-law reference height.");
   args.AddOption(&ctx.power_alpha, "-alpha", "--power-alpha", "Power-law exponent.");
   args.AddOption(&ctx.roughness_length, "-z0", "--roughness-length", "Log-law roughness length.");
   args.AddOption(&ctx.friction_velocity, "-ustar", "--friction-velocity", "Log-law friction velocity.");
   args.AddOption(&ctx.turbulence_file, "-turb", "--turbulence-field",
                  "DRDMannTurb VTI field; replaces the analytical inlet profile.");
   args.AddOption(&ctx.advection_velocity, "-uadv", "--advection-velocity",
                  "Frozen-turbulence advection speed (defaults to inlet velocity).");
   args.AddOption(&ctx.check_field_only, "-check-field", "--check-field-only", "-no-check-field",
                  "--no-check-field-only", "Validate the turbulence field and exit.");
   args.AddOption(&ctx.check_mesh_only, "-check-mesh", "--check-mesh-only", "-no-check-mesh",
                  "--no-check-mesh-only", "Load, relabel, and validate the mesh, then exit.");
   args.AddOption(&ctx.partial_assembly, "-pa", "--enable-pa", "-no-pa", "--disable-pa",
                  "Enable partial assembly.");
   args.AddOption(&ctx.numerical_integration, "-ni", "--enable-ni", "-no-ni", "--disable-ni",
                  "Enable numerical integration rules.");
   args.AddOption(&ctx.paraview, "-pv", "--paraview", "-no-pv", "--no-paraview",
                  "Write ParaView data.");
   args.AddOption(&ctx.paraview_name, "-pvn", "--paraview-name", "ParaView collection path/name.");
   args.AddOption(&ctx.paraview_steps, "-pvs", "--paraview-steps", "Write every N time steps.");
   args.AddOption(&ctx.visualization, "-vis", "--visualization", "-no-vis", "--no-visualization",
                  "Send the final velocity to GLVis.");
   args.AddOption(&ctx.vis_port, "-p", "--send-port", "GLVis socket port.");
   args.Parse();

   if (!args.Good())
   {
      if (Mpi::Root()) { args.PrintUsage(mfem::out); }
      return 1;
   }
   if (Mpi::Root()) { args.PrintOptions(mfem::out); }

   MFEM_VERIFY(ctx.order >= 1, "Finite-element order must be at least one.");
   MFEM_VERIFY(ctx.time_step > 0.0 && ctx.final_time > 0.0, "Time step and final time must be positive.");
   MFEM_VERIFY(ctx.paraview_steps >= 1, "ParaView step interval must be at least one.");
   MFEM_VERIFY(ctx.inlet_profile >= 0 && ctx.inlet_profile <= 2, "Inlet profile must be 0, 1, or 2.");
   MFEM_VERIFY(ctx.reference_height > 0.0 && ctx.roughness_length > 0.0,
               "Reference height and roughness length must be positive.");
   if (ctx.advection_velocity < 0.0) { ctx.advection_velocity = ctx.inlet_velocity; }

   try
   {
      if (!ctx.turbulence_file.empty())
      {
         turbulence_field = std::make_unique<VtiVectorField>(ctx.turbulence_file);
         if (Mpi::Root()) { turbulence_field->PrintSummary(mfem::out); }
      }
   }
   catch (const std::exception &error)
   {
      if (Mpi::Root()) { mfem::err << "Turbulence-field error: " << error.what() << '\n'; }
      return 2;
   }
   if (ctx.check_field_only)
   {
      if (!turbulence_field && Mpi::Root())
      {
         mfem::err << "--check-field-only requires --turbulence-field.\n";
      }
      return turbulence_field ? 0 : 2;
   }

   const bool use_external_mesh = ctx.external_mesh || !ctx.mesh_file.empty();
   std::unique_ptr<Mesh> mesh;
   if (use_external_mesh)
   {
      MFEM_VERIFY(!ctx.mesh_file.empty(), "--external-mesh requires --mesh.");
      mesh = std::make_unique<Mesh>(ctx.mesh_file.c_str(), 1, 1, true);
      MFEM_VERIFY(mesh->Dimension() == 3, "The wind-tunnel mesh must be three-dimensional.");
   }
   else
   {
      MFEM_VERIFY(ctx.nx > 0 && ctx.ny > 0 && ctx.nz > 0, "Generated mesh dimensions must be positive.");
      mesh = std::make_unique<Mesh>(Mesh::MakeCartesian3D(
         ctx.nx, ctx.ny, ctx.nz, Element::HEXAHEDRON, ctx.length, ctx.width, ctx.height));
   }
   if (ctx.relabel_boundaries) { RelabelWindTunnelBoundaries(*mesh, ctx.boundary_tolerance); }

   for (int refinement = 0; refinement < ctx.serial_refinements; ++refinement)
   {
      mesh->UniformRefinement();
   }
   if (mesh->GetElementType(0) != Element::HEXAHEDRON)
   {
      if (ctx.partial_assembly && Mpi::Root())
      {
         mfem::out << "Disabling partial assembly for a non-hexahedral mesh.\n";
      }
      ctx.partial_assembly = false;
      ctx.numerical_integration = false;
   }
   if (Mpi::Root()) { PrintMeshSummary(*mesh); }
   const auto boundary_counts = BoundaryCounts(*mesh);
   MFEM_VERIFY(boundary_counts.size() > 5 && boundary_counts[5] > 0,
               "Mesh has no inlet boundary attribute 5.");
   if (ctx.check_mesh_only) { return 0; }

   auto pmesh = std::make_unique<ParMesh>(MPI_COMM_WORLD, *mesh);
   mesh.reset();
   NavierSolver flow(pmesh.get(), ctx.order, ctx.viscosity);
   flow.EnablePA(ctx.partial_assembly);
   flow.EnableNI(ctx.numerical_integration);

   ParGridFunction *velocity = flow.GetCurrentVelocity();
   Vector zero(3);
   zero = 0.0;
   VectorConstantCoefficient zero_vector(zero);
   velocity->ProjectCoefficient(zero_vector);

   const int max_boundary_attribute = pmesh->bdr_attributes.Max();
   Array<int> walls(max_boundary_attribute);
   Mark(walls, {1, 2, 4, 6, 7});
   flow.AddVelDirichletBC(new VectorConstantCoefficient(zero), walls);

   Array<int> inlet(max_boundary_attribute);
   Mark(inlet, {5});
   flow.AddVelDirichletBC(InletVelocity, inlet);
   // Attribute 3 is the natural outflow boundary.

   real_t time = 0.0;
   flow.Setup(ctx.time_step);
   ParGridFunction *pressure = flow.GetCurrentPressure();

   std::unique_ptr<ParaViewDataCollection> paraview;
   if (ctx.paraview)
   {
      paraview = std::make_unique<ParaViewDataCollection>(ctx.paraview_name, pmesh.get());
      paraview->SetDataFormat(VTKFormat::BINARY32);
      // Conforming output keeps shared surface connectivity, which lets the
      // renderer distinguish the outer tunnel from an internal CAARC surface.
      paraview->SetHighOrderOutput(false);
      paraview->RegisterField("velocity", velocity);
      paraview->RegisterField("pressure", pressure);
      paraview->SetCycle(0);
      paraview->SetTime(time);
      paraview->Save();
   }

   if (Mpi::Root())
   {
      mfem::out << std::setw(7) << "step" << std::setw(13) << "time" << std::setw(13) << "CFL"
                << std::setw(16) << "||u||_inf" << '\n';
   }

   int step = 0;
   while (time < ctx.final_time - ctx.time_step / 2.0)
   {
      const bool last_step = time + ctx.time_step >= ctx.final_time - ctx.time_step / 2.0;
      flow.Step(time, ctx.time_step, step);
      ++step;

      if (paraview && (step % ctx.paraview_steps == 0 || last_step))
      {
         paraview->SetCycle(step);
         paraview->SetTime(time);
         paraview->Save();
      }

      const real_t cfl = flow.ComputeCFL(*velocity, ctx.time_step);
      const real_t local_norm = velocity->Normlinf();
      real_t global_norm = 0.0;
      MPI_Allreduce(&local_norm, &global_norm, 1, MPITypeMap<real_t>::mpi_type, MPI_MAX, MPI_COMM_WORLD);
      if (Mpi::Root())
      {
         mfem::out << std::setw(7) << step << std::setw(13) << std::setprecision(5) << time
                   << std::setw(13) << cfl << std::setw(16) << global_norm << '\n';
      }
   }

   if (ctx.visualization)
   {
      socketstream socket("localhost", ctx.vis_port);
      socket.precision(8);
      socket << "parallel " << Mpi::WorldSize() << ' ' << Mpi::WorldRank() << '\n'
             << "solution\n" << *pmesh << *velocity << std::flush;
   }

   flow.PrintTimingData();
   return 0;
}
