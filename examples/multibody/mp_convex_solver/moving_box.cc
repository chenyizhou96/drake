#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>

#include <gflags/gflags.h>

#include "drake/common/nice_type_name.h"
#include "drake/common/temp_directory.h"
#include "drake/geometry/drake_visualizer.h"
#include "drake/multibody/plant/externally_applied_spatial_force.h"
#include "drake/geometry/scene_graph.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/math/random_rotation.h"
#include "drake/math/rigid_transform.h"
#include "drake/math/roll_pitch_yaw.h"
#include "drake/math/rotation_matrix.h"
#include "drake/multibody/contact_solvers/unconstrained_primal_solver.h"
#include "drake/multibody/contact_solvers/admm_solver.h"
#include "drake/multibody/plant/compliant_contact_computation_manager.h"
#include "drake/multibody/plant/contact_results.h"
#include "drake/multibody/plant/point_pair_contact_info.h"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/systems/analysis/implicit_integrator.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/analysis/simulator_gflags.h"
#include "drake/systems/analysis/simulator_print_stats.h"
#include "drake/systems/framework/diagram_builder.h"

// To profile with Valgrind run with (the defaults are good):
// valgrind --tool=callgrind --instr-atstart=no
// bazel-bin/examples/multibody/mp_convex_solver/clutter
#include <valgrind/callgrind.h>
namespace drake {
namespace multibody {
namespace examples {
namespace mp_convex_solver {
namespace {

// Simulation parameters.
DEFINE_double(simulation_time, 5.0, "Simulation duration in seconds");
DEFINE_double(
    mbp_time_step, 1E-2,
    "If mbp_time_step > 0, the fixed-time step period (in seconds) of discrete "
    "updates for the plant (modeled as a discrete system). "
    "If mbp_time_step = 0, the plant is modeled as a continuous system "
    "and no contact forces are displayed.  mbp_time_step must be >= 0.");
// For this demo, penetration_allowance and stiction_tolerance are only used
// when either continuous integration or TAMSI are used.
DEFINE_double(penetration_allowance, 5.0E-4, "Allowable penetration (meters).");
DEFINE_double(stiction_tolerance, 1.0E-4,
              "Allowable drift speed during stiction (m/s).");

// Physical parameters.
DEFINE_double(density, 1000.0, "The density of all objects, in kg/m³.");
DEFINE_double(friction_coefficient, 1.0,
              "All friction coefficients have this value.");
DEFINE_double(stiffness, 1.0E3, "Point contact stiffness in N/m."); //For boxes: 1.0e8 recommended
DEFINE_double(dissipation_rate, 0.01, "Linear dissipation rate in seconds.");

// Contact geometry parameters.
DEFINE_int32(
    num_spheres_per_face, 3,
    "Multi-contact emulation. We place num_sphere x num_spheres_per_face on "
    "each box face, when emulate_box_multicontact = true.");

// Visualization.
DEFINE_bool(visualize, true, "Whether to visualize (true) or not (false).");
DEFINE_bool(visualize_forces, false,
            "Whether to visualize forces (true) or not (false).");
DEFINE_bool(visualize_multicontact, false,
            "Whether to visualize (true) or not the spheres used to emulate "
            "multicontact.");
DEFINE_double(viz_period, 1.0 / 60.0, "Viz period.");

// Discrete contact solver.
//DEFINE_bool(tamsi, false, "Use TAMSI (true) or new solver (false).");
DEFINE_int32(verbosity_level, 0,
             "Verbosity level of the new primal solver. See "
             "UnconstrainedPrimalSolverParameters.");
DEFINE_string(line_search, "exact",
              "Primal solver line-search. 'exact', 'inexact'");
DEFINE_double(ls_alpha_max, 1.5, "Maximum line search step.");
DEFINE_double(rt_factor, 1.0e-3, "Rt_factor");
DEFINE_double(abs_tol, 1.0e-6, "Absolute tolerance [m/s].");
DEFINE_double(rel_tol, 1.0e-4, "Relative tolerance [-].");
DEFINE_int32(solver_type, 2, "define solver type, 0 for TAMSI, 1 for unconstrained primal solver, 2 for admm solver");
DEFINE_double(radius0, 0.05, "radius or side length/2 for the smallest sphere/box");
DEFINE_int32(max_iterations, 100, "max iterations for the admm solver, for debugging purpose");
DEFINE_bool(initialize_force, true, "admm solver specific: whether initialize force analytically");
DEFINE_double(input_force_x, 1.0, "defines the x direction of the external force on the box, y and z dir are 0");
DEFINE_bool(dynamic_rho, false, "whether or not to use dynamic rho for admm solver");
DEFINE_double(rho, 1.0, "rho parameter in admm to start with");

using drake::math::RigidTransform;
using drake::math::RigidTransformd;
using drake::math::RollPitchYawd;
using drake::math::RotationMatrixd;
using drake::multibody::ContactResults;
using drake::multibody::MultibodyPlant;
using drake::multibody::contact_solvers::internal::UnconstrainedPrimalSolver;
using drake::multibody::contact_solvers::internal::
    UnconstrainedPrimalSolverIterationMetrics;
using drake::multibody::contact_solvers::internal::
    UnconstrainedPrimalSolverParameters;
using drake::multibody::contact_solvers::internal::
    UnconstrainedPrimalSolverStats;
using drake::multibody::contact_solvers::internal::AdmmSolver;
using drake::multibody::contact_solvers::internal::
    AdmmSolverIterationMetrics;
using drake::multibody::contact_solvers::internal::
    AdmmSolverParameters;
using drake::multibody::contact_solvers::internal::
    AdmmSolverStats;
using Eigen::Translation3d;
using Eigen::Vector3d;
using clock = std::chrono::steady_clock;


std::vector<geometry::GeometryId> box_geometry_ids;


const RigidBody<double>& AddBox(const std::string& name,
                                const Vector3<double>& block_dimensions,
                                double mass, double friction,
                                const Vector4<double>& color,
                                bool emulate_box_multicontact,
                                MultibodyPlant<double>* plant) {
  // Ensure the block's dimensions are mass are positive.
  const double LBx = block_dimensions.x();
  const double LBy = block_dimensions.y();
  const double LBz = block_dimensions.z();

  // Describe body B's mass, center of mass, and inertia properties.
  const Vector3<double> p_BoBcm_B = Vector3<double>::Zero();
  //const UnitInertia<double> G_BBcm_B =
      //UnitInertia<double>::TriaxiallySymmetric(1.0e20);
  const UnitInertia<double> G_BBcm_B =
      UnitInertia<double>::SolidBox(LBx, LBy, LBz);
  const SpatialInertia<double> M_BBcm_B(mass, p_BoBcm_B, G_BBcm_B);

  // Create a rigid body B with the mass properties of a uniform solid block.
  const RigidBody<double>& box = plant->AddRigidBody(name, M_BBcm_B);

  // Box's visual.
  // The pose X_BG of block B's geometry frame G is an identity transform.
  const RigidTransform<double> X_BG;  // Identity transform.
  plant->RegisterVisualGeometry(box, X_BG, geometry::Box(LBx, LBy, LBz),
                                name + "_visual", color);

  // When the TAMSI solver is used, we simply let MultibodyPlant estimate
  // contact parameters based on penetration_allowance and stiction_tolerance.
  geometry::ProximityProperties props;
  if (FLAGS_solver_type != 0 || FLAGS_mbp_time_step == 0) {
    props.AddProperty(geometry::internal::kMaterialGroup,
                      geometry::internal::kPointStiffness, FLAGS_stiffness);
    props.AddProperty(geometry::internal::kMaterialGroup, "dissipation_rate",
                      FLAGS_dissipation_rate);
  }
  props.AddProperty(geometry::internal::kMaterialGroup,
                    geometry::internal::kFriction,
                    CoulombFriction<double>(friction, friction));

  //add spheres to emulate box collision:
  if (emulate_box_multicontact) {
    const Vector4<double> red(1.0, 0.0, 0.0, 1.0);
    const Vector4<double> red_50(1.0, 0.0, 0.0, 0.5);
    const double radius_x = LBx / FLAGS_num_spheres_per_face / 2.0;
    const double radius_y = LBy / FLAGS_num_spheres_per_face / 2.0;
    const double radius_z = LBz / FLAGS_num_spheres_per_face / 2.0;
    double dx = 2 * radius_x;
    double dy = 2 * radius_y;
    double dz = 2 * radius_z;
    const int ns = FLAGS_num_spheres_per_face;

    auto add_sphere = [&](const std::string& sphere_name, double x, double y,
                          double z, double radius) {
      const Vector3<double> p_BoSpherei_B(x, y, z);
      const RigidTransform<double> X_BSpherei(p_BoSpherei_B);
      geometry::Sphere shape(radius);
      // Ellipsoid might not be accurate. From console [warning]:
      // "Ellipsoid is primarily for ComputeContactSurfaces in
      // hydroelastic contact model. The accuracy of other collision
      // queries and signed distance queries are not guaranteed."
      // geometry::Ellipsoid shape(radius_x, radius_y, radius_z);
      plant->RegisterCollisionGeometry(box, X_BSpherei, shape, sphere_name,
                                        props);
      if (FLAGS_visualize_multicontact) {
        plant->RegisterVisualGeometry(box, X_BSpherei, shape, sphere_name, red);
      }
    };

    // Add points (zero size spheres) at the corners to avoid spurious
    // interpentrations between boxes and the sink.
    add_sphere("c1", -LBx / 2, -LBy / 2, -LBz / 2, 0);
    add_sphere("c2", +LBx / 2, -LBy / 2, -LBz / 2, 0);
    add_sphere("c3", -LBx / 2, +LBy / 2, -LBz / 2, 0);
    add_sphere("c4", +LBx / 2, +LBy / 2, -LBz / 2, 0);
    add_sphere("c5", -LBx / 2, -LBy / 2, +LBz / 2, 0);
    add_sphere("c6", +LBx / 2, -LBy / 2, +LBz / 2, 0);
    add_sphere("c7", -LBx / 2, +LBy / 2, +LBz / 2, 0);
    add_sphere("c8", +LBx / 2, +LBy / 2, +LBz / 2, 0);

    // Make a "mesh" of non-zero radii spheres.
    for (int i = 0; i < ns; ++i) {
      const double x = -LBx / 2 + radius_x + i * dx;
      for (int j = 0; j < ns; ++j) {
        const double y = -LBy / 2 + radius_y + j * dy;
        for (int k = 0; k < ns; ++k) {
          const double z = -LBz / 2 + radius_z + k * dz;
          if (i == 0 || j == 0 || k == 0 || i == ns - 1 || j == ns - 1 ||
              k == ns - 1) {
            const std::string name_spherei =
                fmt::format("{}_sphere_{}{}{}_collision", name, i, j, k);
            add_sphere(name_spherei, x, y, z, radius_x);
          }
        }  // k
      }    // j
    }   // i
  }    

  auto id = plant->RegisterCollisionGeometry(
      box, X_BG, geometry::Box(LBx, LBy, LBz), name + "_collision", props);
  box_geometry_ids.push_back(id);

  return box;
}

// This method fixes MultibodyPlant::get_applied_spatial_force_input_port() so
// that a constant force `f_Bo_W` is applied on `body`, at its origin Bo. The
// force is expressed in the world frame.
// @pre We called Initialize().
void FixAppliedForce(const BodyIndex& body_index, const Vector3d& f_Bo_W, 
                  MultibodyPlant<double>* plant,systems::Context<double>* plant_context) {
  std::vector<ExternallyAppliedSpatialForce<double>> forces(1);
  //TODO: adjust here so that box doesn't rotate in sliding case 
  forces[0].body_index = body_index;
  forces[0].p_BoBq_B = Vector3d::Zero();
  //forces[0].p_BoBq_B[0] = -FLAGS_radius0;
  forces[0].F_Bq_W = SpatialForce<double>(Vector3d(0.0, 0.0, 0.0), f_Bo_W);
  DRAKE_DEMAND (plant != nullptr);
  DRAKE_DEMAND(plant_context != nullptr);
  plant->get_applied_spatial_force_input_port().FixValue(plant_context,
                                                          forces);
}

void AddGround(MultibodyPlant<double>* plant) {
  DRAKE_THROW_UNLESS(plant != nullptr);
  // Parameters for the sink.
  const double length = 10;
  const double width = 10;
  const double wall_thickness = 0.04;
  const double wall_mass = 1.0;
  const double friction_coefficient = FLAGS_friction_coefficient;
  const Vector4<double> light_blue(0.5, 0.8, 1.0, 0.3);
  const Vector4<double> light_green(0., 0.7, 0.0, 1.0);

  auto add_wall =
      [&](const std::string& name, const Vector3d& dimensions,
          const RigidTransformd& X_WB,
          const Vector4<double>& color) -> const RigidBody<double>& {
    const auto& wall = AddBox(name, dimensions, wall_mass, friction_coefficient,
                              color, false,  plant);
    plant->WeldFrames(plant->world_frame(), wall.body_frame(), X_WB);
    return wall;
  };

  const Vector3d bottom_dimensions(50 * length, 50 * width, wall_thickness);

  add_wall("sink_bottom", bottom_dimensions,
           Translation3d(0, 0, -wall_thickness / 2.0), light_green);
}


int do_main() {
  // Build a generic multibody plant.
  systems::DiagramBuilder<double> builder;
  auto [plant, scene_graph] =
      AddMultibodyPlantSceneGraph(&builder, FLAGS_mbp_time_step);

  AddGround(&plant);

  // set up one box on ground:
  const double radius0 = FLAGS_radius0;
  const double density = FLAGS_density;  // kg/m^3.

  const double friction = FLAGS_friction_coefficient;
  //color is orange
  const Vector4<double> orange(1.0, 0.55, 0.0, 0.5);
  const Vector3d box_size = 2 * radius0 * Vector3d::Ones() ;
  const double volume = box_size(0) * box_size(1) * box_size(2);
  const double mass = density * volume;
  auto box_index = AddBox("box", box_size, mass, friction, 
                  orange, true,  &plant).index();

  geometry::GeometrySet all_boxes(box_geometry_ids);
  scene_graph.ExcludeCollisionsWithin(all_boxes);                

  plant.Finalize();


  //solver related code:
  if (FLAGS_solver_type == 0 || FLAGS_mbp_time_step == 0) {
    plant.set_penetration_allowance(FLAGS_penetration_allowance);
    plant.set_stiction_tolerance(FLAGS_stiction_tolerance);
  }

  UnconstrainedPrimalSolver<double>* primal_solver{nullptr};
  AdmmSolver<double>* admm_solver{nullptr};
  CompliantContactComputationManager<double>* manager{nullptr};
  if (FLAGS_solver_type == 1) {
    auto owned_manager =
        std::make_unique<CompliantContactComputationManager<double>>();
    manager = owned_manager.get();
    plant.SetDiscreteUpdateManager(std::move(owned_manager));
    primal_solver =
        &manager->mutable_contact_solver<UnconstrainedPrimalSolver>();

    // N.B. These lines to set solver parameters are only needed if you want to
    // experiment with these values. Default values should work ok for most
    // applications. Thus, for your general case you can omit these lines.
    UnconstrainedPrimalSolverParameters params;
    params.abs_tolerance = FLAGS_abs_tol;
    params.rel_tolerance = FLAGS_rel_tol;
    params.Rt_factor = FLAGS_rt_factor;
    params.max_iterations = 300;
    params.ls_alpha_max = FLAGS_ls_alpha_max;
    params.use_supernodal_solver = true;
    params.compare_with_dense = false;
    params.verbosity_level = FLAGS_verbosity_level;
    params.log_stats = true;
    if (FLAGS_line_search == "exact") {
      params.ls_method =
          UnconstrainedPrimalSolverParameters::LineSearchMethod::kExact;
    } else {
      params.ls_max_iterations = 100;
      params.ls_method =
          UnconstrainedPrimalSolverParameters::LineSearchMethod::kArmijo;
    }
    primal_solver->set_parameters(params);
  } 
  if (FLAGS_solver_type == 2) {
    auto owned_manager =
        std::make_unique<CompliantContactComputationManager<double>>();
    manager = owned_manager.get();
    plant.SetDiscreteUpdateManager(std::move(owned_manager));
    manager->set_contact_solver(std::make_unique<AdmmSolver<double>>());
    admm_solver =
        &manager->mutable_contact_solver<AdmmSolver>();

    AdmmSolverParameters params;
    params.dynamic_rho = FLAGS_dynamic_rho;
    params.rho = FLAGS_rho;
    params.verbosity_level = FLAGS_verbosity_level;
    params.initialize_force = FLAGS_initialize_force;
    params.log_stats = true;
    admm_solver->set_parameters(params);
  }

  fmt::print("Num positions: {:d}\n", plant.num_positions());
  fmt::print("Num velocities: {:d}\n", plant.num_velocities());

  // Publish contact results for visualization.
  if (FLAGS_visualize) {
    geometry::DrakeVisualizerParams viz_params;
    viz_params.publish_period = FLAGS_viz_period;
    geometry::DrakeVisualizerd::AddToBuilder(&builder, scene_graph, nullptr,
                                             viz_params);
  }
  if (FLAGS_visualize_forces) {
    ConnectContactResultsToDrakeVisualizer(&builder, plant);
  }
  auto diagram = builder.Build();

  // Create a context for this system:
  std::unique_ptr<systems::Context<double>> diagram_context =
      diagram->CreateDefaultContext();
  diagram->SetDefaultContext(diagram_context.get());
  systems::Context<double>& plant_context =
      diagram->GetMutableSubsystemContext(plant, diagram_context.get());

  // In the plant's default context, we assume the state of body B in world W is
  // such that X_WB is an identity transform and B's spatial velocity is zero.
  plant.SetDefaultContext(&plant_context);
  
  //set initial position of the box:
  const Vector3d p_WB0(0.0, 0.0, FLAGS_radius0);  // set radius ...
  plant.SetFreeBodyPose(&plant_context, plant.get_body(box_index),
                        RigidTransformd(p_WB0));

  // set external force on the box:
  const Vector3d external_force(FLAGS_input_force_x, 0.0, 0.0);
  FixAppliedForce(box_index, external_force, &plant, &plant_context);

  std::ofstream log_file("solution.dat");
  log_file << fmt::format("time x y z roll pitch yaw fx fy fz\n");
  const auto& body = plant.get_body(box_index);

  auto simulator =
      MakeSimulatorFromGflags(*diagram, std::move(diagram_context));

  simulator->set_monitor([&plant, &body, &log_file](
                             const systems::Context<double>& root_context) {
    const auto& context = plant.GetMyContextFromRoot(root_context);

    // Pose of body B in the world frame W.
    const math::RigidTransformd& X_WB =
        plant.EvalBodyPoseInWorld(context, body);

    // Position of body B in the world frame W.
    const Vector3d p_WB = X_WB.translation();

    // Orientation of body B in the world frame W.
    const math::RotationMatrixd& R_WB = X_WB.rotation();

    // Convert to roll-pitch-yaw for reporting.
    const math::RollPitchYawd rpy(R_WB);

    const auto& contact_results =
        plant.get_contact_results_output_port().Eval<ContactResults<double>>(
            context);

    // Compute resultant force on body B, expressedin world frame W.
    // NOTE: You could compute the resultant torque also since you have the
    // point of application p_WC.
    Vector3d f_WB = Vector3d::Zero();
    for (int i = 0; i < contact_results.num_point_pair_contacts(); ++i) {
      const PointPairContactInfo<double>& point_pair_info =
          contact_results.point_pair_contact_info(i);
      f_WB += point_pair_info.contact_force();
    }

    log_file << fmt::format("{} {} {} {} {} {} {} {} {} {}\n",
                            context.get_time(), p_WB.x(), p_WB.y(), p_WB.z(),
                            rpy.roll_angle(), rpy.pitch_angle(),
                            rpy.yaw_angle(), f_WB.x(), f_WB.y(), f_WB.z());

    return systems::EventStatus::Succeeded();
  });

  clock::time_point sim_start_time = clock::now();
  CALLGRIND_START_INSTRUMENTATION;
  simulator->AdvanceTo(FLAGS_simulation_time);
  CALLGRIND_STOP_INSTRUMENTATION;
  clock::time_point sim_end_time = clock::now();
  log_file.close();
  const double sim_time =
      std::chrono::duration<double>(sim_end_time - sim_start_time).count();
  std::cout << "AdvanceTo() time [sec]: " << sim_time << std::endl;
  if (primal_solver) {
    std::cout << "ContactSolver total time [sec]: "
              << primal_solver->get_total_time() << std::endl;
  }

  if (manager) {
    manager->LogStats("manager_log.dat");
    if (primal_solver){
      primal_solver->LogIterationsHistory("log.dat");
    }
    if (admm_solver) {
      admm_solver->LogIterationsHistory("log.dat");
      admm_solver->LogOneTimestepHistory("one_step_log.dat", 4);
    }
  }

  PrintSimulatorStatistics(*simulator);

  return 0;
}

}  // namespace
}  // namespace mp_convex_solver
}  // namespace examples
}  // namespace multibody
}  // namespace drake

int main(int argc, char* argv[]) {
  gflags::SetUsageMessage(
      "\nSimulation of a clutter of objects falling into a box container.");
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  return drake::multibody::examples::mp_convex_solver::do_main();
}
