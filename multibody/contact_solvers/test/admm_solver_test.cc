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
#include "drake/geometry/scene_graph.h"
#include "drake/lcm/drake_lcm.h"
#include "drake/multibody/plant/externally_applied_spatial_force.h"
#include "drake/math/random_rotation.h"
#include "drake/math/rigid_transform.h"
#include "drake/multibody/plant/point_pair_contact_info.h"
#include "drake/math/roll_pitch_yaw.h"
#include "drake/math/rotation_matrix.h"
#include "drake/multibody/contact_solvers/admm_solver.h"
#include "drake/multibody/plant/compliant_contact_computation_manager.h"
#include "drake/multibody/plant/contact_results_to_lcm.h"
#include "drake/systems/analysis/implicit_integrator.h"
#include "drake/systems/analysis/simulator.h"
#include "drake/systems/analysis/simulator_gflags.h"
#include "drake/systems/analysis/simulator_print_stats.h"
#include "drake/systems/framework/diagram_builder.h"
#include "gtest/gtest.h"

#undef NDEBUG

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {
namespace {

using drake::math::RigidTransform;
using drake::math::RigidTransformd;
using drake::math::RollPitchYawd;
using drake::math::RotationMatrixd;
using drake::multibody::ContactResults;
using drake::multibody::PointPairContactInfo;
using drake::multibody::MultibodyPlant;
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

// This method fixes MultibodyPlant::get_applied_spatial_force_input_port() so
// that a constant force `f_Bo_W` is applied on `body`, at its origin Bo. The
// force is expressed in the world frame.
// @pre We called Initialize().
void FixAppliedForce(const BodyIndex& body_index, const Vector3d& f_Bo_W, 
                  MultibodyPlant<double>* plant,systems::Context<double>* plant_context) {
  //DRAKE_DEMAND(initialized_);
  std::vector<ExternallyAppliedSpatialForce<double>> forces(1);
  forces[0].body_index = body_index;
  forces[0].p_BoBq_B = Vector3d::Zero();
  forces[0].F_Bq_W = SpatialForce<double>(Vector3d(0.0, 0.0, 0.0), f_Bo_W);
  DRAKE_DEMAND (plant != nullptr);
  DRAKE_DEMAND(plant_context != nullptr);
  plant->get_applied_spatial_force_input_port().FixValue(plant_context,
                                                          forces);
}


const ContactResults<double>& GetContactResults(const MultibodyPlant<double>& plant,
                                          const systems::Context<double>& plant_context) {
  const ContactResults<double>& contact_results =
      plant.get_contact_results_output_port().Eval<ContactResults<double>>(
          plant_context);
  return contact_results;
}

const RigidBody<double>& AddBox(const std::string& name,
                                const Vector3<double>& block_dimensions,
                                double mass, double friction,
                                //const Vector4<double>& color,
                                bool add_box_collision,
                                MultibodyPlant<double>* plant) {
  // Ensure the block's dimensions are mass are positive.
  const double LBx = block_dimensions.x();
  const double LBy = block_dimensions.y();
  const double LBz = block_dimensions.z();
  const int num_spheres_per_face = 3;
  const double stiffness = 1000.0;

  // Describe body B's mass, center of mass, and inertia properties.
  const Vector3<double> p_BoBcm_B = Vector3<double>::Zero();
  const UnitInertia<double> G_BBcm_B =
      UnitInertia<double>::SolidBox(LBx, LBy, LBz);
  const SpatialInertia<double> M_BBcm_B(mass, p_BoBcm_B, G_BBcm_B);

  // Create a rigid body B with the mass properties of a uniform solid block.
  const RigidBody<double>& box = plant->AddRigidBody(name, M_BBcm_B);

  // Box's visual.
  // The pose X_BG of block B's geometry frame G is an identity transform.
  const RigidTransform<double> X_BG;  // Identity transform.
  //plant->RegisterVisualGeometry(box, X_BG, geometry::Box(LBx, LBy, LBz),
                                //name + "_visual", color);

  // When the TAMSI solver is used, we simply let MultibodyPlant estimate
  // contact parameters based on penetration_allowance and stiction_tolerance.
  geometry::ProximityProperties props;
  props.AddProperty(geometry::internal::kMaterialGroup,
                    geometry::internal::kPointStiffness, stiffness);
  props.AddProperty(geometry::internal::kMaterialGroup, "dissipation_rate",
                    0.01);
  props.AddProperty(geometry::internal::kMaterialGroup,
                    geometry::internal::kFriction,
                    CoulombFriction<double>(friction, friction));

  // Box's collision geometry is a solid box.
  const Vector4<double> red(1.0, 0.0, 0.0, 1.0);
  const Vector4<double> red_50(1.0, 0.0, 0.0, 0.5);
  const double radius_x = LBx / num_spheres_per_face / 2.0;
  const double radius_y = LBy / num_spheres_per_face / 2.0;
  const double radius_z = LBz / num_spheres_per_face / 2.0;
  double dx = 2 * radius_x;
  double dy = 2 * radius_y;
  double dz = 2 * radius_z;
  const int ns = num_spheres_per_face;

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
    // if (FLAGS_visualize_multicontact) {
    //   plant->RegisterVisualGeometry(box, X_BSpherei, shape, sphere_name, red);
    // }
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
  }      // i
  

  if (add_box_collision) {
    auto id = plant->RegisterCollisionGeometry(
        box, X_BG, geometry::Box(LBx, LBy, LBz), name + "_collision", props);
    box_geometry_ids.push_back(id);
  }
  return box;
}

void AddGround(MultibodyPlant<double>* plant) {
  DRAKE_THROW_UNLESS(plant != nullptr);
  // Parameters for the sink.
  const double length = 10;
  const double width = 10;
  const double wall_thickness = 0.04;
  const double wall_mass = 1.0;
  const double friction_coefficient = 1.0;
  const Vector4<double> light_green(0., 0.7, 0.0, 1.0);

  auto add_wall =
      [&](const std::string& name, const Vector3d& dimensions,
          const RigidTransformd& X_WB,
          const Vector4<double>& color) -> const RigidBody<double>& {
    const auto& wall = AddBox(name, dimensions, wall_mass, friction_coefficient,
                              //color, 
                              true, plant);
    plant->WeldFrames(plant->world_frame(), wall.body_frame(), X_WB);
    return wall;
  };

  const Vector3d bottom_dimensions(50 * length, 50 * width, wall_thickness);

  add_wall("sink_bottom", bottom_dimensions,
           Translation3d(0, 0, -wall_thickness / 2.0), light_green);
}



const RigidBody<double>& AddSphere(const std::string& name, const double radius,
                                   double mass, double friction,
                                   //const Vector4<double>& color,
                                   MultibodyPlant<double>* plant) {
  const UnitInertia<double> G_Bcm = UnitInertia<double>::SolidSphere(radius);
  const SpatialInertia<double> M_Bcm(mass, Vector3<double>::Zero(), G_Bcm);

  const RigidBody<double>& ball = plant->AddRigidBody(name, M_Bcm);
  const double stiffness = 1000.0;
  const double dissipation_rate = 0.01;

  geometry::ProximityProperties props;
  props.AddProperty(geometry::internal::kMaterialGroup,
                    geometry::internal::kPointStiffness, stiffness);
  props.AddProperty(geometry::internal::kMaterialGroup, "dissipation_rate",
                    dissipation_rate);
  
  props.AddProperty(geometry::internal::kMaterialGroup,
                    geometry::internal::kFriction,
                    CoulombFriction<double>(friction, friction));

  // Add collision geometry.
  const RigidTransformd X_BS = RigidTransformd::Identity();
  plant->RegisterCollisionGeometry(ball, X_BS, geometry::Sphere(radius),
                                   name + "_collision", props);

  // Add visual geometry.
  // plant->RegisterVisualGeometry(ball, X_BS, geometry::Sphere(radius),
  //                               name + "_visual", color);

  return ball;
}

std::vector<BodyIndex> AddObjects(const int num_objects, const bool use_spheres,
                                  MultibodyPlant<double>* plant) {
  const double radius0 = 0.05;
  const double scale_factor = 2.0;
  const double density = 1000.0;  // kg/m^3.

  const double friction = 1.0;
  const int num_bodies = plant->num_bodies();

  std::vector<BodyIndex> bodies;
  for (int i = 1; i <= num_objects; ++i) {
    const std::string name = "object" + std::to_string(i + num_bodies);
    double e = scale_factor > 0 ? i - 1 : num_objects - i;
    double scale = std::pow(std::abs(scale_factor), e);

    int choice = -1;
    if (num_objects > 1) {
      choice = 0;
    }
    if (use_spheres) {
      choice = 0;
    } else {
      choice = 1;
    }

    if (choice == 1) {
      const Vector3d box_size = 2 * radius0 * Vector3d::Ones() * scale;
      const double volume = box_size(0) * box_size(1) * box_size(2);
      const double mass = density * volume;
      //Vector4<double> color50(color);
      //color50.z() = 0.5;
      bodies.push_back(AddBox(name, box_size, mass, friction, //color50,
                             true, plant)
                           .index());
    } else {
        const double radius = radius0 * scale;
        const double volume = 4. / 3. * M_PI * radius * radius * radius;
        const double mass = density * volume;
        bodies.push_back(
            AddSphere(name, radius, mass, friction, //color, 
                           plant).index());
    }
    scale *= scale_factor;
  }

  return bodies;
}

void SetObjectsIntoAPile(const int num_objects, const MultibodyPlant<double>& plant,
                         const Vector3d& offset,
                         const std::vector<BodyIndex>& bodies,
                         systems::Context<double>* plant_context) {
  double delta_z = 0.05;  // assume objects have a BB of about 10 cm.
  const double scale_factor = 2.0;

  const int seed = 41;
  std::mt19937 generator(seed);


  double z = 0;
  int i = 1;
  for (auto body_index : bodies) {
    const auto& body = plant.get_body(body_index);
    if (body.is_floating()) {
      double e = scale_factor > 0 ? i - 1 : num_objects - i;
      double scale = std::pow(std::abs(scale_factor), e);

      z+= delta_z*scale;
      const Vector3d p_WB = offset + Vector3d(0.0, 0.0, z);  //set radius ...

      plant.SetFreeBodyPose(plant_context, body, RigidTransformd(p_WB));
      
      z += delta_z * scale;
      ++i;
    }
  }
}

GTEST_TEST(AdmmSolver, StandingSphereTest) {

  const double simulation_time = 3.0;
  const double mbp_time_step = 0.01;

    // Build a generic multibody plant.
  systems::DiagramBuilder<double> builder;
  auto [plant, scene_graph] =
      AddMultibodyPlantSceneGraph(&builder, mbp_time_step);
  if (false) {
    geometry::GeometrySet all_boxes(box_geometry_ids);
    scene_graph.ExcludeCollisionsWithin(all_boxes);
  }

  AddGround(&plant);

  // AddSphere("sphere", radius, mass, friction, orange, &plant);
  auto pile1 = AddObjects(1,true, &plant);
  plant.Finalize();
  AdmmSolver<double>* admm_solver{nullptr};
  CompliantContactComputationManager<double>* manager{nullptr};
      auto owned_manager =
        std::make_unique<CompliantContactComputationManager<double>>();
  manager = owned_manager.get();
  plant.SetDiscreteUpdateManager(std::move(owned_manager));
  manager->set_contact_solver(std::make_unique<AdmmSolver<double>>());
  admm_solver =
      &manager->mutable_contact_solver<AdmmSolver>();

  AdmmSolverParameters params;
  params.log_stats = false;
  admm_solver->set_parameters(params);

  //fmt::print("Num positions: {:d}\n", plant.num_positions());
  //fmt::print("Num velocities: {:d}\n", plant.num_velocities());
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

  SetObjectsIntoAPile(1, plant, Vector3d(0, 0, 0), pile1,
                      &plant_context);
  auto simulator =
      MakeSimulatorFromGflags(*diagram, std::move(diagram_context));

  simulator->AdvanceTo(simulation_time);

  ContactResults<double> contact_results = GetContactResults(plant,plant_context);

  EXPECT_EQ(contact_results.num_point_pair_contacts(), 1);
  PointPairContactInfo contact_info = contact_results.point_pair_contact_info(0);
  Vector3<double> force = contact_info.contact_force();
  EXPECT_EQ(force[0], 0);
}

GTEST_TEST(AdmmSolver, BoxStictionTest) {
  const double simulation_time = 3.0;
  const double mbp_time_step = 0.01;

    // Build a generic multibody plant.
  systems::DiagramBuilder<double> builder;
  auto [plant, scene_graph] =
      AddMultibodyPlantSceneGraph(&builder, mbp_time_step);
  if (false) {
    geometry::GeometrySet all_boxes(box_geometry_ids);
    scene_graph.ExcludeCollisionsWithin(all_boxes);
  }

  AddGround(&plant);

  // AddSphere("sphere", radius, mass, friction, orange, &plant);
  auto pile1 = AddObjects(1,false, &plant);
  plant.Finalize();
  AdmmSolver<double>* admm_solver{nullptr};
  CompliantContactComputationManager<double>* manager{nullptr};
      auto owned_manager =
        std::make_unique<CompliantContactComputationManager<double>>();
  manager = owned_manager.get();
  plant.SetDiscreteUpdateManager(std::move(owned_manager));
  manager->set_contact_solver(std::make_unique<AdmmSolver<double>>());
  admm_solver =
      &manager->mutable_contact_solver<AdmmSolver>();

  AdmmSolverParameters params;
  params.log_stats = false;
  admm_solver->set_parameters(params);

  //fmt::print("Num positions: {:d}\n", plant.num_positions());
  //fmt::print("Num velocities: {:d}\n", plant.num_velocities());
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

  SetObjectsIntoAPile(1, plant, Vector3d(0, 0, 0), pile1,
                      &plant_context);
  const Vector3d external_force(0.5, 0, 0);
  FixAppliedForce(pile1[0], external_force, &plant,
                                        &plant_context);

  auto simulator =
      MakeSimulatorFromGflags(*diagram, std::move(diagram_context));

  simulator->AdvanceTo(simulation_time);


  ContactResults<double> contact_results = GetContactResults(plant,plant_context);

  EXPECT_EQ(contact_results.num_point_pair_contacts(), 1);
  PointPairContactInfo contact_info = contact_results.point_pair_contact_info(0);
  Vector3<double> force = contact_info.contact_force();
  EXPECT_EQ(force[0], 0);

}




}  // namespace
}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake