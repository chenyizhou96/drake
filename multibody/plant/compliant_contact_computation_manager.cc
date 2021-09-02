#include "drake/multibody/plant/compliant_contact_computation_manager.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <utility>
#include <vector>

#include "drake/common/default_scalars.h"
#include "drake/common/drake_copyable.h"
#include "drake/common/eigen_types.h"
#include "drake/multibody/contact_solvers/block_sparse_linear_operator.h"
#include "drake/multibody/contact_solvers/contact_solver.h"
#include "drake/multibody/contact_solvers/timer.h"
#include "drake/multibody/contact_solvers/unconstrained_primal_solver.h"
#include "drake/multibody/plant/contact_permutation_utils.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/systems/framework/context.h"

#define PRINT_VAR(a) std::cout << #a ": " << a << std::endl;
#define PRINT_VARn(a) std::cout << #a ":\n" << a << std::endl;

namespace drake {
namespace multibody {

template <typename T>
CompliantContactComputationManager<T>::CompliantContactComputationManager()
    : contact_solver_(
          std::make_unique<
              contact_solvers::internal::UnconstrainedPrimalSolver<T>>()) {}

template <typename T>
CompliantContactComputationManager<T>::CompliantContactComputationManager(
    std::unique_ptr<contact_solvers::internal::ContactSolver<T>> contact_solver)
    : contact_solver_(std::move(contact_solver)) {}

template <typename T>
void CompliantContactComputationManager<T>::ExtractModelInfo() {
  const internal::MultibodyTreeTopology& topology =
      this->internal_tree().get_topology();

  internal::ComputeBfsToDfsPermutation(topology, &velocities_permutation_,
                                       &body_to_tree_map_);
  num_trees_ = velocities_permutation_.size();
  num_velocities_ = plant().num_velocities();

  // Allocate some needed workspace.
  const int nv = plant().num_velocities();
  workspace_.M.resize(nv, nv);
  tau_c_.resize(nv);
  tau_c_.setZero();
}

template <typename T>
void CompliantContactComputationManager<T>::CalcContactGraph(
    const geometry::QueryObject<T>& query_object,
    const std::vector<internal::DiscreteContactPair<T>>& contact_pairs) const {
  std::vector<SortedPair<int>> contacts;
  const geometry::SceneGraphInspector<T>& inspector = query_object.inspector();
  for (const auto& pp : contact_pairs) {
    const geometry::FrameId frameA = inspector.GetFrameId(pp.id_A);
    const geometry::FrameId frameB = inspector.GetFrameId(pp.id_B);
    const Body<T>* bodyA = plant().GetBodyFromFrameId(frameA);
    const Body<T>* bodyB = plant().GetBodyFromFrameId(frameB);
    DRAKE_DEMAND(bodyA != nullptr);
    DRAKE_DEMAND(bodyB != nullptr);
    const BodyIndex bodyA_index = bodyA->index();
    const BodyIndex bodyB_index = bodyB->index();

    const int treeA = body_to_tree_map_[bodyA_index];
    const int treeB = body_to_tree_map_[bodyB_index];
    // SceneGraph does not report collisions between anchored geometries.
    // We verify this.
    DRAKE_DEMAND(!(treeA < 0 && treeB < 0));
    contacts.push_back({treeA, treeB});
  }

  graph_ = internal::ComputeContactGraph(num_trees_, contacts,
                                         &participating_trees_);
  num_participating_trees_ = participating_trees_.size();

  num_participating_velocities_ = 0;
  participating_velocities_permutation_.resize(num_participating_trees_);
  for (int tp = 0; tp < num_participating_trees_; ++tp) {
    const int t = participating_trees_[tp];
    const int nt = velocities_permutation_[t].size();
    participating_velocities_permutation_[tp].resize(nt);
    // participating_velocities_permutation_[tp] =
    // num_participating_velocities_:num_participating_velocities_+nt-1.
    std::iota(participating_velocities_permutation_[tp].begin(),
              participating_velocities_permutation_[tp].end(),
              num_participating_velocities_);
    num_participating_velocities_ += nt;
  }
}

template <typename T>
void CompliantContactComputationManager<T>::
    CalcVelocityUpdateWithoutConstraints(
        const systems::Context<T>& context0, VectorX<T>* vstar,
        VectorX<T>* participating_vstar, VectorX<T>* v_guess,
        VectorX<T>* participating_v_guess) const {
  DRAKE_DEMAND(vstar != nullptr);
  DRAKE_DEMAND(participating_vstar != nullptr);
  DRAKE_DEMAND(vstar->size() == plant().num_velocities());
  DRAKE_DEMAND(v_guess->size() == plant().num_velocities());

  // MultibodyTreeSystem::CalcArticulatedBodyForceCache()
  MultibodyForces<T> forces0(plant());

  const internal::PositionKinematicsCache<T>& pc =
      plant().EvalPositionKinematics(context0);
  const internal::VelocityKinematicsCache<T>& vc =
      plant().EvalVelocityKinematics(context0);

  // Compute forces applied by force elements. Note that this resets forces
  // to empty so must come first.
  this->internal_tree().CalcForceElementsContribution(context0, pc, vc,
                                                      &forces0);

  // We need only handle MultibodyPlant-specific forces here.
  this->AddInForcesFromInputPorts(context0, &forces0);

  // Perform the tip-to-base pass to compute the force bias terms needed by ABA.
  const auto& tree_topology = this->internal_tree().get_topology();
  internal::ArticulatedBodyForceCache<T> aba_force_cache(tree_topology);
  this->internal_tree().CalcArticulatedBodyForceCache(context0, forces0,
                                                      &aba_force_cache);

  // MultibodyTreeSystem::CalcArticulatedBodyAccelerations()
  internal::AccelerationKinematicsCache<T> ac(tree_topology);
  this->internal_tree().CalcArticulatedBodyAccelerations(context0,
                                                         aba_force_cache, &ac);

  // Notice we use an explicit Euler scheme here since all forces are evaluated
  // at context0.
  const VectorX<T>& vdot0 = ac.get_vdot();
  const double dt = plant().time_step();

  const auto x0 = context0.get_discrete_state(0).get_value();
  // const VectorX<T> q0 = x0.topRows(this->num_positions());
  const VectorX<T> v0 = x0.bottomRows(plant().num_velocities());

  *vstar = v0 + dt * vdot0;

  // Extract "reduced" velocities for participating trees only.
  // TODO: rename to vstar_part?
  participating_vstar->resize(num_participating_velocities_);
  PermuteFullToParticipatingVelocities(*vstar, participating_vstar);

  *v_guess = v0;
  participating_v_guess->resize(num_participating_velocities_);
  PermuteFullToParticipatingVelocities(*v_guess, participating_v_guess);

// This strategy actually increased the number of iterations by about 20%.
// Thus var v_guess = v0 works best.
#if 0
  // Compute guess by using previous tau_c.
  // const auto& tau_c = plant().EvalContactSolverResults(context0).tau_contact;
  forces0.mutable_generalized_forces() += tau_c_;
  plant().internal_tree().CalcArticulatedBodyForceCache(context0, forces0,
                                                        &aba_force_cache);
  plant().internal_tree().CalcArticulatedBodyAccelerations(
      context0, aba_force_cache, &ac);
  *v_guess = v0 + dt * ac.get_vdot();
  participating_v_guess->resize(num_participating_velocities_);
  PermuteFullToParticipatingVelocities(*v_guess, participating_v_guess);
#endif
}

template <typename T>
void CompliantContactComputationManager<T>::CalcLinearDynamics(
    const systems::Context<T>& context,
    contact_solvers::internal::BlockSparseMatrix<T>* A) const {
  DRAKE_DEMAND(A != nullptr);
  plant().CalcMassMatrix(context, &workspace_.M);
  *A = internal::ExtractBlockDiagonalMassMatrix(
      workspace_.M, velocities_permutation_, participating_trees_);
}

template <typename T>
void CompliantContactComputationManager<T>::CalcContactQuantities(
    const systems::Context<T>& context,
    const std::vector<internal::DiscreteContactPair<T>>& contact_pairs,
    contact_solvers::internal::BlockSparseMatrix<T>* Jc, VectorX<T>* phi0,
    VectorX<T>* vc0, VectorX<T>* mu, VectorX<T>* stiffness,
    VectorX<T>* linear_damping) const {
  DRAKE_DEMAND(Jc != nullptr);
  const internal::ContactJacobians<T>& contact_jacobians =
      this->EvalContactJacobians(context);
  *Jc = internal::ExtractBlockJacobian(contact_jacobians.Jc, graph_,
                                       velocities_permutation_,
                                       participating_trees_);

  const std::vector<CoulombFriction<double>> combined_friction_pairs =
      this->CalcCombinedFrictionCoefficients(context,
                                                    contact_pairs);

  const int num_contacts = contact_pairs.size();
  phi0->resize(num_contacts);
  vc0->resize(3 * num_contacts);
  mu->resize(num_contacts);
  stiffness->resize(num_contacts);
  linear_damping->resize(num_contacts);

  const auto x0 = context.get_discrete_state(0).get_value();
  const VectorX<T> v0 = x0.bottomRows(plant().num_velocities());
  VectorX<T> v0_part(num_participating_velocities_);
  PermuteFullToParticipatingVelocities(v0, &v0_part);
  Jc->Multiply(v0_part, vc0);

  int k_permuted = 0;
  for (const auto& p : graph_.patches) {
    for (int k : p.contacts) {
      (*phi0)[k_permuted] = contact_pairs[k].phi0;
      (*stiffness)[k_permuted] = contact_pairs[k].stiffness;
      (*linear_damping)[k_permuted] = contact_pairs[k].damping;
      (*mu)[k_permuted] = combined_friction_pairs[k].dynamic_friction();
      ++k_permuted;
    }
  }
}

template <typename T>
void CompliantContactComputationManager<T>::DoCalcContactSolverResults(
    const systems::Context<T>& context,
    contact_solvers::internal::ContactSolverResults<T>* results) const {
  contact_solvers::internal::Timer total_timer;

  ContactManagerStats stats;
  stats.time = ExtractDoubleOrThrow(context.get_time());

  contact_solvers::internal::Timer timer;
  // const auto& query_object = plant().EvalGeometryQueryInput(context);
  const auto& query_object =
      plant()
          .get_geometry_query_input_port()
          .template Eval<geometry::QueryObject<T>>(context);
  const std::vector<internal::DiscreteContactPair<T>> contact_pairs =
      this->CalcDiscreteContactPairs(context);
  stats.geometry_time = timer.Elapsed();

  timer.Reset();
  CalcContactGraph(query_object, contact_pairs);
  stats.graph_time = timer.Elapsed();

  // After CalcContactGraph() we know the problem size and topology.
  const int nc = contact_pairs.size();

  stats.num_contacts = nc;

  timer.Reset();
  VectorX<T> vstar(plant().num_velocities());
  VectorX<T> participating_vstar(num_participating_velocities_);
  VectorX<T> v_guess(plant().num_velocities());
  VectorX<T> participating_v_guess(num_participating_velocities_);
  CalcVelocityUpdateWithoutConstraints(context, &vstar, &participating_vstar,
                                       &v_guess, &participating_v_guess);
  stats.vstar_time = timer.Elapsed();

  contact_solvers::internal::ContactSolverResults<T>
      participating_trees_results;
  if (nc > 0) {
    timer.Reset();
    // Computes the lienarized dynamics matrix A in A(v-v*) = Jᵀγ.
    contact_solvers::internal::BlockSparseMatrix<T> A;
    CalcLinearDynamics(context, &A);
    stats.linear_dynamics_time = timer.Elapsed();

    timer.Reset();
    // Computes quantities to define contact constraints.
    contact_solvers::internal::BlockSparseMatrix<T> Jc;
    VectorX<T> phi0, vc0, mu, stiffness, linear_damping;
    CalcContactQuantities(context, contact_pairs, &Jc, &phi0, &vc0, &mu,
                          &stiffness, &linear_damping);
    stats.contact_jacobian_time = timer.Elapsed();

    // Create data structures to call the contact solver.
    contact_solvers::internal::BlockSparseLinearOperator<T> Aop("A", &A);
    contact_solvers::internal::BlockSparseLinearOperator<T> Jop("Jc", &Jc);
    contact_solvers::internal::SystemDynamicsData<T> dynamics_data(
        &Aop, nullptr, &participating_vstar);
    contact_solvers::internal::PointContactData<T> contact_data(
        &phi0, &Jop, &stiffness, &linear_damping, &mu);

    // Initial guess.
    // const auto x0 = context.get_discrete_state(0).get_value();
    // const VectorX<T> v0 = x0.bottomRows(plant().num_velocities());
    // VectorX<T> participating_v0(num_participating_velocities_);
    // PermuteFullToParticipatingVelocities(v0, &participating_v0);

    timer.Reset();
    // Call contact solver.
    // TODO: consider using participating_v0 as the initial guess.
    participating_trees_results.Resize(num_participating_velocities_, nc);
    const contact_solvers::internal::ContactSolverStatus info =
        contact_solver_->SolveWithGuess(plant().time_step(), dynamics_data,
                                        contact_data, participating_v_guess,
                                        &participating_trees_results);
    if (info != contact_solvers::internal::ContactSolverStatus::kSuccess) {
      const std::string msg =
          fmt::format("MultibodyPlant's contact solver of type '" +
                          NiceTypeName::Get(*contact_solver_) +
                          "' failed to converge at "
                          "simulation time = {:7.3g} with discrete update "
                          "period = {:7.3g}.",
                      context.get_time(), plant().time_step());
      throw std::runtime_error(msg);
    }
    stats.contact_solver_time = timer.Elapsed();
  }  // nc > 0

  // ==========================================================================
  // ==========================================================================
  timer.Reset();
  // Permute "backwards" to the original ordering.
  // TODO: avoid heap allocations.
  results->Resize(num_velocities_, nc);
  // This effectively updates the non-participating velocities.
  results->v_next = vstar;
  // This effectively updates the non-participating generalized forces.
  results->tau_contact.setZero();
  tau_c_.setZero();

  // Update participating quantities.
  if (nc > 0) {
    PermuteParticipatingToFullVelocities(participating_trees_results.v_next,
                                         &results->v_next);
    PermuteParticipatingToFullVelocities(
        participating_trees_results.tau_contact, &results->tau_contact);

    // Save tau_c to compute initial guess in the next time step.
    tau_c_ = results->tau_contact;

    // TODO(amcastro-tri): Remove these when queries are computed in patch
    // order.
    PermuteFromPatches(1, participating_trees_results.fn, &results->fn);
    PermuteFromPatches(2, participating_trees_results.ft, &results->ft);
    PermuteFromPatches(1, participating_trees_results.vn, &results->vn);
    PermuteFromPatches(2, participating_trees_results.vt, &results->vt);
  }
  stats.pack_results_time = timer.Elapsed();

  stats.total_time = total_timer.Elapsed();

  stats_.push_back(stats);

  total_time_ += total_timer.Elapsed();
}

// forward = true --> vp(ip) = v(i).
// forward = false --> v(i) = vp(ip).
template <typename T>
void CompliantContactComputationManager<T>::PermuteVelocities(
    bool forward, VectorX<T>* v, VectorX<T>* vp) const {
  DRAKE_DEMAND(v != nullptr);
  DRAKE_DEMAND(vp != nullptr);
  DRAKE_DEMAND(v->size() == plant().num_velocities());
  DRAKE_DEMAND(vp->size() == num_participating_velocities_);
  for (int tp = 0; tp < num_participating_trees_; ++tp) {
    const int t = participating_trees_[tp];
    const int nt = participating_velocities_permutation_[tp].size();
    for (int vt = 0; vt < nt; ++vt) {
      const int i = velocities_permutation_[t][vt];
      const int ip = participating_velocities_permutation_[tp][vt];
      if (forward) {
        (*vp)(ip) = (*v)(i);
      } else {
        (*v)(i) = (*vp)(ip);
      }
    }
  }
}

template <typename T>
void CompliantContactComputationManager<
    T>::PermuteFullToParticipatingVelocities(const VectorX<T>& v,
                                             VectorX<T>* vp) const {
  PermuteVelocities(true, const_cast<VectorX<T>*>(&v), vp);
}

template <typename T>
void CompliantContactComputationManager<
    T>::PermuteParticipatingToFullVelocities(const VectorX<T>& vp,
                                             VectorX<T>* v) const {
  PermuteVelocities(false, v, const_cast<VectorX<T>*>(&vp));
}

// forward = true --> xp(kp) = x(k).
// forward = false --> x(k) = xp(kp).
template <typename T>
void CompliantContactComputationManager<T>::PermuteContacts(
    bool forward, int stride, VectorX<T>* x, VectorX<T>* xp) const {
  DRAKE_DEMAND(x != nullptr);
  DRAKE_DEMAND(xp != nullptr);
  DRAKE_DEMAND(x->size() == xp->size());
  int k_perm = 0;
  for (const auto& p : graph_.patches) {
    for (int k : p.contacts) {
      if (forward) {
        xp->segment(stride * k_perm, stride) = x->segment(stride * k, stride);
      } else {
        x->segment(stride * k, stride) = xp->segment(stride * k_perm, stride);
      }
      ++k_perm;
    }
  }
};

template <typename T>
void CompliantContactComputationManager<T>::PermuteIntoPatches(
    int stride, const VectorX<T>& x, VectorX<T>* xp) const {
  PermuteContacts(true, stride, const_cast<VectorX<T>*>(&x), xp);
}

template <typename T>
void CompliantContactComputationManager<T>::PermuteFromPatches(
    int stride, const VectorX<T>& xp, VectorX<T>* x) const {
  PermuteContacts(false, stride, x, const_cast<VectorX<T>*>(&xp));
}

template <typename T>
void CompliantContactComputationManager<T>::DoCalcAccelerationKinematicsCache(
    const drake::systems::Context<T>& context,
    internal::AccelerationKinematicsCache<T>* ac) const {
  // Evaluate contact results.
  const contact_solvers::internal::ContactSolverResults<T>& solver_results =
      this->EvalContactSolverResults(context);

  // Retrieve the solution velocity for the next time step.
  const VectorX<T>& v_next = solver_results.v_next;

  auto x0 = context.get_discrete_state(0).get_value();
  const VectorX<T> v0 = x0.bottomRows(plant().num_velocities());

  ac->get_mutable_vdot() = (v_next - v0) / plant().time_step();

  // N.B. Pool of spatial accelerations indexed by BodyNodeIndex.
  this->internal_tree().CalcSpatialAccelerationsFromVdot(
      context, plant().EvalPositionKinematics(context),
      plant().EvalVelocityKinematics(context), ac->get_vdot(),
      &ac->get_mutable_A_WB_pool());
}

template <typename T>
void CompliantContactComputationManager<T>::DoCalcDiscreteValues(
    const drake::systems::Context<T>& context0,
    drake::systems::DiscreteValues<T>* updates) const {
  // Get the system state as raw Eigen vectors
  // (solution at the previous time step).
  DRAKE_DEMAND(this->multibody_state_index() == 0);
  DRAKE_DEMAND(context0.num_discrete_state_groups() == 1);

  auto x0 = context0.get_discrete_state(0).get_value();
  const int nq = plant().num_positions();
  const int nv = plant().num_velocities();
  VectorX<T> q0 = x0.topRows(nq);
  VectorX<T> v0 = x0.bottomRows(nv);

  // For a discrete model this evaluates vdot = (v_next - v0)/time_step() and
  // includes contact forces.
  const VectorX<T>& vdot = plant().EvalForwardDynamics(context0).get_vdot();

  // TODO(amcastro-tri): Consider replacing this by:
  //   const VectorX<T>& v_next = solver_results.v_next;
  // to avoid additional vector operations.
  const VectorX<T>& v_next = v0 + plant().time_step() * vdot;

  VectorX<T> qdot_next(nq);
  plant().MapVelocityToQDot(context0, v_next, &qdot_next);
  VectorX<T> q_next = q0 + plant().time_step() * qdot_next;

  VectorX<T> x_next(nq + nv);
  x_next << q_next, v_next;
  updates->get_mutable_vector(0).SetFromVector(x_next);
}

template <typename T>
void CompliantContactComputationManager<T>::LogStats(
    const std::string& log_file_name) const {
  std::cout << fmt::format(
      "CompliantContactComputationManager total wall-clock: {:12.4g}\n",
      total_time());

  const std::vector<ContactManagerStats>& hist = get_stats_history();
  std::ofstream file(log_file_name);

  file << fmt::format(
    "{} {} {} {} {} {} {} {} {} {}\n",
    // Problem size.
    "time",
    // Number of iterations.
    "num_contacts",
    //parameters
    "total_time", "geometry_time",
    //error metrics
    "vstar_time", "graph_time",
    //norms:
    "linear_dynamics_time", "contact_jacobian_time", "contact_solver_time", "pack_results_time"
    //variable data:
    // "v_tilde_0", "v_tilde_1","v_tilde_2","v_tilde_3","v_tilde_4","v_tilde_5",
    // "sigma_tilde_0", "sigma_tilde_1","sigma_tilde_2","sigma_tilde_3","sigma_tilde_4",
    // "sigma_tilde_5", "sigma_tilde_6", "sigma_tilde_7", "sigma_tilde_8",
    // "z_tilde_0", "z_tilde_1","z_tilde_2","z_tilde_3","z_tilde_4",
    // "z_tilde_5", "z_tilde_6", "z_tilde_7", "z_tilde_8",
    // "u_tilde_0", "u_tilde_1","u_tilde_2","u_tilde_3","u_tilde_4",
    // "u_tilde_5", "u_tilde_6", "u_tilde_7", "u_tilde_8"
  );

  for (const auto& s : hist) {
    file << fmt::format(
        "{:18.6g} {:d}  {:18.6g}  {:18.6g} {:18.6g} {:18.6g} {:18.6g} "
        "{:18.6g} "
        "{:18.6g}  {:18.6g}\n",
        s.time, s.num_contacts, s.total_time, s.geometry_time, s.vstar_time,
        s.graph_time, s.linear_dynamics_time, s.contact_jacobian_time,
        s.contact_solver_time, s.pack_results_time);
  }
  file.close();
}

}  // namespace multibody
}  // namespace drake

DRAKE_DEFINE_CLASS_TEMPLATE_INSTANTIATIONS_ON_DEFAULT_NONSYMBOLIC_SCALARS(
    class ::drake::multibody::CompliantContactComputationManager);
