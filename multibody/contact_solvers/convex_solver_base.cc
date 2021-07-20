#include "drake/multibody/contact_solvers/convex_solver_base.h"

#include <algorithm>
#include <fstream>
#include <numeric>

#include "drake/multibody/contact_solvers/contact_solver_utils.h"
#include "drake/multibody/contact_solvers/timer.h"

#include <iostream>
#define PRINT_VAR(a) std::cout << #a ": " << a << std::endl;
#define PRINT_VARn(a) std::cout << #a ":\n" << a << std::endl;


namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Vector3d;
using Eigen::VectorXd;

template <typename T>
ContactSolverStatus ConvexSolverBase<T>::SolveWithGuess(
    const T& time_step, const SystemDynamicsData<T>& dynamics_data,
    const PointContactData<T>& contact_data, const VectorX<T>& v_guess,
    ContactSolverResults<T>* results) {
  // TODO: notice that data_ only is valid withing this scope.
  // Therefore make this look like:
  //   ProcessedData data = MakePreProcessedData(...);
  Timer timer;
  PreProcessData(time_step, dynamics_data, contact_data, parameters_.Rt_factor);
  pre_process_time_ = timer.Elapsed();
  return DoSolveWithGuess(data_, v_guess, results);
}

template <typename T>
void ConvexSolverBase<T>::PreProcessData(
    const T& time_step, const SystemDynamicsData<T>& dynamics_data,
    const PointContactData<T>& contact_data, double Rt_factor) {
  using std::max;
  using std::sqrt;

  // Keep references to data.
  data_.time_step = time_step;
  data_.dynamics_data = &dynamics_data;
  data_.contact_data = &contact_data;
  data_.Resize(dynamics_data.num_velocities(), contact_data.num_contacts());

  // Aliases to data.
  const auto& mu = contact_data.get_mu();
  const auto& phi0 = contact_data.get_phi0();
  const auto& stiffness = contact_data.get_stiffness();
  const auto& dissipation = contact_data.get_dissipation();

  // Aliases to mutable pre-processed data workspace.
  auto& R = data_.R;
  auto& vc_stab = data_.vc_stab;
  auto& Djac = data_.Djac;

  dynamics_data.get_A().AssembleMatrix(&data_.Mblock);
  contact_data.get_Jc().AssembleMatrix(&data_.Jblock);

  // Extract M's per-tree diagonal blocks.
  // Compute Jacobi pre-conditioner Djac.
  data_.Mt.clear();
  data_.Mt.reserve(data_.Mblock.num_blocks());
  for (const auto& block : data_.Mblock.get_blocks()) {
    const int t1 = std::get<0>(block);
    const int t2 = std::get<1>(block);
    const MatrixX<T>& Mij = std::get<2>(block);
    DRAKE_DEMAND(t1 == t2);
    data_.Mt.push_back(Mij);

    DRAKE_DEMAND(Mij.rows() == Mij.cols());
    const int nt = Mij.rows();  // == cols(), the block is squared.

    const int start = data_.Mblock.row_start(t1);
    DRAKE_DEMAND(start == data_.Mblock.col_start(t2));

    Djac.template segment(start, nt) =
        Mij.diagonal().cwiseInverse().cwiseSqrt();
  }

  const int nc = phi0.size();
  for (int ic = 0, ic3 = 0; ic < nc; ic++, ic3 += 3) {
    // Regularization.
    auto Ric = R.template segment<3>(ic3);
    const T& k = stiffness(ic);
    DRAKE_DEMAND(k > 0);
    const T& c = dissipation(ic);
    const T taud = c / k;  // Damping rate.
    const T Rn =
        1.0 / (time_step * k * (time_step + taud));
    DRAKE_DEMAND(Rn > 0);
    const T Rt = Rt_factor * Rn;
    Ric = Vector3<T>(Rt, Rt, Rn);

    // Stabilization velocity.
    const T vn_hat = -phi0(ic) / (time_step + taud);
    vc_stab.template segment<3>(ic3) = Vector3<T>(0, 0, vn_hat);
  }
  data_.Rinv = R.cwiseInverse();
}

template <typename T>
Vector3<T> ConvexSolverBase<T>::CalcProjection(
    const ProjectionParams& params, const Eigen::Ref<const Vector3<T>>& y,
    const T& yr, const T& yn, const Eigen::Ref<const Vector2<T>>& that,
    int* region, Matrix3<T>* dPdy) const {
  const T& mu = params.mu;
  const T& Rt = params.Rt;
  const T& Rn = params.Rn;
  const T mu_hat = mu * Rt / Rn;

  Vector3<T> gamma;
  // Analytical projection of y onto the friction cone ℱ using the R norm.
  if (yr < mu * yn) {  // Region I, stiction.
    *region = 1;
    gamma = y;
    if (dPdy) dPdy->setIdentity();
  } else if (-mu_hat * yr < yn && yn <= yr / mu) {  // Region II, sliding.
    *region = 2;
    // Common terms:
    const T mu_tilde2 = mu * mu_hat;  // mu_tilde = mu * sqrt(Rt/Rn).
    const T factor = 1.0 / (1.0 + mu_tilde2);

    // Projection P(y).
    const T gn = (yn + mu_hat * yr) * factor;
    const Vector2<T> gt = mu * gn * that;
    gamma.template head<2>() = gt;
    gamma(2) = gn;

    // Gradient:
    if (dPdy) {
      const Matrix2<T> P = that * that.transpose();
      const Matrix2<T> Pperp = Matrix2<T>::Identity() - P;

      // We split dPdy into separate blocks:
      //
      // dPdy = |dgt_dyt dgt_dyn|
      //        |dgn_dyt dgn_dyn|
      // where dgt_dyt ∈ ℝ²ˣ², dgt_dyn ∈ ℝ², dgn_dyt ∈ ℝ²ˣ¹ and dgn_dyn ∈ ℝ.
      const Matrix2<T> dgt_dyt = mu * (gn / yr * Pperp + mu_hat * factor * P);
      const Vector2<T> dgt_dyn = mu * factor * that;
      const RowVector2<T> dgn_dyt = mu_hat * factor * that.transpose();
      const T dgn_dyn = factor;

      dPdy->template topLeftCorner<2, 2>() = dgt_dyt;
      dPdy->template topRightCorner<2, 1>() = dgt_dyn;
      dPdy->template bottomLeftCorner<1, 2>() = dgn_dyt;
      (*dPdy)(2, 2) = dgn_dyn;
    }
  } else {  // yn <= -mu_hat * yr
    *region = 3;
    // Region III, no contact.
    gamma.setZero();
    if (dPdy) dPdy->setZero();
  }

  return gamma;
}

template <typename T>
void ConvexSolverBase<T>::CalcAnalyticalInverseDynamics(
    double soft_norm_tolerance, const VectorX<T>& vc, VectorX<T>* gamma,
    std::vector<Matrix3<T>>* dgamma_dy, VectorX<int>* regions) const {
    //eg.vc input gamma is output,
    // VectorX<double> gamma;

    // CalcAnalyticalInverseDynamics(soft_tol, vc, &gamma, &dgamma_dy, &regions); dgamma_dy and regions are not needed 
  const int nc = data_.nc;
  const int nc3 = 3 * nc;
  DRAKE_DEMAND(vc.size() == nc3);
  DRAKE_DEMAND(gamma->size() == nc3);

  // Pre-processed data.
  const auto& R = data_.R;
  const auto& vc_stab = data_.vc_stab;

  // Problem data.
  const auto& mu_all = data_.contact_data->get_mu();

  if (dgamma_dy != nullptr) DRAKE_DEMAND(regions != nullptr);

  for (int ic = 0, ic3 = 0; ic < nc; ic++, ic3 += 3) {
    const auto& vc_ic = vc.template segment<3>(ic3);
    const auto& vc_stab_ic = vc_stab.template segment<3>(ic3);
    const auto& R_ic = R.template segment<3>(ic3);
    const T& mu = mu_all(ic);
    const T& Rt = R_ic(0);
    const T& Rn = R_ic(2);
    const Vector3<T> y_ic = (vc_stab_ic - vc_ic).array() / R_ic.array();
    const auto yt = y_ic.template head<2>();
    const T yr = SoftNorm(yt, soft_norm_tolerance);
    const T yn = y_ic[2];
    const Vector2<T> that = yt / yr;

    // Analytical projection of y onto the friction cone ℱ using the R norm.
    auto gamma_ic = gamma->template segment<3>(ic3);
    if (dgamma_dy != nullptr) {
      auto& dgamma_dy_ic = (*dgamma_dy)[ic];
      gamma_ic = CalcProjection({mu, Rt, Rn}, y_ic, yr, yn, that,
                                &(*regions)(ic), &dgamma_dy_ic);
    } else {
      int region_ic{-1};
      gamma_ic =
          CalcProjection({mu, Rt, Rn}, y_ic, yr, yn, that, &region_ic);
    }
  }
}


//IMPORTANT: this function does projection with D norm, if want to use R norm just pass in R into the D argument 
template <typename T>
void ConvexSolverBase<T>::CalcStarAnalyticalInverseDynamics(
    double soft_norm_tolerance, const VectorX<T>& vc, const VectorX<T>& D, VectorX<T>* gamma)const {
    //eg.vc input gamma is output,
    // VectorX<double> gamma;

    // CalcAnalyticalInverseDynamics(soft_tol, vc, &gamma, &dgamma_dy, &regions); dgamma_dy and regions are not needed 
  const int nc = data_.nc;
  const int nc3 = 3 * nc;
  DRAKE_DEMAND(vc.size() == nc3);
  DRAKE_DEMAND(gamma->size() == nc3);
  DRAKE_DEMAND(D.size() == nc3);

  // Pre-processed data.
  const auto& R = data_.R;
  const auto& vc_stab = data_.vc_stab;

  // Problem data.
  const auto& mu_all = data_.contact_data->get_mu();


  for (int ic = 0, ic3 = 0; ic < nc; ic++, ic3 += 3) {
    const auto& vc_ic = vc.template segment<3>(ic3);
    const auto& vc_stab_ic = vc_stab.template segment<3>(ic3);
    const auto& R_ic = R.template segment<3>(ic3);
    const auto& D_ic = D.template segment<3>(ic3);
    const T& mu = mu_all(ic);
    const T& mustar = 1/mu;
    const T& Rt = R_ic(0);
    const T& Rn = R_ic(2);

    //TODO: change expression for y here, need to be consistent with gtilda+utilda in admm
    const Vector3<T> y_ic = vc_ic.array();
    const auto yt = y_ic.template head<2>();
    const T yr = SoftNorm(yt, soft_norm_tolerance);
    const T yn = y_ic[2];
    const Vector2<T> that = yt / yr;
    int* dummy_region = new int(-1);

    // Analytical projection of y onto the friction cone ℱ^* using the D^{-1} norm.
    auto gamma_ic = gamma->template segment<3>(ic3);
    gamma_ic = CalcProjection({mustar, 1/D_ic(0), 1/D_ic(2)}, y_ic, yr, yn, that, dummy_region);
  }
}

template <typename T>
bool ConvexSolverBase<T>::CheckConvergenceCriteria(const VectorX<T>& vc,
                                                   const VectorX<T>& dvc,
                                                   double abs_tolerance,
                                                   double rel_tolerance) const {
  const int nc = vc.size() / 3;

  // N.B. Notice that because of line-search, dvc can be much larger than
  // (vc-vc_prev). Therefore we check convergence using dvc directly.

  for (int ic = 0, ic3 = 0; ic < nc; ic++, ic3 += 3) {
    const auto vc_ic = vc.template segment<3>(ic3);
    const auto dvc_ic = dvc.template segment<3>(ic3);
    const T dvc_norm = dvc_ic.norm();
    const T vc_norm = vc_ic.norm();
    if (dvc_norm > abs_tolerance + rel_tolerance * vc_norm) {
      return false;
    }
  }

  return true;
}

template <typename T>
void ConvexSolverBase<T>::PackContactResults(
    const PreProcessedData& data, const VectorX<T>& v, const VectorX<T>& vc,
    const VectorX<T>& gamma, ContactSolverResults<T>* results) const {
  results->Resize(data.nv, data.nc);
  results->v_next = v;
  ExtractNormal(vc, &results->vn);
  ExtractTangent(vc, &results->vt);
  ExtractNormal(gamma, &results->fn);
  ExtractTangent(gamma, &results->ft);
  // N.B. While contact solver works with impulses, results are reported as
  // forces.
  results->fn /= data.time_step;
  results->ft /= data.time_step;
  const auto& Jop = data.contact_data->get_Jc();
  Jop.MultiplyByTranspose(gamma, &results->tau_contact);
  results->tau_contact /= data.time_step;
}

template <typename T>
std::pair<T, T> ConvexSolverBase<T>::CalcScaledMomentumError(
    const PreProcessedData& data, const VectorX<T>& v,
    const VectorX<T>& gamma) const {
  const int nv = data.nv;
  const auto& v_star = data.dynamics_data->get_v_star();
  const auto& Djac = data.Djac;

  VectorX<T> v_aux = v - v_star;
  VectorX<T> momentum_balance(nv);
  // momentum_balance = M(v-v*)
  data.Mblock.Multiply(v_aux, &momentum_balance);
  // momentum_balance = M(v-v*) - J^T*gamma
  data.Jblock.MultiplyByTranspose(gamma, &v_aux);
  momentum_balance -= v_aux;

  // Scale momentum balance using the mass matrix's Jacobi preconditioner so
  // that all entries have the same units and we can compute a fair error
  // metric.
  momentum_balance = Djac.asDiagonal() * momentum_balance;

  const T mom_l2 = momentum_balance.norm();
  const T mom_max = momentum_balance.template lpNorm<Eigen::Infinity>();
  return std::make_pair(mom_l2, mom_max);
}

template <typename T>
std::pair<T, T> ConvexSolverBase<T>::CalcRelativeMomentumError(
    const PreProcessedData& data, const VectorX<T>& v,
    const VectorX<T>& gamma) const {
  const int nv = data.nv;
  const auto& v_star = data.dynamics_data->get_v_star();
  const auto& Djac = data.Djac;

  VectorX<T> p = v - v_star;
  VectorX<T> Mdv(nv);
  // Mdv = M(v-v*)
  data.Mblock.Multiply(p, &Mdv);
  // j = J^T*gamma
  VectorX<T> j(nv);
  data.Jblock.MultiplyByTranspose(gamma, &j);

  // r(v) = M⋅(v−v*)−Jᵀγ.
  VectorX<T> r = Mdv - j;

  // p = M⋅v
  data.Mblock.Multiply(v, &p);

  // We using Jacobi scaling to make the metric "fair" for all dofs even if they
  // have different units.
  p = Djac.asDiagonal() * p;
  r = Djac.asDiagonal() * r;
  j = Djac.asDiagonal() * j;

  using std::max;
  const T eps_2 = r.norm() / max(p.norm(), j.norm());

  using std::abs;
  T eps_max = 0;
  for (int i = 0; i < nv; ++i) {
    const T eps_i = abs(r(i)) / max(abs(p(i)), abs(j(i)));
    eps_max = max(eps_max, eps_i);
  }

  return std::make_pair(eps_2, eps_max);
}

template <typename T>
T ConvexSolverBase<T>::CalcOptimalityCondition(const PreProcessedData& data,
                                               const VectorX<T>& v,
                                               const VectorX<T>& gamma) const {
  using std::abs;
  const int nc = data.contact_data->num_contacts();

  VectorX<T> vc(3 * nc);
  data.contact_data->get_Jc().Multiply(v, &vc);
  const VectorX<T>& vc_hat = data.vc_stab;
  const VectorX<T>& R = data.R;

  const VectorX<T> g = vc - vc_hat + R.asDiagonal() * gamma;

  T m = 0;
  for (int ic = 0, ic3 = 0; ic < nc; ic++, ic3 += 3) {
    const auto g_ic = g.template segment<3>(ic3);
    const auto gamma_ic = gamma.template segment<3>(ic3);
    m += abs(g_ic.dot(gamma_ic));
  }
  m /= nc;

  return m;
}

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake

template class ::drake::multibody::contact_solvers::internal::ConvexSolverBase<
    double>;
