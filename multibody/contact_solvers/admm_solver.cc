#include "drake/multibody/contact_solvers/admm_solver.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>

#include "fmt/format.h"

#include "drake/common/test_utilities/limit_malloc.h"
#include "drake/multibody/contact_solvers/contact_solver_utils.h"
#include "drake/multibody/contact_solvers/block_sparse_linear_operator.h"
#include "drake/multibody/contact_solvers/rtsafe.h"
#include "drake/multibody/contact_solvers/supernodal_solver.h"
#include "drake/multibody/contact_solvers/timer.h"

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

using Eigen::SparseMatrix;
using Eigen::SparseVector;
//using drake::multibody::contact_solvers::internal::BlockSparseMatrix;

/*int admm_total_solver_iterations = -1;
int admm_instance = 0;
double admm_error = 0;*/
template <typename T>
AdmmSolver<T>::AdmmSolver()
    : ConvexSolverBase<T>({AdmmSolverParameters().Rt_factor}) {}

template <typename T>
ContactSolverStatus AdmmSolver<T>::DoSolveWithGuess(
    const typename ConvexSolverBase<T>::PreProcessedData& data,
    const VectorX<T>& v_guess, ContactSolverResults<T>* result) {
  throw std::logic_error("Only T = double is supported.");
}

template <typename T>
void AdmmSolver<T>::InitializeD(const int& nc, const std::vector<MatrixX<T>>& Mt, 
                      const BlockSparseMatrix<T>& Jblock, VectorX<T>* D) {
  // const int& nc = data_.nc;
  // const int& nv = data_.nv;
  // auto& D  = data_.D;
  const int nv = Mt.size();
  std::vector<Eigen::LLT<MatrixX<T>>> M_ldlt;
  M_ldlt.resize(nv);
  std::vector<Matrix3<T>> W(nc, Matrix3<T>::Zero());

  for (int iv = 0; iv< nv; iv++) {
    const auto& Mt_local = Mt[iv];
    M_ldlt[iv] = Mt_local.llt();
  }

  for (auto [p, t, Jpt] : Jblock.get_blocks()) {
    //loop over 3 row blocks each time 
    for (int k = 0; k < Jpt.rows()/3; k ++) {
      const int i0 = Jblock.row_start(p) + k*3;
      //DRAKE_DEMAND(k*3 < Jpt.rows());
      const auto& Jkt = Jpt.block(k*3, 0, 3, Jpt.cols());
      W[i0/3] += Jkt*M_ldlt[t].solve(Jkt.transpose());
      //DRAKE_DEMAND(i0/3 < nc);
      if (parameters_.verbosity_level >= 3) {
        
        // if (i0/3 == 0) {
        //   PRINT_VAR(k);
        //   PRINT_VAR(Jkt);
        //   PRINT_VAR(Jpt);
        //   PRINT_VAR(Mt[t]);
        // }
        //PRINT_VAR(i0/3);
      }
    }
    
    if (parameters_.verbosity_level >= 3) {
      PRINT_VAR(W[0]);
      //PRINT_VAR(Jkt);
      //PRINT_VAR(Mt[t]);
      //PRINT_VAR(k);
      //PRINT_VAR(t);
    }
  }

  for (int ic = 0, ic3 = 0; ic < nc; ic++, ic3 += 3) {
    (*D)[ic3] = W[ic].norm()/3;
    (*D)[ic3+1] = W[ic].norm()/3;
    (*D)[ic3+2] = W[ic].norm()/3;
    if (parameters_.verbosity_level >= 3) {
      if ((*D)[ic3] == 0) {
        PRINT_VAR(W[ic]);
        PRINT_VAR(ic);
      }
    }
  }
  if (parameters_.verbosity_level >= 3) {
    PRINT_VAR(*D);
  }
}

template <>
ContactSolverStatus AdmmSolver<double>::DoSolveWithGuess(
    const ConvexSolverBase<double>::PreProcessedData& data,
    const VectorX<double>& v_guess, ContactSolverResults<double>* results) {
  using std::abs;
  using std::max;

  // Starts a timer for the overall execution time of the solver.
  Timer global_timer;

  const auto& dynamics_data = *data.dynamics_data;
  const auto& contact_data = *data.contact_data;

  const int nv = dynamics_data.num_velocities();
  const int nc = contact_data.num_contacts();
  const int nc3 = 3 * nc;
  const auto& Rinv = data_.Rinv;

  // The primal method needs the inverse dynamics data.
  DRAKE_DEMAND(dynamics_data.has_inverse_dynamics());

  // We should not attempt solving zero sized problems for no reason since the
  // solution is trivially v = v*.
  DRAKE_DEMAND(nc != 0);

  // Print stuff for debugging.
  // TODO: refactor into PrintProblemSize().
  if (parameters_.verbosity_level >= 1) {
    PRINT_VAR(nv);
    PRINT_VAR(nc);
    PRINT_VAR(data_.Mt.size());
    PRINT_VAR(data_.Mblock.rows());
    PRINT_VAR(data_.Mblock.cols());
    PRINT_VAR(data_.Mblock.num_blocks());

    PRINT_VAR(data_.Jblock.block_rows());
    PRINT_VAR(data_.Jblock.block_cols());
    PRINT_VAR(data_.Jblock.rows());
    PRINT_VAR(data_.Jblock.cols());
    PRINT_VAR(data_.Jblock.num_blocks());
  }
  // TODO: refactor into PrintProblemStructure().
  if (parameters_.verbosity_level >= 2) {
    for (const auto& [p, t, Jb] : data_.Jblock.get_blocks()) {
      std::cout << fmt::format("(p,t) = ({:d},{:d}). {:d}x{:d}.\n", p, t,
                               Jb.rows(), Jb.cols());
    }
  }

  State state(nv, nc);
  //aux_state_.Resize(nv, nc, parameters_.compare_with_dense);

  state.mutable_v() = v_guess;
  // Compute velocity and impulses here to use in the computation of convergence
  // metrics later for the very first iteration.
  auto& cache = state.mutable_cache();
  const auto& const_cache = state.cache();
  const auto& v_star = data_.dynamics_data->get_v_star();
  const auto& J = data_.Jblock;
  const auto& vc_stab = data_.vc_stab;
  const auto& R = data_.R;

  //CalcVelocityAndImpulses(state, &cache.vc, &cache.gamma);

  // Previous iteration state, for error computation and reporting.
  State state_prev = state;

  // Reset stats.
  stats_ = {};
  stats_.num_contacts = nc;
  // Log the time it took to pre-process data.
  stats_.preproc_time = this->pre_process_time();

  AdmmSolverIterationMetrics metrics;

  //Initialize D related data:
  this->InitializeD(nc, data_.Mt, data_.Jblock, &data_.D);
  data_.Dinv = data_.D.cwiseInverse();
  data_.Dinv_sqrt = data_.Dinv.cwiseSqrt();
  data_.D_sqrt = data_.D.cwiseSqrt();

  const auto& rho = parameters_.rho;

  //TODO: initialize utilde, sigma...... eveyrthing here
  J.Multiply(state.v(), &cache.vc);
  VectorX<double> y_tilde = -R.cwiseSqrt().cwiseProduct(Rinv.cwiseProduct(cache.vc - vc_stab));
  VectorX<double> sigma_tilde(nc3);
  const auto& mu = data_.contact_data->get_mu();
  ConvexSolverBase<double>::ProjectIntoDWarpedCone(parameters_.soft_tolerance, 
                                     mu, Rinv, y_tilde, 
                                      &sigma_tilde);
  //sigma = 1/sqrt(R) * sigma_tilde
  state.mutable_sigma() = Rinv.cwiseSqrt().cwiseProduct(sigma_tilde); 
  //z = Jv - vhat + R*sigma
  VectorX<double> z = cache.vc - vc_stab + R.cwiseProduct(state.sigma());
  //z_tilde = 1/sqrt(D) * z
  state.mutable_z_tilde() = data_.Dinv_sqrt.cwiseProduct(z);
  state.mutable_u_tilde() = -data_.D_sqrt.cwiseProduct(state.sigma())/rho;

  // Super nodal solver is constructed once per time-step to reuse structure
  // of M and J.
  std::unique_ptr<conex::SuperNodalSolver> solver;
  if (parameters_.use_supernodal_solver) {
    Timer timer;
    solver = std::make_unique<conex::SuperNodalSolver>(
        data_.Jblock.block_rows(), data_.Jblock.get_blocks(), data_.Mt);
    stats_.supernodal_construction_time = timer.Elapsed();
  }
  if (parameters_.verbosity_level >=3) {
      PRINT_VAR(state.v().norm());
      PRINT_VAR(state.sigma().norm());
      PRINT_VAR(state.z_tilde().norm());
      PRINT_VAR(state.u_tilde().norm());
      PRINT_VAR(data_.D.minCoeff());
    }

  // Start Newton iterations.
  int k = 0;
  for (; k < parameters_.max_iterations; ++k) {
    if (parameters_.verbosity_level >= 3) {
      std::cout << std::string(80, '=') << std::endl;
      std::cout << std::string(80, '=') << std::endl;
      std::cout << "Iteration: " << k << std::endl;
    }

    //for debugging only, delete later:
    //r_norm = ||g-z||
    data_.Jblock.Multiply(state.v(), & cache.vc);
    cache.g = cache.vc+ data_.R.cwiseProduct(state.sigma()) - data_.vc_stab;
    cache.z = data_.D_sqrt.cwiseProduct(state.z_tilde());
    cache.u = data_.Dinv_sqrt.cwiseProduct(state.u_tilde());
    double r_norm = (cache.g - cache.z).norm();
    //s_norm = ||R(y+sigma)||
    double s_norm = R.cwiseProduct(rho*cache.u+state.sigma()).norm();
    if (parameters_.verbosity_level >= 3){
      PRINT_VAR(r_norm);
      PRINT_VAR(s_norm);
    }


    if (k == 0 || state.cache().rho_changed){
      this -> InitializeSolveForXData(state, solver.get());
    }
    this->SolveForX(state, v_star, &state.mutable_v(), 
          &state.mutable_sigma(), solver.get());

    //debugging code, delete later:
    // VectorX<double> j(nv);
    // data_.Jblock.MultiplyByTranspose(state.sigma(), &j);
    // VectorX<double> p(nv);
    // data_.Mblock.Multiply(state.v(), &p);
    // p = data_.Djac.asDiagonal() * p;
    // j = data_.Djac.asDiagonal() * j;
    // double sigma_smallest = j.cwiseAbs().minCoeff();
    // double p_smallest = p.cwiseAbs().minCoeff();



    // Check if SolveForX is doing the right thing:
    const auto [mom_l2, mom_max] =
      this->CalcScaledMomentumError(data_, state.v(), state.sigma());
    if (parameters_.verbosity_level >=3) {
      //PRINT_VAR(mom_rel_l2);
      //PRINT_VAR(mom_rel_max);
      PRINT_VAR(mom_l2);
      PRINT_VAR(mom_max);
      PRINT_VAR(state.v().norm());
      PRINT_VAR(state.sigma().norm());
      PRINT_VAR(state.z_tilde().norm());
      //PRINT_VAR(state.z_tilde());
      //PRINT_VAR(state.u_tilde());
      PRINT_VAR(state.u_tilde().norm());
    }
    DRAKE_DEMAND(mom_l2 < parameters_.abs_tolerance);
    DRAKE_DEMAND(mom_max < parameters_.abs_tolerance);
    // Update change in contact velocities.
    J.Multiply(state.v(), &cache.vc);
    
    //calculate g:
    CalcG(state.v(), state.sigma(), &cache.g);
    cache.g_tilde = data_.Dinv_sqrt.cwiseProduct(cache.g);
    
    //update z:
    CalcZTilde(const_cache.g_tilde, state.u_tilde(), &state.mutable_z_tilde());
    cache.z = data_.D_sqrt.cwiseProduct(state.z_tilde());
    
    //update u and u_tilde:
    state.mutable_u_tilde() += const_cache.g_tilde - state.z_tilde();
    cache.u = data_.Dinv_sqrt.cwiseProduct(state.u_tilde());

    double uz_product = state.z_tilde().dot(state.u_tilde());
    
    if (parameters_.verbosity_level >= 3) {
      PRINT_VAR(uz_product);
    }

    DRAKE_DEMAND(abs(uz_product) < 1e-12);

    const bool converged =
        this -> CheckConvergenceCriteria(cache.g, cache.z, parameters_.rho*cache.u,
                                 state.sigma(), cache.vc);
    // Update iteration statistics.
/*    metrics = CalcIterationMetrics(state, state_kp, num_ls_iters, alpha);
    if (parameters_.log_stats) {
      stats_.iteration_metrics.push_back(metrics);
    }

    // TODO: refactor into PrintNewtonStats().
    if (parameters_.verbosity_level >= 3) {
      //TODO: Think about what would be helpful here 
      PRINT_VAR(cache.g);
    }*/
    if (parameters_.log_stats) {
      stats_.iteration_metrics.push_back(metrics);
    }

    if (converged) {
      // TODO: refactor into PrintConvergedIterationStats().
      if (parameters_.verbosity_level >= 1) {
        //TODO: Think about what makes sense here
        std::cout << "Iteration converged at: " << k << std::endl;
        PRINT_VAR(cache.vc.norm());
        std::cout << std::string(80, '=') << std::endl;
      }
      break;
    }

    //state_kp = state;
  }
  

  if (k == parameters_.max_iterations) return ContactSolverStatus::kFailure;
  
  //TODO: Work on all the functions here. Keep some output...... 
  //Also check momentum equation!
  auto& last_metrics =
      parameters_.log_stats ? stats_.iteration_metrics.back() : metrics;
  std::tie(last_metrics.mom_l2, last_metrics.mom_max) =
      this->CalcScaledMomentumError(data, state.v(), state.sigma());
  std::tie(last_metrics.mom_rel_l2, last_metrics.mom_rel_max) =
      this->CalcRelativeMomentumError(data, state.v(), state.sigma());
  if (!parameters_.log_stats) stats_.iteration_metrics.push_back(metrics);
  stats_.num_iters = stats_.iteration_metrics.size();

  PackContactResults(data_, state.v(), cache.vc, state.sigma(), results);
  stats_.total_time = global_timer.Elapsed();

  solution_history_.emplace_back(AdmmSolutionData<double>{
      nc, cache.vc, state.sigma(), data_.contact_data->get_mu(), data_.R});

  // Update stats history.
  stats_history_.push_back(stats_);

  total_time_ += global_timer.Elapsed();

  return ContactSolverStatus::kSuccess;
}

template <typename T>
bool AdmmSolver<T>::CheckConvergenceCriteria(const VectorX<T>& g, 
                        const VectorX<T>& z, const VectorX<T>& y, 
                        const VectorX<T>& sigma, const VectorX<T>& vc) const{
  using std::max;

  const double& abs_tol = parameters_.abs_tolerance;
  const double& rel_tol = parameters_.rel_tolerance;
  const auto& R = data_.R;
  const auto& vc_stab = data_.vc_stab;
  const auto& r_s_ratio = parameters_.r_s_ratio;
  auto& rho = parameters_.rho;
  const auto& rho_factor = parameters_.rho_factor;
  auto& rho_changed = parameters_.rho_changed;
  
  //r_norm = ||g-z||
  double r_norm = (g-z).norm();
  
  //s_norm = ||R(y+sigma)||
  double s_norm = R.cwiseProduct(y+sigma).norm();
  
  const double bound = abs_tol+rel_tol*max(vc.norm(), vc_stab.norm());


  //dynamic rho code:
  if (r_norm > r_s_ratio* s_norm && rho < 10000) {
    rho *= rho_factor;
    rho_changed = True;
  }

  if(r_norm < r_s_ratio * s_norm && rho > 1e-10) {
    rho = rho/rho_factor;
    rho_changed = True;
  }

  if (r_norm < bound && s_norm < bound)
    return true;

  return false;
}

template <typename T>
void AdmmSolver<T>::CalcGMatrix(const VectorX<T>& D, const VectorX<T>& R, 
      const double& rho, std::vector<MatrixX<T>>* G) const{

  const int& nc = data_.nc;
  
  for (int ic = 0, ic3 = 0; ic < nc; ic++, ic3 += 3) {
    const auto& R_ic = R.template segment<3>(ic3);
    const auto& D_ic = D.template segment<3>(ic3);
    //const Vector3<T> Rinv = R_ic.cwiseInverse();
    MatrixX<T>& G_ic = (*G)[ic];
    G_ic = rho * (D_ic+rho*R_ic).cwiseInverse().asDiagonal();
  }

}

template <typename T>
void AdmmSolver<T>::InitializeSolveForXData(const State& s, conex::SuperNodalSolver* solver) const {
  auto& cache = s.mutable_cache();
  const auto& D = data_.D;
  const auto& R = data_.R;
  const double& rho = parameters_.rho;

  this->CalcGMatrix(D, R, rho, &cache.G);
  solver->SetWeightMatrix(cache.G);
  solver->Factor();
  cache.Finv = (R+ rho * R.cwiseProduct(R.cwiseProduct(D.cwiseInverse()))).cwiseInverse();

}

//TO THINK ABOUT: use state as argument and make things simpler?
template <typename T>
void AdmmSolver<T>::SolveForX(const State& s, const VectorX<T>& v_star, 
     VectorX<T>* v, VectorX<T>* sigma, conex::SuperNodalSolver* solver) const{
 //pseudo code: 
//r_sigma = rho*R*D^-0.5*(z_tilde+D^-0.5 vhat-u_tilde) - rho*R*u
//r_v = J^T*R^-1*r_sigma + M*v_star
  //auto rhs = cache.r_v - E^T F^{-1}r_sigma
  //*v = solver->Solve(rhs);
  //*sigma = F^{-1} (cache.r_sigma - E.Multiply(*v))
  //Think about: maybe make most of this function another one to compare with dense solver?
  auto& cache = s.mutable_cache();
  const auto& z_tilde = s.z_tilde();
  const auto& u_tilde = s.u_tilde();
  const auto& R = data_.R;
  const auto& Rinv = data_.Rinv;
  const auto& J = data_.Jblock;
  const auto& M = data_.Mblock;
  const auto& D = data_.D;
  const auto& Dinv_sqrt = data_.Dinv_sqrt;
  const auto& Dinv = data_.Dinv;
  const auto& vc_stab = data_.vc_stab;
  const double& rho = parameters_.rho;
  const int& nc3 = data_.nc*3;
  const int& nc = data_.nc;
  const int& nv = data_.nv;

  VectorX<T> R_Dinv_sqrt = R.cwiseProduct(Dinv_sqrt);
  VectorX<T> R_Dinv = R.cwiseProduct(Dinv);
  
  //calculate r_sigma = rho*R*D^-0.5*(z_tilde+D^-0.5 vhat-u_tilde) - rho*R*u
  VectorX<T> r_sigma(nc3);
  r_sigma = rho * R_Dinv_sqrt.cwiseProduct(
                    z_tilde - u_tilde + Dinv_sqrt.cwiseProduct(vc_stab));
  
  // Print stuff for debugging.
  // TODO: refactor into PrintProblemSize().
  if (parameters_.verbosity_level >= 1) {
    PRINT_VAR(nv);
    PRINT_VAR(nc);
    PRINT_VAR(r_sigma.size());
  }
  
  
  //calculate r_v = J^T*R^-1*r_sigma + M*v_star
  VectorX<T> r_v(nv);
  VectorX<T> JtRinvr_sigma(nv);
  Rinv.cwiseProduct(r_sigma);
  J.MultiplyByTranspose(Rinv.cwiseProduct(r_sigma), &JtRinvr_sigma);
  VectorX<T> Mv_star(nv);
  M.Multiply(v_star, &Mv_star);
  r_v = JtRinvr_sigma+Mv_star;
  
  //rhs = r_v - J^T G r_sigma
  VectorX<T> Gr_sigma(nc3);
  for (int ic = 0, ic3 = 0; ic < nc; ic++, ic3 += 3) {
    const MatrixX<T>& G_ic = cache.G[ic];
    const auto& r_sigma_ic = r_sigma.template segment<3>(ic3);
    auto Gr_sigma_ic = Gr_sigma.template segment<3>(ic3);
    Gr_sigma_ic = G_ic*r_sigma_ic;
  }
  VectorX<T> rhs(nv);
  VectorX<T> JtGr_sigma(nv);
  J.MultiplyByTranspose(Gr_sigma, &JtGr_sigma);
  rhs = r_v - JtGr_sigma;

  //TODO: build full matrix for debugging?
  //local_timer.Reset();
  *v = solver->Solve(rhs);
  //sigma = F^-1(r_sigma - rhoRD^-1J v)
  VectorX<T> Jv(nc3);
  J.Multiply(*v, &Jv);
  *sigma = cache.Finv.cwiseProduct(
          r_sigma - rho*R_Dinv.cwiseProduct(Jv));
  //stats_.linear_solver_time += local_timer.Elapsed();
}


template <typename T>
void AdmmSolver<T>::CalcG(const VectorX<T>& v, const VectorX<T>& sigma, 
            VectorX<T>* g) const{
  
  const auto& J = data_.Jblock;
  
  //Afte debugging: use cache.vc to be more convenient 
  VectorX<T> vc(J.rows());
  J.Multiply(v, &vc);
  *g = vc + data_.R.cwiseProduct(sigma) - data_.vc_stab;

}


template <typename T>
void AdmmSolver<T>::CalcZTilde(const VectorX<T>& g_tilde, 
      const VectorX<T>& u_tilde, VectorX<T>*z_tilde) const{
  
  const auto& mu_star = data_.contact_data->get_mu().cwiseInverse();
  ConvexSolverBase<double>::ProjectIntoDWarpedCone(parameters_.soft_tolerance, 
                                      mu_star, data_.D, g_tilde+u_tilde, 
                                      z_tilde);
}


template <typename T>
AdmmSolverIterationMetrics
AdmmSolver<T>::CalcIterationMetrics(const State& s,
                                                   const State& s0,
                                                   int num_ls_iterations,
                                                   double alpha) const {
  auto& cache = s.mutable_cache();
  //CalcVelocityAndImpulses(s, &cache.vc, &s.sigma());

  AdmmSolverIterationMetrics metrics;
  //metrics.gamma_norm = s.sigma().norm();

  return metrics;
}




template <typename T>
void AdmmSolver<T>::LogIterationsHistory(
    const std::string& file_name) const {
  const std::vector<AdmmSolverStats>& stats_hist =
      this->get_stats_history();
  std::ofstream file(file_name);
  file << fmt::format(
      "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} "
      "{} {} {} {}\n",
      // Problem size.
      "num_contacts",
      // Number of iterations.
      "num_iters",
      // Error metrics.
      "vc_error_max_norm", "v_error_max_norm", "gamma_error_max_norm", "mom_l2",
      "mom_max", "mom_rel_l2", "mom_rel_max","opt_cond",
      // Some norms. We can use them as reference scales.
      "vc_norm", "gamma_norm",
      // Line search metrics.
      "total_ls_iters", "max_ls_iters", "last_alpha", "mean_alpha", "alpha_min",
      "alpha_max",
      // Gradient and Hessian metrics.
      "grad_ell_max_norm", "dv_max_norm", "rcond",
      // Timing metrics.
      "total_time", "preproc_time", "assembly_time", "linear_solve_time",
      "ls_time", "supernodal_construction");

  for (const auto& s : stats_hist) {
    const auto& metrics = s.iteration_metrics.back();
    // const int iters = s.iteration_metrics.size();
    const int iters = s.num_iters;

    file << fmt::format(
        "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} "
        "{} {} {} {}\n",
        // Problem size.
        s.num_contacts,
        // Number of iterations.
        iters);
        // Error metrics.
        // metrics.vc_error_max_norm, metrics.v_error_max_norm,
        // metrics.gamma_error_max_norm, metrics.mom_l2, metrics.mom_max,
        // metrics.mom_rel_l2, metrics.mom_rel_max, metrics.opt_cond,
        // // Some norms.
        // metrics.vc_norm, metrics.gamma_norm,
        // // Timing metrics.
        // s.total_time, s.preproc_time, s.assembly_time, s.linear_solver_time,
        //  s.supernodal_construction_time);
  }
  file.close();
}

template <typename T>
void AdmmSolver<T>::LogSolutionHistory(
    const std::string& file_name) const {
  const std::vector<AdmmSolutionData<T>>& sol = this->solution_history();
  std::ofstream file(file_name);

  file << fmt::format("{} {} {} {} {} {} {} {} {} {} {} {}\n", "sol_num", "nc",
                      "vc_x", "vc_y", "vc_z", "gamma_x", "gamma_y", "gamma_z",
                      "mu", "Rx", "Ry", "Rz");

  auto format_vec = [](const Eigen::Ref<const Vector3<T>>& x) {
    return fmt::format("{} {} {}", x(0), x(1), x(2));
  };

  for (int k = 0; k < static_cast<int>(sol.size()); ++k) {
    const auto& s = sol[k];
    const int nc = s.nc;
    for (int i = 0; i < nc; ++i) {
      const int i3 = 3 * i;
      const auto vc = s.vc.template segment<3>(i3);
      const auto gamma = s.gamma.template segment<3>(i3);
      const T mu = s.mu(i);
      const auto R = s.R.template segment<3>(i3);
      file << k << " " << nc << " " << format_vec(vc) << " "
           << format_vec(gamma) << " " << mu << " " << format_vec(R)
           << std::endl;
    }
  }
  file.close();
}

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake

template class ::drake::multibody::contact_solvers::internal::
    AdmmSolver<double>;
