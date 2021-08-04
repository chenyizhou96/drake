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
    for (auto [p ,t, Jpt] : data_.Jblock.get_blocks()) {
      PRINT_VAR(Jpt);
    }
    PRINT_VAR(data_.Mt[0]);
    PRINT_VAR(data_.R);
    //PRINT_VAR(parameters_.max_iterations);
  }
  // TODO: refactor into PrintProblemStructure().
  if (parameters_.verbosity_level >= 2) {
    for (const auto& [p, t, Jb] : data_.Jblock.get_blocks()) {
      std::cout << fmt::format("(p,t) = ({:d},{:d}). {:d}x{:d}.\n", p, t,
                               Jb.rows(), Jb.cols());
    }
  }

  Timer preproc_timer;

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
  //stats_.preproc_time = this->pre_process_time();

  AdmmSolverIterationMetrics metrics;

  //Initialize D related data:
  this->InitializeD(nc, data_.Mt, data_.Jblock, &data_.D);
  data_.Dinv = data_.D.cwiseInverse();
  data_.Dinv_sqrt = data_.Dinv.cwiseSqrt();
  data_.D_sqrt = data_.D.cwiseSqrt();
  const auto& D = data_.D;
  const auto& rho = parameters_.rho;
  
  //each entry of D must be positive
  DRAKE_DEMAND(D.minCoeff() > 0);

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

  stats_.preproc_time += preproc_timer.Elapsed();

  // Super nodal solver is constructed once per time-step to reuse structure
  // of M and J.
  std::unique_ptr<conex::SuperNodalSolver> solver;
  Timer construction_timer;
  solver = std::make_unique<conex::SuperNodalSolver>(
      data_.Jblock.block_rows(), data_.Jblock.get_blocks(), data_.Mt);
  stats_.supernodal_construction_time = construction_timer.Elapsed();
  
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

    // //for debugging only, delete later:
    // //r_norm = ||g-z||
    // data_.Jblock.Multiply(state.v(), & cache.vc);
    // cache.g = cache.vc+ data_.R.cwiseProduct(state.sigma()) - data_.vc_stab;
    // cache.z = data_.D_sqrt.cwiseProduct(state.z_tilde());
    // cache.u = data_.Dinv_sqrt.cwiseProduct(state.u_tilde());
    // double r_norm = (cache.g - cache.z).norm();
    // //s_norm = ||R(y+sigma)||
    // double s_norm = R.cwiseProduct(rho*cache.u+state.sigma()).norm();
    // if (parameters_.verbosity_level >= 3){
    //   PRINT_VAR(r_norm);
    //   PRINT_VAR(s_norm);
    // }
    Timer local_timer;

  
    auto& rho_changed = parameters_.rho_changed;

    local_timer.Reset();
    if (k == 0 || rho_changed){
      this -> InitializeSolveForXData(state, solver.get());
      rho_changed = false;
    }
    this->SolveForX(state, v_star, &state.mutable_v(), 
          &state.mutable_sigma(), solver.get());

    stats_.solve_for_x_time+= local_timer.Elapsed();

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
    
    local_timer.Reset();
    // Update change in contact velocities.
    J.Multiply(state.v(), &cache.vc);
    
    //calculate g and g_tilde:
    CalcG(state.v(), state.sigma(), &cache.g);
    cache.g_tilde = data_.Dinv_sqrt.cwiseProduct(cache.g);
    
    //update z and z_tilde:
    CalcZTilde(const_cache.g_tilde, state.u_tilde(), &state.mutable_z_tilde());
    cache.z = data_.D_sqrt.cwiseProduct(state.z_tilde());
    
    //update u and u_tilde:
    state.mutable_u_tilde() += const_cache.g_tilde - state.z_tilde();
    cache.u = data_.Dinv_sqrt.cwiseProduct(state.u_tilde());

    stats_.solve_for_z_u_time += local_timer.Elapsed();
    
    //test whether or not u_tilde and z_tilde are correct
    double uz_product = state.z_tilde().dot(state.u_tilde());
    //calculate quadratic cost l:
    //double l = 0.5*()
    ///((max(state.cache().vc.norm(), data_.vc_stab.norm())* max(state.cache().vc.norm(), data_.vc_stab.norm())));
    
    const double& abs_tol = parameters_.abs_tolerance;
    // if (state.u_tilde().norm() > abs_tol && state.z_tilde().norm() > abs_tol) {
    //   //uz_product = uz_product/(max(state.u_tilde().norm(), state.z_tilde().norm())* max(state.u_tilde().norm(), state.z_tilde().norm()));
    // }


    if (parameters_.verbosity_level > 0) {
      VectorX<double> u_tilde_slope(nc);
      this -> CalcSlope(state.u_tilde(), &u_tilde_slope);
      VectorX<double> z_slope(nc);
      this -> CalcSlope(state.cache().z, &z_slope);
      if (state.u_tilde().norm() > 1e-8) {
        for (int i = 0; i < nc; i++)
          DRAKE_DEMAND(u_tilde_slope[i] < data_.contact_data->get_mu()[i]);
      }
      int index = -1;
      if (state.cache().z.norm() > 1e-8) {
        for (int i = 0; i < nc; i++)
          if (z_slope[i] >= data_.contact_data->get_mu().cwiseInverse()[i]){
            index = i;
            break;
          }
      }
      if (index != -1) {
        PRINT_VAR(data_.contact_data->get_mu().cwiseInverse()[index]);
        PRINT_VAR(z_slope[index]);
        PRINT_VAR(state.cache().z.template segment<3>(index*3));
      }
      if (state.cache().z.norm() > 1e-8) {
        for (int i = 0; i < nc; i++)
          DRAKE_DEMAND(z_slope[i] < data_.contact_data->get_mu().cwiseInverse()[i]);
      }
      //PRINT_VAR(state.u_tilde());
      //PRINT_VAR(u_tilde_slope);
    }

    if (parameters_.verbosity_level >= 1) {
      if (abs(uz_product) > 1e-12) {
        PRINT_VAR(k);
        PRINT_VAR(state.u_tilde().norm());
        PRINT_VAR(state.z_tilde().norm());
        PRINT_VAR(state.cache().vc.norm());
        PRINT_VAR(data_.vc_stab.norm());
        PRINT_VAR(state.u_tilde());
        PRINT_VAR(state.z_tilde());
        PRINT_VAR(uz_product);
      }
    }

    
    DRAKE_DEMAND(abs(uz_product) < parameters_.rel_tolerance);
   
    //rhis function also does dynamic rho
    //TODO: make dynamic_rho a flag in the future and rename this function
    const bool converged =
        this -> CheckConvergenceCriteria(cache.g, cache.z, parameters_.rho*cache.u,
                                 state.sigma(), cache.vc, &state.mutable_u_tilde());
    // Update iteration statistics.
   //  metrics = CalcIterationMetrics(state, state_kp, num_ls_iters, alpha);
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
  }
  

  if (k == parameters_.max_iterations) return ContactSolverStatus::kFailure;
  
  //TODO: Work on all the functions here. Keep some output...... 

  auto& last_metrics =
      parameters_.log_stats ? stats_.iteration_metrics.back() : metrics;
  //std::tie(last_metrics.mom_l2, last_metrics.mom_max) =
      //this->CalcScaledMomentumError(data, state.v(), state.sigma());
  last_metrics.r_norm_l2 = (const_cache.g- const_cache.z).norm();
  last_metrics.s_norm_l2 = R.cwiseProduct(rho*const_cache.u+state.sigma()).norm();
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
bool AdmmSolver<T>::CheckConvergenceCriteria( const VectorX<T>& g, 
                        const VectorX<T>& z, const VectorX<T>& y, 
                        const VectorX<T>& sigma, const VectorX<T>& vc, VectorX<T>* u_tilde){
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
  
  //s_norm = ||R(rho*u+sigma)||
  double s_norm = R.cwiseProduct(y+sigma).norm();
  
  const double bound = abs_tol+rel_tol*max(vc.norm(), vc_stab.norm());

  //dynamic rho code:
  if (parameters_.dynamic_rho) {
    if (r_norm > r_s_ratio* s_norm && rho < 10000) {
      rho *= rho_factor;
      *u_tilde = *u_tilde/rho_factor;
      rho_changed = true;
    }

    if(r_norm < r_s_ratio * s_norm && rho > 1e-10) {
      rho = rho/rho_factor;
      *u_tilde = *u_tilde * rho_factor;
      rho_changed = true;
    }
  }
  if (r_norm < bound && s_norm < bound)
    return true;

  return false;
}


template <typename T>
void AdmmSolver<T>::CalcSlope(const VectorX<T>& u, VectorX<T>* slope) const{
  const int& nc = data_.nc;
  const int nc3 = 3*nc;
  //PRINT_VAR(u);
  //PRINT_VAR(u.size());
  DRAKE_DEMAND(u.size() == nc3);
  DRAKE_DEMAND ((*slope).size() == nc);
  
  using std::abs;
  using std::sqrt;
  using std::pow;

  for (int ic = 0, ic3 = 0; ic < nc; ic++, ic3 += 3) {
    const auto& u_ic = u.template segment<3>(ic3);
    if (u_ic[2] == 0) {
      DRAKE_DEMAND(u_ic[0]== 0);
      DRAKE_DEMAND(u_ic[1]== 0);
      (*slope)[ic] = 0;
    } else{
      (*slope)[ic] = abs(sqrt(pow(u_ic[0],2)+ pow(u_ic[1], 2))/u_ic[2]);
    }
  }
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
      "{} {} {} {} {} {} {} {}\n",
      // Problem size.
      "num_contacts",
      // Number of iterations.
      "num_iters",
      //error metrics
      "r norm", "s norm",
      //time metric
      "solve_for_x_time", "solve_for_z_u_time", "total time", "preproc time"
      );

  for (const auto& s : stats_hist) {
    const auto& metrics = s.iteration_metrics.back();
    // const int iters = s.iteration_metrics.size();
    const int iters = s.num_iters;
    // Compute some totals and averages.

    file << fmt::format(
        "{} {} {} {} {} {} {} {}\n",
        // Problem size.
        s.num_contacts,
        // Number of iterations.
        iters,
        // Error metrics.
        metrics.r_norm_l2,
        metrics.s_norm_l2,
        //time metrics:
        s.solve_for_x_time,
        s.solve_for_z_u_time,
        s.total_time, 
        s.preproc_time
        );
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
