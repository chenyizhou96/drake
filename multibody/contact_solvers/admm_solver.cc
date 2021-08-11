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

  const int nv = Mt.size();
  std::vector<Eigen::LLT<MatrixX<T>>> M_ldlt;
  M_ldlt.resize(nv);
  std::vector<Matrix3<T>> W(nc, Matrix3<T>::Zero());
  if (!parameters_.scale_with_R) {
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
  } else {
    *D = data_.R;
  }
  if (parameters_.verbosity_level >= 3) {
    PRINT_VAR(*D);
  }

}

template <typename T>
void AdmmSolver<T>::Preprocess( TildeData* tilde_data) {
  const int nv = data_.nv;
  const int nc = data_.nc;
  const auto& Jblock = data_.Jblock;
  const auto& Mblock = data_.Mblock;
  const auto& Mt = data_.Mt;
  const auto& vc_stab = data_.vc_stab;
  const auto& v_star = data_.dynamics_data->get_v_star();
  const auto& D = data_.D;
  const auto& Dinv_sqrt = data_.Dinv_sqrt;
  
  //H = diag(M);
  VectorX<T> H(nv);
  //Note:can't use varied length for template segment, maybe use Map from eigen?
  //doing simple loops for now
  for (int i = 0, ic = 0; i < Mt.size(); i++, ic+= Mt[i].rows()) {
    for (int j  = 0; j < Mt[i].rows(); j++) {
       H(ic+j) = Mt[i](j, j);
    }
  }

  tilde_data -> H = H;
  
  //Mt_tilde = T^-0.5 Mt T^-0.5
  VectorX<double> Hinv_sqrt = H.cwiseSqrt().cwiseInverse();
  tilde_data -> Mt_tilde.resize(Mt.size());
  for (int i = 0, ic = 0; i < Mt.size(); i++, ic+= Mt[i].rows()) {
    int length = Mt[i].rows();
    const auto& Hinv_sqrt_ic = Hinv_sqrt.template segment(ic, length);
    tilde_data -> Mt_tilde[i] = Hinv_sqrt_ic.asDiagonal()*Mt[i]*Hinv_sqrt_ic.asDiagonal();
  }

  //M_tilde_v_star_tilde = H^-0.5 M vstar
  VectorX<double> Mv_star(nv);
  Mblock.Multiply(v_star, &Mv_star);
  tilde_data -> M_tilde_v_star_tilde = Hinv_sqrt.cwiseProduct(Mv_star);
  
  //vc_stab_tilde = D^-0.5 vc_stab
  tilde_data -> vc_stab_tilde = Dinv_sqrt.cwiseProduct(vc_stab);

  //v_star_tilde = H^0.5 v_star;
  tilde_data -> v_star_tilde = H.cwiseSqrt().cwiseProduct(v_star);
  
  //Mblock_tilde = T^-0.5 M T^-0.5
  BlockSparseMatrixBuilder<T> Mblock_tilde_builder(Mblock.block_rows(), Mblock.block_cols(),
                                      Mblock.num_blocks());
  for (auto [p, t, Mpt] : Mblock.get_blocks()) {
    const int i0 = Mblock.row_start(p);
    const int j0 = Mblock.col_start(p);
    const auto& Hinv_sqrt_p = Hinv_sqrt.template segment(i0, Mpt.rows());
    const auto& Hinv_sqrt_t = Hinv_sqrt.template segment(j0, Mpt.cols());

    MatrixX<T> Mpt_tilde = Hinv_sqrt_p.asDiagonal()*Mpt*Hinv_sqrt_t.asDiagonal();
    Mblock_tilde_builder.PushBlock(p, t, Mpt_tilde);
  }
  tilde_data -> Mblock_tilde = Mblock_tilde_builder.Build();

  //Jblock_tilde = D^-0.5 J T^-0.5
  BlockSparseMatrixBuilder<T> Jblock_tilde_builder(Jblock.block_rows(), Jblock.block_cols(),
                                      Jblock.num_blocks());
  for (auto [p, t, Jpt] : Jblock.get_blocks()) {
    const int i0 = Jblock.row_start(p);
    const int j0 = Jblock.col_start(p);
    const auto& Dinv_sqrt_p = Dinv_sqrt.template segment(i0, Jpt.rows());
    const auto& Hinv_sqrt_p = Hinv_sqrt.template segment(j0, Jpt.cols());

    MatrixX<T> Jpt_tilde = Dinv_sqrt_p.asDiagonal()*Jpt*Hinv_sqrt_p.asDiagonal();
    Jblock_tilde_builder.PushBlock(p, t, Jpt_tilde);
  }
  tilde_data -> Jblock_tilde = Jblock_tilde_builder.Build();

}


template <>
ContactSolverStatus AdmmSolver<double>::DoSolveWithGuess(
    const ConvexSolverBase<double>::PreProcessedData& data,
    const VectorX<double>& v_guess, ContactSolverResults<double>* results) {
  using std::abs;
  using std::max;
  using std::pow;

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
  if (parameters_.verbosity_level >= 2) {
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
  auto& mutable_cache = state.mutable_cache();
  const auto& cache = state.cache();

  // Previous iteration state, for error computation and reporting.
  State state_prev = state;

  // Reset stats.
  stats_ = {};
  stats_.num_contacts = nc;

  AdmmSolverIterationMetrics metrics;

  //Initialize D related data:
  this->InitializeD(nc, data_.Mt, data_.Jblock, &data_.D);
  data_.Dinv = data_.D.cwiseInverse();
  data_.Dinv_sqrt = data_.Dinv.cwiseSqrt();
  data_.D_sqrt = data_.D.cwiseSqrt();
  const auto& D = data_.D;
  const auto& Dinv_sqrt = data_.Dinv_sqrt;
  const auto& D_sqrt = data_.D_sqrt;
  const auto& rho = parameters_.rho;

  PRINT_VAR(D.size() / 3);
  PRINT_VAR(D.transpose());
  PRINT_VAR(rho);

  //each entry of D must be positive
  DRAKE_DEMAND(D.minCoeff() > 0);

  this-> Preprocess(&tilde_data_);

  const auto& J_tilde = tilde_data_.Jblock_tilde;
  const auto& H = tilde_data_.H;
  const auto& M_tilde = tilde_data_.Mblock_tilde;
  const auto& Mt_tilde = tilde_data_.Mt_tilde;
  const auto& vc_stab_tilde = tilde_data_.vc_stab_tilde;
  const auto& Rinv_sqrt = data_.Rinv.cwiseSqrt();
  const auto& R = data_.R;
  const auto& R_sqrt = data_.R.cwiseSqrt();
  const auto& M_tilde_v_star_tilde = tilde_data_.M_tilde_v_star_tilde;

  if (parameters_.verbosity_level >= 5) {
    PRINT_VAR(data_.Mt[0]);
    PRINT_VAR(Mt_tilde[0]);
    PRINT_VAR(H);
    for (auto [p ,t, Jpt] : data_.Jblock.get_blocks()) {
      PRINT_VAR(Jpt);
    }
    for (auto [p ,t, Jtildept] : J_tilde.get_blocks()) {
      PRINT_VAR(Jtildept);
    }
    for (auto [p ,t, Mtildept] : M_tilde.get_blocks()) {
      PRINT_VAR(Mtildept);
    }
    for (auto [p ,t, Mpt] : data_.Mblock.get_blocks()) {
      PRINT_VAR(Mpt);
    }
    PRINT_VAR(vc_stab_tilde);
    PRINT_VAR(data_.vc_stab);
  }


  //Initialization goes here
  //TODO: make a separate function?
  //initialize v_tilde = H^0.5 v_guess
  state.mutable_v_tilde() = H.cwiseSqrt().cwiseProduct(v_guess);
  //sigma_tilde =  Proj_mutilde(-R^-0.5D^0.5 (J_tilde v_tilde - vhat_tilde))
  J_tilde.Multiply(state.v_tilde(), &mutable_cache.vc_tilde);
  VectorX<double> y_tilde = -Rinv_sqrt.cwiseProduct(D_sqrt.cwiseProduct(cache.vc_tilde-vc_stab_tilde));
  const auto& mu = data_.contact_data->get_mu();
  ConvexSolverBase<double>::ProjectIntoDWarpedCone(parameters_.soft_tolerance, 
                                     mu, Rinv, y_tilde, 
                                      &state.mutable_sigma_tilde());
  //sigma = 1/sqrt(R) * sigma_tilde
  if (!parameters_.initialize_force){
    state.mutable_sigma_tilde() = VectorX<double>::Zero(nc3);
  }

  //z_tilde = J_tilde v_tilde - vhat_tilde + D^-0.5 R^0.5*sigma_tilde
  state.mutable_z_tilde() = cache.vc_tilde - vc_stab_tilde + Dinv_sqrt.cwiseProduct(
                          R_sqrt.cwiseProduct(state.sigma_tilde()));

  state.mutable_u_tilde() = -D_sqrt.cwiseProduct(
                          Rinv_sqrt.cwiseProduct(state.sigma_tilde()))/rho;

  stats_.preproc_time += preproc_timer.Elapsed();

  // Super nodal solver is constructed once per time-step to reuse structure
  // of M and J.
  std::unique_ptr<conex::SuperNodalSolver> solver;
  Timer construction_timer;
  solver = std::make_unique<conex::SuperNodalSolver>(
      J_tilde.block_rows(), J_tilde.get_blocks(), Mt_tilde);
  stats_.supernodal_construction_time = construction_timer.Elapsed();
  
  if (parameters_.verbosity_level >=3) {
      PRINT_VAR(state.v_tilde().norm());
      PRINT_VAR(state.sigma_tilde().norm());
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


    Timer local_timer;

    local_timer.Reset();
    if (k == 0 || parameters_.rho_changed){
      this -> InitializeSolveForVData(state, solver.get());
      parameters_.rho_changed = false;
    }
    
    //step to solve for v:
    VectorX<double> h = state.z_tilde()-state.u_tilde()+ vc_stab_tilde;
    h = rho* D.cwiseProduct((D+ rho* R).cwiseInverse().cwiseProduct(h));
    VectorX<double> rhs(nv);
    J_tilde.MultiplyByTranspose(h, &rhs);
    rhs += M_tilde_v_star_tilde;
    state.mutable_v_tilde() = solver->Solve(rhs);
    
    //update vc_tilde:
    J_tilde.Multiply(state.v_tilde(), &mutable_cache.vc_tilde); 

    //step to solve for sigma:
    state.mutable_sigma_tilde() = rho* R_sqrt.cwiseProduct(
                D_sqrt.cwiseProduct((D+ rho*R).cwiseInverse().cwiseProduct(
                  state.z_tilde() - state.u_tilde() + vc_stab_tilde - cache.vc_tilde)));
    
    //calculate g_tilde for the ease of life:
    VectorX<double> g_tilde = cache.vc_tilde + Dinv_sqrt.cwiseProduct(R_sqrt.cwiseProduct(
                              state.sigma_tilde())) - vc_stab_tilde;

    //step for z_tilde:
    const auto& mu_star = mu.cwiseInverse();
    ConvexSolverBase<double>::ProjectIntoDWarpedCone(parameters_.soft_tolerance, 
                                      mu_star, data_.D, g_tilde+state.u_tilde(), 
                                      &state.mutable_z_tilde());

    //step for u tilde:
    state.mutable_u_tilde() += g_tilde - state.z_tilde();

    const auto [mom_l2, mom_max] =
        this->CalcTildeScaledMomentumError( state.v_tilde(), state.sigma_tilde());
    
    if (parameters_.verbosity_level >= 4) {
      PRINT_VAR(mom_l2);
      PRINT_VAR(mom_max);
    }
    //TODO: make these two variables dimensionless
    //DRAKE_DEMAND(mom_l2 < 2.0E-14);
    //DRAKE_DEMAND(mom_max < 2.0E-14);

    double uz_product = state.u_tilde().dot(state.z_tilde());
    VectorX<double> dv_tilde = state.v_tilde()-tilde_data_.v_star_tilde;
    VectorX<double> M_tilde_dv_tilde(nv);
    M_tilde.Multiply(dv_tilde, &M_tilde_dv_tilde);
    double l = M_tilde_dv_tilde.dot(dv_tilde);
    l += state.sigma_tilde().dot(state.sigma_tilde());
    uz_product /= l;
    //IMPORTANT: uz_product can be machine epsilon if soft_tolerance is machine epsilon
    DRAKE_DEMAND(uz_product < parameters_.rel_tolerance);

    if (parameters_.log_stats) {
       stats_.iteration_metrics.push_back(metrics);
    }

    const bool converged =
        this -> CheckConvergenceCriteria(g_tilde, state.z_tilde(), state.sigma_tilde(),
                                 rho* state.u_tilde(), cache.vc_tilde);

    if (converged) {
      if (parameters_.verbosity_level >= 1) {
        std::cout << "Iteration converged at: " << k << std::endl;
        //PRINT_VAR(cache.vc.norm());
        std::cout << std::string(80, '=') << std::endl;
      }
      break;
    }

  }

  if (k == parameters_.max_iterations) {
    stats_history_.push_back(stats_);
    this -> LogFailureData("failure_log.dat");
  }
  

  if (k == parameters_.max_iterations) return ContactSolverStatus::kFailure;
  
  VectorX<double> v = H.cwiseInverse().cwiseSqrt().cwiseProduct(state.v_tilde());
  VectorX<double> sigma = Rinv.cwiseSqrt().cwiseProduct(state.sigma_tilde());
  VectorX<double> vc = D_sqrt.cwiseProduct(cache.vc_tilde);

  //  auto& last_metrics = stats_.iteration_metrics.back();
  // last_metrics.r_norm_l2 = (const_cache.g- const_cache.z).norm();
  // last_metrics.s_norm_l2 = R.cwiseProduct(rho*const_cache.u+state.sigma()).norm();
  // last_metrics.rho = parameters_.rho;

  // if (!parameters_.log_stats) stats_.iteration_metrics.push_back(metrics);
  stats_.num_iters = stats_.iteration_metrics.size();

  PackContactResults(data_,v,vc, sigma, results);
  // stats_.total_time = global_timer.Elapsed();

  // solution_history_.emplace_back(AdmmSolutionData<double>{
  //     nc, cache.vc, state.sigma(), data_.contact_data->get_mu(), data_.R});

  // // Update stats history.
   stats_history_.push_back(stats_);

  // total_time_ += global_timer.Elapsed();

  return ContactSolverStatus::kSuccess;
}

template<typename T>
std::pair<T, T> AdmmSolver<T>::CalcTildeScaledMomentumError(const VectorX<T>& v_tilde,
                               const VectorX<T>& sigma_tilde) const{
  const int nv = data_.nv;
  const auto& v_star = data_.dynamics_data->get_v_star();
  const auto& Djac = data_.Djac;
  const auto& M_tilde = tilde_data_.Mblock_tilde;
  const auto& D_sqrt = data_.D_sqrt;
  const auto& Rinv_sqrt = data_.Rinv.cwiseSqrt();
  const auto& J_tilde = tilde_data_.Jblock_tilde;
  const auto& M_tilde_v_star_tilde = tilde_data_.M_tilde_v_star_tilde;
  const auto& H = tilde_data_.H;

  VectorX<T> M_tilde_v_tilde(nv);
  M_tilde.Multiply(v_tilde, &M_tilde_v_tilde);
  VectorX<T> momentum_balance = M_tilde_v_tilde - M_tilde_v_star_tilde;

  VectorX<T> sigma_tilde_term(nv);

  J_tilde.MultiplyByTranspose(D_sqrt.cwiseProduct(Rinv_sqrt.cwiseProduct(sigma_tilde)), &sigma_tilde_term);
  momentum_balance -= sigma_tilde_term;
  momentum_balance = H.cwiseSqrt().cwiseProduct(momentum_balance);

  // Scale momentum balance using the mass matrix's Jacobi preconditioner so
  // that all entries have the same units and we can compute a fair error
  // metric.
  momentum_balance = Djac.asDiagonal() * momentum_balance;

  const T mom_l2 = momentum_balance.norm();
  const T mom_max = momentum_balance.template lpNorm<Eigen::Infinity>();
  return std::make_pair(mom_l2, mom_max);

}


template <typename T>
bool AdmmSolver<T>::CheckConvergenceCriteria( const VectorX<T>& g_tilde, 
                        const VectorX<T>& z_tilde, const VectorX<T>& sigma_tilde, 
                        const VectorX<T>& y_tilde, const VectorX<T>& vc_tilde){
  using std::max;

  const double& abs_tol = parameters_.abs_tolerance;
  const double& rel_tol = parameters_.rel_tolerance;
  const auto& D_sqrt = data_.D_sqrt;
  const auto& Dinv_sqrt = data_.Dinv_sqrt;
  const auto& R = data_.R;
  const auto& R_sqrt = R.cwiseSqrt();
  const auto& vc_stab = data_.vc_stab;
  const auto& r_s_ratio = parameters_.r_s_ratio;
  auto& rho = parameters_.rho;
  const auto& rho_factor = parameters_.rho_factor;
  auto& rho_changed = parameters_.rho_changed;
  const auto& vc = D_sqrt.cwiseProduct(vc_tilde);
  
  //r_norm = ||D^0.5(g_tilde-z_tilde)||
  double r_norm = D_sqrt.cwiseProduct((g_tilde-z_tilde)).norm();
  
  //s_norm = ||R^0.5 (sigma_tilde + D^-0.5 R^0.5 y_tilde)||
  double s_norm = R_sqrt.cwiseProduct(Dinv_sqrt.cwiseProduct(R_sqrt.cwiseProduct(y_tilde))
                                      +sigma_tilde).norm();
  
  const double bound = abs_tol+rel_tol*max(vc.norm(), vc_stab.norm());

  if (parameters_.log_stats) {
    stats_.iteration_metrics.back().r_norm_l2 = r_norm;
    stats_.iteration_metrics.back().s_norm_l2 = s_norm;
    stats_.iteration_metrics.back().bound = bound;
    stats_.iteration_metrics.back().rho = rho;
  }

  // //dynamic rho code:
  // if (parameters_.dynamic_rho) {
  //   if (r_norm > r_s_ratio* s_norm && rho < 10000) {
  //     rho *= rho_factor;
  //     *u_tilde = *u_tilde/rho_factor;
  //     rho_changed = true;
  //   }

  //   if(r_norm < r_s_ratio * s_norm && rho > 1e-10) {
  //     rho = rho/rho_factor;
  //     *u_tilde = *u_tilde * rho_factor;
  //     rho_changed = true;
  //   }
  // }
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

//matrix G = rhoD(D+ rho R)^-1, used for the weight matrix in the supernodal solver 
template <typename T>
void AdmmSolver<T>::CalcGMatrix(const VectorX<T>& D, const VectorX<T>& R, 
      const double& rho, std::vector<MatrixX<T>>* G) const{

  const int& nc = data_.nc;
  
  for (int ic = 0, ic3 = 0; ic < nc; ic++, ic3 += 3) {
    const auto& R_ic = R.template segment<3>(ic3);
    const auto& D_ic = D.template segment<3>(ic3);
    //const Vector3<T> Rinv = R_ic.cwiseInverse();
    MatrixX<T>& G_ic = (*G)[ic];
    G_ic = rho * (D_ic.cwiseProduct((D_ic+rho*R_ic).cwiseInverse()).asDiagonal());
  }
}

template <typename T>
void AdmmSolver<T>::InitializeSolveForVData(const State& s, conex::SuperNodalSolver* solver) const {
  auto& cache = s.mutable_cache();
  const auto& D = data_.D;
  const auto& R = data_.R;
  const double& rho = parameters_.rho;

  this->CalcGMatrix(D, R, rho, &cache.G);
  solver->SetWeightMatrix(cache.G);
  solver->Factor();
  //cache.Finv = (R+ rho * R.cwiseProduct(R.cwiseProduct(D.cwiseInverse()))).cwiseInverse();

}


//TODO: implement this function to get more metrics
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

//logs iteration metrics at step of the whole stats
template <typename T>
void AdmmSolver<T>::LogOneTimestepHistory(
    const std::string& file_name, const int& step) const {
  const std::vector<AdmmSolverStats>& stats_hist =this->get_stats_history();
  const auto& stats = stats_hist[step];
  std::ofstream file(file_name);
  file << fmt::format(
      "{} {} {}\n",
      //error metrics
      "r_norm", "s_norm",
      //parameters
      "rho"
      );
  for (const auto& metrics:stats.iteration_metrics) {
    file << fmt::format(
    "{} {} {}\n",
    // Error metrics.
    metrics.r_norm_l2,
    metrics.s_norm_l2,
    metrics.rho
    );
  }
  file.close();
}


//log iteration metrics of last steps in case the algorithm doesn't converge
template <typename T>
void AdmmSolver<T>::LogFailureData(
    const std::string& file_name) const {
  const std::vector<AdmmSolverStats>& stats_hist =this->get_stats_history();
  const auto& stats = stats_hist.back();
  std::ofstream file(file_name);
  file << fmt::format(
      "{} {} {}\n",
      //error metrics
      "r_norm", "s_norm", "bound"
      );
  for (const auto& metrics:stats.iteration_metrics) {
    file << fmt::format(
    "{} {} {}\n",
    // Error metrics.
    metrics.r_norm_l2,
    metrics.s_norm_l2,
    metrics.bound
    );
  }
  file.close();
}



template <typename T>
void AdmmSolver<T>::LogIterationsHistory(
    const std::string& file_name) const {
  const std::vector<AdmmSolverStats>& stats_hist =
      this->get_stats_history();
  if (parameters_.verbosity_level >= 1) {
    PRINT_VAR(stats_hist.size());
  }
  std::ofstream file(file_name);
  file << fmt::format(
      "{} {} {} {} {} {} {} {} {} {} {}\n",
      // Problem size.
      "num_contacts",
      // Number of iterations.
      "num_iters",
      //force output related:
      "sigma_x_sum", "sigma_y_sum", "sigma_z_sum",
      //parameters
      "rho",
      //error metrics
      "r_norm", "s_norm",
      //time metric
      "solve_for_x_time", "solve_for_z_u_time", "total_time", "preproc_time"
      );

  for (const auto& s : stats_hist) {
    const auto& metrics = s.iteration_metrics.back();
    // const int iters = s.iteration_metrics.size();
    const int iters = s.num_iters;
    // Compute some totals and averages.

    file << fmt::format(
        "{} {} {} {} {} {} {} {} {} {} {}\n",
        // Problem size.
        s.num_contacts,
        // Number of iterations.
        iters,
        //force output related: 
        metrics.sigma_x_sum, metrics.sigma_y_sum, metrics.sigma_z_sum,
        //parameters
        metrics.rho,
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
