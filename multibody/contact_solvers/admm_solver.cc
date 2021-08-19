#include "drake/multibody/contact_solvers/admm_solver.h"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>
#include <Eigen/Dense>
#include <vector>

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
using Eigen::Matrix;
//using drake::multibody::contact_solvers::internal::BlockSparseMatrix;

//using namespace Eigen;

template<typename M>
M load_csv (const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Map<const VectorX<double>>(values.data(), rows, values.size()/rows);
}


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
  const auto& R = data_.R;
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
    }
  }

  for (int ic = 0, ic3 = 0; ic < nc; ic++, ic3 += 3) {
    (*D)[ic3] = W[ic].norm()/3 + R[ic3];
    (*D)[ic3+1] = W[ic].norm()/3 + R[ic3+1];
    (*D)[ic3+2] = W[ic].norm()/3 + R[ic3+2];
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

template <typename T>
void AdmmSolver<T>::Preprocess( ProcessedData* processed_data) {
  const int nv = data_.nv;
  const int nc = data_.nc;
  const int nc3 = 3*nc;
  const auto& Jblock = data_.Jblock;
  const auto& Mblock = data_.Mblock;
  const auto& Mt = data_.Mt;
  const auto& vc_stab = data_.vc_stab;
  const auto& v_star = data_.dynamics_data->get_v_star();
  const auto& D = data_.D;
  const auto& Dinv = data_.Dinv;
  const auto& Dinv_sqrt = data_.Dinv_sqrt;
  const auto& R = data_.R;

  //Mt_inverse = Mt^-1
  processed_data -> Mt_inverse.resize(Mt.size());
  for (int i = 0; i < Mt.size(); i++) {
    processed_data -> Mt_inverse[i] = Mt[i].inverse();
  }
  
  processed_data-> R_tilde = R.cwiseProduct(Dinv);

  //r_tilde = D^-0.5 (J v_star - v_hat)
  VectorX<T> J_v_star(nc3);
  Jblock.Multiply(v_star, &J_v_star);
  processed_data -> r_tilde = Dinv_sqrt.cwiseProduct(J_v_star - vc_stab);

  //Jblock_tilde_transpose = J^TD^-0.5
  BlockSparseMatrixBuilder<T> Jblock_tilde_transpose_builder(Jblock.block_cols(), Jblock.block_rows(),
                                      Jblock.num_blocks());
  for (auto [p, t, Jpt] : Jblock.get_blocks()) {
    const int i0 = Jblock.row_start(p);
    const auto& Dinv_sqrt_p = Dinv_sqrt.template segment(i0, Jpt.rows());
    MatrixX<T> Jpt_tilde_transpose = (Dinv_sqrt_p.asDiagonal()*Jpt).transpose();
    Jblock_tilde_transpose_builder.PushBlock(t, p, Jpt_tilde_transpose);
  }
  processed_data -> Jblock_tilde_transpose = Jblock_tilde_transpose_builder.Build();

  BlockSparseMatrixBuilder<T> Mblock_inverse_builder(Mblock.block_cols(), Mblock.block_rows(),
                                      Mblock.num_blocks());
  for (auto [p, t, Mpt] : Mblock.get_blocks()) {
    Mblock_inverse_builder.PushBlock(p, t, Mpt.inverse());
  }
  processed_data -> Mblock_inverse = Mblock_inverse_builder.Build();

}


template <>
ContactSolverStatus AdmmSolver<double>::DoSolveWithGuess(
    const ConvexSolverBase<double>::PreProcessedData& data,
    const VectorX<double>& v_guess, ContactSolverResults<double>* results) {
  using std::abs;
  using std::max;
  using std::pow;
  using std::sqrt;

  //pseudocode for the whole process:
  //InitializeD    <- initialize D, D
  //Preprocess    <- calculate J_block_transpose_tilde, M_t_inverse, R_tilde_rho_I as blocks, r_tilde
  //initialization
  //loop:
  //three steps for solving 
  //checkforconvergence
  //end of loop
  //postprocessing       <- calculates v

  // Starts a timer for the overall execution time of the solver.
  Timer global_timer;

  const auto& dynamics_data = *data.dynamics_data;
  const auto& contact_data = *data.contact_data;

  const int nv = dynamics_data.num_velocities();
  const int nc = contact_data.num_contacts();
  const int nc3 = 3 * nc;
  const auto& Rinv = data_.Rinv;
  const auto& J = data_.Jblock;
  const auto& v_star = data_.dynamics_data->get_v_star();

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
  const auto& vc_stab = data_.vc_stab;

  // PRINT_VAR(D.size() / 3);
  // PRINT_VAR(D.transpose());
  // PRINT_VAR(rho);

  //each entry of D must be positive
  DRAKE_DEMAND(D.minCoeff() > 0);
  
  this-> Preprocess(&processed_data_);

  const auto& J_tilde_transpose = processed_data_.Jblock_tilde_transpose;
  const auto& r_tilde = processed_data_.r_tilde;
  const auto& Mt_inverse = processed_data_.Mt_inverse;
  const auto& Rinv_sqrt = data_.Rinv.cwiseSqrt();
  const auto& M_inverse = processed_data_.Mblock_inverse;
  const auto& R = data_.R;
  const auto& R_sqrt = data_.R.cwiseSqrt();
  const auto& R_tilde = processed_data_.R_tilde;


  if (parameters_.verbosity_level >= 5) {
    PRINT_VAR(data_.vc_stab);
    PRINT_VAR(R_tilde);
    PRINT_VAR(D);
    PRINT_VAR(R);
    for (auto [p, t, Jpt] : data_.Jblock.get_blocks()) {
      PRINT_VAR(Jpt);
    }
    for (auto [p, t, J_tilde_transpose_pt] : J_tilde_transpose.get_blocks()) {
      PRINT_VAR(J_tilde_transpose_pt);
    }
    for (auto [p, t, Mpt]: data_.Mblock.get_blocks()){
      PRINT_VAR(Mpt);
    }
    for (auto [p, t, M_inversept]: processed_data_.Mblock_inverse.get_blocks()) {
      PRINT_VAR(M_inversept);
    }
    for (int i = 0; i< data_.Mt.size(); i++) {
      PRINT_VAR(data_.Mt[i]);
      PRINT_VAR(Mt_inverse[i]);
    }
    PRINT_VAR(r_tilde);
    PRINT_VAR(v_star);
  }


  const double original_rho = parameters_.rho;

  
  if (!parameters_.use_stiction_guess){
    J.Multiply(v_guess, &mutable_cache.vc);
    VectorX<double> y_tilde = -Rinv_sqrt.cwiseProduct(cache.vc-vc_stab);
    const auto& mu = data_.contact_data->get_mu();
    ConvexSolverBase<double>::ProjectIntoDWarpedCone(parameters_.soft_tolerance, 
                                      mu, Rinv, y_tilde, 
                                        &state.mutable_sigma_tilde());
    state.mutable_sigma_tilde() = D_sqrt.cwiseProduct(Rinv_sqrt.cwiseProduct(state.sigma_tilde()));
    
    if (!parameters_.initialize_force){
      state.mutable_sigma_tilde() = VectorX<double>::Zero(nc3);
    }

    //z_tilde = sigma_tilde
    state.mutable_z_tilde() = state.sigma_tilde();

    //u_tilde = -(delta_v_c_tilde+R_tilde sigma_tilde + r_tilde)/rho
    VectorX<double> temp1(nv);
    J_tilde_transpose.Multiply(state.sigma_tilde(), &temp1);
    VectorX<double> temp2(nv);
    M_inverse.Multiply(temp1, &temp2);
    J_tilde_transpose.MultiplyByTranspose(temp2, &mutable_cache.delta_v_c_tilde);
    auto& u_tilde = state.mutable_u_tilde();
    u_tilde = -(cache.delta_v_c_tilde+R_tilde.cwiseProduct(state.sigma_tilde()) + r_tilde)/rho;
  } else {
    //initialize according to stiction criterion, first set rho = 0 then change it back
    state.mutable_sigma_tilde() = VectorX<double>::Zero(nc3);
    state.mutable_z_tilde() = state.sigma_tilde();
    state.mutable_u_tilde() = state.sigma_tilde();
    parameters_.rho = 0;
    

  }
  



  stats_.preproc_time += preproc_timer.Elapsed();

  // Super nodal solver is constructed once per time-step to reuse structure
  // of M and J.
  std::unique_ptr<conex::SuperNodalSolver> solver;
  Timer construction_timer;
  stats_.supernodal_construction_time = construction_timer.Elapsed();
  
  if (parameters_.verbosity_level >=3) {
      PRINT_VAR(state.sigma_tilde());
      PRINT_VAR(state.sigma_tilde().norm());
      PRINT_VAR(state.z_tilde().norm());
      PRINT_VAR(state.u_tilde().norm());
      PRINT_VAR(data_.D.minCoeff());
    }
  VectorX<double> z_tilde_old = state.z_tilde();

  VectorX<double> z_tilde_star;
  VectorX<double> u_tilde_star;

  if (parameters_.do_max_iterations) {
    z_tilde_star = load_csv<VectorX<double>>("z_tilde_star.csv");
    u_tilde_star = load_csv<VectorX<double>>("u_tilde_star.csv");
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

    //log start data for analysis:
    if (parameters_.log_stats) {
      stats_.iteration_metrics.push_back(metrics);
      //collect data to see if scaling is right:
      stats_.iteration_metrics.back().v_tilde = state.sigma_tilde();
      stats_.iteration_metrics.back().sigma_tilde = state.sigma_tilde();
      stats_.iteration_metrics.back().u_tilde = state.u_tilde();
      stats_.iteration_metrics.back().z_tilde = state.z_tilde();
      stats_.iteration_metrics.back().v_tilde_norm = 0;
      stats_.iteration_metrics.back().sigma_tilde_norm = state.sigma_tilde().norm();
      stats_.iteration_metrics.back().z_tilde_norm = state.z_tilde().norm();
      stats_.iteration_metrics.back().u_tilde_norm = state.u_tilde().norm();
    }

    
    local_timer.Reset();
    if (k == 0 || parameters_.rho_changed){
      std::vector<MatrixX<double>>R_tilde_rho_I(nc3, MatrixX<double>(1,1));
      for (int i = 0; i < nc3; i++) {
        R_tilde_rho_I[i] << R_tilde[i] + rho;
      }
      solver = std::make_unique<conex::SuperNodalSolver>(
        J_tilde_transpose.block_rows(), J_tilde_transpose.get_blocks(), R_tilde_rho_I);

      solver->SetWeightMatrix(Mt_inverse);
      if (parameters_.verbosity_level > 1) {
        PRINT_VAR(solver -> FullMatrix());
        PRINT_VAR(parameters_.rho);
      }
      solver->Factor();
      parameters_.rho_changed = false;
      if (parameters_.rho == 0) {
        parameters_.rho = original_rho;
        parameters_.rho_changed = true;
      }
    }

    
    //step to solve for sigma_tilde:
    VectorX<double> rhs = rho*(state.z_tilde()-state.u_tilde())-r_tilde;
    state.mutable_sigma_tilde() = solver->Solve(rhs);



    VectorX<double> temp1(nv);
    J_tilde_transpose.Multiply(state.sigma_tilde(), &temp1);
    VectorX<double> temp2(nv);
    //PRINT_VAR(temp1);
    M_inverse.Multiply(temp1, &temp2);
    //PRINT_VAR(temp2);
    J_tilde_transpose.MultiplyByTranspose(temp2, &mutable_cache.delta_v_c_tilde);
    VectorX<double> lhs = cache.delta_v_c_tilde+ R_tilde.cwiseProduct(state.sigma_tilde()) +rho*state.sigma_tilde();
    double residue = (lhs - rhs).norm()/max(rhs.norm(),lhs.norm());
    if (k == 0 && parameters_.use_stiction_guess){
      double initial_residue = (cache.delta_v_c_tilde+ R_tilde.cwiseProduct(state.sigma_tilde())+
                                    r_tilde).norm()/max(rhs.norm(),lhs.norm());
      PRINT_VAR(initial_residue);
      DRAKE_DEMAND(initial_residue < 1.0E-14);
    } else {
      //PRINT_VAR(residue);
      DRAKE_DEMAND(residue < 1.0E-14);
    }
    
    
    z_tilde_old = state.z_tilde();
    if(parameters_.verbosity_level > 1){
      PRINT_VAR(rhs);
      PRINT_VAR(z_tilde_old);
      PRINT_VAR(state.sigma_tilde());
    }

    //step for z_tilde:
    const auto& mu = data_.contact_data->get_mu();
    ConvexSolverBase<double>::ProjectIntoDWarpedCone(parameters_.soft_tolerance, 
                                      mu, data_.Dinv, state.sigma_tilde()+state.u_tilde(), 
                                      &state.mutable_z_tilde());

    if(parameters_.verbosity_level > 1){
      PRINT_VAR(state.z_tilde());
      PRINT_VAR(state.sigma_tilde());
    }

    if (parameters_.verbosity_level >=4) {
      if (false) {
        PRINT_VAR(k);
        PRINT_VAR(state.z_tilde());
        PRINT_VAR(state.u_tilde());
        PRINT_VAR(state.sigma_tilde());
        PRINT_VAR(mu);
        PRINT_VAR(data_.D);
        //SaveVector("z_tilde_big.csv", state.z_tilde());
      }
    }

    //step for u tilde:
    state.mutable_u_tilde() += state.sigma_tilde() - state.z_tilde();
    
    if (parameters_.verbosity_level >=2){
      PRINT_VAR(mu);
      VectorX<double> sigma = Dinv_sqrt.cwiseProduct(state.sigma_tilde());
      VectorX<double> J_t_sigma(nv);
      VectorX<double> J_v(nc3);
      J.MultiplyByTranspose(sigma, &J_t_sigma);
      VectorX<double> velocity(nv);
      M_inverse.Multiply(J_t_sigma, &velocity);
      velocity += v_star;
      if (k == 0) {
        PRINT_VAR(velocity);
      }
      J.Multiply(velocity, &J_v);
      if (k == 0){
        PRINT_VAR(data_.R);
        PRINT_VAR(J_v);
        PRINT_VAR(vc_stab);
        VectorX<double> other_sigma = data_.Rinv.cwiseProduct(J_v - vc_stab);
        PRINT_VAR(other_sigma);
        PRINT_VAR(sigma);
        PRINT_VAR(state.sigma_tilde());
        PRINT_VAR(state.z_tilde());
        PRINT_VAR(state.u_tilde());
      }
    }

    
    stats_.iteration_metrics.back().V = pow((state.u_tilde() - u_tilde_star).norm(), 2)
                                        + rho* pow((state.z_tilde() - z_tilde_star).norm(), 2);

    double uz_product = state.u_tilde().dot(state.z_tilde());
    VectorX<double> N_tilde_sigma_tilde = lhs-rho*state.sigma_tilde();
    double l =  r_tilde.dot(r_tilde);
    uz_product /= l;
    //IMPORTANT: uz_product is machine epsilon if soft_tolerance is machine epsilon
    if (uz_product > 1.0E-13){
      PRINT_VAR(uz_product);
    }
    DRAKE_DEMAND(uz_product < 1.0E-13);


    if (parameters_.verbosity_level >=4) {
      const auto& y_tilde = rho* state.u_tilde();
      const auto& R_sqrt = data_.R.cwiseSqrt();
      double s_norm = R_sqrt.cwiseProduct(Dinv_sqrt.cwiseProduct(R_sqrt.cwiseProduct(y_tilde))
                                      +state.sigma_tilde()).norm();
      if (false) {
        PRINT_VAR(k);
        PRINT_VAR(state.z_tilde());
        PRINT_VAR(state.u_tilde());
      }
    }

    // VectorX<double> temp1(nv);
    // J_tilde_transpose.Multiply(state.sigma_tilde(), &temp1);
    // VectorX<double> temp2(nv);
    // M_inverse.Multiply(temp1, &temp2);
    // J_tilde_transpose.MultiplyByTranspose(temp2, &mutable_cache.delta_v_c_tilde);
     mutable_cache.delta_v_c = D_sqrt.cwiseProduct(cache.delta_v_c_tilde);
    
    VectorX<double> y_tilde = rho*state.u_tilde();
     bool converged =
        this -> CheckConvergenceCriteria(state.sigma_tilde(), state.z_tilde(), z_tilde_old, 
                                        cache.delta_v_c,&state.mutable_u_tilde());
    if (parameters_.do_max_iterations) {
      converged = false;
    } 
    

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
    //SaveVector("z_tilde_star.csv", state.z_tilde());
    //SaveVector("u_tilde_star.csv", state.u_tilde());
  }
  

  if (k == parameters_.max_iterations) return ContactSolverStatus::kFailure;
  
  VectorX<double> Jt_sigma(nv);
  VectorX<double> v(nv);
  VectorX<double> sigma = Dinv_sqrt.cwiseProduct(state.sigma_tilde());
  J.MultiplyByTranspose(sigma, &Jt_sigma);
  M_inverse.Multiply(Jt_sigma, &v);
  v += v_star;
  VectorX<double> vc(nc3);
  J.Multiply(v, &vc);

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



template <typename T>
bool AdmmSolver<T>::CheckConvergenceCriteria( const VectorX<T>& sigma_tilde, 
                        const VectorX<T>& z_tilde, const VectorX<T>& z_tilde_old, 
                        const VectorX<T> delta_v_c, VectorX<T>* u_tilde){
  using std::max;

  const double& abs_tol = parameters_.abs_tolerance;
  const double& rel_tol = parameters_.rel_tolerance;
  const int nc = data_.nc;
  const int nc3 = 3*nc;
  const auto& D_sqrt = data_.D_sqrt;
  const auto& Dinv_sqrt = data_.Dinv_sqrt;
  const auto& R = data_.R;
  const auto& R_sqrt = R.cwiseSqrt();
  const auto& vc_stab = data_.vc_stab;
  const auto& r_s_ratio = parameters_.r_s_ratio;
  const auto& J = data_.Jblock;
  auto& rho = parameters_.rho;
  const auto& rho_factor = parameters_.rho_factor;
  auto& rho_changed = parameters_.rho_changed;
  const auto& v_star = data_.dynamics_data->get_v_star();
  
  VectorX<T> v_c(nc3);
  J.Multiply(v_star, &v_c);
  v_c += delta_v_c;
  //r_norm = ||D^0.5(sigma_tilde-z_tilde)||
  double r_norm = D_sqrt.cwiseProduct((sigma_tilde-z_tilde)).norm();
  
  //s_norm = ||R^0.5 (sigma_tilde + D^-0.5 R^0.5 y_tilde)||
  double s_norm = rho* D_sqrt.cwiseProduct((z_tilde_old-z_tilde)).norm();
  
  const double bound = abs_tol+rel_tol*max(v_c.norm(), vc_stab.norm());

  if (parameters_.log_stats) {
    stats_.iteration_metrics.back().r_norm_l2 = r_norm;
    stats_.iteration_metrics.back().s_norm_l2 = s_norm;
    stats_.iteration_metrics.back().bound = bound;
    stats_.iteration_metrics.back().rho = rho;
  }

  // //dynamic rho code:
  if (parameters_.dynamic_rho) {
    if (r_norm > r_s_ratio* s_norm && rho < 500) {
      rho *= rho_factor;
      *u_tilde = *u_tilde/rho_factor;
      rho_changed = true;
    }

    if(s_norm > r_s_ratio * r_norm && rho > 1e-5) {
      rho = rho/rho_factor;
      *u_tilde = *u_tilde * rho_factor;
      rho_changed = true;
    }
  }
  if (rho_changed == true){
    parameters_.num_rho_changed +=1;
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
void AdmmSolver<T>::InitializeSolveForSigmaTildeData(const State& s, conex::SuperNodalSolver* solver) const {
  auto& cache = s.mutable_cache();
  const int nc = data_.nc;
  const int nc3 = nc*3;
  const auto& D = data_.D;
  const auto& R = data_.R;
  const auto& Mt_inverse = processed_data_.Mt_inverse;
  const double& rho = parameters_.rho;
  const auto& J_tilde_transpose = processed_data_.Jblock_tilde_transpose;
  const auto& R_tilde = processed_data_.R_tilde;

  std::vector<MatrixX<T>>R_tilde_rho_I(nc3, MatrixX<T>(1,1));
  //R_tilde_rho_I[0] = processed_data_.R_tilde.asDiagonal()+ rho*MatrixX<T>::Identity(nc3, nc3);

  //solver = std::make_unique<conex::SuperNodalSolver>(
    //J_tilde_transpose.block_rows(), J_tilde_transpose.get_blocks(), R_tilde_rho_I);

  solver->SetWeightMatrix(Mt_inverse);
  solver->Factor();

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
  const auto& stats = stats_hist[step-1];
  std::ofstream file(file_name);
  file << fmt::format(
      "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n",
      //error metrics
      "r_norm", "s_norm",
      //parameters
      "rho", "num_rho_changed",
      //variable data:
      "v_tilde_0", "v_tilde_1","v_tilde_2","v_tilde_3","v_tilde_4","v_tilde_5",
      "sigma_tilde_0", "sigma_tilde_1","sigma_tilde_2","sigma_tilde_3","sigma_tilde_4",
      "sigma_tilde_5", "sigma_tilde_6", "sigma_tilde_7", "sigma_tilde_8",
      "z_tilde_0", "z_tilde_1","z_tilde_2","z_tilde_3","z_tilde_4",
      "z_tilde_5", "z_tilde_6", "z_tilde_7", "z_tilde_8",
      "u_tilde_0", "u_tilde_1","u_tilde_2","u_tilde_3","u_tilde_4",
      "u_tilde_5", "u_tilde_6", "u_tilde_7", "u_tilde_8"
      );
  for (const auto& metrics:stats.iteration_metrics) {
    file << fmt::format(
    "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n",
    // Error metrics.
    metrics.r_norm_l2,
    metrics.s_norm_l2,
    metrics.rho, parameters_.num_rho_changed,
    metrics.v_tilde[0], metrics.v_tilde[1], metrics.v_tilde[2],
    metrics.v_tilde[3], metrics.v_tilde[4], metrics.v_tilde[5],
    metrics.sigma_tilde[0], metrics.sigma_tilde[1], metrics.sigma_tilde[2],
    metrics.sigma_tilde[3], metrics.sigma_tilde[4], metrics.sigma_tilde[5],
    metrics.sigma_tilde[6], metrics.sigma_tilde[7], metrics.sigma_tilde[8],
    metrics.z_tilde[0], metrics.z_tilde[1], metrics.z_tilde[2],
    metrics.z_tilde[3], metrics.z_tilde[4], metrics.z_tilde[5],
    metrics.z_tilde[6], metrics.z_tilde[7], metrics.z_tilde[8],
    metrics.u_tilde[0], metrics.u_tilde[1], metrics.u_tilde[2],
    metrics.u_tilde[3], metrics.u_tilde[4], metrics.u_tilde[5],
    metrics.u_tilde[6], metrics.u_tilde[7], metrics.u_tilde[8]
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
      "{} {} {} {}\n",
      //error metrics
      "r_norm", "s_norm", "bound", "V"
      );
  for (const auto& metrics:stats.iteration_metrics) {
    file << fmt::format(
    "{} {} {} {}\n",
    // Error metrics.
    metrics.r_norm_l2,
    metrics.s_norm_l2,
    metrics.bound,
    metrics.V
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
      "{} {} {} {} {} {} {} {} {} {}\n",
      // Problem size.
      "num_contacts",
      // Number of iterations.
      "num_iters",
      //parameters
      "rho", "num_rho_changed",
      //error metrics
      "r_norm", "s_norm",
      //norms:
      "v_tilde_norm", "sigma_tilde_norm", "z_tilde_norm", "u_tilde_norm"
      //variable data:
      // "v_tilde_0", "v_tilde_1","v_tilde_2","v_tilde_3","v_tilde_4","v_tilde_5",
      // "sigma_tilde_0", "sigma_tilde_1","sigma_tilde_2","sigma_tilde_3","sigma_tilde_4",
      // "sigma_tilde_5", "sigma_tilde_6", "sigma_tilde_7", "sigma_tilde_8",
      // "z_tilde_0", "z_tilde_1","z_tilde_2","z_tilde_3","z_tilde_4",
      // "z_tilde_5", "z_tilde_6", "z_tilde_7", "z_tilde_8",
      // "u_tilde_0", "u_tilde_1","u_tilde_2","u_tilde_3","u_tilde_4",
      // "u_tilde_5", "u_tilde_6", "u_tilde_7", "u_tilde_8"
      );

  for (const auto& s : stats_hist) {
    const auto& metrics = s.iteration_metrics.back();
    const auto& metrics_0 = s.iteration_metrics[0];
    // const int iters = s.iteration_metrics.size();
    const int iters = s.num_iters;
    // Compute some totals and averages.

    file << fmt::format(
        "{} {} {} {} {} {} {} {} {} {}\n",
        // Problem size.
        s.num_contacts,
        // Number of iterations.
        iters,
        //parameters
        metrics.rho,
        parameters_.num_rho_changed,
        // Error metrics.
        metrics.r_norm_l2,
        metrics.s_norm_l2,
        //norms:
        metrics_0.v_tilde_norm, metrics_0.sigma_tilde_norm, metrics_0.z_tilde_norm, metrics_0.u_tilde_norm
        //initialization related:
        // metrics_0.v_tilde[0], metrics_0.v_tilde[1], metrics_0.v_tilde[2],
        // metrics_0.v_tilde[3], metrics_0.v_tilde[4], metrics_0.v_tilde[5],
        // metrics_0.sigma_tilde[0], metrics_0.sigma_tilde[1], metrics_0.sigma_tilde[2],
        // metrics_0.sigma_tilde[3], metrics_0.sigma_tilde[4], metrics_0.sigma_tilde[5],
        // metrics_0.sigma_tilde[6], metrics_0.sigma_tilde[7], metrics_0.sigma_tilde[8],
        // metrics_0.z_tilde[0], metrics_0.z_tilde[1], metrics_0.z_tilde[2],
        // metrics_0.z_tilde[3], metrics_0.z_tilde[4], metrics_0.z_tilde[5],
        // metrics_0.z_tilde[6], metrics_0.z_tilde[7], metrics_0.z_tilde[8],
        // metrics_0.u_tilde[0], metrics_0.u_tilde[1], metrics_0.u_tilde[2],
        // metrics_0.u_tilde[3], metrics_0.u_tilde[4], metrics_0.u_tilde[5],
        // metrics_0.u_tilde[6], metrics_0.u_tilde[7], metrics_0.u_tilde[8]


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
