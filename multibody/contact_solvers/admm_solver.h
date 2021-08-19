#pragma once

#include <iostream>
#include <memory>
#include <fstream>
#include <iostream>
#include<Eigen/Dense>
#include <numeric>
#include <string>

#include <Eigen/SparseCore>
 
using namespace std;
using namespace Eigen;

#include "drake/multibody/contact_solvers/block_sparse_matrix.h"
#include "drake/multibody/contact_solvers/contact_solver.h"
#include "drake/multibody/contact_solvers/contact_solver_utils.h"
#include "drake/multibody/contact_solvers/convex_solver_base.h"
#include "drake/multibody/contact_solvers/supernodal_solver.h"

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

struct AdmmSolverParameters {

  // We monitor convergence of the contact velocities.
  double abs_tolerance{1.0e-6};  // m/s
  double rel_tolerance{1.0e-5};  // Unitless.
  int max_iterations{300};       // Maximum number of iterations.

  bool initialize_force{true}; //whether or not to analytically initialize force
  // Tolerance used in impulse soft norms and soft cones. In Ns.
  // TODO(amcastro-tri): Consider this to have units of N and scale by time
  // step.
  double soft_tolerance{1.0e-7};

  // Tangential regularization factor. We make Rt = Rt_factor * Rn.
  double Rt_factor{1.0e-3};

  //scaling factor for the augmented lagrangian, 
  //maybe move to another place?
  bool dynamic_rho{false};
  double rho{1};
  double rho_factor{2.0};
  double r_s_ratio{10.0};
  bool rho_changed{false};
  int num_rho_changed{0};
  
  // Use supernodal algebra for the linear solver.
  bool use_supernodal_solver{true};

  // The verbosity level determines how much information to print into stdout.
  // These levels are additive. E.g.: level 2 also prints level 0 and 1 info.
  //  0: Nothing gets printed.
  //  1: Prints problem size and error at convergence.
  //  2: Prints sparsity structure.
  //  3: Prints stats at each iteration.
  int verbosity_level{0};

  bool do_max_iterations{false}; //only for getting lyponov function

  bool log_stats{true};
  //whether to use D = R
  bool scale_with_R{false};

  bool use_stiction_guess{false};
};

struct AdmmSolverIterationMetrics {
  // L2 norm of the scaled momentum equation (always satisfied, good check for algorithm)
  double mom_l2{0};
  // Max norm of the scaled momentum equation (always satisfied, good check for algorithm)
  double mom_max{0};

  double mom_rel_l2{0};
  double mom_rel_max{0};

  //values for sigma:
  double sigma_x_sum{0.0};
  double sigma_y_sum{0.0};
  double sigma_z_sum{0.0};

  double rho{0.0};

  double V{0.0};    //lyaponov function


  //For debuggin  purpose only, delete later
  VectorX<double> v_tilde;
  VectorX<double> sigma_tilde;
  VectorX<double> u_tilde;
  VectorX<double> z_tilde;
  
  double v_tilde_norm{0.0};
  double sigma_tilde_norm{0.0};
  double z_tilde_norm{0.0};
  double u_tilde_norm{0.0};


  // Some norms.
  //double vc_norm{0.0};
  //double gamma_norm{0.0};

  //checking whether or nor u_tilda and z_tilda are always perpendicular:
  double u_z_product{0.0};

  //Optimality Condition norms:
  //r_norm = ||g-z||, 
  //r_norm_max is max norm and .._l2 the l2 norm
  double r_norm_max{0.0};
  double r_norm_l2{0.0};
  
  //s_norm = ||R*(y+sigma)||
  //s_norm_max is max norm and .._l2 the l2 norm 
  double s_norm_max{0.0};
  double s_norm_l2{0.0};

  //bound for residuals:
  double bound{0.0};

};

// Intended for debugging only. Remove.
template <typename T>
struct AdmmSolutionData {
#if 0  
  SolutionData(int nv_, int nc_) : nc(nc_) {
    const int nc3 = 3 * nc;
    vc.resize(nc3);
    gamma.resize(nc3);
    mu.resize(nc);
    R.resize(nc3);
  }
#endif 
  int nc;
  VectorX<T> vc;
  VectorX<T> gamma;
  VectorX<T> mu;
  VectorX<T> R;
};

struct AdmmSolverStats {
  int num_contacts{0};
  int num_iters;  // matches iteration_metrics.size() 
  std::vector<AdmmSolverIterationMetrics> iteration_metrics;

  // Performance statistics. All these times are in seconds.

  // Total time for the last call to SolveWithGuess().
  double total_time{0};
  double supernodal_construction_time{0};
  // Time used in pre-processing the data: forming the Delassus operator,
  // computing regularization parameters, etc.
  double preproc_time{0};
  // Time used to assembly the Hessian.
  double solve_for_x_time{0};
  // Time used by the underlying linear solver.
  double solve_for_z_u_time{0};

  double global_time{0};
};

// This solver uses the regularized convex formulation from [Todorov 2014].
// This class must only implement the API ContactSolver::SolveWithGuess(),
// please refer to the documentation in ContactSolver for details.
//
// - [Todorov, 2014] Todorov, E., 2014, May. Convex and analytically-invertible
// dynamics with contacts and constraints: Theory and implementation in MuJoCo.
// In 2014 IEEE International Conference on Robotics and Automation (ICRA) (pp.
// 6054-6061). IEEE.
template <typename T>
class AdmmSolver final : public ConvexSolverBase<T> {
 public:
  AdmmSolver();

  virtual ~AdmmSolver() = default;

  void set_parameters(AdmmSolverParameters& parameters) {
    ConvexSolverBaseParameters base_parameters{parameters.Rt_factor};
    ConvexSolverBase<T>::set_parameters(base_parameters);
    parameters_ = parameters;
  }

  // Retrieves solver statistics since the last call to SolveWithGuess().
  const AdmmSolverStats& get_iteration_stats() const {
    return stats_;
  }

  // Retrieves the history of statistics during the lifetime of this solver for
  // each call to SolveWithGuess().
  const std::vector<AdmmSolverStats>& get_stats_history() const {
    return stats_history_;
  }

  const std::vector<AdmmSolutionData<T>>& solution_history() const {
    return solution_history_;
  }

  // Returns the total time spent on calls to SolveWithGuess().
  double get_total_time() const { return total_time_; }

  void LogIterationsHistory(const std::string& file_name) const;

  void LogFailureData(const std::string& file_name) const;

  void LogSolutionHistory(const std::string& file_name) const;

  void LogOneTimestepHistory(const std::string& file_name, const int& step) const;

  void SaveMatrix(const string& fileName, const MatrixX<T>&  matrix) {
    const static IOFormat CSVFormat(FullPrecision, DontAlignCols, ", ", "\n");
 
    ofstream file(fileName);
    if (file.is_open())
    {
        file << matrix.format(CSVFormat);
        file.close();
    }
  }

  void SaveVector(const string& fileName, const VectorX<T>&  vector) {
    const static IOFormat CSVFormat(FullPrecision, DontAlignCols, ", ", "\n");
 
    ofstream file(fileName);
    if (file.is_open())
    {
        file << vector.format(CSVFormat);
        file.close();
    }
  }


 private:
  // This is not a real cache in the CS sense (i.e. there is no tracking of
  // dependencies nor automatic validity check) but in the sense that this
  // objects stores computations that are function of the solver's state. It is
  // the responsability of the solver to keep these computations
  // properly in sync.
  struct Cache {
    DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(Cache);

    Cache() = default;

    void Resize(int nv, int nc) {
      const int nc3 = 3 * nc;
      vc.resize(nc3);
      delta_v_c_tilde.resize(nc3);
      delta_v_c.resize(nc3);
      // N.B. The supernodal solver needs MatrixX instead of Matrix3.
      G.resize(nc, Matrix3<T>::Zero());
    }

    VectorX<T> vc;     // tilde Contact velocities.
    
    VectorX<T> delta_v_c_tilde;
    VectorX<T> delta_v_c;    //variables for calculating v_c 

    std::vector<MatrixX<T>> G;  //G = rho(D+rhoR)^-1, for building the weight in supernodal solver

  };

  // Everything in this solver is a function of the generalized velocities v.
  // State stores generalized velocities v and cached quantities that are
  // function of v.
  class State {
   public:
    DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(State);

    State() = default;

    State(int nv, int nc) { Resize(nv, nc); }

    void Resize(int nv, int nc) {
      const int nc3 = 3*nc;
      sigma_tilde_.resize(nc3);
      z_tilde_.resize(nc3);
      u_tilde_.resize(nc3);
      cache_.Resize(nv, nc);
    }


    const VectorX<T>& sigma_tilde() const { return sigma_tilde_; }
    VectorX<T>& mutable_sigma_tilde() { return sigma_tilde_;}

    const VectorX<T>& u_tilde() const { return u_tilde_; }
    VectorX<T>& mutable_u_tilde() { return u_tilde_;}

    const VectorX<T>& z_tilde() const { return z_tilde_; }
    VectorX<T>& mutable_z_tilde() { return z_tilde_;}

    const Cache& cache() const { return cache_; }
    Cache& mutable_cache() const { return cache_; }

   private:
    VectorX<T> sigma_tilde_;
    VectorX<T> u_tilde_;
    VectorX<T> z_tilde_;
    mutable Cache cache_;
  };

  struct ProcessedData {
    DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(ProcessedData);

    ProcessedData() = default;
    std::vector<MatrixX<T>> Mt_inverse;   //Mass matrix tilde for supernodal
    VectorX<T> r_tilde;
    BlockSparseMatrix<T> Jblock_tilde_transpose;
    BlockSparseMatrix<T> Mblock_inverse;  //inverse of mass matrix
    VectorX<T> R_tilde;  //R_tilde = D^-1 R;

  };

  //Preprocess data and form tilde data
  void Preprocess(ProcessedData* processed_data);

  //initialize scaling matrix D as approximation of diag(JM^-1 J^T)+R = diag(N)
  void InitializeD(const int& nc,const std::vector<MatrixX<T>>& Mt, 
                  const BlockSparseMatrix<T>& Jblock, VectorX<T>* D);
  // This is the one and only API from ContactSolver that must be implemented.
  // Refere to ContactSolverBase's documentation for details.
  ContactSolverStatus DoSolveWithGuess(
      const typename ConvexSolverBase<T>::PreProcessedData& data,
      const VectorX<T>& v_guess, ContactSolverResults<T>* result) final;

  //LLT factorization for matrix M_tilde+J_tilde^T G J_tilde 
  void InitializeSolveForSigmaTildeData(const State& s, conex::SuperNodalSolver* solver) const;


  //calculate G = rho(D+rhoR)^-1, used in the SetWeightMatrix of supernodal solver
  void CalcGMatrix(const VectorX<T>& D, const VectorX<T>& R, const double& rho, std::vector<MatrixX<T>>* G) const;

  bool CheckConvergenceCriteria(const VectorX<T>& sigma_tilde, 
                        const VectorX<T>& z_tilde, const VectorX<T>& z_tilde_old, 
                        const VectorX<T> delta_v_c, VectorX<T>* u_tilde); 
  
  //calculate normal/tangential rate for utilde/ztilde, for debugging purpose only
  //u should be of length nc3 and slope of length nc
  //slope[i] = abs(sqrt(u[3*i]^2+ u[3*i+1])/u[3*i+2])
  void CalcSlope(const VectorX<T>& u, VectorX<T>* slope) const;

  // Computes iteration metrics between iterations k and k-1 at states s_k and
  // s_kp respectively.
  AdmmSolverIterationMetrics CalcIterationMetrics(
      const State& s_k, const State& s_kp, int num_ls_iterations,
      double alpha) const;

  using ConvexSolverBase<T>::data_;
  AdmmSolverParameters parameters_;
  AdmmSolverStats stats_;
  double total_time_{0};
  std::vector<AdmmSolverStats> stats_history_;
  std::vector<AdmmSolutionData<T>> solution_history_;

  // previous state used in DoSolveWithGuess
  // TODO: think about whether it is really needed
  //Yizhou: probably not needed in ADMM
  mutable State state_prev;

  ProcessedData processed_data_;
};

template <>
ContactSolverStatus AdmmSolver<double>::DoSolveWithGuess(
    const ConvexSolverBase<double>::PreProcessedData&, const VectorX<double>&,
    ContactSolverResults<double>*);

}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake

extern template class ::drake::multibody::contact_solvers::internal::
    AdmmSolver<double>;
