#include "drake/multibody/contact_solvers/convex_solver_base.h"

#include "drake/common/eigen_types.h"
#include "conex/clique_ordering.h"
#include "gtest/gtest.h"

#undef NDEBUG

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::get;
using std::vector;
using drake::Vector3;
using drake::VectorX;

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

struct ConvexSolverBaseTester {
  static void CalcStarAnalyticalInverseDynamics(double soft_norm_tolerance,
                                                const VectorXd& mu,
                                                const VectorXd& D,
                                                const VectorXd& g,
                                                VectorXd* z) const {
    ConvexSolverBase<double>::CalcStarAnalyticalInverseDynamics(
        soft_norm_tolerance, mu, D, g, z);
  }
};

namespace {

GTEST_TEST(AdmmSolver, ProjectionTest) {
  VectorX<double> mu(1);
  mu(0) = 0.5;
  VectorX<double> D(3);
  D(0) = 0.1;
  D(1) = 0.1;
  D(2) = 2;
  VectorX<double> g1(3);
  g1(0) = 0;
  g1(1) = 0;
  g1(2) = 2;
  VectorX<double> z1(3);

  const double soft_tol = 1e-7;
  ConvexSolverBaseTester::CalcStarAnalyticalInverseDynamics(soft_tol, mu, D, g1,
                                                            &z1);
  
  EXPECT_EQ(z1(0), 0);
  EXPECT_EQ(z1(1), 0);
  EXPECT_EQ(z1(2), 2);
}

}  // namespace
}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake