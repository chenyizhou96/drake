#include "drake/multibody/contact_solvers/convex_solver_base.h"

#include "drake/common/eigen_types.h"
#include "conex/clique_ordering.h"
#include "gtest/gtest.h"

#undef NDEBUG

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {

using Eigen::VectorXd;

struct ConvexSolverBaseTester {
  static void ProjectIntoConeWithDnorm(double soft_norm_tolerance,
                                                const VectorXd& mu,
                                                const VectorXd& D,
                                                const VectorXd& g,
                                                VectorXd* z) {
    ConvexSolverBase<double>::ProjectIntoConeWithDnorm(
        soft_norm_tolerance, mu, D, g, z);
  }
};


namespace {
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::get;
using std::vector;
using drake::Vector3;
using drake::VectorX;
using drake::multibody::contact_solvers::internal::AdmmSolver;
using drake::multibody::contact_solvers::internal::ConvexSolverBase;


GTEST_TEST(AdmmSolver, ProjectionTest) {
  std::cout << "reached here";
  VectorX<double> mu(1);
  mu(0) = 0.5;
  VectorX<double> D(3);
  D(0) = 0.1; D(1) = 0.1; D(2) = 2;
  VectorX<double> g1 (3);
  g1(0) = 0; g1(1) = 0; g1(2) = 2;
  VectorX<double> g2(3);
  g2(0) = 1; g2(1) = 1; g2(2) = 0;
  VectorX<double> g3 (3);
  g3(0) = 0; g3(1) = 0; g3(2) = -1;
  VectorX<double> z1(3), z2(3), z3(3);

  const double soft_tol = 1e-7;

  ConvexSolverBaseTester::ProjectIntoConeWithDnorm(soft_tol, mu, D, g1,
                                                            &z1);

  
  EXPECT_EQ(z1(0), 0);
  EXPECT_EQ(z1(1), 0);
  EXPECT_EQ(z1(2), 2);
/*  EXPECT_NEAR(z2(0), 0.98765432098765, 1e-14);
  EXPECT_NEAR(z2(1),  0.98765432098765, 1e-14);
  EXPECT_NEAR(z2(2), 0.698377067838567, 1e-14);
  EXPECT_EQ(z3(0), 0);
  EXPECT_EQ(z3(1), 0);
  EXPECT_EQ(z3(2), 0);*/
}



}  // namespace
}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake