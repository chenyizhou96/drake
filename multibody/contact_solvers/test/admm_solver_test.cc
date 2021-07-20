#include "drake/multibody/contact_solvers/admm_solver.h"

#include "drake/common/eigen_types.h"
#include "conex/clique_ordering.h"
#include "gtest/gtest.h"

#undef NDEBUG

namespace drake {
namespace multibody {
namespace contact_solvers {
namespace internal {
namespace {
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::get;
using std::vector;
using drake::Vector3;
using drake::VectorX;
using drake::multibody::contact_solvers::internal::AdmmSolver;

/*GTEST_TEST(AdmmSolver, BasicTests) {

}
*/

template <typename T>
class TestLinearOperator final : public LinearOperator<T> {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(TestLinearOperator)

  explicit TestLinearOperator(const std::string& name)
      : LinearOperator<T>(name) {}

  ~TestLinearOperator() = default;

  int rows() const final { return 3; }
  int cols() const final { return 2; }

  VectorX<T> ExpectedMultiplyResult() const {
    return Vector3<T>(1.0, 2.0, 3.0);
  }

 protected:
  void DoMultiply(const Eigen::Ref<const VectorX<T>>& x,
                  VectorX<T>* y) const final {
    *y = ExpectedMultiplyResult();
  };

  void DoMultiply(const Eigen::Ref<const Eigen::SparseVector<T>>& x,
                  Eigen::SparseVector<T>* y) const final {
    *y = ExpectedMultiplyResult().sparseView();
  }
};

GTEST_TEST(AdmmSolver, ProjectionTest) {
  const TestLinearOperator<double> Aop("A");
  VectorX<double> mu(1);
  mu(0) = 0.5;
  VectorX<double> D(3);
  D(0) = 0.1; D(1) = 0.1; D(2) = 2;
  VectorX<double> vc1 (3);
  vc1(0) = 0; vc1(1) = 0; vc1(2) = 2;
  VectorX<double> vc2(3);
  vc2(0) = 1; vc2(1) = 1; vc2(2) = 0;
  VectorX<double> vc3 (3);
  vc3(0) = 0; vc3(1) = 0; vc3(2) = -1;
  AdmmSolver<double> admm_solver;
  VectorX<double> gamma1(3), gamma2(3), gamma3(3);

  const double soft_tol = 1e-7;

  admm_solver.TestCalcStarAnalyticalInverseDynamicsHelper(soft_tol, mu, D, vc1, Aop, &gamma1);
  admm_solver.TestCalcStarAnalyticalInverseDynamicsHelper(soft_tol, mu, D, vc2, Aop, &gamma2);
  admm_solver.TestCalcStarAnalyticalInverseDynamicsHelper(soft_tol, mu, D, vc3, Aop, &gamma3);
  
  EXPECT_EQ(gamma1(0), 0);
  EXPECT_EQ(gamma1(1), 0);
  EXPECT_EQ(gamma1(2), 2);
  EXPECT_NEAR(gamma2(0), 0.98765432098765, 1e-14);
  EXPECT_NEAR(gamma2(1),  0.98765432098765, 1e-14);
  EXPECT_NEAR(gamma2(2), 0.698377067838567, 1e-14);
  EXPECT_EQ(gamma3(0), 0);
  EXPECT_EQ(gamma3(1), 0);
  EXPECT_EQ(gamma3(2), 0);
}



}  // namespace
}  // namespace internal
}  // namespace contact_solvers
}  // namespace multibody
}  // namespace drake