#ifndef DSO_G2O_VERTEX_H
#define DSO_G2O_VERTEX_H

#include "util/NumType.h"

#include <g2o/core/base_vertex.h>
#include <g2o/core/solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/slam3d/se3quat.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer_terminate_action.h>

namespace dso {

class VertexSE3PoseDSO final : public g2o::BaseVertex<6, SE3> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VertexSE3PoseDSO();

  bool read(std::istream& is) override;

  bool write(std::ostream& os) const override;

  void setToOriginImpl() override;

  void oplusImpl(const number_t* update_) override;
};

class VertexPhotometricDSO final : public g2o::BaseVertex<2, AffLight> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VertexPhotometricDSO();

  bool read(std::istream& is) override;

  bool write(std::ostream& os) const override;

  void setToOriginImpl() override;

  void oplusImpl(const number_t* update_) override;
};

class VertexInverseDepthDSO final : public g2o::BaseVertex<1, double> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VertexInverseDepthDSO();

  bool read(std::istream& is) override;

  bool write(std::ostream& os) const override;

  void setToOriginImpl() override;

  void oplusImpl(const number_t* update_) override;

  void SetCenterProjectedTo(float Ku, float Kv, float new_idepth);

  Vec3f GetCenterProjectedTo() { return CenterProjectedTo_;}

 private:
  //// Ku, Kv, new_idepth for windowed optimization.
  Vec3f CenterProjectedTo_;
};

class VertexUVDSO final : public g2o::BaseVertex<1, Vec2> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VertexUVDSO();

  bool read(std::istream& is) override;

  bool write(std::ostream& os) const override;

  void setToOriginImpl() override;

  void oplusImpl(const number_t* update_) override;

  //// Set dx,dy for traceStereo().
  void SetDxDy(double dx, double dy) {
    dx_ = dx;
    dy_ = dy;
  }

 private:
  double dx_;
  double dy_;
};

class VertexCamDSO final : public g2o::BaseVertex<4, Vec4> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  VertexCamDSO();

  bool read(std::istream& is) override;

  bool write(std::ostream& os) const override;

  void setToOriginImpl() override;

  void oplusImpl(const number_t* update_) override;
};

} // namespace dso

#endif // DSO_G2O_VERTEX_H
