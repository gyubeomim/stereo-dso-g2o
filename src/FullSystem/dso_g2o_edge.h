#ifndef DSO_G2O_EDGE_H_
#define DSO_G2O_EDGE_H_

#include "dso_g2o_vertex.h"
#include "dso_util.hpp"

#include "util/NumType.h"
#include "util/globalFuncs.h"
#include "FullSystem/HessianBlocks.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>

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


#include <Eigen/Core>
#include <Eigen/Dense>

#include <iostream>
#include <sstream>
#include <vector>

namespace dso {

/// \class EdgeLBASE3PosePhotoIdepthCamDSO class.
class EdgeLBASE3PosePhotoIdepthCamDSO : public ::g2o::BaseMultiEdge<8, Vec8f> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EdgeLBASE3PosePhotoIdepthCamDSO(PointFrameResidual* r)
      : r_(r)
  {}

  virtual void computeError();

  virtual void linearizeOplus();

  // deprecated.
  virtual bool read(std::istream& in) { return false;}
  virtual bool write(std::ostream& out) const {return false; }

  void SetB(double b0) { b0_ = b0; }

 private:
  PointFrameResidual* r_;

  double b0_;
};

/// \class EdgeSE3PosePhotoDSO class.
class EdgeSE3PosePhotoDSO : public ::g2o::BaseBinaryEdge<1, double, VertexSE3PoseDSO, VertexPhotometricDSO> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// \brief The constructor.
  EdgeSE3PosePhotoDSO(Vec3f Xref, Eigen::Vector3f* dINewl, int level, int wl, int hl, double ab_exposure_ref, double ab_exposure_curr, AffLight a0b0, int nl)
      : Xref_(Xref), dINewl_(dINewl), level_(level), wl_(wl), hl_(hl), ab_exposure_ref_(ab_exposure_ref), ab_exposure_curr_(ab_exposure_curr), a0b0_(a0b0), nl_(nl)
  {}

  virtual void computeError();

  virtual void linearizeOplus();

  // deprecated.
  virtual bool read(std::istream& in) { return false;}
  virtual bool write(std::ostream& out) const {return false; }

 private:
  /// \brief 3D point in world coordinate.
  Vec3f Xref_;

  /// \brief Current image.
  Eigen::Vector3f* dINewl_;

  /// \brief Pyramid level.
  int level_;

  /// \brief Image width, height.
  int wl_, hl_;

  /// \brief Exposure time of reference image.
  double ab_exposure_ref_;

  /// \brief Exposure time of current image.
  double ab_exposure_curr_;

  /// \brief Reference image's affine brightness params
  AffLight a0b0_;

  int nl_;
};

/// \class EdgePointActivationIdepthDSO class.
class EdgePointActivationIdepthDSO : public ::g2o::BaseUnaryEdge<1, double, VertexInverseDepthDSO> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// \brief The constructor.
  EdgePointActivationIdepthDSO(double u_pt, double v_pt, Vec2f affLL, const Eigen::Vector3f* dIl, CalibHessian* HCalib, const Mat33f R, const Vec3f t)
      : u_pt_(u_pt), v_pt_(v_pt), affLL_(affLL), dIl_(dIl), HCalib_(HCalib), R_(R), t_(t)
  {}

  virtual void computeError();

  virtual void linearizeOplus();

  void projectPoint();

  // deprecated.
  virtual bool read(std::istream& in) { return false;}
  virtual bool write(std::ostream& out) const {return false; }

 private:
  /// \brief u before projection.
  double u_pt_;

  /// \brief v before projection.
  double v_pt_;

  /// \brief Nomarlized u after projection.
  double u_;

  /// \brief Nomarlized v after projection.
  double v_;

  /// \brief Affine brightness params (a,b)
  Vec2f affLL_;

  /// \brief Image vector per level. (AFAIK [0]: intensity, [1]: dx, [2]: dy)
  const Eigen::Vector3f* dIl_;

  /// \brief Calibration paramteres.
  CalibHessian* HCalib_;

  /// \brief Precomputed rotation matrix.
  const Mat33f R_;

  /// \brief Precomputed translation vector.
  const Vec3f t_;

  /// \brief Current frame's idepth.
  double new_idepth_;

  /// \brief Rescale value of idepth (1/ptp[2])
  double drescale_;

  /// \brief Projected u.
  double Ku_;

  /// \brief Projected v.
  double Kv_;

  Vec3f Klip_;

};

/// \class EdgeTracePointUVDSO class.
class EdgeTracePointUVDSO : public ::g2o::BaseUnaryEdge<1, double, VertexUVDSO> {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  /// \brief The constructor.
  EdgeTracePointUVDSO(Vec2f affLL, const Eigen::Vector3f* dI, double dx, double dy, Vec2f rotatePattern)
      : affLL_(affLL), dI_(dI), dx_(dx), dy_(dy), rotatePattern_(rotatePattern)
  {}

  virtual void computeError();

  virtual void linearizeOplus();

  // deprecated.
  virtual bool read(std::istream& in) { return false;}
  virtual bool write(std::ostream& out) const {return false; }

 private:
  /// \brief Affine brightness params (a,b)
  Vec2f affLL_;

  /// \brief Image vector per level. (AFAIK [0]: intensity, [1]: dx, [2]: dy)
  const Eigen::Vector3f* dI_;

  double dx_;

  double dy_;

  Vec2f rotatePattern_;
};
}  // namespace dso
#endif /* DSO_G2O_EDGE_H_ */
