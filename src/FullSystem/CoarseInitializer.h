/**
 * This file is part of DSO.
 *
 * Copyright 2016 Technical University of Munich and Intel.
 * Developed by Jakob Engel <engelj at in dot tum dot de>,
 * for more information see <http://vision.in.tum.de/dso>.
 * If you use this code, please cite the respective publications as
 * listed on the above website.
 *
 * DSO is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * DSO is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with DSO. If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "util/NumType.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "IOWrapper/Output3DWrapper.h"
#include "util/settings.h"
#include "vector"
#include <math.h>

namespace dso {

struct CalibHessian;
struct FrameHessian;

struct Pnt {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  // index in jacobian. never changes (actually, there is no reason why).
  float u,v;

  // idepth / isgood / energy during optimization.
  /// This point corresponds to the inverse depth of the reference frame
  float idepth;

  /// Point in the new image, in front of the camera, the pixel value is good
  bool isGood;

  /// [0] the square of the residual, [1] the regularization term (the square of the inverse depth minus one)
  Vec2f energy;		// (UenergyPhotometric, energyRegularizer)


  bool isGood_new;

  /// Inverse depth of the point on the new frame (current frame)
  float idepth_new;

  /// New energy for iterative calculations
  Vec2f energy_new;

  /// Expected value of inverse depth
  float iR;

  /// Sum of sub-point inverse depth information matrix
  float iRSumNum;

  /// Hessian with inverse depth, i.e. covariance, dd * dd
  float lastHessian;

  /// Covariance of new iteration
  float lastHessian_new;

  // max stepsize for idepth (corresponding to max. movement in pixel-space).
  /// Maximum step size for inverse depth increase
  float maxstep;

  // idx (x+y*w) of closest point one pyramid level above.
  /// The id of the parent node (closest) at the point in the previous layer
  int parent;
  /// Distance from parent node in previous layer
  float parentDist;

  // idx (x+y*w) of up to 10 nearest points in pixel space.
  /// 10 points closest to the point in the image
  int neighbours[10];

  /// The distance of the nearest 10 points
  float neighboursDist[10];

  /// Layer 0 extraction is 1, 2, 4, corresponding to d, 2d, 4d, other layers are 1
  float my_type;

  /// Outlier threshold
  float outlierTH;
}; // class Pnt

class CoarseInitializer {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  CoarseInitializer(int w, int h);
  ~CoarseInitializer();

  /// \comment(edward): deprecated.
  void setFirst(CalibHessian* HCalib, FrameHessian* newFrameHessian);

  void setFirstStereo(CalibHessian* HCalib, FrameHessian* newFrameHessian, FrameHessian* newFrameHessian_Right);

  bool trackFrame(FrameHessian* newFrameHessian, FrameHessian* newFrameHessian_Right, std::vector<IOWrap::Output3DWrapper*> &wraps);

  void calcTGrads(FrameHessian* newFrameHessian);

  /// Number of frames currently joined
  int frameID;

  /// Whether to optimize photometric parameters
  bool fixAffine;
  bool printDebug;

  /// The point class on each layer is extracted from the first frame
  Pnt* points[PYR_LEVELS];

  /// Number of points per layer
  int numPoints[PYR_LEVELS];

  /// Photometric coefficient between reference frame and current frame
  AffLight thisToNext_aff;

  /// Pose between reference frame and current frame
  SE3 thisToNext;

  FrameHessian* firstFrame;

  /// Newly added frames in the track
  FrameHessian* newFrame;

  FrameHessian* firstRightFrame;

 private:
  /// camera parameter
  Mat33 K[PYR_LEVELS];
  Mat33 Ki[PYR_LEVELS];
  double fx[PYR_LEVELS];
  double fy[PYR_LEVELS];
  double fxi[PYR_LEVELS];
  double fyi[PYR_LEVELS];
  double cx[PYR_LEVELS];
  double cy[PYR_LEVELS];
  double cxi[PYR_LEVELS];
  double cyi[PYR_LEVELS];
  int w[PYR_LEVELS];
  int h[PYR_LEVELS];

  void makeK(CalibHessian* HCalib);

  float* idepth[PYR_LEVELS];

  /// Whether to scale convergence (tentative)
  bool snapped;

  /// The scale converges on the first few frames
  int snappedAt;

  // pyramid images & levels on all levels
  Eigen::Vector3f* dINew[PYR_LEVELS];
  Eigen::Vector3f* dIFist[PYR_LEVELS];

  Eigen::DiagonalMatrix<float, 8> wM;

  // temporary buffers for H and b.
  /// Used to calculate Schur
  Vec10f* JbBuffer;			// 0-7: sum(dd * dp). 8: sum(res*dd). 9: 1/(1+sum(dd*dd))=inverse hessian entry.

  /// New value after waiting for update
  Vec10f* JbBuffer_new;

  /// 9-dimensional vector, product to get 9 * 9 matrix, and make the accumulator
  /// Hessian matrix
  Accumulator9 acc9;
  /// Schur part Hessian
  Accumulator9 acc9SC;

  Vec3f dGrads[PYR_LEVELS];

  /// These parameters are fascinating
  float alphaK;          /// 2.5*2.5
  float alphaW;          /// 150*150
  float regWeight;       /// Weighted inverse depth, 0.8
  float couplingWeight;  /// 1

  Vec3f calcResAndGS(
      int lvl,
      Mat88f &H_out, Vec8f &b_out,
      Mat88f &H_out_sc, Vec8f &b_out_sc,
      SE3 refToNew, AffLight refToNew_aff,
      bool plot);

  Vec3f calcEC(int lvl); // returns OLD ENERGY, NEW ENERGY, NUM TERMS.
  void optReg(int lvl);

  void propagateUp(int srcLvl);
  void propagateDown(int srcLvl);
  float rescale();

  void resetPoints(int lvl);
  void doStep(int lvl, float lambda, Vec8f inc);
  void applyStep(int lvl);

  void makeGradients(Eigen::Vector3f** data);

  void debugPlot(int lvl, std::vector<IOWrap::Output3DWrapper*> &wraps);
  void makeNN();
}; // class CoarseInitializer

/// As the second template parameter of the KDTreeSingleIndexAdaptor class must be given, including the following interface
struct FLANNPointcloud {
  inline FLANNPointcloud() {num=0; points=0;}
  inline FLANNPointcloud(int n, Pnt* p) :  num(n), points(p) {}
  int num;
  Pnt* points;

  /// Returns the number of data points
  inline size_t kdtree_get_point_count() const { return num; }

  /// Used when using the L2 metric, returns the vector p1, the Euclidean distance to the idx_p2 data point
  inline float kdtree_distance(const float *p1, const size_t idx_p2,size_t /*size*/) const
  {
    const float d0=p1[0]-points[idx_p2].u;
    const float d1=p1[1]-points[idx_p2].v;
    return d0*d0+d1*d1;
  }

  /// Returns the dim dimension data of the idx point
  inline float kdtree_get_pt(const size_t idx, int dim) const
  {
    if (dim==0) return points[idx].u;
    else return points[idx].v;
  }

  /// Optional calculation bounding box
  /// false means default
  /// true this function should return bb
  template <class BBOX>
  bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }
}; // class FLANNPointCloud

} // namespace dso
