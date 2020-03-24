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

#include "dso_g2o_vertex.h"
#include "dso_g2o_edge.h"
 
#include "util/NumType.h"
#include "vector"
#include <math.h>
#include "util/settings.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "IOWrapper/Output3DWrapper.h"

#include <g2o/core/solver.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/solvers/cholmod/linear_solver_cholmod.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/sparse_optimizer_terminate_action.h>

namespace dso
{

struct CalibHessian;
struct FrameHessian;
struct PointFrameResidual;

class CoarseTracker {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  CoarseTracker(int w, int h);
  ~CoarseTracker();

  bool trackNewestCoarse(
      FrameHessian* newFrameHessian,
      SE3 &lastToNew_out, AffLight &aff_g2l_out,
      int coarsestLvl, Vec5 minResForAbort,
      IOWrap::Output3DWrapper* wrap=0);

  void setCTRefForFirstFrame(
      std::vector<FrameHessian*> frameHessians);

  void setCoarseTrackingRef(
      std::vector<FrameHessian*> frameHessians, FrameHessian* fh_right, CalibHessian Hcalib);

  void makeCoarseDepthForFirstFrame(FrameHessian* fh);

  void makeK(
      CalibHessian* HCalib);

  bool debugPrint, debugPlot;

  Mat33f K[PYR_LEVELS];
  Mat33f Ki[PYR_LEVELS];
  float fx[PYR_LEVELS];
  float fy[PYR_LEVELS];
  float fxi[PYR_LEVELS];
  float fyi[PYR_LEVELS];
  float cx[PYR_LEVELS];
  float cy[PYR_LEVELS];
  float cxi[PYR_LEVELS];
  float cyi[PYR_LEVELS];
  int w[PYR_LEVELS];
  int h[PYR_LEVELS];

  void debugPlotIDepthMap(float* minID, float* maxID, std::vector<IOWrap::Output3DWrapper*> &wraps);
  void debugPlotIDepthMapFloat(std::vector<IOWrap::Output3DWrapper*> &wraps);

  /// Reference frame
  FrameHessian* lastRef;

  AffLight lastRef_aff_g2l;

  /// New frame
  FrameHessian* newFrame;

  /// Reference frame id
  int refFrameID;

  // act as pure ouptut
  Vec5 lastResiduals;

  /// Optical flow indication, only pan and pan, rotation + pan pixel movement
  Vec3 lastFlowIndicators;
  double firstCoarseRMSE;
 private:

  void makeCoarseDepthL0(std::vector<FrameHessian*> frameHessians, FrameHessian* fh_right, CalibHessian Hcalib);
  float* idepth[PYR_LEVELS];
  float* weightSums[PYR_LEVELS];
  float* weightSums_bak[PYR_LEVELS];


  Vec6 calcResAndGS(int lvl, Mat88 &H_out, Vec8 &b_out, SE3 refToNew, AffLight aff_g2l, float cutoffTH);
  Vec6 calcRes(int lvl, SE3 refToNew, AffLight aff_g2l, float cutoffTH,
               g2o::SparseOptimizer* optimizer,
               VertexSE3PoseDSO* pose,
               VertexPhotometricDSO* photo);
  void calcGSSSE(int lvl, Mat88 &H_out, Vec8 &b_out, SE3 refToNew, AffLight aff_g2l);
  void calcGS(int lvl, Mat88 &H_out, Vec8 &b_out, SE3 refToNew, AffLight aff_g2l);

  // pc buffers
  /// X coordinate of inverse depth point on each layer
  float* pc_u[PYR_LEVELS];
  /// Coordinate y with inverse depth points on each layer
  float* pc_v[PYR_LEVELS];
  /// Inverse depth of points on each layer
  float* pc_idepth[PYR_LEVELS];
  /// Color value of points on each layer
  float* pc_color[PYR_LEVELS];
  /// Number of points on each layer
  int pc_n[PYR_LEVELS];

  // warped buffers
  /// Inverse depth of the points from the projection
  float* buf_warped_idepth;
  /// Normalized coordinates obtained by projection
  float* buf_warped_u;
  float* buf_warped_v;
  /// Image gradient of projection point
  float* buf_warped_dx;
  float* buf_warped_dy;
  /// Projected residuals
  float* buf_warped_residual;
  /// Projected Huber Function Weights
  float* buf_warped_weight;
  /// Gray value at reference point of projection point
  float* buf_warped_refColor;
  /// Number of projection points
  int buf_warped_n;

  Accumulator9 acc;
};


class CoarseDistanceMap {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  CoarseDistanceMap(int w, int h);
  ~CoarseDistanceMap();

  void makeDistanceMap(
      std::vector<FrameHessian*> frameHessians,
      FrameHessian* frame);

  void makeInlierVotes(
      std::vector<FrameHessian*> frameHessians);

  void makeK( CalibHessian* HCalib);

  /// The value of the distance field
  float* fwdWarpedIDDistFinal;

  Mat33f K[PYR_LEVELS];
  Mat33f Ki[PYR_LEVELS];
  float fx[PYR_LEVELS];
  float fy[PYR_LEVELS];
  float fxi[PYR_LEVELS];
  float fyi[PYR_LEVELS];
  float cx[PYR_LEVELS];
  float cy[PYR_LEVELS];
  float cxi[PYR_LEVELS];
  float cyi[PYR_LEVELS];
  int w[PYR_LEVELS];
  int h[PYR_LEVELS];

  void addIntoDistFinal(int u, int v);


 private:

  PointFrameResidual** coarseProjectionGrid;
  int* coarseProjectionGridNum;

  /// The coordinates projected into the frame
  Eigen::Vector2i* bfsList1;

  /// Use with 1 rotation
  Eigen::Vector2i* bfsList2;

  void growDistBFS(int bfsNum);
};

}

