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
#include "vector"
#include <math.h>
#include "OptimizationBackend/RawResidualJacobian.h"

namespace dso {

class PointFrameResidual;
class CalibHessian;
class FrameHessian;
class PointHessian;

class EFResidual;
class EFPoint;
class EFFrame;
class EnergyFunctional;

class EFResidual {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  inline EFResidual(PointFrameResidual* org, EFPoint* point_, EFFrame* host_, EFFrame* target_)
      : data(org), point(point_), host(host_), target(target_)
  {
    isLinearized=false;
    isActiveAndIsGoodNEW=false;

    J = new RawResidualJacobian();

    assert(((long)this)%16==0);
    assert(((long)J)%16==0);
  }

  inline ~EFResidual() {
    delete J;
  }

  void takeDataF();

  void fixLinearizationF(EnergyFunctional* ef);

  // structural pointers
  PointFrameResidual* data;
  int hostIDX, targetIDX;       /// Host and Target ID numbers for the residuals

  EFPoint* point;               /// Residual point
  EFFrame* host;
  EFFrame* target;

  int idxInAll;                 /// id in all residuals

  RawResidualJacobian* J;       /// Used to calculate jacob, res value

  EIGEN_ALIGN16 VecNRf res_toZeroF;  /// Linear residual after updating delta
  EIGEN_ALIGN16 Vec8f JpJdF;         /// Inverse Depth Jaco and Pose + Photometric Jaco's Hessian

  // status.
  bool isLinearized;   /// Completion of res_toZeroF

  // if residual is not OOB & not OUTLIER & should be used during accumulations
  /// Activated can also participate in optimization
  bool isActiveAndIsGoodNEW;

  /// Whether it is activated depends on the residual state
  inline const bool &isActive() const {
    return isActiveAndIsGoodNEW;
  }
};

enum EFPointStatus {PS_GOOD=0, PS_MARGINALIZE, PS_DROP};

class EFPoint {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  EFPoint(PointHessian* d, EFFrame* host_) : data(d),host(host_)
  {
    takeData();
    stateFlag=EFPointStatus::PS_GOOD;
  }
  void takeData();

  /// PointHessian data
  PointHessian* data;

  /// Inverse matrix depth prior information, there are after initialization
  float priorF;

  /// The difference between the current idepth and linearization, without using FEJ, is 0
  float deltaF;

  // constant info (never changes in-between).
  /// Current point id in EFFrame
  int idxInPoints;

  //// host frame.
  EFFrame* host;

  // contains all residuals.
  /// All residuals at that point
  std::vector<EFResidual*> residualsAll;

  /// Current residual + marginalized prior residual
  float bdSumF;

  /// inverse hessian of idepth (= covariance of idepth) (= 1/H)
  float HdiF;

  //// LF====================
  //// Hessian of marginalized idepth.
  float Hdd_accLF;

  //// Hessian of marginalized idepth and intrinsic parameters.
  VecCf Hcd_accLF;

  //// Marginalized J_didepth * residual
  float bd_accLF;

  //// AF====================
  //// Hessian of normal idepth
  float Hdd_accAF;

  //// Hessian of normal idepth and intrinsic parameters.
  VecCf Hcd_accAF;

  //// Normal J_didepth * residual
  float bd_accAF;

  /// Status of points
  EFPointStatus stateFlag;
};

class EFFrame {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  EFFrame(FrameHessian* d) : data(d) {
    takeData();
  }
  void takeData();

  /// Pose 0-5, photometric params ab 6-7
  Vec8 prior;			/// Only the first frame has a priori pose, prior hessian (diagonal)
  Vec8 delta_prior;		/// Increase relative to prior, = state - state_prior (E_prior = (delta_prior)' * diag(prior) * (delta_prior)
  Vec8 delta;			/// Photometric increments relative to linearized point pose, state - state_zero.

  /// All points on the frame
  std::vector<EFPoint*> points;

  /// Corresponding FrameHessian data
  FrameHessian* data;

  int idx;	/// Frame id in energy function, idx in frames.

  /// All historical frame IDs
  int frameID;
};

} // namespace dso
