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

 
#include "util/globalCalib.h"
#include "vector"
 
#include "util/NumType.h"
#include <iostream>
#include <fstream>
#include "util/globalFuncs.h"
#include "OptimizationBackend/RawResidualJacobian.h"

namespace dso
{
class PointHessian;
class FrameHessian;
class CalibHessian;

class EFResidual;


enum ResLocation {ACTIVE=0, LINEARIZED, MARGINALIZED, NONE};

/// IN is inside, OOB point is beyond the image, OUTLIER is outside point
enum ResState {IN=0, OOB, OUTLIER};

struct FullJacRowT
{
  Eigen::Vector2f projectedTo[MAX_RES_PER_POINT];
};

class PointFrameResidual
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  EFResidual* efResidual;

  static int instanceCounter;

  /// Last residual status
  ResState state_state;

  /// Last energy value
  double state_energy;

  /// New calculation status
  ResState state_NewState;

  /// New energy, if it is greater than the threshold, it is equal to the threshold
  double state_NewEnergy;

  /// Energy that may have outer points, may be greater than a threshold
  double state_NewEnergyWithOutlier;


  void setState(ResState s) {state_state = s;}


  PointHessian* point;
  FrameHessian* host;
  FrameHessian* target;

  /// Various Jacobian ratios of residuals to variables
  RawResidualJacobian* J;


  bool isNew;

  /// Projection coordinates of each patch
  Eigen::Vector2f projectedTo[MAX_RES_PER_POINT];

  /// Patch center point projection [pixel x, pixel y, new frame inverse depth]
  Vec3f centerProjectedTo;

  ~PointFrameResidual();
  PointFrameResidual();
  PointFrameResidual(PointHessian* point_, FrameHessian* host_, FrameHessian* target_);
  double linearize(CalibHessian* HCalib);


  void resetOOB()
  {
    state_NewEnergy = state_energy = 0;
    state_NewState = ResState::OUTLIER;

    setState(ResState::IN);
  };
  void applyRes( bool copyJacobians);

  void debugPlot();

  void printRows(std::vector<VecX> &v, VecX &r, int nFrames, int nPoints, int M, int res);
};
}

