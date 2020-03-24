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
#include "util/IndexThreadReduce.h"
#include "vector"
#include <math.h>
#include "map"

namespace dso {

class PointFrameResidual;
class CalibHessian;
class FrameHessian;
class PointHessian;

class EFResidual;
class EFPoint;
class EFFrame;
class EnergyFunctional;
class AccumulatedTopHessian;
class AccumulatedTopHessianSSE;
class AccumulatedSCHessian;
class AccumulatedSCHessianSSE;

extern bool EFAdjointsValid;
extern bool EFIndicesValid;
extern bool EFDeltaValid;

class EnergyFunctional {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
  friend class EFFrame;
  friend class EFPoint;
  friend class EFResidual;
  friend class AccumulatedTopHessian;
  friend class AccumulatedTopHessianSSE;
  friend class AccumulatedSCHessian;
  friend class AccumulatedSCHessianSSE;

  EnergyFunctional();
  ~EnergyFunctional();

  EFResidual* insertResidual(PointFrameResidual* r);
  EFFrame* insertFrame(FrameHessian* fh, CalibHessian* Hcalib);
  EFPoint* insertPoint(PointHessian* ph);

  void dropResidual(EFResidual* r);
  void marginalizeFrame(EFFrame* fh);
  void removePoint(EFPoint* ph);

  void marginalizePointsF();
  void dropPointsF();
  void solveSystemF(int iteration, double lambda, CalibHessian* HCalib);
  double calcMEnergyF();
  double calcLEnergyF_MT();

  void makeIDX();

  void setDeltaF(CalibHessian* HCalib);

  void setAdjointsF(CalibHessian* Hcalib);

  std::vector<EFFrame*> frames;       /// Frames in the energy function
  int nPoints, nFrames, nResiduals;   /// Number of EFPoint, number of EFframe key frames, number of residuals

  MatXX HM;   /// Optimized Hessian matrix with marginalized inverse depth
  VecX bM;    /// Optimized Jr term, marginalized inverse depth

  int resInA, resInL, resInM;  /// The number of residuals in calculating A, L, marginalized H, and b, respectively
  MatXX lastHS;
  VecX lastbS;
  VecX lastX;
  std::vector<VecX> lastNullspaces_forLogging;
  std::vector<VecX> lastNullspaces_pose;
  std::vector<VecX> lastNullspaces_scale;
  std::vector<VecX> lastNullspaces_affA;
  std::vector<VecX> lastNullspaces_affB;

  IndexThreadReduce<Vec10>* red;

  /// History ID
  /// 64-bit alignment
  /// The connection relationship between keyframes,
  /// first: the first 32 represents the host ID, and the last 32 bits represents the target ID;
  /// second: the number [0] ordinary, [1] marginalized
  std::map<long,Eigen::Vector2i> connectivityMap;

 private:

  VecX getStitchedDeltaF() const;

  void resubstituteF_MT(VecX x, CalibHessian* HCalib, bool MT);
  void resubstituteFPt(const VecCf &xc, Mat18f* xAd, int min, int max, Vec10* stats, int tid);

  void accumulateAF_MT(MatXX &H, VecX &b, bool MT);
  void accumulateLF_MT(MatXX &H, VecX &b, bool MT);
  void accumulateSCF_MT(MatXX &H, VecX &b, bool MT);

  void calcLEnergyPt(int min, int max, Vec10* stats, int tid);

  void orthogonalize(VecX* b, MatXX* H);
  Mat18f* adHTdeltaF;  /// Pose increment between host and target, total number of frames × number of frames

  //// arbitrary public
 public:
  Mat88* adHost;       /// Adjoint matrix, double
  Mat88* adTarget;

  Mat88f* adHostF;     /// Adjoint matrix, float
  Mat88f* adTargetF;

 private:
  VecC cPrior;        /// <setting_initialCalibHessian information matrix
  VecCf cDeltaF;      /// Camera Increment
  VecCf cPriorF;      /// float type of cPrior

  AccumulatedTopHessianSSE* accSSE_top_L;
  AccumulatedTopHessianSSE* accSSE_top_A;

  AccumulatedSCHessianSSE* accSSE_bot;

  std::vector<EFPoint*> allPoints;
  std::vector<EFPoint*> allPointsToMarg;

  float currentLambda;
};

} // namespace dso
