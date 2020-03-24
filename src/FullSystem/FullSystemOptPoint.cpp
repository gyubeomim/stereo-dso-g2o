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


/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "dso_g2o_vertex.h"
#include "dso_g2o_edge.h"

#include "FullSystem/FullSystem.h"
 
#include "stdio.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <algorithm>
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"

#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include "FullSystem/ImmaturePoint.h"
#include "math.h"

namespace dso {

/// optimize the immature point idepth, and create it as PointHessian
PointHessian* FullSystem::optimizeImmaturePoint(ImmaturePoint* point, int minObs,
                                                ImmaturePointTemporaryResidual* residuals)
{
  auto linear_solver = g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
  auto block_solver = g2o::make_unique<g2o::BlockSolverX>(std::move(linear_solver));

  // auto algorithm = new g2o::OptimizationAlgorithmGaussNewton(std::move(block_solver));
  auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));
  algorithm->setUserLambdaInit(0.1);

  g2o::SparseOptimizer* optimizer = new g2o::SparseOptimizer();
  optimizer->setAlgorithm(algorithm);

  // Set the terminate action (eps = 1e-3)
  g2o::SparseOptimizerTerminateAction* terminateAction = new g2o::SparseOptimizerTerminateAction;
  terminateAction->setGainThreshold(g2o::cst(1e-3));
  optimizer->addPostIterationAction(terminateAction);

  /// STEP1: initialization and res of other keyframes (points are projected on other keyframes)
  int nres = 0;
  for(FrameHessian* fh : frameHessians) {
    /// not created and own
    if(fh != point->host) {
      residuals[nres].state_NewEnergy = residuals[nres].state_energy = 0;
      residuals[nres].state_NewState = ResState::OUTLIER;
      residuals[nres].state_state = ResState::IN;
      residuals[nres].target = fh;
      nres++;   /// number of observations.
    }
  }
  assert(nres == ((int)frameHessians.size())-1);

  bool print = false;//rand()%50==0;

  float lastEnergy = 0;
  float lastHdd=0;
  float lastbd=0;
  float currentIdepth=(point->idepth_max + point->idepth_min)*0.5f;

  //// add idepth vertex.
  VertexInverseDepthDSO* vtx_idepth = new VertexInverseDepthDSO();
  vtx_idepth->setEstimate(currentIdepth);
  vtx_idepth->setId(0);
  optimizer->addVertex(vtx_idepth);

  /// STEP2: use LM (GN)-like method to optimize idepth instead of using triangulation
  /// TODO this optimization method, and the method of triangulation, which is better
  for(int i=0;i<nres;i++) {
    lastEnergy += point->linearizeResidual(&Hcalib, 1000, residuals+i,lastHdd, lastbd, currentIdepth, optimizer, vtx_idepth);
    residuals[i].state_state = residuals[i].state_NewState;
    residuals[i].state_energy = residuals[i].state_NewEnergy;
  }

  // if(!std::isfinite(lastEnergy) || lastHdd < setting_minIdepthH_act)
  // {
  //   if(print)
  //     printf("OptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n",
  //            nres, lastHdd, lastEnergy);
  //   return 0;
  // }

  if(print) printf("Activate point. %d residuals. H=%f. Initial Energy: %f. Initial Id=%f\n" ,
                   nres, lastHdd,lastEnergy,currentIdepth);

  // float lambda = 0.1;
  //// iterative optimization

  optimizer->initializeOptimization();
  optimizer->setVerbose(false);
  // std::cout << "[*] optimizing immaturepoints idepth... " << std::endl;
  optimizer->optimize(setting_GNItsOnPointActivation);

  for(int iteration=0;iteration<setting_GNItsOnPointActivation;iteration++) {
    // float H = lastHdd;
    // H *= 1+lambda;
    // float step = (1.0/H) * lastbd;
    // float newIdepth = currentIdepth - step;

    // float newHdd=0; float newbd=0; float newEnergy=0;
    // for(int i=0;i<nres;i++)
    //   newEnergy += point->linearizeResidual(&Hcalib, 1, residuals+i,newHdd, newbd, newIdepth);

    // if(!std::isfinite(lastEnergy) || newHdd < setting_minIdepthH_act)
    // {
    //   if(print) printf("OptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n",
    //                    nres,
    //                    newHdd,
    //                    lastEnergy);
    //   return 0;
    // }

    // if(print) printf("%s %d (L %.2f) %s: %f -> %f (idepth %f)!\n",
    //                  (true || newEnergy < lastEnergy) ? "ACCEPT" : "REJECT",
    //                  iteration,
    //                  log10(lambda),
    //                  "",
    //                  lastEnergy, newEnergy, newIdepth);

    // if(newEnergy < lastEnergy)
    // {
    //   currentIdepth = newIdepth;
    //   lastHdd = newHdd;
    //   lastbd = newbd;
    //   lastEnergy = newEnergy;
    //   for(int i=0;i<nres;i++)
    //   {
    //     residuals[i].state_state = residuals[i].state_NewState;
    //     residuals[i].state_energy = residuals[i].state_NewEnergy;
    //   }

    //   lambda *= 0.5;
    // }
    // else
    // {
    //   lambda *= 5;
    // }

    // if(fabsf(step) < 0.0001*currentIdepth)
    //   break;
  }

  currentIdepth = vtx_idepth->estimate();

  if(!std::isfinite(currentIdepth))
  {
    printf("MAJOR ERROR! point idepth is nan after initialization (%f).\n", currentIdepth);
    return (PointHessian*)((long)(-1));		// yeah I'm like 99% sure this is OK on 32bit systems.
  }

  /// count the good number in all observations, and return if it is small.
  int numGoodRes=0;
  for(int i=0;i<nres;i++)
    if(residuals[i].state_state == ResState::IN) {
      numGoodRes++;
    }

  if(numGoodRes < minObs) {
    if(print) {
     printf("OptPoint: OUTLIER!\n");
    }
    return (PointHessian*)((long)(-1));		// yeah I'm like 99% sure this is OK on 32bit systems.
  }

  /// STEP3: create points that can be PointHessian
  PointHessian* p = new PointHessian(point, &Hcalib);

  /// discard infinited points.
  if(!std::isfinite(p->energyTH)) {
    delete p;
    return (PointHessian*)((long)(-1));
  }

  p->lastResiduals[0].first = 0;
  p->lastResiduals[0].second = ResState::OOB;
  p->lastResiduals[1].first = 0;
  p->lastResiduals[1].second = ResState::OOB;
  p->setIdepthZero(currentIdepth);             /// set the idepth at the linearization
  p->setIdepth(currentIdepth);
  p->setPointStatus(PointHessian::ACTIVE);

  /// calculate PointFrameResidual
  for(int i=0;i<nres;i++)
    if(residuals[i].state_state == ResState::IN) {
      PointFrameResidual* r = new PointFrameResidual(p, p->host, residuals[i].target);
      r->state_NewEnergy = r->state_energy = 0;
      r->state_NewState = ResState::OUTLIER;
      r->setState(ResState::IN);
      p->residuals.push_back(r);

      /// and the residual of the latest frame
      if(r->target == frameHessians.back()) {
        p->lastResiduals[0].first = r;
        p->lastResiduals[0].second = ResState::IN;
      }
      /// and the latest frame
      else if(r->target == (frameHessians.size()<2 ? 0 : frameHessians[frameHessians.size()-2]))
      {
        p->lastResiduals[1].first = r;
        p->lastResiduals[1].second = ResState::IN;
      }
    }

  if(print) printf("point activated!\n");

  statistics_numActivatedPoints++;
  return p;
}



}
