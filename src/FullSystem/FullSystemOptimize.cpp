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



#include "FullSystem/FullSystem.h"
#include "dso_g2o_vertex.h"
#include "dso_g2o_edge.h"
#include "dso_util.hpp"
 
#include "stdio.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <algorithm>
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include "FullSystem/ResidualProjections.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include <cmath>

#include <algorithm>

namespace dso {

/// linearize the residuals
/// Parameter: [true is applyRes, and remove bad residuals] [false does not perform fixed linearization]
void FullSystem::linearizeAll_Reductor(bool fixLinearization,
                                       std::vector<PointFrameResidual*>* toRemove,
                                       int min, int max,
                                       Vec10* stats, int tid) {
  for(int k=min; k<max; k++) {
    PointFrameResidual* r = activeResiduals[k];
    (*stats)[0] += r->linearize(&Hcalib); /// linearize to get energy

    /// fixed linearization (executed after optimization)
    if(fixLinearization) {
      /// give the value to efResidual.
      r->applyRes(true);

      /// ths residual is in.
      if(r->efResidual->isActive()) {
        if(r->isNew) {
          PointHessian* p = r->point;
          Vec3f ptp_inf = r->host->targetPrecalc[r->target->idx].PRE_KRKiTll*Vec3f(p->u,p->v, 1);	// projected point assuming infinite depth.
          Vec3f ptp = ptp_inf + r->host->targetPrecalc[r->target->idx].PRE_KtTll*p->idepth_scaled;	// projected point with real depth.
          float relBS = 0.01*((ptp_inf.head<2>()/ptp_inf[2]) - (ptp.head<2>()/ptp[2])).norm();	        // 0.01 = one pixel.

          if(relBS > p->maxRelBaseline) {
            p->maxRelBaseline = relBS;   /// proportional to the baseline length of the point
          }

          p->numGoodResiduals++;
        }
      }
      else {
        /// delete OOB, outlier
        /// remove residulas too large to remove.
        toRemove[tid].push_back(activeResiduals[k]);
      }
    }
  }
}

/// pass the linearization result to the energy function EFResidual
/// copyJacobians [true: update jacobian] [false: not update]
void FullSystem::applyRes_Reductor(bool copyJacobians, int min, int max, Vec10* stats, int tid) {
  for(int k=min; k < max; k++) {
    activeResiduals[k]->applyRes(true);
  }
}

/// calculate the energy threshold of the current latest frame, too metaphysical.
void FullSystem::setNewFrameEnergyTH() {
  // collect all residuals and make decision on TH.
  allResVec.clear();
  allResVec.reserve(activeResiduals.size()*2);
  FrameHessian* newFrame = frameHessians.back();

  for(PointFrameResidual* r : activeResiduals)
    /// residual on new frame.
    if(r->state_NewEnergyWithOutlier >= 0 && r->target == newFrame) {
      allResVec.push_back(r->state_NewEnergyWithOutlier);
    }

  if(allResVec.size()==0) {
    newFrame->frameEnergyTH = 12*12*patternNum;
    return;		// should never happen, but lets make sure.
  }

  /// take the energy of settings_frameEnergy THN as the threshold.
  int nthIdx = setting_frameEnergyTHN*allResVec.size();

  assert(nthIdx < (int)allResVec.size());
  assert(setting_frameEnergyTHN < 1);

  /// sort.
  std::nth_element(allResVec.begin(), allResVec.begin()+nthIdx, allResVec.end());

  /// 70% of the values are less than this value.
  float nthElement = sqrtf(allResVec[nthIdx]);

  /// ? this threshold is set so
  /// first expand, multiply by a robust function ?, then calculate the square to get the threshold.
  newFrame->frameEnergyTH = nthElement*setting_frameEnergyTHFacMedian;
  newFrame->frameEnergyTH = 26.0f*setting_frameEnergyTHConstWeight + newFrame->frameEnergyTH*(1-setting_frameEnergyTHConstWeight);
  newFrame->frameEnergyTH = newFrame->frameEnergyTH*newFrame->frameEnergyTH;
  newFrame->frameEnergyTH *= setting_overallEnergyTHWeight*setting_overallEnergyTHWeight;

  //int good=0,bad=0;
  //for(float f : allResVec) if(f<newFrame->frameEnergyTH) good++; else bad++;
  //printf("EnergyTH: mean %f, median %f, result %f (in %d, out %d)! \n",
  //meanElement, nthElement, sqrtf(newFrame->frameEnergyTH),
  //good, bad);
}

/// linearize the residuals and remove those that are not in the image, and the residuals are large.
Vec3 FullSystem::linearizeAll(bool fixLinearization) {
  double lastEnergyP = 0;
  double lastEnergyR = 0;
  double num = 0;

  std::vector<PointFrameResidual*> toRemove[NUM_THREADS];
  for(int i=0;i<NUM_THREADS;i++) {
    toRemove[i].clear();
  }

  if(multiThreading) {
    /// TODO see the multi-threaded IndexThreadReduce
    treadReduce.reduce(boost::bind(&FullSystem::linearizeAll_Reductor, this, fixLinearization, toRemove, _1, _2, _3, _4), 0, activeResiduals.size(), 0);
    lastEnergyP = treadReduce.stats[0];
  }
  else {
    Vec10 stats;
    linearizeAll_Reductor(fixLinearization, toRemove, 0, activeResiduals.size(), &stats, 0);
    lastEnergyP = stats[0];
  }

  setNewFrameEnergyTH();

  if(fixLinearization) {
    /// linearized before, state_state is update after apply, if there is the same, the state is updated.
    for(PointFrameResidual* r : activeResiduals) {
      PointHessian* ph = r->point;
      if(ph->lastResiduals[0].first == r)
        ph->lastResiduals[0].second = r->state_state;
      else if(ph->lastResiduals[1].first == r)
        ph->lastResiduals[1].second = r->state_state;
    }

    /// residual is created when it is created, and then remove the bad ones.
    int nResRemoved=0;

    for(int i=0;i<NUM_THREADS;i++) {   /// number of thread.
      std::cout << "toRemove size " << toRemove[i].size() << std::endl;
      for(PointFrameResidual* r : toRemove[i]) {
        PointHessian* ph = r->point;

        /// delete bad lastResiduals
        if(ph->lastResiduals[0].first == r)
          ph->lastResiduals[0].first=0;
        else if(ph->lastResiduals[1].first == r)
          ph->lastResiduals[1].first=0;

        for(unsigned int k=0; k<ph->residuals.size();k++)
          if(ph->residuals[k] == r) {
            ef->dropResidual(r->efResidual);
            deleteOut<PointFrameResidual>(ph->residuals,k);
            nResRemoved++;
            break;
          }
      }
    }
    //printf("FINAL LINEARIZATION: removed %d / %d residuals!\n", nResRemoved, (int)activeResiduals.size());
  }

  /// the latter two variables are useless.
  return Vec3(lastEnergyP, lastEnergyR, num);
}

// applies step to linearization point.
/// update each state, and determine if you can stop optimization.
bool FullSystem::doStepFromBackup(float stepfacC,
                                  float stepfacT,
                                  float stepfacR,
                                  float stepfacA,
                                  float stepfacD) {
  //	float meanStepC=0,meanStepP=0,meanStepD=0;
  //	meanStepC += Hcalib.step.norm();

  /// equiv to step size.
  Vec10 pstepfac;
  pstepfac.segment<3>(0).setConstant(stepfacT);
  pstepfac.segment<3>(3).setConstant(stepfacR);
  pstepfac.segment<4>(6).setConstant(stepfacA);

  float sumA=0, sumB=0, sumT=0, sumR=0, sumID=0, numID=0;

  float sumNID=0;

  if(setting_solverMode & SOLVER_MOMENTUM) {
    /// the value of the intrinsic param is updated.
    Hcalib.setValue(Hcalib.value_backup + Hcalib.step);

    for(FrameHessian* fh : frameHessians) {
      Vec10 step = fh->step;
      /// ? why add half
      /// ! this solution is very strange? do not care
      step.head<6>() += 0.5f*(fh->step_backup.head<6>());

      fh->setState(fh->state_backup + step);     /// pose + photometric update
      sumA += step[6]*step[6];                   /// squared  brightness increment a
      sumB += step[7]*step[7];                   ///                               b
      sumT += step.segment<3>(0).squaredNorm();  /// translation increment.
      sumR += step.segment<3>(3).squaredNorm();  /// rotation increpand.

      for(PointHessian* ph : fh->pointHessians) {
        /// ? whya dd half
        float step = ph->step + 0.5f*(ph->step_backup);

        ph->setIdepth(ph->idepth_backup + step);
        sumID += step*step;                   /// squared idepth increment.
        sumNID += fabsf(ph->idepth_backup);   /// idepth sum.
        numID++;

        /// idepth without FEJ
        ph->setIdepthZero(ph->idepth_backup + step);
      }
    }
  }
  else {
    /// update intrinsic params status
    Hcalib.setValue(Hcalib.value_backup + stepfacC*Hcalib.step);

    /// pose, photometric params a,b update.
    for(FrameHessian* fh : frameHessians) {
      fh->setState(fh->state_backup + pstepfac.cwiseProduct(fh->step));
      sumA += fh->step[6]*fh->step[6];
      sumB += fh->step[7]*fh->step[7];
      sumT += fh->step.segment<3>(0).squaredNorm();
      sumR += fh->step.segment<3>(3).squaredNorm();

      /// update point idepth, note that point idepth does not use FEJ.
      for(PointHessian* ph : fh->pointHessians) {
        ph->setIdepth(ph->idepth_backup + stepfacD*ph->step);
        sumID += ph->step*ph->step;
        sumNID += fabsf(ph->idepth_backup);
        numID++;

        ph->setIdepthZero(ph->idepth_backup + stepfacD*ph->step);
      }
    }
  }

  sumA /= frameHessians.size();
  sumB /= frameHessians.size();
  sumR /= frameHessians.size();
  sumT /= frameHessians.size();
  sumID /= numID;
  sumNID /= numID;

  if(!setting_debugout_runquiet)
    printf("STEPS: A %.1f; B %.1f; R %.1f; T %.1f. \t",
           sqrtf(sumA) / (0.0005*setting_thOptIterations),
           sqrtf(sumB) / (0.00005*setting_thOptIterations),
           sqrtf(sumR) / (0.00005*setting_thOptIterations),
           sqrtf(sumT)*sumNID / (0.00005*setting_thOptIterations));

  EFDeltaValid=false;

  /// update relative pose, photometric params (a,b).
  setPrecalcValues();

  return sqrtf(sumA) < 0.0005*setting_thOptIterations &&
        sqrtf(sumB) < 0.00005*setting_thOptIterations &&
        sqrtf(sumR) < 0.00005*setting_thOptIterations &&
        sqrtf(sumT)*sumNID < 0.00005*setting_thOptIterations;

  //	printf("mean steps: %f %f %f!\n",
  //			meanStepC, meanStepP, meanStepD);
}

// sets linearization point.
/// backup frame, point, step and state of internal params
void FullSystem::backupState(bool backupLastStep) {

  if(setting_solverMode & SOLVER_MOMENTUM) {
    if(backupLastStep) {
      Hcalib.step_backup = Hcalib.step;
      Hcalib.value_backup = Hcalib.value;
      for(FrameHessian* fh : frameHessians)
      {
        fh->step_backup = fh->step;
        fh->state_backup = fh->get_state();
        for(PointHessian* ph : fh->pointHessians)
        {
          ph->idepth_backup = ph->idepth;
          ph->step_backup = ph->step;
        }
      }
    }
    /// initialize before iteration
    else {
      Hcalib.step_backup.setZero();
      Hcalib.value_backup = Hcalib.value;
      for(FrameHessian* fh : frameHessians)
      {
        fh->step_backup.setZero();
        fh->state_backup = fh->get_state();
        for(PointHessian* ph : fh->pointHessians)
        {
          ph->idepth_backup = ph->idepth;
          ph->step_backup=0;
        }
      }
    }
  }
  else {
    Hcalib.value_backup = Hcalib.value;
    for(FrameHessian* fh : frameHessians)
    {
      fh->state_backup = fh->get_state();
      for(PointHessian* ph : fh->pointHessians)
        ph->idepth_backup = ph->idepth;
    }
  }
}

// sets linearization point.
/// return to original value.
void FullSystem::loadSateBackup() {
  Hcalib.setValue(Hcalib.value_backup);
  for(FrameHessian* fh : frameHessians) {
    fh->setState(fh->state_backup);

    for(PointHessian* ph : fh->pointHessians) {
      ph->setIdepth(ph->idepth_backup);
      ph->setIdepthZero(ph->idepth_backup);  /// useless FEJ
    }
  }

  EFDeltaValid=false;

  /// update current status.
  setPrecalcValues();
}

/// calculate absolute energy.
double FullSystem::calcMEnergy() {
  if(setting_forceAceptStep) {
    return 0;
  }

  // calculate (x - x0)^T*[2b + H*(x - x0)] for everything saved in L.
  //ef->makeIDX();
  //ef->setDeltaF(&Hcalib);

  return ef->calcMEnergyF();
}

void FullSystem::printOptRes(Vec3 res,
                             double resL,
                             double resM,
                             double resPrior,
                             double LExact,
                             float a, float b)
{
  printf("A(%f)=(AV %.3f). Num: A(%'d) + M(%'d); ab %f %f!\n",
         res[0],
         sqrtf((float)(res[0] / (patternNum*ef->resInA))),
         ef->resInA,
         ef->resInM,
         a,
         b
         );
}

#if 1
/// perfrom GN optimization on the current keyframe.
float FullSystem::optimize(int mnumOptIts) {
  //// customizing
  if(frameHessians.size() < 2) {
    return 0;
  }
  else if(frameHessians.size() < 3) {
    /// number of iterations.
    mnumOptIts = 10;
  }
  else if(frameHessians.size() < 4) {
    mnumOptIts = 7;
  }
  else {
    mnumOptIts = 3;
  }

  auto linear_solver = g2o::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
  auto block_solver = g2o::make_unique<g2o::BlockSolverX>(std::move(linear_solver));
  block_solver->setSchur(true);

  // auto algorithm = new g2o::OptimizationAlgorithmGaussNewton(std::move(block_solver));
  auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(block_solver));
  algorithm->setUserLambdaInit(0.1);
  g2o::SparseOptimizer* optimizer = new g2o::SparseOptimizer();
  optimizer->setAlgorithm(algorithm);

  // Set the terminate action.
  g2o::SparseOptimizerTerminateAction* terminateAction = new g2o::SparseOptimizerTerminateAction;
  terminateAction->setGainThreshold(g2o::cst(1e-3));
  optimizer->addPostIterationAction(terminateAction);

  // get statistics and active residuals.
  /// STEP1: find the residuals that are not linearized (marginalized), add activeResiduals.
  activeResiduals.clear();
  int numPoints = 0;
  int numLRes = 0;

  for(FrameHessian* fh : frameHessians) {
    for(PointHessian* ph : fh->pointHessians) {
      for(PointFrameResidual* r : ph->residuals) {
        if(!r->efResidual->isLinearized) {  /// if no linear error
          activeResiduals.push_back(r);  /// newly added residuals
          r->resetOOB();                 /// residual state reset.
        }
        else
          numLRes++;   /// counted by linearization.
      }
      numPoints++;
    }
  }

  int id=0;
  //// add camera intrinsic vertex.
  VertexCamDSO* vtx_cam = new VertexCamDSO;
  vtx_cam->setEstimate(Vec4(Hcalib.fxl(), Hcalib.fyl(), Hcalib.cxl(), Hcalib.cyl()));
  vtx_cam->setId(id++);
  optimizer->addVertex(vtx_cam);

  Mat33f K;
  K << Hcalib.fxl(), 0,            Hcalib.cxl(),
       0,            Hcalib.fyl(), Hcalib.cyl(),
       0,            0,            1;

  std::map<PointFrameResidual*, VertexSE3PoseDSO*> vpose;
  std::map<PointFrameResidual*, VertexPhotometricDSO*> vphoto;
  std::map<PointFrameResidual*, VertexInverseDepthDSO*> videpth;

  std::vector<VertexSE3PoseDSO*> v_vtx_pose;
  std::vector<VertexPhotometricDSO*> v_vtx_photo;
  v_vtx_pose.resize(8);
  v_vtx_photo.resize(8);

  bool vtxused[8] = {false};

  for(PointFrameResidual* r : activeResiduals) {
    if(vtxused[r->host->idx] == false) {
      vtxused[r->host->idx] = true;

      //// add pose vertex.
      VertexSE3PoseDSO* vtx_pose = new VertexSE3PoseDSO();
      //// Twh
      vtx_pose->setEstimate(r->host->PRE_camToWorld);
      std::cout << "before\n" << vtx_pose->estimate().matrix() << std::endl;
      vtx_pose->setId(id++);
      optimizer->addVertex(vtx_pose);

      AffLight a0b0 = r->host->aff_g2l();

      //// add photometric vertex.
      VertexPhotometricDSO* vtx_photo = new VertexPhotometricDSO();
      vtx_photo->setEstimate(a0b0);
      vtx_photo->setId(id++);
      optimizer->addVertex(vtx_photo);

      v_vtx_pose[r->host->idx] = vtx_pose;
      v_vtx_photo[r->host->idx] = vtx_photo;

      vpose[r] = vtx_pose;
      vphoto[r] = vtx_photo;
    }

    //// add idepth vertex.
    VertexInverseDepthDSO* vtx_idepth = new VertexInverseDepthDSO();
    vtx_idepth->setEstimate(r->point->idepth);
    vtx_idepth->setId(id++);
    vtx_idepth->setMarginalized(true);
    optimizer->addVertex(vtx_idepth);

    videpth[r] = vtx_idepth;

    const float* color = r->point->color;

    EdgeLBASE3PosePhotoIdepthCamDSO* edge = new EdgeLBASE3PosePhotoIdepthCamDSO(r);

    edge->resize(4);
    edge->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_vtx_pose[r->host->idx]));
 edge->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(v_vtx_photo[r->host->idx]));
    edge->setVertex(2, dynamic_cast<g2o::OptimizableGraph::Vertex*>(vtx_idepth));
    edge->setVertex(3, dynamic_cast<g2o::OptimizableGraph::Vertex*>(vtx_cam));

    Vec8f colors;
    colors << *(color), *(color+1), *(color+2), *(color+3), *(color+4), *(color+5), *(color+6), *(color+7);

    edge->setMeasurement(colors);
    edge->setInformation(Eigen::Matrix<double,8,8>::Identity());
    g2o::RobustKernelHuber* huber = new g2o::RobustKernelHuber;
    huber->setDelta(setting_huberTH);
    edge->setRobustKernel(huber);
    edge->setId(id++);
    edge->setLevel(0);

    AffLight a0b0 = v_vtx_photo[r->host->idx]->estimate();
    edge->SetB(a0b0.b);

    //// project point from host frame to target frame.
    edge->computeError();

    optimizer->addEdge(edge);
  }

  std::cout << "activeResiduals size: " << activeResiduals.size() << std::endl;
  std::cout << "vertex size: " << optimizer->vertices().size() << std::endl;
  std::cout << "edge size: " << optimizer->edges().size() << std::endl;

  //// arbitrarily added.
  if(multiThreading)
    treadReduce.reduce(boost::bind(&FullSystem::applyRes_Reductor, this, true, _1, _2, _3, _4), 0, activeResiduals.size(), 50);
  else
    applyRes_Reductor(true,0,activeResiduals.size(),0,0);

  if(!setting_debugout_runquiet)
    printf("OPTIMIZE %d pts, %d active res, %d lin res!\n",ef->nPoints,(int)activeResiduals.size(), numLRes);

#if 0
  /// STEP2: linearize the residuals of activeResiduals and calculate the marginalized energy value (however, it is set to 0 here)
  /// linearization, parameters: [true is fixed linearization and removes bad residuals] [false does not perform fixed linearization]
  Vec3 lastEnergy = linearizeAll(false);

  /// what's the difference b/w linearizAll and those?
  double lastEnergyL = calcLEnergy();  /// islinearized amount of energy.
  double lastEnergyM = calcMEnergy();  /// energy of HM part.

  /// give the linearized result top EFResidual.
  if(multiThreading)
    treadReduce.reduce(boost::bind(&FullSystem::applyRes_Reductor, this, true, _1, _2, _3, _4), 0, activeResiduals.size(), 50);
  else
    applyRes_Reductor(true,0,activeResiduals.size(),0,0);

  if(!setting_debugout_runquiet) {
    printf("Initial Error       \t");
    printOptRes(lastEnergy, lastEnergyL, lastEnergyM, 0, 0, frameHessians.back()->aff_g2l().a, frameHessians.back()->aff_g2l().b);
  }

  debugPlotTracking();

  double lambda = 1e-1;
  float stepsize=1;
  VecX previousX = VecX::Constant(CPARS+ 8*frameHessians.size(), NAN);
#endif

  debugPlotTracking();

  /// STEP3: iterative solution
  optimizer->initializeOptimization();
  optimizer->setVerbose(true);
  std::cout << "[*] window optimizing " << mnumOptIts << " times..." << std::endl;
  optimizer->optimize(mnumOptIts);

#if 0
  for(int iteration=0; iteration < mnumOptIts; iteration++) {
    /// STEP3.1: back up the current state.
    backupState(iteration!=0);

    /// STEP3.2: solve the solution.
    // solve!
    solveSystem(iteration, lambda);

    double incDirChange = (1e-20 + previousX.dot(ef->lastX)) / (1e-20 + previousX.norm()*ef->lastX.norm());

    previousX = ef->lastX;

    /// ? TUM's own solution?
    if(std::isfinite(incDirChange) && (setting_solverMode & SOLVER_STEPMOMENTUM)) {
      float newStepsize = exp(incDirChange*1.4);

      if(incDirChange<0 && stepsize>1) {
        stepsize=1;
      }

      stepsize = sqrtf(sqrtf(newStepsize*stepsize*stepsize*stepsize));
      if(stepsize > 2) {
        stepsize=2;
      }
      if(stepsize <0.25) {
        stepsize=0.25;
      }
    }

    /// STEP3.3: update status.
    /// update variables to determine whether to stop.
    bool canbreak = doStepFromBackup(stepsize, stepsize, stepsize, stepsize, stepsize);

    /// eval new energy!
    /// re-calculate after update
    Vec3 newEnergy = linearizeAll(false);
    double newEnergyL = calcLEnergy();
    double newEnergyM = calcMEnergy();

    if(!setting_debugout_runquiet) {
      printf("%s %d (L %.2f, dir %.2f, ss %.1f): \t",
             (newEnergy[0] + newEnergy[1] + newEnergyL + newEnergyM <
              lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM) ? "ACCEPT" : "REJECT",
             iteration,
             log10(lambda),
             incDirChange,
             stepsize);
      printOptRes(newEnergy, newEnergyL, newEnergyM , 0, 0, frameHessians.back()->aff_g2l().a, frameHessians.back()->aff_g2l().b);
    }

    /// STEP4: determine whether to accept this calcuation.
    if(setting_forceAceptStep || (newEnergy[0] + newEnergy[1] + newEnergyL + newEnergyM <
                                  lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM))
    {
      /// accept the updated amount.
      if(multiThreading)
        treadReduce.reduce(boost::bind(&FullSystem::applyRes_Reductor, this, true, _1, _2, _3, _4), 0, activeResiduals.size(), 50);
      else
        applyRes_Reductor(true,0,activeResiduals.size(),0,0);

      lastEnergy = newEnergy;
      lastEnergyL = newEnergyL;
      lastEnergyM = newEnergyM;

      /// fixed lambda.
      lambda *= 0.25;
    }
    else {
      /// not accepted, roll back.
      loadSateBackup();
      lastEnergy = linearizeAll(false);
      lastEnergyL = calcLEnergy();
      lastEnergyM = calcMEnergy();
      lastEnergyL = 0;
      lastEnergyM = 0;
      lambda *= 1e2;
    }

    if(canbreak && iteration >= setting_minOptIterations) {
      break;
    }
  }
#endif

  backupState(true);

#if 1
  std::cout << std::endl << "vtx_cam: " << vtx_cam->estimate().transpose() << std::endl << std::endl;
  //// update estimates after optimization.
  Vec4 update = vtx_cam->estimate();

  // [0-3: Kl, 4-7: Kr, 8-12: l2r]
  Hcalib.value = update;
  Hcalib.value_scaled[0] = 1 * Hcalib.value[0];
  Hcalib.value_scaled[1] = 1 * Hcalib.value[1];
  Hcalib.value_scaled[2] = 1 * Hcalib.value[2];
  Hcalib.value_scaled[3] = 1 * Hcalib.value[3];

  Hcalib.value_scaledf = Hcalib.value_scaled.cast<float>();
  Hcalib.value_scaledi[0] = 1.0f / Hcalib.value_scaledf[0];
  Hcalib.value_scaledi[1] = 1.0f / Hcalib.value_scaledf[1];
  Hcalib.value_scaledi[2] = - Hcalib.value_scaledf[2] / Hcalib.value_scaledf[0];
  Hcalib.value_scaledi[3] = - Hcalib.value_scaledf[3] / Hcalib.value_scaledf[1];
  Hcalib.value_minus_value_zero = Hcalib.value - Hcalib.value_zero;
#endif

  bool vtxused2[8] = {false};
  for(PointFrameResidual* r : activeResiduals) {
    boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

    if(vtxused2[r->host->idx] == false) {
      vtxused2[r->host->idx] = true;

      VertexSE3PoseDSO* vtx_pose = vpose[r];
      VertexPhotometricDSO* vtx_photo = vphoto[r];

      //// Twh
      r->host->PRE_camToWorld = vtx_pose->estimate();
      std::cout << "after\n" << vtx_pose->estimate().matrix() << std::endl;
      //// Thw
      r->host->PRE_worldToCam = r->host->PRE_camToWorld.inverse();

      r->host->state[6] = vtx_photo->estimate().a;
      r->host->state[7] = vtx_photo->estimate().b;
      //// custom scaling
      r->host->state_scaled[6] = 1 * r->host->state[6];
      r->host->state_scaled[7] = 1 * r->host->state[7];
    }

    VertexInverseDepthDSO* vtx_idepth = videpth[r];
    r->centerProjectedTo = vtx_idepth->GetCenterProjectedTo();
    r->point->setIdepth(vtx_idepth->estimate());
    r->point->setIdepthZero(vtx_idepth->estimate());
    r->efResidual->point->HdiF = 1.0 / r->point->idepth_hessian;
  }

  //// arbitrarily added.
  if(multiThreading)
    treadReduce.reduce(boost::bind(&FullSystem::applyRes_Reductor, this, true, _1, _2, _3, _4), 0, activeResiduals.size(), 50);
  else
    applyRes_Reductor(true,0,activeResiduals.size(),0,0);


  /// STEP5: set the pose of the latest frame to the linearization point.
  /// the pose of the latest frame is set to the linearization point (x=0),
  /// 0-5 is the pose increment and therefore 0
  /// 6-7 is the value, directly assigned.
  Vec10 newStateZero = Vec10::Zero();
  newStateZero.segment<2>(6) = frameHessians.back()->get_state().segment<2>(6);
  /// the latest frame is set to the linearization point, to be estimated.
  frameHessians.back()->setEvalPT(frameHessians.back()->PRE_worldToCam, newStateZero);

  EFDeltaValid=false;
  EFAdjointsValid=false;

  /// re-calculated adjoint. (Hcalib is not used.)
  ef->setAdjointsF(&Hcalib);

  /// update increment.
  setPrecalcValues();

  // if(fixLinearization) {
  {
    setNewFrameEnergyTH();

    std::vector<PointFrameResidual*> toRemove;
    toRemove.clear();

    for(int k=0; k<activeResiduals.size(); k++) {
      PointFrameResidual* r = activeResiduals[k];
      /// give the value to efResidual.
      r->applyRes(true);

      /// ths residual is in.
      if(r->efResidual->isActive()) {
        if(r->isNew) {
          PointHessian* p = r->point;
          Vec3f ptp_inf = r->host->targetPrecalc[r->target->idx].PRE_KRKiTll*Vec3f(p->u,p->v, 1);	// projected point assuming infinite depth.
          Vec3f ptp = ptp_inf + r->host->targetPrecalc[r->target->idx].PRE_KtTll*p->idepth_scaled;	// projected point with real depth.
          float relBS = 0.01*((ptp_inf.head<2>()/ptp_inf[2]) - (ptp.head<2>()/ptp[2])).norm();	        // 0.01 = one pixel.

          if(relBS > p->maxRelBaseline) {
            p->maxRelBaseline = relBS;   /// proportional to the baseline length of the point
          }

          p->numGoodResiduals++;
        }
      }
      else {
        /// delete OOB, outlier
        /// remove residulas too large to remove.
        toRemove.push_back(activeResiduals[k]);
      }
    }

    /// linearized before, state_state is update after apply, if there is the same, the state is updated.
    for(PointFrameResidual* r : activeResiduals) {
      PointHessian* ph = r->point;
      if(ph->lastResiduals[0].first == r)
        ph->lastResiduals[0].second = r->state_state;
      else if(ph->lastResiduals[1].first == r)
        ph->lastResiduals[1].second = r->state_state;
    }

    /// residual is created when it is created, and then remove the bad ones.
    int nResRemoved=0;

     std::cout << "activeResiduals size " << activeResiduals.size() << std::endl;
     std::cout << "toRemove size " << toRemove.size() << std::endl;

    for(PointFrameResidual* r : toRemove) {
      PointHessian* ph = r->point;

      /// delete bad lastResiduals
      if(ph->lastResiduals[0].first == r)
        ph->lastResiduals[0].first=0;
      else if(ph->lastResiduals[1].first == r)
        ph->lastResiduals[1].first=0;

      for(unsigned int k=0; k<ph->residuals.size();k++)
        if(ph->residuals[k] == r) {
          ef->dropResidual(r->efResidual);
          deleteOut<PointFrameResidual>(ph->residuals,k);
          nResRemoved++;
          break;
        }
    }
  }

#if 0
  /// energy after update.
  lastEnergy = linearizeAll(true);

  /// if the energy function is too large, the projection is not good, and the loss.
  if(!std::isfinite((double)lastEnergy[0]) ||
     !std::isfinite((double)lastEnergy[1]) ||
     !std::isfinite((double)lastEnergy[2]))
  {
    printf("KF Tracking failed: LOST!\n");
    isLost=true;
  }


  statistics_lastFineTrackRMSE = sqrtf((float)(lastEnergy[0] / (patternNum * ef->resInA)));

  if(calibLog != 0) {
    (*calibLog) << Hcalib.value_scaled.transpose() <<
        " " << frameHessians.back()->get_state_scaled().transpose() <<
        " " << sqrtf((float)(lastEnergy[0] / (patternNum * ef->resInA))) <<
        " " << ef->resInM << "\n";
    calibLog->flush();
  }
#endif

  /// STEP6: give the optimized result to the shell of each frame.
  /// Note that the linear points of ther frames are not updated here.
  /// give the optimization result to the shell.
  {
    boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
    for(FrameHessian* fh : frameHessians) {
      fh->shell->camToWorld = fh->PRE_camToWorld;
#if 1
      fh->shell->aff_g2l = fh->aff_g2l();
      std::cout << "fh->shell->aff_g2l: " << fh->shell->aff_g2l.a << ", " << fh->shell->aff_g2l.b << std::endl;
#else
      fh->shell->aff_g2l = AffLight();
#endif
    }
  }

  debugPlotTracking();

  /// return average error rmse.
  // return sqrtf((float)(lastEnergy[0] / (patternNum*ef->resInA)));
  return sqrtf(optimizer->activeRobustChi2() / (patternNum * optimizer->edges().size()));
}

#else
float FullSystem::optimize(int mnumOptIts)
{

  if(frameHessians.size() < 2) return 0;
  if(frameHessians.size() < 3) mnumOptIts = 20;
  if(frameHessians.size() < 4) mnumOptIts = 15;

  // get statistics and active residuals.

  activeResiduals.clear();
  int numPoints = 0;
  int numLRes = 0;
  for(FrameHessian* fh : frameHessians)
    for(PointHessian* ph : fh->pointHessians)
    {
      for(PointFrameResidual* r : ph->residuals)
      {
        if(!r->efResidual->isLinearized)
        {
          activeResiduals.push_back(r);
          r->resetOOB();
        }
        else
          numLRes++;
      }
      numPoints++;
    }

  if(!setting_debugout_runquiet)
    printf("OPTIMIZE %d pts, %d active res, %d lin res!\n",ef->nPoints,(int)activeResiduals.size(), numLRes);


  Vec3 lastEnergy = linearizeAll(false);
  double lastEnergyL = calcLEnergy();
  double lastEnergyM = calcMEnergy();





  if(multiThreading)
    treadReduce.reduce(boost::bind(&FullSystem::applyRes_Reductor, this, true, _1, _2, _3, _4), 0, activeResiduals.size(), 50);
  else
    applyRes_Reductor(true,0,activeResiduals.size(),0,0);


  if(!setting_debugout_runquiet)
  {
    printf("Initial Error       \t");
    printOptRes(lastEnergy, lastEnergyL, lastEnergyM, 0, 0, frameHessians.back()->aff_g2l().a, frameHessians.back()->aff_g2l().b);
  }

  debugPlotTracking();



  double lambda = 1e-1;
  float stepsize=1;
  VecX previousX = VecX::Constant(CPARS+ 8*frameHessians.size(), NAN);
  for(int iteration=0;iteration<mnumOptIts;iteration++)
  {
    // solve!
    backupState(iteration!=0);
    //solveSystemNew(0);
    solveSystem(iteration, lambda);
    double incDirChange = (1e-20 + previousX.dot(ef->lastX)) / (1e-20 + previousX.norm() * ef->lastX.norm());
    previousX = ef->lastX;


    if(std::isfinite(incDirChange) && (setting_solverMode & SOLVER_STEPMOMENTUM))
    {
      float newStepsize = exp(incDirChange*1.4);
      if(incDirChange<0 && stepsize>1) stepsize=1;

      stepsize = sqrtf(sqrtf(newStepsize*stepsize*stepsize*stepsize));
      if(stepsize > 2) stepsize=2;
      if(stepsize <0.25) stepsize=0.25;
    }

    bool canbreak = doStepFromBackup(stepsize,stepsize,stepsize,stepsize,stepsize);

    // eval new energy!
    Vec3 newEnergy = linearizeAll(false);
    double newEnergyL = calcLEnergy();
    double newEnergyM = calcMEnergy();

    if(!setting_debugout_runquiet)
    {
      printf("%s %d (L %.2f, dir %.2f, ss %.1f): \t",
             (newEnergy[0] +  newEnergy[1] +  newEnergyL + newEnergyM <
              lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM) ? "ACCEPT" : "REJECT",
             iteration,
             log10(lambda),
             incDirChange,
             stepsize);
      printOptRes(newEnergy, newEnergyL, newEnergyM , 0, 0, frameHessians.back()->aff_g2l().a, frameHessians.back()->aff_g2l().b);
    }

    if(setting_forceAceptStep || (newEnergy[0] +  newEnergy[1] +  newEnergyL + newEnergyM <
                                  lastEnergy[0] + lastEnergy[1] + lastEnergyL + lastEnergyM))
    {

      if(multiThreading)
        treadReduce.reduce(boost::bind(&FullSystem::applyRes_Reductor, this, true, _1, _2, _3, _4), 0, activeResiduals.size(), 50);
      else
        applyRes_Reductor(true,0,activeResiduals.size(),0,0);

      lastEnergy = newEnergy;
      lastEnergyL = newEnergyL;
      lastEnergyM = newEnergyM;

      lambda *= 0.25;
    }
    else
    {
      loadSateBackup();
      lastEnergy = linearizeAll(false);
      lastEnergyL = calcLEnergy();
      lastEnergyM = calcMEnergy();
      lambda *= 1e2;
    }


    if(canbreak && iteration >= setting_minOptIterations) break;
  }

  Vec10 newStateZero = Vec10::Zero();
  newStateZero.segment<2>(6) = frameHessians.back()->get_state().segment<2>(6);

  frameHessians.back()->setEvalPT(frameHessians.back()->PRE_worldToCam,
                                  newStateZero);
  EFDeltaValid=false;
  EFAdjointsValid=false;
  ef->setAdjointsF(&Hcalib);
  setPrecalcValues();

  std::cout << "activeResiduals size " << activeResiduals.size() << std::endl;
  lastEnergy = linearizeAll(true);

  if(!std::isfinite((double)lastEnergy[0]) || !std::isfinite((double)lastEnergy[1]) || !std::isfinite((double)lastEnergy[2]))
  {
    printf("KF Tracking failed: LOST!\n");
    isLost=true;
  }


  statistics_lastFineTrackRMSE = sqrtf((float)(lastEnergy[0] / (patternNum*ef->resInA)));

  if(calibLog != 0)
  {
    (*calibLog) << Hcalib.value_scaled.transpose() <<
        " " << frameHessians.back()->get_state_scaled().transpose() <<
        " " << sqrtf((float)(lastEnergy[0] / (patternNum*ef->resInA))) <<
        " " << ef->resInM << "\n";
    calibLog->flush();
  }

  {
    boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
    for(FrameHessian* fh : frameHessians)
    {
      fh->shell->camToWorld = fh->PRE_camToWorld;
      fh->shell->aff_g2l = fh->aff_g2l();
    }
  }

  debugPlotTracking();

  return sqrtf((float)(lastEnergy[0] / (patternNum*ef->resInA)));

}
#endif

/// restoring system
void FullSystem::solveSystem(int iteration, double lambda) {
  ef->lastNullspaces_forLogging = getNullspaces(ef->lastNullspaces_pose,
                                                ef->lastNullspaces_scale,
                                                ef->lastNullspaces_affA,
                                                ef->lastNullspaces_affB);

  ef->solveSystemF(iteration, lambda, &Hcalib);
}

/// calculate energy E (chi2) which is relative.
double FullSystem::calcLEnergy() {
  if(setting_forceAceptStep) return 0;

  double Ef = ef->calcLEnergyF_MT();
  return Ef;
}

/// remove outliers (the number of residuals become 0)
void FullSystem::removeOutliers() {
  int numPointsDropped=0;

  for(FrameHessian* fh : frameHessians){
    for(unsigned int i=0;i<fh->pointHessians.size();i++) {
      PointHessian* ph = fh->pointHessians[i];

      if(ph==0) continue;

      /// if the number of residuals at this point is 0, then discard.
      if(ph->residuals.size() == 0) {
        fh->pointHessiansOut.push_back(ph);
        ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
        fh->pointHessians[i] = fh->pointHessians.back();
        fh->pointHessians.pop_back();
        i--;
        numPointsDropped++;
      }
    }
  }
  ef->dropPointsF();
}

/// get nullspace for each state.
std::vector<VecX> FullSystem::getNullspaces(std::vector<VecX> &nullspaces_pose,
                                            std::vector<VecX> &nullspaces_scale,
                                            std::vector<VecX> &nullspaces_affA,
                                            std::vector<VecX> &nullspaces_affB)
{
  nullspaces_pose.clear();  /// size: 6, vec: 4 + 8*n
  nullspaces_scale.clear(); /// size: 1
  nullspaces_affA.clear();  /// size: 1
  nullspaces_affB.clear();  /// size: 1

  int n=CPARS+frameHessians.size()*8;

  /// all nullspaces
  std::vector<VecX> nullspaces_x0_pre;

  /// nullspace for pose.
  /// nullspace of the i-th variable
  for(int i=0; i<6; i++) {
    VecX nullspace_x0(n);
    nullspace_x0.setZero();

    for(FrameHessian* fh : frameHessians) {
      nullspace_x0.segment<6>(CPARS+fh->idx*8) = fh->nullspaces_pose.col(i);
      nullspace_x0.segment<3>(CPARS+fh->idx*8) *= SCALE_XI_TRANS_INVERSE;     /// remove scale
      nullspace_x0.segment<3>(CPARS+fh->idx*8+3) *= SCALE_XI_ROT_INVERSE;
    }
    nullspaces_x0_pre.push_back(nullspace_x0);
    nullspaces_pose.push_back(nullspace_x0);
  }

  /// nullspace of photometric params a,b
  for(int i=0;i<2;i++) {
    VecX nullspace_x0(n);
    nullspace_x0.setZero();

    for(FrameHessian* fh : frameHessians) {
      nullspace_x0.segment<2>(CPARS+fh->idx*8+6) = fh->nullspaces_affine.col(i).head<2>();
      nullspace_x0[CPARS+fh->idx*8+6] *= SCALE_A_INVERSE;
      nullspace_x0[CPARS+fh->idx*8+7] *= SCALE_B_INVERSE;
    }

    nullspaces_x0_pre.push_back(nullspace_x0);

    if(i==0) nullspaces_affA.push_back(nullspace_x0);
    if(i==1) nullspaces_affB.push_back(nullspace_x0);
  }

  /// nullspace of scale
  VecX nullspace_x0(n);

  nullspace_x0.setZero();
  for(FrameHessian* fh : frameHessians) {
    nullspace_x0.segment<6>(CPARS+fh->idx*8) = fh->nullspaces_scale;
    nullspace_x0.segment<3>(CPARS+fh->idx*8) *= SCALE_XI_TRANS_INVERSE;
    nullspace_x0.segment<3>(CPARS+fh->idx*8+3) *= SCALE_XI_ROT_INVERSE;
  }
  nullspaces_x0_pre.push_back(nullspace_x0);
  nullspaces_scale.push_back(nullspace_x0);

  return nullspaces_x0_pre;
}

} // namespace dso
