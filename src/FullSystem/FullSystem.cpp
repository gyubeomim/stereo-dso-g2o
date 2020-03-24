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

#include "FullSystem/FullSystem.h"
 
#include "stdio.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <algorithm>
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/ResidualProjections.h"
#include "FullSystem/ImmaturePoint.h"

#include "FullSystem/CoarseTracker.h"
#include "FullSystem/CoarseInitializer.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "IOWrapper/Output3DWrapper.h"

#include "util/ImageAndExposure.h"

#include <cmath>
#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace dso {

int FrameHessian::instanceCounter=0;
int PointHessian::instanceCounter=0;
int CalibHessian::instanceCounter=0;

FullSystem::FullSystem() {

  int retstat =0;
  if(setting_logStuff) {

    retstat += system("rm -rf logs");
    retstat += system("mkdir logs");

    retstat += system("rm -rf mats");
    retstat += system("mkdir mats");

    calibLog = new std::ofstream();
    calibLog->open("logs/calibLog.txt", std::ios::trunc | std::ios::out);
    calibLog->precision(12);

    numsLog = new std::ofstream();
    numsLog->open("logs/numsLog.txt", std::ios::trunc | std::ios::out);
    numsLog->precision(10);

    coarseTrackingLog = new std::ofstream();
    coarseTrackingLog->open("logs/coarseTrackingLog.txt", std::ios::trunc | std::ios::out);
    coarseTrackingLog->precision(10);

    eigenAllLog = new std::ofstream();
    eigenAllLog->open("logs/eigenAllLog.txt", std::ios::trunc | std::ios::out);
    eigenAllLog->precision(10);

    eigenPLog = new std::ofstream();
    eigenPLog->open("logs/eigenPLog.txt", std::ios::trunc | std::ios::out);
    eigenPLog->precision(10);

    eigenALog = new std::ofstream();
    eigenALog->open("logs/eigenALog.txt", std::ios::trunc | std::ios::out);
    eigenALog->precision(10);

    DiagonalLog = new std::ofstream();
    DiagonalLog->open("logs/diagonal.txt", std::ios::trunc | std::ios::out);
    DiagonalLog->precision(10);

    variancesLog = new std::ofstream();
    variancesLog->open("logs/variancesLog.txt", std::ios::trunc | std::ios::out);
    variancesLog->precision(10);


    nullspacesLog = new std::ofstream();
    nullspacesLog->open("logs/nullspacesLog.txt", std::ios::trunc | std::ios::out);
    nullspacesLog->precision(10);
  }
  else
  {
    nullspacesLog=0;
    variancesLog=0;
    DiagonalLog=0;
    eigenALog=0;
    eigenPLog=0;
    eigenAllLog=0;
    numsLog=0;
    calibLog=0;
  }

  assert(retstat!=293847);



  selectionMap = new float[wG[0]*hG[0]];

  coarseDistanceMap = new CoarseDistanceMap(wG[0], hG[0]);
  coarseTracker = new CoarseTracker(wG[0], hG[0]);
  coarseTracker_forNewKF = new CoarseTracker(wG[0], hG[0]);
  coarseInitializer = new CoarseInitializer(wG[0], hG[0]);
  pixelSelector = new PixelSelector(wG[0], hG[0]);

  statistics_lastNumOptIts=0;
  statistics_numDroppedPoints=0;
  statistics_numActivatedPoints=0;
  statistics_numCreatedPoints=0;
  statistics_numForceDroppedResBwd = 0;
  statistics_numForceDroppedResFwd = 0;
  statistics_numMargResFwd = 0;
  statistics_numMargResBwd = 0;

  lastCoarseRMSE.setConstant(100);

  currentMinActDist=2;
  initialized=false;


  ef = new EnergyFunctional();
  ef->red = &this->treadReduce;

  isLost=false;
  initFailed=false;

  needNewKFAfter = -1;

  linearizeOperation=true;
  runMapping=true;
  mappingThread = boost::thread(&FullSystem::mappingLoop, this);
  lastRefStopID=0;

  minIdJetVisDebug = -1;
  maxIdJetVisDebug = -1;
  minIdJetVisTracker = -1;
  maxIdJetVisTracker = -1;
}

FullSystem::~FullSystem() {
  blockUntilMappingIsFinished();

  if(setting_logStuff)
  {
    calibLog->close(); delete calibLog;
    numsLog->close(); delete numsLog;
    coarseTrackingLog->close(); delete coarseTrackingLog;
    //errorsLog->close(); delete errorsLog;
    eigenAllLog->close(); delete eigenAllLog;
    eigenPLog->close(); delete eigenPLog;
    eigenALog->close(); delete eigenALog;
    DiagonalLog->close(); delete DiagonalLog;
    variancesLog->close(); delete variancesLog;
    nullspacesLog->close(); delete nullspacesLog;
  }

  delete[] selectionMap;

  for(FrameShell* s : allFrameHistory)
    delete s;
  for(FrameHessian* fh : unmappedTrackedFrames)
    delete fh;

  delete coarseDistanceMap;
  delete coarseTracker;
  delete coarseTracker_forNewKF;
  delete coarseInitializer;
  delete pixelSelector;
  delete ef;
}

void FullSystem::setOriginalCalib(VecXf originalCalib, int originalW, int originalH)
{}

// set camera response function.
void FullSystem::setGammaFunction(float* BInv) {
  if(BInv==0) return;

  // copy BInv.
  memcpy(Hcalib.Binv, BInv, sizeof(float)*256);


  // invert.
  for(int i=1;i<255;i++)
  {
    // find val, such that Binv[val] = i.
    // I dont care about speed for this, so do it the stupid way.

    for(int s=1;s<255;s++)
    {
      if(BInv[s] <= i && BInv[s+1] >= i)
      {
        Hcalib.B[i] = s+(i - BInv[s]) / (BInv[s+1]-BInv[s]);
        break;
      }
    }
  }
  Hcalib.B[0] = 0;
  Hcalib.B[255] = 255;
}

void FullSystem::printResult(std::string file) {
  boost::unique_lock<boost::mutex> lock(trackMutex);
  boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

  std::ofstream myfile;
  myfile.open (file.c_str());
  myfile << std::setprecision(15);
  int i = 0;

  Eigen::Matrix<double,3,3> last_R = (*(allFrameHistory.begin()))->camToWorld.so3().matrix();
  Eigen::Matrix<double,3,1> last_T = (*(allFrameHistory.begin()))->camToWorld.translation().transpose();

  for(FrameShell* s : allFrameHistory)
  {
    if(!s->poseValid)
    {
      myfile<< last_R(0,0) <<" "<<last_R(0,1)<<" "<<last_R(0,2)<<" "<<last_T(0,0)<<" "<<
          last_R(1,0) <<" "<<last_R(1,1)<<" "<<last_R(1,2)<<" "<<last_T(1,0)<<" "<<
          last_R(2,0) <<" "<<last_R(2,1)<<" "<<last_R(2,2)<<" "<<last_T(2,0)<<"\n";
      continue;
    }

    if(setting_onlyLogKFPoses && s->marginalizedAt == s->id)
    {
      myfile<< last_R(0,0) <<" "<<last_R(0,1)<<" "<<last_R(0,2)<<" "<<last_T(0,0)<<" "<<
          last_R(1,0) <<" "<<last_R(1,1)<<" "<<last_R(1,2)<<" "<<last_T(1,0)<<" "<<
          last_R(2,0) <<" "<<last_R(2,1)<<" "<<last_R(2,2)<<" "<<last_T(2,0)<<"\n";
      continue;
    }

    const Eigen::Matrix<double,3,3> R = s->camToWorld.so3().matrix();
    const Eigen::Matrix<double,3,1> T = s->camToWorld.translation().transpose();

    last_R = R;
    last_T = T;

    myfile<< R(0,0) <<" "<<R(0,1)<<" "<<R(0,2)<<" "<<T(0,0)<<" "<<
        R(1,0) <<" "<<R(1,1)<<" "<<R(1,2)<<" "<<T(1,0)<<" "<<
        R(2,0) <<" "<<R(2,1)<<" "<<R(2,2)<<" "<<T(2,0)<<"\n";

    //myfile << s->timestamp <<
    //" " << s->camToWorld.translation().transpose()<<
    //" " << s->camToWorld.so3().unit_quaternion().x()<<
    //" " << s->camToWorld.so3().unit_quaternion().y()<<
    //" " << s->camToWorld.so3().unit_quaternion().z()<<
    //" " << s->camToWorld.so3().unit_quaternion().w() << "\n";
    i++;
  }
  myfile.close();
}

// use the determined motion model to track the new frame, an get the pose and photometric parameters.
Vec4 FullSystem::trackNewCoarse(FrameHessian* fh, FrameHessian* fh_right) {
  assert(allFrameHistory.size() > 0);

  // set pose initialization.
  // printf("the size of allFrameHistory is %d \n", (int)allFrameHistory.size());

  // show original images
  for(IOWrap::Output3DWrapper* ow : outputWrapper) {
    ow->pushStereoLiveFrame(fh, fh_right);
  }

  //// lastF = last TrackingRef.
  FrameHessian* lastF = coarseTracker->lastRef;

  AffLight aff_last_2_l = AffLight(0,0);

  // STEP1: set different cases.
  std::vector<SE3,Eigen::aligned_allocator<SE3>> lastF_2_fh_tries;

  // for first two frames process differently
  if(allFrameHistory.size() == 2) {
    initializeFromInitializer(fh);

    lastF_2_fh_tries.push_back(SE3(Eigen::Matrix<double, 3, 3>::Identity(), Eigen::Matrix<double,3,1>::Zero() ));

    for(float rotDelta=0.02; rotDelta < 0.05; rotDelta = rotDelta + 0.02)
    {
      lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
      lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
      lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,0,rotDelta), Vec3(0,0,0)));			// assume constant motion.
      lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
      lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,-rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
      lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,0,-rotDelta), Vec3(0,0,0)));			// assume constant motion.
      lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,0), Vec3(0,0,0)));	                // assume constant motion.
      lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,rotDelta,rotDelta), Vec3(0,0,0)));	                // assume constant motion.
      lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,0,rotDelta), Vec3(0,0,0)));	                // assume constant motion.
      lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,0), Vec3(0,0,0)));	        // assume constant motion.
      lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,-rotDelta,rotDelta), Vec3(0,0,0)));	        // assume constant motion.
      lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,0,rotDelta), Vec3(0,0,0)));	        // assume constant motion.
      lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,0), Vec3(0,0,0)));	        // assume constant motion.
      lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,rotDelta,-rotDelta), Vec3(0,0,0)));	        // assume constant motion.
      lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,0,-rotDelta), Vec3(0,0,0)));	        // assume constant motion.
      lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,0), Vec3(0,0,0)));	        // assume constant motion.
      lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,-rotDelta,-rotDelta), Vec3(0,0,0)));	        // assume constant motion.
      lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,0,-rotDelta), Vec3(0,0,0)));	        // assume constant motion.
      lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
      lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
      lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
      lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	        // assume constant motion.
      lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
      lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	        // assume constant motion.
      lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	        // assume constant motion.
      lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	        // assume constant motion.
    }

    coarseTracker->makeK(&Hcalib);

    //// set the first keyframe into 'coarseTracker->lastRef'.
    coarseTracker->setCTRefForFirstFrame(frameHessians);

    lastF = coarseTracker->lastRef;
  }
  else {
    FrameShell* slast = allFrameHistory[allFrameHistory.size()-2];
    FrameShell* sprelast = allFrameHistory[allFrameHistory.size()-3];
    SE3 slast_2_sprelast;
    SE3 lastF_2_slast;

    // lock on global pose consistency!
    {
      boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

      //// T_c-2_c-1
      slast_2_sprelast = sprelast->camToWorld.inverse() * slast->camToWorld;

      //// T_c-1_k
      lastF_2_slast = slast->camToWorld.inverse() * lastF->shell->camToWorld;

      aff_last_2_l = slast->aff_g2l;
    }

    SE3 fh_2_slast = slast_2_sprelast;  // assumed to be the same as fh_2_slast (T_c-2_c-1).

    // get last delta-movement.
    lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast);	                        // assume constant motion.
    lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() * lastF_2_slast);	// assume double motion (frame skipped)
    lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log()*0.5).inverse() * lastF_2_slast);       // assume half motion.
    lastF_2_fh_tries.push_back(lastF_2_slast);                                                  // assume zero motion.
    lastF_2_fh_tries.push_back(SE3());                                                          // assume zero motion FROM KF.

    //// already commented out.
    // lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log()*1.5).inverse() * SE3::exp(fh_2_slast.log()*1.5).inverse() * lastF_2_slast);
    // lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() * fh_2_slast.inverse() * lastF_2_slast);
    // lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log()*1.5).inverse() * SE3::exp(fh_2_slast.log()*1.5).inverse() *  SE3::exp(fh_2_slast.log()*1.5).inverse() * lastF_2_slast);
    // lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() * fh_2_slast.inverse() * fh_2_slast.inverse() * lastF_2_slast);
    // lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log()*1.5).inverse() * SE3::exp(fh_2_slast.log()*1.5).inverse() *  SE3::exp(fh_2_slast.log()*1.5).inverse() * SE3::exp(fh_2_slast.log()*1.5).inverse() * lastF_2_slast);
    // lastF_2_fh_tries.push_back(fh_2_slast.inverse() * fh_2_slast.inverse() * fh_2_slast.inverse() * fh_2_slast.inverse() * fh_2_slast.inverse() * lastF_2_slast);
    // lastF_2_fh_tries.push_back(SE3::exp(fh_2_slast.log()*1.5).inverse() * SE3::exp(fh_2_slast.log()*1.5).inverse() *  SE3::exp(fh_2_slast.log()*1.5).inverse() * SE3::exp(fh_2_slast.log()*1.5).inverse() * SE3::exp(fh_2_slast.log()*1.5).inverse() * lastF_2_slast);

    // just try a TON of different initializations (all rotations). In the end,
    // if they don't work they will only be tried on the coarsest level, which is super fast anyway.
    // also, if tracking rails here we loose, so we really, really want to avoid that.
    for(float rotDelta=0.02; rotDelta < 0.02; rotDelta++) {
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,0), Vec3(0,0,0)));			  // assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,0), Vec3(0,0,0)));			  // assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,0,rotDelta), Vec3(0,0,0)));			  // assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,0), Vec3(0,0,0)));		  // assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,0), Vec3(0,0,0)));		  // assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,0,-rotDelta), Vec3(0,0,0)));		  // assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,0), Vec3(0,0,0)));	          // assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,rotDelta), Vec3(0,0,0)));	          // assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,rotDelta), Vec3(0,0,0)));	          // assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,0), Vec3(0,0,0)));	          // assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,rotDelta), Vec3(0,0,0)));	          // assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,rotDelta), Vec3(0,0,0)));	          // assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,0), Vec3(0,0,0)));	          // assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,rotDelta,-rotDelta), Vec3(0,0,0)));	          // assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,0,-rotDelta), Vec3(0,0,0)));	          // assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,0), Vec3(0,0,0)));	  // assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,0,-rotDelta,-rotDelta), Vec3(0,0,0)));	  // assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,0,-rotDelta), Vec3(0,0,0)));	  // assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));  // assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	  // assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	  // assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	  // assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	  // assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	  // assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	  // assume constant motion.
      lastF_2_fh_tries.push_back(fh_2_slast.inverse() * lastF_2_slast * SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	  // assume constant motion.
    }

    if(!slast->poseValid || !sprelast->poseValid || !lastF->shell->poseValid) {
      lastF_2_fh_tries.clear();
      lastF_2_fh_tries.push_back(SE3());
    }
  }

  Vec3 flowVecs = Vec3(100,100,100);

  //// TrackingRef(lastF) To Cam(fh). (T_ck)
  SE3 lastF_2_fh = SE3();

  AffLight aff_g2l = AffLight(0,0);

  // as long as maxResForImmediateAccept is not reached, I'll continue through the options.
  // I'll keep track of the so-far best achieved residual for each level in achievedRes.
  // If on a coarse level, tracking is WORSE than achievedRes, we will not continue to save time.

  Vec5 achievedRes = Vec5::Constant(NAN);
  bool haveOneGood = false;
  int tryIterations=0;

  // STEP2: try different cases to get a good tracking result.
  for(unsigned int i=0; i<lastF_2_fh_tries.size(); i++) {
    AffLight aff_g2l_this = aff_last_2_l;      /// assignment of the previous frame to the current frame.
    SE3 lastF_2_fh_this = lastF_2_fh_tries[i];

    bool trackingIsGood = coarseTracker->trackNewestCoarse(fh,
                                                           lastF_2_fh_this,
                                                           aff_g2l_this,
                                                           pyrLevelsUsed-1,
                                                           achievedRes);	// in each level has to be at least as good as the last try.

    tryIterations++;

    //// this for loop is usually finisihed when i == 0.
    if(i != 0) {
      printf("RE-TRACK ATTEMPT %d with initOption %d and start-lvl %d (ab %f %f): %f %f %f %f %f -> %f %f %f %f %f \n",
             i,
             i,
             pyrLevelsUsed-1,
             aff_g2l_this.a,
             aff_g2l_this.b,
             achievedRes[0],
             achievedRes[1],
             achievedRes[2],
             achievedRes[3],
             achievedRes[4],
             coarseTracker->lastResiduals[0],
             coarseTracker->lastResiduals[1],
             coarseTracker->lastResiduals[2],
             coarseTracker->lastResiduals[3],
             coarseTracker->lastResiduals[4]);
    }

    /// STEP3: If the tracking is good, the 0-level residual is better than the best, leave a pose, and save the best energy value of each layer.
    // do we have a new winner?
    if(trackingIsGood &&
       std::isfinite((float)coarseTracker->lastResiduals[0]) &&
       !(coarseTracker->lastResiduals[0] >= achievedRes[0]))
    {
      //printf("take over. minRes %f -> %f!\n", achievedRes[0], coarseTracker->lastResiduals[0]);
      flowVecs = coarseTracker->lastFlowIndicators;
      aff_g2l = aff_g2l_this;
      lastF_2_fh = lastF_2_fh_this;
      haveOneGood = true;
    }

    // take over achieved res (always).
    if(haveOneGood) {
      for(int i=0; i<5; i++) {
        // take over if achievedRes is either bigger or NAN.
        if(!std::isfinite((float)achievedRes[i]) ||
           achievedRes[i] > coarseTracker->lastResiduals[i])
        {
          achievedRes[i] = coarseTracker->lastResiduals[i];
        }
      }
    }

    /// STEP4: pause if energy is less than the threshold, and set the threshold for the next time.
    if(haveOneGood &&  achievedRes[0] < lastCoarseRMSE[0]*setting_reTrackThreshold)
      break;
  }

  if(!haveOneGood) {
    printf("BIG ERROR! tracking failed entirely. Take predictred pose and hope we may somehow recover.\n");
    flowVecs = Vec3(0,0,0);
    aff_g2l = aff_last_2_l;
    lastF_2_fh = lastF_2_fh_tries[0];
  }

  /// give the best value obtained this time as the threshold value next time.
  lastCoarseRMSE = achievedRes;

  /// STEP5: At this time the shell is in the tracking phase, no one uses it, set the value.
  // no lock required, as fh is not used anywhere yet.
  //// Trc = Tcr.inv
  fh->shell->camToTrackingRef = lastF_2_fh.inverse();
  fh->shell->trackingRef = lastF->shell;
  fh->shell->aff_g2l = aff_g2l;
  //// Twc = Twr * Trc
  fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;

  // std::cout << "AFTER fh->shell->trackingRef(lastF)->camToWorld\n" << lastF->shell->camToWorld.matrix() << "\nAFTER fh->shell->camToTrackingRef\n" << fh->shell->camToTrackingRef.matrix() << "\nAFTER fh->shell->camToWorld\n" << fh->shell->camToWorld.matrix() << std::endl;

  //// comment out.
  // Eigen::Matrix<double,3,1> last_T = fh->shell->camToWorld.translation().transpose();
  // std::cout<<"[+] xyz: [" << last_T(0,0) << ", " << last_T(1,0) << ", "<<last_T(2,0) << "]" << std::endl;

  if(coarseTracker->firstCoarseRMSE < 0)
    coarseTracker->firstCoarseRMSE = achievedRes[0];

  if(!setting_debugout_runquiet)
    printf("Coarse Tracker tracked ab = %f %f (exp %f). Res %f!\n", aff_g2l.a, aff_g2l.b, fh->ab_exposure, achievedRes[0]);

  if(setting_logStuff) {
    (*coarseTrackingLog) << std::setprecision(16)
                         << fh->shell->id << " "
                         << fh->shell->timestamp << " "
                         << fh->ab_exposure << " "
                         << fh->shell->camToWorld.log().transpose() << " "
                         << aff_g2l.a << " "
                         << aff_g2l.b << " "
                         << achievedRes[0] << " "
                         << tryIterations << "\n";
  }

  return Vec4(achievedRes[0], flowVecs[0], flowVecs[1], flowVecs[2]);
}

void FullSystem::stereoMatch( ImageAndExposure* image, ImageAndExposure* image_right, int id, cv::Mat &idepthMap) {
  // =========================== add into allFrameHistory =========================
  FrameHessian* fh = new FrameHessian();
  FrameHessian* fh_right = new FrameHessian();
  FrameShell* shell = new FrameShell();

  shell->camToWorld = SE3(); 		// no lock required, as fh is not used anywhere yet.
  shell->aff_g2l = AffLight(0,0);
  shell->marginalizedAt = shell->id = allFrameHistory.size();
  shell->timestamp = image->timestamp;
  shell->incoming_id = id; // id passed into DSO
  fh->shell = shell;
  fh_right->shell=shell;

  // =========================== make Images / derivatives etc. =========================
  fh->ab_exposure = image->exposure_time;
  fh->makeImages(image->image, &Hcalib);
  fh_right->ab_exposure = image_right->exposure_time;
  fh_right->makeImages(image_right->image,&Hcalib);

  Mat33f K = Mat33f::Identity();
  K(0,0) = Hcalib.fxl();
  K(1,1) = Hcalib.fyl();
  K(0,2) = Hcalib.cxl();
  K(1,2) = Hcalib.cyl();

  int counter = 0;

  makeNewTraces(fh, fh_right, 0);

  unsigned  char * idepthMapPtr = idepthMap.data;

  for(ImmaturePoint* ph : fh->immaturePoints) {
    ph->u_stereo = ph->u;
    ph->v_stereo = ph->v;
    ph->idepth_min_stereo = ph->idepth_min = 0;
    ph->idepth_max_stereo = ph->idepth_max = NAN;

    ImmaturePointStatus phTraceRightStatus = ph->traceStereo(fh_right, K, 1);

    if(phTraceRightStatus == ImmaturePointStatus::IPS_GOOD) {
      ImmaturePoint* phRight = new ImmaturePoint(ph->lastTraceUV(0), ph->lastTraceUV(1), fh_right, &Hcalib );

      phRight->u_stereo = phRight->u;
      phRight->v_stereo = phRight->v;
      phRight->idepth_min_stereo = ph->idepth_min = 0;
      phRight->idepth_max_stereo = ph->idepth_max = NAN;
      ImmaturePointStatus  phTraceLeftStatus = phRight->traceStereo(fh, K, 0);

      float u_stereo_delta = abs(ph->u_stereo - phRight->lastTraceUV(0));
      float depth = 1.0f/ph->idepth_stereo;

      if(phTraceLeftStatus == ImmaturePointStatus::IPS_GOOD && u_stereo_delta < 1 && depth > 0 && depth < 70)    //original u_stereo_delta 1 depth < 70
      {
        ph->idepth_min = ph->idepth_min_stereo;
        ph->idepth_max = ph->idepth_max_stereo;

        *((float *)(idepthMapPtr + int(ph->v) * idepthMap.step) + (int)ph->u *3) = ph->idepth_stereo;
        *((float *)(idepthMapPtr + int(ph->v) * idepthMap.step) + (int)ph->u *3 + 1) = ph->idepth_min;
        *((float *)(idepthMapPtr + int(ph->v) * idepthMap.step) + (int)ph->u *3 + 2) = ph->idepth_max;

        counter++;
      }
    }
  }

  //std::sort(error.begin(), error.end());
  //std::cout << 0.25 <<" "<<error[error.size()*0.25].first<<" "<<
  //0.5 <<" "<<error[error.size()*0.5].first<<" "<<
  //0.75 <<" "<<error[error.size()*0.75].first<<" "<<
  //0.1 <<" "<<error.back().first << std::endl;
  //for(int i = 0; i < error.size(); i++)
  //std::cout << error[i].first << " " << error[i].second.first << " " << error[i].second.second << std::endl;

  std::cout<<" frameID " << id << " got good matches " << counter << std::endl;

  delete fh;
  delete fh_right;

  return;
}

// process nonkey frame to refine key frame idepth
void FullSystem::traceNewCoarseNonKey(FrameHessian* fh, FrameHessian* fh_right) {
  boost::unique_lock<boost::mutex> lock(mapMutex);

  // new idepth after refinement
  float idepth_min_update = 0;
  float idepth_max_update = 0;

  Mat33f K = Mat33f::Identity();
  K(0, 0) = Hcalib.fxl();
  K(1, 1) = Hcalib.fyl();
  K(0, 2) = Hcalib.cxl();
  K(1, 2) = Hcalib.cyl();

  Mat33f Ki = K.inverse();

  // go through all active frames
  for (FrameHessian *host : frameHessians) {
    //// deprecated.
    // number++;
    // int trace_total = 0, trace_good = 0, trace_oob = 0, trace_out = 0, trace_skip = 0, trace_badcondition = 0, trace_uninitialized = 0;

    // trans from reference keyframe to newest frame
    SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
    // KRK-1
    Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
    // KRi
    Mat33f KRi = K * hostToNew.rotationMatrix().inverse().cast<float>();
    // Kt
    Vec3f Kt = K * hostToNew.translation().cast<float>();
    // t
    Vec3f t = hostToNew.translation().cast<float>();

    //aff
    Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure, host->aff_g2l(), fh->aff_g2l()).cast<float>();

    for (ImmaturePoint *ph : host->immaturePoints) {
      // do temperol stereo match
      ImmaturePointStatus phTrackStatus = ph->traceOn(fh, KRKi, Kt, aff, &Hcalib, false);

      if (phTrackStatus == ImmaturePointStatus::IPS_GOOD) {
        ImmaturePoint *phNonKey = new ImmaturePoint(ph->lastTraceUV(0), ph->lastTraceUV(1), fh, &Hcalib);

        // project onto newest frame
        Vec3f ptpMin = KRKi * (Vec3f(ph->u, ph->v, 1) / ph->idepth_min) + Kt;
        float idepth_min_project = 1.0f / ptpMin[2];

        Vec3f ptpMax = KRKi * (Vec3f(ph->u, ph->v, 1) / ph->idepth_max) + Kt;
        float idepth_max_project = 1.0f / ptpMax[2];

        phNonKey->idepth_min = idepth_min_project;
        phNonKey->idepth_max = idepth_max_project;
        phNonKey->u_stereo = phNonKey->u;
        phNonKey->v_stereo = phNonKey->v;
        phNonKey->idepth_min_stereo = phNonKey->idepth_min;
        phNonKey->idepth_max_stereo = phNonKey->idepth_max;

        // do static stereo match from left image to right
        ImmaturePointStatus phNonKeyStereoStatus = phNonKey->traceStereo(fh_right, K, 1);

        if(phNonKeyStereoStatus == ImmaturePointStatus::IPS_GOOD) {
          ImmaturePoint* phNonKeyRight = new ImmaturePoint(phNonKey->lastTraceUV(0), phNonKey->lastTraceUV(1), fh_right, &Hcalib);

          phNonKeyRight->u_stereo = phNonKeyRight->u;
          phNonKeyRight->v_stereo = phNonKeyRight->v;
          phNonKeyRight->idepth_min_stereo = phNonKey->idepth_min;
          phNonKeyRight->idepth_max_stereo = phNonKey->idepth_max;

          // do static stereo match from right image to left
          ImmaturePointStatus  phNonKeyRightStereoStatus = phNonKeyRight->traceStereo(fh, K, 0);

          // change of u after two different stereo match
          float u_stereo_delta = abs(phNonKey->u_stereo - phNonKeyRight->lastTraceUV(0));
          float disparity = phNonKey->u_stereo - phNonKey->lastTraceUV[0];

          // free to debug the threshold
          if(u_stereo_delta > 1 && disparity < 10) {
            ph->lastTraceStatus = ImmaturePointStatus :: IPS_OUTLIER;
            continue;
          }
          else {
            // project back
            Vec3f pinverse_min = KRi * (Ki * Vec3f(phNonKey->u_stereo, phNonKey->v_stereo, 1) / phNonKey->idepth_min_stereo - t);
            idepth_min_update = 1.0f / pinverse_min(2);

            Vec3f pinverse_max = KRi * (Ki * Vec3f(phNonKey->u_stereo, phNonKey->v_stereo, 1) / phNonKey->idepth_max_stereo - t);
            idepth_max_update = 1.0f / pinverse_max(2);

            ph->idepth_min = idepth_min_update;
            ph->idepth_max = idepth_max_update;

            delete phNonKey;
            delete phNonKeyRight;
          }
        }
        else {
          delete phNonKey;
          continue;
        }
      }

    //// deprecated.
      // if (ph->lastTraceStatus == ImmaturePointStatus::IPS_GOOD) trace_good++;
      // if (ph->lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION) trace_badcondition++;
      // if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OOB) trace_oob++;
      // if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER) trace_out++;
      // if (ph->lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED) trace_skip++;
      // if (ph->lastTraceStatus == ImmaturePointStatus::IPS_UNINITIALIZED) trace_uninitialized++;
      // trace_total++;
    }
  }
}

//process keyframe
void FullSystem::traceNewCoarseKey(FrameHessian* fh, FrameHessian* fh_right) {
  boost::unique_lock<boost::mutex> lock(mapMutex);

  //// deprecated.
  // int trace_total=0, trace_good=0, trace_oob=0, trace_out=0, trace_skip=0, trace_badcondition=0, trace_uninitialized=0;

  Mat33f K = Mat33f::Identity();
  K(0,0) = Hcalib.fxl();
  K(1,1) = Hcalib.fyl();
  K(0,2) = Hcalib.cxl();
  K(1,2) = Hcalib.cyl();

  // go through all active frames
  for(FrameHessian* host : frameHessians){
    // trans from reference key frame to the newest one
    SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
    //KRK-1
    Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
    //Kt
    Vec3f Kt = K * hostToNew.translation().cast<float>();

    Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure, host->aff_g2l(), fh->aff_g2l()).cast<float>();

    for(ImmaturePoint* ph : host->immaturePoints) {
      ImmaturePointStatus phTrackStatus = ph->traceOn(fh, KRKi, Kt, aff, &Hcalib, false );

      //// deprecated.
      // if(ph->lastTraceStatus==ImmaturePointStatus::IPS_GOOD) trace_good++;
      // if(ph->lastTraceStatus==ImmaturePointStatus::IPS_BADCONDITION) trace_badcondition++;
      // if(ph->lastTraceStatus==ImmaturePointStatus::IPS_OOB) trace_oob++;
      // if(ph->lastTraceStatus==ImmaturePointStatus::IPS_OUTLIER) trace_out++;
      // if(ph->lastTraceStatus==ImmaturePointStatus::IPS_SKIPPED) trace_skip++;
      // if(ph->lastTraceStatus==ImmaturePointStatus::IPS_UNINITIALIZED) trace_uninitialized++;
      // trace_total++;
    }
  }
}

/// handle picking out points to be activated.
void FullSystem::activatePointsMT_Reductor(std::vector<PointHessian*>* optimized,
                                           std::vector<ImmaturePoint*>* toOptimize,
                                           int min, int max, Vec10* stats, int tid) {
  ImmaturePointTemporaryResidual* tr = new ImmaturePointTemporaryResidual[frameHessians.size()];
  for(int k=min;k<max;k++)
  {
    (*optimized)[k] = optimizeImmaturePoint((*toOptimize)[k],1,tr);
  }
  delete[] tr;
}

/// activate immature points, join optimization.
void FullSystem::activatePointsMT() {
  /// STEP1: threshold calculation, control the number by distance map. currentMinActDist initial value is 2.
  if(ef->nPoints < setting_desiredPointDensity*0.66)   //setting_desiredPointDensity 是2000
    currentMinActDist -= 0.8;  //original 0.8
  if(ef->nPoints < setting_desiredPointDensity*0.8)
    currentMinActDist -= 0.5;  //original 0.5
  else if(ef->nPoints < setting_desiredPointDensity*0.9)
    currentMinActDist -= 0.2;  //original 0.2
  else if(ef->nPoints < setting_desiredPointDensity)
    currentMinActDist -= 0.1;  //original 0.1

  if(ef->nPoints > setting_desiredPointDensity*1.5)
    currentMinActDist += 0.8;
  if(ef->nPoints > setting_desiredPointDensity*1.3)
    currentMinActDist += 0.5;
  if(ef->nPoints > setting_desiredPointDensity*1.15)
    currentMinActDist += 0.2;
  if(ef->nPoints > setting_desiredPointDensity)
    currentMinActDist += 0.1;

  if(currentMinActDist < 0) currentMinActDist = 0;
  if(currentMinActDist > 4) currentMinActDist = 4;

  if(!setting_debugout_runquiet)
    printf("SPARSITY:  MinActDist %f (need %d points, have %d points)!\n",
           currentMinActDist, (int)(setting_desiredPointDensity), ef->nPoints);

  FrameHessian* newestHs = frameHessians.back();

  // make dist map.
  coarseDistanceMap->makeK(&Hcalib);
  coarseDistanceMap->makeDistanceMap(frameHessians, newestHs);

  //coarseTracker->debugPlotDistMap("distMap");

// points to be activated.
  std::vector<ImmaturePoint*> toOptimize;
  toOptimize.reserve(20000);

  /// STEP2: handle immature points, activate/delete/skip.
  // go through all active frames
  for(FrameHessian* host : frameHessians) {
    if(host == newestHs) continue;

    SE3 fhToNew = newestHs->PRE_worldToCam * host->PRE_camToWorld;
    Mat33f KRKi = (coarseDistanceMap->K[1] * fhToNew.rotationMatrix().cast<float>() * coarseDistanceMap->Ki[0]);
    Vec3f Kt = (coarseDistanceMap->K[1] * fhToNew.translation().cast<float>());

    // for all immaturePoints in frameHessian
    for(unsigned int i=0;i<host->immaturePoints.size();i+=1) {
      ImmaturePoint* ph = host->immaturePoints[i];
      ph->idxInImmaturePoints = i;

      // delete points that have never been traced successfully, or that are outlier on the last trace.
      if(!std::isfinite(ph->idepth_max) || ph->lastTraceStatus == IPS_OUTLIER) {
        //				immature_invalid_deleted++;
        // remove point.
        delete ph;
        host->immaturePoints[i]=0;
        continue;
      }

      /// activation conditions for immature points.
      // can activate only if this is true.
      bool canActivate = (ph->lastTraceStatus == IPS_GOOD
                          || ph->lastTraceStatus == IPS_SKIPPED
                          || ph->lastTraceStatus == IPS_BADCONDITION
                          || ph->lastTraceStatus == IPS_OOB )
                         && ph->lastTracePixelInterval < 8
                                                         && ph->quality > setting_minTraceQuality
                         && (ph->idepth_max+ph->idepth_min) > 0;

      // if I cannot activate the point, skip it. Maybe also delete it.
      if(!canActivate) {
        /// delete the marginalized frames, and OOB points.
        // if point will be out afterwards, delete it instead.
        if(ph->host->flaggedForMarginalization || ph->lastTraceStatus == IPS_OOB)
        {
          //immature_notReady_deleted++;
          delete ph;
          host->immaturePoints[i]=0;
        }
        //immature_notReady_skipped++;
        continue;
      }

      // see if we need to activate point due to distance map.
      Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt*(0.5f*(ph->idepth_max + ph->idepth_min));
      int u = ptp[0]/ptp[2] + 0.5f;
      int v = ptp[1]/ptp[2] + 0.5f;

      if((u > 0 && v > 0 && u < wG[1] && v < hG[1])) {
        /// distance map + decimal point
        float dist = coarseDistanceMap->fwdWarpedIDDistFinal[u+wG[1]*v] + (ptp[0]-floorf((float)(ptp[0])));

        /// the more points, the larger the distance thresohld.
        if(dist >= currentMinActDist*ph->my_type) {
          coarseDistanceMap->addIntoDistFinal(u,v);
          toOptimize.push_back(ph);
        }
      }
      else {
        delete ph;
        host->immaturePoints[i]=0; // delete points.
      }
    }
  }

  //	printf("ACTIVATE: %d. (del %d, notReady %d, marg %d, good %d, marg-skip %d)\n",
  //			(int)toOptimize.size(), immature_deleted, immature_notReady, immature_needMarg, immature_want, immature_margskip);

  /// STEP3: optimize the immature points selected in the previous step, perform idepth optimization, an get PointHessian.
  std::vector<PointHessian*> optimized;
  optimized.resize(toOptimize.size());

  if(multiThreading) {
    treadReduce.reduce(boost::bind(&FullSystem::activatePointsMT_Reductor, this, &optimized, &toOptimize, _1, _2, _3, _4), 0, toOptimize.size(), 50);
  }
  else {
    activatePointsMT_Reductor(&optimized, &toOptimize, 0, toOptimize.size(), 0, 0);
  }

  /// STEP4: Add PointHessian to the energy function, remove the immature points of convergence, or bad points.
  for(unsigned k=0; k<toOptimize.size(); k++) {
    PointHessian* newpoint = optimized[k];
    ImmaturePoint* ph = toOptimize[k];

    if(newpoint != 0 && newpoint != (PointHessian*)((long)(-1))) {
      newpoint->host->immaturePoints[ph->idxInImmaturePoints]=0;
      newpoint->host->pointHessians.push_back(newpoint);

      /// insert point in energy function
      ef->insertPoint(newpoint);

      /// insert residual in energy function.
      for(PointFrameResidual* r : newpoint->residuals)
        ef->insertResidual(r);

      assert(newpoint->efPoint != 0);
      delete ph;
    }
    else if(newpoint == (PointHessian*)((long)(-1)) || ph->lastTraceStatus==IPS_OOB) {
      // bug: the original order is wrong.
      ph->host->immaturePoints[ph->idxInImmaturePoints]=0;
      delete ph;
    }
    else {
      assert(newpoint == 0 || newpoint == (PointHessian*)((long)(-1)));
    }
  }

  /// STEP5: throw away the deleted points.
  for(FrameHessian* host : frameHessians) {
    for(int i=0;i<(int)host->immaturePoints.size();i++) {
      if(host->immaturePoints[i]==0) {
        /// bug if the back is also empty.
        host->immaturePoints[i] = host->immaturePoints.back(); /// there is no order requirement, the last one is directly empty.
        host->immaturePoints.pop_back();
        i--;
      }
    }
  }
}

void FullSystem::activatePointsOldFirst() {
  assert(false);
}

/// mark the state of the point to be removed, marginalized or lost.
void FullSystem::flagPointsForRemoval() {
  assert(EFIndicesValid);

  std::vector<FrameHessian*> fhsToKeepPoints;
  std::vector<FrameHessian*> fhsToMargPoints;

  //if(setting_margPointVisWindow>0)
  /// bug is another statement that is not used.
  {
    for(int i=((int)frameHessians.size())-1;i>=0 && i >= ((int)frameHessians.size());i--)
      if(!frameHessians[i]->flaggedForMarginalization)
        fhsToKeepPoints.push_back(frameHessians[i]);

    for(int i=0; i< (int)frameHessians.size();i++)
      if(frameHessians[i]->flaggedForMarginalization)
        fhsToMargPoints.push_back(frameHessians[i]);
  }

  //ef->setAdjointsF();
  //ef->setDeltaF(&Hcalib);
  int flag_oob=0, flag_in=0, flag_inin=0, flag_nores=0;

  // go through all active frames
  for(FrameHessian* host : frameHessians) {
    for(unsigned int i=0; i < host->pointHessians.size(); i++) {
      PointHessian* ph = host->pointHessians[i];

      if(ph==0) {
        continue;
      }

      /// throw away behind the camera, no residual points.
      if(ph->idepth_scaled < 0 || ph->residuals.size()==0) {
        host->pointHessiansOut.push_back(ph);
        ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
        host->pointHessians[i]=0;
        flag_nores++;
      }
      /// mark the points on the marginalized frame an the points that are more affected as marginalized or delete.
      else if(ph->isOOB(fhsToKeepPoints, fhsToMargPoints) || host->flaggedForMarginalization) {
        flag_oob++;

        /// if it is an internal point, linearize the residual in the current state and calculate the residual to zero.
        if(ph->isInlierNew()) {
          flag_in++;

          int ngoodRes=0;
          for(PointFrameResidual* r : ph->residuals) {
            r->resetOOB();
            r->linearize(&Hcalib);
            r->efResidual->isLinearized = false;
            r->applyRes(true);

            /// if it is an active (can participate in optimization) residual, fix it and calculate res_toZeroF
            if(r->efResidual->isActive()) {
              r->efResidual->fixLinearizationF(ef);
              ngoodRes++;
            }
          }

          /// if the covariance of the idepth is large, throw it away and small marginalize.
          if(ph->idepth_hessian > setting_minIdepthH_marg) {
            flag_inin++;
            ph->efPoint->stateFlag = EFPointStatus::PS_MARGINALIZE;
            host->pointHessiansMarginalized.push_back(ph);
          }
          else {
            ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
            host->pointHessiansOut.push_back(ph);
          }
        }
        /// not throw away the inner point.
        else {
          host->pointHessiansOut.push_back(ph);
          ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
          //printf("drop point in frame %d (%d goodRes, %d activeRes)\n", ph->host->idx, ph->numGoodResiduals, (int)ph->residuals.size());
        }
        host->pointHessians[i]=0;
      }
    }

    /// delete marginalized or deleted points.
    for(int i=0; i<(int)host->pointHessians.size(); i++) {
      if(host->pointHessians[i] == 0) {
        host->pointHessians[i] = host->pointHessians.back();
        host->pointHessians.pop_back();
        i--;
      }
    }

  }
}

void FullSystem::addActiveFrame( ImageAndExposure* image, ImageAndExposure* image_right, int id ) {
  if(isLost)
    return;

  /// STEP1: lock the tacking thread.
  boost::unique_lock<boost::mutex> lock(trackMutex);

  /// STEP2: create framehessian and frameshell, initialize them accordingly, and store all frames.
  // =========================== add into allFrameHistory =========================
  FrameHessian* fh = new FrameHessian();
  FrameHessian* fh_right = new FrameHessian();
  FrameShell* shell = new FrameShell();

  shell->camToWorld = SE3(); 		// no lock required, as fh is not used anywhere yet.
  shell->aff_g2l = AffLight(0,0);
  shell->marginalizedAt = shell->id = allFrameHistory.size();
  shell->timestamp = image->timestamp;
  shell->incoming_id = id; // id passed into DSO
  fh->shell = shell;
  fh_right->shell=shell;
  allFrameHistory.push_back(shell); /// save only a brief shell.

  /// STEP3: get the exposure time, generate a pyramid, and calculate the entire image gradient.
  // =========================== make Images / derivatives etc. =========================
  fh->ab_exposure = image->exposure_time;
  fh->makeImages(image->image, &Hcalib);
  fh_right->ab_exposure = image_right->exposure_time;
  fh_right->makeImages(image_right->image,&Hcalib);

  /// STEP4: initialization.
  if(!initialized) {
    // use initializer!
    /// STEP4.1: add the first frame.
    // first frame set. fh is kept by coarseInitializer.
    if(coarseInitializer->frameID<0) {
      coarseInitializer->setFirstStereo(&Hcalib, fh,fh_right);
      initialized=true;
    }
    return;
  }
  // do front-end operation.
  else {
    /// STEP5: tracking the new frame, get the pose photometric, and determine the tracking status.
    // =========================== SWAP tracking reference?. =========================
    if(coarseTracker_forNewKF->refFrameID > coarseTracker->refFrameID) {
      boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);

      /// exchange the reference frame and the current tracker's coarseTracker.
      CoarseTracker* tmp = coarseTracker;
      coarseTracker=coarseTracker_forNewKF;
      coarseTracker_forNewKF=tmp;
    }

    //// tres[0]: achievedRes[0] = sqrtf((float)(resOld[0] / resOld[1])) at LEVEL 0
    //// tres[1]: flowVecs[0] = sumSquaredShiftT/(sumSquaredShiftNum+0.1);    /// the size of the average pixel movement during pure translation.
    //// tres[2]: flowVecs[1] = 0
    //// tres[3]: flowVecs[2] = sumSquaredShiftRT/(sumSquaredShiftNum+0.1);   /// the size of the avaerage pixel movement during translation + rotation.
    Vec4 tres = trackNewCoarse(fh,fh_right);

    if(!std::isfinite((double)tres[0]) ||
       !std::isfinite((double)tres[1]) ||
       !std::isfinite((double)tres[2]) ||
       !std::isfinite((double)tres[3])) {
      printf("Initial Tracking failed: LOST!\n");
      isLost=true;
      return;
    }

    /// determine where to insert a keyframe.
    bool needToMakeKF = false;

    /// how ofen are keyframes inserted.
    if(setting_keyframesPerSecond > 0) {
      needToMakeKF = allFrameHistory.size()== 1 ||
                     (fh->shell->timestamp - allKeyFramesHistory.back()->timestamp) > 0.95f/setting_keyframesPerSecond;
    }
    else {
      Vec2 refToFh = AffLight::fromToVecExposure(coarseTracker->lastRef->ab_exposure,
                                                 fh->ab_exposure,
                                                 coarseTracker->lastRef_aff_g2l,
                                                 fh->shell->aff_g2l);

      float delta = setting_kfGlobalWeight*setting_maxShiftWeightT *  sqrtf((double)tres[1]) / (wG[0]+hG[0]) +  /// translation pixel shift.
                    setting_kfGlobalWeight*setting_maxShiftWeightR *  sqrtf((double)tres[2]) / (wG[0]+hG[0]) +  /// TODO rotation pixel shift, set to 0 ???
                    setting_kfGlobalWeight*setting_maxShiftWeightRT * sqrtf((double)tres[3]) / (wG[0]+hG[0]) +  /// rotation + translation pixel shift.
                    setting_kfGlobalWeight*setting_maxAffineWeight * fabs(logf((float)refToFh[0]));  /// brightness changes

      //// comment out.
      // std::cout << "\tdelta: " << delta << std::endl;

      // BRIGHTNESS CHECK
      needToMakeKF = allFrameHistory.size()== 1 ||
                     delta > 1 ||
                     2*coarseTracker->firstCoarseRMSE < tres[0];  /// the error energy changes too much. (double the initial value.)
    }

    for(IOWrap::Output3DWrapper* ow : outputWrapper) {
      ow->publishCamPose(fh->shell, &Hcalib);
      //// semi-direct method.
      Twc_posterior_ = fh->shell->camToWorld;
    }

    /// STEP7: publish the frame.
    lock.unlock();
    deliverTrackedFrame(fh, fh_right, needToMakeKF);
    return;
  }
}

/// give the traced frame to the graphics thread, and set it as a keyframe or a non-keyframe.
void FullSystem::deliverTrackedFrame(FrameHessian* fh, FrameHessian* fh_right, bool needKF) {
  /// execute sequentially.
#if 0
  if(linearizeOperation) {
#else
    if(true) {
#endif
    std::cout << "[+] linearizeOperation " << std::endl;

    if(goStepByStep && lastRefStopID != coarseTracker->refFrameID) {
      MinimalImageF3 img(wG[0], hG[0], fh->dI);
      IOWrap::displayImage("frameToTrack", &img);

      while(true) {
        char k=IOWrap::waitKey(0);

        if(k==' ') {
          break;
        }

        handleKey(k);
      }
      lastRefStopID = coarseTracker->refFrameID;
    }
    else {
      handleKey(IOWrap::waitKey(1));
    }

    if(needKF) {
      makeKeyFrame(fh, fh_right);
    }
    else {
      makeNonKeyFrame(fh, fh_right);
    }
  }
  else {
    /// lock the track and map synchronization.
    boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
    unmappedTrackedFrames.push_back(fh);
    unmappedTrackedFrames_right.push_back(fh_right);

    if(needKF) {
      needNewKFAfter = fh->shell->trackingRef->id;
    }

    trackedFrameSignal.notify_all();

    while(coarseTracker_forNewKF->refFrameID == -1 && coarseTracker->refFrameID == -1) {
      /// when there is no tracked image, block trackMapSyncMutex until it is notified.
      mappedFrameSignal.wait(lock);
    }

    lock.unlock();
  }
}

void FullSystem::mappingLoop() {
  boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);

  while(runMapping) {
    while(unmappedTrackedFrames.size()==0) {
      /// no image waiting for trackedFrameSignal to wake up.
      trackedFrameSignal.wait(lock);

      if(!runMapping) {
        return;
      }
    }

    FrameHessian* fh = unmappedTrackedFrames.front();
    unmappedTrackedFrames.pop_front();
    FrameHessian* fh_right = unmappedTrackedFrames_right.front();
    unmappedTrackedFrames_right.pop_front();

    // guaranteed to make a KF for the very first two tracked frames.
    if(allKeyFramesHistory.size() <= 2) {
      lock.unlock();
      /// run makeKeyFrame will not affect unmappedTrackedFrames, so unlock.
      makeKeyFrame(fh, fh_right);
      lock.lock();
      /// wake up before end.
      mappedFrameSignal.notify_all();
      continue;
    }

    if(unmappedTrackedFrames.size() > 3)
      needToKetchupMapping=true;

    // if there are other frames to track, do that first.
    if(unmappedTrackedFrames.size() > 0) {
      lock.unlock();
      makeNonKeyFrame(fh, fh_right);
      lock.lock();

      /// too much to deal with.
      if(needToKetchupMapping && unmappedTrackedFrames.size() > 0) {
        FrameHessian* fh = unmappedTrackedFrames.front();
        unmappedTrackedFrames.pop_front();

        {
          boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
          assert(fh->shell->trackingRef != 0);
          fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
          fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);
        }
        delete fh;
        // delete fh_right;
     }
    }
    else {
      /// layer need keyframes.
      if(setting_realTimeMaxKF || needNewKFAfter >= frameHessians.back()->shell->id) {
        //// make new KF.
        lock.unlock();
        makeKeyFrame(fh, fh_right);
        needToKetchupMapping=false;
        lock.lock();
      }
      else {
        //// make new non-KF.
        lock.unlock();
        makeNonKeyFrame(fh, fh_right);
        lock.lock();
      }
    }
    mappedFrameSignal.notify_all();
  }

  printf("MAPPING FINISHED!\n");
}

void FullSystem::blockUntilMappingIsFinished() {
  boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
  runMapping = false;
  trackedFrameSignal.notify_all();
  lock.unlock();

  mappingThread.join();
}

/// set as non-keyframe.
void FullSystem::makeNonKeyFrame( FrameHessian* fh, FrameHessian* fh_right) {
  // needs to be set by mapping thread. no lock required since we are in mapping thread.
  {
    boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
    assert(fh->shell->trackingRef != 0);

    /// take out its current pose during mapping to get camToWorld.
    //// T_w_curr = T_w_ref * T_ref_curr
    fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;

    /// remove the estimated pose at this time.
    fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(), fh->shell->aff_g2l);
  }

  /// update the immature point (the point where the depth has no converged).
  traceNewCoarseNonKey(fh, fh_right);

  delete fh;
  delete fh_right;
}

/// keyframe generation, optimization, activation points, extraction points, marginalized keyframes.
void FullSystem::makeKeyFrame( FrameHessian* fh, FrameHessian* fh_right) {
  /// STEP1: set the pose and photometric parameters of the currently estimated fh.
  // needs to be set by mapping thread
  {
    boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
    assert(fh->shell->trackingRef != 0);
    fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
    fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(), fh->shell->aff_g2l);
  }

  /// STEP2: use this frame to update the immature point of the previous frame.
  /// update the immature points (the points where the depth has not converged)
  traceNewCoarseKey(fh, fh_right);

  boost::unique_lock<boost::mutex> lock(mapMutex);

  /// STEP3: select frames to be marginalized.
  // =========================== Flag Frames to be Marginalized. =========================
  flagFramesForMarginalization(fh);

  /// STEP4: add to keyframe sequence.
  // =========================== add New Frame to Hessian Struct. =========================
  fh->idx = frameHessians.size();

  //// Activated Keyframes. size is usually 8 fixed.
  frameHessians.push_back(fh);

  fh->frameID = allKeyFramesHistory.size();
  allKeyFramesHistory.push_back(fh->shell);
  ef->insertFrame(fh, &Hcalib);

  /// this will be run every time a keyframe is added to set the pose. set the pose linearization point.
  setPrecalcValues();

  /// STEP5: residuals of the previous keyframe and the current frame fh (old)
  // =========================== add new residuals for old points =========================
  int numFwdResAdde=0;

  // go through all active frames
  for(FrameHessian* fh1 : frameHessians) {
    if(fh1 == fh) continue;

    /// delete all after construction.
    for(PointHessian* ph : fh1->pointHessians) {
      /// create the residual b/w the current frame fh and the previous frame.
      PointFrameResidual* r = new PointFrameResidual(ph, fh1, fh);
      r->setState(ResState::IN);
      ph->residuals.push_back(r);
      ef->insertResidual(r);

      /// set the last residual.
      ph->lastResiduals[1] = ph->lastResiduals[0];
      /// the current setting is the previous one.
      ph->lastResiduals[0] = std::pair<PointFrameResidual*, ResState>(r, ResState::IN);
      numFwdResAdde+=1;
    }
  }

  /// STEP6: activate some immature points on all keyframes (construct new residuals)
  // =========================== Activate Points (& flag for marginalization). =========================
  activatePointsMT();
  /// ? why do you want to reset the ID? is it because of adding a new frame?
  ef->makeIDX();

  /// optimize the keyframes in the sliding window (easy to say, there are many problems in it)
  // =========================== OPTIMIZE ALL =========================
  /// aren't those two values the same???
  fh->frameEnergyTH = frameHessians.back()->frameEnergyTH;

  float rmse = optimize(setting_maxOptIterations);
  //// commented out.
  std::cout << "rmse: " << rmse << std::endl;

  // =========================== Figure Out if INITIALIZATION FAILED =========================
  if(allKeyFramesHistory.size() <= 4) {
    if(allKeyFramesHistory.size()==2 && rmse > 20*benchmark_initializerSlackFactor) {
      printf("[1] I THINK INITIALIZATINO FAILED! Resetting.\n");
      initFailed=true;
    }
    if(allKeyFramesHistory.size()==3 && rmse > 13*benchmark_initializerSlackFactor) {
      printf("[2] I THINK INITIALIZATINO FAILED! Resetting.\n");
      initFailed=true;
    }
    if(allKeyFramesHistory.size()==4 && rmse > 9*benchmark_initializerSlackFactor) {
      printf("[3] I THINK INITIALIZATINO FAILED! Resetting.\n");
      initFailed=true;
    }
  }

  /// the optimized energy function is too large, it is considered to be lost.
  if(isLost) {
    return;
  }

  /// STEP8: remove the outlier points and set the latest frame as the latest frame.
  // =========================== REMOVE OUTLIER =========================
  removeOutliers();

  {
    boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
    /// update intrinsic parameteres.
    coarseTracker_forNewKF->makeK(&Hcalib);

    //// set last Reference Keyframe into 'coarseTracker_forNewKF->lastRef'
    coarseTracker_forNewKF->setCoarseTrackingRef(frameHessians, fh_right, Hcalib);

    if(Twc_prior_.translation().norm() > 1.1) {
      std::cout << "[+] Got Twc_prior from OpenVSLAM!" << std::endl;
      fh->shell->camToWorld = Twc_prior_;
      coarseTracker_forNewKF->lastRef = fh;
    }

    //// plot color coded depth image.
    coarseTracker_forNewKF->debugPlotIDepthMap(&minIdJetVisTracker, &maxIdJetVisTracker, outputWrapper);
    coarseTracker_forNewKF->debugPlotIDepthMapFloat(outputWrapper);
  }

  /// STEP9: mark delete and marginalized points, and delete & marginalize.
  // =========================== (Activate-)Marginalize Points =========================
  flagPointsForRemoval();
  ef->dropPointsF();

  getNullspaces(ef->lastNullspaces_pose,
                ef->lastNullspaces_scale,
                ef->lastNullspaces_affA,
                ef->lastNullspaces_affB);

  ef->marginalizePointsF();


  /// STEP10: generate new points.
  // =========================== add new Immature points & new residuals =========================
  makeNewTraces(fh, fh_right, 0);

  for(IOWrap::Output3DWrapper* ow : outputWrapper) {
    ow->publishGraph(ef->connectivityMap);
    ow->publishKeyframes(frameHessians, false, &Hcalib);
  }

  /// STEP11: marginalize keyframes.
  /// marginalize a frame to delete or marginalize all points above.
  // =========================== Marginalize Frames =========================
  for(unsigned int i=0; i<frameHessians.size(); i++) {
    if(frameHessians[i]->flaggedForMarginalization) {
      marginalizeFrame(frameHessians[i]);
      i=0;
    }
  }

  delete fh_right;
  // printLogLine();
  // printEigenValLine();
}

/// extract information from initialization for tracking.
// insert the first Frame into FrameHessians
void FullSystem::initializeFromInitializer(FrameHessian* newFrame) {
  boost::unique_lock<boost::mutex> lock(mapMutex);

  Mat33f K = Mat33f::Identity();
  K(0,0) = Hcalib.fxl();
  K(1,1) = Hcalib.fyl();
  K(0,2) = Hcalib.cxl();
  K(1,2) = Hcalib.cyl();

  /// STEP1: set the first frame as a keyframe, join the queue, and add EnergyFunctional.
  // add firstframe.
  FrameHessian* firstFrame = coarseInitializer->firstFrame;   /// the first frame is added to the map.
  firstFrame->idx = frameHessians.size();                     /// assign it an id (starting at 0)
  frameHessians.push_back(firstFrame);                        /// keyframe container in the map.
  firstFrame->frameID = allKeyFramesHistory.size();           /// all historical keyframe ids.
  allKeyFramesHistory.push_back(firstFrame->shell);           /// all historical keyframes.
  ef->insertFrame(firstFrame, &Hcalib);
  setPrecalcValues();                                         /// set relative pose pre-calculated values.

  FrameHessian* firstFrameRight = coarseInitializer->firstRightFrame;
  frameHessiansRight.push_back(firstFrameRight);

  firstFrame->pointHessians.reserve(wG[0]*hG[0]*0.2f);             /// the number of points 20% of image size.
  firstFrame->pointHessiansMarginalized.reserve(wG[0]*hG[0]*0.2f); /// marginalization.
  firstFrame->pointHessiansOut.reserve(wG[0]*hG[0]*0.2f);          /// lost point.

  /// STEP2: find the average scale factor.
  float idepthStereo = 0;
  float sumID=1e-5, numID=1e-5;
  for(int i=0;i<coarseInitializer->numPoints[0];i++)
  {
    /// ? what is the value of iR
    sumID += coarseInitializer->points[0][i].iR;  /// equivalent to median value of point 0.
    numID++;
  }

  // randomly sub-select the points I need.
  /// target points / actual points
  float keepPercentage = setting_desiredPointDensity / coarseInitializer->numPoints[0];

  if(!setting_debugout_runquiet)
    printf("Initialization: keep %.1f%% (need %d, have %d)!\n", 100*keepPercentage,
           (int)(setting_desiredPointDensity), coarseInitializer->numPoints[0] );

  /// STEP3: create PointHessian, and keyframes, and add EnergyFunctional
  // initialize first frame by idepth computed by static stereo matching
  for(int i=0;i<coarseInitializer->numPoints[0];i++) {
    /// if fewer points are extracted, do not excute, if more points are extracted, randomly kill them.
    if(rand()/(float)RAND_MAX > keepPercentage) continue;

    Pnt* point = coarseInitializer->points[0]+i;
    ImmaturePoint* pt = new ImmaturePoint(point->u+0.5f, point->v+0.5f, firstFrame, point->my_type, &Hcalib);

    pt->u_stereo = pt->u;
    pt->v_stereo = pt->v;
    pt->idepth_min_stereo = 0;
    pt->idepth_max_stereo = NAN;

    pt->traceStereo(firstFrameRight, K, 1);

    pt->idepth_min = pt->idepth_min_stereo;
    pt->idepth_max = pt->idepth_max_stereo;
    idepthStereo = pt->idepth_stereo;


    if(!std::isfinite(pt->energyTH) ||
       !std::isfinite(pt->idepth_min) ||
       !std::isfinite(pt->idepth_max) ||
       pt->idepth_min < 0 || pt->idepth_max < 0)
    {
      delete pt;
      continue;
    }

    PointHessian* ph = new PointHessian(pt, &Hcalib);
    delete pt;
    if(!std::isfinite(ph->energyTH)) {delete ph; continue;}

    ph->setIdepthScaled(idepthStereo);
    ph->setIdepthZero(idepthStereo);
    ph->hasDepthPrior=true;
    ph->setPointStatus(PointHessian::ACTIVE); /// activate point.


    firstFrame->pointHessians.push_back(ph);
    ef->insertPoint(ph);
  }

  /// STEP4: set the amount to be optimized for the first and latest frames, reference frame.
  SE3 firstToNew = coarseInitializer->thisToNext;

  // really no lock required, as we are initializing.
  {
    boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
    firstFrame->shell->camToWorld = SE3();    /// empty initial value?
    firstFrame->shell->aff_g2l = AffLight(0,0);
    firstFrame->setEvalPT_scaled(firstFrame->shell->camToWorld.inverse(), firstFrame->shell->aff_g2l);
    firstFrame->shell->trackingRef=0;
    firstFrame->shell->camToTrackingRef = SE3();

    newFrame->shell->camToWorld = firstToNew.inverse();
    newFrame->shell->aff_g2l = AffLight(0,0);
    newFrame->setEvalPT_scaled(newFrame->shell->camToWorld.inverse(), newFrame->shell->aff_g2l);
    newFrame->shell->trackingRef = firstFrame->shell;
    newFrame->shell->camToTrackingRef = firstToNew.inverse();

  }

  initialized=true;
  printf("INITIALIZE FROM INITIALIZER (%d pts)!\n", (int)firstFrame->pointHessians.size());
}

/// extract new pixels for tracking.
void FullSystem::makeNewTraces(FrameHessian* newFrame, FrameHessian* newFrameRight, float* gtDepth) {
  /// bug is uesless
  pixelSelector->allowFast = true;
  //int numPointsTotal = makePixelStatus(newFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
  int numPointsTotal = pixelSelector->makeMaps(newFrame, selectionMap,setting_desiredImmatureDensity);

  newFrame->pointHessians.reserve(numPointsTotal*1.2f);
  //fh->pointHessiansInactive.reserve(numPointsTotal*1.2f);
  newFrame->pointHessiansMarginalized.reserve(numPointsTotal*1.2f);
  newFrame->pointHessiansOut.reserve(numPointsTotal*1.2f);

  for(int y=patternPadding+1;y<hG[0]-patternPadding-2;y++) {
    for(int x=patternPadding+1;x<wG[0]-patternPadding-2;x++) {
      int i = x+y*wG[0];
      if(selectionMap[i]==0) continue;

      ImmaturePoint* impt = new ImmaturePoint(x,y,newFrame, selectionMap[i], &Hcalib);

      /// the projection should not be an infinite number.
      if(!std::isfinite(impt->energyTH)) {
        delete impt;
      }
      else {
        newFrame->immaturePoints.push_back(impt);
      }
    }
  }

  printf("[+] made %d immature points!\n", (int)newFrame->immaturePoints.size());
}

/// calculate the pre-calculated value of frameHessian, and the delta value of the state.
/// set the relationshp b/w keyframes.
void FullSystem::setPrecalcValues() {
  /// pre-calculation container for each target frame, the size is the number of keyframes.
  for(FrameHessian* fh : frameHessians) {
    fh->targetPrecalc.resize(frameHessians.size());
    for(unsigned int i=0; i < frameHessians.size(); i++) {  /// ? and myself and my own ?
      /// calculate the conversion relationship b/w host and target.
      fh->targetPrecalc[i].set(fh, frameHessians[i], &Hcalib);
    }
  }

  ef->setDeltaF(&Hcalib);
}

void FullSystem::printLogLine() {
  if(frameHessians.size()==0) return;

  if(!setting_debugout_runquiet)
    printf("LOG %d: %.3f fine. Res: %d A, %d L, %d M; (%'d / %'d) forceDrop. a=%f, b=%f. Window %d (%d)\n",
           allKeyFramesHistory.back()->id,
           statistics_lastFineTrackRMSE,
           ef->resInA,
           ef->resInL,
           ef->resInM,
           (int)statistics_numForceDroppedResFwd,
           (int)statistics_numForceDroppedResBwd,
           allKeyFramesHistory.back()->aff_g2l.a,
           allKeyFramesHistory.back()->aff_g2l.b,
           frameHessians.back()->shell->id - frameHessians.front()->shell->id,
           (int)frameHessians.size());


  if(!setting_logStuff) return;

  if(numsLog != 0)
  {
    (*numsLog) << allKeyFramesHistory.back()->id << " "  <<
        statistics_lastFineTrackRMSE << " "  <<
        (int)statistics_numCreatedPoints << " "  <<
        (int)statistics_numActivatedPoints << " "  <<
        (int)statistics_numDroppedPoints << " "  <<
        (int)statistics_lastNumOptIts << " "  <<
        ef->resInA << " "  <<
        ef->resInL << " "  <<
        ef->resInM << " "  <<
        statistics_numMargResFwd << " "  <<
        statistics_numMargResBwd << " "  <<
        statistics_numForceDroppedResFwd << " "  <<
        statistics_numForceDroppedResBwd << " "  <<
        frameHessians.back()->aff_g2l().a << " "  <<
        frameHessians.back()->aff_g2l().b << " "  <<
        frameHessians.back()->shell->id - frameHessians.front()->shell->id << " "  <<
        (int)frameHessians.size() << " "  << "\n";
    numsLog->flush();
  }
}

void FullSystem::printEigenValLine() {
  if(!setting_logStuff) return;
  if(ef->lastHS.rows() < 12) return;


  MatXX Hp = ef->lastHS.bottomRightCorner(ef->lastHS.cols()-CPARS,ef->lastHS.cols()-CPARS);
  MatXX Ha = ef->lastHS.bottomRightCorner(ef->lastHS.cols()-CPARS,ef->lastHS.cols()-CPARS);
  int n = Hp.cols()/8;
  assert(Hp.cols()%8==0);

  // sub-select
  for(int i=0;i<n;i++)
  {
    MatXX tmp6 = Hp.block(i*8,0,6,n*8);
    Hp.block(i*6,0,6,n*8) = tmp6;

    MatXX tmp2 = Ha.block(i*8+6,0,2,n*8);
    Ha.block(i*2,0,2,n*8) = tmp2;
  }
  for(int i=0;i<n;i++)
  {
    MatXX tmp6 = Hp.block(0,i*8,n*8,6);
    Hp.block(0,i*6,n*8,6) = tmp6;

    MatXX tmp2 = Ha.block(0,i*8+6,n*8,2);
    Ha.block(0,i*2,n*8,2) = tmp2;
  }

  VecX eigenvaluesAll = ef->lastHS.eigenvalues().real();
  VecX eigenP = Hp.topLeftCorner(n*6,n*6).eigenvalues().real();
  VecX eigenA = Ha.topLeftCorner(n*2,n*2).eigenvalues().real();
  VecX diagonal = ef->lastHS.diagonal();

  std::sort(eigenvaluesAll.data(), eigenvaluesAll.data()+eigenvaluesAll.size());
  std::sort(eigenP.data(), eigenP.data()+eigenP.size());
  std::sort(eigenA.data(), eigenA.data()+eigenA.size());

  int nz = std::max(100,setting_maxFrames*10);

  if(eigenAllLog != 0)
  {
    VecX ea = VecX::Zero(nz); ea.head(eigenvaluesAll.size()) = eigenvaluesAll;
    (*eigenAllLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
    eigenAllLog->flush();
  }
  if(eigenALog != 0)
  {
    VecX ea = VecX::Zero(nz); ea.head(eigenA.size()) = eigenA;
    (*eigenALog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
    eigenALog->flush();
  }
  if(eigenPLog != 0)
  {
    VecX ea = VecX::Zero(nz); ea.head(eigenP.size()) = eigenP;
    (*eigenPLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
    eigenPLog->flush();
  }

  if(DiagonalLog != 0)
  {
    VecX ea = VecX::Zero(nz); ea.head(diagonal.size()) = diagonal;
    (*DiagonalLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
    DiagonalLog->flush();
  }

  if(variancesLog != 0)
  {
    VecX ea = VecX::Zero(nz); ea.head(diagonal.size()) = ef->lastHS.inverse().diagonal();
    (*variancesLog) << allKeyFramesHistory.back()->id << " " <<  ea.transpose() << "\n";
    variancesLog->flush();
  }

  std::vector<VecX> &nsp = ef->lastNullspaces_forLogging;
  (*nullspacesLog) << allKeyFramesHistory.back()->id << " ";
  for(unsigned int i=0;i<nsp.size();i++)
    (*nullspacesLog) << nsp[i].dot(ef->lastHS * nsp[i]) << " " << nsp[i].dot(ef->lastbS) << " " ;
  (*nullspacesLog) << "\n";
  nullspacesLog->flush();

}

void FullSystem::printFrameLifetimes() {
  if(!setting_logStuff) return;


  boost::unique_lock<boost::mutex> lock(trackMutex);

  std::ofstream* lg = new std::ofstream();
  lg->open("logs/lifetimeLog.txt", std::ios::trunc | std::ios::out);
  lg->precision(15);

  for(FrameShell* s : allFrameHistory)
  {
    (*lg) << s->id
          << " " << s->marginalizedAt
          << " " << s->statistics_goodResOnThis
          << " " << s->statistics_outlierResOnThis
          << " " << s->movedByOpt;



    (*lg) << "\n";
  }


  lg->close();
  delete lg;
}

void FullSystem::printEvalLine() {
  return;
}

} // namespace dso
