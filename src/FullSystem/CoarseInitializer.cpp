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

#include "FullSystem/CoarseInitializer.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/ImmaturePoint.h"
#include "util/nanoflann.h"

namespace dso {

CoarseInitializer::CoarseInitializer(int ww, int hh) : thisToNext_aff(0,0), thisToNext(SE3()) {
  for(int lvl=0; lvl<pyrLevelsUsed; lvl++) {
    int wl = ww>>lvl;
    int hl = hh>>lvl;
    points[lvl] = 0;
    numPoints[lvl] = 0;
    idepth[lvl] = new float[wl*hl];
  }

  JbBuffer = new Vec10f[ww*hh];
  JbBuffer_new = new Vec10f[ww*hh];

  frameID=-1;
  fixAffine=true;
  printDebug=false;

  wM.diagonal()[0] = wM.diagonal()[1] = wM.diagonal()[2] = SCALE_XI_ROT;
  wM.diagonal()[3] = wM.diagonal()[4] = wM.diagonal()[5] = SCALE_XI_TRANS;
  wM.diagonal()[6] = SCALE_A;
  wM.diagonal()[7] = SCALE_B;
}

CoarseInitializer::~CoarseInitializer() {
  for(int lvl=0; lvl<pyrLevelsUsed; lvl++) {
    if(points[lvl] != 0) delete[] points[lvl];
    delete[] idepth[lvl];
  }

  delete[] JbBuffer;
  delete[] JbBuffer_new;
}

//// never used function in stereo-dso.
bool CoarseInitializer::trackFrame(FrameHessian* newFrameHessian, FrameHessian* newFrameHessian_Right, std::vector<IOWrap::Output3DWrapper*> &wraps) {
  newFrame = newFrameHessian;

  /// STEP1: displace the new frame first.
  for(IOWrap::Output3DWrapper* ow : wraps) {
    ow->pushStereoLiveFrame(newFrameHessian,newFrameHessian_Right);
  }

  int maxIterations[] = {5,5,10,30,50};

  /// tuning parameters.
  alphaK = 2.5*2.5;   //*freeDebugParam1*freeDebugParam1;
  alphaW = 150*150;   //*freeDebugParam2*freeDebugParam2;
  regWeight = 0.8;    //*freeDebugParam4;
  couplingWeight = 1; //*freeDebugParam5;

  /// STEP2: initialize the idepth of each point to 1, initialize the photometric parameters, pose SE3
  /// snapped should mean that the displacement is large enough, re-optimize.
  if(!snapped) {
    /// initialize.
    thisToNext.translation().setZero();

    for(int lvl=0;lvl<pyrLevelsUsed;lvl++) {
      int npts = numPoints[lvl];
      Pnt* ptsl = points[lvl];

      for(int i=0;i<npts;i++) {
        ptsl[i].iR = 1;
        ptsl[i].idepth_new = 1;
        ptsl[i].lastHessian = 0;
      }
    }
  }

  SE3 refToNew_current = thisToNext;
  AffLight refToNew_aff_current = thisToNext_aff;

  /// if there are affine coefficients, estimate an initial value.
  if(firstFrame->ab_exposure>0 && newFrame->ab_exposure>0) {
    refToNew_aff_current = AffLight(logf(newFrame->ab_exposure /  firstFrame->ab_exposure),0); // coarse approximation.
  }

  Vec3f latestRes = Vec3f::Zero();

  /// estimated from the top level.
  for(int lvl=pyrLevelsUsed-1; lvl>=0; lvl--) {
    /// STEP3: use the calcualted previous level to initialize the next level.
    if(lvl<pyrLevelsUsed-1) {
      propagateDown(lvl+1);
    }

    Mat88f H,Hsc;
    Vec8f b,bsc;
    resetPoints(lvl);  /// initialize the top level here.

    /// STEP4: calculate energy before iteration, Hessian, etc.
    Vec3f resOld = calcResAndGS(lvl, H, b, Hsc, bsc, refToNew_current, refToNew_aff_current, false);
    applyStep(lvl);   /// new energy is paid to the old one.

    float lambda = 0.1;
    float eps = 1e-4;
    int fails=0;

    /// initial information.
    if(printDebug) {
      printf("lvl %d, it %d (l=%f) %s: %.3f+%.5f -> %.3f+%.5f (%.3f->%.3f) (|inc| = %f)! \t",
             lvl, 0, lambda,
             "INITIA",
             sqrtf((float)(resOld[0] / resOld[2])),   /// chi-square (res * res) average
             sqrtf((float)(resOld[1] / resOld[2])),   /// idepth energy average.
             sqrtf((float)(resOld[0] / resOld[2])),
             sqrtf((float)(resOld[1] / resOld[2])),
             (resOld[0]+resOld[1]) / resOld[2],
             (resOld[0]+resOld[1]) / resOld[2],
             0.0f);
      std::cout << refToNew_current.log().transpose() << " AFF " << refToNew_aff_current.vec().transpose() <<"\n";
    }

    /// STEP5: iterative solution.
    int iteration=0;
    while(true) {
      /// STEP5.1: calculate the marginalized hessian matrix, and some operations.
      Mat88f Hl = H;

      /// this is not LM.
      for(int i=0;i<8;i++) {
        Hl(i,i) *= (1+lambda);
      }

      /// Schur complement, marginalize the idepth state.
      /// beacuse dd must be diagonal, so also multiply the reciprocal.
      Hl -= Hsc*(1/(1+lambda));
      Vec8f bl = b - bsc*(1/(1+lambda));

      /// ? why does wM multiply, it corresponds to the state SCALE.
      /// ? (0.01f / (w[lvl] * h[lvl])) is to reduuce the value and be more stable?
      Hl = wM * Hl * wM * (0.01f/(w[lvl]*h[lvl]));
      bl = wM * bl * (0.01f/(w[lvl]*h[lvl]));

      /// STEP5.2: solve the increment.
      Vec8f inc;

      /// fixed photometric params.
      if(fixAffine) {
        inc.head<6>() = -(wM.toDenseMatrix().topLeftCorner<6,6>() * (Hl.topLeftCorner<6,6>().ldlt().solve(bl.head<6>())));
        inc.tail<2>().setZero();
      }
      else {
        inc = -(wM * (Hl.ldlt().solve(bl)));	//=-H^-1 * b.
      }

      /// STEP5.3: update status, update idepth in doStep.
      SE3 refToNew_new = SE3::exp(inc.head<6>().cast<double>()) * refToNew_current;
      AffLight refToNew_aff_new = refToNew_aff_current;
      refToNew_aff_new.a += inc[6];
      refToNew_aff_new.b += inc[7];
      doStep(lvl, lambda, inc);

      /// STEP5.4: calculate the updated energy and compare it with the old one to determine where it is accept.
      Mat88f H_new, Hsc_new; Vec8f b_new, bsc_new;
      Vec3f resNew = calcResAndGS(lvl, H_new, b_new, Hsc_new, bsc_new, refToNew_new, refToNew_aff_new, false);
      Vec3f regEnergy = calcEC(lvl);

      float eTotalNew = (resNew[0] + resNew[1] + regEnergy[1]);
      float eTotalOld = (resOld[0] + resOld[1] + regEnergy[0]);

      bool accept = eTotalOld > eTotalNew;

      if(printDebug) {
        printf("lvl %d, it %d (l=%f) %s: %.5f + %.5f + %.5f -> %.5f + %.5f + %.5f (%.2f->%.2f) (|inc| = %f)! \t",
               lvl, iteration, lambda,
               (accept ? "ACCEPT" : "REJECT"),
               sqrtf((float)(resOld[0] / resOld[2])),
               sqrtf((float)(regEnergy[0] / regEnergy[2])),
               sqrtf((float)(resOld[1] / resOld[2])),
               sqrtf((float)(resNew[0] / resNew[2])),
               sqrtf((float)(regEnergy[1] / regEnergy[2])),
               sqrtf((float)(resNew[1] / resNew[2])),
               eTotalOld / resNew[2],
               eTotalNew / resNew[2],
               inc.norm());
        std::cout << refToNew_new.log().transpose() << " AFF " << refToNew_aff_new.vec().transpose() <<"\n";
      }

      /// STEP5.5: if accepted, update the status; if not accepted, increate the lambda.
      if(accept) {
        if(resNew[1] == alphaK*numPoints[lvl]) {
          snapped = true;
        }

        H = H_new;
        b = b_new;
        Hsc = Hsc_new;
        bsc = bsc_new;
        resOld = resNew;
        refToNew_aff_current = refToNew_aff_new;
        refToNew_current = refToNew_new;
        applyStep(lvl);
        optReg(lvl);
        lambda *= 0.5;
        fails=0;

        if(lambda < 0.0001) {
          lambda = 0.0001;
        }
      }
      else {
        fails++;
        lambda *= 4;
        if(lambda > 10000) {
          lambda = 10000;
        }
      }

      bool quitOpt = false;

      /// iteration stop condition, convergence / greater than the maximum number of times / failure more than 2 times.
      if(!(inc.norm() > eps) || iteration >= maxIterations[lvl] || fails >= 2) {
        Mat88f H,Hsc;
        Vec8f b,bsc;
        quitOpt = true;
      }

      if(quitOpt) {
        break;
      }

      iteration++;
    }

    latestRes = resOld;
  }

  /// STEP6: assign the pose after optimization, and calculate the depth of the upper poiont from the bottom.
  thisToNext = refToNew_current;
  thisToNext_aff = refToNew_aff_current;

  for(int i=0; i<pyrLevelsUsed-1; i++) {
    propagateUp(i);
  }

  frameID++;
  //printf("frameID is %d \n", frameID);

  if(!snapped) {
    snappedAt=0;
  }

  if(snapped && snappedAt==0) {
    /// number of frames shifted enough.
    snappedAt = frameID;
  }

  debugPlot(0,wraps);

  /// the displacement is large enough, and then optimize 5 times.
  return snapped && frameID > snappedAt+1;
}

void CoarseInitializer::debugPlot(int lvl, std::vector<IOWrap::Output3DWrapper*> &wraps) {
  bool needCall = false;
  for(IOWrap::Output3DWrapper* ow : wraps)
    needCall = needCall || ow->needPushDepthImage();
  if(!needCall) return;


  int wl = w[lvl], hl = h[lvl];
  Eigen::Vector3f* colorRef = firstFrame->dIp[lvl];

  MinimalImageB3 iRImg(wl,hl);

  for(int i=0;i<wl*hl;i++)
    iRImg.at(i) = Vec3b(colorRef[i][0],colorRef[i][0],colorRef[i][0]);


  int npts = numPoints[lvl];

  float nid = 0, sid=0;
  for(int i=0;i<npts;i++)
  {
    Pnt* point = points[lvl]+i;
    if(point->isGood)
    {
      nid++;
      sid += point->iR;
    }
  }
  float fac = nid / sid;



  for(int i=0;i<npts;i++)
  {
    Pnt* point = points[lvl]+i;

    if(!point->isGood)
      iRImg.setPixel9(point->u+0.5f,point->v+0.5f,Vec3b(0,0,0));

    else
      iRImg.setPixel9(point->u+0.5f,point->v+0.5f,makeRainbow3B(point->iR*fac));
  }


  //IOWrap::displayImage("idepth-R", &iRImg, false);
  for(IOWrap::Output3DWrapper* ow : wraps)
    ow->pushDepthImage(&iRImg);
}

/// calculate energy function and hessian matrix, and schur complement, sc represents Schur.
// calculates residual, Hessian and Hessian-block neede for re-substituting depth.
Vec3f CoarseInitializer::calcResAndGS(int lvl, Mat88f &H_out, Vec8f &b_out,
                                      Mat88f &H_out_sc, Vec8f &b_out_sc,
                                      SE3 refToNew, AffLight refToNew_aff,
                                      bool plot)
{
  int wl = w[lvl], hl = h[lvl];

  /// current level image and graident.
  Eigen::Vector3f* colorRef = firstFrame->dIp[lvl];
  Eigen::Vector3f* colorNew = newFrame->dIp[lvl];

  /// rotation matrix R * inverse of intrinsic matrix K_inv.
  Mat33f RKi = (refToNew.rotationMatrix() * Ki[lvl]).cast<float>();
  /// pan
  Vec3f t = refToNew.translation().cast<float>();
  /// photomatrix params.
  Eigen::Vector2f r2new_aff = Eigen::Vector2f(exp(refToNew_aff.a), refToNew_aff.b);

  /// camera params for this level.
  float fxl = fx[lvl];
  float fyl = fy[lvl];
  float cxl = cx[lvl];
  float cyl = cy[lvl];

  /// 1*1 accumulator.
  Accumulator11 E;
  /// initial value, allocate space.
  acc9.initialize();

  E.initialize();

  int npts = numPoints[lvl];
  Pnt* ptsl = points[lvl];

  for(int i=0; i<npts; i++) {
    Pnt* point = ptsl+i;
    point->maxstep = 1e10;

    /// if it is not good.
    if(!point->isGood) {
      E.updateSingle((float)(point->energy[0]));  /// accumulate
      point->energy_new = point->energy;
      point->isGood_new = false;
      continue;
    }

    /// 8x1 matrix, the number of residuals near each point is 8.
    EIGEN_ALIGN16 VecNRf dp0;
    EIGEN_ALIGN16 VecNRf dp1;
    EIGEN_ALIGN16 VecNRf dp2;
    EIGEN_ALIGN16 VecNRf dp3;
    EIGEN_ALIGN16 VecNRf dp4;
    EIGEN_ALIGN16 VecNRf dp5;
    EIGEN_ALIGN16 VecNRf dp6;
    EIGEN_ALIGN16 VecNRf dp7;
    EIGEN_ALIGN16 VecNRf dd;
    EIGEN_ALIGN16 VecNRf r;
    JbBuffer_new[i].setZero();  /// 10x1 vector.

    // sum over all residuals.
    bool isGood = true;
    float energy=0;
    for(int idx=0; idx<patternNum; idx++) {
      /// pattern offset.
      int dx = patternP[idx][0];
      int dy = patternP[idx][1];

      /// Pj '= R*(x/z, y/z, 1) + t/z, transform to a new point. the depth still uses the host frame!.
      Vec3f pt = RKi * Vec3f(point->u+dx, point->v+dy, 1) + t*point->idepth_new;
      /// normarlized corrdinates Pj
      float u = pt[0] / pt[2];
      float v = pt[1] / pt[2];
      /// pixel coordinates pj
      float Ku = fxl * u + cxl;
      float Kv = fyl * v + cyl;

      /// dpi/pz'
      float new_idepth = point->idepth_new/pt[2];  /// idepth on a new frame.

      /// it is near the image border, the depth is less than 0, it is not good
      if(!(Ku > 1 && Kv > 1 && Ku < wl-2 && Kv < hl-2 && new_idepth > 0)) {
        isGood = false;
        break;
      }

      /// interpolate to get patch pixel values in the new image, (input 3D, output 3D pixel values + x-direction gradient + y-direction gradient)
      Vec3f hitColor = getInterpolatedElement33(colorNew, Ku, Kv, wl);
      //Vec3f hitColor = getInterpolatedElement33BiCub(colorNew, Ku, Kv, wl);

      /// reference the pixel value on the patch on the reference frame, and output the one-dimensional pixel value.
      //float rlR = colorRef[point->u+dx + (point->v+dy) * wl][0];
      float rlR = getInterpolatedElement31(colorRef, point->u+dx, point->v+dy, wl);

      /// if pixel values are finite, good.
      if(!std::isfinite(rlR) || !std::isfinite((float)hitColor[0])) {
        isGood = false;
        break;
      }

      /// residual
      float residual = hitColor[0] - r2new_aff[0] * rlR - r2new_aff[1];

      /// huber weights
      float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

      /// huberweight * (2-huberweight) = Objective Function
      /// relationship b/w robust weights and functions.
      energy += hw*residual*residual*(2 - hw);

      /// Pj differentiates idepth idi.
      /// 1/pz * (tx-u * tz), u = px/pz
      float dxdd = (t[0]-t[2]*u)/pt[2];

      /// 1/pz * (ty-v * tz), v = py/pz
      float dydd = (t[1]-t[2]*v)/pt[2];

      /// ? why open root
      /// ! robust kernel function is equivalent to weighted least squares.
      if(hw < 1)
        hw = sqrtf(hw);

      /// dxfx, dyfy
      float dxInterp = hw*hitColor[1]*fxl;
      float dyInterp = hw*hitColor[2]*fyl;

      /// derivative of j (new state) pose by residuals.
      dp0[idx] = new_idepth*dxInterp;                    ///  dpi/pz' * dxfx
      dp1[idx] = new_idepth*dyInterp;                    ///  dpi/pz' * dyfy
      dp2[idx] = -new_idepth*(u*dxInterp + v*dyInterp);  /// -dpi/pz' * (px'/pz' * dxfx+py'/pz' * dyfy)
      dp3[idx] = -u*v*dxInterp - (1+v*v)*dyInterp;       /// -px'py'/pz'^2 * dxfy - (1 + py'^2 / pz'^2 * dyfy)
      dp4[idx] = (1+u*u)*dxInterp + u*v*dyInterp;        /// (1 + px'^2/pz'^2) * dxfx + px'py'/pz'^2 * dxfy
      dp5[idx] = -v*dxInterp + u*dyInterp;               /// -py'/pz' * dxfx + px'/pz' * dyfy

      /// derivative of photometric parameters with residulas
      dp6[idx] = - hw*r2new_aff[0] * rlR;                /// exp(aj-ai) * I(pi)
      dp7[idx] = - hw*1;                                 /// for b

      /// idepth derivative of i (old state) for residuals
      dd[idx] = dxInterp * dxdd  + dyInterp * dydd;      /// dxfx*1/pz * (tx-u *tz) + dyfy*1/pz*(tx-u * tz)
      r[idx] = hw*residual;                              /// residual error res

      /// derivative of pixel error to idepth, modulo inverse.
      float maxstep = 1.0f / Vec2f(dxdd*fxl, dydd*fyl).norm(); /// ? why do yo set?
      if(maxstep < point->maxstep) point->maxstep = maxstep;

      // immediately compute dp*dd' and dd*dd' in JbBuffer1.
      /// calculate the first row (column) of hessian, and Jr's row about inverse depth.
      /// used to calculate Schur complement.
      JbBuffer_new[i][0] += dp0[idx]*dd[idx];
      JbBuffer_new[i][1] += dp1[idx]*dd[idx];
      JbBuffer_new[i][2] += dp2[idx]*dd[idx];
      JbBuffer_new[i][3] += dp3[idx]*dd[idx];
      JbBuffer_new[i][4] += dp4[idx]*dd[idx];
      JbBuffer_new[i][5] += dp5[idx]*dd[idx];
      JbBuffer_new[i][6] += dp6[idx]*dd[idx];
      JbBuffer_new[i][7] += dp7[idx]*dd[idx];
      JbBuffer_new[i][8] += r[idx]*dd[idx];
      JbBuffer_new[i][9] += dd[idx]*dd[idx];
    }

    /// if the point's pattern (one of the pixels) exceeds the image, the pixel value is infinite, or the residual is greater than the threshold.
    if(!isGood || energy > point->outlierTH*20) {
      E.updateSingle((float)(point->energy[0]));  /// add the previous frame.
      point->isGood_new = false;
      point->energy_new = point->energy;          /// last time grives the current time.
      continue;
    }

    /// inner points are added to the energy function
    // add into energy.
    E.updateSingle(energy);
    point->isGood_new = true;
    point->energy_new[0] = energy;

    /// because using 128 bits is equivalent to adding 4 number at a times, so i += 4.
    // update Hessian matrix.
    for(int i=0; i+3<patternNum; i+=4)
      acc9.updateSSE(_mm_load_ps(((float*)(&dp0))+i),
                     _mm_load_ps(((float*)(&dp1))+i),
                     _mm_load_ps(((float*)(&dp2))+i),
                     _mm_load_ps(((float*)(&dp3))+i),
                     _mm_load_ps(((float*)(&dp4))+i),
                     _mm_load_ps(((float*)(&dp5))+i),
                     _mm_load_ps(((float*)(&dp6))+i),
                     _mm_load_ps(((float*)(&dp7))+i),
                     _mm_load_ps(((float*)(&r))+i));

    /// add the extra value after 0, 4, 8 because SSE2 is added in 128 units, the extra is added separately.
    for(int i=((patternNum>>2)<<2); i < patternNum; i++)
      acc9.updateSingle((float)dp0[i],(float)dp1[i],(float)dp2[i],(float)dp3[i],
                        (float)dp4[i],(float)dp5[i],(float)dp6[i],(float)dp7[i],
                        (float)r[i]);
  }

  E.finish();
  acc9.finish();

  /// ? what are you doing?

  // calculate alpha energy, and decide if we cap it.
  Accumulator11 EAlpha;
  EAlpha.initialize();

  for(int i=0;i<npts;i++) {
    Pnt* point = ptsl+i;

    /// if point is not good.
    if(!point->isGood_new) {
      /// this is intentionally written again, useless code.
      E.updateSingle((float)(point->energy[1]));
    }
    else {
      /// the initial initalization is 1.
      point->energy_new[1] = (point->idepth_new-1)*(point->idepth_new-1); /// ? what is the principle?
      E.updateSingle((float)(point->energy_new[1]));
    }
  }

  EAlpha.finish(); /// just calculate if the displacement is large enough.
  float alphaEnergy = alphaW*(EAlpha.A + refToNew.translation().squaredNorm() * npts); /// the larger the translation, the easier it is to initalize successfully?

  //printf("AE = %f * %f + %f\n", alphaW, EAlpha.A, refToNew.translation().squaredNorm() * npts);

  // compute alpha opt.
  float alphaOpt;

  /// translation is greater than a certain value.
  if(alphaEnergy > alphaK*npts) {
    alphaOpt = 0;
    alphaEnergy = alphaK*npts;
  }
  else {
    alphaOpt = alphaW;
  }

  acc9SC.initialize();
  for(int i=0; i < npts; i++) {
    Pnt* point = ptsl+i;

    if(!point->isGood_new) {
      continue;
    }

    /// for idepth dd*dd;
    point->lastHessian_new = JbBuffer_new[i][9];

    /// ? what is this? weight the idepth value? normalize the depth value?
    /// energy adds (d-1)*(d-1), so dd = 1, r += (d-1)
    JbBuffer_new[i][8] += alphaOpt * (point->idepth_new - 1);  /// r * dd
    JbBuffer_new[i][9] += alphaOpt;                            /// derivative for idepth is 1 / dd*dd

    if(alphaOpt==0) {
      JbBuffer_new[i][8] += couplingWeight*(point->idepth_new - point->iR);
      JbBuffer_new[i][9] += couplingWeight;
    }

    /// inverse is covariance, weight
    JbBuffer_new[i][9] = 1/(1+JbBuffer_new[i][9]);

    /// 9 is the weight, the Schur complement is calculated!
    /// dp * dd * (dd^2)^-1 *dd * dp
    acc9SC.updateSingleWeighted((float)JbBuffer_new[i][0],(float)JbBuffer_new[i][1],(float)JbBuffer_new[i][2],(float)JbBuffer_new[i][3],
                                (float)JbBuffer_new[i][4],(float)JbBuffer_new[i][5],(float)JbBuffer_new[i][6],(float)JbBuffer_new[i][7],
                                (float)JbBuffer_new[i][8],(float)JbBuffer_new[i][9]);
  }
  acc9SC.finish();


  //printf("nelements in H: %d, in E: %d, in Hsc: %d / 9!\n", (int)acc9.num, (int)E.num, (int)acc9SC.num*9);
  H_out = acc9.H.topLeftCorner<8,8>();        /// acc9.num; dp^T*dp
  b_out = acc9.H.topRightCorner<8,1>();       /// acc9.num; dp^T*r
  H_out_sc = acc9SC.H.topLeftCorner<8,8>();   /// acc9.num; (dp*dd)^T*(dd*dd)^-1*(dd*dp)
  b_out_sc = acc9SC.H.topRightCorner<8,1>();  /// acc9.num; (dp*dd)^T*(dd*dd)^-1*(dp^T*r)

  /// ? what do you mean
  /// t * t * ntps
  /// add hessian to t, add a number to the diagonal, and add b to
  H_out(0,0) += alphaOpt*npts;
  H_out(1,1) += alphaOpt*npts;
  H_out(2,2) += alphaOpt*npts;

  Vec3f tlog = refToNew.log().head<3>().cast<float>(); /// lie algebra, translation part (last pose value)
  b_out[0] += tlog[0]*alphaOpt*npts;
  b_out[1] += tlog[1]*alphaOpt*npts;
  b_out[2] += tlog[2]*alphaOpt*npts;

  /// energy value? the number of points used.
  return Vec3f(E.A, alphaEnergy, E.num);
}

float CoarseInitializer::rescale() {
  float factor = 20*thisToNext.translation().norm();
  //	float factori = 1.0f/factor;
  //	float factori2 = factori*factori;
  //
  //	for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
  //	{
  //		int npts = numPoints[lvl];
  //		Pnt* ptsl = points[lvl];
  //		for(int i=0;i<npts;i++)
  //		{
  //			ptsl[i].iR *= factor;
  //			ptsl[i].idepth_new *= factor;
  //			ptsl[i].lastHessian *= factori2;
  //		}
  //	}
  //	thisToNext.translation() *= factori;

  return factor;
}

/// calculate the difference b/w the old and new idepth and iR, return the old difference, the new difference, the number
/// ? what is iR?
/// ! iR is the mean of idepth, and the scale converges to iR.
Vec3f CoarseInitializer::calcEC(int lvl) {
  if(!snapped) {
    return Vec3f(0,0,numPoints[lvl]);
  }

  AccumulatorX<2> E;
  E.initialize();

  int npts = numPoints[lvl];

  for(int i=0; i<npts; i++) {
    Pnt* point = points[lvl]+i;

    if(!point->isGood_new) {
      continue;
    }

    float rOld = (point->idepth - point->iR);
    float rNew = (point->idepth_new - point->iR);
    E.updateNoWeight(Vec2f(rOld*rOld, rNew*rNew)); /// sum

    //printf("%f %f %f!\n", point->idepth, point->idepth_new, point->iR);
  }
  E.finish();

  //printf("ER: %f %f %f!\n", couplingWeight*E.A1m[0], couplingWeight*E.A1m[1], (float)E.num.numIn1m);
  return Vec3f(couplingWeight*E.A1m[0], couplingWeight*E.A1m[1], E.num);
}

/// use the nearest point to update the iR of each point, the feeling of smooth
void CoarseInitializer::optReg(int lvl) {
  int npts = numPoints[lvl];
  Pnt* ptsl = points[lvl];

  /// if the displacement is not enough, set iR to 1
  if(!snapped) {
    for(int i=0; i<npts; i++)
      ptsl[i].iR = 1;
    return;
  }

  for(int i=0; i<npts; i++) {
    Pnt* point = ptsl+i;

    if(!point->isGood)
      continue;

    float idnn[10];
    int nnn=0;

    /// get the nearest 10 points around the current point, iR of good quality point.
    for(int j=0; j<10; j++) {
      if(point->neighbours[j] == -1)
        continue;

      Pnt* other = ptsl + point->neighbours[j];

      if(!other->isGood) continue;

      idnn[nnn] = other->iR;
      nnn++;
    }

    /// weighted with the nearest point median to get the new iR.
    if(nnn > 2) {
      /// get the median.
      std::nth_element(idnn, idnn+nnn/2, idnn+nnn);
      point->iR = (1-regWeight)*point->idepth + regWeight*idnn[nnn/2];
    }
  }
}

/// use normalized product to update high-level's idepth value.
void CoarseInitializer::propagateUp(int srcLvl) {
  assert(srcLvl+1<pyrLevelsUsed);
  // set idepth of target

  int nptss= numPoints[srcLvl];
  int nptst= numPoints[srcLvl+1];
  Pnt* ptss = points[srcLvl];
  Pnt* ptst = points[srcLvl+1];

  // set to zero.
  for(int i=0;i<nptst;i++) {
    Pnt* parent = ptst+i;
    parent->iR=0;
    parent->iRSumNum=0;
  }

  /// update the parent in the previous level.
  for(int i=0;i<nptss;i++) {
    Pnt* point = ptss+i;
    if(!point->isGood) continue;

    Pnt* parent = ptst + point->parent;
    /// mean * information matrix
    parent->iR += point->iR * point->lastHessian;
    /// new information matrix sigma
    parent->iRSumNum += point->lastHessian;
  }

  for(int i=0;i<nptst;i++) {
    Pnt* parent = ptst+i;

    if(parent->iRSumNum > 0) {
      /// mean value after gauss normalized product.
      parent->idepth = parent->iR = (parent->iR / parent->iRSumNum);
      parent->isGood = true;
    }
  }

  /// use nearby points to update iR and idepth.
  optReg(srcLvl+1);
}

// @ use upper level information to initialize lower level
// @param: current pyramid layer +1
// @note: cannot initialize the top-level value
void CoarseInitializer::propagateDown(int srcLvl) {
  assert(srcLvl>0);
  // set idepth of target

  int nptst= numPoints[srcLvl-1];   /// number of points in the current level.
  Pnt* ptss = points[srcLvl];       /// the current level+1, the point set of the previous level.
  Pnt* ptst = points[srcLvl-1];     /// point set of current level.

  for(int i=0; i<nptst; i++) {
    Pnt* point = ptst + i;                /// traverse the points of the current level.
    Pnt* parent = ptss + point->parent;   /// find the parent of the current point.

    if(!parent->isGood || parent->lastHessian < 0.1)
      continue;

    if(!point->isGood) {
      /// if the current point is not good, then give the value of the parent point directly to it. and set good
      point->iR = point->idepth = point->idepth_new = parent->iR;
      point->isGood=true;
      point->lastHessian=0;
    }
    else {
      /// get the new iR by weighting point and parent with hessian.
      /// iR can be seen as the depth value, the gauss normalized product used, hessian is the information matrix.
      float newiR = (point->iR*point->lastHessian*2 + parent->iR*parent->lastHessian) / (point->lastHessian*2+parent->lastHessian);
      point->iR = point->idepth = point->idepth_new = newiR;
    }
  }

  /// ? why iR is updated here, idepth is not updated.
  /// it feels more to consider the smoothing effect of nearby points.
  optReg(srcLvl-1); /// current layer.
}

/// calculate low level, high level, pixel value and gradient.
void CoarseInitializer::makeGradients(Eigen::Vector3f** data) {
  for(int lvl=1; lvl<pyrLevelsUsed; lvl++) {
    int lvlm1 = lvl-1;
    int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

    Eigen::Vector3f* dINew_l = data[lvl];
    Eigen::Vector3f* dINew_lm = data[lvlm1];

    /// use the previous level to get the value of current level.
    for(int y=0; y<hl; y++)
      for(int x=0; x<wl; x++)
        dINew_l[x + y*wl][0] = 0.25f * (dINew_lm[2*x   + 2*y*wlm1][0] +
                                        dINew_lm[2*x+1 + 2*y*wlm1][0] +
                                        dINew_lm[2*x   + 2*y*wlm1+wlm1][0] +
                                        dINew_lm[2*x+1 + 2*y*wlm1+wlm1][0]);

    /// calculate gradient based on pixels.
    for(int idx=wl;idx < wl*(hl-1);idx++) {
      dINew_l[idx][1] = 0.5f*(dINew_l[idx+1][0] - dINew_l[idx-1][0]);
      dINew_l[idx][2] = 0.5f*(dINew_l[idx+wl][0] - dINew_l[idx-wl][0]);
    }
  }
}

// set first frame
void CoarseInitializer::setFirstStereo(CalibHessian* HCalib,
                                       FrameHessian* newFrameHessian,
                                       FrameHessian* newFrameHessian_Right) {

  /// STEP1: calculate the intrinsic params of each level images.
  makeK(HCalib);
  firstFrame = newFrameHessian;
  firstRightFrame = newFrameHessian_Right;

  /// pixel selection.
  PixelSelector sel(w[0],h[0]);

  float* statusMap = new float[w[0]*h[0]];
  bool* statusMapB = new bool[w[0]*h[0]];

  Mat33f K = Mat33f::Identity();
  K(0,0) = HCalib->fxl();
  K(1,1) = HCalib->fyl();
  K(0,2) = HCalib->cxl();
  K(1,2) = HCalib->cyl();

  /// get point density in different layers.
  float densities[] = {0.03,0.05,0.15,0.5,1};
  memset(idepth[0], 0, sizeof(float)*w[0]*h[0]);

  for(int lvl=0; lvl<pyrLevelsUsed; lvl++) {
    /// STEP2: select large gradient pixels for different levels, the 0 level is more complicated. 1d,2d,4d block to select 3 levels of pixels.
    sel.currentPotential = 3;   /// set the grid size, 3x3 size grid.

    int npts,npts_right;        /// number of selected pixel.

    /// extract feature pixels at level 0.
    if(lvl == 0) {
      npts = sel.makeMaps(firstFrame, statusMap,densities[lvl]*w[0]*h[0],1,false,2);

    }
    /// goodpoints are selected for other levels.
    else {
      npts = makePixelStatus(firstFrame->dIp[lvl], statusMapB, w[lvl], h[lvl], densities[lvl]*w[0]*h[0]);
    }

    /// if the points is not empty, release the space and create a new one.
    if(points[lvl] != 0) {
      delete[] points[lvl];
    }
    points[lvl] = new Pnt[npts];

    // set idepth map by static stereo matching. if no idepth is available, set 0.01.
    int wl = w[lvl], hl = h[lvl];  /// image size of each level.
    Pnt* pl = points[lvl];         /// points on each level.
    int nl = 0;

    /// to leave space for pattern, 2 border
    /// STEP3: in the selected pixels, add point information.
    for(int y=patternPadding+1; y<hl-patternPadding-2; y++)
      for(int x=patternPadding+1; x<wl-patternPadding-2; x++) {

        /// if it is selected pixel. LEVEL 0
        if(lvl==0 && statusMap[x+y*wl] != 0) {
          ImmaturePoint* pt = new ImmaturePoint(x, y, firstFrame, statusMap[x+y*wl], HCalib);

          pt->u_stereo = pt->u;
          pt->v_stereo = pt->v;
          pt->idepth_min_stereo = 0;
          pt->idepth_max_stereo = NAN;
          ImmaturePointStatus stat = pt->traceStereo(firstRightFrame, K, 1);

          if(stat==ImmaturePointStatus::IPS_GOOD) {
            // assert(patternNum==9);
            pl[nl].u = x;
            pl[nl].v = y;

            pl[nl].idepth = pt->idepth_stereo;
            pl[nl].iR = pt->idepth_stereo;

            pl[nl].isGood=true;
            pl[nl].energy.setZero();
            pl[nl].lastHessian=0;
            pl[nl].lastHessian_new=0;
            pl[nl].my_type= (lvl!=0) ? 1 : statusMap[x+y*wl];
            idepth[0][x+y*wl] = pt->idepth_stereo;

            /// the pixel graident.
            // Eigen::Vector3f* cpt = firstFrame->dIp[lvl] + x + y*w[lvl];

            //// sumGrad2 is deprecated.
            // float sumGrad2 = 0;

            /// calculate the pixel gradient sum in the pattern.
            // for(int idx=0; idx < patternNum; idx++) {
            //   /// offset of pattern.
            //   int dx = patternP[idx][0];
            //   int dy = patternP[idx][1];
            //   float absgrad = cpt[dx + dy*w[lvl]].tail<2>().squaredNorm();
            //   sumGrad2 += absgrad;
            // }

            /// ! the threshold of the oulier point is related to the size of the pattern, a pixel is 12x12
            /// ? how is this threshold determined...
            pl[nl].outlierTH = patternNum * setting_outlierTH;

            nl++;
            assert(nl <= npts);
          }
          else {
            pl[nl].u = x;
            pl[nl].v = y;
            pl[nl].idepth = 0.01;
            //printf("the idepth is: %f\n", pl[nl].idepth);
            pl[nl].iR = 0.01;
            pl[nl].isGood=true;
            pl[nl].energy.setZero();
            pl[nl].lastHessian=0;
            pl[nl].lastHessian_new=0;
            pl[nl].my_type= (lvl!=0) ? 1 : statusMap[x+y*wl];
            //// set idepth to 0.01;
            idepth[0][x+wl*y] = 0.01;

            // Eigen::Vector3f* cpt = firstFrame->dIp[lvl] + x + y*w[lvl];

            // float sumGrad2=0;
            // for(int idx=0;idx<patternNum;idx++) {
            //   int dx = patternP[idx][0];
            //   int dy = patternP[idx][1];
            //   float absgrad = cpt[dx + dy*w[lvl]].tail<2>().squaredNorm();
            //   sumGrad2 += absgrad;
            // }

            pl[nl].outlierTH = patternNum*setting_outlierTH;
            nl++;
            assert(nl <= npts);
          }

          delete pt;
        }

        /// LEVEL != 0, higher level
        if(lvl!=0 && statusMapB[x+y*wl]) {
          int lvlm1 = lvl-1;
          int wlm1 = w[lvlm1];
          float* idepth_l = idepth[lvl];
          float* idepth_lm = idepth[lvlm1];

          //assert(patternNum==9);

          pl[nl].u = x+0.1;
          pl[nl].v = y+0.1;
          pl[nl].idepth = 1;
          pl[nl].iR = 1;
          pl[nl].isGood=true;
          pl[nl].energy.setZero();
          pl[nl].lastHessian=0;
          pl[nl].lastHessian_new=0;
          pl[nl].my_type= (lvl!=0) ? 1 : statusMap[x+y*wl];

          int bidx = 2*x   + 2*y*wlm1;

          idepth_l[x + y*wl] = idepth_lm[bidx] +
                               idepth_lm[bidx+1] +
                               idepth_lm[bidx+wlm1] +
                               idepth_lm[bidx+wlm1+1];

          // Eigen::Vector3f* cpt = firstFrame->dIp[lvl] + x + y*w[lvl];
          // float sumGrad2=0;
          // for(int idx=0;idx<patternNum;idx++) {
          //   int dx = patternP[idx][0];
          //   int dy = patternP[idx][1];
          //   float absgrad = cpt[dx + dy*w[lvl]].tail<2>().squaredNorm();
          //   sumGrad2 += absgrad;
          // }

          pl[nl].outlierTH = patternNum*setting_outlierTH;

          nl++;
          assert(nl <= npts);
        }
      }

    /// number of points, removed some points on the boundary.
    numPoints[lvl]=nl;
  }

  delete[] statusMap;
  delete[] statusMapB;

  /// STEP4: calculate the nearst neighbor and parent of a point.
  makeNN();

  /// parameter initialization
  thisToNext = SE3();
  snapped = false;
  frameID = snappedAt = 0;

  for(int i=0;i<pyrLevelsUsed;i++) {
    dGrads[i].setZero();
  }
}

//// deprecated.
void CoarseInitializer::setFirst(CalibHessian* HCalib, FrameHessian* newFrameHessian) {

  makeK(HCalib);
  firstFrame = newFrameHessian;

  PixelSelector sel(w[0],h[0]);

  float* statusMap = new float[w[0]*h[0]];
  bool* statusMapB = new bool[w[0]*h[0]];

  float densities[] = {0.03,0.05,0.15,0.5,1};

  for(int lvl=0; lvl<pyrLevelsUsed; lvl++) {
    sel.currentPotential = 3;
    int npts;
    if(lvl == 0)
      npts = sel.makeMaps(firstFrame, statusMap,densities[lvl]*w[0]*h[0],1,false,2);
    else
      npts = makePixelStatus(firstFrame->dIp[lvl], statusMapB, w[lvl], h[lvl], densities[lvl]*w[0]*h[0]);

    if(points[lvl] != 0) delete[] points[lvl];
    points[lvl] = new Pnt[npts];

    // set idepth map to initially 1 everywhere.
    int wl = w[lvl], hl = h[lvl];
    Pnt* pl = points[lvl];
    int nl = 0;
    for(int y=patternPadding+1;y<hl-patternPadding-2;y++)
      for(int x=patternPadding+1;x<wl-patternPadding-2;x++)
      {
        //if(x==2) printf("y=%d!\n",y);
        if(lvl!=0 && statusMapB[x+y*wl])
        {
          //assert(patternNum==9);
          pl[nl].u = x+0.1;
          pl[nl].v = y+0.1;
          pl[nl].idepth = 1;
          pl[nl].iR = 1;
          pl[nl].isGood=true;
          pl[nl].energy.setZero();
          pl[nl].lastHessian=0;
          pl[nl].lastHessian_new=0;
          pl[nl].my_type= (lvl!=0) ? 1 : statusMap[x+y*wl];

          Eigen::Vector3f* cpt = firstFrame->dIp[lvl] + x + y*w[lvl];
          float sumGrad2=0;
          for(int idx=0;idx<patternNum;idx++)
          {
            int dx = patternP[idx][0];
            int dy = patternP[idx][1];
            float absgrad = cpt[dx + dy*w[lvl]].tail<2>().squaredNorm();
            sumGrad2 += absgrad;
          }

          //float gth = setting_outlierTH * (sqrtf(sumGrad2)+setting_outlierTHSumComponent);
          //pl[nl].outlierTH = patternNum*gth*gth;

          pl[nl].outlierTH = patternNum*setting_outlierTH;



          nl++;
          assert(nl <= npts);
        }
      }


    numPoints[lvl]=nl;
  }
  delete[] statusMap;
  delete[] statusMapB;

  makeNN();

  thisToNext=SE3();
  snapped = false;
  frameID = snappedAt = 0;

  for(int i=0;i<pyrLevelsUsed;i++) {
    dGrads[i].setZero();
  }
}

/// reset point energy, idepth_new parameters
void CoarseInitializer::resetPoints(int lvl) {
  Pnt* pts = points[lvl];
  int npts = numPoints[lvl];

  for(int i=0; i<npts; i++) {
    /// reset.
    pts[i].energy.setZero();
    pts[i].idepth_new = pts[i].idepth;

    /// if it is the top level, use the average value of the surrounding points to reset.
    if(lvl==pyrLevelsUsed-1 && !pts[i].isGood) {
      float snd=0, sn=0;

      for(int n = 0;n<10;n++) {
        if(pts[i].neighbours[n] == -1 || !pts[pts[i].neighbours[n]].isGood) continue;
        snd += pts[pts[i].neighbours[n]].iR;
        sn += 1;
      }

      if(sn > 0) {
        pts[i].isGood=true;
        pts[i].iR = pts[i].idepth = pts[i].idepth_new = snd/sn;
      }
    }
  }
}

/// after finding the state increment, calculate the idepth that is marginalized and update the idepth.
void CoarseInitializer::doStep(int lvl, float lambda, Vec8f inc) {

  const float maxPixelStep = 0.25;
  const float idMaxStep = 1e10;
  Pnt* pts = points[lvl];
  int npts = numPoints[lvl];

  for(int i=0;i<npts;i++) {
    if(!pts[i].isGood) {
      continue;
    }

    /// dd*r + (dp*dd)^T * delta_p
    float b = JbBuffer[i][8] + JbBuffer[i].head<8>().dot(inc);

    /// dd*delta_d = dd*r - (dp*dd)^T * delta_p = b
    /// delta_d = b * dd^-1
    float step = -b * JbBuffer[i][9] / (1+lambda);

    /// maximum idepth can only increase these
    float maxstep = maxPixelStep*pts[i].maxstep;

    if(maxstep > idMaxStep) {
      maxstep=idMaxStep;
    }

    if(step > maxstep) {
      step = maxstep;
    }

    if(step < -maxstep) {
      step = -maxstep;
    }

    /// update to get new idepth.
    float newIdepth = pts[i].idepth + step;

    if(newIdepth < 1e-3 ) {
      newIdepth = 1e-3;
    }
    if(newIdepth > 50) {
      newIdepth = 50;
    }

    pts[i].idepth_new = newIdepth;
  }
}

/// assign new value to old (energy, point state, idepth, hessian)
void CoarseInitializer::applyStep(int lvl) {
  Pnt* pts = points[lvl];
  int npts = numPoints[lvl];

  for(int i=0;i<npts;i++) {
    if(!pts[i].isGood) {
      pts[i].idepth = pts[i].idepth_new = pts[i].iR;
      continue;
    }

    pts[i].energy = pts[i].energy_new;
    pts[i].isGood = pts[i].isGood_new;
    pts[i].idepth = pts[i].idepth_new;
    pts[i].lastHessian = pts[i].lastHessian_new;
  }

  std::swap<Vec10f*>(JbBuffer, JbBuffer_new);
}

/// calculate intrinsic params for each level.
void CoarseInitializer::makeK(CalibHessian* HCalib) {
  w[0] = wG[0];
  h[0] = hG[0];

  fx[0] = HCalib->fxl();
  fy[0] = HCalib->fyl();
  cx[0] = HCalib->cxl();
  cy[0] = HCalib->cyl();

  for (int level = 1; level < pyrLevelsUsed; ++ level) {
    w[level] = w[0] >> level;
    h[level] = h[0] >> level;
    fx[level] = fx[level-1] * 0.5;
    fy[level] = fy[level-1] * 0.5;
    cx[level] = (cx[0] + 0.5) / ((int)1<<level) - 0.5;
    cy[level] = (cy[0] + 0.5) / ((int)1<<level) - 0.5;
  }

  //// make K and Kinv.
  for (int level = 0; level < pyrLevelsUsed; ++ level) {
    K[level]  << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
    Ki[level] = K[level].inverse();
    fxi[level] = Ki[level](0,0);
    fyi[level] = Ki[level](1,1);
    cxi[level] = Ki[level](0,2);
    cyi[level] = Ki[level](1,2);
  }
}

//// make nearest neighbor.
/// generate a kt-tree for each layer of points, and use it to find the neighboring point set and the parent point.
void CoarseInitializer::makeNN() {
  const float NNDistFactor=0.05;

  /// the first param is distance, the second is datasetadapter, and third is dimension.
  typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, FLANNPointcloud> ,FLANNPointcloud,2> KDTree;

  // build indices
  FLANNPointcloud pcs[PYR_LEVELS];   /// each level creates a point cloud.
  KDTree* indexes[PYR_LEVELS];       /// kdtree for point cloud establishment.

  /// create a kd-tree indexed 2D point cloud per level.
  for(int i=0;i<pyrLevelsUsed;i++) {
    pcs[i] = FLANNPointcloud(numPoints[i], points[i]);
    indexes[i] = new KDTree(2, pcs[i], nanoflann::KDTreeSingleIndexAdaptorParams(5));
    indexes[i]->buildIndex();
  }

  const int nn=10;

  // find NN & parents
  for(int lvl=0;lvl<pyrLevelsUsed;lvl++) {
    Pnt* pts = points[lvl];
    int npts = numPoints[lvl];

    int ret_index[nn];   /// proximity points found.
    float ret_dist[nn];  /// search distance to point.

    /// search results, nearest nn and 1.
    nanoflann::KNNResultSet<float, int, int> resultSet(nn);
    nanoflann::KNNResultSet<float, int, int> resultSet1(1);

    for(int i=0; i<npts; i++) {
      //resultSet.init(pts[i].neighbours, pts[i].neighboursDist );
      resultSet.init(ret_index, ret_dist);

      Vec2f pt = Vec2f(pts[i].u,pts[i].v);  /// current point.

      /// use the kd-tree to query the nearest neighbor.
      indexes[lvl]->findNeighbors(resultSet, (float*)&pt, nanoflann::SearchParams());

      int myidx=0;
      float sumDF = 0;

      /// assign values to the neighbors of each point.
      for(int k=0; k<nn; k++) {
        pts[i].neighbours[myidx]=ret_index[k];       /// recent index.

        float df = expf(-ret_dist[k]*NNDistFactor);  /// distance uses exponential form.

        sumDF += df; /// distance sum.

        pts[i].neighboursDist[myidx]=df;
        assert(ret_index[k]>=0 && ret_index[k] < npts);
        myidx++;
      }

      /// reduction of distance to 10
      for(int k=0; k<nn; k++) {
        pts[i].neighboursDist[k] *= 10/sumDF;
      }

      /// find the parent node of the point in the image one level higher.
      if(lvl < pyrLevelsUsed-1 ) {
        resultSet1.init(ret_index, ret_dist);

        /// converted to a high level.
        pt = pt*0.5f-Vec2f(0.25f,0.25f);

        indexes[lvl+1]->findNeighbors(resultSet1, (float*)&pt, nanoflann::SearchParams());

        /// parent node.
        pts[i].parent = ret_index[0];
        /// distance to the parent node (in higher levels)
        pts[i].parentDist = expf(-ret_dist[0]*NNDistFactor);

        assert(ret_index[0]>=0 && ret_index[0] < numPoints[lvl+1]);
      }
      /// no parent node at the highest level.
      else {
        pts[i].parent = -1;
        pts[i].parentDist = -1;
      }
    }
  }

  // done.
  for(int i=0;i<pyrLevelsUsed;i++)
    delete indexes[i];
}

} // namespace dso
