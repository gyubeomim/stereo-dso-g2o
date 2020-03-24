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


#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "OptimizationBackend/AccumulatedSCHessian.h"
#include "OptimizationBackend/AccumulatedTopHessian.h"

namespace dso {

bool EFAdjointsValid = false; /// Whether to set the state adjoint matrix
bool EFIndicesValid = false;  /// Whether to set frame, point, res ID
bool EFDeltaValid = false;    /// Whether to set the status increment value

/// Calculate adHost(F), adTarget(F)
/// the parameters passed are useless
void EnergyFunctional::setAdjointsF(CalibHessian* Hcalib) {
  if(adHost != 0) {
    delete[] adHost;
  }
  if(adTarget != 0) {
    delete[] adTarget;
  }

  adHost = new Mat88[nFrames*nFrames];
  adTarget = new Mat88[nFrames*nFrames];

  for(int h=0;h<nFrames;h++)    /// host frame
    for(int t=0;t<nFrames;t++)  /// target frame
    {
      FrameHessian* host = frames[h]->data;
      FrameHessian* target = frames[t]->data;

      //// Tth = Ttw * Thw.inv
      SE3 hostToTarget = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();

      Mat88 AH = Mat88::Identity();
      Mat88 AT = Mat88::Identity();

      /// See note derivation, or https://www.cnblogs.com/JingeTU/p/9077372.html
      /// The transpose is because the stitchDoubleInternal will not be transposed when calculating the hessian later
      AH.topLeftCorner<6,6>() = -hostToTarget.Adj().transpose();
      AT.topLeftCorner<6,6>() = Mat66::Identity();

      /// photometric parameter, the derivative of the combined term
      /// E = Ij - tj*exp(aj)/ti*exp(ai)*Ii - (bj - tj*exp(aj)/ti*exp(ai)*bi)
      /// a = -tj*exp(aj)/ti*exp(ai)
      /// b = -(bj - tj*exp(aj)/ti*exp(ai)*bi)
      Vec2f affLL = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2l_0(), target->aff_g2l_0()).cast<float>();

      AT(6,6) = -affLL[0];  /// a'(aj)
      AT(7,7) = -1;         /// b'(bj)

      AH(6,6) = affLL[0];   /// a'(ai)
      AH(7,7) = affLL[0];   /// b'(bi)

      AH.block<3,8>(0,0) *= SCALE_XI_TRANS;
      AH.block<3,8>(3,0) *= SCALE_XI_ROT;
      AH.block<1,8>(6,0) *= SCALE_A;
      AH.block<1,8>(7,0) *= SCALE_B;

      AT.block<3,8>(0,0) *= SCALE_XI_TRANS;
      AT.block<3,8>(3,0) *= SCALE_XI_ROT;
      AT.block<1,8>(6,0) *= SCALE_A;  /// ? Already multiplied, how to multiply again
      AT.block<1,8>(7,0) *= SCALE_B;

      adHost[h + t*nFrames] = AH;
      adTarget[h + t*nFrames] = AT;
    }

  /// constant matrix
  cPrior = VecC::Constant(setting_initialCalibHessian);

  /// float type
  if(adHostF != 0) {
    delete[] adHostF;
  }
  if(adTargetF != 0) {
    delete[] adTargetF;
  }

  adHostF = new Mat88f[nFrames*nFrames];
  adTargetF = new Mat88f[nFrames*nFrames];

  for(int h=0;h<nFrames;h++)
    for(int t=0;t<nFrames;t++)
    {
      adHostF[h+t*nFrames] = adHost[h+t*nFrames].cast<float>();
      adTargetF[h+t*nFrames] = adTarget[h+t*nFrames].cast<float>();
    }

  cPriorF = cPrior.cast<float>();

  EFAdjointsValid = true;
}

EnergyFunctional::EnergyFunctional() {
  adHost=0;
  adTarget=0;

  red=0;

  adHostF=0;
  adTargetF=0;
  adHTdeltaF=0;

  nFrames = nResiduals = nPoints = 0;

  /// Initially, frame changes are added later
  HM = MatXX::Zero(CPARS,CPARS);
  bM = VecX::Zero(CPARS);

  accSSE_top_L = new AccumulatedTopHessianSSE();
  accSSE_top_A = new AccumulatedTopHessianSSE();
  accSSE_bot = new AccumulatedSCHessianSSE();

  resInA = resInL = resInM = 0;
  currentLambda=0;
}

EnergyFunctional::~EnergyFunctional() {
  for(EFFrame* f : frames) {
    for(EFPoint* p : f->points) {
      for(EFResidual* r : p->residualsAll) {
        r->data->efResidual=0;
        delete r;
      }
      p->data->efPoint=0;
      delete p;
    }
    f->data->efFrame=0;
    delete f;
  }

  if(adHost != 0) delete[] adHost;
  if(adTarget != 0) delete[] adTarget;


  if(adHostF != 0) delete[] adHostF;
  if(adTargetF != 0) delete[] adTargetF;
  if(adHTdeltaF != 0) delete[] adHTdeltaF;

  delete accSSE_top_L;
  delete accSSE_top_A;
  delete accSSE_bot;
}

/// Calculate the increment of relative quantities for various states
void EnergyFunctional::setDeltaF(CalibHessian* HCalib) {
  if(adHTdeltaF != 0) {
    delete[] adHTdeltaF;
  }

  adHTdeltaF = new Mat18f[nFrames*nFrames];

  for(int h=0;h<nFrames;h++)
    for(int t=0;t<nFrames;t++) {
      int idx = h + t*nFrames;
      /// delta_th = Adj * delta_t (or)
      /// delta_th = Adj * delta_h
      /// Add it together, it should be the increment of pose transformation between two frames, because h becomes a bit, t becomes a bit
      adHTdeltaF[idx] = frames[h]->data->get_state_minus_stateZero().head<8>().cast<float>().transpose() * adHostF[idx] +
                        frames[t]->data->get_state_minus_stateZero().head<8>().cast<float>().transpose() * adTargetF[idx];
    }

  /// camera intrinsic parameter increment
  cDeltaF = HCalib->value_minus_value_zero.cast<float>();

  for(EFFrame* f : frames) {
    /// frame pose increment
    f->delta = f->data->get_state_minus_stateZero().head<8>();

    /// a priori increment
    f->delta_prior = (f->data->get_state() - f->data->getPriorZero()).head<8>();

    for(EFPoint* p : f->points) {
      /// idepth increment
      p->deltaF = p->data->idepth - p->data->idepth_zero;
    }
  }

  EFDeltaValid = true;
}

// accumulates & shifts L.
/// Calculate the normal equation formed by the frame points in the energy equation
//// A: Active
void EnergyFunctional::accumulateAF_MT(MatXX &H, VecX &b, bool MT) {
  if(MT) {
    red->reduce(boost::bind(&AccumulatedTopHessianSSE::setZero,
                            accSSE_top_A, nFrames,  _1, _2, _3, _4), 0, 0, 0);
    red->reduce(boost::bind(&AccumulatedTopHessianSSE::addPointsInternal<0>,
                            accSSE_top_A, &allPoints, this,  _1, _2, _3, _4), 0, allPoints.size(), 50);
    accSSE_top_A->stitchDoubleMT(red,H,b,this,false,true);
    resInA = accSSE_top_A->nres[0];
  }
  else {
    accSSE_top_A->setZero(nFrames);
    for(EFFrame* f : frames)
      for(EFPoint* p : f->points)
        accSSE_top_A->addPoint<0>(p,this);                  /// mode 0 increase EF point
    accSSE_top_A->stitchDoubleMT(red,H,b,this,false,false); /// without prior, get H, b
    resInA = accSSE_top_A->nres[0];                         /// All residuals count
  }
}

// accumulates & shifts L.
/// Calculate H and b, add the prior, res is subtract the linearized residual
//// L: Linearized
void EnergyFunctional::accumulateLF_MT(MatXX &H, VecX &b, bool MT) {
  if(MT) {
    red->reduce(boost::bind(&AccumulatedTopHessianSSE::setZero,
                            accSSE_top_L, nFrames,  _1, _2, _3, _4), 0, 0, 0);
    red->reduce(boost::bind(&AccumulatedTopHessianSSE::addPointsInternal<1>,
                            accSSE_top_L, &allPoints, this,  _1, _2, _3, _4), 0, allPoints.size(), 50);
    accSSE_top_L->stitchDoubleMT(red,H,b,this,true,true);
    resInL = accSSE_top_L->nres[0];
  }
  else {
    accSSE_top_L->setZero(nFrames);
    for(EFFrame* f : frames)
      for(EFPoint* p : f->points)
        accSSE_top_L->addPoint<1>(p,this);                 /// mode 1
    accSSE_top_L->stitchDoubleMT(red,H,b,this,true,false);
    resInL = accSSE_top_L->nres[0];
  }
}

/// Calculate the Schur complement part of the idepth
void EnergyFunctional::accumulateSCF_MT(MatXX &H, VecX &b, bool MT) {
  if(MT) {
    red->reduce(boost::bind(&AccumulatedSCHessianSSE::setZero, accSSE_bot, nFrames,  _1, _2, _3, _4), 0, 0, 0);
    red->reduce(boost::bind(&AccumulatedSCHessianSSE::addPointsInternal,
                            accSSE_bot, &allPoints, true,  _1, _2, _3, _4), 0, allPoints.size(), 50);
    accSSE_bot->stitchDoubleMT(red, H, b, this, true);
  }
  else
  {
    accSSE_bot->setZero(nFrames);
    for(EFFrame* f : frames)
      for(EFPoint* p : f->points)
        accSSE_bot->addPoint(p, true);
    accSSE_bot->stitchDoubleMT(red, H, b,this, false);
  }
}

/// Calculate camera parameters, poses, and photometric increments
void EnergyFunctional::resubstituteF_MT(VecX x, CalibHessian* HCalib, bool MT) {
  assert(x.size() == CPARS + nFrames*8);

  VecXf xF = x.cast<float>();

  /// Camera  intrinsic parameters, this time increment
  HCalib->step = -x.head<CPARS>();

  Mat18f* xAd = new Mat18f[nFrames*nFrames];
  VecCf cstep = xF.head<CPARS>();

  for(EFFrame* h : frames) {
    /// increment of frame pose and photometric solution
    h->data->step.head<8>() = -x.segment<8>(CPARS + 8*h->idx);
    h->data->step.tail<2>().setZero();

    /// The absolute pose increment becomes relative
    for(EFFrame* t : frames)
      xAd[nFrames*h->idx + t->idx] = xF.segment<8>(CPARS + 8*h->idx).transpose() * adHostF[h->idx + nFrames*t->idx] +
                                     xF.segment<8>(CPARS + 8*t->idx).transpose() * adTargetF[h->idx + nFrames*t->idx];
  }

  /// Calculate the idepth increment of a point
  if(MT)
    red->reduce(boost::bind(&EnergyFunctional::resubstituteFPt,
                            this, cstep, xAd,  _1, _2, _3, _4), 0, allPoints.size(), 50);
  else
    resubstituteFPt(cstep, xAd, 0, allPoints.size(), 0,0);

  delete[] xAd;
}

/// Calculate the increment of point idepth
void EnergyFunctional::resubstituteFPt(const VecCf &xc, Mat18f* xAd, int min, int max, Vec10* stats, int tid)
{
  for(int k=min;k<max;k++) {
    EFPoint* p = allPoints[k];

    int ngoodres = 0;

    for(EFResidual* r : p->residualsAll)
      if(r->isActive())
        ngoodres++;

    if(ngoodres==0) {
      p->data->step = 0;
      continue;
    }

    //// delta_idepth = -H_idepth_idepth^-1 * (J_idepth^{T} * r - H_idepth_y * delta_y)
    //// where y = [ xi, a, b, c ]

    float b = p->bdSumF;
    /// minus idepth and intrinsic parameters
    b -= xc.dot(p->Hcd_accAF + p->Hcd_accLF);

    for(EFResidual* r : p->residualsAll) {
      if(!r->isActive()) {
        continue;
      }

      /// Absolutely relative, xAd is transposed
      b -= xAd[r->hostIDX*nFrames + r->targetIDX] * r->JpJdF;
    }

    /// idepth increment
    p->data->step = -b*p->HdiF;
    assert(std::isfinite(p->data->step));
  }
}

/// Also find energy, use HM and bM to find, delta is absolute
double EnergyFunctional::calcMEnergyF() {
  assert(EFDeltaValid);
  assert(EFAdjointsValid);
  assert(EFIndicesValid);

  VecX delta = getStitchedDeltaF();
  return delta.dot(2*bM + HM*delta);
}

/// Calculate the sum of the energy E at all points, delta is relative
void EnergyFunctional::calcLEnergyPt(int min, int max, Vec10* stats, int tid) {
  Accumulator11 E;
  E.initialize();
  VecCf dc = cDeltaF;

  for(int i=min; i<max; i++) {
    EFPoint* p = allPoints[i];
    float dd = p->deltaF;

    for(EFResidual* r : p->residualsAll) {
      //// skip if it is not linearized or activated.
      /// While satisfying
      if(!r->isLinearized || !r->isActive()) {
        continue;
      }

      Mat18f dp = adHTdeltaF[r->hostIDX + nFrames*r->targetIDX];
      RawResidualJacobian* rJ = r->J;

      // compute Jp*delta
      float Jp_delta_x_1 = rJ->Jpdxi[0].dot(dp.head<6>()) +
                           rJ->Jpdc[0].dot(dc) +
                           rJ->Jpdd[0]*dd;

      float Jp_delta_y_1 = rJ->Jpdxi[1].dot(dp.head<6>()) +
                           rJ->Jpdc[1].dot(dc) +
                           rJ->Jpdd[1]*dd;

      __m128 Jp_delta_x = _mm_set1_ps(Jp_delta_x_1);
      __m128 Jp_delta_y = _mm_set1_ps(Jp_delta_y_1);
      __m128 delta_a = _mm_set1_ps((float)(dp[6]));
      __m128 delta_b = _mm_set1_ps((float)(dp[7]));

      for(int i=0; i+3 < patternNum; i+=4) {
        // PATTERN: E = (2*res_toZeroF + J*delta)*J*delta.
        /// E = (f(x0) + J*dx)^2 = dx*H*dx + 2*J*dx*f(x0) + f(x0)^2
        /// throw away the constant f(x0)^2
        __m128 Jdelta =             _mm_mul_ps(_mm_load_ps(((float*)(rJ->JIdx))+i), Jp_delta_x);
        Jdelta = _mm_add_ps(Jdelta, _mm_mul_ps(_mm_load_ps(((float*)(rJ->JIdx+1))+i), Jp_delta_y));
        Jdelta = _mm_add_ps(Jdelta, _mm_mul_ps(_mm_load_ps(((float*)(rJ->JabF))+i), delta_a));
        Jdelta = _mm_add_ps(Jdelta, _mm_mul_ps(_mm_load_ps(((float*)(rJ->JabF+1))+i), delta_b));

        __m128 r0 = _mm_load_ps(((float*)&r->res_toZeroF)+i);
        r0 = _mm_add_ps(r0, r0);
        r0 = _mm_add_ps(r0, Jdelta);
        Jdelta = _mm_mul_ps(Jdelta, r0);
        E.updateSSENoShift(Jdelta);
      }

      /// 128-bit aligned, extra part
      /// 4% remainder
      for(int i=((patternNum>>2)<<2); i < patternNum; i++) {
        float Jdelta = rJ->JIdx[0][i]*Jp_delta_x_1 +
                       rJ->JIdx[1][i]*Jp_delta_y_1 +
                       rJ->JabF[0][i]*dp[6] +
                       rJ->JabF[1][i]*dp[7];
        E.updateSingleNoShift((float)(Jdelta*(Jdelta + 2*r->res_toZeroF[i])));
      }
    }
    /// idepth prior
    E.updateSingle(p->deltaF * p->deltaF * p->priorF);
  }

  E.finish();
  (*stats)[0] += E.A;
}

/// MT is multi-threaded, calculating energy, including prior + square of residuals
double EnergyFunctional::calcLEnergyF_MT() {
  assert(EFDeltaValid);
  assert(EFAdjointsValid);
  assert(EFIndicesValid);

  double E = 0;

  /// Prior energy (x - x_prior)^T*∑*(x - x_prior)
  /// Because f->prior is a diagonal of hessian, which is represented by a vector, so use cwiseProduct to multiply one by one
  for(EFFrame* f : frames)
    //// cwiseProduct == coefficient wise product.
    /// Pose prior
    E += f->delta_prior.cwiseProduct(f->prior).dot(f->delta_prior);

  /// Camera intrinsic parameters
  E += cDeltaF.cwiseProduct(cPriorF).dot(cDeltaF);

  red->reduce(boost::bind(&EnergyFunctional::calcLEnergyPt, this, _1, _2, _3, _4), 0, allPoints.size(), 50);

  return E + red->stats[0];
}

/// Insert a residual into the energy function and update the connection graph relationship
EFResidual* EnergyFunctional::insertResidual(PointFrameResidual* r) {
  EFResidual* efr = new EFResidual(r, r->point->efPoint, r->host->efFrame, r->target->efFrame);
  /// The ids of all residuals at this point
  efr->idxInAll = r->point->efPoint->residualsAll.size();
  /// All residuals at this point
  r->point->efPoint->residualsAll.push_back(efr);

  /// Increment the res count between two frames
  connectivityMap[(((long)efr->host->frameID) << 32) + ((long)efr->target->frameID)][0]++;

  nResiduals++;
  r->efResidual = efr;
  return efr;
}

/// Add a frame to the energy function
/// operations: change the normal equation, rearrange ID, common view relationship
EFFrame* EnergyFunctional::insertFrame(FrameHessian* fh, CalibHessian* Hcalib) {
  /// Build energy function frames for optimization. Add to energy function frames
  EFFrame* eff = new EFFrame(fh);
  eff->idx = frames.size();
  frames.push_back(eff);

  nFrames++;
  /// FrameHessian points to the energy function frame
  fh->efFrame = eff;

  /// Marginalize one frame, missing 8
  assert(HM.cols() == 8*nFrames + CPARS - 8);

  /// 8 parameters per frame + camera  intrinsic parameters
  bM.conservativeResize(8*nFrames + CPARS);
  HM.conservativeResize(8*nFrames + CPARS, 8*nFrames + CPARS);

  /// block of new frame is 0
  bM.tail<8>().setZero();
  HM.rightCols<8>().setZero();
  HM.bottomRows<8>().setZero();

  EFIndicesValid = false;
  EFAdjointsValid=false;
  EFDeltaValid=false;

  /// Set up the adjoint matrix
  setAdjointsF(Hcalib);

  /// Set id
  makeIDX();

  for(EFFrame* fh2 : frames) {
    /// The first 32 bits are the historical ID of the host frame, and the last 32 bits are the historical ID of the Target
    connectivityMap[(((long)eff->frameID) << 32) + ((long)fh2->frameID)] = Eigen::Vector2i(0,0);

    if(fh2 != eff) {
      connectivityMap[(((long)fh2->frameID) << 32) + ((long)eff->frameID)] = Eigen::Vector2i(0,0);
    }
  }

  return eff;
}

/// Insert a point into the energy function and put it into the corresponding EFframe
EFPoint* EnergyFunctional::insertPoint(PointHessian* ph) {
  EFPoint* efp = new EFPoint(ph, ph->host->efFrame);
  efp->idxInPoints = ph->host->efFrame->points.size();
  ph->host->efFrame->points.push_back(efp);
  //printf("the host of this PointHessian is: %d, and the num of present pointsis: %d\n", ph->host->frameID, ph->host->efFrame->points.size());


  nPoints++;
  ph->efPoint = efp;

  /// There are IDs that need to be re-combed the residuals
  EFIndicesValid = false;

  return efp;
}

/// Discard a residual, and update the relationship
void EnergyFunctional::dropResidual(EFResidual* r) {
  EFPoint* p = r->point;
  assert(r == p->residualsAll[r->idxInAll]);

  /// Last one for current
  p->residualsAll[r->idxInAll] = p->residualsAll.back();
  /// The current id becomes the current position
  p->residualsAll[r->idxInAll]->idxInAll = r->idxInAll;
  /// Pop up most
  p->residualsAll.pop_back();

  /// count
  if(r->isActive()) {
    r->host->data->shell->statistics_goodResOnThis++;
  }
  else {
    r->host->data->shell->statistics_outlierResOnThis++;
  }

  /// residual key minus one
  connectivityMap[(((long)r->host->frameID) << 32) + ((long)r->target->frameID)][0]--;
  nResiduals--;

  /// PointFrameHessian pointer to the residual
  r->data->efResidual=0;

  delete r;
}

/// Marginalize one frame fh
void EnergyFunctional::marginalizeFrame(EFFrame* fh) {
  assert(EFDeltaValid);
  assert(EFAdjointsValid);
  assert(EFIndicesValid);

  assert((int)fh->points.size()==0);
  int ndim = nFrames*8+CPARS-8;// new dimension
  int odim = nFrames*8+CPARS;// old dimension


  //	VecX eigenvaluesPre = HM.eigenvalues().real();
  //	std::sort(eigenvaluesPre.data(), eigenvaluesPre.data()+eigenvaluesPre.size());
  //

  /// [*** step 1 ***] Move the marginalized frame to the far right, farthest
  /// HM bM is obtained by the marginalization point
  if((int)fh->idx != (int)frames.size()-1) {
    int io = fh->idx*8+CPARS;	                 // index of frame to move to end
    int ntail = 8*(nFrames-fh->idx-1);           /// Number of variables behind marginalized frames
    assert((io+8+ntail) == nFrames*8+CPARS);

    Vec8 bTmp = bM.segment<8>(io);      /// Number of variables behind marginalized frames...
    VecX tailTMP = bM.tail(ntail);      /// Back to front
    bM.segment(io,ntail) = tailTMP;
    bM.tail<8>() = bTmp;

    /// Marginalize the frame to the right and move it to the front
    MatXX HtmpCol = HM.block(0,io,odim,8);
    MatXX rightColsTmp = HM.rightCols(ntail);
    HM.block(0,io,odim,ntail) = rightColsTmp;
    HM.rightCols(8) = HtmpCol;

    /// Move the bottom edge of the frame up
    MatXX HtmpRow = HM.block(io,0,8,odim);
    MatXX botRowsTmp = HM.bottomRows(ntail);
    HM.block(io,0,ntail,odim) = botRowsTmp;
    HM.bottomRows(8) = HtmpRow;
  }

  /// [*** step 2 ***] plus prior
  /// If the frame obtained from the initialization has a priori, it needs to be added when marginalizing. The lightness also has a priori
  //	// marginalize. First add prior here, instead of to active.
  HM.bottomRightCorner<8,8>().diagonal() += fh->prior;
  bM.tail<8>() += fh->prior.cwiseProduct(fh->delta_prior);

  /// [*** step 3 ***] First scaled and then calculate Schur complement
  //	std::cout << std::setprecision(16) << "HMPre:\n" << HM << "\n\n";
  VecX SVec = (HM.diagonal().cwiseAbs()+VecX::Constant(HM.cols(), 10)).cwiseSqrt();
  VecX SVecI = SVec.cwiseInverse();

  //	std::cout << std::setprecision(16) << "SVec: " << SVec.transpose() << "\n\n";
  //	std::cout << std::setprecision(16) << "SVecI: " << SVecI.transpose() << "\n\n";

  // scale!
  MatXX HMScaled = SVecI.asDiagonal() * HM * SVecI.asDiagonal();
  VecX bMScaled =  SVecI.asDiagonal() * bM;

  // invert bottom part!
  Mat88 hpi = HMScaled.bottomRightCorner<8,8>();
  hpi = 0.5f*(hpi+hpi);
  hpi = hpi.inverse();
  hpi = 0.5f*(hpi+hpi);

  // schur-complement!
  MatXX bli = HMScaled.bottomLeftCorner(8,ndim).transpose() * hpi;
  HMScaled.topLeftCorner(ndim,ndim).noalias() -= bli * HMScaled.bottomLeftCorner(8,ndim);
  bMScaled.head(ndim).noalias() -= bli*bMScaled.tail<8>();

  //unscale!
  HMScaled = SVec.asDiagonal() * HMScaled * SVec.asDiagonal();
  bMScaled = SVec.asDiagonal() * bMScaled;

  // set.
  HM = 0.5*(HMScaled.topLeftCorner(ndim,ndim) + HMScaled.topLeftCorner(ndim,ndim).transpose());
  bM = bMScaled.head(ndim);

  /// [*** step 4 ***] Change the ID number of EFFrame, and delete
  // remove from vector, without changing the order!
  for(unsigned int i=fh->idx; i+1<frames.size();i++) {
    frames[i] = frames[i+1];
    frames[i]->idx = i;
  }

  frames.pop_back();
  nFrames--;
  fh->data->efFrame=0;

  assert((int)frames.size()*8+CPARS == (int)HM.rows());
  assert((int)frames.size()*8+CPARS == (int)HM.cols());
  assert((int)frames.size()*8+CPARS == (int)bM.size());
  assert((int)frames.size() == (int)nFrames);

  //VecX eigenvaluesPost = HM.eigenvalues().real();
  //std::sort(eigenvaluesPost.data(), eigenvaluesPost.data()+eigenvaluesPost.size());

  //std::cout << std::setprecision(16) << "HMPost:\n" << HM << "\n\n";

  //std::cout << "EigPre:: " << eigenvaluesPre.transpose() << "\n";
  //std::cout << "EigPost: " << eigenvaluesPost.transpose() << "\n";

  EFIndicesValid = false;
  EFAdjointsValid=false;
  EFDeltaValid=false;

  makeIDX();
  delete fh;
}

/// Marginalize a point
void EnergyFunctional::marginalizePointsF() {
  assert(EFDeltaValid);
  assert(EFAdjointsValid);
  assert(EFIndicesValid);

  /// [*** step 1 ***] record marginalized points
  allPointsToMarg.clear();
  for(EFFrame* f : frames) {
    for(int i=0; i<(int)f->points.size(); i++) {
      EFPoint* p = f->points[i];
      if(p->stateFlag == EFPointStatus::PS_MARGINALIZE) {
        p->priorF *= setting_idepthFixPriorMargFac;   /// ? What is this ???

        for(EFResidual* r : p->residualsAll) {
          /// Marginalized residual count
          if(r->isActive()) {
            connectivityMap[(((long)r->host->frameID) << 32) + ((long)r->target->frameID)][1]++;
          }
        }

        allPointsToMarg.push_back(p);
      }
    }
  }

  /// [*** step 2 ***] Calculate H, b, HSC, bSC formed by the residuals connected at this point
  accSSE_bot->setZero(nFrames);
  accSSE_top_A->setZero(nFrames);

  for(EFPoint* p : allPointsToMarg) {
    accSSE_top_A->addPoint<2>(p,this);   /// residual at this point, calculate H b
    accSSE_bot->addPoint(p,false);       /// marginalized part
    removePoint(p);
  }

  MatXX M, Msc;
  VecX Mb, Mbsc;

  accSSE_top_A->stitchDouble(M, Mb, this, false, false);  /// no prior, added later
  accSSE_bot->stitchDouble(Msc, Mbsc, this);

  resInM+= accSSE_top_A->nres[0];

  MatXX H = M - Msc;
  VecX b = Mb - Mbsc;

  /// [*** step 3 ***] handle nullspace
  /// subtract the nullspace part
  if(setting_solverMode & SOLVER_ORTHOGONALIZE_POINTMARG) {
    // have a look if prior is there.
    bool haveFirstFrame = false;
    for(EFFrame* f : frames) {
      if(f->frameID==0) {
        haveFirstFrame=true;
      }
    }

    if(!haveFirstFrame) {
      orthogonalize(&b, &H);
    }
  }

  /// Weights the amount of marginalization, inaccurate linearization
  /// So the marginalized part is directly added to HM bM
  HM += setting_margWeightFac*H;
  bM += setting_margWeightFac*b;

  if(setting_solverMode & SOLVER_ORTHOGONALIZE_FULL)
    orthogonalize(&bM, &HM);

  EFIndicesValid = false;
  /// Grooming ID
  makeIDX();
}

/// Drop points directly, not marginalized
void EnergyFunctional::dropPointsF() {
  for(EFFrame* f : frames) {
    for(int i=0;i<(int)f->points.size();i++) {
      EFPoint* p = f->points[i];
      if(p->stateFlag == EFPointStatus::PS_DROP) {
        removePoint(p);
        i--;
      }
    }
  }

  EFIndicesValid = false;
  makeIDX();
}

/// Remove a point p from EFFrame
void EnergyFunctional::removePoint(EFPoint* p) {
  for(EFResidual* r : p->residualsAll) {
    /// throw away all the residuals of the change
    dropResidual(r);
  }

  EFFrame* h = p->host;
  h->points[p->idxInPoints] = h->points.back();
  h->points[p->idxInPoints]->idxInPoints = p->idxInPoints;
  h->points.pop_back();

  nPoints--;
  p->data->efPoint = 0;

  EFIndicesValid = false;

  delete p;
}

/// Calculate the nullspace matrix pseudo-inverse and subtract the nullspace from H and b, which is equivalent to setting the corresponding Jacob to 0
void EnergyFunctional::orthogonalize(VecX* b, MatXX* H) {
  //	VecX eigenvaluesPre = H.eigenvalues().real();
  //	std::sort(eigenvaluesPre.data(), eigenvaluesPre.data()+eigenvaluesPre.size());
  //	std::cout << "EigPre:: " << eigenvaluesPre.transpose() << "\n";

  // decide to which nullspaces to orthogonalize.
  std::vector<VecX> ns;
  ns.insert(ns.end(), lastNullspaces_pose.begin(), lastNullspaces_pose.end());
  ns.insert(ns.end(), lastNullspaces_scale.begin(), lastNullspaces_scale.end());
  //	if(setting_affineOptModeA <= 0)
  //		ns.insert(ns.end(), lastNullspaces_affA.begin(), lastNullspaces_affA.end());
  //	if(setting_affineOptModeB <= 0)
  //		ns.insert(ns.end(), lastNullspaces_affB.begin(), lastNullspaces_affB.end());

  // make Nullspaces matrix
  /// 7 degrees of freedom
  MatXX N(ns[0].rows(), ns.size());   /// size (4 + 8 * n) × 7
  for(unsigned int i=0;i<ns.size();i++) {
    N.col(i) = ns[i].normalized();
  }

  /// Find pseudoinverse
  // compute Npi := N * (N' * N)^-1 = pseudo inverse of N.
  Eigen::JacobiSVD<MatXX> svdNN(N, Eigen::ComputeThinU | Eigen::ComputeThinV);

  VecX SNN = svdNN.singularValues();
  double minSv = 1e10, maxSv = 0;

  for(int i=0;i<SNN.size();i++) {
    if(SNN[i] < minSv) minSv = SNN[i];
    if(SNN[i] > maxSv) maxSv = SNN[i];
  }

  /// Setting_solverModeDelta (1e-5) times smaller than the maximum singular value, it is considered to be 0
  /// find the inverse
  for(int i=0;i<SNN.size();i++) {
    if(SNN[i] > setting_solverModeDelta*maxSv)
      SNN[i] = 1.0 / SNN[i];
    else
      SNN[i] = 0;
  }

  MatXX Npi = svdNN.matrixU() * SNN.asDiagonal() * svdNN.matrixV().transpose(); 	// [dim] x 9.
  /// Npi.transpose () is the pseudo-inverse of N
  MatXX NNpiT = N*Npi.transpose(); 	// [dim] x [dim].
  MatXX NNpiTS = 0.5*(NNpiT + NNpiT.transpose());	// = N * (N' * N)^-1 * N'.

  /// Why did you do this?
  /// Subtract the nullspace from H and b ???
  if(b!=0) {
    *b -= NNpiTS * *b;
  }
  if(H!=0) {
    *H -= NNpiTS * *H * NNpiTS;
  }

  //	std::cout << std::setprecision(16) << "Orth SV: " << SNN.reverse().transpose() << "\n";
  //	VecX eigenvaluesPost = H.eigenvalues().real();
  //	std::sort(eigenvaluesPost.data(), eigenvaluesPost.data()+eigenvaluesPost.size());
  //	std::cout << "EigPost:: " << eigenvaluesPost.transpose() << "\n";
}

/// Calculate normal equations and solve
void EnergyFunctional::solveSystemF(int iteration, double lambda, CalibHessian* HCalib) {
  /// Different bits control different modes
  if(setting_solverMode & SOLVER_USE_GN) {
    lambda=0;
  }
  /// Really damn GN, just a little damping
  if(setting_solverMode & SOLVER_FIX_LAMBDA) {
    lambda = 1e-5;
  }

  assert(EFDeltaValid);
  assert(EFAdjointsValid);
  assert(EFIndicesValid);

  /// [*** step 1 ***] Calculate normal equations first, involving marginalization, prior, Schur complement, etc.
  MatXX HL_top, HA_top, H_sc;
  VecX  bL_top, bA_top, bM_top, b_sc;

  /// For the new residual, the current residual used, without the idepth
  accumulateAF_MT(HA_top, bA_top,multiThreading);

  /// Residuals of marginalization fix, with marginalization pairs, use res_toZeroF minus the linearization part, plus the prior, the part without idepth
  /// bug: There are no points involved at all here, only a priori information, because the marginalized and deleted points are not there anymore
  /// ! The only effect here is to zero the p
  /// Calculated from previous calculations
  accumulateLF_MT(HL_top, bL_top,multiThreading);

  /// Schur part about idepth
  accumulateSCF_MT(H_sc, b_sc, multiThreading);

  /// TODO What are HM and bM
  /// Due to the fixed linearization point, the residuals are updated every iteration
  bM_top = (bM + HM*getStitchedDeltaF());

  MatXX HFinal_top;
  VecX bFinal_top;

  /// [*** step 2 ***] If it is set to solve the orthogonal system, set the corresponding nullspace part of Jacobian to 0, otherwise calculate the schur normally
  if(setting_solverMode & SOLVER_ORTHOGONALIZE_SYSTEM) {
    // have a look if prior is there.
    bool haveFirstFrame = false;

    for(EFFrame* f : frames)
      if(f->frameID==0)
        haveFirstFrame=true;

    /// calculate after Schur
    MatXX HT_act =  HL_top + HA_top - H_sc;
    VecX bT_act =   bL_top + bA_top - b_sc;

    /// Include the first frame without subtracting the nullspace
    /// Does not include the first frame, because the first frame must be fixed and unified with the first frame, minus nullspace, to prevent floating in nullspace
    if(!haveFirstFrame)
      orthogonalize(&bT_act, &HT_act);

    HFinal_top = HT_act + HM;
    bFinal_top = bT_act + bM_top;

    lastHS = HFinal_top;
    lastbS = bFinal_top;

    /// LM
    /// This damping is also added after the Schur complement calculation
    for(int i=0; i<8*nFrames+CPARS; i++) {
      HFinal_top(i,i) *= (1 + lambda);
    }
  }
  else {
    HFinal_top = HL_top + HM + HA_top;
    bFinal_top = bL_top + bM_top + bA_top - b_sc;

    lastHS = HFinal_top - H_sc;
    lastbS = bFinal_top;

    /// And this is the damping added to the entire Hessian
    /// Why is it because the nullspace is subtracted ??
    for(int i=0; i < 8*nFrames + CPARS; i++) {
      HFinal_top(i,i) *= (1+lambda);
    }

    HFinal_top -= H_sc * (1.0f/(1+lambda)); /// Because Schur has a diagonal inverse, so it is a countdown
  }

  /// [*** step 3 ***] Solve using SVD, or solve directly with ldlt
  VecX x;

  if(setting_solverMode & SOLVER_SVD) {
    /// Scale for numerical stability
    VecX SVecI = HFinal_top.diagonal().cwiseSqrt().cwiseInverse();
    MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top * SVecI.asDiagonal();
    VecX bFinalScaled  = SVecI.asDiagonal() * bFinal_top;

    /// Hx = b ---> U∑V^T * x = b
    Eigen::JacobiSVD<MatXX> svd(HFinalScaled, Eigen::ComputeThinU | Eigen::ComputeThinV);

    /// Singular Value
    VecX S = svd.singularValues();
    double minSv = 1e10, maxSv = 0;

    for(int i=0;i<S.size();i++) {
      if(S[i] < minSv) minSv = S[i];
      if(S[i] > maxSv) maxSv = S[i];
    }

    /// Hx = b ---> U∑V^T * x = b ---> ΣV^T * x = U^T * b
    VecX Ub = svd.matrixU().transpose()*bFinalScaled;
    int setZero=0;

    for(int i=0;i<Ub.size();i++) {
      /// Small singular value is set to 0
      if(S[i] < setting_solverModeDelta*maxSv) {
        Ub[i] = 0;
        setZero++;
      }

      /// Set aside 7 unobservable, nullspaces
      if((setting_solverMode & SOLVER_SVD_CUT7) && (i >= Ub.size()-7)) {
        Ub[i] = 0;
        setZero++;
      }
      /// V^T * x = ∑^-1 * U^T * b
      else {
        Ub[i] /= S[i];
      }
    }
    /// x = V * ∑^-1 * U^T * b multiply scaled back
    x = SVecI.asDiagonal() * svd.matrixV() * Ub;
  }
  else {
    VecX SVecI = (HFinal_top.diagonal() + VecX::Constant(HFinal_top.cols(), 10)).cwiseSqrt().cwiseInverse();
    MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top * SVecI.asDiagonal();

#if 0
    std::cout << "original\n" << HFinal_top << std::endl;
    std::cout << "SVecI\n" << SVecI.transpose() << std::endl;
    std::cout << "scaled\n" << HFinalScaled << std::endl;
#endif

    x = SVecI.asDiagonal() * HFinalScaled.ldlt().solve(SVecI.asDiagonal() * bFinal_top);//  SVec.asDiagonal() * svd.matrixV() * Ub;
  }

  /// [*** step 4 ***] If it is set to process the solution directly, directly remove the nullspace in the solution x
  if((setting_solverMode & SOLVER_ORTHOGONALIZE_X) ||
     (iteration >= 2 && (setting_solverMode & SOLVER_ORTHOGONALIZE_X_LATER))) {
    VecX xOld = x;
    orthogonalize(&x, 0);
  }

  lastX = x;

  /// [*** step 5 ***] Find the increment value of each quantity to be calculated separately
  //resubstituteF(x, HCalib);
  currentLambda= lambda;

  resubstituteF_MT(x, HCalib,multiThreading);

  currentLambda=0;
}

/// Set the ID number corresponding to EFFrame, EFPoint, EFResidual
void EnergyFunctional::makeIDX() {
  /// reassign ID
  for(unsigned int idx=0;idx<frames.size();idx++)
    frames[idx]->idx = idx;

  allPoints.clear();

  for(EFFrame* f : frames) {
    for(EFPoint* p : f->points)
    {
      allPoints.push_back(p);
      /// Reisdual ID
      for(EFResidual* r : p->residualsAll)
      {
        r->hostIDX = r->host->idx; /// EFFrame idx
        r->targetIDX = r->target->idx;
      }
    }
  }
  EFIndicesValid=true;
}

/// Returns the state increment, where the frame pose and photometric parameters are absolute for each frame
VecX EnergyFunctional::getStitchedDeltaF() const {
  VecX d = VecX(CPARS + nFrames*8);
  /// Camera Increment
  d.head<CPARS>() = cDeltaF.cast<double>();

  //// pose + photometric params.
  for(int h=0; h < nFrames; h++) {
    d.segment<8>(CPARS + 8*h) = frames[h]->delta;
  }

  return d;
}

} // namespace dso
