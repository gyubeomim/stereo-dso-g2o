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


 
#include "FullSystem/HessianBlocks.h"
#include "util/FrameShell.h"
#include "FullSystem/ImmaturePoint.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

namespace dso {


/// constructor, immature point and map point.
PointHessian::PointHessian(const ImmaturePoint* const rawPoint, CalibHessian* Hcalib) {
  instanceCounter++;
  host = rawPoint->host;  /// main frame.
  hasDepthPrior=false;

  idepth_hessian=0;
  maxRelBaseline=0;
  numGoodResiduals=0;

  // set static values & initialization.
  u = rawPoint->u;
  v = rawPoint->v;
  assert(std::isfinite(rawPoint->idepth_max));
  //idepth_init = rawPoint->idepth_GT;

  /// seems to be used for display.
  my_type = rawPoint->my_type;

  /// mean depth.
  setIdepthScaled((rawPoint->idepth_max + rawPoint->idepth_min)*0.5);
  setPointStatus(PointHessian::INACTIVE);

  int n = patternNum;

  /// one pixel corresponds to 8 pixels.
  memcpy(color, rawPoint->color, sizeof(float)*n);
  memcpy(weights, rawPoint->weights, sizeof(float)*n);
  energyTH = rawPoint->energyTH;

  efPoint=0;

  //// arbitrary added
  idx = instanceCounter;
}

void PointHessian::release() {
  for(unsigned int i=0;i<residuals.size();i++) delete residuals[i];
  residuals.clear();
}

/// set the state of the fixed linearization point position.
/// I didn't understand the nullspace TODO.
//// set the nullspaces.
void FrameHessian::setStateZero(Vec10 state_zero) {
  /// the first 6DOF pose must be 0
  assert(state_zero.head<6>().squaredNorm() < 1e-20);

  this->state_zero = state_zero;

  /// ! I feel this nullspaces_pose is Adj_T
  /// ! exp(Adj_T*zeta) = T*exp(zeta)*T^{-1}
  /// Global to local, multiply left and right
  /// ! T_c_w * delta_T_g * T_c_w_inv = delta_T_l
  /// TODO Is this a method of numerical differentiation ???
  for(int i=0; i < 6; i++) {
    Vec6 eps;
    eps.setZero();
    eps[i] = 1e-3;

    SE3 EepsP = Sophus::SE3::exp(eps);
    SE3 EepsM = Sophus::SE3::exp(-eps);

    SE3 w2c_leftEps_P_x0 = (get_worldToCam_evalPT()*EepsP)*get_worldToCam_evalPT().inverse();

    SE3 w2c_leftEps_M_x0 = (get_worldToCam_evalPT()*EepsM)*get_worldToCam_evalPT().inverse();

    //// set nullspaces pose
    nullspaces_pose.col(i) = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log())/(2e-3);
  }
  //ncurrullspaces_pose.topRows<3>() *= SCALE_XI_TRANS_INVERSE;
  //nullspaces_pose.bottomRows<3>() *= SCALE_XI_ROT_INVERSE;

  // scale change
  SE3 w2c_leftEps_P_x0 = (get_worldToCam_evalPT());
  w2c_leftEps_P_x0.translation() *= 1.00001;
  w2c_leftEps_P_x0 = w2c_leftEps_P_x0 * get_worldToCam_evalPT().inverse();

  SE3 w2c_leftEps_M_x0 = (get_worldToCam_evalPT());
  w2c_leftEps_M_x0.translation() /= 1.00001;
  w2c_leftEps_M_x0 = w2c_leftEps_M_x0 * get_worldToCam_evalPT().inverse();

  //// set nullspaces scale.
  nullspaces_scale = (w2c_leftEps_P_x0.log() - w2c_leftEps_M_x0.log())/(2e-3);

  nullspaces_affine.setZero();
  nullspaces_affine.topLeftCorner<2,1>()  = Vec2(1,0);
  assert(ab_exposure > 0);
  nullspaces_affine.topRightCorner<2,1>() = Vec2(0, expf(aff_g2l_0().a)*ab_exposure);
};

void FrameHessian::release() {
  // DELETE POINT
  // DELETE RESIDUAL
  for(unsigned int i=0;i<pointHessians.size();i++) delete pointHessians[i];
  for(unsigned int i=0;i<pointHessiansMarginalized.size();i++) delete pointHessiansMarginalized[i];
  for(unsigned int i=0;i<pointHessiansOut.size();i++) delete pointHessiansOut[i];
  for(unsigned int i=0;i<immaturePoints.size();i++) delete immaturePoints[i];


  pointHessians.clear();
  pointHessiansMarginalized.clear();
  pointHessiansOut.clear();
  immaturePoints.clear();
}

/// calculate the pixel value and gradient of the pyramid image of each level.
void FrameHessian::makeImages(float* color, CalibHessian* HCalib) {

  /// create image values for each level, and storage space for image gradients.
  for(int i=0;i<pyrLevelsUsed;i++) {
    dIp[i] = new Eigen::Vector3f[wG[i]*hG[i]];
    absSquaredGrad[i] = new float[wG[i]*hG[i]];
  }

  /// turns out they point to the same place.
  dI = dIp[0];

  // make d0
  int w=wG[0];   /// 0 level width
  int h=hG[0];   /// 0 level height

  for(int i=0;i<w*h;i++)
    dI[i][0] = color[i];

  for(int lvl=0; lvl<pyrLevelsUsed; lvl++) {
    /// image size of this level.
    int wl = wG[lvl], hl = hG[lvl];

    Eigen::Vector3f* dI_l = dIp[lvl];
    float* dabs_l = absSquaredGrad[lvl];

    if(lvl>0) {
      int lvlm1 = lvl-1;
      int wlm1 = wG[lvlm1];  /// number of columns.
      Eigen::Vector3f* dI_lm = dIp[lvlm1];

      /// pixels 0.25 to generate pyramids.
      for(int y=0;y<hl;y++)
        for(int x=0;x<wl;x++) {
          dI_l[x + y*wl][0] = 0.25f * (dI_lm[2*x   + 2*y*wlm1][0] +
                                       dI_lm[2*x+1 + 2*y*wlm1][0] +
                                       dI_lm[2*x   + 2*y*wlm1+wlm1][0] +
                                       dI_lm[2*x+1 + 2*y*wlm1+wlm1][0]);
        }
    }

    /// the second line starts
    for(int idx=wl;idx < wl*(hl-1);idx++) {
      float dx = 0.5f*(dI_l[idx+1][0] - dI_l[idx-1][0]);
      float dy = 0.5f*(dI_l[idx+wl][0] - dI_l[idx-wl][0]);

      if(!std::isfinite(dx)) dx=0;
      if(!std::isfinite(dy)) dy=0;

      /// gradient.
      dI_l[idx][1] = dx;
      dI_l[idx][2] = dy;

      /// sqaured gradient.
      dabs_l[idx] = dx*dx + dy*dy;

      if(setting_gammaWeightsPixelSelect==1 && HCalib!=0) {
        /// multiply by the response function, and change back to normal color, because I = G^-1(I) / V(x) when photometric correction.
        float gw = HCalib->getBGradOnly((float)(dI_l[idx][0]));
        dabs_l[idx] *= gw*gw;	// convert to gradient of original color space (before removing response).
      }
    }
  }
}

/// calculate relative pose (Tth) before and after optimization, change in relative photometric params (a,b), and intermediate variables.
void FrameFramePrecalc::set(FrameHessian* host, FrameHessian* target, CalibHessian* HCalib) {
  // printf("whether this->host is NULL: yes is 1, no is 0. Answer: %x\n", this);

  /// this is an assignment, the count will increase, not a copy.
  this->host = host;
  this->target = target;

  //// Tth = Ttw * Thw.inv = Ttw * Twh
  /// ? I really don't understand the meaning of the name leftToLeft_0
  /// optimize pose transformation b/w host and targets before optimization.
  SE3 leftToLeft_0 = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();
  PRE_RTll_0 = (leftToLeft_0.rotationMatrix()).cast<float>();
  PRE_tTll_0 = (leftToLeft_0.translation()).cast<float>();

  //// Tth = Ttw * Twh
  /// position transformation b/w host and trarget after optimization.
  SE3 leftToLeft = target->PRE_worldToCam * host->PRE_camToWorld;
  PRE_RTll = (leftToLeft.rotationMatrix()).cast<float>();
  PRE_tTll = (leftToLeft.translation()).cast<float>();
  distanceLL = leftToLeft.translation().norm();

  /// mutliply the internal parameter, the middle amount?
  Mat33f K = Mat33f::Zero();
  K(0,0) = HCalib->fxl();
  K(1,1) = HCalib->fyl();
  K(0,2) = HCalib->cxl();
  K(1,2) = HCalib->cyl();
  K(2,2) = 1;

  PRE_KRKiTll = K*PRE_RTll*K.inverse();
  PRE_RKiTll = PRE_RTll*K.inverse();
  PRE_KtTll = K*PRE_tTll;

  /// photometric affine value.
  PRE_aff_mode = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2l(), target->aff_g2l()).cast<float>();
  PRE_b0_mode = host->aff_g2l_0().b;
}

}

