#include "dso_g2o_edge.h"

namespace dso {

void EdgeLBASE3PosePhotoIdepthCamDSO::computeError() {
  const VertexSE3PoseDSO* vertex_pose = static_cast<const VertexSE3PoseDSO*>(_vertices[0]);
  const VertexPhotometricDSO* vertex_photo = static_cast<const VertexPhotometricDSO*>(_vertices[1]);
  VertexInverseDepthDSO* vertex_idepth = static_cast<VertexInverseDepthDSO*>(_vertices[2]);
  const VertexCamDSO* vertex_cam = static_cast<const VertexCamDSO*>(_vertices[3]);

  double idepth = vertex_idepth->estimate();

  Vec4 vecK = vertex_cam->estimate();
  double fx = vecK(0);
  double fy = vecK(1);
  double cx = vecK(2);
  double cy = vecK(3);
  Mat33f K;
  K << fx,0,cx,
      0,fy,cy,
      0,0,1;

  SE3 Ttw = r_->target->PRE_worldToCam;
  //// Tth = Ttw * Twh
  SE3 Tth = Ttw * vertex_pose->estimate();

  Mat33f R = (Tth.rotationMatrix()).cast<float>();
  Vec3f t = (Tth.translation()).cast<float>();

  //----------------
  float energyLeft=0;
  float wJI2_sum=0;
  //----------------

  for(int idx=0; idx<patternNum; idx++) {
    double u_host = r_->point->u + patternP[idx][0];
    double v_host = r_->point->v + patternP[idx][1];

    Vec3f Klip = Vec3f((u_host - cx)/fx,
                       (v_host - cy)/fy,
                       1);

    Vec3f ptp = R*Klip + t*idepth;

    //// drescale = idepth_new / idepth_old
    double drescale = 1.0f / ptp[2];

    if(drescale <= 0) {
      r_->state_NewState = ResState::OOB;
      _error.setZero();
      return;
    }
    double new_idepth = idepth*drescale;

    double _u = ptp[0]*drescale;
    double _v = ptp[1]*drescale;
    double _Ku = _u*fx + cx;
    double _Kv = _v*fy + cy;

    if(util::CheckBoundary(_Ku, _Kv, wG[0]-3, hG[0]-3)) {
      r_->state_NewState = ResState::OOB;
      _error.setZero();
      this->setLevel(1);
      return;
    }
    else if(u_host == r_->point->u && v_host == r_->point->v) {
      //// set CenterProjectedTo_ variable.
      vertex_idepth->SetCenterProjectedTo(_Ku, _Kv, new_idepth);
    }

    const Eigen::Vector3f* dIl = r_->target->dI;

    // interpolate on new frame.
    // hitcolor ([0]: intensity, [1]: dx, [2]: dy)
    Vec3f hitcolor = getInterpolatedElement33(dIl, _Ku, _Kv, wG[0]);

    // check if intensity is invalid.
    if(!std::isfinite((float)hitcolor[0])) {
      r_->state_NewState = ResState::OOB;
      _error(idx) = 0;
      continue;
    }

#if 1
    // photometric parameter.
    AffLight a0b0 = vertex_photo->estimate();
    AffLight a1b1 = r_->target->aff_g2l();
    double ab_exposure_host = r_->host->ab_exposure;
    double ab_exposure_target = r_->target->ab_exposure;

    Vec2f ab = AffLight::fromToVecExposure(ab_exposure_host, ab_exposure_target, a0b0, a1b1).cast<float>();
#else
    Vec2f ab;
    ab<<1,0;
#endif

    // calculate photometric error.
    _error(idx) = hitcolor[0] - (ab[0]*_measurement(idx) + ab[1]);

#if 0
    std::cout << "_error("<<idx<<"): " << _error(idx) << std::endl;
    std::cout << "hitcolor[0]: " << hitcolor[0] << ", ab[0]: " << ab[0] << ", _measurement("<<idx<<"): "<<_measurement(idx) << ", ab[1]: " << ab[1] << std::endl;
#endif

    /// weight proportional to gradient size
    float w = sqrtf(setting_outlierTHSumComponent / (setting_outlierTHSumComponent + hitcolor.tail<2>().squaredNorm()));
    w = 0.5f*(w + r_->point->weights[idx]);

    /// huber function, energy value (chi2)
    float hw = fabsf((float)_error(idx)) < setting_huberTH ? 1 : setting_huberTH / fabsf((float)_error(idx));
    energyLeft += w*w*hw*_error(idx)*_error(idx)*(2 - hw);
    /// squared gradient.
    wJI2_sum += hw*hw*(hitcolor[1]*hitcolor[1] + hitcolor[2]*hitcolor[2]);
  }

  r_->state_NewEnergyWithOutlier = energyLeft;

  /// is greater than the threshold value.
  if(energyLeft > std::max<float>(r_->host->frameEnergyTH, r_->target->frameEnergyTH) || wJI2_sum < 2)
  {
    energyLeft = std::max<float>(r_->host->frameEnergyTH, r_->target->frameEnergyTH);
    r_->state_NewState = ResState::OUTLIER;
  }
  else {
    r_->state_NewState = ResState::IN;
  }
  r_->state_NewEnergy = energyLeft;
}

void EdgeLBASE3PosePhotoIdepthCamDSO::linearizeOplus() {
  if(level() == 1 || r_->state_NewState == ResState::OOB) {
    return;
  }

  const VertexSE3PoseDSO* vertex_pose = static_cast<const VertexSE3PoseDSO*>(_vertices[0]);
  const VertexPhotometricDSO* vertex_photo = static_cast<const VertexPhotometricDSO*>(_vertices[1]);
  VertexInverseDepthDSO* vertex_idepth = static_cast<VertexInverseDepthDSO*>(_vertices[2]);
  const VertexCamDSO* vertex_cam = static_cast<const VertexCamDSO*>(_vertices[3]);

  Eigen::Matrix<double,2,4> J_dp2_dC;
  Eigen::Matrix<double,1,2> J_dr_dp2;
  Eigen::Matrix<double,8,4> J_dr_dC;

  Eigen::Matrix<double,8,6> J_dr_dxi;

  Eigen::Matrix<double,8,2> J_dr_dphoto;

  Eigen::Matrix<double,8,1> J_dr_didepth;
  float H_idepth_idepth = 0;

  double idepth = vertex_idepth->estimate();

  Vec4 vecK = vertex_cam->estimate();
  double fx = vecK(0);
  double fy = vecK(1);
  double cx = vecK(2);
  double cy = vecK(3);
  Mat33f K;
  K << fx,0,cx,
      0,fy,cy,
      0,0,1;

  SE3 Ttw = r_->target->PRE_worldToCam;
  //// Tth = Ttw * Twh
  SE3 Tth = Ttw * vertex_pose->estimate();

  Mat33f R = (Tth.rotationMatrix()).cast<float>();
  Vec3f t = (Tth.translation()).cast<float>();

  for(int idx=0; idx<patternNum; idx++) {
    double u_host = r_->point->u + patternP[idx][0];
    double v_host = r_->point->v + patternP[idx][1];

    Vec3f Klip = Vec3f((u_host - cx)/fx,
                       (v_host - cy)/fy,
                       1);

    Vec3f ptp = R*Klip + t*idepth;

    //// drescale = idepth_new / idepth_old
    double drescale = 1.0f / ptp[2];

    if(drescale <= 0) {
      r_->state_NewState = ResState::OOB;
      return;
    }
    double new_idepth = idepth*drescale;

    double _u = ptp[0]*drescale;
    double _v = ptp[1]*drescale;
    double _Ku = _u*fx + cx;
    double _Kv = _v*fy + cy;

    if(util::CheckBoundary(_Ku, _Kv, wG[0]-3, hG[0]-3)) {
      r_->state_NewState = ResState::OOB;
      return;
    }

    const Eigen::Vector3f* dIl = r_->target->dI;

    // hitcolor ([0]: intensity, [1]: dx, [2]: dy)
    Vec3f hitcolor = getInterpolatedElement33(dIl, _Ku, _Kv, wG[0]);
    // check if intensity is invalid.
    if(!std::isfinite((float)hitcolor[0])) {
      r_->state_NewState = ResState::OOB;
      return;
    }

    // camera intrinsics----------------------------------------
    double fxi = 1/fx;
    double fyi = 1/fy;

    J_dr_dp2(0,0) = hitcolor[1];
    J_dr_dp2(0,1) = hitcolor[2];

    // x
    J_dp2_dC(0,2) = drescale*(R(2,0)*_u - R(0,0));
    J_dp2_dC(0,3) = fx*fyi*drescale*(R(2,1)*_u - R(0,1));
    J_dp2_dC(0,0) = Klip[0]*J_dp2_dC(0,2);
    J_dp2_dC(0,1) = Klip[1]*J_dp2_dC(0,3);
    // y
    J_dp2_dC(1,2) = fy*fxi*drescale*(R(2,0)*_v - R(1,0));
    J_dp2_dC(1,3) = drescale*(R(2,1)*_v - R(1,1));
    J_dp2_dC(1,0) = Klip[0]*J_dp2_dC(1,2);
    J_dp2_dC(1,1) = Klip[1]*J_dp2_dC(1,3);

    //// 8*(1x4) = 8*(1x2 * 2x4)
    J_dr_dC.block<1,4>(idx,0) = J_dr_dp2 * J_dp2_dC;
    // ---------------------------------------------------------

    // pose-----------------------------------------------
    double dx = hitcolor[1]*fx;
    double dy = hitcolor[2]*fy;

    /// translation part.
    J_dr_dxi(idx,0) = new_idepth * dx;
    J_dr_dxi(idx,1) = new_idepth * dy;
    J_dr_dxi(idx,2) = -new_idepth * (_u*dx + _v*dy);
    /// rotation part.
    J_dr_dxi(idx,3) = -(_u*_v*dx + (1 + _v*_v)*dy);
    J_dr_dxi(idx,4) = _u*_v*dy + (1 + _u*_u)*dx;
    J_dr_dxi(idx,5) = _u*dy - _v*dx;
    // ---------------------------------------------------------

    // photometric parameters a,b-------------------------------
    // calculate jacobian of affine brightness function.
    AffLight a0b0 = vertex_photo->estimate();
    AffLight a1b1 = r_->target->aff_g2l();

    double ab_exposure_host = r_->host->ab_exposure;
    double ab_exposure_target = r_->target->ab_exposure;

    Vec2f ab = AffLight::fromToVecExposure(ab_exposure_host, ab_exposure_target, a0b0, a1b1).cast<float>();

    J_dr_dphoto(idx,0) = ab[0]*(b0_ - _measurement(idx));
    J_dr_dphoto(idx,1) = -1;
    // ---------------------------------------------------------

    // inverse depth------------------------------
    // jacobian for inverse depth. dr/dinvd
    J_dr_didepth(idx,0) = dx*drescale*(t[0] - t[2]*_u) + dy*drescale*(t[1] - t[2]*_v);
    // ---------------------------------------------------------

    H_idepth_idepth += J_dr_didepth(idx,0) * J_dr_didepth(idx,0);
  }

  if(H_idepth_idepth < 1e-10) {
    H_idepth_idepth = 1e-10;
  }
  r_->point->idepth_hessian = H_idepth_idepth;

#if 0
  std::cout << "J_dr_dxi: " << J_dr_dxi << std::endl;
  std::cout << "J_dr_dphoto: " << J_dr_dphoto << std::endl;
  std::cout << "J_dr_didepth: " << J_dr_didepth << std::endl << std::endl;
#endif

  _jacobianOplus[0] = J_dr_dxi;
  _jacobianOplus[1] = J_dr_dphoto;
  _jacobianOplus[2] = J_dr_didepth;
  _jacobianOplus[3] = J_dr_dC;
}

#if 0
void EdgeLBASE3PosePhotoIdepthCamDSO::projectPoint() {
  const VertexSE3PoseDSO* vertex_pose = static_cast<const VertexSE3PoseDSO*>(_vertices[0]);
  const VertexCamDSO* vertex_cam = static_cast<const VertexCamDSO*>(_vertices[3]);
  VertexInverseDepthDSO* vertex_idepth = static_cast<VertexInverseDepthDSO*>(_vertices[2]);

  SE3 Ttw = r_->target->get_worldToCam_evalPT();
  //// Tth = Ttw * Twh
  SE3 Tth = Ttw * vertex_pose->estimate();
  R = (Tth.rotationMatrix()).cast<float>();
  t = (Tth.translation()).cast<float>();

  double idepth = vertex_idepth->estimate();
  Vec4 vecK = vertex_cam->estimate();
  double fx = vecK(0);
  double fy = vecK(1);
  double cx = vecK(2);
  double cy = vecK(3);

  Mat33f K;
  K << fx,0,cx,
      0,fy,cy,
      0,0,1;

  Mat33f KRKi = K*R*K.inverse();
  Vec3f Kt = K*t;

  double u_host = r_->point->u;
  double v_host = r_->point->v;

  Klip_ = Vec3f((u_host - cx)/fx,
                (v_host - cy)/fy,
                1);

  Vec3f ptp = R*Klip_ + t*idepth;

  //// drescale = idepth_new / idepth_old
  drescale_ = 1.0f / ptp[2];

  if(drescale_ <= 0) {
    r_->state_NewState = ResState::OOB;
    return;
  }
  new_idepth_ = idepth*drescale_;

  double _u = ptp[0] * drescale_;
  double _v = ptp[1] * drescale_;
  double _Ku = _u*fx + cx;
  double _Kv = _v*fy + cy;

  if(util::CheckBoundary(_Ku, _Kv, wG[0]-3, hG[0]-3)) {
    r_->state_NewState = ResState::OOB;
    // set as outlier if out of boundary.
    this->setLevel(1);

    u_.at(0) = -1;
    v_.at(0) = -1;
    Ku_.at(0) = -1;
    Kv_.at(0) = -1;
    return;
  }
  else {
    //// set CenterProjectedTo_ variable.
    vertex_idepth->SetCenterProjectedTo(_Ku, _Kv, new_idepth_);
  }

  for(int idx=0; idx<patternNum; idx++) {
#if 0
    Klip_ = Vec3f((u_host + patternP[idx][0] - cx)/fx,
                  (v_host + patternP[idx][1] - cy)/fy,
                  1);
    Vec3f ptp = R*Klip_ + t*idepth;

    double _u = ptp[0] / ptp[2];
    double _v = ptp[1] / ptp[2];
    double _Ku = _u*fx + cx;
    double _Kv = _v*fy + cy;
#else
    double _u = u_host + patternP[idx][0];
    double _v = v_host + patternP[idx][1];

    Vec3f ptp = KRKi*Vec3f(_u, _v, 1) + Kt*idepth;

    double _Ku = ptp[0] / ptp[2];
    double _Kv = ptp[1] / ptp[2];
#endif

    if(util::CheckBoundary(_Ku, _Kv, wG[0]-3, hG[0]-3)) {
      r_->state_NewState = ResState::OOB;
      // set as outlier if out of boundary.
      this->setLevel(1);

      u_.at(idx) = -1;
      v_.at(idx) = -1;
      Ku_.at(idx) = -1;
      Kv_.at(idx) = -1;
    }
    else {
      r_->state_NewState = ResState::IN;

      u_.at(idx) = _u;
      v_.at(idx) = _v;
      Ku_.at(idx) = _Ku;
      Kv_.at(idx) = _Kv;
    }
  }
}
#endif

//===============================================================================

void EdgeSE3PosePhotoDSO::computeError() {
  const VertexSE3PoseDSO* vertex_pose = static_cast<const VertexSE3PoseDSO*>(_vertices[0]);
  const VertexPhotometricDSO* vertex_photo = static_cast<const VertexPhotometricDSO*>(_vertices[1]);

  Vec3 Xcurr = vertex_pose->estimate() * Vec3(Xref_[0], Xref_[1], Xref_[2]);
  Vec2 uv = util::ProjectToImageCoordinate(Xcurr, level_);

  // check boundary.
  if(util::CheckBoundary(uv, wl_, hl_)) {
    _error(0,0) = 0.0;
    return;
  }

  // photometric parameter.
  AffLight a1b1 = vertex_photo->estimate();
  Vec2f ab = AffLight::fromToVecExposure(ab_exposure_ref_, ab_exposure_curr_, a0b0_, a1b1).cast<float>();

  // interpolate on new frame.
  // hitcolor ([0]: intensity, [1]: dx, [2]: dy)
  Vec3f hitcolor = getInterpolatedElement33(dINewl_, uv(0), uv(1), wl_);

  // check if intensity is invalid.
  if(!std::isfinite((float)hitcolor[0])) {
    return;
  }

  // calculate photometric error.
  _error(0,0) = hitcolor[0] - (ab[0]*_measurement + ab[1]);
}

void EdgeSE3PosePhotoDSO::linearizeOplus() {
  const VertexSE3PoseDSO* vertex_pose = static_cast<VertexSE3PoseDSO*>(_vertices[0]);
  Vec3 Xcurr = vertex_pose->estimate() * Vec3(Xref_[0], Xref_[1], Xref_[2]);
  double fx = KG[level_](0,0);
  double fy = KG[level_](1,1);

  double x = Xcurr[0];
  double y = Xcurr[1];
  double invz = 1.0/Xcurr[2];

  Vec2 uv = util::ProjectToImageCoordinate(Xcurr, level_);

  // check boundary.
  if(util::CheckBoundary(uv, wl_, hl_)) {
    _jacobianOplusXi.setZero();
    _jacobianOplusXj.setZero();
    return;
  }

  // hitcolor ([0]: intensity, [1]: dx, [2]: dy)
  Vec3f hitcolor = getInterpolatedElement33(dINewl_, uv(0), uv(1), wl_);

#if 0
  // VERSION1 (3D point version)
  double invz_2 = invz * invz;
  // jacobian from se3 to u,v
  // NOTE that in g2o the Lie algebra is (\omega, \epsilon), where \omega is so(3) and \epsilon the translation
  Eigen::Matrix<double, 2, 6> jacobian_uv_ksai;

  jacobian_uv_ksai(0,0) = invz *fx;
  jacobian_uv_ksai(0,1) = 0;
  jacobian_uv_ksai(0,2) = -x*invz_2 *fx;
  jacobian_uv_ksai(0,3) = -x*y*invz_2 *fx;
  jacobian_uv_ksai(0,4) = (1+(x*x*invz_2) ) *fx;
  jacobian_uv_ksai(0,5) = -y*invz *fx;

  jacobian_uv_ksai(1,0) = 0;
  jacobian_uv_ksai(1,1) = invz *fy;
  jacobian_uv_ksai(1,2) = -y*invz_2 *fy;
  jacobian_uv_ksai(1,3) = -(1+y*y*invz_2) *fy;
  jacobian_uv_ksai(1,4) = x*y*invz_2 *fy;
  jacobian_uv_ksai(1,5) = x*invz *fy;

  Eigen::Matrix<double, 1, 2> jacobian_pixel_uv;

  jacobian_pixel_uv(0,0) = hitcolor[1];
  jacobian_pixel_uv(0,1) = hitcolor[2];

  _jacobianOplusXi = jacobian_pixel_uv * jacobian_uv_ksai;
#else
  // VERSION2 (inverse depth version)
  double u = x*invz;
  double v = y*invz;
  double dx = hitcolor[1]*fx;
  double dy = hitcolor[2]*fy;

  // calculate jacobian of pose.
  /// translation part.
  _jacobianOplusXi(0,0) = invz * dx;
  _jacobianOplusXi(0,1) = invz * dy;
  _jacobianOplusXi(0,2) = -invz * (u*dx + v*dy);
  /// rotation part.
  _jacobianOplusXi(0,3) = -(u*v*dx + (1 + v*v)*dy);
  _jacobianOplusXi(0,4) = u*v*dy + (1 + u*u)*dx;
  _jacobianOplusXi(0,5) = u*dy - v*dx;
#endif

  // photometric parameters.
  const VertexPhotometricDSO* vertex_photo = static_cast<const VertexPhotometricDSO*>(_vertices[1]);
  AffLight a1b1 = vertex_photo->estimate();
  Vec2f ab = AffLight::fromToVecExposure(ab_exposure_ref_, ab_exposure_curr_, a0b0_, a1b1).cast<float>();

  // calculate jacobian of affine brightness function.
  _jacobianOplusXj(0,0) = ab[0] * (a0b0_.b - _measurement);
  _jacobianOplusXj(0,1) = -1;
}

//===============================================================================

void EdgePointActivationIdepthDSO::projectPoint() {
  const VertexInverseDepthDSO* vertex_idepth = static_cast<const VertexInverseDepthDSO*>(_vertices[0]);
  double idepth = vertex_idepth->estimate();

  Klip_ = Vec3f((u_pt_ - HCalib_->cxl())*HCalib_->fxli(),
                (v_pt_ - HCalib_->cyl())*HCalib_->fyli(),
                1);

  Vec3f ptp = R_ * Klip_ + t_ * idepth;

  drescale_ = 1.0f / ptp[2];
  new_idepth_ = idepth * drescale_;

  if(drescale_ <= 0) {
    return;
  }

  u_ = ptp[0] * drescale_;
  v_ = ptp[1] * drescale_;

  Ku_ = u_*HCalib_->fxl() + HCalib_->cxl();
  Kv_ = v_*HCalib_->fyl() + HCalib_->cyl();

  if(util::CheckBoundary(Ku_, Kv_, wG[0]-3, hG[0]-3)) {
    // set as outlier if out of boundary.
    setLevel(1);
    return;
  }
}

void EdgePointActivationIdepthDSO::computeError() {
  // check boundary.
  if(util::CheckBoundary(Ku_, Kv_, wG[0]-3, hG[0]-3)) {
    _error(0,0) = 0.0;
    return;
  }

  // interpolate on new frame.
  // hitcolor ([0]: intensity, [1]: dx, [2]: dy)
  Vec3f hitcolor = getInterpolatedElement33(dIl_, Ku_, Kv_, wG[0]);

  // check if intensity is invalid.
  if(!std::isfinite((float)hitcolor[0])) {
    return;
  }

  // calculate photometric error.
  _error(0,0) = hitcolor[0] - (affLL_[0]*_measurement + affLL_[1]);
}

void EdgePointActivationIdepthDSO::linearizeOplus() {
  if(util::CheckBoundary(Ku_, Kv_, wG[0]-3, hG[0]-3)) {
    return;
  }

  // hitcolor ([0]: intensity, [1]: dx, [2]: dy)
  Vec3f hitcolor = getInterpolatedElement33(dIl_, Ku_, Kv_, wG[0]);

  double dx = hitcolor[1]*HCalib_->fxl();
  double dy = hitcolor[2]*HCalib_->fyl();

  // jacobian for inverse depth. dr/dinvd
  _jacobianOplusXi(0,0) = dx*drescale_*(t_[0] - t_[2]*u_) + dy*drescale_*(t_[1] - t_[2]*v_);
}

//===============================================================================

void EdgeTracePointUVDSO::computeError() {
  const VertexUVDSO* vertex_uv = static_cast<const VertexUVDSO*>(_vertices[0]);
  double bestU = vertex_uv->estimate()(0);
  double bestV = vertex_uv->estimate()(1);

  // check boundary.
  if(util::CheckBoundary(bestU, bestV, wG[0]-3, hG[0]-3)) {
    _error(0,0) = 0.0;
    return;
  }

  // interpolate on new frame.
  // hitcolor ([0]: intensity, [1]: dx, [2]: dy)
  Vec3f hitcolor = getInterpolatedElement33(dI_,
                                            bestU + rotatePattern_[0],
                                            bestV + rotatePattern_[1],
                                            wG[0]);

  // check if intensity is invalid.
  if(!std::isfinite(hitcolor[0])) {
    return;
  }

  // calculate photometric error.
  _error(0,0) = hitcolor[0] - (affLL_[0]*_measurement + affLL_[1]);
}

void EdgeTracePointUVDSO::linearizeOplus() {
  const VertexUVDSO* vertex_uv = static_cast<const VertexUVDSO*>(_vertices[0]);
  double bestU = vertex_uv->estimate()(0);
  double bestV = vertex_uv->estimate()(1);

  if(util::CheckBoundary(bestU, bestV, wG[0]-3, hG[0]-3)) {
    return;
  }

  // hitcolor ([0]: intensity, [1]: dx, [2]: dy)
  Vec3f hitcolor = getInterpolatedElement33(dI_,
                                            bestU + rotatePattern_[0],
                                            bestV + rotatePattern_[1],
                                            wG[0]);

  if(!std::isfinite(hitcolor[0])) {
    return;
  }

  // jacobian for inverse depth. dI/dp
  _jacobianOplusXi(0,0) = dx_*hitcolor[1] + dy_*hitcolor[2];
}
} // namespace dso
