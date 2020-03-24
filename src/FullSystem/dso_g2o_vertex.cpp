#include "dso_g2o_vertex.h"

namespace dso {

VertexSE3PoseDSO::VertexSE3PoseDSO()
{}

bool VertexSE3PoseDSO::read(std::istream& is) { return false; }
bool VertexSE3PoseDSO::write(std::ostream& os) const { return false; }

void VertexSE3PoseDSO::setToOriginImpl() {
  _estimate = SE3();
}

void VertexSE3PoseDSO::oplusImpl(const number_t* update_) {
  Eigen::Map<const Vec6> update(update_);
  _estimate = SE3::exp(update) * estimate();
}

//===================================================================

VertexPhotometricDSO::VertexPhotometricDSO() {}
bool VertexPhotometricDSO::read(std::istream& is) { return false; }
bool VertexPhotometricDSO::write(std::ostream& os) const { return false; }

void VertexPhotometricDSO::setToOriginImpl() {
  _estimate = AffLight();
}

void VertexPhotometricDSO::oplusImpl(const number_t* update_) {
  Eigen::Map<const Vec2> update(update_);

#if 0
  _estimate.a = update(0);
  _estimate.b = update(1);
#else
  _estimate.a += update(0);
  _estimate.b += update(1);
#endif
}

//===================================================================

VertexInverseDepthDSO::VertexInverseDepthDSO() {
  CenterProjectedTo_ = Vec3f(2,2,0);
}

bool VertexInverseDepthDSO::read(std::istream& is) { return false; }

bool VertexInverseDepthDSO::write(std::ostream& os) const { return false; }

void VertexInverseDepthDSO::setToOriginImpl() {
  _estimate = 0.0;
}

void VertexInverseDepthDSO::oplusImpl(const number_t* update_) {
  _estimate += (double)*update_;
}

void VertexInverseDepthDSO::SetCenterProjectedTo(float Ku, float Kv, float new_idepth) {
  CenterProjectedTo_ = Vec3f(Ku, Kv, new_idepth);
}
//===================================================================

VertexUVDSO::VertexUVDSO() {}
bool VertexUVDSO::read(std::istream& is) { return false; }
bool VertexUVDSO::write(std::ostream& os) const { return false; }

void VertexUVDSO::setToOriginImpl() {
  _estimate = Vec2();
}

void VertexUVDSO::oplusImpl(const number_t* update_) {
  double update = (double)*update_;

  if(update < -0.5) {
    update = -0.5;
  }
  else if(update > 0.5) {
    update = 0.5;
  }
  else if(!std::isfinite(update)) {
    update = 0;
  }

  _estimate(0) += update*dx_;
  _estimate(1) += update*dy_;
}

//===================================================================

VertexCamDSO::VertexCamDSO() {}
bool VertexCamDSO::read(std::istream& is) { return false; }
bool VertexCamDSO::write(std::ostream& os) const { return false; }

void VertexCamDSO::setToOriginImpl() {
  _estimate = Eigen::Vector4d::Zero();
}

void VertexCamDSO::oplusImpl(const number_t* update_) {
  Eigen::Map<const Vec4> update(update_);
  _estimate(0) += update(0);  // fx
  _estimate(1) += update(1);  // fy
  _estimate(2) += update(2);  // cx
  _estimate(3) += update(3);  // cy
}

}  // namespace dso
