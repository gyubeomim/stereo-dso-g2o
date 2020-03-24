#ifndef DSO_UTIL_H_
#define DSO_UTIL_H_

#include "util/globalCalib.h"

namespace dso {
namespace util {

/// \brief Projection from 3D point to 2D point.
inline Vec2 ProjectToImageCoordinate(Vec3& Xcurr, int level) {
  double fx = KG[level](0,0);
  double fy = KG[level](1,1);
  double cx = KG[level](0,2);
  double cy = KG[level](1,2);

  Vec2 res;
  res(0) = fx*(Xcurr(0)/Xcurr(2)) + cx;
  res(1) = fy*(Xcurr(1)/Xcurr(2)) + cy;

  return res;

}

/// \brief Check image boundary. (true: pixel is outside border.)
inline bool CheckBoundary(const Vec2 uv, int wl, int hl) {
  double u = uv(0);
  double v = uv(1);

  if((u-2)<0 || (u+3)>wl || (v-2)<0 || (v+3)>hl) {
    return true;
  }

  return false;
}

/// \brief Check image boundary. (true: pixel is outside border.)
inline bool CheckBoundary(const double u, const double v, int wl, int hl) {
  if((u-2)<0 || (u+3)>wl || (v-2)<0 || (v+3)>hl) {
    //// Outlier.
    return true;
  }

  //// Inlier.
  return false;
}

/// \brief Convert radian to degree.
inline double R2D(double rad) {
  // rad to deg.
  const double PI = std::atan(1.0)*4;
  return rad * 180 / PI;
}

} // namespace util
} // namespace dso

#endif /* DSO_UTIL_H_ */
