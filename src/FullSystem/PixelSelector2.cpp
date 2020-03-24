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


#include "FullSystem/PixelSelector2.h"
 
// 



#include "util/NumType.h"
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include "FullSystem/HessianBlocks.h"
#include "util/globalFuncs.h"

namespace dso
{


PixelSelector::PixelSelector(int w, int h)
{
  randomPattern = new unsigned char[w*h];
  std::srand(3141592);	// want to be deterministic.
  for(int i=0;i<w*h;i++) randomPattern[i] = rand() & 0xFF;

  currentPotential=3;

  /// 32x32 blocks to calculate the threshold.
  gradHist = new int[100*(1+w/32)*(1+h/32)];
  ths = new float[(w/32)*(h/32)+100];
  thsSmoothed = new float[(w/32)*(h/32)+100];

  allowFast=false;
  gradHistFrame=0;
}

PixelSelector::~PixelSelector()
{
  delete[] randomPattern;
  delete[] gradHist;
  delete[] ths;
  delete[] thsSmoothed;
}

/// gradient value taking below% as threshold.
int computeHistQuantil(int* hist, float below) {
  /// minimum number of pixels.
  int th = hist[0]*below + 0.5f;

  /// 90? so casual...
  for(int i=0; i<90; i++) {
    /// all pixels with graident 0-i account for below%
    th -= hist[i+1];
    if(th<0) {
      return i;
    }
  }

  return 90;
}

/// generate graident histogram, calculate threshold for each block.
void PixelSelector::makeHists(const FrameHessian* const fh) {
  gradHistFrame = fh;

  /// summation of squared gradients at level 0.
  float * mapmax0 = fh->absSquaredGrad[0];

  /// width and height.
  int w = wG[0];
  int h = hG[0];

  /// still each block size is 32x32, not 32x32 grids in the paper.
  int w32 = w/32;
  int h32 = h/32;
  thsStep = w32;

  for(int y=0;y<h32;y++)
    for(int x=0;x<w32;x++) {
      /// grid of y rows and x columns.
      float* map0 = mapmax0 + 32*x + 32*y*w;
      int* hist0 = gradHist; // + 50*(x+y*w32);

      // divide into 50 cells.
      memset(hist0,0,sizeof(int)*50);

      for(int j=0;j<32;j++)
        for(int i=0;i<32;i++) {
          /// the entire image coordinates of the grid's (j,i) pixels.
          int it = i+32*x;
          int jt = j+32*y;

          /// inside.
          if(it>w-2 || jt>h-2 || it<1 || jt<1) continue;

          /// squared square root.
          int g = sqrtf(map0[i+j*w]);

          /// ? why is the number 48 because it is divied into 50 cells?
          if(g>48) {
            g=48;
          }

          /// 1-49 stores the number of cooresponding gradients.
          hist0[g+1]++;

          /// number of all pixels.
          hist0[0]++;
        }
      /// get the threshold of each block.
      ths[x+y*w32] = computeHistQuantil(hist0,setting_minGradHistCut) + setting_minGradHistAdd;
    }

  /// use 3x3 window to average to smooth.
  for(int y=0; y<h32 ;y++)
    for(int x=0; x<w32; x++) {
      float sum=0, num=0;

      if(x>0) {
        if(y>0) {
          num++;
          sum+=ths[x-1+(y-1)*w32];
        }
        if(y<h32-1) {
          num++;
          sum+=ths[x-1+(y+1)*w32];
        }
        num++; sum+=ths[x-1+(y)*w32];
      }

      if(x<w32-1) {
        if(y>0) {
          num++;
          sum+=ths[x+1+(y-1)*w32];
        }
        if(y<h32-1) {
          num++;
          sum+=ths[x+1+(y+1)*w32];
        }
        num++;
        sum+=ths[x+1+(y)*w32];
      }

      if(y>0) {
        num++;
        sum+=ths[x+(y-1)*w32];
      }
      if(y<h32-1) {
        num++;
        sum+=ths[x+(y+1)*w32];
      }
      num++;
      sum+=ths[x+y*w32];

      thsSmoothed[x+y*w32] = (sum/num) * (sum/num);
    }
}

/* *******************************
 * @ function:
 *
 * @param: fh frame Hessian data structure
 * @ map_out selected map points
 * @ density The number of points (density) required for each pyramid layer
 * @recursionsLeft maximum recursion
 * @ plot
 * @thFactor threshold factor, settingImaturePointDensity
 * @
 * @note: use recursion
****************************** */
int PixelSelector::makeMaps(const FrameHessian* const fh,
                            float* map_out,
                            float density,
                            int recursionsLeft,
                            bool plot,
                            float thFactor)
{
  float numHave=0;
  float numWant=density;
  float quotia;
  int idealPotential = currentPotential;

  {
    // the number of selected pixels behaves approximately as
    // K / (pot+1)^2, where K is a scene-dependent constant.
    // we will allow sub-selecting pixels by up to a quotia of 0.25, otherwise we will re-select.

    /// STEP1: without calculating the histogram and the threshld of the selected point, cell the function to generate the block threshold.
    if(fh != gradHistFrame) {
      /// first time in, if the frame of the gradient histogram is not fh, the histogram will be generated.
      makeHists(fh);
    }

    /// STEP2: select eligible pixels on the current frame.
    // select!
    Eigen::Vector3i n = this->select(fh, map_out,currentPotential, thFactor);

    // sub-select!
    /// select the obtained points.
    numHave = n[0]+n[1]+n[2];
    /// the ratio of obtained points.
    quotia = numWant / numHave;

    /// STEP3: calculate the number of pixels to be collected, the size of the range is equivalent to the dynamic grid, the smaller the pot, the more points will be obtained.
    // by default we want to over-sample by 40% just to be sure.
    float K = numHave * (currentPotential+1) * (currentPotential+1); /// equivalent to the area covered, each pixel correspoinds to a pot*pot;
    idealPotential = sqrtf(K/numWant)-1;	// round down.
    if(idealPotential<1) idealPotential=1;

    /// STEP4: the number you want and the number you have obtained will be resampled if it is greater than or less than 0.25
    if(recursionsLeft>0 && quotia>1.25 && currentPotential>1) {
      // re-sample to get more points!
      // potential needs to be smaller
      if(idealPotential>=currentPotential)    /// idealPotential should be small.
        idealPotential = currentPotential-1;  /// decrease, multiply points.

      //printf("PixelSelector: have %.2f%%, need %.2f%%. RESAMPLE with pot %d -> %d.\n",
      //100*numHave/(float)(wG[0]*hG[0]),
      //100*numWant/(float)(wG[0]*hG[0]),
      //currentPotential,
      //idealPotential);

      currentPotential = idealPotential;
      return makeMaps(fh,map_out, density, recursionsLeft-1, plot,thFactor); /// recursion.
    }
    else if(recursionsLeft>0 && quotia < 0.25) {
      // re-sample to get less points!
      if(idealPotential<=currentPotential)   /// idealPotential should be large.
        idealPotential = currentPotential+1; /// increase, take less points.

      //printf("PixelSelector: have %.2f%%, need %.2f%%. RESAMPLE with pot %d -> %d.\n",
      //100*numHave/(float)(wG[0]*hG[0]),
      //100*numWant/(float)(wG[0]*hG[0]),
      //currentPotential,
      //idealPotential);

      currentPotential = idealPotential;
      return makeMaps(fh,map_out, density, recursionsLeft-1, plot,thFactor);
    }
  }

  /// STEP5: there are still many extractions, and some points are randomly deleted.
  int numHaveSub = numHave;

  if(quotia < 0.95) {
    int wh=wG[0]*hG[0];
    int rn=0;
    unsigned char charTH = 255*quotia;
    for(int i=0;i<wh;i++)
    {
      if(map_out[i] != 0)
      {
        if(randomPattern[rn] > charTH )
        {
          map_out[i]=0;
          numHaveSub--;
        }
        rn++;
      }
    }
  }

  //printf("PixelSelector: have %.2f%%, need %.2f%%. KEEPCURR with pot %d -> %d. Subsampled to %.2f%%\n",
  //100*numHave/(float)(wG[0]*hG[0]),
  //100*numWant/(float)(wG[0]*hG[0]),
  //currentPotential,
  //idealPotential,
  //100*numHaveSub/(float)(wG[0]*hG[0]));

  currentPotential = idealPotential;

  /// draw the selection result.
  if(plot) {
    int w = wG[0];
    int h = hG[0];


    MinimalImageB3 img(w,h);

    for(int i=0; i<w*h; i++) {
      /// pixel value.
      float c = fh->dI[i][0]*0.7;
      if(c > 255) {
        c=255;
      }
      img.at(i) = Vec3b(c,c,c);
    }

    IOWrap::displayImage("Selector Image", &img);

    /// an zhao according to different layers of pixels, draw different colors.
    for(int y=0; y<h; y++)
      for(int x=0; x<w; x++) {
        int i=x+y*w;
        if(map_out[i] == 1)
          img.setPixelCirc(x,y,Vec3b(0,255,0));
        else if(map_out[i] == 2)
          img.setPixelCirc(x,y,Vec3b(255,0,0));
        else if(map_out[i] == 4)
          img.setPixelCirc(x,y,Vec3b(0,0,255));
      }
    IOWrap::displayImage("Selector Pixels", &img);
  }

  return numHaveSub;
}

/// ? is this selected point on a different level, or is it a different threshold, different block mentioned in the paper?
/* *******************************
 * @ function: Select pixels that meet the requirements on different layers according to the threshold
 *
 * @param: some information of fh frame
 * @ map_out selected pixels and layer
 * @ pot (currentPotential) The size of the selected point, one within a pot
 * @thFactor threshold factor (multiplier)
 *
 * @note: Returns the number of points selected for each layer
****************************** */
Eigen::Vector3i PixelSelector::select(const FrameHessian* const fh,
                                      float* map_out,
                                      int pot,
                                      float thFactor)
{
  /// const on *left: pointer content cannot be changed, on *right pointer cannot be changed.
  /// equiv const Eigen::Vector3f* const
  Eigen::Vector3f const * const map0 = fh->dI;

  /// sum of squared gradients of 0,1,2 levels.
  float * mapmax0 = fh->absSquaredGrad[0];
  float * mapmax1 = fh->absSquaredGrad[1];
  float * mapmax2 = fh->absSquaredGrad[2];

  /// image size of different layers.
  int w = wG[0];
  int w1 = wG[1];
  int w2 = wG[2];
  int h = hG[0];

  /// ? what is this for?
  /// ! randomly select gradients and thresholds in the 16 corresponding directions.
  /// ! the directions in each pot are randomly selected to prevent the same features and duplicates.
  /// modules are all 1.
  const Vec2f directions[16] = {
    Vec2f(0,    1.0000),
    Vec2f(0.3827,    0.9239),
    Vec2f(0.1951,    0.9808),
    Vec2f(0.9239,    0.3827),
    Vec2f(0.7071,    0.7071),
    Vec2f(0.3827,   -0.9239),
    Vec2f(0.8315,    0.5556),
    Vec2f(0.8315,   -0.5556),
    Vec2f(0.5556,   -0.8315),
    Vec2f(0.9808,    0.1951),
    Vec2f(0.9239,   -0.3827),
    Vec2f(0.7071,   -0.7071),
    Vec2f(0.5556,    0.8315),
    Vec2f(0.9808,   -0.1951),
    Vec2f(1.0000,    0.0000),
    Vec2f(0.1951,   -0.9808)};

  /// ? where do I change the status of PixelSelectorStatus?
  memset(map_out,0,w*h*sizeof(PixelSelectorStatus));

  /// reduce the multiple of the pyramid layer threshold.
  /// dw: Down Weights.
  float dw1 = setting_gradDownweightPerLevel;   // 0.75, second level.
  float dw2 = dw1*dw1; /// third lvel.

  //x4, y4; x3, y3; x2, y2; x1, y1, what's the difference?
  int n3=0, n2=0, n4=0;
  /// in the second level, select a point to traverse every pot.
  for(int y4=0; y4<h; y4+=(4*pot))
    for(int x4=0; x4<w; x4+=(4*pot)) {
      /// the size of the neighborhood of the point (take up 4pot or the remainder at the end).
      int my3 = std::min((4*pot), h-y4);
      int mx3 = std::min((4*pot), w-x4);

      int bestIdx4=-1; float bestVal4=0;

      /// random coefficient.
      /// take the lower 4 bits, 0-15, corresponding to directions.
      Vec2f dir4 = directions[randomPattern[n2] & 0xF];

      /// within the range of the above, iterate on the first layer, every other point.
      for(int y3=0; y3<my3; y3+=(2*pot))
        for(int x3=0; x3<mx3; x3+=(2*pot)) {
          /// corresponds to level 0 coordinates.
          int x34 = x3+x4;
          int y34 = y3+y4;

          /// continue to determine the neighborhood on this layer.
          int my2 = std::min((2*pot), h-y34);
          int mx2 = std::min((2*pot), w-x34);

          int bestIdx3=-1; float bestVal3=0;

          Vec2f dir3 = directions[randomPattern[n2] & 0xF];

          /// in the neighborhood above, transform to level 0, traverse every pot.
          /// ! the largest pixel in each pot size grid that is greater than the threshold.
          for(int y2=0; y2<my2; y2+=pot)
            for(int x2=0; x2<mx2; x2+=pot) {
              /// cordinates.
              int x234 = x2+x34;
              int y234 = y2+y34;

              int my1 = std::min(pot, h-y234);
              int mx1 = std::min(pot, w-x234);

              int bestIdx2=-1; float bestVal2=0;

              Vec2f dir2 = directions[randomPattern[n2] & 0xF];

              /// in level 0, traversal within the pot size neighborhood.
              for(int y1=0;y1<my1;y1+=1)
                for(int x1=0;x1<mx1;x1+=1) {
                  assert(x1+x234 < w);
                  assert(y1+y234 < h);

                  /// pixel id.
                  int idx = x1+x234 + w*(y1+y234);

                  /// pixel coordinates.
                  int xf = x1+x234;
                  int yf = y1+y234;

                  if(xf<4 || xf>=w-5 || yf<4 || yf>h-4) continue;

                  /// histogram to find the threshold, divide by 32 to determine which threshold range,
                  /// ! it can be determined that each grid is 32 grids in size.
                  float pixelTH0 = thsSmoothed[(xf>>5) + (yf>>5) * thsStep];
                  float pixelTH1 = pixelTH0*dw1; //DownweightPerPixel = 0.75
                  float pixelTH2 = pixelTH1*dw2;

                  /// gradient modulus of level 0.
                  float ag0 = mapmax0[idx];

                  if(ag0 > pixelTH0*thFactor) {
                    /// the derivative image is two.
                    Vec2f ag0d = map0[idx].tail<2>();
                    /// determine by the graident in this direction.
                    float dirNorm = fabsf((float)(ag0d.dot(dir2)));

                    if(!setting_selectDirectionDistribution) dirNorm = ag0;

                    /// take the largest graident.
                    if(dirNorm > bestVal2) {
                      bestVal2 = dirNorm;
                      bestIdx2 = idx;
                      bestIdx3 = -2;
                      bestIdx4 = -2;
                    }
                  }

                  /// if there is, it will not choose points in other levels, but it will also choose the largest on in the pot.
                  if(bestIdx3==-2) {
                    continue;
                  }

                  float ag1 = mapmax1[(int)(xf*0.5f+0.25f) + (int)(yf*0.5f+0.25f)*w1];  /// first level.

                  if(ag1 > pixelTH1*thFactor) {
                    Vec2f ag0d = map0[idx].tail<2>();
                    float dirNorm = fabsf((float)(ag0d.dot(dir3)));
                    if(!setting_selectDirectionDistribution) dirNorm = ag1;

                    if(dirNorm > bestVal3) {
                      bestVal3 = dirNorm;
                      bestIdx3 = idx;
                      bestIdx4 = -2;
                    }
                  }

                  if(bestIdx4==-2) continue;

                  float ag2 = mapmax2[(int)(xf*0.25f+0.125) + (int)(yf*0.25f+0.125)*w2];  /// second level.
                  if(ag2 > pixelTH2*thFactor)
                  {
                    Vec2f ag0d = map0[idx].tail<2>();
                    float dirNorm = fabsf((float)(ag0d.dot(dir4)));
                    if(!setting_selectDirectionDistribution) dirNorm = ag2;

                    if(dirNorm > bestVal4)
                    { bestVal4 = dirNorm; bestIdx4 = idx; }
                  }
                }

              /// the 0 level of the pot is cycled, and if there is, add a flag.
              if(bestIdx2>0) {
                map_out[bestIdx2] = 1;
                /// there are better in high-level pots, those that meet more stringent requirements do not need to meet pixelTH1.
                /// bug bestVal3 is useless, because bestIdx3 = -2 continues directly.
                bestVal3 = 1e10;  /// if the 0 level is found, it will not be found at the high level.
                n2++;  /// count.
              }
            }

          /// No in level 0, select in level 1.
          if(bestIdx3 > 0) {
            map_out[bestIdx3] = 2;
            bestVal4 = 1e10;
            n3++;
          }
        }

      /// if the first level does not exist, select it on the second level.
      if(bestIdx4>0) {
        map_out[bestIdx4] = 4;
        n4++;
      }
    }

  /// number of points selected at levels 0,1,2.
  return Eigen::Vector3i(n2, n3, n4);
}


}

