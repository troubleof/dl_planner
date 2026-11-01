#pragma once
#include "collision_map.hpp"
#include <limits>

// Auxiliary evidence timestamps, not occupancy probabilities or state-change
// times. Only rays in scans actually fused by the worker update this grid.
class ObservationAge {
  NavConfig cfg;Eigen::Vector3i dims,base;std::vector<double> stamps;
 public:
  explicit ObservationAge(const NavConfig& c):cfg(c) {
    base=(c.lower/c.resolution).array().floor().cast<int>();
    dims=((c.upper-c.lower)/c.resolution).array().round().cast<int>();
    stamps.assign(size_t(dims.x())*dims.y()*dims.z(),-1.);
  }
  int index(const Eigen::Vector3i& global) const {
    auto p=global-base;if((p.array()<0).any()||(p.array()>=dims.array()).any())return -1;
    return (p.x()*dims.y()+p.y())*dims.z()+p.z();
  }
  double stamp(const Vec& p) const {
    int i=index(((p.array()+1e-7)/cfg.resolution).floor().cast<int>());return i<0?-1.:stamps[i];
  }
  void update(const Route& points,const Vec& origin,double t) {
    for(const auto& endpoint:points) {
      Vec delta=endpoint-origin;double distance=delta.norm();if(distance<1e-9)continue;
      Vec end=origin+delta*std::min(1.,cfg.range/distance);
      Vec from=(origin.array()+1e-7).matrix()/cfg.resolution,to=(end.array()+1e-7).matrix()/cfg.resolution,d=to-from;
      Eigen::Vector3i cell=from.array().floor().cast<int>(),last=to.array().floor().cast<int>(),step;
      Vec next,period;
      for(int k=0;k<3;++k){step[k]=(d[k]>0)-(d[k]<0);
        if(step[k]){next[k]=(cell[k]+(step[k]>0?1.:0.)-from[k])/d[k];period[k]=1./std::abs(d[k]);}
        else next[k]=period[k]=std::numeric_limits<double>::infinity();}
      for(int iteration=0;iteration<1000;++iteration) {
        int i=index(cell);if(i>=0)stamps[i]=t;
        if(cell==last)break;
        double tau=std::numeric_limits<double>::infinity();for(int k=0;k<3;++k)if(cell[k]!=last[k])tau=std::min(tau,next[k]);
        for(int k=0;k<3;++k)if(cell[k]!=last[k]&&std::abs(next[k]-tau)<1e-10){cell[k]+=step[k];next[k]+=period[k];}
      }
    }
  }
};
