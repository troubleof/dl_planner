#pragma once
#include <Eigen/Core>
#include <memory>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <rclcpp/rclcpp.hpp>
using Vec = Eigen::Vector3d;
using Route = std::vector<Vec>;

struct NavConfig {
  double resolution=0.1, radius=0.3272634126602354, range=8.0;
  double min_z=2., max_z=3., max_v=3., max_a=6., speed=1.5, horizon=3.;
  Vec lower{-10.,-6.,0.}, upper{10.,6.,5.};
  double handoff=.30, deadline_guard=.025, validation_dt=.005;
  double p_hit=.9,p_miss=.35,p_min=.12,p_max=.97,p_occ=.8,p_free=.3;
  int guidance_steps=1;
  std::string route_policy="full";
  double route_rejoin_distance=1.0, repair_padding=1.2, repair_span=4.0;
  double repair_search_seconds=.04;
  int max_local_repairs=2;
  int inflationSteps() const {return int(std::ceil(radius/resolution));}
};

class CollisionMap {
 public:
  using Ptr=std::shared_ptr<CollisionMap>;
  NavConfig cfg;
  bool unknown_is_occupied=false; // optimistic only during geometric search
  bool route_mode=false;
  bool search_box_enabled=false;
  Vec search_lower=Vec::Zero(),search_upper=Vec::Zero();
  uint64_t version=0;
  double stamp=-1.;
  Vec last_block=Vec::Zero();
  virtual ~CollisionMap()=default;
  virtual void update(const Route& points,const Vec& origin)=0;
  virtual int state(const Vec& p)=0; // 0 known free, 1 occupied, 2 unknown
  virtual bool inflated(const Vec& p)=0;
  bool inside(const Vec& p) const {return (p.array()>cfg.lower.array()+1e-6).all()&&(p.array()<cfg.upper.array()-1e-6).all();}
  // A 0.1 micrometre tie tolerance prevents spline roundoff selecting the
  // opposite cell at an exact voxel face. The same rule applies to both maps.
  Vec cellCenter(const Vec& p) const {return (((p.array()+1e-7)/cfg.resolution).floor()+.5).matrix()*cfg.resolution;}
  bool flight(const Vec& p) const {return p.allFinite() && p.z()>=cfg.min_z && p.z()<=cfg.max_z && p.x()>cfg.lower.x()+cfg.radius && p.x()<cfg.upper.x()-cfg.radius && p.y()>cfg.lower.y()+cfg.radius && p.y()<cfg.upper.y()-cfg.radius;}
  double getResolution() const {return cfg.resolution;}
  bool guidanceFree(const Vec& p) {
    if(!flight(p))return false;
    // One extra voxel for geometric guidance only; final safety still uses
    // the common physical envelope. Avoid cutting a tight reference corner.
    int n=cfg.guidance_steps;
    for(int x=-n;x<=n;++x)for(int y=-n;y<=n;++y)for(int z=-n;z<=n;++z)
      if(inflated(p+cfg.resolution*Vec(x,y,z)))return false;
    return true;
  }
  int getInflateOccupancy(const Vec& p) {
    bool outside_box=search_box_enabled&&((p.array()<search_lower.array()).any()||(p.array()>search_upper.array()).any());
    bool blocked=outside_box||!flight(p)||inflated(p)||(route_mode&&!guidanceFree(p))||(unknown_is_occupied&&state(p)!=0);
    if(blocked)last_block=p;return blocked;
  }
  bool safe(const Vec& p) {return flight(p)&&state(p)==0&&!inflated(p);}
};
CollisionMap::Ptr makeMap(const std::string& backend,const NavConfig& cfg,const std::string& rog_config);
rclcpp::Node::SharedPtr makeOptimizerNode(const NavConfig& cfg);
