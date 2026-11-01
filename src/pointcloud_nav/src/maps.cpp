#include "collision_map.hpp"
#include <plan_env/grid_map.h>
#include <rog_map/rog_map.h>
#include <chrono>

namespace {
using P=rclcpp::Parameter;
auto node(const std::string& name,const std::vector<P>& params) {
  rclcpp::NodeOptions opts;opts.parameter_overrides(params);
  return std::make_shared<rclcpp::Node>(name,opts);
}
class EgoMap final:public CollisionMap {
  GridMap map_;
  rclcpp::Node::SharedPtr node_;
 public:
  explicit EgoMap(const NavConfig& c) {
    cfg=c;
    node_=node("single_uav_ego_map",{
      P("grid_map/resolution",c.resolution),P("grid_map/map_size_x",c.upper.x()-c.lower.x()),P("grid_map/map_size_y",c.upper.y()-c.lower.y()),P("grid_map/map_size_z",c.upper.z()-c.lower.z()),
      P("grid_map/local_update_range_x",30.),P("grid_map/local_update_range_y",30.),P("grid_map/local_update_range_z",10.),P("grid_map/local_map_margin",200),
      P("grid_map/obstacles_inflation",c.radius),P("grid_map/ground_height",c.lower.z()),P("grid_map/skip_pixel",2),
      P("grid_map/p_hit",c.p_hit),P("grid_map/p_miss",c.p_miss),P("grid_map/p_min",c.p_min),P("grid_map/p_max",c.p_max),P("grid_map/p_occ",c.p_occ),
      P("grid_map/min_ray_length",0.),P("grid_map/max_ray_length",c.range),P("grid_map/virtual_ceil_height",-1.),P("grid_map/virtual_ceil_yp",-1.),P("grid_map/virtual_ceil_yn",-1.),P("grid_map/frame_id",std::string("world"))});
    map_.initMap(node_); // no executor: callbacks/timers never mutate the map
  }
  void update(const Route& points,const Vec& origin) override {map_.updateEndpoints(points,origin);++version;}
  int state(const Vec& p) override {
    if(!inside(p))return 2;
    const double l=map_.rawLogOdds(cellCenter(p));
    if(l>std::log(cfg.p_occ/(1-cfg.p_occ)))return 1;
    if(l>=std::log(cfg.p_min/(1-cfg.p_min))-1e-6 && l<std::log(cfg.p_free/(1-cfg.p_free)))return 0;
    return 2;
  }
  bool inflated(const Vec& p) override {return !inside(p)||map_.getInflateOccupancy(cellCenter(p))!=0;}
};

class NativeRog final:public rog_map::ROGMap {
  const double getSystemWalltimeNow() override {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  }
 public:
  NativeRog(const std::string& path,const NavConfig& c) {
    cfg_=rog_map::Config(path);
    // EGO uses a cube, so use exactly the same integer-offset kernel.
    cfg_.inf_spherical_neighbor.clear();
    int n=c.inflationSteps();
    for(int x=-n;x<=n;++x)for(int y=-n;y<=n;++y)for(int z=-n;z<=n;++z)cfg_.inf_spherical_neighbor.emplace_back(x,y,z);
    if(std::abs(cfg_.resolution-c.resolution)>1e-9||std::abs(cfg_.inflation_resolution-c.resolution)>1e-9||cfg_.inflation_step!=n||std::abs(cfg_.raycast_range_max-c.range)>1e-9||cfg_.map_sliding_en)throw std::runtime_error("ROG config violates shared map contract");
    init();
  }
};
class RogMap final:public CollisionMap {
  NativeRog map_;
 public:
  RogMap(const NavConfig& c,const std::string& path):map_(path,c){cfg=c;}
  void update(const Route& points,const Vec& origin) override {
    rog_map::PointCloud cloud;cloud.reserve(points.size());
    for(const auto& p:points){pcl::PointXYZI q;q.x=p.x();q.y=p.y();q.z=p.z();q.intensity=1.;cloud.push_back(q);}
    map_.updateMap(cloud,super_utils::Pose{origin,Eigen::Quaterniond::Identity()});++version;
  }
  int state(const Vec& p) override {
    if(!inside(p))return 2;
    Vec q=cellCenter(p);
    if(map_.isOccupied(q))return 1;
    if(map_.isKnownFree(q))return 0;
    return 2;
  }
  bool inflated(const Vec& p) override {return !inside(p)||map_.isOccupiedInflate(cellCenter(p));}
};
}
CollisionMap::Ptr makeMap(const std::string& backend,const NavConfig& cfg,const std::string& path) {
  if(backend=="ego")return std::make_shared<EgoMap>(cfg);
  if(backend=="rog")return std::make_shared<RogMap>(cfg,path);
  throw std::runtime_error("unknown map backend");
}
rclcpp::Node::SharedPtr makeOptimizerNode(const NavConfig& c) {
  return node("single_uav_optimizer",{P("optimization/lambda_smooth",1.),P("optimization/lambda_collision",1.),P("optimization/lambda_feasibility",.1),P("optimization/lambda_fitness",1.),P("optimization/dist0",.3),P("optimization/swarm_clearance",.6),P("optimization/max_vel",c.max_v),P("optimization/max_acc",c.max_a),P("optimization/order",3)});
}
