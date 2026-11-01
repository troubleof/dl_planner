#include "route_manager.hpp"
#include <iostream>
#include <stdexcept>

class TestMap:public CollisionMap {
 public:
  std::vector<std::pair<Vec,Vec>> occupied;
  void update(const Route&,const Vec&)override{++version;}
  void add(const Vec& lo,const Vec& hi){occupied.emplace_back(lo,hi);++version;}
  int state(const Vec& p)override{return inflated(p)?1:0;}
  bool inflated(const Vec& p)override{
    if(!inside(p))return true;
    Vec q=cellCenter(p);
    for(const auto& b:occupied)if((q.array()>=b.first.array()).all()&&(q.array()<=b.second.array()).all())return true;
    return false;
  }
};
static void require(bool value,const std::string& message){if(!value)throw std::runtime_error(message);}
struct Fixture {
  std::shared_ptr<TestMap> map=std::make_shared<TestMap>();AStar astar;
  std::unique_ptr<RouteManager> manager;
  Vec start{-3.,0.,2.5},goal{3.,0.,2.5};
  explicit Fixture(const std::string& policy="repair") {
    map->cfg.route_policy=policy;astar.initGridMap(map,Eigen::Vector3i(80,60,12));
    manager=std::make_unique<RouteManager>(*map,[this](const RouteSearchRequest& q){return searchGridRoute(*map,astar,q);});
  }
  void wall(){map->add(Vec(-.2,-.4,0),Vec(.2,.4,5));}
  void free(const Route& r){
    require(r.size()>=2,"route empty");
    for(size_t i=1;i<r.size();++i) {
      if(manager->stats.source!="full")require(manager->edgeFree(r[i-1],r[i]),"reused/repaired edge crosses obstacle");
      else {
        // Full mode deliberately retains the original sampled A* reference.
        // It is only a seed: the worker still validates the entire final spline.
        int n=std::max(2,int(std::ceil((r[i]-r[i-1]).norm()/.05)));
        for(int j=0;j<=n;++j)require(map->safe(r[i-1]+double(j)/n*(r[i]-r[i-1])),"full-search reference collides");
      }
    }
  }
};
int main(int argc,char** argv) {
  rclcpp::init(argc,argv);int passed=0;
  try {
    {Fixture f("full");f.free(f.manager->get(f.start,f.goal));f.free(f.manager->get(f.start,f.goal));require(f.manager->stats.full_searches==1&&f.manager->stats.source=="full","full policy changed");++passed;}
    {Fixture f;f.free(f.manager->get(f.start,f.goal));f.free(f.manager->get(Vec(-2.8,0,2.5),f.goal));require(f.manager->stats.source=="reuse"&&f.manager->stats.full_searches==0,"unchanged path not reused");++passed;
      f.map->add(Vec(-.1,3.,0),Vec(.1,3.5,5));f.free(f.manager->get(Vec(-2.6,0,2.5),f.goal));require(f.manager->stats.source=="reuse"&&f.manager->stats.queried_cells>0,"unrelated update invalidated path or reused stale query cache");++passed;
      f.wall();f.free(f.manager->get(Vec(-2.4,0,2.5),f.goal));require(f.manager->stats.source=="repair"&&f.manager->stats.local_searches==1&&f.manager->stats.full_searches==0&&f.manager->stats.retained_points>0,"local insertion not repaired: "+f.manager->stats.source+" / "+f.manager->stats.fallback);++passed;
      f.free(f.manager->get(Vec(-2.2,0,2.5),f.goal));require(f.manager->stats.source=="reuse","repaired path not retained");++passed;}
    {Fixture f("reuse");f.manager->get(f.start,f.goal);f.wall();f.free(f.manager->get(f.start,f.goal));require(f.manager->stats.full_searches==1&&f.manager->stats.local_searches==0,"reuse-only policy performed local repair");++passed;}
    {Fixture f;f.manager->get(f.start,f.goal);f.free(f.manager->get(f.start,Vec(3,2,2.5)));require(f.manager->stats.fallback=="goal_changed","goal-change cache was not invalidated");++passed;}
    {Fixture f;f.manager->get(f.start,f.goal);f.free(f.manager->get(Vec(-4,3,2.5),f.goal));require(f.manager->stats.fallback=="cannot_rejoin","remote start incorrectly reused path");++passed;}
    {Fixture f;f.map->version=5;f.manager->get(f.start,f.goal);f.map->version=1;f.manager->get(f.start,f.goal);require(f.manager->stats.fallback=="map_version_reset","reset map reused old certificate");++passed;}
    {Fixture f;f.manager->get(f.start,f.goal);f.manager->invalidate("optimizer_rejected");f.manager->get(f.start,f.goal);require(f.manager->stats.fallback=="optimizer_rejected"&&f.manager->stats.full_searches==1,"optimizer failure did not invalidate cache");++passed;}
    {Fixture f;f.map->cfg.repair_padding=.1;f.manager->get(f.start,f.goal);f.wall();f.free(f.manager->get(f.start,f.goal));require(f.manager->stats.local_searches==1&&f.manager->stats.full_searches==1&&f.manager->stats.fallback=="local_search_failed","bounded search failure did not fall back");require(!f.map->search_box_enabled&&!f.map->route_mode,"local search leaked map query mode");++passed;}
    {Fixture f;f.map->cfg.max_local_repairs=0;f.manager->get(f.start,f.goal);f.wall();f.free(f.manager->get(f.start,f.goal));require(f.manager->stats.fallback=="repair_count_exceeded","repair budget ignored");++passed;}
    {Fixture f;f.map->cfg.guidance_steps=0;f.map->add(Vec(0,0,2.5),Vec(.099,.099,2.599));require(!f.manager->edgeFree(Vec(-.05,.049,2.55),Vec(.25,.351,2.55)),"thin corner intersection missed");++passed;}
    {Fixture f;f.manager=std::make_unique<RouteManager>(*f.map,[&f](const RouteSearchRequest& q){return q.local?Route{q.start,q.goal}:searchGridRoute(*f.map,f.astar,q);});f.manager->get(f.start,f.goal);f.wall();f.free(f.manager->get(f.start,f.goal));require(f.manager->stats.fallback=="local_search_failed"&&f.manager->stats.full_searches==1,"unverified patch accepted");++passed;}
    std::cout<<"ROUTE_MANAGER_CHECK {\"passed\":"<<passed<<"}\n";rclcpp::shutdown();return 0;
  }catch(const std::exception& e){std::cerr<<"FAILED after "<<passed<<" checks: "<<e.what()<<'\n';rclcpp::shutdown();return 1;}
}
