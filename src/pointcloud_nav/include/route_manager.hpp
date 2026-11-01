#pragma once
#include "collision_map.hpp"
#include <path_searching/dyn_a_star.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <functional>
#include <limits>
#include <unordered_map>

struct RouteSearchRequest {
  Vec start,goal;
  bool local=false;
  Vec lower=Vec::Zero(),upper=Vec::Zero();
  double seconds=.2;
  std::function<bool(const Vec&,const Vec&)> edge_check;
};

// Both the worker and regression executable use this exact A* adapter.
inline Route searchGridRoute(CollisionMap& map,AStar& astar,const RouteSearchRequest& q) {
  struct Restore {
    CollisionMap& m;AStar& a;bool unknown,route,box;Vec lower,upper;
    ~Restore(){m.unknown_is_occupied=unknown;m.route_mode=route;m.search_box_enabled=box;m.search_lower=lower;m.search_upper=upper;a.setTimeLimit(.2);a.setEdgeValidator({});}
  } restore{map,astar,map.unknown_is_occupied,map.route_mode,map.search_box_enabled,map.search_lower,map.search_upper};
  map.unknown_is_occupied=false;map.route_mode=true;map.search_box_enabled=q.local;
  map.search_lower=q.lower;map.search_upper=q.upper;astar.setTimeLimit(q.seconds);astar.setEdgeValidator(q.edge_check);
  if(!astar.AstarSearch(.2,q.start,q.goal))return {};
  Route r=astar.getPath();if(r.size()<2)return {};
  r.front()=q.start;r.back()=q.goal;return r;
}

struct RouteStats {
  std::string source="none",fallback="none";
  uint64_t map_version=0;
  int full_searches=0,local_searches=0,repairs=0,invalid_edges=0,checked_edges=0;
  int queried_cells=0,query_cache_hits=0,retained_points=0;
  double astar_ms=0,check_ms=0,manager_ms=0;
};

class RouteManager {
 public:
  using Search=std::function<Route(const RouteSearchRequest&)>;
  RouteStats stats;
  explicit RouteManager(CollisionMap& map,Search search):map_(map),search_(std::move(search)){}

  void invalidate(const std::string& why) {cache_.clear();invalid_reason_=why;}
  const Route& cachedRoute() const {return cache_;}

  // Exact supercover traversal of the public voxel lattice, with the same
  // boundary tie tolerance as CollisionMap. Every touched cell and all
  // configured guidance neighbours are queried. Unknown is optimistic HERE;
  // accepted spline curves still require known-free centre cells in the worker.
  bool edgeFree(const Vec& a,const Vec& b) {
    if(query_version_!=map_.version){query_cache_.clear();query_version_=map_.version;}
    ++stats.checked_edges;
    if(!map_.flight(a)||!map_.flight(b))return false;
    const double h=map_.cfg.resolution;
    Vec from=(a.array()+1e-7).matrix()/h,to=(b.array()+1e-7).matrix()/h,delta=to-from;
    Key id=key(from),end=key(to);if(blocked(id))return false;
    std::array<int,3> step{};Vec next,period;
    for(int k=0;k<3;++k) {
      step[k]=(delta[k]>0)-(delta[k]<0);
      if(step[k]) {next[k]=(id[k]+(step[k]>0?1.:0.)-from[k])/delta[k];period[k]=1./std::abs(delta[k]);}
      else next[k]=period[k]=std::numeric_limits<double>::infinity();
    }
    for(int iteration=0;id!=end&&iteration<10000;++iteration) {
      // Once an axis reaches the endpoint cell, it must not step past it
      // because of a numerically coincident endpoint/voxel-face intersection.
      double tau=std::numeric_limits<double>::infinity();
      for(int k=0;k<3;++k)if(id[k]!=end[k])tau=std::min(tau,next[k]);
      std::array<int,3> tied{};int count=0;
      for(int k=0;k<3;++k)if(id[k]!=end[k]&&std::abs(next[k]-tau)<1e-10)tied[count++]=k;
      if(!count)return false;
      for(int mask=1;mask<(1<<count);++mask) {
        Key touched=id;
        for(int j=0;j<count;++j)if(mask&(1<<j))touched[tied[j]]+=step[tied[j]];
        if(blocked(touched))return false;
      }
      for(int j=0;j<count;++j){int k=tied[j];id[k]+=step[k];next[k]+=period[k];}
    }
    return id==end;
  }

  Route get(const Vec& start,const Vec& goal) {
    const auto begin=Clock::now();stats=RouteStats{};stats.map_version=map_.version;
    if(query_version_!=map_.version) {query_cache_.clear();query_version_=map_.version;}
    struct Timer {RouteStats& stats;Clock::time_point start;~Timer(){stats.manager_ms=elapsed(start);}} timer{stats,begin};
    if(map_.cfg.route_policy=="full")return full(start,goal,"policy_full");
    if(cache_.empty())return full(start,goal,invalid_reason_.empty()?"empty_cache":invalid_reason_);
    if((goal-goal_).norm()>1e-6)return full(start,goal,"goal_changed");
    if(map_.version<cache_version_)return full(start,goal,"map_version_reset");

    auto check_begin=Clock::now();Route route=trim(start);
    if(route.size()<2){stats.check_ms+=elapsed(check_begin);return full(start,goal,"cannot_rejoin");}
    auto bad=badEdges(route);stats.invalid_edges=std::count(bad.begin(),bad.end(),true);
    stats.check_ms+=elapsed(check_begin);
    if(stats.invalid_edges==0) {
      stats.source="reuse";stats.retained_points=int(route.size())-1;store(route,goal);return route;
    }
    if(map_.cfg.route_policy=="reuse")return full(start,goal,"cached_route_blocked");

    for(int attempt=0;attempt<map_.cfg.max_local_repairs;++attempt) {
      auto first=std::find(bad.begin(),bad.end(),true);
      if(first==bad.end())break;
      size_t left=std::distance(bad.begin(),first),last=left;
      while(last+1<bad.size()&&bad[last+1])++last;
      size_t right=last+1;double before=0,after=0;
      while(left>0&&(before<.6||!pointFree(route[left]))) {before+=(route[left]-route[left-1]).norm();--left;}
      while(right+1<route.size()&&(after<.6||!pointFree(route[right]))) {after+=(route[right+1]-route[right]).norm();++right;}
      if(!pointFree(route[left])||!pointFree(route[right]))return full(start,goal,"repair_anchor_blocked");
      double span=0;Vec lower=route[left],upper=route[left];
      for(size_t i=left;i<=right;++i){lower=lower.cwiseMin(route[i]);upper=upper.cwiseMax(route[i]);if(i>left)span+=(route[i]-route[i-1]).norm();}
      if(span>map_.cfg.repair_span)return full(start,goal,"repair_span_exceeded");
      RouteSearchRequest q{route[left],route[right],true,
        (lower.array()-map_.cfg.repair_padding).matrix().cwiseMax(map_.cfg.lower),
        (upper.array()+map_.cfg.repair_padding).matrix().cwiseMin(map_.cfg.upper),map_.cfg.repair_search_seconds};
      q.edge_check=[this](const Vec& a,const Vec& b){return edgeFree(a,b);};
      auto search_begin=Clock::now();++stats.local_searches;Route patch=search_(q);stats.astar_ms+=elapsed(search_begin);
      check_begin=Clock::now();
      bool valid=patch.size()>=2&&(patch.front()-q.start).norm()<1e-8&&(patch.back()-q.goal).norm()<1e-8;
      if(valid)for(const auto& p:patch)if((p.array()<q.lower.array()-1e-8).any()||(p.array()>q.upper.array()+1e-8).any()){valid=false;break;}
      if(valid){auto failures=badEdges(patch);valid=std::none_of(failures.begin(),failures.end(),[](bool x){return x;});}
      stats.check_ms+=elapsed(check_begin);
      if(!valid)return full(start,goal,"local_search_failed");
      Route joined;joined.reserve(route.size()+patch.size());
      for(size_t i=0;i<=left;++i)append(joined,route[i]);
      for(size_t i=1;i<patch.size();++i)append(joined,patch[i]);
      for(size_t i=right+1;i<route.size();++i)append(joined,route[i]);
      stats.retained_points=int(left+1+route.size()-right-1);route=std::move(joined);++stats.repairs;
      check_begin=Clock::now();bad=badEdges(route);stats.check_ms+=elapsed(check_begin);
      if(std::none_of(bad.begin(),bad.end(),[](bool x){return x;})) {
        stats.source="repair";store(route,goal);return route;
      }
    }
    return full(start,goal,"repair_count_exceeded");
  }

 private:
  using Clock=std::chrono::steady_clock;
  using Key=std::array<int,3>;
  struct Hash {size_t operator()(const Key& k)const {size_t h=0;for(int v:k)h^=std::hash<int>{}(v)+0x9e3779b9+(h<<6)+(h>>2);return h;}};
  CollisionMap& map_;Search search_;Route cache_;Vec goal_=Vec::Zero();
  uint64_t cache_version_=0,query_version_=std::numeric_limits<uint64_t>::max();
  std::unordered_map<Key,bool,Hash> query_cache_;
  std::string invalid_reason_;
  static double elapsed(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
  static Key key(const Vec& p){return {int(std::floor(p.x())),int(std::floor(p.y())),int(std::floor(p.z()))};}
  static void append(Route& r,const Vec& p){if(r.empty()||(r.back()-p).norm()>1e-8)r.push_back(p);}
  bool pointFree(const Vec& p){return edgeFree(p,p);}
  bool blocked(const Key& id) {
    auto found=query_cache_.find(id);if(found!=query_cache_.end()){++stats.query_cache_hits;return found->second;}
    ++stats.queried_cells;Vec p=(Vec(id[0],id[1],id[2])+Vec::Constant(.5))*map_.cfg.resolution;
    int n=map_.cfg.guidance_steps;bool bad=false;
    for(int x=-n;x<=n&&!bad;++x)for(int y=-n;y<=n&&!bad;++y)for(int z=-n;z<=n;++z)
      if(map_.inflated(p+map_.cfg.resolution*Vec(x,y,z))){bad=true;break;}
    query_cache_[id]=bad;return bad;
  }
  std::vector<bool> badEdges(const Route& r) {
    std::vector<bool> bad;bad.reserve(r.size()-1);
    for(size_t i=1;i<r.size();++i)bad.push_back(!edgeFree(r[i-1],r[i]));return bad;
  }
  Route trim(const Vec& start) {
    double distance=std::numeric_limits<double>::infinity();Vec projection=Vec::Zero();size_t edge=0;
    for(size_t i=0;i+1<cache_.size();++i) {
      Vec d=cache_[i+1]-cache_[i];double u=std::clamp(d.dot(start-cache_[i])/std::max(d.squaredNorm(),1e-16),0.,1.);
      Vec p=cache_[i]+u*d;double dist=(p-start).squaredNorm();
      if(dist<distance){distance=dist;projection=p;edge=i;}
    }
    if(distance>map_.cfg.route_rejoin_distance*map_.cfg.route_rejoin_distance)return {};
    Route r;append(r,start);append(r,projection);for(size_t i=edge+1;i<cache_.size();++i)append(r,cache_[i]);return r;
  }
  void store(const Route& r,const Vec& goal) {cache_=r;goal_=goal;cache_version_=map_.version;invalid_reason_.clear();}
  Route full(const Vec& start,const Vec& goal,const std::string& why) {
    stats.source="full";stats.fallback=why;++stats.full_searches;
    auto begin=Clock::now();Route raw=search_(RouteSearchRequest{start,goal});stats.astar_ms+=elapsed(begin);
    cache_.clear();
    if(raw.size()>=2&&map_.cfg.route_policy!="full") {
      begin=Clock::now();auto bad=badEdges(raw);stats.check_ms+=elapsed(begin);
      if(std::none_of(bad.begin(),bad.end(),[](bool x){return x;}))store(raw,goal);
    }
    return raw; // unchanged full-search reference still receives whole-curve validation
  }
};
