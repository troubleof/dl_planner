#include "collision_map.hpp"
#include "route_manager.hpp"
#include "observation_age.hpp"
#include "visible_guidance.hpp"
#include <bspline_opt/bspline_optimizer.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <filesystem>
#include <fstream>
using Spline=ego_planner::UniformBspline;
using Clock=std::chrono::steady_clock;
using Y=YAML::Node;

static Vec vec(const Y& x) {
  if(!x.IsSequence()||x.size()!=3)throw std::runtime_error("expected finite 3-vector");
  Vec v(x[0].as<double>(),x[1].as<double>(),x[2].as<double>());
  if(!v.allFinite())throw std::runtime_error("nonfinite vector");return v;
}
static std::string jvec(const Vec& p) {std::ostringstream s;s<<std::setprecision(17)<<'['<<p.x()<<','<<p.y()<<','<<p.z()<<']';return s.str();}
static NavConfig configFrom(const std::string& path) {
  Y y=YAML::LoadFile(path);NavConfig c;
  c.resolution=y["resolution"].as<double>();Vec dims=vec(y["body_dimensions_m"]);
  c.radius=dims.norm()/2+y["safety_margin_m"].as<double>();c.lower=vec(y["bounds"][0]);c.upper=vec(y["bounds"][1]);
  c.min_z=y["height_band"][0].as<double>();c.max_z=y["height_band"][1].as<double>();
  c.max_v=y["max_velocity"].as<double>();c.max_a=y["max_acceleration"].as<double>();c.speed=y["cruise_speed"].as<double>();c.horizon=y["planning_horizon"].as<double>();c.range=y["sensor_range"].as<double>();
  c.handoff=y["handoff"].as<double>();c.deadline_guard=y["deadline_guard"].as<double>();c.validation_dt=y["validation_dt"].as<double>();c.guidance_steps=y["route_guidance_extra_voxels"].as<int>();
  c.p_hit=y["p_hit"].as<double>();c.p_miss=y["p_miss"].as<double>();c.p_min=y["p_min"].as<double>();c.p_max=y["p_max"].as<double>();c.p_occ=y["p_occ"].as<double>();c.p_free=y["p_free"].as<double>();
  c.route_policy=y["route_policy"].as<std::string>("full");
  c.route_rejoin_distance=y["route_rejoin_distance"].as<double>(1.);
  c.repair_padding=y["repair_padding"].as<double>(1.2);c.repair_span=y["repair_span"].as<double>(4.);
  c.repair_search_seconds=y["repair_search_seconds"].as<double>(.04);c.max_local_repairs=y["max_local_repairs"].as<int>(2);
  if((c.route_policy!="full"&&c.route_policy!="reuse"&&c.route_policy!="repair")||!std::isfinite(c.route_rejoin_distance)||c.route_rejoin_distance<=0||!std::isfinite(c.repair_padding)||c.repair_padding<=0||!std::isfinite(c.repair_span)||c.repair_span<=0||!std::isfinite(c.repair_search_seconds)||c.repair_search_seconds<=0||c.repair_search_seconds>.2||c.max_local_repairs<0||c.max_local_repairs>4)throw std::runtime_error("invalid route reuse/repair configuration");
  if(c.resolution<=0||c.range<=0||c.radius<=0||c.validation_dt<=0||c.max_v<=0||c.max_a<=0||c.speed<=0||c.horizon<=0||c.handoff<=c.deadline_guard||c.min_z>=c.max_z||c.guidance_steps<0||c.guidance_steps>3||(dims.array()<=0).any()||(c.upper.array()<=c.lower.array()).any()||std::abs(c.lower.x()+c.upper.x())>1e-8||std::abs(c.lower.y()+c.upper.y())>1e-8)throw std::runtime_error("invalid shared navigation configuration");
  return c;
}
static std::string jroute(const Route& p) {std::ostringstream s;s<<'[';for(size_t i=0;i<p.size();++i)s<<(i?",":"")<<jvec(p[i]);s<<']';return s.str();}
static double ms(Clock::time_point t) {return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
struct State {Vec p=Vec::Zero(),v=Vec::Zero(),a=Vec::Zero();};
static State state(const Y& n){return {vec(n["p"]),vec(n["v"]),vec(n["a"])};}
struct Plan {Spline p;double start=0,dt=0;};
static Plan planFrom(const Y& n) {
  Eigen::MatrixXd q(3,n["controls"].size());
  if(q.cols()<6||q.cols()>500)throw std::runtime_error("invalid spline controls");
  for(int i=0;i<q.cols();++i)q.col(i)=vec(n["controls"][i]);
  double dt=n["dt"].as<double>(),start=n["start"].as<double>();
  if(!std::isfinite(dt)||dt<=0||!std::isfinite(start))throw std::runtime_error("invalid spline time");
  return {Spline(q,3,dt),start,dt};
}
static std::string jplan(Plan& p) {
  Route q;auto m=p.p.getControlPoint();for(int i=0;i<m.cols();++i)q.push_back(m.col(i));
  std::ostringstream s;s<<std::setprecision(17)<<"{\"start\":"<<p.start<<",\"dt\":"<<p.dt<<",\"duration\":"<<p.p.getTimeSum()<<",\"controls\":"<<jroute(q)<<'}';return s.str();
}
static double length(const Route& r){double s=0;for(size_t i=1;i<r.size();++i)s+=(r[i]-r[i-1]).norm();return s;}
static Vec at(const Route& r,double d){for(size_t i=1;i<r.size();++i){double l=(r[i]-r[i-1]).norm();if(d<=l)return r[i-1]+(r[i]-r[i-1])*d/std::max(l,1e-9);d-=l;}return r.back();}
static void boundaries(Eigen::MatrixXd& q,double dt,const State& s,const Vec& goal) {
  q.col(0)=s.p-s.v*dt+s.a*dt*dt/3.;q.col(1)=s.p-s.a*dt*dt/6.;q.col(2)=s.p+s.v*dt+s.a*dt*dt/3.;
  q.rightCols(3).colwise()=goal;
}

class Navigator {
 public:
  NavConfig cfg;
  CollisionMap::Ptr map;
  ego_planner::BsplineOptimizer optimizer;
  ego_planner::SwarmTrajData empty_swarm;
  AStar::Ptr search;
  std::unique_ptr<RouteManager> router;
  std::unique_ptr<ObservationAge> observations;
  double observation_ms=0.;
  uint64_t optimizer_calls=0;
  Clock::time_point request_begin;
  double request_budget_ms=1e12;
  rclcpp::Node::SharedPtr opt_node;
  explicit Navigator(const std::string& backend,const std::string& path,const std::string& config_path):cfg(configFrom(config_path)) {
    if(YAML::LoadFile(config_path)["track_observation_age"].as<bool>(false))observations=std::make_unique<ObservationAge>(cfg);
    map=makeMap(backend,cfg,path);opt_node=makeOptimizerNode(cfg);
    optimizer.setParam(opt_node);optimizer.setEnvironment(map);optimizer.setDroneId(0);optimizer.setSwarmTrajs(&empty_swarm);
    optimizer.a_star_=std::make_shared<AStar>();optimizer.a_star_->initGridMap(map,Eigen::Vector3i(120,80,20));
    search=std::make_shared<AStar>();search->initGridMap(map,Eigen::Vector3i(120,80,20));
    router=std::make_unique<RouteManager>(*map,[this](const RouteSearchRequest& q){return searchGridRoute(*map,*search,q);});
  }
  double cloud(const Y& input) {
    observation_ms=0.;
    if(!input["cloud"])return 0.;
    auto begin=Clock::now();const auto c=input["cloud"];
    if(c["frame"].as<std::string>()!="world")throw std::runtime_error("cloud frame must be world");
    double t=c["stamp"].as<double>();
    if(!std::isfinite(t)||t<=map->stamp||t>input["t"].as<double>()+1e-9)throw std::runtime_error("cloud timestamp is stale or in future");
    Vec origin=vec(c["origin"]);if(!map->inside(origin))throw std::runtime_error("cloud origin outside map");
    Route points;
    if(c["binary_xyz_f32"]) {
      const size_t count=c["count"].as<size_t>();
      if(count>100000)throw std::runtime_error("cloud exceeds point limit");
      std::ifstream f(c["binary_xyz_f32"].as<std::string>(),std::ios::binary|std::ios::ate);
      if(!f||size_t(f.tellg())!=count*3*sizeof(float))throw std::runtime_error("binary cloud length mismatch");
      f.seekg(0);std::vector<float> xyz(count*3);f.read(reinterpret_cast<char*>(xyz.data()),xyz.size()*sizeof(float));
      points.reserve(count);for(size_t i=0;i<count;++i){Vec p(xyz[i*3],xyz[i*3+1],xyz[i*3+2]);if(!p.allFinite())throw std::runtime_error("nonfinite cloud point");points.push_back(p);}
    } else {
      if(!c["points"].IsSequence()||c["points"].size()>100000)throw std::runtime_error("invalid cloud size");
      points.reserve(c["points"].size());for(auto p:c["points"])points.push_back(vec(p));
    }
    map->update(points,origin);map->stamp=t;
    if(observations){auto age_begin=Clock::now();observations->update(points,origin,t);observation_ms=ms(age_begin);}
    return ms(begin);
  }
  bool check(Spline& p,const State& expected,std::string& reason,double& vb,double& ab,bool check_start=true,double from=0.) {
    auto v=p.getDerivative(),a=v.getDerivative();double T=p.getTimeSum();
    if(!std::isfinite(T)||T<=0){reason="invalid_duration";return false;}
    vb=v.getControlPoint().colwise().norm().maxCoeff();ab=a.getControlPoint().colwise().norm().maxCoeff();
    if(vb>cfg.max_v+1e-6||ab>cfg.max_a+1e-6){reason="derivative_bound";return false;}
    if(check_start&&((p.evaluateDeBoorT(0)-expected.p).norm()>1e-6||(v.evaluateDeBoorT(0)-expected.v).norm()>1e-6||(a.evaluateDeBoorT(0)-expected.a).norm()>1e-5)){reason="start_discontinuity";return false;}
    if(v.evaluateDeBoorT(T).norm()>1e-6||a.evaluateDeBoorT(T).norm()>1e-5){reason="terminal_not_rest";return false;}
    int n=std::max(1,int(std::ceil((T-from)/cfg.validation_dt)));
    for(int i=0;i<=n;++i){double u=std::clamp(from+(T-from)*i/n,0.,T);Vec q=p.evaluateDeBoorT(u);
      if(!map->flight(q)){reason="flight_bounds";return false;}
      if(map->inflated(q)){reason="occupied";return false;}
      if(map->state(q)!=0){reason="unknown";return false;}
    }
    reason="ok";return true;
  }
  Route makeRoute(const State& s,const Vec& goal) {
    router->stats=RouteStats{};router->stats.map_version=map->version;
    map->unknown_is_occupied=false;
    if(!map->flight(goal)||!map->safe(s.p))return {};
    Route raw=router->get(s.p,goal);if(raw.size()<2)return {};
    // Sample the optimistic route, then stop before the first unknown/occupied cell.
    Route local{s.p};double lim=std::min(cfg.horizon,length(raw));
    for(double d=.05;d<=lim+1e-9;d+=.05){Vec p=at(raw,std::min(d,lim));if(!map->safe(p))break;local.push_back(p);}
    if(local.size()<3||length(local)<.15){router->invalidate("local_prefix_blocked");return {};}
    // Short-cut only inside known free cells, including the collision envelope.
    Route simple{local.front()};size_t i=0;
    while(i+1<local.size()) {
      size_t chosen=i+1;
      for(size_t j=local.size()-1;j>i+1;--j){Vec delta=local[j]-local[i];int n=std::max(1,int(std::ceil(delta.norm()/.04)));bool free=true;
        for(int k=0;k<=n;++k){Vec p=local[i]+delta*double(k)/n;if(!map->safe(p)||!map->guidanceFree(p)){free=false;break;}}
        if(free){chosen=j;break;}}
      simple.push_back(local[chosen]);i=chosen;
    }
    return simple;
  }
  std::string routeStats() {
    const auto& r=router->stats;std::ostringstream s;s<<std::setprecision(17);
    s<<",\"route_policy\":"<<std::quoted(cfg.route_policy)<<",\"route_source\":"<<std::quoted(r.source)<<",\"route_fallback\":"<<std::quoted(r.fallback)
      <<",\"route_map_version\":"<<r.map_version<<",\"full_astar_calls\":"<<r.full_searches<<",\"local_astar_calls\":"<<r.local_searches<<",\"local_repairs\":"<<r.repairs
      <<",\"invalid_edges\":"<<r.invalid_edges<<",\"checked_edges\":"<<r.checked_edges<<",\"queried_cells\":"<<r.queried_cells<<",\"query_cache_hits\":"<<r.query_cache_hits<<",\"retained_points\":"<<r.retained_points
      <<",\"astar_ms\":"<<r.astar_ms<<",\"route_check_ms\":"<<r.check_ms<<",\"route_manager_ms\":"<<r.manager_ms;
    return s.str();
  }
  std::string solveRoute(const Y& in,Route route,double search_ms) {
    if(in["forbid_optimizer"].as<bool>(false))throw std::runtime_error("optimizer forbidden for direct policy");
    State s=state(in["start_state"]);Vec goal=vec(in["goal"]);
    double t=in["t"].as<double>();
    if(t-map->stamp>.5)return "\"ok\":false,\"reason\":\"stale_cloud\"";
    auto begin=Clock::now();
    if(route.size()<2)return "\"ok\":false,\"reason\":\"no_known_route\",\"search_ms\":"+std::to_string(search_ms)+routeStats();
    double L=length(route),vb=0,ab=0;std::string reason="optimizer_failed";Plan candidate;
    bool success=false;int attempts=0;begin=Clock::now();
    map->unknown_is_occupied=true;
    for(int attempt=0;attempt<3;++attempt) {
      attempts++;int k=std::max(12,int(std::ceil(L/.18))+1);
      double duration=std::max(1.,1.5*L/cfg.speed)*std::pow(1.45,attempt),dt=duration/(k-1);
      Vec tangent=(route[1]-route[0]).normalized();double c1=std::max(0.,s.v.dot(tangent))*duration,c2=.5*s.a.dot(tangent)*duration*duration;
      Eigen::Matrix3d M;M<<1,1,1,3,4,5,6,12,20;Vec c=M.fullPivLu().solve(Vec(L-c1-c2,-c1-2*c2,-2*c2));
      Route samples;for(int i=0;i<k;++i){double u=double(i)/(k-1),arc=c1*u+c2*u*u+c[0]*pow(u,3)+c[1]*pow(u,4)+c[2]*pow(u,5);samples.push_back(at(route,std::clamp(arc,0.,L)));}
      Eigen::MatrixXd q;Spline::parameterizeToBspline(dt,samples,{s.v,Vec::Zero(),s.a,Vec::Zero()},q);boundaries(q,dt,s,route.back());
      optimizer.setLocalTargetPt(route.back());optimizer.initControlPoints(q,true);
      ++optimizer_calls;if(!optimizer.BsplineOptimizeTrajRebound(q,dt)){reason="rebound_failed";continue;}
      boundaries(q,dt,s,route.back());optimizer.ref_pts_.clear();
      for(int i=0;i<=q.cols()-3;++i)optimizer.ref_pts_.push_back(at(route,L*i/(q.cols()-3)));
      Eigen::MatrixXd refined;++optimizer_calls;if(!optimizer.BsplineOptimizeTrajRefine(q,dt,refined)){reason="refine_failed";continue;}
      candidate={Spline(refined,3,dt),t+cfg.handoff,dt};
      if(check(candidate.p,s,reason,vb,ab)){success=true;break;}
    }
    std::ostringstream out;out<<std::setprecision(17)<<"\"ok\":"<<(success?"true":"false")<<",\"reason\":"<<std::quoted(reason)<<",\"attempts\":"<<attempts<<",\"search_ms\":"<<search_ms<<",\"opt_validate_ms\":"<<ms(begin)<<",\"velocity_bound\":"<<vb<<",\"acceleration_bound\":"<<ab<<",\"route\":"<<jroute(route);
    if(success)out<<",\"plan\":"<<jplan(candidate);
    else out<<",\"last_block\":"<<jvec(map->last_block)<<",\"last_block_state\":"<<map->state(map->last_block)<<",\"last_block_inflated\":"<<map->inflated(map->last_block);
    if(!success)router->invalidate("optimizer_rejected");
    out<<routeStats();
    return out.str();
  }
  std::string directOne(const Y& in,const Y& entry) {
    std::string reason="ok";double vb=0,ab=0;bool ok=false;Plan p;
    try {
      p=planFrom(entry["plan"]);
      if(std::abs(p.start-in["t"].as<double>()-cfg.handoff)>1e-8)reason="direct_start_time";
      else if(!entry["plan"]["duration"]||std::abs(entry["plan"]["duration"].as<double>()-p.p.getTimeSum())>1e-8||p.p.getTimeSum()>6.)reason="direct_duration";
      else if(in["t"].as<double>()-map->stamp>.5)reason="stale_cloud";
      else {
        ok=check(p.p,state(in["start_state"]),reason,vb,ab);
        if(ok) {
          Vec goal=vec(in["goal"]);
          // Local detours can temporarily increase GLOBAL goal distance. This
          // changes only the progress merit, never map/dynamics/backup checks.
          if(in["local_progress_target"]) {
            goal=vec(in["local_progress_target"]);
            if(!map->safe(goal)||(goal-state(in["start_state"]).p).norm()>1.81){ok=false;reason="invalid_local_target";}
          }
          double before=(goal-state(in["start_state"]).p).norm(),after=(goal-p.p.evaluateDeBoorT(p.p.getTimeSum())).norm();
          if(before>.10&&after>before-std::min(.10,.5*before)){ok=false;reason="no_terminal_progress";}
        }
        if(ok){Plan b;ok=stop(p,std::min(p.start+.6,p.start+p.p.getTimeSum()),b,reason);if(!ok)reason="no_valid_backup";}
      }
    }catch(const std::exception& e){reason="malformed_direct_plan";}
    std::ostringstream out;out<<std::setprecision(17)<<"\"ok\":"<<(ok?"true":"false")<<",\"reason\":"<<std::quoted(reason)<<",\"velocity_bound\":"<<vb<<",\"acceleration_bound\":"<<ab;
    if(ok)out<<",\"plan\":"<<jplan(p);
    return out.str();
  }
  std::string solve(const Y& in) {
    if(in["direct_plans"]) {
      if(!in["direct_plans"].IsSequence()||in["direct_plans"].size()>15)throw std::runtime_error("direct candidate count exceeds 15");
      std::ostringstream checks;checks<<'[';int tried=0;auto begin=Clock::now();
      for(auto entry:in["direct_plans"]) {
        // Cooperative deadline: never start another candidate after exhaustion.
        // An in-progress map update/check is not preempted; Python rejects late results.
        if(ms(request_begin)>=request_budget_ms)break;
        std::string value=directOne(in,entry);Y parsed=YAML::Load("{"+value+"}");
        if(tried++)checks<<',';checks<<"{\"id\":"<<entry["id"].as<int>()<<",\"reason\":"<<std::quoted(parsed["reason"].as<std::string>())<<'}';
        if(parsed["ok"].as<bool>()) {
          checks<<']';return value+",\"frontend\":\"direct\",\"direct_selected_id\":"+std::to_string(entry["id"].as<int>())+",\"direct_tried\":"+std::to_string(tried)+",\"direct_checks\":"+checks.str()+",\"direct_worker_ms\":"+std::to_string(ms(begin))+",\"optimizer_fallback\":false";
        }
      }
      checks<<']';
      std::string value="\"ok\":false,\"reason\":\"no_valid_direct_candidate\",\"frontend\":\"direct\"";
      if(in["direct_fallback_repair"].as<bool>(false)) {
        Y fallback=YAML::Clone(in);fallback.remove("direct_plans");fallback.remove("direct_fallback_repair");value=solve(fallback);
      }
      return value+",\"direct_tried\":"+std::to_string(tried)+",\"direct_checks\":"+checks.str()+",\"optimizer_fallback\":"+(in["direct_fallback_repair"].as<bool>(false)?"true":"false");
    }
    State s=state(in["start_state"]);Vec goal=vec(in["goal"]);
    if(in["t"].as<double>()-map->stamp>.5)return "\"ok\":false,\"reason\":\"stale_cloud\",\"frontend\":\"repair\"";
    int tried=0,geometric_rejects=0;std::string last="none";auto begin=Clock::now();
    if(in["neural_routes"]&&in["t"].as<double>()-map->stamp<=.5) {
      if(!in["neural_routes"].IsSequence()||in["neural_routes"].size()>3)throw std::runtime_error("neural candidate count exceeds 3");
      for(auto entry:in["neural_routes"]) {
        ++tried;Route raw;
        if(!entry["points"].IsSequence()||entry["points"].size()<2||entry["points"].size()>64){++geometric_rejects;last="invalid_reference_size";continue;}
        for(auto p:entry["points"])raw.push_back(vec(p));
        if((raw.front()-s.p).norm()>1e-6){++geometric_rejects;last="reference_start";continue;}
        Route local{s.p};double limit=std::min(cfg.horizon,length(raw));
        for(double d=.025;d<=limit+1e-9;d+=.025){Vec p=at(raw,d);if(!map->safe(p))break;local.push_back(p);}
        if(local.size()<3||length(local)<.3||(local.back()-goal).norm()>(s.p-goal).norm()-.1){++geometric_rejects;last="no_progressing_known_prefix";continue;}
        // Same rebound/refine optimizer, terminal rest and whole-curve validator.
        router->stats=RouteStats{};router->stats.source="yopo";router->stats.map_version=map->version;
        auto value=solveRoute(in,local,ms(begin));Y parsed=YAML::Load("{"+value+"}");
        bool ok=parsed["ok"].as<bool>();last=parsed["reason"].as<std::string>();
        if(ok&&in["with_backup"].as<bool>(false)) {
          Plan nominal=planFrom(parsed["plan"]),backup;std::string why;
          ok=stop(nominal,std::min(nominal.start+.6,nominal.start+nominal.p.getTimeSum()),backup,why);
          if(!ok)last="no_valid_backup";
        }
        if(ok)return value+",\"frontend\":\"yopo\",\"neural_tried\":"+std::to_string(tried)+",\"neural_geometric_rejects\":"+std::to_string(geometric_rejects)+",\"neural_selected_id\":"+std::to_string(entry["id"].as<int>())+",\"neural_worker_ms\":"+std::to_string(ms(begin));
      }
    }
    if(in["probe_only"].as<bool>(false))return "\"ok\":false,\"reason\":"+std::string("\"")+last+"\"";
    double neural_ms=ms(begin);begin=Clock::now();Route route=makeRoute(s,goal);double search_ms=ms(begin);
    std::ostringstream suffix;suffix<<",\"frontend\":\"repair\",\"neural_tried\":"<<tried<<",\"neural_geometric_rejects\":"<<geometric_rejects<<",\"neural_worker_ms\":"<<neural_ms<<",\"neural_last_rejection\":"<<std::quoted(last);
    return solveRoute(in,route,search_ms)+suffix.str();
  }
  State on(Plan& p,double t) {
    double u=std::clamp(t-p.start,0.,p.p.getTimeSum());
    auto v=p.p.getDerivative(),a=v.getDerivative();
    return {p.p.evaluateDeBoorT(u),v.evaluateDeBoorT(u),a.evaluateDeBoorT(u)};
  }
  // Finite family of cubic uniform splines. Repeated last controls give exact
  // terminal rest; initial controls give exact P/V/A. No state is frozen.
  bool stop(Plan& parent,double branch,Plan& result,std::string& reason) {
    State s=on(parent,branch);double vb=0,ab=0;
    for(double h: {.12,.16,.20,.25,.32,.40,.50,.65,.80,1.,1.25}) {
      Eigen::MatrixXd q(3,6);Vec end=s.p+s.v*h+s.a*h*h/3.;
      boundaries(q,h,s,end);Plan b{Spline(q,3,h),branch,h};
      if(check(b.p,s,reason,vb,ab)){result=b;return true;}
    }
    return false;
  }
  // Closed sampled intervals; expand each bad sample back by one check step.
  // This retains the baseline's sampled map validator, not a swept-volume proof.
  std::string intervals(Plan& p,double from,double to,double& first) {
    from=std::max(from,p.start);to=std::max(from,to);first=-1;
    int n=std::max(1,int(std::ceil((to-from)/cfg.validation_dt)));
    auto controls=p.p.getControlPoint();
    if((controls.colwise()-controls.col(0)).cwiseAbs().maxCoeff()==0.)n=1;
    bool bad=false;double lo=0,last=0;std::ostringstream out;out<<std::setprecision(17)<<'[';int count=0;
    for(int i=0;i<=n;++i) {
      double t=from+(to-from)*i/n;bool hit=!map->safe(p.p.evaluateDeBoorT(std::clamp(t-p.start,0.,p.p.getTimeSum())));
      if(hit&&!bad){lo=std::max(from,t-cfg.validation_dt);if(first<0)first=lo;bad=true;}
      if(hit)last=t;
      if(bad&&(!hit||i==n)){if(count++)out<<',';out<<'['<<lo<<','<<std::min(to,last+cfg.validation_dt)<<']';bad=false;}
    }
    out<<']';return out.str();
  }
  bool prefix(Plan& p,double from,double to) {
    if(to<from)return true;double first;intervals(p,from,to,first);return first<0;
  }
  // Recompute a finite family of earlier stops on the newly received map.
  // Returned witnesses still need the caller's measured-time commit gate.
  // No backdating, no instant halt and no assumption of monotone map safety.
  std::string escape(const Y& in) {
    double t=in["t"].as<double>();auto schedule=in["execution_schedule"];
    if(!schedule||schedule.size()<1||schedule.size()>4)throw std::runtime_error("escape schedule invalid");
    auto original=YAML::Load("{"+guard(in)+"}");
    double bad=original["first_invalid_time"].IsNull()?t+1.:original["first_invalid_time"].as<double>();
    std::ostringstream out;out<<std::setprecision(17)<<"\"ok\":true,\"prior_guard\":{"<<guard(in)<<"},\"stops\":[";
    auto begin=Clock::now();int tried=0,count=0;bool exhausted=false;double latest=-1.;
    if(t-map->stamp<=.5)for(int j=1;j<=16;++j) {
      double branch=t+.05*j;if(branch>=bad-cfg.validation_dt)break;
      if(ms(begin)>12.){exhausted=true;break;}
      bool clear=true;size_t active=0;
      for(size_t i=0;i<schedule.size();++i) {
        Plan p=planFrom(schedule[i]);double end=i+1<schedule.size()?schedule[i+1]["start"].as<double>():branch;
        if(p.start>branch)break;active=i;
        if(end>=t && !prefix(p,std::max(t,p.start),std::min(end,branch))){clear=false;break;}
      }
      ++tried;if(!clear)continue;
      Plan p=planFrom(schedule[active]),b;std::string why;
      if(stop(p,branch,b,why)) {
        if(count++)out<<',';out<<"{\"plan\":"<<jplan(b)<<",\"commit_deadline\":"<<branch-cfg.deadline_guard<<'}';latest=branch;
      }
    }
    out<<"],\"tried\":"<<tried<<",\"search_ms\":"<<ms(begin)<<",\"budget_exhausted\":"<<(exhausted?"true":"false")<<",\"latest_sampled_feasible_branch\":";
    if(latest<0)out<<"null";else out<<latest;
    return out.str();
  }
  std::string guard(const Y& in) {
    double t=in["t"].as<double>();std::ostringstream out;out<<std::setprecision(17);
    bool safe=true;double first=-1;out<<"\"segments\":[";
    auto schedule=in["execution_schedule"];
    if(!schedule||!schedule.IsSequence()||schedule.size()<1||schedule.size()>4)throw std::runtime_error("guard needs one to four scheduled curves");
    for(size_t i=0;i<schedule.size();++i) {
      Plan p=planFrom(schedule[i]);double end=p.start+p.p.getTimeSum();
      if(i+1<schedule.size()) {
        Plan next=planFrom(schedule[i+1]);end=next.start;State before=on(p,end),after=on(next,end);
        if(next.start<p.start||(before.p-after.p).norm()>1e-6||(before.v-after.v).norm()>1e-6||(before.a-after.a).norm()>1e-5)throw std::runtime_error("invalid backup junction");
      }
      double bad=-1;std::string windows=intervals(p,std::max(t,p.start),std::max(t,end),bad);
      if(bad>=0){safe=false;if(first<0)first=bad;}
      if(i)out<<',';out<<"{\"start\":"<<p.start<<",\"invalid_intervals\":"<<windows<<'}';
    }
    out<<"],\"ok\":"<<(safe?"true":"false")<<",\"reason\":\""<<(safe?"ok":"execution_schedule_invalid")<<"\",\"first_invalid_time\":";
    if(first<0)out<<"null";else out<<first;
    if(in["nominal_plan"]) {
      Plan p=planFrom(in["nominal_plan"]);double bad=-1;
      auto windows=intervals(p,t,std::max(t,p.start+p.p.getTimeSum()),bad);
      out<<",\"nominal_invalid_intervals\":"<<windows<<",\"nominal_replace_commit_upper_bound\":";
      if(bad<0)out<<"null";else out<<bad-cfg.deadline_guard;
      // Latest witness on a 50 ms grid, within a 40 ms search budget. This is
      // neither an optimal braking time nor a guarantee that an RPC can finish.
      if(bad>=0&&in["find_stop"].as<bool>(false)) {
        auto begin=Clock::now();bool found=false,exhausted=false;Plan b;std::string why;
        if(t-map->stamp<=.5)for(double branch=std::floor((bad-cfg.validation_dt)*20)/20;branch>=t+cfg.handoff-1e-9;branch-=.05) {
          if(ms(begin)>40){exhausted=true;break;}
          if(prefix(p,t,branch)&&stop(p,branch,b,why)){found=true;break;}
        }
        out<<",\"stop_search_ms\":"<<ms(begin)<<",\"stop_search_budget_exhausted\":"<<(exhausted?"true":"false")<<",\"stop_found\":"<<(found?"true":"false");
        if(found)out<<",\"stop_plan\":"<<jplan(b)<<",\"verified_stop_commit_deadline\":"<<b.start-cfg.deadline_guard;
      }
    }
    return out.str();
  }
  std::string handle(const Y& in) {
    request_begin=Clock::now();request_budget_ms=in["worker_budget_ms"].as<double>(1e12);
    if(!std::isfinite(request_budget_ms)||request_budget_ms<0)throw std::runtime_error("invalid worker budget");
    std::string op=in["op"].as<std::string>();double update_ms=cloud(in);
    std::ostringstream out;out<<std::setprecision(17)<<"\"map_version\":"<<map->version<<",\"map_stamp\":"<<map->stamp<<",\"map_update_ms\":"<<update_ms<<',';
    if(in["execution_schedule"]&&op!="guard"&&op!="escape") {
      std::string checked=guard(in);out<<"\"execution_guard\":{"<<checked<<"},";
      if(!YAML::Load("{"+checked+"}")["ok"].as<bool>()) {
        out<<"\"ok\":false,\"reason\":\"execution_schedule_invalid\"";return out.str();
      }
    }
    if(in["executing_plan"] && in["cloud"]) {
      Plan old=planFrom(in["executing_plan"]);double t=in["t"].as<double>();
      for(double at_t=t;at_t<=t+cfg.handoff+1e-9;at_t+=cfg.validation_dt) {
        Vec p=old.p.evaluateDeBoorT(std::clamp(at_t-old.start,0.,old.p.getTimeSum()));
        if(!map->safe(p)){out<<"\"ok\":false,\"reason\":\"unsafe_execution_prefix\"";return out.str();}
      }
    }
    if(op=="escape")out<<escape(in);
    else if(op=="candidate_features") {
      if(!in["direct_plans"]||in["direct_plans"].size()>15)throw std::runtime_error("candidate features size");
      out<<"\"ok\":true,\"features\":[";int k=0;
      for(auto entry:in["direct_plans"]) {
        if(k++)out<<',';Plan p=planFrom(entry["plan"]);out<<"{\"id\":"<<entry["id"].as<int>()<<",\"values\":[";int n=0;
        for(double u:{.15,.3,.6,.9,1.2}) {
          Vec v=on(p,p.start+u).p;double unknown=0,occupied=0;
          // Sparse contextual probes, NOT a body swept-volume certificate.
          for(double x:{-.35,0.,.35})for(double y:{-.35,0.,.35})for(double z:{-.2,0.,.2}) {
            Vec q=v+Vec(x,y,z);unknown+=map->state(q)==2;occupied+=map->inflated(q);
          }
          if(n++)out<<',';out<<unknown/27.<<','<<occupied/27.<<','<<(map->safe(v)?1:0);
        }
        out<<"]}";
      }out<<']';
    }
    else if(op=="label_schedules") {
      if(!in["schedules"]||in["schedules"].size()>45)throw std::runtime_error("label schedules size");
      out<<"\"ok\":true,\"branches\":[";int i=0;
      for(auto entry:in["schedules"]) {if(i++)out<<',';Y q=YAML::Clone(entry);q["t"]=in["t"];
        // Offline future labels must drop commands that have already been
        // superseded, exactly as the online executor pops its schedule.
        auto old=entry["execution_schedule"];size_t active=0;
        for(size_t j=1;j<old.size();++j)if(old[j]["start"].as<double>()<=in["t"].as<double>()+1e-9)active=j;
        Y remaining(YAML::NodeType::Sequence);for(size_t j=active;j<old.size();++j)remaining.push_back(old[j]);q["execution_schedule"]=remaining;
        out<<"{\"id\":"<<entry["id"].as<int>()<<','<<guard(q)<<'}';}out<<']';
    }
    else if(op=="make_stops") {
      if(!in["direct_plans"]||in["direct_plans"].size()>15)throw std::runtime_error("make stops size");
      out<<"\"ok\":true,\"branches\":[";int i=0;
      for(auto entry:in["direct_plans"]) {if(i++)out<<',';Plan p=planFrom(entry["plan"]),b;std::string why;
        bool ok=stop(p,p.start+.6,b,why);out<<"{\"id\":"<<entry["id"].as<int>()<<",\"ok\":"<<(ok?"true":"false");
        if(ok)out<<",\"plan\":"<<jplan(b);out<<'}';}out<<']';
    }
    else if(op=="visible_context") {
      State s=state(in["start_state"]);Vec goal=vec(in["goal"]);
      auto context=visibleGuidance(*map,s.p,goal);
      out<<"\"ok\":true,\"context\":{\"endpoints\":"<<jroute(context.endpoints)<<",\"waypoint\":"<<jvec(context.waypoint)<<",\"known_cells\":"<<context.known
         <<",\"visited\":"<<context.visited<<",\"guidance_ms\":"<<context.elapsed_ms<<",\"budget_exhausted\":"<<(context.exhausted?"true":"false")<<'}';
    }
    else if(op=="context") {
      if(!observations)throw std::runtime_error("observation age tracking was not enabled");
      State s=state(in["start_state"]);Vec goal=vec(in["goal"]);double yaw=std::atan2(goal.y()-s.p.y(),goal.x()-s.p.x()),t=in["t"].as<double>();
      Vec forward(std::cos(yaw),std::sin(yaw),0),left(-std::sin(yaw),std::cos(yaw),0);
      out<<"\"ok\":true,\"observation_update_ms\":"<<observation_ms<<",\"context\":{\"center\":"<<jvec(s.p)<<",\"yaw\":"<<yaw<<",\"shape\":[21,21,6],\"grid\":[";
      for(int x=0;x<21;++x)for(int y=0;y<21;++y) {
        double free=0,occupied=0,unknown=0,observed=0,age=0,safe=0;
        for(double z:{-.2,0.,.2}) {
          Vec p=s.p+.3*(x-10)*forward+.3*(y-10)*left+Vec(0,0,z);
          bool inside=map->inside(p);int st=inside?map->state(p):2;bool occ=!inside||map->inflated(p);
          double stamp=observations->stamp(p);free+=st==0;occupied+=occ;unknown+=st==2;observed+=stamp>=0;
          age+=stamp<0?1.:std::clamp((t-stamp)/2.,0.,1.);safe+=map->flight(p)&&st==0&&!occ;
        }
        if(x||y)out<<',';out<<'['<<free/3<<','<<occupied/3<<','<<unknown/3<<','<<observed/3<<','<<age/3<<','<<safe/3<<']';
      }
      out<<"],\"stamp_semantics\":\"last_fused_ray_observation_per_native_voxel\"}";
    }
    else if(op=="observation_query") {
      if(!observations)throw std::runtime_error("age tracking disabled");
      out<<"\"ok\":true,\"stamps\":[";int i=0;for(auto p:in["points"]){if(i++)out<<',';out<<observations->stamp(vec(p));}out<<']';
    }
    else if(op=="evaluate_direct") {
      if(!in["direct_plans"].IsSequence()||in["direct_plans"].size()>15)throw std::runtime_error("direct label candidates invalid");
      out<<"\"ok\":true,\"branches\":[";int i=0;
      for(auto entry:in["direct_plans"]){if(i++)out<<',';out<<'{'<<directOne(in,entry)<<",\"id\":"<<entry["id"].as<int>()<<'}';}out<<']';
    }
    else if(op=="evaluate_recovery") {
      // Offline labels only. Six independent stop probes on the unchanged map.
      if(!in["direct_plans"].IsSequence()||in["direct_plans"].size()>15)throw std::runtime_error("risk candidates invalid");
      out<<"\"ok\":true,\"branches\":[";int i=0;
      for(auto entry:in["direct_plans"]) {
        if(i++)out<<',';Plan p=planFrom(entry["plan"]);out<<"{\"id\":"<<entry["id"].as<int>()<<",\"stops\":[";int j=0;
        for(double u:{0.,.2,.4,.6,.8,1.2}) {
          if(j++)out<<',';Plan b;std::string why;double at=std::min(u,p.p.getTimeSum());bool ok=stop(p,p.start+at,b,why);
          out<<"{\"offset\":"<<at<<",\"ok\":"<<(ok?"true":"false")<<",\"reason\":"<<std::quoted(why)<<'}';
        }
        out<<"]}";
      }
      out<<']';
    }
    else if(op=="evaluate_routes") {
      // Offline labels only. Each branch starts from the identical route cache
      // and map. No candidate is committed and no future scan is available.
      if(!in["candidates"].IsSequence()||in["candidates"].size()!=15)throw std::runtime_error("label probe requires 15 candidates");
      auto saved=std::make_unique<RouteManager>(*router);bool unknown=map->unknown_is_occupied;
      out<<"\"ok\":true,\"branches\":[";
      for(int i=0;i<16;++i) {
        router=std::make_unique<RouteManager>(*saved);map->unknown_is_occupied=unknown;
        Y q=YAML::Clone(in);q["op"]="plan";q.remove("candidates");q["with_backup"]=true;
        if(i<15){q["neural_routes"].push_back(in["candidates"][i]);q["probe_only"]=true;}
        auto begin=Clock::now();std::string value=solve(q);Y r=YAML::Load("{"+value+"}");
        if(r["ok"].as<bool>()) {Plan nominal=planFrom(r["plan"]),backup;std::string why;
          if(!stop(nominal,std::min(nominal.start+.6,nominal.start+nominal.p.getTimeSum()),backup,why))value="\"ok\":false,\"reason\":\"no_valid_backup\"";}
        if(i)out<<',';out<<'{'<<value<<",\"action\":"<<i<<",\"branch_ms\":"<<ms(begin)<<'}';
      }
      router=std::move(saved);map->unknown_is_occupied=unknown;out<<']';
    }
    else if(op=="guard")out<<guard(in);
    else if(op=="plan") {
      std::string solved=solve(in);
      if(in["with_backup"].as<bool>(false)) {
        Y result=YAML::Load("{"+solved+"}");
        if(result["ok"].as<bool>()) {
          Plan nominal=planFrom(result["plan"]),backup;std::string why;
          double branch=std::min(nominal.start+.6,nominal.start+nominal.p.getTimeSum());
          if(stop(nominal,branch,backup,why))solved+=",\"backup_plan\":"+jplan(backup);
          else solved="\"ok\":false,\"reason\":\"no_valid_backup\"";
        }
      }
      out<<solved;
    }
    else if(op=="validate") {
      Plan p=planFrom(in["plan"]);State s=state(in["start_state"]);double vb=0,ab=0;std::string reason;
      bool ok=check(p.p,s,reason,vb,ab);double t=in["t"].as<double>();
      Plan prepared;
      if(ok&&in["prepare_backup"].as<bool>(false)) {
        if(!stop(p,std::min(p.start+.6,p.start+p.p.getTimeSum()),prepared,reason)){ok=false;reason="no_valid_backup";}
      }
      if(ok&&in["backup_plan"]) {
        Plan b=planFrom(in["backup_plan"]);State junction=on(p,b.start);
        if(b.start<p.start||b.start>p.start+p.p.getTimeSum()){ok=false;reason="backup_branch_time";}
        else ok=check(b.p,junction,reason,vb,ab);
      }
      if(ok&&t>=p.start-cfg.deadline_guard){ok=false;reason="deadline";}
      if(ok&&t-map->stamp>.5){ok=false;reason="stale_cloud";}
      out<<"\"ok\":"<<(ok?"true":"false")<<",\"reason\":"<<std::quoted(reason);
      if(ok&&in["prepare_backup"].as<bool>(false))out<<",\"backup_plan\":"<<jplan(prepared);
    } else if(op=="monitor") {
      bool ok=true;std::string reason="ok";double vb=0,ab=0;
      if(in["plan"]){Plan p=planFrom(in["plan"]);double from=std::clamp(in["t"].as<double>()-p.start,0.,p.p.getTimeSum());ok=check(p.p,State{},reason,vb,ab,false,from);}
      else {ok=map->safe(vec(in["p"]));if(!ok)reason="hover_unknown_or_occupied";}
      out<<"\"ok\":"<<(ok?"true":"false")<<",\"reason\":"<<std::quoted(reason);
    } else if(op=="query") {
      out<<"\"ok\":true,\"cells\":[";size_t i=0;for(auto p:in["points"]){Vec q=vec(p);if(i++)out<<',';out<<'['<<map->state(q)<<','<<(map->inflated(q)?1:0)<<']';}out<<']';
    } else if(op=="update")out<<"\"ok\":true";
    else throw std::runtime_error("unknown operation");
    return out.str();
  }
};

int main(int argc,char** argv) {
  if(argc<4){std::cerr<<"usage: pointcloud_nav_worker ego|rog rog.yaml navigation.json\n";return 2;}
  std::filesystem::create_directories("log");rclcpp::init(argc,argv);
  try {
    Navigator nav(argv[1],argv[2],argv[3]);std::cout<<"NAV_READY\n"<<std::flush;std::string line;
    while(std::getline(std::cin,line)) {
      auto begin=Clock::now();std::string result;auto optimizer_before=nav.optimizer_calls;
      try {result=nav.handle(YAML::Load(line));}
      catch(const std::exception& e){std::ostringstream s;s<<"\"ok\":false,\"error\":"<<std::quoted(std::string(e.what()));result=s.str();}
      std::cout<<"\nNAV_RESPONSE {"<<result<<",\"optimizer_calls\":"<<(nav.optimizer_calls-optimizer_before)<<",\"worker_ms\":"<<std::setprecision(17)<<ms(begin)<<"}\n"<<std::flush;
    }
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';rclcpp::shutdown();return 1;}
  rclcpp::shutdown();return 0;
}
