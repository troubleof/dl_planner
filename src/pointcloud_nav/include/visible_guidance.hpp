#pragma once
#include "collision_map.hpp"
#include <queue>
#include <chrono>
#include <algorithm>

// Bounded 2-D guidance over CURRENT known-free cells. Never changes occupancy.
struct VisibleGuidance {
  Route endpoints;
  Vec waypoint=Vec::Zero();
  int visited=0,known=0;
  bool exhausted=false;
  double elapsed_ms=0;
};
inline VisibleGuidance visibleGuidance(CollisionMap& map,const Vec& start,const Vec& goal,double budget_ms=8.) {
  auto begin=std::chrono::steady_clock::now();
  auto elapsed=[&](){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();};
  VisibleGuidance out;out.waypoint=start;
  constexpr int n=61,mid=30,total=n*n;constexpr double h=.1;
  std::vector<int> safe(total,-1),parent(total,-1);std::vector<double> dist(total,1e9);
  auto point=[&](int i)->Vec{return start+Vec((i/n-mid)*h,(i%n-mid)*h,0);};
  auto free=[&](int i){if(safe[i]<0){safe[i]=map.safe(point(i));out.known+=safe[i];}return safe[i]>0;};
  auto edge=[&](const Vec& a,const Vec& b){int steps=std::max(1,int(std::ceil((b-a).norm()/.025)));for(int k=0;k<=steps;++k)if(!map.safe(a+(b-a)*double(k)/steps))return false;return true;};
  int source=mid*n+mid;
  if(!free(source)){out.elapsed_ms=elapsed();return out;}
  using Q=std::pair<double,int>;std::priority_queue<Q,std::vector<Q>,std::greater<Q>> queue;
  queue.push({0.,source});dist[source]=0.;parent[source]=source;std::vector<int> reachable;
  while(!queue.empty()) {
    if(elapsed()>budget_ms){out.exhausted=true;break;}
    auto [d,i]=queue.top();queue.pop();if(d>dist[i]+1e-9)continue;
    ++out.visited;reachable.push_back(i);int x=i/n,y=i%n;
    for(int dx=-1;dx<=1;++dx)for(int dy=-1;dy<=1;++dy) {
      if(!(dx||dy)||x+dx<0||x+dx>=n||y+dy<0||y+dy>=n)continue;
      int j=(x+dx)*n+y+dy;double nd=d+h*std::hypot(dx,dy);
      if(nd>3.5||nd>=dist[j]||!free(j))continue;
      if(dx&&dy&&(!free((x+dx)*n+y)||!free(x*n+y+dy)))continue;
      if(!edge(point(i),point(j)))continue;
      parent[j]=i;dist[j]=nd;queue.push({nd,j});
    }
  }
  double before=(goal-start).norm();
  auto score=[&](int i){return before-(goal-point(i)).norm()-.08*dist[i];};
  std::sort(reachable.begin(),reachable.end(),[&](int a,int b){return score(a)>score(b);});
  Route anchors;
  if(before<=1.8&&before>.05&&edge(start,goal))anchors.push_back(goal);
  int inspected=0;
  for(int target:reachable) {
    if(++inspected>128||elapsed()>budget_ms+2.){out.exhausted=true;break;}
    if(score(target)<.08||anchors.size()==5)break;
    Route path;for(int i=target;i!=source;i=parent[i])path.push_back(point(i));std::reverse(path.begin(),path.end());
    Vec end=start;for(const Vec& p:path){if((p-start).norm()>1.8||!edge(start,p))break;end=p;}
    if((end-start).norm()<.15)continue;
    bool duplicate=false;for(const Vec& a:anchors)if((a-start).normalized().dot((end-start).normalized())>.97)duplicate=true;
    if(!duplicate)anchors.push_back(end);
  }
  // Directional scans also preserve very short prefixes that the grid rounds away.
  double yaw=std::atan2(goal.y()-start.y(),goal.x()-start.x());
  for(double angle:{0.,-.52,.52,-1.05,1.05,-1.57,1.57}) {
    if(anchors.size()>=5)break;
    Vec dir(std::cos(yaw+angle),std::sin(yaw+angle),0),end=start;
    for(double d=.025;d<=1.8+1e-9;d+=.025){Vec p=start+dir*d;if(!map.safe(p))break;end=p;}
    if((end-start).norm()<.15||before-(goal-end).norm()<.1)continue;
    bool duplicate=false;for(const Vec& a:anchors)if((a-start).normalized().dot((end-start).normalized())>.97)duplicate=true;
    if(!duplicate)anchors.push_back(end);
  }
  for(int i=0;i<5;++i) {
    Vec a=anchors.empty()?start:anchors[std::min(i,int(anchors.size())-1)];double length=(a-start).norm();
    for(double cap:{.25,.65,1.4}) {
      // Grid-edge sampling is only proposal guidance: interpolated caps can
      // hit an unsampled voxel sliver. Explicitly certify each final endpoint.
      double used=std::min(cap,length);Vec direction=(a-start)/std::max(length,1e-9);
      Vec p=start+direction*used;
      while(used>1e-9&&!map.safe(p)){used=std::max(0.,used-.025);p=start+direction*used;}
      out.endpoints.push_back(p);
    }
  }
  if(!anchors.empty())out.waypoint=anchors.front();
  out.elapsed_ms=elapsed();return out;
}
