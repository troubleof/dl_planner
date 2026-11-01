路径复用与局部修复（2026-10-29）

当前运行目录：/home/chaonian/ego-planner-swarm/single-uav-nav
默认策略：repair。原来的完整搜索可用 --route-policy full 选择。

快速运行
  cd /home/chaonian/ego-planner-swarm/single-uav-nav
  CMAKE_BUILD_PARALLEL_LEVEL=2 bash build_navigation.sh
  bash run_navigation.sh --backend both --scenes offset_wall --delays .08 --route-policy repair --out runs/navigation-repair-user-demo

三策略对照
  source /opt/ros/humble/setup.bash
  OPENBLAS_NUM_THREADS=1 python3 tools/run_route_comparison.py --out runs/navigation-repair-my-comparison --scenes offset_wall forest --seeds 11 --delays 0 .08

参数均来自 config/navigation.json；每次运行保存实际配置。
  full：每次完整 A*，保留上一版参考路径搜索行为。
  reuse：复用当前地图上有效的缓存参考路径，否则完整 A*。
  repair：先复用，再尝试局部 A* 修补冲突段，失败则完整 A*。

具体行为
1. 缓存从当前状态到目标的几何参考路径。未知空间在这一层保持乐观处理。
2. 从新的未来接续位置裁剪旧路径，并重新检查连接段和保留段。
3. 新地图版本使查询缓存失效。采用体素穿越算法，检查边跨过的体素及统一的引导余量，避免只检测节点或漏掉短距离穿角。
4. 找到连续失效边，向两侧各保留约 0.6 m 的连接距离。修补段长度上限 4 m，局部搜索盒额外扩展 1.2 m，每次搜索软时间预算 40 ms，每个请求最多修补两次。
5. 局部 A* 使用相同的体素边检查；修补后再次检查拼接路径。端点偏离、越过局部搜索盒、仍有冲突、连接距离过大或搜索失败时回退。
6. 目标变化、地图版本重置、无法连接旧路径都会放弃缓存。复用/修复后的样条若未通过优化或整段验证，下一次请求改用完整搜索。
7. 修复的是参考路径。最终样条仍要通过已有的未知空间、占用、包络、速度/加速度、接续连续性和最新地图检查。发布截止时间保持不变。

实现范围
当前是“按地图版本复核缓存路径，按冲突区间局部搜索”。没有给 EGO/ROG 增加全图变化体素日志，也没有实现 D* Lite/LPA* 的搜索树增量更新。
同一地图版本内缓存体素查询结果；新版本会清空查询缓存。地图更新、检查、搜索、优化和 IPC 耗时仍进入执行时序。
full 的参考边保留原来的采样检查；缓存与局部修复使用更保守的体素穿越检查。三种策略的最终样条验证相同。
参考路径缓存不是可执行轨迹缓存，不绕过停车边界或后端验证。

关键文件
  src/pointcloud_nav/include/route_manager.hpp：路径缓存、连接、体素穿越、失效区间与局部拼接。
  src/pointcloud_nav/src/worker.cpp：接入与完整样条验证、日志。
  src/pointcloud_nav/include/collision_map.hpp：可临时限制局部搜索范围。
  src/pointcloud_nav/third_party/ego/.../dyn_a_star.*：可选连边验证器与搜索时间预算；full 模式仍使用原参考搜索检查。
  tools/run_route_comparison.py：同场景交错运行三策略。
  tools/replay_route_comparison.py：相同采集位姿、点云和状态输入回放；不能将其候选有效率当成闭环成功率。
  tools/audit_route_repair.py：源码冻结、共同参数、提交记录、曲线动力学与接续审计。

验证命令
  source /opt/ros/humble/setup.bash
  install/pointcloud_nav/lib/pointcloud_nav/route_manager_check
  OPENBLAS_NUM_THREADS=1 python3 tools/test_navigation_contract.py --out runs/navigation-repair-my-contract
  OPENBLAS_NUM_THREADS=1 python3 tools/test_route_repair_integration.py --out runs/navigation-repair-my-integration

已有结果
  runs/navigation-repair-evaluation-v1：96 组闭环与 audit.json。
  runs/navigation-repair-matched-replay-v1：36 次固定输入回放。
  runs/navigation-repair-integration-v2：新障碍使旧候选失效、样条验证失败回退及迟交拒绝。
  runs/navigation-repair-dev-v1：开发场景记录，未混入正式统计。

使用边界
固定小地图、静态几何、理想里程计/执行与有限垂向视场的仿真条件不变。尚未验证真实控制误差、动态障碍运动和 ROG 滑动窗口。
复用可能保留较长的绕行路径或出现额外转向；本轮平均飞行时间增加约 3.6%–6.1%。减少计算并不等于飞得更快。
局部修复后也可能因未知空间而无法生成可提交样条。这是应拒绝的情况，不能把几何修补成功当作飞行成功。
若迁移固定输入回放数据到其他机器，requests.jsonl 中保留的原绝对点云路径需要重定位；闭环试验可直接按上述命令重新生成点云。
