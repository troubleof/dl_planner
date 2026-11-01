单机点云避障实验入口（2026-10-28）

2026-10-29 更新：加入路径复用与局部修复，config/navigation.json 默认 route_policy=repair。
--route-policy full 保留完整搜索；reuse 复用有效路径、失效后完整搜索；repair 先修复冲突段，失败再完整搜索。
本轮详细说明与结果见 README_PATH_REPAIR.txt。旧版 2026-10-28 审计需配合旧源码包；新实验使用 audit_route_repair.py。

这是从当前 ROS2 工程中抽出的独立实验组件：点云 + 自身 P/V/A + 目标点 -> EGO B 样条 -> 延迟提交 -> 单机执行。
运行时不需要前机、目标估计、历史跟随、可见性代价，也不需要原 ego-planner 工作区的 install。
采用原工程所依赖的 ROS2 EGO-Swarm 优化器，而非声称实现了论文所有原版设置。ROG 核心取自已固定版本的 SUPER 仓库。

一、运行位置与命令

当前开发工作树：/home/chaonian/ego-planner-swarm/single-uav-nav
分支：codex/single-uav-pointcloud-baseline
原工程：/home/chaonian/ego-planner-swarm/planner（未修改）
在 WSL Ubuntu-22.04 中：
  cd /home/chaonian/ego-planner-swarm/single-uav-nav
  CMAKE_BUILD_PARALLEL_LEVEL=2 bash build_navigation.sh
  bash run_navigation.sh --backend ego --scenes offset_wall --delays 0 --out runs/my_ego_run
  bash run_navigation.sh --backend both --scenes forest --seeds 41 42 43 --delays 0 .08 --out runs/my_comparison

输出目录必须是新的；已有实验不会被覆盖。未写 --max-time 时每例最大模拟飞行 35 秒。
Windows 的 Run-Navigation.ps1 已封装上述入口。图像由 plot_navigation_results.py 生成，不需要 RViz。

二、独立构建依赖

Ubuntu 22.04 + ROS2 Humble；构建只 source /opt/ros/humble/setup.bash。
依赖 ament/colcon、C++17、Eigen3、PCL、OpenCV、yaml-cpp、fmt，以及 rclcpp、geometry_msgs、nav_msgs、sensor_msgs、visualization_msgs、message_filters、cv_bridge、pcl_conversions。
Python 使用系统 python3、numpy、scipy、matplotlib。没有训练步骤、模型权重或标注数据集。
参考安装包名：python3-colcon-common-extensions libeigen3-dev libpcl-dev libopencv-dev libyaml-cpp-dev libfmt-dev python3-numpy python3-scipy python3-matplotlib ros-humble-cv-bridge ros-humble-pcl-conversions ros-humble-message-filters。
源码包已携带所用 EGO/ROG 源码和原始出处记录，无需重新 clone 原跟随工程。

三、唯一参数入口

config/navigation.json 是 Python 传感器、执行器、C++ 规划器与两种地图共同的参数文件。
每次运行写出 navigation_runtime.json 和由它派生的 rog_runtime.json。config/navigation_rog.yaml 仅保留早期人工对照样例，运行不读取它。
当前参数：机体 0.35×0.35×0.25 m；外接球半径 0.277263 m；安全余量 0.05 m；规划半径 0.327263 m。
0.1 m 体素，ceil(R/h)=4，统一 9×9×9 立方膨胀邻域；这比实际外接球更保守，并非精确球形 Minkowski 和。
参考路径另有 1 个体素的转弯余量，两后端相同；最终安全检查仍采用上述统一包络。
质心高度 2–3 m，速度/加速度上限 3 m/s、6 m/s²；巡航参考 1.5 m/s；局部路径长度 3 m。

四、点云与时序契约

world 坐标下 little-endian float32 XYZ；必须附采集时间戳及同一时刻的传感器原点。
仿真 LiDAR：360° 水平、720 方位角、垂向 -8° 到 +8°、1° 间隔、量程 8 m、5 Hz；共 12,240 条射线。
返回点来自第一交点；无遮挡射线用 range+0.1 m 端点，交给原生地图截断成自由射线。
这不是普通“只含命中点”的 PointCloud2 的无条件直接替代：真实传感器需按射线有效性和外参生成该契约，不能把未知的无回波随意当空闲。
文件二进制传输用于可审计的同步仿真，尚未提供实机 DDS 点云订阅/飞控执行桥。worker 中原生地图回调不 spin，唯一更新源是显式输入。
所有输入/输出元数据存入 requests.jsonl，二进制点云存入 clouds；规划端不接收场景真值。

一次周期：
  1. 固定候选接续时间 t+0.30 s，并从正在执行的曲线预测该时刻 P/V/A。
  2. 点云更新、路径复用/搜索、EGO rebound/refine、整段验证；实测二进制写入/读取、序列化、IPC 与 C++ 工作耗时。
  3. 模拟时间按实际耗时向前推进，同时旧轨迹继续运行；额外延迟加在本阶段结果可用时间。
  4. 使用期间到达的最新点云再次验证，计入该次实际耗时、5 ms 发布延迟、10 ms 主机提交检查预算。
  5. 固定接续时刻前至少 25 ms 提交。版本变化、待处理更新点云、接续状态变化或迟交都拒绝，不移动起始时间。
传感器真值渲染、离线碰撞评估、审计日志写入不算规划计算时间，分别属于仿真/记录；非硬实时系统。
启动前静止预热 4 帧，仅用于初始化地图，相关耗时另存，不计入飞行时间。
候选结尾速度/加速度均为零；重规划失败时继续旧的已验停车轨迹，不把运动中的无人机瞬间冻结。
如果新点云使不可替换的执行前缀不再安全，实验以 unsafe_execution_prefix 失败结束；没有把立即刹停当作安全成功，也未实现有动力学保证的应急备份轨迹。

五、公平对照及范围

闭环固定场景/种子、起终点、传感器、包络、优化器与时序，只切换地图；轨迹不同会带来不同采集位姿，因此闭环点云不声称逐帧相同。
replay_navigation_maps.py 另将同一批 39 帧点云和采集位姿逐字节送入两地图，比较更新耗时与查询差异。
公共查询域 20×12×5 m；EGO 固定全域重建膨胀，ROG 固定网格增量计数，关闭滑动/ESDF/可视化。ROG 内部奇数体素尺寸及膨胀 halo 按其实现保留。
两种概率地图的未知先验、射线去重、命中/穿越更新方法不同，统一概率参数不意味着后验地图完全相同。
EGO 接入显式 origin+endpoints API，修复遍历标签 char 周期问题及膨胀索引跨轴越界；未观测的传感器所在体素标为空闲，但不清除已有命中。
查询采用同一 1e-7 m 体素面容差；A* 增加半体素间距的连边检测。以上均有 SOURCE_MANIFEST.json、冻结哈希与开发失败记录。

验证入口：
  source /opt/ros/humble/setup.bash
  OPENBLAS_NUM_THREADS=1 python3 tools/test_navigation_contract.py --out runs/my_contract
  OPENBLAS_NUM_THREADS=1 python3 tools/replay_navigation_maps.py --out runs/my_replay
  python3 tools/audit_route_repair.py --root runs/navigation-repair-evaluation-v1

本轮只是理想里程计、理想轨迹执行、静态几何和有限垂向视野下的单机基线。没有验证动态障碍、姿态变化/俯仰滚转、跟踪误差、丢包、真实传感器噪声、ROG 滑动窗口或 SUPER 备份轨迹。
接受曲线要求中心体素已知空闲与已知障碍包络无碰撞；未要求整个机体扫掠体积都已被观测为空闲，因此不是未知环境中的形式安全保证。
详细结果见 01_单机基线与ROG对照结果.txt；应先保留 EGO 为主基线，再做更大地图/滑动窗口及按地图变化选择性复核的后续实验。
