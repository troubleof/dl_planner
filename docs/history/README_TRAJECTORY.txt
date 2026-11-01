距离图CNN＋多候选轨迹头（2026-10-31）

本目录是当前context-dagger单机工程的隔离副本。EGO、ROG、repair、验证器、停车备份和执行器没有改动。
新前端 tools/trajectory_frontend.py 输出几何参考，不能直接用作飞控命令。

输入：已收到的原生17×720射线，采集原点/采集时间/自机PVA/未来接续PVA/目标。
距离、命中、有效掩码压缩成3×17×72；两层小CNN＋22维状态；15个轨迹分支各3路点的有界偏移，另有16质量值与16可行性logit（含repair）。
最多提出3条参考；EGO重新优化并验证名义曲线和停车备份，提交检查维持原门限。
过期、未来帧、错误点数、非有限输入、非有限输出和近目标都回退repair；默认旧入口仍使用repair。
可行性输出仅是排序特征，未校准为安全概率。

训练不是原YOPO权重微调，也不是原论文guidance learning的复现。
使用已有训练/验证分支中的EGO曲线，按弧长取3点作为几何教师；无效分支不提供几何监督。
质量/可行性标签对应教师原候选，不是模型改变几何后的精确价值。轨迹偏移的重新评价依靠闭环实验和验证器。
短期目标、数据规模与分布偏移是明确限制；旧两帧/DAgger模型保留。本轮验证轻量单帧轨迹头，不声称解决了之前的长期可恢复性问题。
预算特征沿用固定额外80ms协议，没有证明跨预算泛化；模型不含逐体素年龄、全局地图或动态障碍状态。

WSL目录：/home/chaonian/range-trajectory-20261031/single-uav-pointcloud-nav
Python：/home/chaonian/yopo-ego-20261029/venv/bin/python
先运行 build_navigation.sh；source /opt/ros/humble/setup.bash；设置 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1。
使用全新的输出目录：
python tools/test_trajectory_contract.py --out runs/my-contract
python tools/run_trajectory_experiment.py --stage pilot --out runs/my-pilot
python tools/run_trajectory_experiment.py --stage test --out runs/my-test
python tools/audit_trajectory_results.py --root . --run runs/my-test --out runs/my-audit

正式对照：repair、old11（历史固定候选评分器）、fixed11（新模型相同权重但几何偏移置零）、trajectory11/22/33。
训练forest3101—3106，验证3201/3202；新测试3501—3504，offset_wall只作历史回归。
每策略10次，失败按35秒计入汇总，三种子全部报告；同分布新地图不是跨分布/真机泛化证明。

参考YOPO： https://tju-air-lab.github.io/projects/YOPO/
下一条RL路线P2M： https://github.com/arclab-hku/P2M
P2M作者代码需要Isaac Sim/Lab、OmniDrones及流估计相关组件；本轮不安装、不替换现有ROS2工程，也不声称已接入P2M。
后续先把模型动作改成“候选/速度档/repair选择”，在原验证器后评估，再考虑动态点云流与交互RL。
