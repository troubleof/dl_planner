两帧观察、局部地图与多步候选评分实验，2026-10-30

本目录延续单机点云导航和此前YOPO适配代码。默认repair、EGO优化器、地图碰撞验证和已有执行/停车备份不替换。
ContextScorer是重新训练的35461参数评分网络，不是原YOPO权重微调或完整YOPO复现。源码保留原作者模型和上一轮RangeScorer用于历史比较。

新模型输入：两张3×17×72距离/命中/有效图，6×21×21局部地图摘要，33维自机/观察/预算状态。
地图年龄指原生0.1m体素最后一次被已融合射线观测的采集时间，不是占据状态更新时间。
模型给四条几何参考和repair共五动作评分。EGO实际生成B样条，原验证器检查名义轨迹、停车备份和提交条件。
观测、推理和规划耗时均进入执行时序；模型加载和预热在飞行前完成。

训练标签来自相同历史、相同接续P/V/A和感知通道状态下的真实2秒分支续跑。分支之后的新扫描按各自实际位置生成。
第一轮继续repair，第二轮收集第一轮模型访问的训练状态并用该模型继续。此为一次DAgger式聚合，有限候选和窗口不是全局最优专家。
中止、超时和未找到有效执行退路都保留；branch_horizon只表示短分支窗口结束，不是整次任务成功。

先按build_navigation.sh构建。原环境使用Ubuntu-22.04、ROS2 Humble、Python3.10、CPU PyTorch2.4.1、NumPy1.21.5、SciPy1.8.0。
OPENBLAS_NUM_THREADS=1和OMP_NUM_THREADS=1；调用含torch的环境解释器。
模型部署测试示例（必须使用新的输出目录）：
python tools/run_context_experiment.py --stage test --policies repair context_repair ensemble --out runs/my-context-comparison

该命令仅在已具备context_models/round1三个模型时运行；源码交付包含这些权重。
完整训练、数据划分、对照、限制、源码哈希和复现命令见交付根的中文报告、02_复现命令.txt和evidence。
