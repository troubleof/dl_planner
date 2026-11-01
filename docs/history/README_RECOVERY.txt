候选重新评价与可恢复性评分，2026-10-31

本目录继续上一版range-trajectory工程。推荐入口为tools/run_recovery_experiment.py；继承的旧README描述历史实验，原repair入口不变。
当前运行目录：/home/chaonian/candidate-recovery-20261031/single-uav-pointcloud-nav

模型：冻结trajectory_models中的seed11/33候选生成器；新recovery_models评分器同时看距离图、状态和实际候选几何。
四个动作：repair、固定候选束、seed11偏移候选束、seed33偏移候选束。原EGO/验证/停车备份/提交时限全部保留。
short_s11.pt是同状态候选重新评价后的短期监督对照；long_s11/s22/s33.pt是6秒repair续跑监督，不是网络自己续跑的完整RL价值。
风险输出是经验性预测，不是校准安全概率。不得直接把几何路点发给飞控。

源码：recovery_frontend.py；采集：collect_recovery_branches.py；训练：train_recovery_model.py。
标签审计：audit_recovery_labels.py；闭环审计：audit_recovery_results.py。
导出：export_recovery_models.py；导出的评分器仍需两个冻结生成器和完整编码/解码。

环境与现有ROS2 Humble相同。加载ROS环境，设置OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1后运行：
python tools/run_recovery_experiment.py --stage pilot --out runs/new-pilot
python tools/run_recovery_experiment.py --stage test --out runs/new-test
python tools/audit_recovery_results.py --root . --run runs/new-test --out runs/new-audit

输出目录必须是新目录。正式新测试图3601—3604；3503和offset_wall为开发用过的回归图，必须分开报告。
训练数据v2继承v1完整分支并跳过无法恢复的初始化；证据包必须同时保留v1、v2与midpoints目录。初始化失败不算飞行成功或失败。
不根据测试成绩挑一个种子当默认策略。完整结果与限制见交付根中文报告。
