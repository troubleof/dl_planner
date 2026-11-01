# dl_planner

基于现有 EGO/ROG 工程的**单机点云避障研究**。当前工作是 D435i 前向视野下的带时标候选轨迹、地图更新后的退路有效性与小型网络训练。保留 repair/EGO 优化器作为对照；当前学习方案直接解码轨迹，仍经过硬验证和有限停车回退。

这是 2026-11-01 整理的可继续迭代仓库，沿用原工程源码与模型。以后以 Git 提交和标签标记版本，不再每轮复制完整工程或打包整个 `runs/`。

**当前限制：尚未证明模型稳定优于规则；存在包络相交、无有效退路和超时失败。没有 D435i 实机或 PX4 SITL 成绩。**

## 保留范围

| 位置 | 内容 |
|---|---|
| `src/`、`config/`、`tools/` | 原单机规划器、两种地图、模型、训练、仿真与验证工具 |
| `survival_models/` | 最新三个结构各三个种子及两个消融，共 11 份权重 |
| `visible_models/` 等模型目录 | 小体积历史模型与固定几何对照；用于保持已有比较入口 |
| `deployment_survival/` | 三个结构的 CPU TorchScript 与原计时记录 |
| `datasets/recovery-v4/` | 必要训练数组、精简索引、失效事件与数据协议 |
| `examples/cases/` | **5 个典型完整案例**，单例压缩包及逐文件哈希 |
| `reports/2026-11-01/` | 中文报告、指标、失败清单、审计摘要和图 |
| `provenance/` | 导入源码来源与保留资产的 SHA-256 |
| `docs/history/` | 历史说明，仅用于理解过程，当前入口以本 README 为准 |

已按用户要求删除的大部分原始日志和完整 evidence 包没有恢复或上传。历史报告中的“完整归档已校验”是**清理前的状态**，不能理解为全部原始数据仍可回放。当前数据可再训练；已删除案例无法从原始点云重新标注。

原 YOPO 作者的 43 MiB 预训练文件不是当前 D435i/repair/生存预测实验的依赖，未随核心包重复上传。其来源、原哈希与可选恢复方式见 [第三方来源](docs/THIRD_PARTY.md)。

## 编译与验证

目标环境：Ubuntu 22.04 / ROS2 Humble / Python 3.10，PyTorch 2.4.1 CPU。当前桌面是在 Windows + WSL2 上测试，硬件与依赖版本见 [环境记录](reports/2026-11-01/environment.json)。

首次安装需已有 ROS2 Humble，并具备 `colcon`、Eigen3、PCL、OpenCV、yaml-cpp、fmt 和本包声明的 ROS 依赖。Python 需要 numpy、scipy、torch、PyYAML，绘图脚本另需 matplotlib；ROS Python 依赖使用系统环境。

```bash
git clone https://github.com/troubleof/dl_planner.git
cd dl_planner
source /opt/ros/humble/setup.bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
CMAKE_BUILD_PARALLEL_LEVEL=2 bash build_navigation.sh

# py 指向具备上述依赖的 Python；不要把原机器上的虚拟环境绝对路径写进代码。
py=python3
$py tools/check_core.py --replay
$py tools/test_perception_channel.py
install/pointcloud_nav/lib/pointcloud_nav/route_manager_check

tag=$(date +%Y%m%d-%H%M%S)
$py tools/test_d435i_contract.py --out "runs/check-camera-$tag"
$py tools/test_backup_contract.py --out "runs/check-backup-$tag"
```

也可运行 `PYTHON="$py" bash tools/check_checkout.sh`，依次检查感知通道、路径缓存、相机和停车接续，再进行全数据/基础数据各一轮训练试跑。试跑权重位于被忽略的 `runs/`，不会替换正式权重。

本机固定工作目录是 `D:/UAV_Research/ego/dl_planner`，在 WSL 中为 `/mnt/d/UAV_Research/ego/dl_planner`。本机可复用 `/home/chaonian/yopo-ego-20261029/venv/bin/python`，它不属于仓库依赖路径。编译产物和原始运行目录被 `.gitignore` 排除。

`check_core.py` 检查资产哈希、全部新权重加载、训练/验证/测试地图隔离、标签与冻结训练结果一致、五个案例的文件及点云引用。`--replay` 使用当前 worker 重建案例中的融合地图并调用已记录的 escape 请求；这是固定输入检查，不是重跑旧飞行，也不声称复现原 CPU 时序。

## 再训练

```bash
tag=$(date +%Y%m%d-%H%M%S)
$py tools/train_recovery.py --dataset datasets/recovery-v4 \
  --out "runs/train-$tag"

# 只学当前可行性消融
$py tools/train_recovery.py --dataset datasets/recovery-v4 \
  --out "runs/current-only-$tag" --kinds spatial --seeds 11 --current-only

# 不使用候选自身视角增广：取已验证一致的前 827 行，无需再存一份数据集。
$py tools/train_recovery.py --dataset datasets/recovery-v4 \
  --out "runs/base-only-$tag" --kinds spatial --seeds 11 --base-only
```

默认每个结构种子 11/22/33、60 epochs，按验证损失选权重。训练数组保持原始字节与哈希；196 MB 的原查询元数据改为约 0.6 MB 的分组/时间/来源索引，模型实际输入及标签没有因此改动。`label_events` 使用数据集内相对路径，不再依赖旧 `runs/recovery-data-v3`。

数据共 1,027 行，训练 869、验证 158；其中 827 行基础状态和 200 次候选自身视角分支，相关记录不能当作独立地图样本。历史 4382/4482/4483 已用于开发；保留的测试地图为 5401–5404。后续继续调参后，应另取新地图进行最终泛化评价。

## 小规模仿真

```bash
tag=$(date +%Y%m%d-%H%M%S)
$py tools/run_recovery_experiment.py --out "runs/smoke-$tag" \
  --split regression --seeds 3104 \
  --policies repair short_recover riskrule mlp11 spatial11 temporal11

# 使用新训练目录时：--weights "runs/train-$tag"
```

同一入口可比较 `repair`、`repair_recover`、`short`、`short_recover`、`old`、`old_recover`、`bankrule`、`riskrule` 及各模型。完整矩阵每种方案为四张地图 × 两后端；无需为了确认环境就重复跑全部矩阵。运行会产生排错用原始日志，默认不进 Git；完成数据生成或排错后按 [留存约定](docs/RETENTION.md) 整理。

当前仿真是 D435i 106×60 理想深度射线、5 Hz、有界朝向、理想定位/执行与静态几何。0.30 s 接续、25 ms 提交提前量、0.5 s 过期门限和 80 ms 模拟规划延迟仍保留；计算预算与额外模拟延迟分别计量。

## 五个完整案例

| ID | 保留目的 |
|---|---|
| `new-observation-escape-success` | 新观测后提前停车成功，并完成到达 |
| `insufficient-escape-time` | 无及时有效退路，明确记录失败 |
| `temporal-model-envelope-failure` | 双帧模型重复试验中的包络相交失败 |
| `historical-ego-delay400` | 历史 EGO 延迟下的执行前缀失效 |
| `historical-rog-delay400` | 历史 ROG 对应失效；仅保留诊断，不重新开展延迟矩阵 |

```bash
$py tools/extract_case.py new-observation-escape-success \
  --out runs/retained-escape-success
```

解压前验证原压缩包和逐文件哈希，解压后只在新目录内重定位 JSON 路径；原压缩包不变。每例保留场景、配置、轨迹、事件、请求、相机/传感器与原始点云等实际已有内容。

## 当前结果与边界

最新主比较：MLP 23/24、单帧 CNN 23/24、双帧 CNN 22/24（三个种子各 8 次）；同候选库未知惩罚规则 8/8，上一版 CNN 8/8。重复实验出现双帧模型包络失败，新增自身视角分支数据的收益未获本轮消融证实。

桌面 CPU 网络推理 P95 为 0.07–0.46 ms，但完整规划 RPC P95 为 41–60 ms。25 ms 计算预算下新模型均未到达。质心已知自由加膨胀占据检查不等于完整机体扫掠体积安全；未来有效性概率只用于排序，不是安全证书。

详见 [中文实验报告](reports/2026-11-01/01_实验结果与限制.txt) 和 [论文查新](reports/2026-11-01/03_论文查新与架构假设.txt)。
