# 第三方来源与保留内容

当前 ROS 包的原始声明为 GPL-3.0-or-later，见 `src/pointcloud_nav/package.xml`；GPL 文本在根目录 `LICENSE` 和 EGO 子目录保留。各嵌入库自己的版权和许可证说明继续适用，不把第三方代码认作本项目原创。

- EGO 优化器与地图适配使用原工作树中固定的 ROS2 EGO-Swarm 衍生代码，记录提交 `23a8d5a191711dd65633df689bd00f55d4dea8f9`。来源文件哈希、移植前后记录见 `src/pointcloud_nav/SOURCE_MANIFEST.json`；当前版本包含原工程已有的移植和单机修改。
- ROG 子集取自 [SUPER](https://github.com/hku-mars/SUPER) 提交 `2ad3419c127a617c6d7df6925e81a14175a9c096`，其地图来自 [ROG-Map](https://github.com/hku-mars/ROG-Map)。这不是对整个 SUPER 的复现。嵌入的 fmt、tinycolormap、backward 等库保留文件内版权和许可说明。
- [YOPO](https://github.com/TJU-Aerial-Robotics/YOPO) 使用 `YOPO-Simple` 分支提交 `bd8a7f91953cfef9f1aa9a48d7ff0ab7bceee132`，MIT 许可证位于 `third_party/yopo/LICENSE`，原文件哈希在 `third_party/yopo/upstream_manifest.json`。

当前 D435i 多候选/生存预测模型是本工程训练的轻量结构，不调用 YOPO 原版预训练网络。旧 `tools/yopo_frontend.py` 等历史入口仍保留，直接运行这些旧入口另需作者的 `epoch50.pth`，因此不属于 README 中已验证的当前默认入口。

如确需复现旧 YOPO 作者权重对照，可在上述固定上游提交中取得 `YOPO/saved/YOPO_1/epoch50.pth`，放入本仓库同名路径。原 SHA-256 是 `09ec31094ed83e09702efc1facf18076564ebbe1afc164960c41036e16ba229f`；该可选文件已被忽略，不再重复上传。当前仓库中所有自训练权重及必要对照权重均保留。
