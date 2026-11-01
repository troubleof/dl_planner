直接带时标轨迹分支

当前版本 direct_timed_bspline_v2，67,371个参数。
输入：现有距离/命中/有效图3×17×72和32维自机、目标、帧年龄/预算及后端状态。
输出：15条候选，每条持续时间、六个二维控制残差、终点XY和评分。
解析解码：当前P/V/A、终端静止，P/V/A感知的基础曲线和固定线性转换；垂直分量解析追踪目标高度。当前是水平导航研究，不是一般3D学习规划器。

direct11/direct22/direct33：在线禁止EGO优化器，原验证拒绝所有候选时继续已经提交的停止序列；可能停车后仍无法导航，照实记为超时或退路失效。
direct_repair11：直接候选均被拒绝时允许repair＋EGO接管；单独统计优化器调用与接管请求，不当作纯直接模型的成绩。
repair和recovery11/recovery33：原对照权重、地图、优化器与执行协议。
默认navigation.json仍是repair。

核心新增/修改：
tools/direct_frontend.py：网络、带时间控制点解码、前端。
tools/build_direct_teachers.py、convert_direct_teachers.py、train_direct_geometry.py：离线EGO教师与参数化训练。
tools/evaluate_direct_predictions.py、train_direct_scores.py：实际网络输出重评及冻结几何后的评分训练。
tools/test_direct_contract.py、audit_direct_training.py、audit_direct_results.py：拒绝/接续契约、数据核验、独立地图执行回放。
src/pointcloud_nav/src/worker.cpp：直接曲线准入入口和真实优化器调用计数；check/stop/guard原函数保持一致。
tools/navigation_common.py：前端结果按direct_plans传输；实际前端/序列化耗时计入RPC。

保留v1教师、权重和16例先导（8个直接/混合失败），以及v2所有结果。source_stages含修订前版本。
数据、训练、代码冻结与复现详见交付目录的中文报告。
