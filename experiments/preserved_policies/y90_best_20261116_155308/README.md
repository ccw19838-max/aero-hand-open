# Y轴最佳策略保留

保存于2026-11-16 15:53:08（北京时间），后续X轴训练不会更新这些参数。

策略为此前选出的Y90°最佳检查点，本段训练步数589,824，平均最佳姿态误差4.68°。当前保存的是已录制约90°姿态接近效果的版本；五指稳握1秒尚未通过。

- 完整检查点：checkpoint/（policy.zip、奖励归一化状态、参考状态、模型报告、元数据）。
- 每个文件SHA256与原件逐一相同，见manifest.json。
- 检查点文件及目录设为只读，训练分配为0。
- 原速与慢放回放保留在preview/（如存在）。
- 源检查点：/workspace/aero-hand-open/experiments/cube_adaptive_60h/artifacts/goal_20261113_1536/y/segment_006/checkpoints/step_000589824_8b295922

恢复推理可用experiments.cube_adaptive_60h.storage.load读取checkpoint目录；需保留其记录的匹配代码和依赖。
