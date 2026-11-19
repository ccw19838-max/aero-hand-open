# X 轴翻面冻结评估

训练尚未取得 90° 独立验收资格。

截至冻结时，30° 课程尚未连续两轮达到晋级要求。

|冻结评估目标|物理步长|严格成功 / 已测 / 计划|首次违规前转角中位数|瞬时角度峰值|目标保持峰值|
|---:|---:|---:|---:|---:|---:|
|30°|1 ms|0 / 20 / 20|21.25°|21.57°|0.00 s|
|30°|2 ms|0 / 20 / 20|21.25°|21.58°|0.00 s|

- [30° 冻结策略](checkpoints/challenge_30) · [逐初态结果](challenge_30/evaluation.json)
- [可续训检查点](../arms/v4/checkpoints/step_015213568_062844e2a7)
- [最佳样例录像](videos/best/best.mp4) · [受力与轨迹记录](videos/best/best_trajectory.json) · [仿真状态](videos/best/best_states.npz)。冻结评估 30°，2 ms，初态种子 8102011。
- [典型失败录像](videos/typical_failure/typical_failure.mp4) · [受力与轨迹记录](videos/typical_failure/typical_failure_trajectory.json) · [仿真状态](videos/typical_failure/typical_failure_states.npz)。冻结评估 30°，1 ms，初态种子 8102001。

封存初态逐项记录了三指全程受力和完整保持条件。
