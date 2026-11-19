# 只训练 Z 轴正向 90°：激进奖励实验 v1

本实验接替已停止的 16 小时 XYZ 课程，保留所有旧实现和检查点。目标是整块魔方绕**固定掌面法向 Z** 正向转 90°，然后五指重抓、稳定在目标面。不是拧动魔方各层，也不是绕世界竖直轴转动；手模型原有约 20° 倾角保持不变。

## 成功标准

- 从初握真正建立时的姿态计算目标，不把初态误差或握持前滑动算作进度。
- 完整 SO(3) 姿态误差 ≤5°，有向净转角在 +90°±5° 内，不使用魔方对称等价。
- 拇指、食指、中指、无名指、小指均接触，单指法向力 ≥0.02 N。
- 线速度 <0.03 m/s、角速度 <0.5 rad/s，连续保持 **1 秒**。
- 不接触安装座或地面，不借助外力，不越出 80 mm 操作区域，不持续严重穿透。
- 单回合 16 秒。超时属于真正失败，不向失败之后 bootstrap 价值。

仅旋转途中允许部分手指离开；完全失去手指接触 0.15 秒仍判失败。

## 避免初握被新训练破坏

旧策略在转动训练中出现初握退化。实测固定原参考电机目标，验证种子 10000–10019 的 20 个初态均可在 0.50–0.55 秒建立原先的五指稳定初握。因此，新任务的 acquire 阶段保持原电机目标，建立初握后再交给策略。这个控制阶段切换在训练、独立评估、录像中一致执行。

这是一套“初始固定目标保持 + 学习转动/重抓”的控制器，不宣称所有阶段都由神经网络驱动。物体从 reset 起始终自由运动；没有刚性固定、外力、提高摩擦、改质量或回合中瞬移。requested/applied action 均记录在 info，历史动作使用实际 applied action。

## 冻结的奖励

令 `delta = (上一物理步姿态误差 − 当前误差) / 当前课程目标角度`：

- 有合法手指支撑时 `80 × delta`；反向进度无论是否合法都扣分。原权重为 5。
- 目标附近有 `4/s` 的窄范围姿态奖励；同时检查有向角，−270°/450°不能借同姿态获奖。
- 到位后低速五指稳定 `20/s`，每个重抓接触还有边际奖励。
- 对侧翻、目标速度不匹配、过大线速度、动作跳变、失去接触和耗时扣分。目标角速度随剩余角度下降，到位需要刹车。
- 初握只结算一次 `+2`。初握阶段没有按时长增加的正奖励。
- 真正成功 `+500`，失败 `−100`。成功奖励大于整回合能收集的到位/稳定稠密奖励，避免反复打断保持计时器刷分。

奖励每 2 ms 物理子步积分，控制周期 50 ms。正式训练仍使用七个电机目标的增量，每步限有效范围的 **±1%**，最终限幅不变。2% 增量只用于保留的先导试验：同三个初态，旧 Z5 actor 在固定初握、1%增量下最长稳定 0.448–0.512 秒，2%只有 0–0.028 秒，故恢复1%。该对比不是90°成功评估，原始结果见 artifacts/diagnostics/warm_start_controls.json。

## 学习与验收

CPU 四个环境全部训练 Z，不分配 X/Y。课程为 5→10→15→20→30→45→60→75→90°；每级误差容限 `min(5°, 目标角/5)`，不能原地过关。每 16384 个环境控制步，用固定 20 个初态验证；连续两次 ≥90% 才晋级。各级都要求连续稳定 1 秒，不是只有最后一级才要求抓稳。

从旧 Z5° 已通过的 checkpoint 初始化 actor 和观测统计；旧 critic 与奖励尺度不兼容，所以重建 critic、Adam、奖励统计和探索标准差。不是从失败的 X/Y 权重继续。后续课程迁移同样记录来源，重复恢复同一配置则保留优化器。

PPO：两层 128 单元 actor/critic；rollout 256×4、batch 256、6 epochs、学习率 2e−4、gamma 0.995、GAE 0.95、clip 0.2、entropy 0.01、初始 log_std −0.5。启用训练奖励归一化，评估只归一化观测。将 VecNormalize 的 clip_reward 从默认10提高到10000，避免把归一化后的大额完成奖励压平，从而破坏终局奖励相对累计稳定奖励的优势。该设置保存在每个新检查点 metadata 和归一化状态中。

90° 课程过关后，使用 100 个未参与训练/选模型的初态独立评估，成功率 ≥90%才标记完成。独立评估失败会保留报告、继续 Z90 训练；下一次使用新的种子块。通过后自动录制带失败/重置标记的 30 秒真实时间回放。

总预算 **24 小时**，包含训练、课程验证与独立评估；上限 1000 万额外环境控制步。每 4096 步保存，预算结束保存并停止。resume 沿用原始截止时间，不重新获得 24 小时。环境断开或进程被终止后，需要用 resume 恢复，不能仅凭状态文件的 running 字段认定仍在训练。

## 命令

在仓库根目录，Python 为 `/workspace/.venvs/aero-cube/bin/python`：

```bash
python -m unittest discover -s experiments/cube_z90/tests -v

python -m experiments.cube_z90.curriculum \
  --reference experiments/cube_envelop/artifacts/prepared_envelope_v1/reference.npz \
  --initial-checkpoint experiments/cube_envelop_gradual/artifacts/long16h_z_then_xy_v1/stage_000_z5/checkpoints/step_000229376_25a1a93b \
  --output experiments/cube_z90/artifacts/z90_aggressive_delta1_v1 \
  --hours 24 --steps 10000000 --envs 4 --action-delta 0.01

# 中断后原样加 --resume，保留原截止时间。

python -m experiments.cube_z90.evaluate \
  --checkpoint <已保存的检查点目录> --output <新的评估目录> \
  --episodes 100 --axis z --seed-start 90000000 --video 30
```

`status.json` 记录当前级别、已通过角度、真实 PID、截止时间和最新验证；`stage_*/progress.jsonl` 保留各次评估；检查点含 policy.zip、vecnormalize.pkl、参考状态及 SHA256 元数据。权重仍依赖仿真真值观测，不能直接部署为只有七路编码器输入的实机控制器。

增加奖励和算力投入不能保证硬件可达性或学习收敛。未达到完整 90°+稳定标准时，应报告具体失败，不能把小角度课程通过写成已学会翻面。

当前正式运行目录为 `artifacts/z90_aggressive_delta1_v1/`。先导试验 `artifacts/z90_aggressive_v1/` 已停止并保留，2%增量下首轮5°验证0/20，初握20/20；它不代表新训练已经学会转动。修正后的正式训练沿用首次启动的总截止时间 **2026-11-13 02:34:14 UTC / 北京时间10:34:14**，先导试验计入同一个24小时总预算。

## 完整训练预览

`artifacts/preview_20261112_1115_step73732/index.html` 是本次完整预览，可离线打开；视频、逐帧指标与曲线均内嵌。视频权重固定在73,732步，页面另标训练状态快照的实际步数，不把后续学习结果混入旧视频。

- `four_hands_30s.mp4`：四个独立初态同屏，共用冻结策略，30秒、20fps，标注失败与重置。
- `single_hand_30s.mp4`：同一轨迹中手1的完整视角，30秒、20fps。
- `states.npz`、`trajectory.jsonl`：真实物理状态与控制数据；渲染读取这些状态，不重新改变物体运动。
- `checkpoint_evaluation/evaluation.json`：这份权重额外20个初态的检查，目标仍是5°，不是90°验收。
- `preview_report.json`：训练进程、四个采样工作进程PID、课程验证、回放事件与额外检查。

四个训练环境在采样时并行运行，共享一套PPO网络；课程验证期间，训练器顺序评估20个初态。视频中的四个环境是额外创建的确定性评估回放，不是从实时训练工作进程截取的画面。

重新生成时，先锁定一个具体的不可变检查点目录：

```bash
python -m experiments.cube_z90.preview --checkpoint <检查点> --run <训练输出目录> --output <预览目录> --seconds 30
python -m experiments.cube_z90.evaluate --checkpoint <同一检查点> --output <预览目录>/checkpoint_evaluation --episodes 20 --axis z --seed-start 90000000
python -m experiments.cube_z90.build_preview <预览目录>
```
