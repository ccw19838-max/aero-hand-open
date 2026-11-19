# X/Y 各两个 CPU 仿真进程

根据用户最新指示，暂停 Z，把四个并行物理工作进程全部分给 X/Y：**X=2，Y=2，Z=0**。

复用冻结的 `cube_quarter_turn_v2` 模型、133 维观测、完整电机行程、奖励和成功/失败条件。旧 XYZ 任务已收到 SIGTERM 并保存。新任务直接使用旧任务每轴批准的三个初握，以及实际回放能精确复现的最高评分候选；不重跑初握，也不丢弃当前最佳动作。

旧搜索没有保存 CEM 分布和随机数状态，因此迁移属于以最佳动作初始化的新搜索，不声称逐位一致恢复旧 CEM。新的 `planner/search_state.npz` 原子保存分布、随机数状态、最佳参数、代数、尝试计数、参考哈希及截止时间，可用 `--resume` 恢复。恢复不得变更参考、实现或截止时间。

每轴一个 CEM 控制进程加两个原生 MuJoCo 工作进程。每代 48 个候选，由两只手并行尝试，同轴共享搜索分布。始终只提交两个在途任务，停止时不会继续排空整代队列。持续 30 代没有新最佳结果时，扩大采样分布以继续探索；物理参数、评分、90°目标和一秒保持门槛均不变。

一条完整轨迹必须在 2 ms 和 1 ms 独立回放中都通过 90°±5°、完整姿态误差≤5°、拇指+另外两指接触、低线/角速度及连续保持一秒，才会成为示范。找到第一条合格示范后，退出该轴的搜索进程池，进入既有行为克隆及 PPO 流程，**该轴 PPO 同样固定两个并行环境**。失败的高角度轨迹不会作为完整示范。

取消原来“搜索 90 分钟仍没有完整示范就停止”的局部上限，搜索及后续训练共用剩余全局预算：**北京时间 2026-11-13 10:34:14**。如成功门槛始终未满足，会在截止时保存搜索状态和失败结果；不能把持续搜索称为已经开始 PPO。X/Y 的独立通过记录为 `xy_verified`，不会自动声明 XYZ 或魔方复原任务成功。Z 的历史结果及视频保留，但没有 Z 工作进程。

启动命令（仓库根目录）：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/aero-cube-matplotlib \
/workspace/.venvs/aero-cube/bin/python -m experiments.cube_quarter_turn_xy.pipeline \
  --source-run experiments/cube_quarter_turn_v2/artifacts/xyz90_v2 \
  --source-checkpoint experiments/cube_official_baseline/artifacts/official_warmstart_v1/checkpoints/step_001441792_47e53726 \
  --output experiments/cube_quarter_turn_xy/artifacts/xy_two_workers_v1 \
  --deadline-at 2026-11-13T02:34:14.079538+00:00
```

同一输出目录不得重复启动；任务确已停止后可添加 `--resume` 续跑。总览为输出目录的 `status.json`，每轴实际仿真 PID、搜索计数及状态见 `x/planner/status.json` 和 `y/planner/status.json`；之后的教师训练状态见各自 `teacher/status.json`。

验证命令：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/workspace/.venvs/aero-cube/bin/python -m unittest \
  experiments.cube_quarter_turn_xy.tests.test_parallel -v
```

验证包含：两个真实仿真工作进程与单进程的结果一致、停止后最多两个在途任务、分布和随机数恢复、拒绝变更截止时间，以及两个真实并行环境的 PPO/奖励归一化冒烟更新。PPO 测试仅验证代码路径，不生成正式训练示范。
