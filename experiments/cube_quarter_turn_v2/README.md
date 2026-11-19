# XYZ 90° 教师策略 v2

旧版在约 52 万步仍没有完整的 90° + 1 秒停稳成功。具体诊断见
[原始分析](../cube_quarter_turn/IMPROVEMENT_ANALYSIS_20261112.md)。v2 是独立实验，旧版模型、代码和检查点保持可回放。

## 改了什么

- 七个电机从旧版的局部目标区间，改为覆盖各自完整有效行程。动作 0 对应所选初握，−1/+1 对应合法上下限；不是增加电机的物理行程或自由度。
- Actor 和 Critic 都读取同一份 133 维观测，包括所有被动关节和速度、指尖位置、接触力、实际控制目标、目标姿态和阶段。动作历史明确为上一控制周期实际执行的动作 `a(t−1)`。训练、规划、回放共用 `TeacherEnv`。
- 2.5 秒内没有至少 2° 的新误差改善，且没有积累终点稳定时间，就判定停滞。失败终值从 −100 改为 −10；时间成本从每秒 −0.2 改为 −1；手指总法向力超过物重 8 倍的部分，每秒扣 `0.02 × 超出倍数`。接触力总和不是合力或抓握裕量。
- X/Y/Z 分开筛选初握和训练教师，避免一个策略在早期混学相互冲突的动作。筛选只改变电机控制，不在过程里移动物体、施加辅助力或修改摩擦。
- 先用分段动作的 CEM 搜索合法完整轨迹，再进行行为克隆和 PPO。只有在 **2 ms、1 ms 两次独立重放中都完成完整 90° 并停稳** 的轨迹才能进入示范集。失败的大角度片段保留为诊断。
- 行为克隆做 24 个魔方颜色朝向的等价增强。PPO 每次更新后保留两批示范学习，减少遗忘。

## 没有降低的标准

仍是 57 mm / 77.966253 g / 摩擦 0.6 的刚性魔方、掌面原有 20° 倾角、世界重力、2 ms 物理步长。魔方层仍锁定。

成功仍同时要求完整三维姿态误差 ≤5°、净有向转角 +90°±5°、拇指和至少两根其他手指有效接触、线速度 <0.03 m/s、角速度 <0.5 rad/s，连续保持 1 秒。地面和安装支架支撑、持续穿透、越界和脱离手部的失败门槛沿用 v1。没有用 5°、30° 或 60° 作为成功标准。

## 执行流程与预算

每轴先检查最多 192 个电机目标候选，前 0.25 秒正常保持原参考状态，再渐变到候选目标。候选须在末尾连续 0.5 秒满足三指稳定条件；随后从该候选参考状态独立验证 3 秒自由保持，2 ms 和 1 ms 都通过、位移和姿态误差差异在阈值内，才写出批准的参考文件。原始参考也会参加筛选；若最后仍为五指紧握，报告实际接触和间隙，不称为已经优化出换指姿态。

每轴初握筛选上限 12 分钟、轨迹搜索上限 90 分钟。搜索在每个控制周期和每代保存进展。没有合法初握或合法完整示范，该轴会明确停止并保留最佳失败轨迹，不自动跳过门槛进入 PPO。默认一代 48 个候选，五段控制，保留全局随机候选防止局部收敛。

获得合格示范后：最多 3000 次行为克隆更新，再进行 PPO（512/256/128 SiLU、rollout 128、batch 128、4 epochs、学习率 2e−4、gamma 0.99、GAE 0.95、clipping 0.2、entropy 0.001）。X/Z 各 1 个仿真工作进程，Y 为 2 个；搜索阶段每轴 1 个进程，共 3 个。CPU 物理工作进程合计不超过 4；不渲染训练。

每 16384 步用固定 20 个验证初态评估，两次达到 90% 后，用另 100 个初态独立评估。只有三个轴都通过，才自动进行 100 条不重置物理状态的 X→Y→Z 串联检查，串联也 ≥90% 才写 `final_task_verified=true`。独立成功不代表可以串联。

所有阶段共用原始截止时间 **2026-11-13 10:34:14 北京时间**，不重新计算 24 小时。筛选、搜索和评估也计入预算。到时停止并保存；可以早于截止时间因为轨迹门槛失败而停止。

## 命令

在仓库根目录，已安装环境为 `/workspace/.venvs/aero-cube`：

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLCONFIGDIR=/tmp/aero-cube-matplotlib
export XDG_CACHE_HOME=/tmp/aero-cube-cache

/workspace/.venvs/aero-cube/bin/python -m experiments.cube_quarter_turn_v2.pipeline \
  --reference experiments/cube_envelop/artifacts/prepared_envelope_v1/reference.npz \
  --source-checkpoint experiments/cube_official_baseline/artifacts/official_warmstart_v1/checkpoints/step_001441792_47e53726 \
  --output experiments/cube_quarter_turn_v2/artifacts/xyz90_v2 \
  --deadline-at 2026-11-13T02:34:14.079538+00:00
```

`pipeline` 管理三个轴和截止时间。可加 `--axis x` 只运行一轴，或缩小 `--candidates` / `--plan-minutes` 做冒烟检查；正式实验参数变更必须使用新输出目录。不要向同一目录重复启动。

查看 `artifacts/xyz90_v2/status.json`；每轴另有 `prepare/report.json`、`planner/status.json`、`planner/report.json` 和 `teacher/status.json`。`planner/best_candidate.json` 中的大角度不等于完整成功。`planner/demo_*.npz` 只有通过双步长回放才会生成。

获得教师检查点后独立评估、导出逐控制周期轨迹和 30 秒回放：

```bash
MUJOCO_GL=egl /workspace/.venvs/aero-cube/bin/python -m experiments.cube_quarter_turn_v2.evaluate \
  --checkpoint experiments/cube_quarter_turn_v2/artifacts/xyz90_v2/z/teacher/best \
  --output experiments/cube_quarter_turn_v2/artifacts/z_evaluation \
  --episodes 100 --video
```

检查点包含策略、观测归一化缓冲区、奖励归一化状态、参考状态和代码/文件校验。`train_axis(..., resume=checkpoint)` 恢复 SB3 策略、优化器和奖励统计，示范由校验后的独立文件重新加载；不恢复仿真向量环境当前状态/RNG，不能声称逐位一致续训。源作者权重仅作为特征初值，v2 动作和观测语义已有变化，使用新优化器和值函数输出头；不是作者 Brax 训练的原样续训。

## 验证

```bash
/workspace/.venvs/aero-cube/bin/python -m unittest \
  experiments.cube_quarter_turn_v2.tests.test_v2 \
  experiments.cube_quarter_turn.tests.test_quarter_turn \
  experiments.cube_quarter_turn.tests.test_action_mapping_candidate -v
```

测试包括完整行程、133 维真实输入、上一动作一致性、失败与停滞门槛、完整 90°+1 秒、反例、双步长初握、失败示范拒收、截止时间中断、颜色等价增强、策略存取一致性以及 PPO 冒烟更新。优化器测试中的合成标签只用于测试，不生成正式示范。代码验收通过不表示已经学会转动。

这是一版改进实验。开环搜索可能仍找不到合法轨迹；初始几何、欠驱动耦合和中途换指空间仍是主要不确定因素。只有数据通过上述门槛，才会声明某轴完成。
