# X/Y 闭环 PPO：转动时换指，抓稳后五指受力

用户已确认：转动过程中允许短暂换指，抓稳后要求食指、中指、无名指、小指和拇指全部有效受力。X/Y 各两个并行仿真进程，Z 不训练。

本任务独立保存，旧轨迹、录像和策略语义不修改。研究依据与原文链接见 [RESEARCH.md](RESEARCH.md)。

## 本次变化

- 从第一批采样直接进行 PPO，取消“先搜到完整成功示范才训练”的 CEM 门槛。旧实验主要是开环轨迹搜索；本次是实际更新神经网络参数的闭环训练，不声称延续一个已经存在的 CEM 神经策略。
- 每 50 ms 输出七维动作，电机目标在当前目标上最多增减完整有效量程的 2%，最终限幅。使用真实每指接触力和全部被动关节反馈，输入共 138 维；上一控制周期实际动作为 `a(t−1)`。
- 抓稳必须有拇指 ≥0.2 N，其他四指各 ≥0.1 N。目标附近连续一秒期间任何一指不足，计时归零。拇指 >20 N 或其他指 >12 N 时不算合格稳握；过大的力另有惩罚。
- 转动时允许换指，拇指与至少另外两指不足支撑的连续时间上限为 0.15 秒。没有要求五指全程不离开魔方。
- 姿态误差差分奖励、五指覆盖奖励、接近目标奖励、保持时间势函数、动作平滑和过大接触力惩罚一起使用。接触奖励饱和，不能靠无限挤压刷分；一秒保持势函数在失稳时扣回。
- 先验证五指初握，之后按 0°/30°/60°/90° 课程推进，每阶段在 1 ms、2 ms 两组各至少 90% 且连续两次验证通过才晋级。小角度结果不会写成完成 90°。
- 同轴两个仿真进程分别使用 1 ms 和 2 ms 物理步长，直接暴露此前两种步长结果不一致的问题。最终仍须在两种步长独立评估通过。
- 名义 90°任务通过独立评估后，再加入 ±5%、±10% 摩擦与关节阻尼随机化。尺寸、名义质量、重力、20°倾角、5°姿态容差、穿透和安装支架失败条件保持原值。

原始五指参考为 `cube_envelop/artifacts/prepared_envelope_v1/reference.npz`，已验证原生 1 ms 和 2 ms 无外力保持。此次中性电机控制再次通过五指一秒握持检查。

## 策略与奖励

Actor 和 Critic 各为 512/256/128 SiLU MLP。保留作者策略转换后的隐藏特征和输入归一化，**重新初始化动作输出头、Value 输出头和 PPO 优化器**，避免把原绝对位置动作当增量不断积分。当前有完整仿真状态，尚未采用 OpenAI 2019 的 LSTM 或完整 ADR，不宣称是其训练系统的原样复现。

PPO：两环境 ×128 rollout、batch128、4 epochs、学习率3e−4、gamma0.99、GAE0.95、clipping0.2、entropy0.005。训练不渲染，不需要 GPU。

物理检查每个子步执行支架/地面支持判断；每 10 ms 读取真实接触力、姿态和速度并更新保持/失败时间。每个控制周期记录各手指力的最小值和最大值，评估报告给出每指零接触比例和五指接触占比。接触力来自魔方与各手指的原生 MuJoCo 接触力汇总，不给显示值添加偏置，也不伪造接触。

奖励配置在 `task.py`。姿态进展项为 `20 × Δ(目标误差)/(π/2)`，正向进展只在合法区域且有效支撑时给分；保持势函数为 `20 × Δ(合格保持时间)/1s`；成功终值+50、失败−20。覆盖项按接触下限饱和；目标接近奖励乘覆盖率平方；超力惩罚按每指上限归一化。不把只靠三指在目标附近静止算成功。

## 预算、保存和验收

用户本次“继续训练”启动独立 **24 小时**预算，从新任务启动时计时；不篡改旧实验已经到期的截止时间。每5分钟保存策略、优化器、输入/奖励归一化状态、参考和课程状态；每16384步或20分钟（先到者）验证。环境中断后可从检查点恢复，不能声称后台进程一定能跨环境休眠持续运行。

课程验证各步长20个初态；每三次课程验证额外做一次90°探测。90°阶段达到两次验证通过后，各步长另用100个初态独立评估。随机化最大档也通过独立评估，并重新通过名义模型评估，才写该轴 `final_task_verified=true`。两个轴都通过才写总览 `xy_verified=true`。本轮没有验证完整魔方求解或连续 X→Y 组合。

启动（仓库根目录）：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLCONFIGDIR=/tmp/aero-cube-matplotlib \
/workspace/.venvs/aero-cube/bin/python -m experiments.cube_contact_ppo.pipeline \
  --reference experiments/cube_envelop/artifacts/prepared_envelope_v1/reference.npz \
  --source-checkpoint experiments/cube_official_baseline/artifacts/official_warmstart_v1/checkpoints/step_001441792_47e53726 \
  --previous-run experiments/cube_quarter_turn_xy/artifacts/xy_two_workers_v1 \
  --output experiments/cube_contact_ppo/artifacts/contact_ppo_v1 --hours 24
```

恢复同一任务添加 `--resume`；只在确认任务已停止后执行。恢复保留本轮截止时间，不重置24小时。会恢复 SB3 优化器和归一化状态，但不恢复环境在内存中的物理状态及所有进程 RNG，不声称逐位一致。

总览：输出目录 `status.json`。轴状态：`x/status.json`、`y/status.json`。`training_steps` 是本次 PPO 环境步数，`policy_updates` 是实际优化次数；不能和旧 CEM 的候选轨迹数直接相加。

独立检查（默认固定90°）：

```bash
/workspace/.venvs/aero-cube/bin/python -m experiments.cube_contact_ppo.evaluate \
  --checkpoint experiments/cube_contact_ppo/artifacts/contact_ppo_v1/x/best \
  --output experiments/cube_contact_ppo/artifacts/x_evaluation --episodes 100
```

运行测试：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
/workspace/.venvs/aero-cube/bin/python -m unittest experiments.cube_contact_ppo.tests.test_contact -v
```

测试覆盖三指漏洞、五指计时、换指宽限、过大力、错误转角/额外整圈、原物理失败标准、双步长真实初握、增量限幅、课程切换、真实双进程 PPO 更新和检查点推理一致性。实现通过检查不表示已经学会翻面。
