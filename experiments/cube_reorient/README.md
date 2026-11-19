# 掌心朝上、保留原始倾角的魔方 XYZ 单次转向

本实验让右手掌心朝上，按用户要求**保留模型原有约 20° 倾角**。每个回合将整块魔方相对初始姿态绕指定的固定手坐标 X、Y 或 Z 轴转到 **+90° 目标姿态**，然后重新抓稳。X/Y 任务对应翻面，Z 任务对应绕掌面法向转向；魔方各层保持锁定，不进行拧层或解魔方。

当前已启动的运行是 `artifacts/train_xyz90_tilt20_v1`：4 个 CPU 环境，最多 2 小时或 500,000 环境步。本次每 32,768 步验证每轴 20 回合，以减少 CPU 时间被验证占用；入口默认值仍为 8,192 步。训练前每轴运行 3 回合零动作和随机动作对照。实时进度见 [status.json](artifacts/train_xyz90_tilt20_v1/status.json)，检查点见 [latest](artifacts/train_xyz90_tilt20_v1/latest/)，[模型预览](artifacts/prepared_v3/preview.png)。

这是独立的 `cube_reorient` 任务，从头训练带目标输入的 PPO。原竖直抓握任务 `cube_grasp` 的源码、配置及检查点保留。前轮竖直抓握已训练到 **20,480 环境步**，其最后一次固定验证为 **20/20**；这是旧任务的验证结果，不能当作新任务的翻转成绩。可核对旧[运行状态](../cube_grasp/artifacts/train_acquire_v1/status.json)和[验证记录](../cube_grasp/artifacts/train_acquire_v1/validation/step_000020480/evaluation.json)。

## 模型、安装与坐标

继续使用 OpenAI robogym 的 locked Rubik cube：边长 **57 mm**、质量 **77.966253 g**，单个自由刚体、box 碰撞和六面贴图。资产与原抓握任务共享，见[原始模型](../cube_grasp/assets/robogym/rubik_locked.xml)、[来源清单](../cube_grasp/assets/robogym/source_manifest.json)及 [MIT 许可证](../cube_grasp/assets/robogym/LICENSE)。来源固定于 `openai/robogym` 的 `9ff1111fd557b1a01911d7d8c9d59a453071c96e`；质量由上游设置的密度 `421 kg/m³` 得到。

原手模型内部存在约 20° 的掌面倾角，当前按用户指定保留。[model.py](model.py)将安装节点四元数设为 `[1, 0, 0, 0]`（`w,x,y,z`），根节点位置仍为 `[0, 0, 0.15] m`，世界重力仍为 `[0, 0, -9.81] m/s²`。掌面朝上但不严格水平，掌面法向也不等同于世界 `+Z`。

| 固定手轴 | 含义与近似世界方向 | 画面颜色 | 本回合目标 |
|---|---|---|---|
| X | 从腕部指向指尖，`[0.9397242, 0, -0.3419335]` | 红 | 绕手 X 正向转到 +90°，翻面 |
| Y | 从食指侧指向小指侧，`[0, 1, 0]` | 绿 | 绕手 Y 正向转到 +90°，翻面 |
| Z | 掌面向外的法向，`[0.3419335, 0, 0.9397242]` | 蓝 | 绕手 Z 正向转到 +90°，转向 |

正向遵循右手定则。三个轴由固定手的安装姿态定义，魔方运动时不会跟随魔方旋转；它们并非世界 XYZ。目标四元数按本回合**实际随机初始化姿态**计算：

```text
q_goal = R_fixed_hand_axis_in_world(+90°) × q_reset
```

先将选定手轴表示为上述世界方向，再通过左乘构造目标。成功判定比较完整三维姿态，不将立方体的几何对称姿态或不同颜色面视为同一姿态；仅四元数 `q` 与 `-q` 表示的同一姿态等价。有向累计转角用于诊断，成功仍需完整目标姿态及抓稳条件。当前定义评价目标姿态到达，不要求中间动作沿理想的纯单轴轨迹。

## 环境与成功判据

使用原生 MuJoCo 3.3.6、Gymnasium、Stable-Baselines3 PPO，训练与评估都运行于 CPU。各入口共用 [env.py](env.py) 的观测、动作和物理子步判据。

| 项目 | 默认设置 |
|---|---|
| 物理步长 / 控制周期 | 2 ms / 50 ms，即每动作 25 个物理子步 |
| 动作 | 7 维执行器目标增量，每周期最大为有效量程的 ±1%；最终目标再次限幅 |
| 观测 | 75 维：原抓握的 65 维状态，加目标四元数 4 维、姿态误差旋转向量 3 维、目标轴 one-hot 3 维 |
| 目标选择 | 训练时每回合抽取 X/Y/Z 之一；评估时固定轴分别测试 |
| 回合上限 | 10 秒 |
| 姿态要求 | 与完整目标姿态的夹角误差不超过 10° |
| 抓稳接触 | 拇指及至少另外两根手指，每指累计法向接触力至少 0.02 N |
| 抓稳速度 | 线速度 `<0.03 m/s`，角速度 `<0.5 rad/s` |
| 成功保持 | 姿态、接触和低速条件同时连续满足 0.5 秒 |
| 允许支撑 | 掌面接触允许；安装支架和地面支撑禁止 |
| 初始化扰动 | 位置每轴 ±1 mm、随机轴姿态角 ±2°、质量和摩擦分别 ±10% |

稳定时间按物理子步累计，条件中断就清零。魔方离开初始操作区域超过 80 mm、穿透超过 1 mm 持续 0.1 秒，以及掉落、禁止支撑或数值异常，都按对应原因失败。奖励鼓励完整姿态误差下降、接近目标和目标处稳定保持，并约束物体速度、动作幅度与动作变化；仅在手中保持不动不能满足转动成功条件。

初始化仍然是**物体预置在手指可接触的位置**，不包含桌面拾取。观测包含关节状态、魔方位姿和速度、接触标记等仿真真值。当前代码和权重不能直接视为实机可部署策略，实机还需要对应状态估计与验证。

## 准备参考握姿

使用已有 `/workspace/.venvs/aero-cube` 环境，依赖与[抓握任务 requirements.txt](../cube_grasp/requirements.txt)相同。需要补装时执行：

```bash
cd /workspace/aero-hand-open
/workspace/.venvs/aero-cube/bin/python -m pip install -r experiments/cube_grasp/requirements.txt
```

从已批准的竖直参考状态构造保留原始倾角的掌心朝上初态：

```bash
/workspace/.venvs/aero-cube/bin/python -m experiments.cube_reorient.prepare \
  --source experiments/cube_grasp/artifacts/acquire_prepared_v1/reference.npz \
  --output experiments/cube_reorient/artifacts/prepared_v3 \
  --seconds 300
```

`--source` 要求同目录的 `prepare_report.json` 已批准该参考状态。程序把物体与握姿一起刚性变换到新的安装方向，清零初始化速度，然后在 **2 ms 与 1 ms** 两种物理步长下分别执行 **3 秒无辅助自由运动验证**。正式魔方保持自由刚体，没有外加支持力、固定约束或运行中的位置重置。结果写入 `prepare_report.json`，只有检查通过才保存 `reference.npz`。

没有旧参考文件时可省略 `--source`，程序先用固定种子 17 重新构造竖直握姿，再完成同样的安装姿态转换及验证：

```bash
/workspace/.venvs/aero-cube/bin/python -m experiments.cube_reorient.prepare \
  --output experiments/cube_reorient/artifacts/prepare_rebuilt_v1 \
  --seconds 300
```

每次使用新的输出目录，避免覆盖已批准参考状态。`--seconds` 是需要重建初始握姿时的搜索预算；后续物理验证还需运行。**准备通过只证明该掌心朝上初态能握持，不证明任何轴的 +90° 转动已经成功。** 旧的水平补偿参考状态属于不同安装配置，当前应使用保留原始倾角后重新批准的参考文件。

## 从头训练与续训

首次 CPU 训练示例：

```bash
/workspace/.venvs/aero-cube/bin/python -m experiments.cube_reorient.train \
  --reference experiments/cube_reorient/artifacts/prepared_v3/reference.npz \
  --output experiments/cube_reorient/artifacts/train_v1 \
  --hours 2 --steps 500000 --envs 4 \
  --eval-every 8192 --eval-episodes 20 --baseline-episodes 5 --seed 0
```

这是最多 2 小时、额外 500,000 环境步、4 个 CPU 并行环境的运行预算。初始化、基线和周期验证计入时间预算；停止后还需完成检查点落盘。训练不渲染。Actor/Critic 各两层 128 单元，rollout 为每环境 128 步，batch 128，每批更新 4 次，学习率 `3e-4`。

训练前比较零动作和随机动作基线。每 8,192 环境步分别对 **X/Y/Z 各 20 回合**进行固定验证，一次完整验证共 60 回合。**只有三个轴各自连续两次完整验证成功率都达到 90%，才按达标条件停止。** 不能用三个轴的平均成功率掩盖某一轴未学会；20 回合也是评估数量，不是训练轮数。

`best` 检查点先比较三个轴中最低的成功率，再比较平均成功率和最终姿态误差。未完成一次完整验证时可能还没有 `best`，此时可以评价 `latest`。

新任务观测和目标定义已改变，不能直接用旧 `cube_grasp` 检查点续训。相同 `cube_reorient` 任务的续训示例为：

```bash
/workspace/.venvs/aero-cube/bin/python -m experiments.cube_reorient.train \
  --reference experiments/cube_reorient/artifacts/train_v1/latest/reference.npz \
  --resume experiments/cube_reorient/artifacts/train_v1/latest \
  --output experiments/cube_reorient/artifacts/train_v2 \
  --hours 2 --steps 500000 --envs 4 --seed 1
```

完整检查点包含策略与优化器、`VecNormalize` 统计、参考状态、准备报告和配置元数据，使用哈希检查任务定义和文件一致性。恢复时重新开始环境回合和随机序列，因此不是逐位一致的断点恢复。

## 独立评估与回放

对三个轴各使用 100 个未用于选模型的测试种子，并录制 30 秒回放：

```bash
/workspace/.venvs/aero-cube/bin/python -m experiments.cube_reorient.evaluate \
  --checkpoint experiments/cube_reorient/artifacts/train_v1/best \
  --output experiments/cube_reorient/artifacts/eval_v1 \
  --axis all --episodes 100 --seed-start 20000 --video 30
```

`--episodes 100` 表示**每轴 100 回合，总计 300 回合**。也可用 `--axis x`、`--axis y` 或 `--axis z` 单独检查。每轴分别报告完整成功率、最终及最佳姿态误差、掉落率、目标处稳定时间、动作变化和失败原因。

`--axis all --video 30` 按 X、Y、Z 分为各 10 秒，画面标注目标轴、+90° 目标、当前误差、回合及状态，场景显示红/绿/蓝固定轴。包含失败重置时会标记，并将重置时间和原因写入视频旁的 JSON。这是三个轴的分段回放，不代表同一回合连续完成 XYZ 三次翻转，也不代表连续成功保持了 30 秒。

| 记录位置 | 用途 |
|---|---|
| `artifacts/prepared_v3/prepare_report.json` | 保留原始倾角的安装配置、参考状态来源及两种步长的握持验证 |
| `artifacts/train_v1/status.json` | 当前状态、环境步数、耗时、停止原因及最新检查点 |
| `artifacts/train_v1/run_config.json` | 本次预算、命令参数与任务配置 |
| `artifacts/train_v1/baselines.json` | 各基线的分轴结果 |
| `artifacts/train_v1/progress.jsonl` | 分轴验证成功率、姿态误差、掉落率与连续达标次数 |
| `artifacts/train_v1/checkpoints/` | 完整检查点，`latest` 和 `best` 是索引 |
| `artifacts/eval_v1/evaluation.json` | 汇总报告及独立的 X/Y/Z 结果 |
| `artifacts/eval_v1/{x,y,z}/` | 各轴评估报告与 `trajectories.jsonl` |
| `artifacts/eval_v1/replay.mp4` | 回放；同名 JSON 记录真实时间、分段和重置 |

转动成绩以实际 `artifacts/` 运行记录为准。候选握姿、模型可运行、奖励提升和短暂碰到目标姿态，都不能单独表述为转动任务成功；需要满足完整判据并分别通过各轴评估。

代码在 `feature/new-manipulation` 分支本地开发，未推送、未发布。
