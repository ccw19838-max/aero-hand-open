# 竖直右手的魔方抓握训练

第一阶段训练 **57 mm 魔方的稳定握持**，后续目标是整块魔方在手中转动。魔方各层锁定为一个刚体；当前成功判据只评价抓握，没有转动奖励，也不代表已经学会转动。

当前运行结果和完整基线比较见 [2026-11-11 实验记录](RESULTS_20261111.md)。首轮预抓握保持课程的 PPO 与固定目标基线均为 100/100，这个结果不能作为主动抓取成绩。

机械手竖直固定，手指朝上，重力为 `[0, 0, -9.81] m/s²`。魔方在每回合开始时预置于手指可接触的位置，因此本阶段学习的是预放置物体的抓握与保持，尚不包含从桌面接近、拾取物体。

## 模型与来源

使用 OpenAI [robogym 的 locked Rubik cube](https://github.com/openai/robogym/blob/9ff1111fd557b1a01911d7d8c9d59a453071c96e/robogym/assets/xmls/rubik/rubik_locked.xml)，固定版本 `9ff1111fd557b1a01911d7d8c9d59a453071c96e`。

| 参数 | 当前值及依据 |
|---|---|
| 边长 | 57 mm，上游模型原尺寸 |
| 质量 | 77.966253 g，上游 [locked.py:85](https://github.com/openai/robogym/blob/9ff1111fd557b1a01911d7d8c9d59a453071c96e/robogym/envs/dactyl/locked.py#L85) 设置密度 421 kg/m³ 后得到 |
| 主惯量 | 三轴均为 `4.22187259995e-5 kg·m²`，均匀立方体假设 |
| 碰撞与外观 | 单个 box，同时使用上游六面魔方贴图；没有视觉几何重复计质量 |
| 接触 | `condim=3`，滑动摩擦系数 0.6；这是本实验的显式设置 |
| 自由度 | 一个 freejoint，整块物体六维自由运动 |
| 手的安装 | 根节点位置 `[0, 0, 0.15] m`，四元数 `[0.70710678, 0, -0.70710678, 0]` |

原始 [XML](assets/robogym/rubik_locked.xml)、[贴图](assets/robogym/rubik.png)、[MIT 许可证](assets/robogym/LICENSE)和带 SHA-256 的[来源清单](assets/robogym/source_manifest.json)随代码保存。Copyright (c) 2020 OpenAI；复制或修改这些资产时保留许可证与来源。原 XML 的质量由上游运行代码覆写，因此不能用 XML 的默认密度推断标称质量。

手部复用仓库的 Aero 腱绳、关节和执行器模型。场景生成见 [model.py](model.py)，实验参数见 [config.py](config.py)。将来改变魔方尺寸、质量或接触参数时，应保存为新的实验配置。

## 任务与成功判据

使用 **原生 MuJoCo 3.3.6 + Gymnasium + Stable-Baselines3 PPO，全部在 CPU 上运行**。训练和评估共用 [env.py](env.py)，不经过 MJX 接触计算。

| 项目 | 默认值 |
|---|---|
| 物理步长 / 控制周期 | 2 ms / 50 ms，每个动作包含 25 个物理子步 |
| 动作 | 7 维，各执行器目标增量，默认每周期不超过有效量程的 ±1%；配置上限为 ±5%，最后再限幅 |
| 观测 | 65 维，详见下文 |
| 回合上限 | 5 秒 |
| 成功保持时间 | 连续 2 秒 |
| 有效手指接触 | 拇指及至少另外两根手指；每根手指累计法向接触力不小于 0.02 N |
| 稳定速度 | 魔方线速度 `<0.03 m/s`，角速度 `<0.5 rad/s` |
| 操作区域 | 魔方距本回合初始位置不超过 80 mm |
| 接触约束 | 允许掌面接触；禁止安装支架和地面支撑 |
| 初始化随机化 | 位置每轴 ±1 mm、随机轴姿态角 ±2°、质量与摩擦分别 ±10% |

稳定时间按物理子步累计，条件中断即清零。支架或地面支撑、掉落、数值异常会失败；穿透超过 1 mm 或离开操作区域持续 0.1 秒也会失败。有效初态同样要求穿透不超过 1 mm，并通过最多 32 次随机初态筛选。MuJoCo 使用柔性接触约束，承载时会产生小量数值穿透；准备报告和评估保留实际最大深度，超过阈值时不累计稳定时间。

65 维观测包括：16 个手关节位置、16 个速度、7 个归一化控制目标、7 个上一周期动作、魔方相对初始位置 3 维、四元数 4 维、线速度与角速度各 3 维、5 个手指接触标记及 1 个稳定时间比例。**这些包含仿真真值，不能直接删除后部署到实机**；实机需要对应的感知、状态估计或新的观测设计。

奖励鼓励有效多指接触、低速稳定保持及位置稳定，惩罚速度、动作变化和动作幅度。成功事件奖励 `+10`、失败事件 `-5`，只在终止时结算一次。评估使用上述完整成功判据，不能用奖励上升代替成功率。

## 安装与入口

当前云环境使用 Python 3.10 虚拟环境 `/workspace/.venvs/aero-cube`。以下命令从主仓库根目录执行。新机器先创建 Python 3.10 虚拟环境，再按锁定版本安装；`requirements.txt` 选择 PyTorch CPU wheel，避免安装 CUDA 依赖。

```bash
cd /workspace/aero-hand-open
python3.10 -m venv /workspace/.venvs/aero-cube
/workspace/.venvs/aero-cube/bin/python -m pip install --upgrade pip
/workspace/.venvs/aero-cube/bin/python -m pip install -r experiments/cube_grasp/requirements.txt
```

主要依赖是 `mujoco==3.3.6`、`torch==2.6.0+cpu`、`stable-baselines3==2.6.0`、`gymnasium==1.1.1`、NumPy、SciPy 和录像依赖，完整列表见 [requirements.txt](requirements.txt)。云环境已有该虚拟环境时可跳过创建；其他 Linux 机器可替换仓库和虚拟环境路径。

先执行有时间预算的参考姿态准备：

```bash
/workspace/.venvs/aero-cube/bin/python -m experiments.cube_grasp.prepare \
  --output experiments/cube_grasp/artifacts/prepare_v1 \
  --seconds 300 --seed 17
```

`prepare` 搜索手指可接触的初态，并进行无辅助物理验证。准备过程中的临时固定魔方只用于构造手姿；正式参考状态释放后没有固定约束、外加支持力或位置重置。参考姿态需通过 2 ms 和 1 ms 两种物理步长的无辅助握持检查，结果写入 `prepare_report.json`。只有报告批准后才生成 `reference.npz` 并允许训练；准备失败时应查看失败原因，不跳过检查。

开始第一阶段 PPO：

```bash
/workspace/.venvs/aero-cube/bin/python -m experiments.cube_grasp.train \
  --reference experiments/cube_grasp/artifacts/prepare_v1/reference.npz \
  --output experiments/cube_grasp/artifacts/train_v1 \
  --hours 2 --steps 500000 --envs 4 \
  --eval-every 8192 --eval-episodes 20 --baseline-episodes 20 --seed 0
```

这个命令最多使用 4 个 CPU 并行环境，以 2 小时和新增 500,000 环境步为上限；达到停止条件后保存检查点。初始化、基线和周期验证计入时间预算，退出还需要完成检查点落盘。训练不渲染。默认 Actor/Critic 各为两层 128 单元，rollout 为每环境 128 步、batch 128、每批更新 4 次、学习率 `3e-4`。

启动时分别评价零动作、固定闭合和随机动作基线；随后每 8,192 个环境步，用固定的 20 个验证种子评价策略。连续两次验证成功率达到 90% 会停止当前握持训练。**这里的 20 是评估回合数，不是训练轮数；验证达标也不自动开始转动训练。**

独立评价检查点并生成 30 秒真实时间回放：

```bash
/workspace/.venvs/aero-cube/bin/python -m experiments.cube_grasp.evaluate \
  --checkpoint experiments/cube_grasp/artifacts/train_v1/best \
  --output experiments/cube_grasp/artifacts/eval_v1 \
  --episodes 100 --seed-start 20000 --video 30
```

`best` 指已完成验证中成功率最高的检查点；尚未完成一次验证时使用 `latest`。独立测试从种子 20000 开始，与验证使用的 10000 起始种子分开。录像如果包含失败后重置，会明确标记，并在伴随 JSON 中记录重置次数；30 秒录像不等于连续成功握持了 30 秒。

续训示例：

```bash
/workspace/.venvs/aero-cube/bin/python -m experiments.cube_grasp.train \
  --reference experiments/cube_grasp/artifacts/train_v1/latest/reference.npz \
  --resume experiments/cube_grasp/artifacts/train_v1/latest \
  --output experiments/cube_grasp/artifacts/train_v2 \
  --hours 2 --steps 500000 --envs 4 --seed 1
```

检查点包含 `policy.zip`、`vecnormalize.pkl`、参考状态、准备报告和元数据，使用哈希绑定模型、观测和动作定义。续训恢复 PPO 策略、优化器和观测归一化状态；环境回合与随机序列重新开始，因此不是逐位一致的断点恢复。仅加载 `policy.zip` 而遗漏归一化状态不能保证相同推理结果。

## 从保持进入松脱后补抓

首轮保持参考已经夹住物体，因此另提供 [acquire_prepare.py](acquire_prepare.py) 来构造需要主动动作的课程：从合法参考中将无名指目标放松 `0.015 m`，让原始自由体无辅助演化 50 ms，保留真实位置和速度。它属于松脱后的补抓，仍不包含张开手从桌面拾取。

```bash
/workspace/.venvs/aero-cube/bin/python -m experiments.cube_grasp.acquire_prepare \
  --reference experiments/cube_grasp/artifacts/prepared_v2/reference.npz \
  --output experiments/cube_grasp/artifacts/acquire_prepared_v1 --episodes 20

/workspace/.venvs/aero-cube/bin/python -m experiments.cube_grasp.train \
  --reference experiments/cube_grasp/artifacts/acquire_prepared_v1/reference.npz \
  --config experiments/cube_grasp/artifacts/acquire_prepared_v1/config.json \
  --output experiments/cube_grasp/artifacts/train_acquire_v1 \
  --hours 1.9 --steps 500000 --envs 4 --eval-every 4096
```

派生课程必须先验证初态不稳定，并满足随机零动作成功率不高于 20%、限速闭合基线不低于 80%；同时确定性闭合要通过 2 ms/1 ms 检查。所有正式动作仍受每控制周期 1% 量程的限制。新参考保持原尺寸、质量、摩擦及成功判据。更换课程初态时本次从头训练新策略；`--resume` 仅用于参考、配置及观测动作定义完全相同的续训。

## 运行记录

| 文件 | 含义 |
|---|---|
| `artifacts/prepare_v1/prepare_report.json` | 初握可达性与 2 ms / 1 ms 数值检查，属于准备结果 |
| `artifacts/train_v1/status.json` | 当前状态、训练环境步数、时间预算、停止原因、最新检查点 |
| `artifacts/train_v1/baselines.json` | 零动作、固定闭合、随机动作的表现 |
| `artifacts/train_v1/progress.jsonl` | 每次验证的成功率、稳定时间及失败原因 |
| `artifacts/train_v1/checkpoints/` | 完整可恢复检查点；`latest` 和 `best` 为索引 |
| `artifacts/eval_v1/evaluation.json` | 独立测试成功率、各回合结果及失败统计 |
| `artifacts/eval_v1/trajectories.jsonl` | 评估轨迹 |
| `artifacts/eval_v1/replay.mp4` | 回放及同名 JSON 中的时间、帧率、重置说明 |

模型和候选参考握姿已完成初步数值检查；学习策略的进度与成绩以实际 `artifacts/` 运行记录为准。需要比较 PPO 与零动作、固定闭合基线：预置姿态本身若已能保持，高成功率不能单独证明策略学到了有价值的纠正行为。应同时看随机初态下的掉落、位移、连续稳定时间及动作变化，并保留失败轨迹。

本任务代码位于主仓库 `experiments/cube_grasp/`，在 `feature/new-manipulation` 分支本地开发，尚未推送。旧转方块基线和已归档的转笔实验独立保留。
