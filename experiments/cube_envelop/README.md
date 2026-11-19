# 四指包络与拇指支撑的魔方抓握、转向

当前运行：[curriculum_envelope_v1/status.json](artifacts/curriculum_envelope_v1/status.json)。实际课程先训练五指稳握（`hold`），每 4,096 步用 20 回合验证，连续两次达到 90% 后，再通过种子 40000 的独立 100 回合检查。该检查成功率也达到 90%，才自动进入 XYZ 翻面。两阶段共享最多 2 小时、500,000 环境步；握持阶段上限 30 分钟，未达标时保留权重与失败记录，不假称已进入转动训练。

跨阶段通过 `--init-checkpoint` 迁移 78 维策略权重及观测归一化，重建 PPO 优化器，转动阶段步数重新从零统计；父运行状态累计总步数。`--resume` 仍只用于相同模式、相同参考的续训。新初握见[预览](artifacts/prepared_envelope_v1/preview.png)。

首轮记录：独立的固定控制基线在验证种子 10000–10019 上能完成五指初握 20/20，但额外连续稳握 2 秒仅 5/20。新 PPO 在 12,288 步时，同一固定验证集达到初握 20/20、稳握 19/20（95%）；这只是首次达标验证，不代表独立 100 回合已通过，也不是翻面成绩。实时后续结果以运行记录为准。新代码、门槛、保存加载、跨模式初始化与课程预算合计 31 项自动测试通过。

```bash
/workspace/.venvs/aero-cube/bin/python -m experiments.cube_envelop.curriculum \
  --reference experiments/cube_envelop/artifacts/prepared_envelope_v1/reference.npz \
  --config experiments/cube_envelop/artifacts/prepared_envelope_v1/config.json \
  --output experiments/cube_envelop/artifacts/curriculum_envelope_v1 \
  --hours 2 --steps 500000 --envs 4
```

本任务先建立**食指、中指、无名指、小指包络魔方，拇指参与支撑的五指握持**，稳定后再完成固定手坐标 X/Y/Z 中指定一轴的单次 +90° 转向，最后恢复五指稳定接触。它是独立的 `cube_envelop` 实验，从头训练，旧 `cube_reorient` 的代码、权重及结果保留作对照。

掌心朝上并保留原模型约 **20° 倾角**，安装节点四元数为 `[1, 0, 0, 0]`，不做水平补偿。使用同一块 **57 mm、77.966253 g** 的刚体魔方；世界重力、原手的关节、腱绳和执行器模型保持一致。魔方各层锁定，本任务不拧层、不解魔方。

## 模型来源与轴定义

资产来自 OpenAI [robogym locked Rubik cube](https://github.com/openai/robogym/blob/9ff1111fd557b1a01911d7d8c9d59a453071c96e/robogym/assets/xmls/rubik/rubik_locked.xml)，固定版本 `9ff1111fd557b1a01911d7d8c9d59a453071c96e`，MIT 许可，Copyright (c) 2020 OpenAI。复用仓库内的[原始模型](../cube_grasp/assets/robogym/rubik_locked.xml)、[贴图](../cube_grasp/assets/robogym/rubik.png)、[来源与校验清单](../cube_grasp/assets/robogym/source_manifest.json)和[许可证](../cube_grasp/assets/robogym/LICENSE)。质量来自上游设置的密度 `421 kg/m³`，碰撞使用单个 box。

固定手 X 轴从腕部指向指尖，Y 轴从食指侧指向小指侧，Z 轴为掌面外法向；画面分别用红、绿、蓝表示。三轴随安装姿态固定，因保留倾角而不等同于世界 XYZ，也不会随魔方运动。正方向遵循右手定则，X/Y 目标是翻面，Z 目标是绕掌面法向转向。

## 课程与成功定义

默认 `task_mode='reorient'`，一个回合包含以下阶段：

| 阶段 | 要求与转换条件 |
|---|---|
| `acquire`：建立包络握持 | 五根手指各自有效接触，线速度 `<0.03 m/s`、角速度 `<0.5 rad/s`，连续 0.5 秒；2 秒内未建立则失败 |
| `rotate`：转向 | 在建立握持的时刻重新定义 +90° 目标，允许个别手指离开、适度松开和换指；鼓励拇指参与支撑 |
| `recapture`：重新抓稳 | 完整姿态误差不超过 10°，有向净进度位于 +90°±10°，恢复五指接触及低速状态并连续保持 0.5 秒，才算完整成功 |

每根手指的有效接触按其接触点累计法向力 **≥0.02 N** 计算，可由指节或指尖接触建立。握持要求所有五指参与，不能用两三根手指的较大接触力代替缺失的手指。

目标在首次成功建立包络握持时冻结：

```text
q_goal = R_fixed_hand_axis_in_world(+90°) × q_at_capture
```

初始化后的自然沉降和 `acquire` 阶段不计旋转进度。目标比较完整三维姿态，不将魔方的不同颜色面当作几何对称等价面；`q` 与 `-q` 仍表示同一姿态。净进度窗口同时排除转了 +450° 或 −270° 后恰好到达同一终点姿态的情况。

可选 `task_mode='hold'` 用于先训练五指握持：建立 0.5 秒包络握持后，还须**额外连续保持 2 秒**才成功。此模式没有旋转目标，不计旋转进度，评估中的 `recapture_rate` 不适用；握持达标不能写成翻转成功。默认转向模式本身也包含先握稳再转的阶段流程。

| 物理与控制设置 | 默认值 |
|---|---|
| 仿真与训练 | 原生 MuJoCo 3.3.6、Gymnasium、Stable-Baselines3 PPO，CPU |
| 物理步长 / 控制周期 | 2 ms / 50 ms，每周期 25 个物理子步 |
| 动作 | 7 维执行器目标增量，每周期最大有效量程的 ±1%，随后执行目标限幅 |
| 观测 | 78 维，在原 75 维状态与目标输入上增加 3 维阶段 one-hot |
| 回合上限 | 12 秒 |
| 初始化扰动 | 位置每轴 ±1 mm、随机轴角度 ±2°、质量与摩擦分别 ±10% |
| 操作区域 | 相对本回合初始位置的位移不超过 80 mm |
| 支撑规则 | 允许掌面接触，禁止安装支架和地面支撑 |

稳定时间、接触占空比与失败判据按物理子步计算。建立握持后完全失去手指接触持续 0.15 秒会失败；掉落、禁止支撑、数值异常，以及穿透超过 1 mm 或离开操作区域持续 0.1 秒，也会失败。失败事件锁存，不能在后续重新接触时抹去。

初握候选偏紧，因此 `rotate` 阶段降低持续五指接触的奖励，允许局部松开并通过换指完成转动；`acquire` 和 `recapture` 阶段再强调五指稳定握持。拇指接触提供较小的正奖励，过大拇指接触力被惩罚。配置中的 `thumb_force_soft_limit=20 N` 是**仿真奖励的软尺度**，不是实机额定值或已经验证的可用握持力。接触力来自 MuJoCo 真值，不能直接当作实测舵机力或实机指尖力。

正向进度及接近目标的正奖励应要求至少一根手指保持有效接触；完全脱离期间不通过自由飞行获得这些正奖励，反向进度仍扣分。完整成功必须经历建立握持和目标处重新抓稳。实际奖励及阶段逻辑见 [env.py](env.py)，全部参数见 [config.py](config.py)。

## 准备真实五指参考状态

使用现有 `/workspace/.venvs/aero-cube` 环境，依赖沿用 [cube_grasp/requirements.txt](../cube_grasp/requirements.txt)：

```bash
cd /workspace/aero-hand-open
/workspace/.venvs/aero-cube/bin/python -m pip install -r experiments/cube_grasp/requirements.txt

/workspace/.venvs/aero-cube/bin/python -m experiments.cube_envelop.prepare \
  --output experiments/cube_envelop/artifacts/prepared_envelope_v1 \
  --seed 73 --seconds 300
```

准备入口搜索真实的全五指接触初握；已有候选时可使用 `--candidate /path/to/candidate.npz` 进行核验。临时固定魔方仅用于构造初始手姿，正式参考状态必须释放为自由刚体，清理辅助约束、外力及相关仿真数据后，再以 **2 ms / 1 ms** 两种步长验证无辅助握持。只有 `prepare_report.json` 批准的状态才可用于训练。候选被找到、画面上看起来包住、单次静态接触成立，都不能替代这一验收。

准备状态等价于将物体预放在手内，不包含从桌面接近和拾取。初始姿态可能已存在五指接触，因此需要同时评价零动作、固定闭合、随机动作基线，确认学习策略是否提高了扰动下的表现。

当前已进行五指候选搜索，候选数量、种子和最终两种步长的验证结果以准备报告为准。准备通过只证明该初态能无辅助握持，不证明已经学会转向。

## 训练命令

默认训练完整的建立握持、转向、重新抓稳任务：

```bash
/workspace/.venvs/aero-cube/bin/python -m experiments.cube_envelop.train \
  --reference experiments/cube_envelop/artifacts/prepared_envelope_v1/reference.npz \
  --output experiments/cube_envelop/artifacts/train_envelope_xyz_v1 \
  --task-mode reorient --hours 2 --steps 500000 --envs 4 \
  --eval-every 32768 --eval-episodes 20 --baseline-episodes 5 --seed 0
```

首次实验最多 2 小时、500,000 个新增环境步、4 个 CPU 并行环境。预算包含初始化、基线和周期验证；退出时还需完成检查点落盘。训练不渲染。Actor/Critic 各两层 128 单元，rollout 为每环境 128 步，batch 128，每批更新 4 次，学习率 `3e-4`。

每 32,768 环境步对 X/Y/Z **各 20 回合**固定验证，共 60 回合。三个轴必须各自连续两次完整验证成功率均达到 90%，才按转向达标条件停止。不能用三轴平均值代替各轴结果。

如果先单独训练握持，应使用独立输出目录：

```bash
/workspace/.venvs/aero-cube/bin/python -m experiments.cube_envelop.train \
  --reference experiments/cube_envelop/artifacts/prepared_envelope_v1/reference.npz \
  --output experiments/cube_envelop/artifacts/train_envelope_hold_v1 \
  --task-mode hold --hours 2 --steps 500000 --envs 4 \
  --eval-every 32768 --eval-episodes 20 --baseline-episodes 5 --seed 0
```

`hold` 验证只有握持这一组，每次 20 回合，连续两次成功率达到 90% 才按握持达标条件停止。此入口不会自动切换为转向任务，也不会把握持成功率记为 XYZ 成功率。

新任务使用 **78 维观测**，旧 `cube_reorient` 的 75 维权重不能直接 `--resume`。同一新任务、同一模式下可以用 `--resume <run>/latest` 恢复，且 `--reference` 必须匹配检查点中的参考状态；`hold` 与 `reorient` 也不能通过修改 `--task-mode` 直接切换续训。检查点恢复策略、优化器与 `VecNormalize` 统计，但重新开始环境回合和随机序列，因此不是逐位一致的断点恢复。

## 评估、轨迹与视频

独立评价完整转向策略：

```bash
/workspace/.venvs/aero-cube/bin/python -m experiments.cube_envelop.evaluate \
  --checkpoint experiments/cube_envelop/artifacts/train_envelope_xyz_v1/best \
  --output experiments/cube_envelop/artifacts/eval_envelope_xyz_v1 \
  --axis all --episodes 100 --seed-start 20000 --video 30
```

`--episodes 100` 在转向模式下表示每轴 100 回合，总共 300 回合；尚未产生 `best` 时可以评价 `latest`。模式从检查点读取：对 `hold` 检查点评价时，只进行 100 个握持回合，不伪造三轴结果。测试种子独立于选模型的验证种子。

| 指标 | 含义 |
|---|---|
| `capture_rate` | 初始五指稳定握持建立率，需连续满足 0.5 秒 |
| `recapture_rate` | 进入转向后，在目标处恢复五指稳定握持的回合比例；初始握持不能计为重新抓稳 |
| `axis_success_rates` | X/Y/Z 各自完成全部阶段及完整成功判据的比例 |
| `contact_duty_cycle` | 各根手指有效接触的物理子步总时长 / 已完成回合的总时长 |
| 姿态误差与有向进度 | 区分转动不足、反向、多转整圈和目标处未稳定 |
| `phase_failure_counts` | 建立握持、转向或重新抓稳阶段中的具体失败原因 |

同时报告掉落率、物体位移、动作变化及失败轨迹。完整转向的 30 秒录像按 XYZ 分段，并标注阶段、目标误差、握持/重新抓稳事件和重置。握持模式的录像明确标为 HOLD。视频包含重置时，不能当作一次连续成功动作。

| 文件或目录 | 用途 |
|---|---|
| `artifacts/prepared_envelope_v1/prepare_report.json` | 搜索来源、无辅助五指握持及两种步长的验收 |
| `artifacts/train_envelope_xyz_v1/status.json` | 状态、训练步数、时间预算、停止原因、最新检查点 |
| `artifacts/train_envelope_xyz_v1/run_config.json` | 实际命令与任务配置 |
| `artifacts/train_envelope_xyz_v1/baselines.json` | 零动作、固定闭合、随机动作对照 |
| `artifacts/train_envelope_xyz_v1/progress.jsonl` | 分轴成功率、握持建立率、重新抓稳率、接触占空比及阶段失败 |
| `artifacts/train_envelope_xyz_v1/checkpoints/` | 策略、优化器、归一化、参考状态和元数据；`latest`/`best` 为索引 |
| `artifacts/eval_envelope_xyz_v1/evaluation.json` | 独立测试汇总及各轴结果 |
| `artifacts/eval_envelope_xyz_v1/{x,y,z}/` | 各轴评估报告及轨迹 |
| `artifacts/eval_envelope_xyz_v1/replay.mp4` | 回放，同名 JSON 记录时间、分段和重置 |

## 对照与结果边界

旧 `cube_reorient` 实验已在 **142,428 环境步**停止，最后一次 **131,072 步**验证中，X/Y/Z 均为 **0/20**。这是旧任务对照，可核对[停止状态](../cube_reorient/artifacts/train_xyz90_tilt20_v1/status.json)和[最后验证](../cube_reorient/artifacts/train_xyz90_tilt20_v1/validation/step_000131072/evaluation.json)，不能替代新实验结果。

新实验的准备是否批准、当前训练阶段、成功率和失败原因均以 `artifacts/` 实际记录为准。五指握持成功与转向成功分别验收，初始姿态本身已能保持也需通过基线比较解释。当前包含仿真位姿、速度与接触真值，不宣称已完成实机感知或部署。

本任务在 `feature/new-manipulation` 分支本地开发，未推送、未发布。
