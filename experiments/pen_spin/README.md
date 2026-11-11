# 竖直右手单次绕拇指转笔

2026-11-11新增：[先训练多指握持的课程](../../docs/pen_grasp_curriculum.md)。该模式学习主动抓稳，从经过短时检查的释放初态开始；原转笔任务仍要求其独立准备门槛，不能用握持初态报告替代。

让 130 mm 单端偏重笔绕固定参考拇指轴正向转一圈，重新抓稳并保持 0.5 秒。CPU 可以执行仿真与训练；能否在首次 24 小时内学会，要由独立评估判断。新任务在主仓库实现，原转方块任务保留作为回归基线。

物理、奖励与验收规格见 [实施设计](../../docs/pen_spin_plan.md)，本次运行结果见 [CPU 首次实验报告](../../docs/pen_spin_cpu_v1_report.md)。

## 环境与工作目录

以下命令均在仓库根目录执行。本云环境已经安装参考依赖，不需要 ROS 2 或连接 ESP32。

```bash
cd /workspace/aero-hand-open
export PEN_PYTHON=/workspace/.venvs/aero-rl-reference/bin/python
export JAX_PLATFORMS=cpu
export MUJOCO_GL=egl
export EGL_PLATFORM=surfaceless
export LIBGL_ALWAYS_SOFTWARE=1
```

`PEN_PYTHON` 选择已有虚拟环境；`JAX_PLATFORMS` 固定 CPU。后三项供无桌面回放使用 EGL 与软件渲染，训练本身不渲染。模块为 JAX、Matplotlib、XDG 与 Mesa 缓存设置仓库内默认可写目录 `experiments/pen_spin/artifacts/cache/`；已有环境变量优先。

未设置 `DISPLAY` / `WAYLAND_DISPLAY` 时，模块会在导入 MuJoCo 前设置上述 EGL 默认值，以支持训练结束后的自动录像；显式设置的渲染环境变量仍优先。仅在录像函数内部设置后端为时已晚，可能导致无桌面机器的 GLFW 初始化失败。

检查依赖与子模块：

```bash
"$PEN_PYTHON" -c "import jax, mujoco, brax; print('JAX', jax.__version__, 'MuJoCo', mujoco.__version__, 'Brax', brax.__version__); print(jax.devices())"
git submodule status --recursive
```

参考版本为 Python3.10、JAX/JAXLIB0.6.2、MuJoCo/MJX3.3.6、Brax0.13.0、Flax0.10.7、Orbax Checkpoint0.11.25、NumPy1.26.4、ml-dtypes0.5.3。新 Linux 机器需先初始化子模块并安装兼容依赖；不要直接让 `uv` 默认源更新 MJX，子模块的 `tool.uv.sources` 指向 MuJoCo 开发版，会改变本实验条件。

保存本版默认配置：

```bash
"$PEN_PYTHON" -c "from experiments.pen_spin.config import PenSpinConfig; PenSpinConfig().save('experiments/pen_spin/artifacts/baseline-v1.json')"
```

配置会随实验保存。改变模型、物理或任务判定需建立新配置、重新准备，不能沿用旧通过报告；仅改变 PPO 或预算也应保留实验记录。

## 1. 准备与静态初握搜索

```bash
"$PEN_PYTHON" -m experiments.pen_spin.prepare \
  --output experiments/pen_spin/artifacts/prepared-v1 \
  --config experiments/pen_spin/artifacts/baseline-v1.json \
  --max-candidates 2048 \
  --budget-seconds 900 \
  --seed 0
```

命令先检查模型、原生/MJX接触及时间步一致性，再搜索无辅助静态初握。搜索最多2,048个候选或900秒，检查与首次编译另有耗时。每个候选无外力模拟2秒；最后0.5秒需连续满足有效拇指和另一根手指接触、低线速度与角速度、无支撑或明显穿模。参考姿态还需满足旋转任务的空间包络和非奇异角度条件。

读取 `prepare_report.json` 判断结果：

- 退出码 **0**：所有准备检查通过，生成 `reference.npz`。
- 退出码 **2**：完成了有效失败诊断，报告有原因，禁止进入长时间训练。
- 已有 `reference.npz` 的目录不能覆盖；重试请换目录，避免旧通过结果和新报告混淆。

输出包括 `config.json`、`prepare_report.json`；通过时另有 `reference.npz`。失败时可能保留 `best_candidate.npz` 和标记 `approved=false` 的 `diagnostic_reset.npz`，均只用于诊断，**不能替代批准参考**。`--skip-mjx-check` 是诊断选项，会阻止初握搜索和批准。

若手的运动范围无法形成初握、笔依靠掌根/拇指基座/支架、穿透过大或时间步改变后不能保持，应保留失败报告。不得以提高摩擦、外力或瞬移掩盖可达性失败。

准备失败后，如有 `diagnostic_reset.npz`，可输出被动诊断视频：

```bash
"$PEN_PYTHON" -m experiments.pen_spin.evaluate \
  --diagnostic-reference experiments/pen_spin/artifacts/prepared-v1/diagnostic_reset.npz \
  --output experiments/pen_spin/artifacts/diagnostic-v1 \
  --video-seconds 30
```

`--diagnostic-reference` 与 `--checkpoint` 互斥。此命令使用固定执行器目标，每10秒重复同一个未经批准的首次重置，累计30秒真实物理时间。它没有学习权重，也不执行100初态的独立策略评估。输出 `diagnostic.mp4`、`diagnostic_report.json`、`diagnostic_trajectory.npz` 和 `preview.png`；画面明确标注 `NO TRAINED POLICY` 与 `UNAPPROVED RESET`。任务失败锁存后物理继续，以观察笔如何掉落。

## 2. CPU 课程训练

只有准备通过才执行：

```bash
"$PEN_PYTHON" -m experiments.pen_spin.train \
  --reference experiments/pen_spin/artifacts/prepared-v1/reference.npz \
  --output experiments/pen_spin/artifacts/train-v1 \
  --config experiments/pen_spin/artifacts/baseline-v1.json \
  --budget-hours 24
```

`--reference` 指定批准初握；`--output` 保存记录与检查点。首次24小时从 `prepare_report.json` 记录的准备启动时刻计时，包括模型检查、搜索、编译、训练和最终评估；不要准备后搁置一天再启动训练。`--budget-hours` 可以收紧该批预算，不能延长原截止时间。调试可设更小预算并加 `--num-timesteps` 限制总环境步数，这只检验流程，不能代替训练结果。

课程为稳定握笔、90°、180°、360°，阶段上限分别2、4、6、12小时。每30分钟用固定20个验证种子评估，连续两次成功率≥80%才升级；连续四次成功率与有效进度无明显改善时停止并诊断。验证初态默认包含最多1mm位置与2°角度扰动。90°/180°课程的重抓方向是初始方向绕固定轴旋转后的目标方向，360°返回初始方向。

| CPU 参数 | 默认值 |
|---|---:|
| 物理步长 / 控制周期 | 2ms / 50ms |
| 训练 / 评估环境配置 | 8 / 4 |
| Actor / Critic 隐藏层 | 各128、128 |
| PPO rollout / batch / minibatches | 16 / 32 / 4 |
| 每批更新 | 4次 |
| 学习率 / discount / GAE | 3e−4 / 0.99 / 0.95 |
| PPO clipping / entropy | 0.2 / 0.005 |

CPU可批量模拟，但增加环境数可能增加编译、内存和每步耗时。本配置不能理解为1000只手的GPU训练吞吐。

训练输出 `training_report.json`、`curriculum.jsonl` 和 `stages/<hold|turn90|turn180|turn360>/<last|best>/` 检查点目录。检查点包含 `checkpoint.pkl`、`config.json`、`metadata.json`、参考状态及准备报告和校验信息。依据报告选择实际路径，并核对课程目标；握笔或90°检查点不能报告为360°成功。首次编译尚未产生有效参数就耗尽预算时，报告可以没有选定检查点，不应生成假权重。

命令为最终评估预留最多15分钟或剩余预算10%的时间；有有效检查点时自动在 `final_evaluation/` 请求100个独立初态、360°完整目标和30秒视频。即使训练停在早期课程，也按完整目标如实测量。预算不足可能只完成部分回合或来不及生成视频，应读取实际 `status` 和已完成数量，不能把请求100回合当作已完成100回合。

续训示例：

```bash
"$PEN_PYTHON" -m experiments.pen_spin.train \
  --reference experiments/pen_spin/artifacts/prepared-v1/reference.npz \
  --output experiments/pen_spin/artifacts/train-v1-resumed \
  --resume experiments/pen_spin/artifacts/train-v1/stages/hold/last \
  --new-batch \
  --budget-hours 24
```

示例恢复握笔阶段并通过 `--new-batch` 明确开启新预算；恢复其他阶段时替换为报告里的实际路径。省略 `--new-batch` 表示继续原批次的剩余时间。续训恢复策略参数与观测归一化状态，重建优化器、环境状态及随机数流程，不是逐位一致的暂停恢复。新增预算要与首次24小时分开披露。外层看门狗限制编译、评估和阶段时长，提前发送退出信号，到硬截止时强制停止。

## 3. 独立评估与30秒视频

```bash
"$PEN_PYTHON" -m experiments.pen_spin.evaluate \
  --checkpoint experiments/pen_spin/artifacts/train-v1/stages/turn360/best \
  --output experiments/pen_spin/artifacts/evaluation-v1 \
  --episodes 100 \
  --seed-start 10000 \
  --goal-degrees 360 \
  --backend native \
  --video-seconds 30
```

`--checkpoint` 接受检查点目录或 `checkpoint.pkl`。示例要求已经训练到360°并有验证产物；实际以 `training_report.json` 的 `selected_checkpoint` 为准，提前停止时不要照搬不存在的目录。`--goal-degrees 360` 明确评估最终完整任务；省略时使用检查点所属课程目标，报告同时标明训练目标与测试目标。默认原生MuJoCo回放；`--backend mjx` 可核对训练引擎，默认最多4个环境同时评估。10000起始的100个种子应保留为独立测试，不参与挑选模型。只导出统计可设 `--video-seconds 0`。

默认评估沿用检查点的批次截止时间，到期会报告预算耗尽。若日后只想重新分析已有策略，可以显式追加 `--budget-seconds 900`，开启最多15分钟的独立只读评估预算，并使用新的输出目录；该结果标记为后续评估，不能算作首次24小时内完成。自动最终评估不会传入这个额外预算。

| 文件 | 用途 |
|---|---|
| `evaluation.json` | 课程目标、完整成功率、阶段到达率、转角、回报、耗时与逐回合指标 |
| `failure_samples.json` | 失败回合及其逐控制步轨迹 |
| `trajectories/episode_<seed>.npz` | 每回合2ms子步轨迹、字段名、种子和控制频率动作 |
| `rollout.mp4` | 30秒真实仿真时间回放 |
| `video_trajectory.json` | 每帧对应的回合、重置与轨迹指标 |
| `video_info.json` | 帧率、时长、回合数和引擎 |

单回合最多10秒，所以30秒视频可能含多个回合，画面和轨迹会明确标注重置。每帧对应50ms物理时间、20fps，不通过重复终帧或慢放凑时长。若回合在某个2ms子步终止，任务指标当即冻结，但仅用于视频的物理副本继续到控制周期末，画面标注 `TERMINAL` 及额外时间；下帧重置。蓝色箭头表示固定参考轴，橙色弧线表示右手定则正方向。视频完成不表示任务成功。

读报告时注意指标含义：

| 指标 | 解释 |
|---|---|
| `success_rate` | 完整满足本次评估目标的回合比例；先核对 `goal_angle` 与检查点训练目标 |
| `reached_spin_rate` / `reached_catch_rate` | 到达旋转/重抓阶段的比例，不能替代最终稳定成功率 |
| `theta` / `phi` | 笔方向与A端轨道的净转角，单位rad，乘 `180/π` 转为度 |
| `median_valid_progress` | 每回合处于有效包络且轨迹仍可证明时的两种角度联合进度峰值，再取中位数，单位rad；按控制步采样 |
| `failure_reason_counts` | 超时、越界、失去接触、穿透、支撑或数值异常的实际计数 |
| `final_tilt_degrees_mean` | 回合终点平面外倾角的均值，不是全过程最大倾角 |
| `final_action_change_squared_mean` | 最后一个控制步动作变化平方均值，不是完整轨迹平均 |
| `mean_return` | 冻结奖励下的回报，只适合同奖励与同课程的比较 |

`success_rate_std` 是单回合0/1成功结果的标准差，不是成功率估计的标准误或置信区间。失败率包含超时、支撑和数值错误，不能直接叫掉落率；需要查看失败代码和轨迹。失败样例最多保存前10个，汇总统计仍使用全部评估回合。

轨迹NPZ通过 `np.load(path, allow_pickle=False)` 读取，`fields` 定义 `trace` 每列含义；`trace` 是真实2ms子步，终止后的填充被去重，`actions` 是50ms控制周期的7维命令。两种采样率不能按数组行数直接一一对应。

## 观测、动作和实物边界

Actor读取手部反馈、上一周期动作、笔相对位置/方向/姿态、线速度与角速度、手指接触、课程角度、阶段及轨迹状态；Critic另读关节与执行器真值。训练和回放共用观测、动作映射及每个物理子步判定。

输出7维归一化动作，每控制周期改变执行器目标最多有效量程±5%，再限幅。它不是笔的目标位姿或直接指定全部关节轨迹。

该Actor依赖仿真笔位姿与接触真值，不能删除这些输入后直接部署到树莓派或ESP32。实物需要对应感知、时序、标定和执行控制；量化、蒸馏与连续花式转笔属于后续工作。

## 测试与验收

```bash
"$PEN_PYTHON" -m unittest discover -s experiments/pen_spin/tests -v
```

默认测试会跳过需要完整CPU MJX编译的集成项。需要验证训练引擎的重置、控制步和观测一致性时，显式运行：

```bash
PEN_SPIN_MJX_TESTS=1 "$PEN_PYTHON" -m unittest discover -s experiments/pen_spin/tests -v
```

首次完整编译可能较慢，耗时不代表策略已经开始学习。

测试验证被覆盖的实现条件，机械可达性与学习结果仍由实际准备和评估报告给出。完整成功要求两种净转角同时达标、四个象限有有效拇指接触、脱手和空间包络受限、最后低速重抓0.5秒，且无掌根、拇指基座、支架或地面支撑。

三个不同的偏离门槛是：笔平面外倾角45°、实际拇指中点到固定参考轴的横向距离10mm、实际拇指轴相对固定参考轴夹角30°。后者是本版对“拇指明显偏离”的明确实现，不能通过保持中点不动而大幅转动拇指来绕过检查。

“代码能执行”和“学会转笔”分开验收。若准备失败，应交付有效代码和真实失败报告，不虚构检查点、策略视频或成功率；可提供明确标注“无训练策略”的被动诊断视频。若训练后学不会，应保留实际权重与失败样例。
