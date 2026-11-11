# 多指握笔：GitHub 源码调研与适配建议

调研日期：2026-11-11（北京时间）。对象是本仓库的竖直 Aero 右手、130 mm / 16 g 单端偏重笔、7 维执行器控制和 CPU 训练。下面区分已经查证的上游实现、当前实验数据和建议的新实验；本次调研没有修改正在运行的 `grasp_train_v4` 奖励、模型或检查点。

## 1. 当前最需要解决的问题

当前 Actor 直接使用 MuJoCo 的笔位置、方向、四元数、速度和接触力，**尚不存在相机姿态估计误差**。先让控制器在真值观测下握稳，再引入真实感知的噪声与延迟，才能区分控制问题和视觉问题。

2,048 训练步检查点的早期原生 MuJoCo 验证结果如下。这不是最终策略结果，也不足以断言训练不能收敛。

| 项目 | 早期验证结果 |
|---|---:|
| 验证回合 | 20，种子 1000–1019 |
| 连续稳定握持 1 s 成功 | 0 / 20 |
| 穿透超限终止 | 17 / 20 |
| 无效支撑终止 | 3 / 20 |
| 平均回合时长 | 0.1873 s |
| 曾同时满足“拇指 + 至少另外两指有效接触”的回合 | 0 / 20 |

来源：[早期评估 JSON](../experiments/pen_spin/artifacts/grasp_train_v4/early_policy_check/evaluation.json) 及其 `trajectories/` 中的 2 ms 子步记录；有效接触门槛为每指法向力合计 ≥0.02 N。有些回合曾有三指接触，但不包含拇指，因此不满足目标。当前进度另见 [live_status.json](../experiments/pen_spin/artifacts/grasp_train_v4/live_status.json)。

这些记录提示优先检查预握姿的可达性、接触方向、闭合控制和接触数值稳定性。仅增加“接触手指数”奖励，不能保证形成对向夹持；只有三指数量也不能证明力闭合。穿透终止还可能涉及控制速度、接触柔度、求解器或步长，需要用轨迹和对照仿真定位，不能只归因于奖励权重。

## 2. 哪些项目值得借鉴

| 项目 | 从源码确认的设计 | 对本项目的用途与限制 |
|---|---|---|
| [PenSpin](https://github.com/HaozhiQi/penspin) | 6 类典型笔握姿；筛选并缓存可保持的初态；旋转奖励配合位移、速度、姿态偏移和用力惩罚；教师/学生与实机数据流程 | 最贴近细长物体操控。使用 Allegro 16 维动作，握姿、权重不能直接用于 Aero；连续轴向旋转奖励不能代替我们的绕拇指完整轨迹验收 |
| [LEAP Hand Sim](https://github.com/leap-hand/LEAP_Hand_Sim) | 从随机扰动握姿中筛选存活状态，缓存供后续旋转训练使用 | 借鉴初态库和课程衔接。其抓握生成配置关闭策略动作，不应把这部分当成“从任意释放状态学抓握”的 RL 实现 |
| [DexGraspNet](https://github.com/PKU-EPIC/DexGraspNet) | 接触距离、力闭合相关代理能量、物体穿透、自穿透和关节限位共同优化，再进行物理验证 | 借鉴预握姿搜索目标；不是可直接复制的在线 PPO 奖励，也不是机械可达性的保证 |
| [Wuji mjlab](https://github.com/wuji-technology/wuji-mjlab) | 参考动作跟踪、残差控制、动作平滑、用力约束；多标记 PnP 笔位姿估计；相机噪声/延迟/丢帧模型 | 奖励与实机观测衔接最完整。使用 20 关节 Wuji 手、250 mm 笔及参考轨迹，原训练要求 NVIDIA GPU；需要重新适配 Aero |
| [AprilTag](https://github.com/AprilRobotics/apriltag) | 小型 C 库进行标记检测；已知尺寸与相机内参后估计标记位姿 | 适合作为 CPU 实机感知候选，须实际测试遮挡、分辨率和延迟 |
| [FoundationPose](https://github.com/NVlabs/FoundationPose) | 物体 6D 位姿初始化与跟踪，可使用 CAD 模型；官方安装依赖 CUDA/PyTorch3D/NVDiffRast | 可作为未来 GPU 视觉方案；不作为当前 CPU 握持阶段的默认依赖 |

上游展示不能证明 Aero 的 7 个执行器能实现同样动作。特别是我们的目标包括竖直安装、单次绕拇指、重抓与保持，不能用沿笔杆自转或其他手型的成功视频替代验收。

## 3. 奖励、握姿和动作控制的具体发现

### PenSpin / LEAP：先解决“从哪里开始”

PenSpin 的 [握姿生成代码][penspin-grasp] 明确列出拇指对食指/中指/无名指等典型构型，然后扰动、仿真筛选，保存满足存活条件的手与物体状态。LEAP 的 [抓握配置][leap-grasp-config] 设置 `disable_actions: True`，其 [缓存代码][leap-grasp] 保存完成筛选时长的状态。两者都给旋转策略提供了更容易学习的起点。

对我们可借鉴的是：为 Aero 构建“拇指对食指 + 中指”“拇指对中指 + 无名指”等可达候选，再训练主动闭合与保持。候选应在 **7 维执行器目标空间** 搜索，并让原腱绳模型运动到实际姿态。不能直接填写 Allegro 的 16 个关节角，也不能自由调整 Aero 原本耦合的关节。

当前主动抓握初态只证明释放后短时可存活，并不证明随后能形成夹持。过去的被动握姿搜索已经失败；借鉴典型构型可以改善搜索，但仍需如实记录可达性失败，不能假定一定找得到握姿。

PenSpin 的 [奖励实现][penspin-reward] 和 [基础配置][penspin-config] 包括沿目标轴的旋转速度、物体线速度、偏离初始手姿态、执行器力矩、用力与运动的组合惩罚、位置偏移等。基础配置中的旋转速度截断为 −0.5 到 0.5 rad/s，部分权重为：旋转 1、线速度 −0.3、姿态偏移 −0.1、力矩 −0.1、work −1、位置偏移 −0.1。

这些数值仅说明上游设计，**不直接作为我们的新权重**：频率、单位、动作维度、奖励积分和动力学均不同；基础配置的点云维度为 0，相应笔点云高度分散项也不能一概说成始终启用。本阶段只握持，继续关闭旋转进度奖励。

### DexGraspNet：接触数量之外，还要看接触几何

[能量函数][dex-energy] 的结构是：

```text
E = E_fc + 100 E_distance + 100 E_penetration
         + 10 E_self_penetration + E_joint_limits
```

其中 `E_fc` 使用接触法向与抓取映射矩阵构造平方范数，其他项限制接触点到表面的距离、穿透和关节越界。这是优化候选抓握的代理目标；低能量不等于已经证明在摩擦、重力和执行器限制下稳定，项目也提供 [物理验证流程][dex-validation]。

对 Aero 的建议是先把这类几何目标用于预握姿排序：接触来自笔的相对两侧，拇指能与另外两指配合，接触位置具有合理的轴向间距，同时无支架支撑、无深穿透。笔的偏心质心也必须进入力矩判断。筛选后的候选仍需在原生 MuJoCo 与 MJX 中接受无外力保持测试。

### Wuji：参考目标、动作平滑和用力约束

Wuji 的 [奖励实现][wuji-reward] 使用 `exp(-(error/sigma)^2)` 奖励笔位置与笔轴上采样点对参考轨迹的跟踪，还包括关节/连杆参考跟踪。它的 [动作实现][wuji-actions] 是参考动作上的残差加指数平滑；动作惩罚同时考虑一阶和二阶差分：

```text
||a_t - a_(t-1)||² + ||a_t - 2 a_(t-1) + a_(t-2)||²
```

这与本任务从头寻找抓握不同。其 [训练说明][wuji-train] 提供 5 组共 40 段参考轨迹，包括稳定抓握旋转；不能把“有参考动作的跟踪策略”描述为仅靠任意设计的奖励自行发现完整动作。

对当前任务可以分开验证两项改进：

1. 在找到有效握姿后，增加笔中心和有方向笔轴相对握持目标的误差项；初始释放位置如果不可握，不应强行将其当成必须保持的目标。对细长近似轴对称物体，不必为了保持而强行锁定无关的笔杆轴向滚转角。
2. 单独比较动作二阶差分惩罚或执行器目标平滑。仍保留每控制周期有效量程 ±5% 的增量上限和最终限幅，训练与回放共享全部滤波状态；平滑过强也可能让闭合来不及，需做对照。

一个不能直接照抄的细节：Wuji 的归一化力矩项显式要求一关节一执行器且传动比为 1，因此 `actuator_force` 可按 N·m 解释。Aero 包含腱绳传动，原始执行器力可能是 N。适配时应读取手部自由度的 `qfrc_actuator`，或利用传动映射计算关节力矩；再用物理上有效的力矩限值归一化，或分别归一化腱绳拉力和关节力矩。不能混用 N 与 N·m，更不能未经标定就宣称与实机电流一致。

## 4. 姿态估计：仿真真值与实机观测分别处理

### 现阶段保持仿真真值

本地 [env.py](../experiments/pen_spin/env.py) 的 `_sample()` 和 `observe()` 直接读取模型状态。Actor 当前有 54 个输入，包括完整四元数、速度和接触力。更换视觉算法不能修复当前真值环境中的抓握失败。

### 优先研究 Wuji 的多标记 PnP

Wuji 的 [pen_pose.py][wuji-pose] 不是直接对整根笔运行大模型：它根据自定义 18 种标记图案的已知三维角点，检测图像角点，联合求解 PnP。默认至少需要两个标记，使用重投影误差、整标记一致性、几何可见性检查和共面多解处理筛选候选。源码默认最大重投影误差为 4 px，这个门槛需要按我们的相机标定与图像尺寸重新确定。

它没有对每帧笔位姿做时间平均；上一帧只用于近似同样合理解之间的连续性判断等求解辅助。不能误写为“加了滤波，所以遮挡也能准确估计”。其 [可见性检查][wuji-visibility] 主要覆盖标记朝向和笔自身遮挡，不包含完整的手指外部遮挡模型。

[世界坐标观测器][wuji-world] 用固定腕部标记建立相机到手部坐标的变换。借鉴时应明确本项目使用笔局部 Z 轴；Wuji 训练跟踪代码采用的笔轴约定不同，需要显式转换，不能直接复用数值。上线数据协议还应自行携带采集时间、有效标志、重投影误差、有效标记数和观测年龄；不能假设这些字段都已经由上游发布接口提供。

对我们的 130 mm、直径 10 mm 笔，建议先验证多面小标记或两端不同颜色/图案的可观测性。Wuji 的 250 mm 笔及其标记布局不能等比例直接当作已验证方案。若增加标记端帽导致质量、外形或碰撞变化，必须回写仿真模型。AprilTag/OpenCV PnP 可在 CPU 上实现，但树莓派的实际帧率、遮挡可用率和端到端延迟仍需要实测。

### 只需要哪些姿态信息

已知 A、B 两端的三维位置，可以计算几何中心和从 B 指向 A 的单位轴，表达位置 3 个自由度加轴方向 2 个自由度。这与 Wuji [观测代码][wuji-observation] 用笔轴采样点消除无关滚转的思路一致。但有三点需要区分：

- 单目图像里的两个二维端点不能无条件唯一恢复完整三维姿态；需要额外几何约束、标记 PnP、双目或深度观测。
- 两个三维端点也不能确定绕笔杆自身轴的滚转角。A/B 必须可区分，否则可能发生轴方向翻转和错误的转圈累计。几何中心与向 A 偏移 6.875 mm 的质心也不能混淆。
- 当前 Actor 仍读取完整四元数及其他仿真真值。改成中心加有方向轴，需要更新观测版本、归一化和策略训练；不能在实机端把缺失输入随意填零。接触力和速度等输入也需逐项解决传感、估计或重新训练替代的问题。

后续可参照 Wuji 的相机观测模型，分别加入每回合标定偏差、逐帧噪声、延迟和连续丢帧，并跟踪数据年龄。先根据实际摄像头录像测出分布，再用于训练；当前基线不凭空加入未经测量的大幅噪声。

PenSpin 的 [实机视觉代码][penspin-real] 另有 Tracking SAM 分割、深度点云与端点提取流程，依赖和模型较重。FoundationPose 的 [安装说明][foundation-readme] 也明确采用 CUDA 路线。两者可作为未来视觉研究参照，当前优先验证轻量标记方案。

## 5. 适配到本仓库的实验顺序

下面是调研后建议的新实验，**尚未实现或训练验证**，不改变本轮已冻结的基线与验收标准。

| 顺序 | 修改位置 | 实验内容 | 需要观察的证据 |
|---|---|---|---|
| A：接触诊断 | `model.py`、`env.py`、`evaluate.py` | 导出逐子步接触点/法向、穿透深度、执行器目标及关节力矩；分别比较 2 ms 和 1 ms 下同一初态与控制序列 | 穿透发生在什么几何对、什么闭合速度下；是否与步长敏感有关 |
| B：预握姿 | `grasp_prepare.py` | 在 7 维控制空间搜索有拇指对向接触的典型构型，按几何和无辅助仿真结果排序；记录主动闭合的可行性 | 可行候选比例、首次三指有效接触时间、无支撑存活时长；没有候选也应输出失败原因 |
| C：握持奖励 | `grasp.py`、`config.py` | 在 B 的可行初态上比较选定手指组合的接近/接触项；有有效目标后再加入位置/笔轴保持项及合理归一化的用力项 | 真实稳定保持时间、成功率、偏移、用力；防止仅提高总奖励却不握稳 |
| D：闭合控制 | `env.py`、`config.py`、检查点状态 | 分别比较二阶动作差分惩罚和目标平滑，不一次叠加所有改变 | 动作变化、峰值接触、穿透率、闭合速度与保持成功率 |
| E：感知与迁移 | 新的实机观测适配层 | 摄像头标定、标记 PnP、中心/轴观测版本、可用输入核对、噪声延迟训练 | 位置/方向误差、A/B 混淆、遮挡恢复、无效帧比例、延迟分位数 |

新奖励的结构可写为：

```text
dt * [接近选定接触区域 + 对向接触 + 稳定保持
      + 有效握姿的位置/笔轴跟踪
      - 穿透趋势 - 过度用力 - 速度 - 动作变化]
+ 一次性成功/失败事件
```

这只是分解结构，不是已验证的权重表。各项量纲和尺度需先测量，按新配置记录；参考项目也没有给出能保证本手型握稳的一组通用权重。连续软惩罚不能取代已有硬失败条件。

对照时固定评估种子、总时间预算、网络和 PPO 配置，先逐项比较，保留基线。保持原重力、摩擦、碰撞、动作限幅以及支撑/穿透失败规则，不通过改物理或放宽成功门槛提高成功率。先完成一轮握持再进入旋转：固定 20 初态验证连续两次 ≥80%，再做独立 100 初态评估；当前微小初态扰动还需要逐步扩大，不能作为实机鲁棒性的证明。新初态分布与旧初态分布分别报告，避免把更容易的重置条件误认为策略进步。

当前模式的握持通过后，还需验证绕拇指几何约束、完整转角判断以及控制器交接。可以从成功轨迹保存有效握姿；若需要主动控制才能保持，就验证主动交接，而不把它标为被动稳定握姿。

## 6. 可复核来源与版本

下列源码链接锁定本次实际读取的提交，避免以后默认分支更新改变结论。这里只研究设计，没有移植上游代码或权重。若以后复制实现，应分别核对代码、模型和数据许可证；尤其 FoundationPose 使用 NVIDIA Source Code License，不能笼统按 MIT 处理。

| 仓库 | 本次读取提交 |
|---|---|
| HaozhiQi/penspin | `5035c52dc949e81313af71d9a0ca5ead8b463a5a` |
| leap-hand/LEAP_Hand_Sim | `150bc3d4b61fd6619193ba5a8ef209f3609ced89` |
| PKU-EPIC/DexGraspNet | `bd1b13d7248729af117e1d46aaa6266b147a3c7b` |
| wuji-technology/wuji-mjlab | `b2b98d7054a8d1b0f5cf295d5acaebc4358bc6a9` |
| AprilRobotics/apriltag | `b7c0ebe9aa20f82ec7a828579004f9e706bfecd9` |
| NVlabs/FoundationPose | `a1b694b83e633c2cb6115b9063d940a687759392` |

[penspin-grasp]: https://github.com/HaozhiQi/penspin/blob/5035c52dc949e81313af71d9a0ca5ead8b463a5a/penspin/tasks/allegro_hand_grasp.py#L19
[penspin-reward]: https://github.com/HaozhiQi/penspin/blob/5035c52dc949e81313af71d9a0ca5ead8b463a5a/penspin/tasks/allegro_hand_hora.py#L520
[penspin-config]: https://github.com/HaozhiQi/penspin/blob/5035c52dc949e81313af71d9a0ca5ead8b463a5a/configs/task/AllegroHandHora.yaml#L43
[penspin-real]: https://github.com/HaozhiQi/penspin/blob/5035c52dc949e81313af71d9a0ca5ead8b463a5a/real/robot_controller/real_robot/robot.py#L135
[leap-grasp-config]: https://github.com/leap-hand/LEAP_Hand_Sim/blob/150bc3d4b61fd6619193ba5a8ef209f3609ced89/leapsim/cfg/task/LeapHandGrasp.yaml#L22
[leap-grasp]: https://github.com/leap-hand/LEAP_Hand_Sim/blob/150bc3d4b61fd6619193ba5a8ef209f3609ced89/leapsim/tasks/leap_hand_grasp.py#L73
[dex-energy]: https://github.com/PKU-EPIC/DexGraspNet/blob/bd1b13d7248729af117e1d46aaa6266b147a3c7b/grasp_generation/utils/energy.py#L10
[dex-validation]: https://github.com/PKU-EPIC/DexGraspNet/blob/bd1b13d7248729af117e1d46aaa6266b147a3c7b/grasp_generation/scripts/validate_grasps.py
[wuji-reward]: https://github.com/wuji-technology/wuji-mjlab/blob/b2b98d7054a8d1b0f5cf295d5acaebc4358bc6a9/src/wuji_mjlab/tasks/pen_spin/mdp/rewards.py#L19
[wuji-actions]: https://github.com/wuji-technology/wuji-mjlab/blob/b2b98d7054a8d1b0f5cf295d5acaebc4358bc6a9/src/wuji_mjlab/tasks/pen_spin/mdp/actions.py
[wuji-train]: https://github.com/wuji-technology/wuji-mjlab/blob/b2b98d7054a8d1b0f5cf295d5acaebc4358bc6a9/src/wuji_mjlab/tasks/pen_spin/docs/train_zh.md#L7
[wuji-pose]: https://github.com/wuji-technology/wuji-mjlab/blob/b2b98d7054a8d1b0f5cf295d5acaebc4358bc6a9/deploy/pen_spin/lib/pen_pose.py#L105
[wuji-visibility]: https://github.com/wuji-technology/wuji-mjlab/blob/b2b98d7054a8d1b0f5cf295d5acaebc4358bc6a9/deploy/pen_spin/lib/pose_visibility.py
[wuji-world]: https://github.com/wuji-technology/wuji-mjlab/blob/b2b98d7054a8d1b0f5cf295d5acaebc4358bc6a9/deploy/pen_spin/scripts/pen_world_observer.py#L38
[wuji-observation]: https://github.com/wuji-technology/wuji-mjlab/blob/b2b98d7054a8d1b0f5cf295d5acaebc4358bc6a9/src/wuji_mjlab/tasks/pen_spin/mdp/observations.py#L144
[foundation-readme]: https://github.com/NVlabs/FoundationPose/blob/a1b694b83e633c2cb6115b9063d940a687759392/readme.md#L104
