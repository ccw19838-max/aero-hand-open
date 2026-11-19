# 2019 OpenAI Dactyl 与本次改动的依据

查阅时间：2026-11-13。原始网页、论文文本及选取的代码快照保存在 `research/`。

## 第一手资料

1. [OpenAI：Solving Rubik’s Cube（2019）](https://openai.com/index/solving-rubiks-cube/)
2. [论文：Solving Rubik’s Cube with a Robot Hand，arXiv:1910.07113](https://arxiv.org/abs/1910.07113)，重点为 §6.1–6.3 和附录 C。
3. [OpenAI robogym](https://github.com/openai/robogym)，本次核对的提交为 `9ff1111fd557b1a01911d7d8c9d59a453071c96e`：
   [locked.py](https://github.com/openai/robogym/blob/9ff1111fd557b1a01911d7d8c9d59a453071c96e/robogym/envs/dactyl/locked.py)、
   [locked_parallel.py](https://github.com/openai/robogym/blob/9ff1111fd557b1a01911d7d8c9d59a453071c96e/robogym/envs/dactyl/goals/locked_parallel.py)、
   [cube_env.py](https://github.com/openai/robogym/blob/9ff1111fd557b1a01911d7d8c9d59a453071c96e/robogym/envs/dactyl/common/cube_env.py)。
4. 对照方案：[NVIDIA IsaacGymEnvs AllegroHand](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/isaacgymenvs/tasks/allegro_hand.py) 的姿态/位置/动作/成功/掉落奖励。该任务代码读取自公开主分支，不能把它当作本项目的性能证明。

## 原方案实际做了什么

论文 §6.1 写明：20 个驱动关节，每个使用 11 档离散动作，动作是关节位置的相对变化。核心奖励是距离目标的改善、完成目标额外 +5、掉落 −20。随机生成整体翻转和顶层转动目标，而不是等待一条完整成功示范之后才启用强化学习。

§6.2 使用 PPO 和循环策略，包含 2048 单元前馈层与 1024 单元 LSTM；Actor/Value 分开，Value 可读取仿真真值。ADR 从简单环境开始，按能力逐渐扩大随机化范围，帮助记忆网络适应不同动力学。高层魔方求解算法负责给动作目标，低层策略负责操作。

§6.3 报告：完整魔方任务使用 64 张 V100 和 920 台各 32 核 CPU 的采样机器，持续训练数月；方块重定向也使用 32 张 V100 与 400 台 32 核采样机器。当前四个 CPU 工作进程只能采用其方法思路，不能承诺类似训练速度或成功率。

另一个重要差别：论文和 robogym 锁定魔方的目标姿态容差是 0.4 rad（约 22.9°）。本项目保留更严格的 **5°、有向 +90°±5°、连续保持一秒**，不能把原论文的成功数字直接当作这里的成功率。

核心奖励列表没有“五指必须始终接触”的项。本次五指稳握是按用户要求新增的约束，不是声称 OpenAI 原作者用了同样的接触奖励。

## 如何适配当前手

| 原工作思路 | CPU 适配 | 边界 |
|---|---|---|
| 闭环 PPO | 从第一批 rollout 就更新策略 | 不再以 CEM 完整示范为启动门槛 |
| 相对关节动作 | 七个电机目标每控制周期最多增减有效量程 2% | 控制腱绳电机目标，不增加自由度 |
| 目标距离改善奖励 | 完整三维姿态误差的差分奖励 | 同时检查有向角，拒绝额外整圈别名 |
| ADR 逐步加难 | 名义任务通过后才使用 ±5%、±10% 摩擦/阻尼随机化 | 这是有限能力课程，不是完整 ADR 实现 |
| 记忆网络适应未知动力学 | 当前用 138 维仿真真值和 CPU MLP；两工作进程分别用 1/2 ms | 本版不实现论文 LSTM，不宣称其元学习效果 |
| 分阶段目标 | 五指保持 → 30° → 60° → 90° | 低角度成绩单独标识，最终评估固定 90° |

旧轨迹中 0 N 是某根手指没有接触魔方，和该指电机是否有驱动力不是同一件事。过去只要求拇指加另外两指，因此中指/无名指 0 N 仍可积累目标保持时间。本次抓稳必须同时满足：拇指 ≥0.2 N，食指/中指/无名指/小指各 ≥0.1 N，且不过大；五指不足时保持计时归零。转动允许短暂换指，但连续超过 0.15 秒没有拇指和至少另外两指的有效支撑会失败。

奖励对接触覆盖做饱和处理，超过上限的力会扣分，避免策略单纯挤压。五指有力仍不能单独证明抓稳；完整姿态、低线/角速度、操作区域、穿透和支架支撑检查同时生效。

这些力阈值、2% 控制增量、课程角度和奖励系数是本项目第一版工程配置，需通过实验验证，不是从 OpenAI 原论文抄来的常数。当前手只有 7 个驱动，原 Shadow Hand 的网络和权重不能直接搬过来执行。
