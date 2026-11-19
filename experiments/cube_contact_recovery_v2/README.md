# X 轴接触恢复 V2

2026-11-17 的诊断确认两处反馈错误：有电机驱动的拇指根部 `right_t_link/th_mp_collision` 没有计入拇指力；接触力为拇指优先，但奖励使用的指尖距离原本为食指优先。V2 在独立环境中修复这两处，保留旧检查点及其代码，不混用新旧评估。

最终任务仍是原始初握出发绕固定 X 轴正向转 90°，有符号转角和完整姿态误差均在 5° 内，五指真实受力、低速度连续保持 1 秒。最终独立验收为 1 ms 与 2 ms 步长各 100 个新初态全部通过。手保持 20° 倾角，57 mm 魔方、质量、摩擦、重力和七电机模型不变。Y 最佳策略保持只读，不训练 Y/Z。

本实验先处理真实故障：食指先伸开失触、小指随后失触，而拇指持续推动。只增加转角奖励会继续强化这一行为。

## 课程与奖励

1. 先学接触恢复。80% 训练初态由旧策略实际运行 0.8 秒到达，物理状态不注入、不瞬移；其余 20% 从原始初握学习保持。辅助目标是回到该已到达姿态并五指抓稳 1 秒。辅助成功不是 X90 成功。
2. 每 30 分钟固定验证：两种步长各 20 个补位初态，另做原始初握 X90 探测。补位在两种步长下均达到 80%，且连续两次通过，才开始完整轨迹搜索。
3. 搜索上限 120 秒、48 个候选。候选必须从原始初握真实执行，并在两种步长都完成 X90 严格目标，才能作为示范。没有合格路径则保持补位阶段，保存诊断，不宣称机构不可能或任务成功。
4. 有合格路径后训练原始初握 X90，保留 15% 补位混合和小权重的合格轨迹模仿。正式评价始终从原始初握开始，没有前缀。连续两次达到 90% 再做最终独立验收。

奖励强化缺失手指靠近和恢复接触；缺指时暂停正向旋转奖励，降低拇指推进附加奖励；保持原有反向惩罚、失败条件和接触力上限。目标增量仍为每周期有效量程 2%。继承旧策略特征与动作网络，动作头初始缩小至 0.5，采用标准差 0.08、上限 0.12 的探索，避免持续饱和推动。

4 个 CPU 环境，步长分配为 1/2/1/2 ms，控制周期 50 ms。PPO rollout 256、batch 128、4 epochs、学习率 1e-4。V2 首次迁移重建优化器与奖励归一化；此后 V2 断点恢复保留优化器和归一化，不重置时间预算。

## 当前实验与命令

实验目录：`artifacts/recovery_20261117_v2/`。24 小时包含准备、训练和评估，截止 **2026-11-18 14:15:26 北京时间**。每 5 分钟保存，每 3 小时以额外初态审查；到时保存停止。监督进程最多重启失败的训练进程 3 次，均沿用同一截止时间。

在仓库根目录执行：

```bash
# 查看进度和独立评估（不要将训练样本中的五指占比当作成功率）。
cat experiments/cube_contact_recovery_v2/artifacts/recovery_20261117_v2/status.md

# 仅在当前监督进程停止后恢复；文件锁阻止重复启动。
/workspace/.venvs/aero-cube/bin/python -m experiments.cube_contact_recovery_v2.supervise \
  --root experiments/cube_contact_recovery_v2/artifacts/recovery_20261117_v2

# 独立评估 / 导出逐控制周期轨迹。必须使用 V2 入口。
/workspace/.venvs/aero-cube/bin/python -m experiments.cube_contact_recovery_v2.evaluate \
  --checkpoint experiments/cube_contact_recovery_v2/artifacts/recovery_20261117_v2/latest \
  --target 90 --episodes 20 --seed 5800000 --trace --output /tmp/x90_v2_evaluation.json

# 接触语义、原生/C 加速一致性及真实前缀测试。
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /workspace/.venvs/aero-cube/bin/python \
  -m unittest experiments.cube_contact_recovery_v2.tests.test_semantics -v
```

模型、奖励、训练和判断代码的指纹保存在检查点；修改后会拒绝无声明地加载。旧录像中拇指 0 N 的显示可能漏掉拇指根部，必须保留版本标注。V2 初始评估与后续评估分别存档，不能把修正造成的读数变化算作学习进步。
