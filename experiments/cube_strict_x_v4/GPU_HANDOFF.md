# 在 GPU 机器上继续训练 X 轴翻面

从 `checkpoints/x_latest` 接着训练 **30° → 60° → 90°**。每次从原始握姿向当前目标转动七电机右手中的 57 mm 魔方。训练使用四个原生 MuJoCo CPU 进程计算接触和动力学，PyTorch 在 CUDA 上执行策略推理与 PPO 更新。吞吐量也取决于 CPU 采样速度。此入口配有四个环境；上千只手的 GPU 物理仿真需要另行适配 MJX 或其他 GPU 物理后端。

## 交接包中的策略

| 包内目录 | 保存的训练步数 | 用途 |
|---|---:|---|
| `checkpoints/x_latest/` | 15,213,568 | 默认续训起点，含策略、优化器、奖励归一化状态、四个环境的在途回合以及三指补位示范。|
| `checkpoints/x_best/` | 12,984,320 | 已完成独立评估的 X 轴策略，可用作改进前的对照，或在不同输出目录发起训练。|
| `checkpoints/y_best/` | 589,824 | 历史 Y 轴策略，经原 Y 轴环境载入。此前回放接近 90°，五指稳握 1 秒的目标尚未通过。|

包内保留各阶段实验源码、机械手与魔方资产，以及原作者的转方块基准权重。旧实验报告中的步数和文件名可用于追溯结果。复现那些依赖早期指定权重的历史测试，需要相应的旧检查点。当前 X 轴续训所需的归一化状态、优化器、握姿和补位示范随检查点提供。

交接时，X 轴最佳策略在 20 个新初态上用 1 ms 和 2 ms 物理步长分别测试，30° 抓稳均为 **0/20**。首次三指受力违规前的转角中位数均为 **21.25°**，最高约 **21.57°**。两组的首个越界手指都是中指。可先观看[最佳样例](../../results/x/videos/best/best.mp4)和[典型失败](../../results/x/videos/typical_failure/typical_failure.mp4)，再对照[冻结评估报告](../../results/x/REPORT.md)。

## 1. 获取训练代码和权重

从 GitHub 获取时，在新机器上克隆训练分支及其两个仿真子模块，然后进入第 2 步。

```bash
git clone --branch feature/new-manipulation --recurse-submodules https://github.com/ccw19838-max/aero-hand-open.git
cd aero-hand-open
```

使用离线交接包时，把 `aero-hand-gpu.tar.gz` 下载到 GPU 机器并在该文件所在目录解包、校验。

```bash
tar -xzf aero-hand-gpu.tar.gz
cd aero-hand-open
sha256sum --quiet -c SHA256SUMS
```

最后一条命令正常完成时退出码为 0。后续命令均在此项目根目录执行，使用当前用户可写的解包位置；本地 C 接触计算组件会在第一次启动时编译。

## 2. 创建训练环境

以下以 **Ubuntu 22.04 x86_64、Python 3.10、NVIDIA GPU** 为例。训练端使用 Python 3.10.22、MuJoCo 3.3.6、Stable-Baselines3 2.6.0 和 PyTorch 2.6.0。先用 `nvidia-smi` 确认主机或容器能识别显卡。

```bash
nvidia-smi
sudo apt-get update
sudo apt-get install -y python3.10-venv build-essential libegl1 libgl1 ffmpeg fonts-noto-cjk
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu124 'torch==2.6.0+cu124'
python -m pip install -r experiments/cube_strict_x_v4/requirements-gpu.txt
python -m pip check
```

这里选用 CUDA 12.4 版 PyTorch，需要显卡架构及宿主驱动支持这套 wheel。其他硬件可先核对官方适配表。[PyTorch 历史版本与 CUDA 安装命令](https://pytorch.org/get-started/previous-versions/#v260)

在新机上执行实际的 CUDA 运算并编译 MuJoCo 接触组件。成功时会打印显卡名称、PyTorch 的 CUDA 版本以及 MuJoCo 版本。

```bash
python - <<'PY'
import mujoco
import torch
from experiments.cube_adaptive_60h.fast_contacts import library

assert torch.cuda.is_available(), "PyTorch 无法使用 CUDA"
x = torch.ones((32, 32), device="cuda")
assert (x @ x)[0, 0].item() == 32
library()
print(torch.cuda.get_device_name(), torch.__version__, torch.version.cuda, mujoco.__version__)
PY
```

## 3. 先跑一个完整更新

```bash
python -m experiments.cube_strict_x_v4.continue_training \
  --checkpoint checkpoints/x_latest --output runs/x_gpu \
  --device cuda --updates 1
```

一次更新采集 4 × 256 = 1,024 个控制步。从本包的 `x_latest` 首次运行并完整结束后，累计步数应为 **15,214,592**。查看新保存的 `runs/x_gpu/latest/metadata.json`。CPU 型号、指令集或数值库变化可能使旧在途回合逐步重放时出现数值差异。若程序明确报告该差异，可在同一命令末尾加 `--reset-episodes`，以原始握姿重新开始四个回合，同时继承策略、优化器与奖励归一化统计。

## 4. 继续训练

```bash
python -m experiments.cube_strict_x_v4.continue_training \
  --checkpoint checkpoints/x_latest --output runs/x_gpu \
  --device cuda --updates 1024
```

同一输出目录会从 `runs/x_gpu/latest` 恢复。这里的 1,024 是本次最多追加的 PPO 更新数，相当于 **1,048,576 个采样步**。新实验的训练预算由本次调用决定。输出保留最近更新的检查点和按严格验证结果选出的最佳检查点。压缩包中的三个起始检查点可一直用作只读对照。

每半小时在 1 ms 和 2 ms 物理步长下各评估 20 次，两种步长都连续两轮至少 16/20 才晋级。观察首次食指、中指或无名指受力越出 0.1–12 N 前的转角、五指抓稳时间及严格成功数。三个阶段均允许 ±5° 姿态误差，到位后还需低速并连续保持 1 秒。若连续四次验证停滞，每个阶段最多进行一次 10 分钟的补位搜索。90° 课程取得验证资格后，入口自动冻结策略，用两种步长各 100 个独立初态验收，结果保存到输出目录的 `final/evaluation.json`。
