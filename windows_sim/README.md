# DiffPhysDrone → PyBullet 实时可视化桥接

把训练好的策略（`Model` + checkpoint）接到 PyBullet 四旋翼上实时飞，做可视化演示。
桥接层只依赖 **torch + numpy + pybullet**（CPU 即可），不需要 `quadsim_cuda`。

## 文件

| 文件 | 作用 |
|---|---|
| `pybullet_sim.py` | 主桥接：加载模型+场景 → PyBullet 建世界 → 15Hz 闭环（深度→状态→策略→控制） |
| `scene_export.py` | 在 WSL 侧用真实 `env_cuda` 导出 40 障碍场景 npz |
| `scene.npz` | 已生成：默认场景（seed 0，z≈0） |
| `scene_highz.npz` | 已生成：高空场景（seed 0，`--spawn_z 2.5`） |

## 用法

### 1) WSL 侧导出场景（可选，已有默认场景可跳过）

```bash
cd ~/DiffPhysDrone
python3 windows_sim/scene_export.py --seed 0 --out windows_sim/scene.npz
python3 windows_sim/scene_export.py --seed 0 --spawn_z 2.5 --out windows_sim/scene_highz.npz
```

### 2) Windows 侧跑实时可视化（GUI 窗口）

```bash
# 装依赖（Windows Python 3.10+）
pip install pybullet torch numpy

# 跑（仓库在 WSL，脚本自动通过 UNC 路径 \\wsl.localhost\... 读 model.py/checkpoint）
cd <本目录，或任意>
python pybullet_sim.py --gui --steps 600            # 每次随机新障碍
python pybullet_sim.py --gui --steps 600 --seed 7   # 固定布局（复现）
python pybullet_sim.py --gui --steps 600 --seed 7 --spawn_z 2.5   # 高空场景
python pybullet_sim.py --gui --steps 600 --speed 3  # 提速到 3m/s+
python pybullet_sim.py --gui --steps 600 --target_speed 1.5  # 追逐水平移动目标
```

GUI 内按 **R** 键随时重置障碍物。若 UNC 路径没自动解析到，用 `DIFFPHYSDRONE` 环境变量指定仓库位置，或把 `model.py`、`checkpoint0004.pth` 拷到本目录再跑。

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--checkpoint` | `checkpoint0004.pth` | 权重路径 |
| `--scene` | 无 | 加载场景 npz（不指定则用 `--seed` 生成，**默认每次随机**） |
| `--seed` | 无 | 场景随机种子（固定后每次一致，用于复现） |
| `--spawn_z` | 无 | 出生/目标高度 z（生成场景时生效，如 2.5 高空） |
| `--gui` | 关 | 开 GUI 实时 3D 窗口（Windows 演示必加）；否则 DIRECT 无头 |
| `--mode` | `kinematic` | `kinematic`=运动学积分（最稳，实测到达目标）；`physics`=牛顿力控（实验性，导航不稳） |
| `--steps` | 600 | 控制步数（15Hz，600≈40s） |
| `--flip` | `ud` | 深度图像方向：WSL 无头用 `lr`；Windows 用 `ud`。用 `find_flip.py` 一键探测（跑 4 种各 150 步，dist_target 最小的即正确） |
| `--no-camera` | 关 | 跳过相机（恒定深度，冒烟测试用） |
| `--speed` | 1.0 | 目标速度缩放（×max_speed） |
| `--target_speed` | 0 | 目标水平移动速度（m/s，0=静止；>0 则匀速水平移动并在场景边界反弹，无人机追逐） |
| `--solid` | 关 | kinematic 模式下障碍物也视为实体（撞到即停）；默认只拦地面不穿地 |
| `--ground` | 关 | 显示可见地面薄片（默认关，避免深度相机拍到地面干扰策略） |

**GUI 内快捷键**：按 `R` 键即时重置场景——随机生成新障碍布局、无人机复位到新起点、视角自动拉远（无需重启程序）。

## 实测效果（无头 DIRECT 验证）

- 真实 40 障碍场景（kinematic + flip=lr）：从起点穿过障碍带，**到达目标 ~0.65m**，随后在目标附近盘旋。
- 高空场景（spawn_z=2.5，kinematic）：同样到达目标 ~1.05m。
- 无障碍空场景对模型是训练外分布（深度全远），指令会奇怪，属正常。

## 已知限制与调参

1. **sim-to-sim 分布偏移**：模型在可微物理仿真训练，PyBullet 深度/动力学不同，性能会下降；这本身是迁移观察点。
1. **kinematic 模式的地面碰撞**：默认**始终拦截地面**（机身不穿地，质心 z ≥ -0.94）；障碍实体需 `--solid`（撞到即停，导航可能绕行）。`physics` 模式有 PyBullet 原生碰撞。
2. **深度方向是关键**：训练 env 的 canvas 约定未知，实测 `--flip lr` 最优（上下翻转会卡死）。若换场景仍不顺，可试 `none`。
3. **kinematic 模式不参与碰撞响应**（飞过障碍是视觉穿过）；`physics` 模式有碰撞但导航不稳，需进一步调偏航增益 `YAW_GAIN`/阻尼 `DRAG`。
4. **相机模型**：按训练复刻（64×48、fov_x_half_tan=0.53、cam_angle=10°）；PyBullet 深度缓冲经透视重建转成沿视线距离。
5. **无地面入镜**：地面碰撞面在 z=-1（与训练一致），无可见形状，避免把地面当"近墙"干扰策略。
6. `--speed` 只缩放目标速度（max_speed）；场景里某 seed 的无人机可能偏慢（如 1.01 m/s），换 `--seed` 导出新场景可得到更快无人机。
