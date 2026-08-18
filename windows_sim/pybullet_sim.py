#!/usr/bin/env python3
"""DiffPhysDrone 策略 → PyBullet 实时可视化桥接。

把训练好的策略（Model + checkpoint）接到 PyBullet 四旋翼上实时飞，
深度相机按训练环境复刻（64x48, fov_x_half_tan=0.53, cam_angle≈10°）。
桥接层只依赖 torch + numpy + pybullet，不需要 quadsim_cuda。

用法（Windows 或 WSL 均可）:
    # 1) WSL 侧先导出一份场景
    python3 windows_sim/scene_export.py --seed 0 [--spawn_z 2.5] --out windows_sim/scene.npz
    # 2) 跑桥接（Windows 上开 --gui 弹出实时 3D 窗口）
    python3 windows_sim/pybullet_sim.py --checkpoint checkpoint0004.pth --scene windows_sim/scene.npz --gui
    # 冒烟测试（无 GUI、可跳过相机）:
    python3 windows_sim/pybullet_sim.py --checkpoint checkpoint0004.pth --scene windows_sim/scene.npz \
        --no-camera --steps 20

Windows 侧说明:
    - 需 pip install pybullet torch；CPU 推理即可（模型小）。
    - 仓库在 WSL，Windows Python 直接读 UNC 路径（脚本会自动尝试
      \\\\wsl.localhost\\Ubuntu-22.04\\home\\fengrenhan\\DiffPhysDrone），
      或用环境变量 DIFFPHYSDRONE 指定，或把 model.py + checkpoint 拷到本目录。
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import pybullet as p
import pybullet_data


# ---------- 训练环境常量（与 eval_demo.py / env_cuda.py 对齐） ----------
W, H = 64, 48                  # 深度相机分辨率（W 列, H 行）
FOV_X_HALF_TAN = 0.53          # 水平半角正切
CAM_ANGLE_DEG = 10.0           # 相机俯仰下倾角（度）
CTL_DT = 1.0 / 15.0            # 控制周期 15Hz
G = 9.80665
CAM_NEAR, CAM_FAR = 0.3, 24.0
PHYS_DT = 1.0 / 240.0          # 物理子步
CTRL_EVERY = int(round(CTL_DT / PHYS_DT))   # 每控制步的子步数 = 16
MASS = 1.0
DRAG = 0.15                    # 线性阻尼（帮助稳定）
MAX_ACCEL = 25.0               # 期望加速度限幅（m/s^2）
YAW_GAIN = 0.4                 # 偏航力矩增益（physics 模式朝向目标用）
YAW_DAMP = 0.05                # 偏航角速度阻尼
GROUND_Z = -1.0                # 地面碰撞面高度（与训练 env 一致）
DRONE_R = 0.15                 # 无人机近似半径（障碍碰撞检测用）
DRONE_HALF_H = 0.06            # 无人机机身半高（贴地时质心最小高度 = GROUND_Z + 半高）

STATIC_BODIES = []             # 障碍物 + marker 的 body id（重置场景时删除重建）


def check_obstacle_collision(sc, p, r=DRONE_R):
    """kinematic 模式的可选障碍碰撞：位置 p（含机身半径 r）是否与任何障碍相交。"""
    for b in sc['balls']:
        cx, cy, cz, br = b
        if br > 0 and np.linalg.norm(p - np.array([cx, cy, cz])) < br + r:
            return True
    for c in sc['cyl']:
        cx, cy, cr = c
        if cr > 0 and np.linalg.norm(p[:2] - np.array([cx, cy])) < cr + r:
            return True
    for c in sc['cyl_h']:
        cx, cz, cr = c
        if cr > 0 and np.linalg.norm(p[[0, 2]] - np.array([cx, cz])) < cr + r:
            return True
    for v in sc['voxels']:
        cx, cy, cz, sx, sy, sz = v
        if min(sx, sy, sz) > 0 and np.all(np.abs(p - np.array([cx, cy, cz])) < np.array([sx, sy, sz]) + r):
            return True
    return False


def gen_scene(seed, n_batch=8, spawn_z=None, n_obs=40):
    """在桥接内生成障碍场景（复刻 env_cuda.reset 的生成结构，纯 torch，Windows 可用）。
    注：RNG 调用顺序与 env 不完全一致，布局分布一致但不必逐位相同。"""
    g = torch.Generator()
    g.manual_seed(seed)

    def rand(*s):
        return torch.rand(*s, generator=g)

    def randn(*s):
        return torch.randn(*s, generator=g)

    B = n_batch
    ball_w = torch.tensor([8., 18, 6, 0.2]); ball_b = torch.tensor([0., -9, -1, 0.4])
    voxel_w = torch.tensor([8., 18, 6, 0.1, 0.1, 0.1]); voxel_b = torch.tensor([0., -9, -1, 0.2, 0.2, 0.2])
    cyl_w = torch.tensor([8., 18, 0.35]); cyl_b = torch.tensor([0., -9, 0.05])
    cyl_h_w = torch.tensor([8., 6, 0.1]); cyl_h_b = torch.tensor([0., 0, 0.05])
    roof_add = torch.tensor([0., 0., 2.5, 1.5, 1.5, 1.5])

    balls = rand(B, n_obs, 4) * ball_w + ball_b
    voxels = rand(B, n_obs, 6) * voxel_w + voxel_b
    cyl = rand(B, n_obs, 3) * cyl_w + cyl_b
    cyl_h = rand(B, 2, 3) * cyl_h_w + cyl_h_b
    max_speed = 0.75 + 2.5 * rand(B, 1)
    scale0 = (max_speed - 0.5).clamp_min(1)
    roof = rand(B) < 0.5
    balls[~roof, :15, :2] = cyl[~roof, :15, :2]
    voxels[~roof, :15, :2] = cyl[~roof, 15:30, :2]
    balls[~roof, :15] = balls[~roof, :15] + roof_add[:4]
    voxels[~roof, :15] = voxels[~roof, :15] + roof_add
    balls[..., 0] = balls[..., 0].clamp(balls[..., 3] + 0.3 / scale0,
                                        8 - 0.3 / scale0 - balls[..., 3])
    voxels[..., 0] = voxels[..., 0].clamp(voxels[..., 3] + 0.3 / scale0,
                                          8 - 0.3 / scale0 - voxels[..., 3])
    cyl[..., 0] = cyl[..., 0].clamp(cyl[..., 2] + 0.3 / scale0,
                                    8 - 0.3 / scale0 - cyl[..., 2])
    cyl_h[..., 0] = cyl_h[..., 0].clamp(cyl_h[..., 2] + 0.3 / scale0,
                                        8 - 0.3 / scale0 - cyl_h[..., 2])
    voxels[roof, 0, 2] = voxels[roof, 0, 2] * 0.5 + 201
    voxels[roof, 0, 3:] = 200
    voxels[:, :, 1] *= (max_speed + 4) / scale0
    balls[:, :, 1] *= (max_speed + 4) / scale0
    cyl[:, :, 1] *= (max_speed + 4) / scale0
    voxels[..., 0] *= scale0
    balls[..., 0] *= scale0
    cyl[..., 0] *= scale0
    cyl_h[..., 0] *= scale0

    p_init = torch.tensor([[-1.5, -3., 1], [9.5, -3., 1], [-0.5, 1., 1], [8.5, 1., 1],
                           [0., 3., 1], [8., 3., 1], [-1., -1., 1], [9., -1., 1]], dtype=torch.float32)
    p_end = torch.tensor([[8., 3., 1], [0., 3., 1], [8., -1., 1], [0., -1., 1],
                          [8., -3., 1], [0., -3., 1], [8., 1., 1], [0., 1., 1]], dtype=torch.float32)
    scale = torch.cat([scale0, rand(B, 1) + 0.5, rand(B, 1) - 0.5], -1)  # (B,3)
    start = p_init[0] * scale[0] + randn(3) * 0.1
    target = p_end[0] * scale[0] + randn(3) * 0.1
    if spawn_z is not None:
        start[2] = spawn_z
        target[2] = spawn_z
    return {
        'balls': balls[0].numpy(), 'voxels': voxels[0].numpy(),
        'cyl': cyl[0].numpy(), 'cyl_h': cyl_h[0].numpy(),
        'start': start.numpy(), 'target': target.numpy(),
        'margin': np.array([float((rand(1) * 0.2 + 0.1).item())]),
        'max_speed': np.array([max_speed[0, 0].item()]),
    }


def set_view(start, target):
    """初始视角拉远，让起点与终点都在画面内（以两者中点为注视点，距离随场景跨度缩放）。"""
    center = (start + target) / 2.0
    span = float(np.linalg.norm(target - start))
    p.resetDebugVisualizerCamera(
        cameraDistance=float(np.clip(span * 1.5, 8.0, 30.0)),
        cameraYaw=-45, cameraPitch=-35,
        cameraTargetPosition=[float(center[0]), float(center[1]), float(center[2])])


def resolve_repo():
    """定位 DiffPhysDrone 仓库目录（model.py 所在）。"""
    if os.environ.get('DIFFPHYSDRONE'):
        return os.environ['DIFFPHYSDRONE']
    home = os.path.expanduser('~')
    cands = [
        os.path.join(home, 'DiffPhysDrone'),
        r'\\wsl.localhost\Ubuntu-22.04\home\fengrenhan\DiffPhysDrone',
    ]
    for c in cands:
        if os.path.isfile(os.path.join(c, 'model.py')):
            return c
    return cands[0]


REPO = resolve_repo()
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from model import Model  # noqa: E402


def quat_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def yaw_quat_from_dir(fwd_h):
    """水平前方向量 -> 偏航四元数（机体 x 轴指向该方向，水平姿态）。"""
    fx, fy = float(fwd_h[0]), float(fwd_h[1])
    yaw = math.atan2(fy, fx)
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def compute_projection():
    """按训练相机模型构造透视投影矩阵（垂直 fov 由水平 fov + 纵横比推出）。"""
    fov_vert = 2 * math.degrees(math.atan(FOV_X_HALF_TAN * H / W))
    return p.computeProjectionMatrixFOV(fov=fov_vert, aspect=W / H,
                                        nearVal=CAM_NEAR, farVal=CAM_FAR)


def depth_to_distance(depth):
    """pybullet 深度缓冲 [0,1] -> 沿视线距离 (H,W)，透视重建。"""
    buf = 2.0 * depth - 1.0
    z = 2.0 * CAM_NEAR * CAM_FAR / (CAM_FAR + CAM_NEAR - buf * (CAM_FAR - CAM_NEAR))
    return np.asarray(z, dtype=np.float32)


def build_obstacles(sc):
    """建/重建障碍物与起终点 marker（body id 记入 STATIC_BODIES，供重置时清理）。"""
    # 球（roof 大球包含在内，半径最大 ~2.1m；跳过退化 r<=0）
    for b in sc['balls']:
        cx, cy, cz, r = b
        if r <= 0:
            continue
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=float(r))
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=float(r),
                                  rgbaColor=[0.55, 0.55, 0.6, 0.9])
        STATIC_BODIES.append(p.createMultiBody(0, col, vis, basePosition=[cx, cy, cz]))

    # 柱（[cx,cy,r]，沿 z 无限，取高 12m 覆盖飞行区）
    for c in sc['cyl']:
        cx, cy, r = c
        if r <= 0:
            continue
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=float(r), height=12.0)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=float(r), length=12.0,
                                  rgbaColor=[0.6, 0.6, 0.5, 0.9])
        STATIC_BODIES.append(p.createMultiBody(0, col, vis, basePosition=[cx, cy, 3.0]))

    # 水平柱（[cx,cz,r]，沿 y 无限，取长 24m）
    for c in sc['cyl_h']:
        cx, cz, r = c
        if r <= 0:
            continue
        q = p.getQuaternionFromEuler([math.pi / 2, 0, 0])  # 圆柱轴转成沿 y
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=float(r), height=24.0)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=float(r), length=24.0,
                                  rgbaColor=[0.5, 0.6, 0.6, 0.9])
        STATIC_BODIES.append(p.createMultiBody(0, col, vis,
                                               basePosition=[cx, 0, cz], baseOrientation=q))

    # 盒子（[cx,cy,cz,sx,sy,sz]）
    for v in sc['voxels']:
        cx, cy, cz, sx, sy, sz = v
        if min(sx, sy, sz) <= 0:
            continue
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[sx, sy, sz])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[sx, sy, sz],
                                  rgbaColor=[0.6, 0.5, 0.5, 0.9])
        STATIC_BODIES.append(p.createMultiBody(0, col, vis, basePosition=[cx, cy, cz]))

    # 起终点 marker（无碰撞形状用 -1；返回目标 marker 的 body id 供移动目标使用）
    tgt_marker = p.createMultiBody(0, -1, p.createVisualShape(
        p.GEOM_SPHERE, radius=0.25, rgbaColor=[1, 0.2, 0.2, 1]),
        basePosition=sc['target'].tolist())
    STATIC_BODIES.append(tgt_marker)
    STATIC_BODIES.append(p.createMultiBody(0, -1, p.createVisualShape(
        p.GEOM_SPHERE, radius=0.25, rgbaColor=[0.2, 1, 0.2, 1]),
        basePosition=sc['start'].tolist()))
    return tgt_marker


def build_world(sc, gui, show_ground=False):
    """建 PyBullet 世界：地面（常驻，不随重置删除）+ 障碍/marker + 无人机。返回无人机 id。"""
    p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -G)
    p.setTimeStep(PHYS_DT)
    # 地面：碰撞面在 z=-1（与训练 env 的地面一致）
    ground_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[50, 50, 0.5])
    p.createMultiBody(0, ground_col, -1, basePosition=[0, 0, -1.5])  # 顶面 z=-1
    # 可见地面薄片（外观；会出现在深度相机里，可能轻微影响策略——默认关）
    if show_ground:
        ground_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[50, 50, 0.03],
                                         rgbaColor=[0.55, 0.6, 0.5, 1])
        p.createMultiBody(0, -1, ground_vis, basePosition=[0, 0, -1.03])
    tgt_marker = build_obstacles(sc)

    # 无人机：单刚体箱体（质心施力 + 展示朝向）
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.18, 0.18, 0.06])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.18, 0.18, 0.06],
                              rgbaColor=[0.1, 0.4, 0.9, 1])
    drone = p.createMultiBody(MASS, col, vis, basePosition=sc['start'].tolist())
    p.resetBaseVelocity(drone, [0, 0, 0])
    return drone, tgt_marker


def reset_world(sc, drone):
    """删除旧障碍/marker，用新场景重建，并把无人机复位到新起点。返回新目标 marker id。"""
    for bid in STATIC_BODIES:
        p.removeBody(bid)
    STATIC_BODIES.clear()
    tgt_marker = build_obstacles(sc)
    start, target = sc['start'], sc['target']
    p.resetBasePositionAndOrientation(drone, start.tolist(),
                                      list(yaw_quat_from_dir(target - start)))
    p.resetBaseVelocity(drone, [0, 0, 0])
    return tgt_marker


def render_depth(drone, flip='both'):
    """从无人机机体相机取深度图，返回 (dist (H,W), pos, R)。

    flip 控制深度图像方向（训练 env 的 canvas 约定未知，可尝试对齐）:
      none=仅转置  ud=上下翻转  lr=左右翻转  both=上下+左右
    """
    pos, quat = p.getBasePositionAndOrientation(drone)
    R = quat_to_R(quat)
    fwd, up = R[:, 0], R[:, 2]
    a = math.radians(CAM_ANGLE_DEG)
    cam_fwd = fwd * math.cos(a) - up * math.sin(a)          # 前视下倾 cam_angle
    eye = np.array(pos, dtype=np.float64)
    view = p.computeViewMatrix(eye.tolist(), (eye + cam_fwd).tolist(), up.tolist())
    proj = compute_projection()
    _, _, _, depth_buf, _ = p.getCameraImage(
        W, H, viewMatrix=view, projectionMatrix=proj,
        flags=p.ER_NO_SEGMENTATION_MASK)
    if depth_buf is None:
        return None, eye, R
    img = np.asarray(depth_buf)
    if img.ndim == 3:                                       # 有的渲染器返回 (W,H,1)
        img = img[..., 0]
    if img.ndim == 1:                                       # GUI 模式返回一维扁平数组（行优先 H,W）
        img = img.reshape(H, W)
    if img.shape == (W, H):
        img = img.T                                         # (W,H) -> (H,W)
    elif img.shape != (H, W):
        raise ValueError(f"unexpected depth shape {img.shape}")
    if flip in ('ud', 'both'):
        img = np.flipud(img)
    if flip in ('lr', 'both'):
        img = np.fliplr(img)
    return depth_to_distance(img), eye, R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=os.path.join(REPO, 'checkpoint0004.pth'))
    ap.add_argument('--scene', default=None, help='加载场景 npz（不指定则用 --seed 生成，默认每次随机）')
    ap.add_argument('--seed', type=int, default=None, help='场景随机种子（缺省=每次随机）')
    ap.add_argument('--spawn_z', type=float, default=None, help='出生/目标高度 z（生成场景时生效）')
    ap.add_argument('--gui', action='store_true', help='GUI 实时窗口（默认 DIRECT 无头）')
    ap.add_argument('--mode', choices=['physics', 'kinematic'], default='kinematic',
                    help='kinematic=运动学积分（最稳，实测到达目标）；physics=牛顿力控（实验性）')
    ap.add_argument('--steps', type=int, default=600, help='控制步数（15Hz，600≈40s）')
    ap.add_argument('--no-camera', action='store_true', help='跳过相机（冒烟测试用）')
    ap.add_argument('--flip', choices=['none', 'ud', 'lr', 'both'], default='ud',
                    help='深度图像方向（WSL 无头用 lr；Windows 用 ud——用 find_flip.py 探测）')
    ap.add_argument('--speed', type=float, default=1.0, help='速度缩放')
    ap.add_argument('--target_speed', type=float, default=0.0,
                    help='目标水平移动速度（m/s，0=静止目标；>0 则在场景边界内匀速移动并反弹）')
    ap.add_argument('--solid', action='store_true',
                    help='kinematic 模式下障碍物也视为实体（撞到即停；默认只拦地面）')
    ap.add_argument('--ground', action='store_true',
                    help='显示可见地面薄片（默认关，避免深度相机拍到地面干扰策略）')
    args = ap.parse_args()

    # ---- 模型 ----
    sd = torch.load(args.checkpoint, map_location='cpu')
    dim_obs = sd['v_proj.weight'].shape[1]
    model = Model(dim_obs, 6).eval()
    model.load_state_dict(sd, strict=False)
    print(f"model dim_obs={dim_obs}（with odom）")

    # ---- 场景（指定 --scene 则加载 npz；否则按 --seed 生成，缺省每次随机）----
    if args.scene:
        sc = np.load(args.scene)
    else:
        seed = args.seed if args.seed is not None else int(time.time() * 1000) % (2 ** 31)
        sc = gen_scene(seed, spawn_z=args.spawn_z)
        print(f"generated scene (seed={seed})")
    start, target = sc['start'], sc['target']
    margin = float(sc['margin'][0])
    max_speed = float(sc['max_speed'][0]) * args.speed
    print(f"scene: start={np.round(start,2).tolist()} target={np.round(target,2).tolist()} "
          f"margin={margin:.3f} max_speed={max_speed:.2f}")

    drone, target_marker = build_world(sc, args.gui, args.ground)
    p.resetBasePositionAndOrientation(drone, start.tolist(),
                                      list(yaw_quat_from_dir(target - start)))

    if args.gui:
        set_view(start, target)

    # 目标水平运动：--target_speed>0 时在场景边界内匀速移动并反弹（保持可见）
    target_vel = np.zeros(3)
    if args.target_speed > 0:
        th = np.random.uniform(0, 2 * math.pi)
        target_vel = args.target_speed * np.array([np.cos(th), np.sin(th), 0.0])
    TX_MIN, TX_MAX, TY_MIN, TY_MAX = -1.5, 9.5, -4.0, 4.0

    g_std = torch.tensor([0.0, 0.0, -G])
    h = None
    prev = time.time()
    vel = np.zeros(3, dtype=np.float64)   # kinematic 模式用 Python 维护速度

    for step in range(args.steps):
        # ---- 键盘：R 键重置障碍场景 ----
        if args.gui:
            _keys = p.getKeyboardEvents()
            if _keys.get(ord('r'), 0) & p.KEY_WAS_TRIGGERED:
                seed2 = int(time.time() * 1000) % (2 ** 31)
                sc = gen_scene(seed2, spawn_z=args.spawn_z)
                start, target = sc['start'], sc['target']
                margin = float(sc['margin'][0])
                max_speed = float(sc['max_speed'][0]) * args.speed
                target_marker = reset_world(sc, drone)
                h = None
                vel[:] = 0.0
                if args.target_speed > 0:
                    th = np.random.uniform(0, 2 * math.pi)
                    target_vel = args.target_speed * np.array([np.cos(th), np.sin(th), 0.0])
                set_view(start, target)
                prev = time.time()
                print(f"[重置] 新场景 seed={seed2} "
                      f"start={np.round(start,2).tolist()} target={np.round(target,2).tolist()}")

        # ---- 目标移动（水平匀速 + 边界反弹）----
        if args.target_speed > 0:
            target = target + target_vel * CTL_DT
            if target[0] < TX_MIN or target[0] > TX_MAX:
                target_vel[0] *= -1.0
                target[0] = np.clip(target[0], TX_MIN, TX_MAX)
            if target[1] < TY_MIN or target[1] > TY_MAX:
                target_vel[1] *= -1.0
                target[1] = np.clip(target[1], TY_MIN, TY_MAX)
            p.resetBasePositionAndOrientation(target_marker, target.tolist(), [0, 0, 0, 1])

        # ---- 感知 ----
        pos, quat = p.getBasePositionAndOrientation(drone)
        R = quat_to_R(quat)
        pos = np.asarray(pos, dtype=np.float64)
        if args.mode == 'physics':
            vel = np.asarray(p.getBaseVelocity(drone)[0], dtype=np.float64)

        if args.no_camera:
            dist = np.full((H, W), 5.0, dtype=np.float32)   # 恒定 5m，冒烟测试
        else:
            dist, pos, R = render_depth(drone, args.flip)
            if dist is None:
                print("警告: 无深度（DIRECT 模式可能不支持相机），改用恒定深度")
                dist = np.full((H, W), 5.0, dtype=np.float32)

        x = torch.from_numpy(dist)[None, None]              # (1,1,H,W)
        x = 3.0 / x.clamp(CAM_NEAR, CAM_FAR) - 0.6
        x = F.max_pool2d(x, 4, 4)                           # (1,1,12,16)

        # 状态向量（与 eval_demo.py 对齐：local_v, target_v_body, up_world, margin）
        tdir = target - pos
        tnorm = np.linalg.norm(tdir)
        tv_world = tdir / max(tnorm, 1e-6) * min(tnorm, max_speed)
        local_v = R.T @ vel
        tv_body = R.T @ tv_world
        up_world = R[:, 2]
        state = torch.tensor(np.concatenate([local_v, tv_body, up_world, [margin]]),
                             dtype=torch.float32)[None]

        # ---- 策略 ----
        with torch.no_grad():
            act, _, h = model(x, state, h)                  # (1,6)
        R_t = torch.from_numpy(R).float()[None]             # (1,3,3)
        a_pred, v_pred = (R_t @ act.reshape(1, 3, 2)).unbind(-1)
        act_cmd = (a_pred - v_pred - g_std) + g_std         # 世界系期望加速度 (1,3)
        a = np.clip(act_cmd[0].numpy(), -MAX_ACCEL, MAX_ACCEL)
        if step < 3:
            print(f"  diag a_pred={[round(v,2) for v in a_pred[0].tolist()]} "
                  f"act_cmd={[round(v,2) for v in a.tolist()]} "
                  f"depth[min/mean/max]={dist.min():.2f}/{dist.mean():.2f}/{dist.max():.2f} "
                  f"state={[round(v,2) for v in state[0].tolist()]}")

        # ---- 控制 ----
        yawq = yaw_quat_from_dir(target - pos)
        if args.mode == 'physics':
            # 每个物理子步都施力（重力补偿使净加速度≈a + 线性阻尼）；
            # 偏航力矩朝目标（不 reset，避免破坏速度累积）
            cur_yaw = math.atan2(2.0 * (quat[3] * quat[2] + quat[0] * quat[1]),
                                 1.0 - 2.0 * (quat[1] * quat[1] + quat[2] * quat[2]))
            des_yaw = math.atan2(target[1] - pos[1], target[0] - pos[0])
            yerr = (des_yaw - cur_yaw + math.pi) % (2 * math.pi) - math.pi
            w_z = p.getBaseVelocity(drone)[1][2]
            f_ext = (MASS * (a + [0, 0, G])).tolist()
            f_drag = (-DRAG * vel).tolist()
            for _ in range(CTRL_EVERY):
                p.applyExternalForce(drone, -1, f_ext, pos.tolist(), p.WORLD_FRAME)
                p.applyExternalForce(drone, -1, f_drag, pos.tolist(), p.WORLD_FRAME)
                p.applyExternalTorque(drone, -1, [0, 0, YAW_GAIN * yerr - YAW_DAMP * w_z],
                                      p.WORLD_FRAME)
                p.stepSimulation()
        else:
            # 运动学积分：速度在 Python 维护，不读 pybullet（避免被覆盖）
            vel = vel + a * CTL_DT
            newpos = pos + vel * CTL_DT
            # 地面碰撞（始终生效）：质心不低于 地面 + 机身半高，防穿地
            if newpos[2] < GROUND_Z + DRONE_HALF_H:
                newpos[2] = GROUND_Z + DRONE_HALF_H
                vel[2] = 0.0
            # 障碍碰撞（--solid 可选）：撞到即停
            if args.solid and check_obstacle_collision(sc, newpos, DRONE_R):
                vel = np.zeros(3)
            else:
                pos = newpos
            p.resetBasePositionAndOrientation(drone, pos.tolist(), list(yawq))
            p.stepSimulation()

        # 实时节奏（GUI 模式）
        if args.gui:
            el = time.time() - prev
            wait = CTL_DT - el
            if wait > 0:
                time.sleep(wait)
            prev = time.time()

        if step % 15 == 0:
            d2t = np.linalg.norm(pos - target)
            print(f"step {step:4d}  pos=({pos[0]:6.2f},{pos[1]:6.2f},{pos[2]:5.2f}) "
                  f"speed={np.linalg.norm(vel):5.2f} a=({a[0]:5.2f},{a[1]:5.2f},{a[2]:5.2f}) "
                  f"dist_target={d2t:5.2f} tgt=({target[0]:5.2f},{target[1]:5.2f},{target[2]:5.2f})")

    print("done")

    if args.gui:
        time.sleep(1)


if __name__ == '__main__':
    main()
