#!/usr/bin/env python3
"""导出 40 障碍场景 npz（在 WSL 侧运行，用真实 env_cuda 保证场景与训练/评估一致）。

用法:
    python3 windows_sim/scene_export.py [--seed 0] [--spawn_z 2.5] [--out windows_sim/scene.npz]

生成的 scene.npz 含: balls(40,4) voxels(40,6) cyl(40,3) cyl_h(2,3)
                       start(3) target(3) margin(1) max_speed(1)
供 PyBullet 桥接（pybullet_sim.py）加载重建世界。
"""
import argparse
import os
import sys

import numpy as np
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # windows_sim/.. = 仓库根
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=0, help='随机种子（决定障碍布局）')
    ap.add_argument('--spawn_z', type=float, default=None, help='出生/目标高度（世界坐标 z），如 2.5')
    ap.add_argument('--out', default='windows_sim/scene.npz')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    from env_cuda import Env  # 仅 WSL 侧可 import（依赖 quadsim_cuda）
    env = Env(8, 64, 48, 0.01, device='cuda', single=True, spawn_z=args.spawn_z)
    env.reset()

    np.savez_compressed(
        args.out,
        balls=env.balls[0].cpu().numpy(),        # (40,4) [cx,cy,cz,r]
        voxels=env.voxels[0].cpu().numpy(),      # (40,6) [cx,cy,cz,sx,sy,sz]
        cyl=env.cyl[0].cpu().numpy(),            # (40,3) [cx,cy,r] 沿 z 无限柱
        cyl_h=env.cyl_h[0].cpu().numpy(),        # (2,3)  [cx,cz,r] 沿 y 无限柱
        start=env.p[0].cpu().numpy(),            # (3,)
        target=env.p_target[0].cpu().numpy(),    # (3,)
        margin=np.array([env.margin[0].item()]),
        max_speed=np.array([env.max_speed[0].item()]),
    )
    print(f"saved {args.out}")
    print(f"  start   = {[round(x,2) for x in env.p[0].tolist()]}")
    print(f"  target  = {[round(x,2) for x in env.p_target[0].tolist()]}")
    print(f"  margin  = {env.margin[0].item():.3f}   max_speed = {env.max_speed[0].item():.2f} m/s")
    print(f"  balls   = {tuple(env.balls.shape)}   voxels = {tuple(env.voxels.shape)}"
          f"   cyl = {tuple(env.cyl.shape)}   cyl_h = {tuple(env.cyl_h.shape)}")


if __name__ == '__main__':
    main()
