#!/usr/bin/env python3
"""
高角度 3D 飞行渲染器（自包含，仅依赖 numpy/matplotlib/Pillow）。
用法 A（独立运行，读 npz 缓存）：
    python3 render_high_view.py flight_data.npz -o flight_high.gif --elev 70 --mode follow --spin 0.6
用法 B（被 eval_demo.py 调用）：
    from render_high_view import render_high_angle_gif
npz 约定：pos (T,3) 轨迹；obstacles (K,4) [cx,cy,cz,r]；target (3,)；可选 voxels (N,3)。
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
def _np(x):
    if hasattr(x, "detach"):          # torch tensor（可能在 GPU 上）
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float64)
def _sphere(cx, cy, cz, r, n=16):
    u = np.linspace(0, 2 * np.pi, n)
    v = np.linspace(0, np.pi, max(n // 2, 4))
    return (cx + r * np.outer(np.cos(u), np.sin(v)),
            cy + r * np.outer(np.sin(u), np.sin(v)),
            cz + r * np.outer(np.ones_like(u), np.cos(v)))
def render_high_angle_gif(pos, obstacles, target, out="flight_high.gif",
                          elev=70, azim=-60, cam_mode="follow", margin=8.0,
                          fps=20, trail=25, spin=0.0, voxels=None, start=None):
    pos = _np(pos).reshape(-1, 3)
    T = len(pos)
    target = _np(target).reshape(3)
    if start is None:
        start = pos[0]
    else:
        start = _np(start).reshape(3)
    obs = []
    for o in obstacles:
        if isinstance(o, (tuple, list)) and len(o) == 2:
            c = _np(o[0]).reshape(3)
            r = float(o[1])
        else:
            o = _np(o).reshape(-1)
            c = o[:3]
            r = float(o[3])
        obs.append((c, r))
    fig = plt.figure(figsize=(9, 9), dpi=110)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_axis_off()
    ax.xaxis.pane.set_facecolor("0.97")
    ax.yaxis.pane.set_facecolor("0.97")
    ax.zaxis.pane.set_facecolor("0.97")
    ax.set_box_aspect((1, 1, 0.55))
    allpts = np.vstack([pos, target[None, :], start[None, :]]
                       + [c[None, :] for c, _ in obs])
    xmin, xmax = allpts[:, 0].min(), allpts[:, 0].max()
    ymin, ymax = allpts[:, 1].min(), allpts[:, 1].max()
    zmin, zmax = allpts[:, 2].min(), allpts[:, 2].max()
    pad = max(2.0, 0.1 * max(xmax - xmin, ymax - ymin))
    # ---- 静态场景 ----
    for c, r in obs:
        X, Y, Z = _sphere(c[0], c[1], c[2], r)
        ax.plot_surface(X, Y, Z, color="0.35", alpha=0.9,
                        rstride=1, cstride=1, linewidth=0, antialiased=False)
    if voxels is not None:
        v = _np(voxels).reshape(-1, 3)
        ax.scatter(v[:, 0], v[:, 1], v[:, 2], c="0.6", s=5, alpha=0.45, depthshade=False)
    gx = np.linspace(xmin - pad, xmax + pad, 2)
    gy = np.linspace(ymin - pad, ymax + pad, 2)
    gX, gY = np.meshgrid(gx, gy)
    ax.plot_surface(gX, gY, np.zeros_like(gX), color="0.85", alpha=0.25,
                    rstride=1, cstride=1, linewidth=0)
    ax.plot([target[0]], [target[1]], [target[2]], "s", color="tab:red",
            ms=11, zorder=9, label="target")
    ax.plot([target[0], target[0]], [target[1], target[1]], [0, target[2]],
            "--", color="tab:red", lw=1.2, alpha=0.7)
    ax.plot([start[0]], [start[1]], [start[2]], "o", color="tab:green",
            ms=9, zorder=9, label="start")
    # ---- 动态元素 ----
    traj_line, = ax.plot([], [], [], color="tab:orange", lw=2.2, alpha=0.95)
    trail_line, = ax.plot([], [], [], color="tab:blue", lw=1.4, alpha=0.5)
    head_line, = ax.plot([], [], [], color="tab:red", lw=1.6)
    drone_dot, = ax.plot([], [], [], "o", color="tab:blue", ms=8, zorder=10)
    # 机头朝向由速度推出（无需额外记录 yaw）
    vel = np.diff(pos, axis=0)
    heads = np.zeros((T, 2)); heads[:] = [1.0, 0.0]
    for i in range(1, T):
        v = vel[i - 1]
        if np.linalg.norm(v[:2]) > 1e-6:
            heads[i] = v[:2] / np.linalg.norm(v[:2])
        else:
            heads[i] = heads[i - 1]
    def set_lims(p):
        if cam_mode == "follow":
            ax.set_xlim3d(p[0] - margin, p[0] + margin)
            ax.set_ylim3d(p[1] - margin, p[1] + margin)
            ax.set_zlim3d(0, max(zmax + 1.5, p[2] + margin * 0.6))
        else:
            ax.set_xlim3d(xmin - pad, xmax + pad)
            ax.set_ylim3d(ymin - pad, ymax + pad)
            ax.set_zlim3d(max(0, zmin - 1), zmax + 1.5)
    set_lims(pos[0])
    def update(t):
        p = pos[t]
        traj_line.set_data(pos[:t + 1, 0], pos[:t + 1, 1])
        traj_line.set_3d_properties(pos[:t + 1, 2])
        i0 = max(0, t - trail)
        trail_line.set_data(pos[i0:t + 1, 0], pos[i0:t + 1, 1])
        trail_line.set_3d_properties(pos[i0:t + 1, 2])
        hx, hy = heads[t]
        L = 0.6
        head_line.set_data([p[0], p[0] + hx * L], [p[1], p[1] + hy * L])
        head_line.set_3d_properties([p[2], p[2]])
        drone_dot.set_data([p[0]], [p[1]])
        drone_dot.set_3d_properties([p[2]])
        set_lims(p)
        ax.view_init(elev=elev, azim=azim + t * spin)
        ax.set_title(f"t = {t + 1}/{T}  elev={elev}  azim={azim + t * spin:.0f}", fontsize=10)
        return traj_line, trail_line, head_line, drone_dot
    ani = FuncAnimation(fig, update, frames=T, interval=1000 / fps, blit=False)
    if out.endswith(".mp4"):
        writer = FFMpegWriter(fps=fps, bitrate=2400)
        ani.save(out, writer=writer, dpi=110)
    else:
        writer = PillowWriter(fps=fps)
        ani.save(out, writer=writer, dpi=110)
    plt.close(fig)
    print(f"[render] saved {out}  ({T} frames, {fps} fps, mode={cam_mode}, "
          f"elev={elev}, azim={azim}, spin={spin}/frame)")
    return out
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="高角度 3D 飞行 GIF/MP4 渲染器")
    ap.add_argument("npz", help="flight_data.npz：pos/obstacles/target[/voxels]")
    ap.add_argument("-o", "--out", default="flight_high.gif")
    ap.add_argument("--elev", type=float, default=70, help="俯仰角（90=正上方）")
    ap.add_argument("--azim", type=float, default=-60, help="方位角")
    ap.add_argument("--mode", choices=["follow", "fixed"], default="follow",
                    help="follow=无人机居中跟拍；fixed=全景固定机位")
    ap.add_argument("--margin", type=float, default=8.0, help="follow 模式取景半宽（米）")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--trail", type=int, default=25, help="尾迹步数")
    ap.add_argument("--spin", type=float, default=0.0, help="每帧方位角旋转度数（全景镜头用 0.5~1 很有电影感）")
    args = ap.parse_args()
    d = np.load(args.npz)
    def get(k):
        return d[k] if k in d.files else None
    render_high_angle_gif(
        pos=d["pos"], obstacles=d["obstacles"], target=d["target"],
        out=args.out, elev=args.elev, azim=args.azim, cam_mode=args.mode,
        margin=args.margin, fps=args.fps, trail=args.trail, spin=args.spin,
        voxels=get("voxels"), start=get("start"))
