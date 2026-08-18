#!/usr/bin/env python3
"""跑 4 种深度翻转（无头 DIRECT），各 150 步，报告最终 dist_target。
dist_target 最小的翻转 = 深度方向最正确（越小越接近目标）。"""
import subprocess
import sys

for fl in ['none', 'ud', 'lr', 'both']:
    print(f'=== flip={fl} ===', flush=True)
    r = subprocess.run(
        [sys.executable, 'pybullet_sim.py', '--steps', '150', '--flip', fl],
        capture_output=True, text=True)
    lines = [l for l in r.stdout.splitlines() if 'dist_target' in l]
    if lines:
        print('   ' + lines[-1].strip())
    else:
        print(f'   ERR rc={r.returncode} last_stderr: {(r.stderr.splitlines() or [""])[-1][:120]}')
