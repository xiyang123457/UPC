"""B1 附件：把七段骨架换成「阻尼弹簧振子」，用于验证结构的可复用性。

用途：与 skeleton_template.py 逐段对照，看清楚"换题时哪几段要动、哪一段逐字不动"。
输入：第 2 段 PARAMS。
输出：终端 3 行自检数字 + figs/demo_swap.png（300 dpi）。
跑完应该看到（实测值）：
    unit check          : OK  (omega0 = 2.00 rad/s, zeta = 0.10)
    x(T)                : numeric -0.007664 m  analytic -0.007664 m  diff 7.8e-09%
    scan c              : -0.067 / -0.046 / -0.032 / -0.022 / -0.015 m/s
    [fig] <脚本所在目录>/figs/demo_swap.png
"""
import os

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ==== 第 2 段 · 参数区 ====
PARAMS = {
    "m": 1.0,          # 质量，kg
    "k": 4.0,          # 弹簧刚度，N/m
    "c": 0.4,          # 阻尼系数，kg/s
    "y0": [1.0, 0.0],  # 初始状态 [位移 m, 速度 m/s]
    "T": 20.0,         # 积分终止时间，s
}


# ==== 第 3 段 · 量纲自检 ====
def check_units(p=PARAMS):
    """量纲自检：omega0 = sqrt(k/m) 必须是角频率量级，zeta 必须无量纲且 < 1（欠阻尼）。"""
    omega0 = np.sqrt(p["k"] / p["m"])                    # sqrt((N/m)/kg) = 1/s
    assert 1e-6 < omega0 < 1e6, f"omega0={omega0} 1/s 不合理：检查 k(N/m) 与 m(kg)"
    zeta = p["c"] / (2.0 * np.sqrt(p["k"] * p["m"]))     # 无量纲
    assert 0.0 <= zeta < 1.0, f"zeta={zeta} 不在欠阻尼区间：精确解公式不适用"
    print(f"unit check          : OK  (omega0 = {omega0:.2f} rad/s, zeta = {zeta:.2f})")


# ==== 第 4 段 · 物理模型（换题时只改这里的内容）====
def rhs(t, y, p=PARAMS):
    """ODE 右端。y=[位移 m, 速度 m/s]，返回 [速度 m/s, 加速度 m/s^2]。"""
    x, v = y
    return [v, -(p["k"] / p["m"]) * x - (p["c"] / p["m"]) * v]


# ==== 第 5 段 · 求解器封装（本题逐字未改，与骨架完全相同）====
def integrate(y0=None, T=None, rtol=1e-9):
    """积分一次并返回 solve_ivp 结果对象（含 .t / .y / .sol 插值）。

    参数：y0 初始状态向量（None 时取 PARAMS["y0"]，单位见段 4 docstring）；
          T 终止时间 s（None 时取 PARAMS["T"]）；rtol 相对容差（无量纲）。
    返回：OdeResult。收敛性检验就是扫这个 rtol。
    """
    y0 = PARAMS["y0"] if y0 is None else y0
    T = PARAMS["T"] if T is None else T
    return solve_ivp(rhs, (0.0, T), y0, method="RK45",
                     rtol=rtol, atol=1e-12, dense_output=True)


def speed_at(t_end=None, rtol=1e-9):
    """返回 t_end 时刻的速度，单位 m/s（状态向量的第 2 个分量）。"""
    return float(integrate(T=t_end, rtol=rtol).y[1, -1])


# ==== 第 6 段 · 分析 ====
def check_vs_analytic(T=None):
    """退化检验：欠阻尼振子精确解 vs 数值解。

    参数：T 对照时刻 s（None 取 PARAMS["T"]）。
    返回：(x 数值, x 解析, 相对偏差)，位移单位 m，偏差以初始振幅归一（无量纲）。
    """
    T = PARAMS["T"] if T is None else T
    m, k, c = PARAMS["m"], PARAMS["k"], PARAMS["c"]
    x0 = PARAMS["y0"][0]
    omega0 = np.sqrt(k / m)
    zeta = c / (2.0 * np.sqrt(k * m))
    omega_d = omega0 * np.sqrt(1.0 - zeta**2)            # 有阻尼固有角频率，rad/s
    x_num = float(integrate(T=T).y[0, -1])
    x_ana = x0 * np.exp(-zeta * omega0 * T) * (
        np.cos(omega_d * T) + zeta * omega0 / omega_d * np.sin(omega_d * T))
    return x_num, x_ana, abs(x_num - x_ana) / abs(x0)


def scan_c(factors=(0.8, 0.9, 1.0, 1.1, 1.2), T=None):
    """结果灵敏度：只改阻尼 c，看同一时刻速度变多少。

    参数：factors 缩放因子元组（无量纲）；T 取样时刻 s（None 取 PARAMS["T"]）。
    返回：[(factor, speed)]，speed 单位 m/s。
    """
    T = PARAMS["T"] if T is None else T
    base, out = PARAMS["c"], []
    for f in factors:
        PARAMS["c"] = base * f
        out.append((f, speed_at(T)))
    PARAMS["c"] = base                                   # 必须复位，否则后面全算错
    return out


# ==== 第 7 段 · 出图 + 主流程 ====
# 本文件在 code/ 下，图统一出到仓库根的 figs/
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "figs")


def plot_velocity(out=None):
    """画 v(t) 曲线。

    参数：out 输出路径；None 时写到 figs/demo_swap.png。
    返回：实际写入的绝对路径。
    """
    out = out or os.path.join(FIGS, "demo_swap.png")
    sol = integrate()
    plt.figure(figsize=(6, 4))
    plt.plot(sol.t, sol.y[1], lw=1.5)
    plt.xlabel("Time $t$ (s)")
    plt.ylabel("Velocity $v$ (m/s)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    return out


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    check_units()
    x_num, x_ana, rel = check_vs_analytic()
    print(f"x(T)                : numeric {x_num:+.6f} m  analytic {x_ana:+.6f} m  "
          f"diff {rel * 100:.1e}%")
    print("scan c              : "
          + " / ".join(f"{v:+.3f}" for _, v in scan_c()) + " m/s")
    print(f"[fig] {plot_velocity()}")
