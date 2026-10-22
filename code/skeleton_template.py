# ============================================================================
# 第 1 段 · 文件头：用途 / 输入 / 输出 / 自检数字
#   用途：UPC 七段结构的最小可运行骨架。示例物理是阻尼自由落体，
#         换成你的方程只需改第 4 段，其余六段一行不动。
#   输入：第 2 段 PARAMS（全部输入只出现在这一处）
#   输出：终端 4 行自检数字 + figs/skeleton.png（300 dpi）
#   跑完应该看到（实测值，对不上就是代码有问题）：
#     unit check          : OK  (tau = 1.00 s)
#     terminal speed      : numeric 9.80955 m/s  analytic 9.81000 m/s  diff 0.005%
#     exact-solution diff : 2.3e-08 %          <- 与 rtol=1e-9 同量级 = 积分器误差主导
#     scan b              : 12.26 / 10.90 / 9.81 / 8.92 / 8.17 m/s
#     [fig] <仓库根>/figs/skeleton.png
# ============================================================================
import os

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib

matplotlib.use("Agg")  # 无界面后端：断网/无显示器/SSH 下都能出图
import matplotlib.pyplot as plt

# ==== 第 2 段 · 参数区：一个 dict 装全部输入 ====
# 为什么用 dict 而不是散装全局变量：灵敏度扫描时改一个 key 就能跑一组，
# 且可以整体 copy 一份当基准，不会互相污染。
PARAMS = {
    "m": 1.0,          # 质量，kg
    "b": 1.0,          # 线性阻尼系数，kg/s
    "g": 9.81,         # 重力加速度，m/s^2
    "y0": [0.0, 0.0],  # 初始状态 [位移 m, 速度 m/s]，顺序与段 4 的 rhs 拆包顺序一致
    "T": 10.0,         # 积分终止时间，s（取 10 个特征时间，足够收敛到终端速度）
}


# ==== 第 3 段 · 量纲自检：贴着参数区放，公式一错立刻抛异常 ====
def check_units(p=PARAMS):
    """量纲自检：tau = m/b 必须是秒量级。

    参数：p 参数字典。
    返回：None。量纲或数量级不对时抛 AssertionError（不返回错误码，直接炸）。
    """
    tau = p["m"] / p["b"]                     # kg / (kg/s) = s
    assert 1e-6 < tau < 1e6, f"特征时间 tau={tau} s 不合理：检查 m(kg) 与 b(kg/s) 的单位"
    print(f"unit check          : OK  (tau = {tau:.2f} s)")


# ==== 第 4 段 · 物理模型：改物理只改这一处 ====
def rhs(t, y, p=PARAMS):
    """ODE 右端，f(t, y) -> dy/dt。

    参数：t 时间 s；y 状态向量 [x 位移 m, v 速度 m/s]。
    返回：list [dx/dt, dv/dt]，单位 [m/s, m/s^2]。
    """
    x, v = y
    return [v, -p["g"] - (p["b"] / p["m"]) * v]


# ==== 第 5 段 · 求解器封装：所有分析都调它，只写一次 ====
# 为什么 y0 / T 的默认值写 None 而不是 PARAMS["..."]：
#   Python 的默认参数在**函数定义时**求值一次，之后再也不变。
#   写成 T=PARAMS["T"] 的话，后面把 PARAMS["T"] 改成 20.0，integrate() 仍用定义时的 10.0
#   —— 48 小时里换参数重跑会得到"改了没生效"的诡异结果。
#   传 None 当哨兵、进函数再读 PARAMS，才是安全的写法。
#   （例外：p=PARAMS 这种默认值是**同一个 dict 对象**，读属性时看到的是最新内容，所以安全。）
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


# ==== 第 6 段 · 分析：退化检验 + 灵敏度。本段不 import matplotlib ====
def check_vs_analytic(T=None):
    """退化检验：数值解 vs 精确解 v(t) = -v_t (1 - e^{-t/tau})。

    参数：T 对照时刻 s（None 取 PARAMS["T"]）。
    返回：(v 数值, v 解析, 相对偏差)，速度单位 m/s，偏差无量纲。
    """
    T = PARAMS["T"] if T is None else T
    m, b, g = PARAMS["m"], PARAMS["b"], PARAMS["g"]
    v_num = speed_at(T)
    v_ana = -(m * g / b) * (1.0 - np.exp(-(b / m) * T))
    return v_num, v_ana, abs(v_num - v_ana) / abs(v_ana)


def scan_b(factors=(0.8, 0.9, 1.0, 1.1, 1.2), T=None):
    """结果灵敏度：只改 b，看同一时刻速度变多少。

    参数：factors 缩放因子元组（无量纲）；T 取样时刻 s（None 取 PARAMS["T"]）。
    返回：[(factor, speed)]，speed 单位 m/s。
    """
    T = PARAMS["T"] if T is None else T
    base, out = PARAMS["b"], []
    for f in factors:
        PARAMS["b"] = base * f
        out.append((f, speed_at(T)))
    PARAMS["b"] = base                        # 必须复位，否则后面全算错
    return out


# ==== 第 7 段 · 出图 + 主流程：放最后，因为要改的最多 ====
# 本文件在 code/ 下，图统一出到仓库根的 figs/（与 test.py、README 的目录约定一致）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "figs")
# 为什么用 __file__ 向上定位而不是相对路径：从任何 CWD 调用都能出到同一个 figs/


def plot_velocity(out=None):
    """画 v(t) 曲线，并叠一条终端速度参考线。

    参数：out 输出路径；None 时写到 figs/skeleton.png。
    返回：实际写入的绝对路径。
    """
    out = out or os.path.join(FIGS, "skeleton.png")
    sol = integrate()
    v_t = PARAMS["m"] * PARAMS["g"] / PARAMS["b"]
    plt.figure(figsize=(6, 4))
    plt.plot(sol.t, sol.y[1], lw=1.5, label="numerical (RK45)")
    plt.axhline(-v_t, ls="--", lw=1.0, label=r"analytic $v_t=mg/b$")
    plt.xlabel("Time $t$ (s)")                # 物理图轴标签必须带单位
    plt.ylabel("Velocity $v$ (m/s)")
    plt.legend(frameon=False)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()
    return out


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)          # 先建目录，否则 savefig 报错

    check_units()                             # 自检 1：量纲

    v_num, v_ana, rel = check_vs_analytic()   # 自检 2/3：退化到解析解
    v_t = PARAMS["m"] * PARAMS["g"] / PARAMS["b"]
    print(f"terminal speed      : numeric {abs(v_num):.5f} m/s  "
          f"analytic {v_t:.5f} m/s  diff {abs(abs(v_num) - v_t) / v_t * 100:.3f}%")
    print(f"exact-solution diff : {rel * 100:.1e} %")

    print("scan b              : "              # 自检 4：灵敏度
          + " / ".join(f"{abs(v):.2f}" for _, v in scan_b()) + " m/s")

    print(f"[fig] {plot_velocity()}")
