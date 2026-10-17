# ============================================================================
# 文件头（L1）
# 用途：UPC 48 小时比赛的数值计算起手模板——从方程到图，一次跑通
#       示例题：2025-B Artillery（5 kg 球弹、直径 11 cm、初速上限 450 m/s、目标 1200 m）
# 输入：本题的物理参数（在 PARAMS 段改）
# 输出：figs/*.png（300 dpi）+ 终端自检输出
# 跑完应该看到的自检数字（已实测，可直接对照）：
#   feasible angles for 1200 m: ['2.72 deg', '77.06 deg']   <- 两个解，不是唯一解！
#   check: range=1200.0 m  flight=3.77 s  maxalt=17.6 m      <- 与 2025 金奖论文的 3.7 s 一致
#   Cd x0.8 -> 1292.1 m ... Cd x1.2 -> 1122.3 m              <- 灵敏度温和，模型稳健
#   vacuum check: diff=0.000%                                <- 关掉空气后与解析解完全吻合
# ============================================================================

# import 说明（为什么需要这个库）：
#   numpy      -> 数组与向量化运算，所有数值计算的基础
#   scipy      -> solve_ivp 解 ODE、brentq 求根、quad 积分，避免自己写积分器
#   matplotlib -> 出图，论文里的图必须 300 dpi 且带单位
#   pint       -> 单位与量纲检查（可选依赖，未安装时自动降级）
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
import matplotlib
matplotlib.use("Agg")           # 无界面后端：无显示环境下也能出图
import matplotlib.pyplot as plt

# ==== 1. 单位系统 ====
# 变量 ureg：pint 单位注册表；装不上时降级为 None，主流程仍可跑
#   例：ureg("m/s") -> 速度单位
# 为什么用 try/except：pint 不是必装依赖，赛时断网装不上也不能让脚本挂掉
try:
    import pint
    ureg = pint.UnitRegistry()
except ImportError:
    ureg = None
    print("[warn] pint 未安装，量纲自检跳过；pip install pint 后重跑")

# ==== 2. 物理参数（本题改这里）====
# 变量 PARAMS：dict，装本题全部输入参数
#   例：{"m": 5.0, "d": 0.11, "v0": 450.0}
# 为什么用 dict：方便整体做灵敏度扫描，改一个 key 就能跑一组
PARAMS = {
    "m": 5.0,        # 弹丸质量，kg（2025-B 题给定 5 kg）
    "d": 0.11,       # 直径，m（2025-B 题给定 11 cm）
    "v0": 450.0,     # 初速度上限，m/s（2025-B 题给定）
    "rho_air": 1.225,  # 海平面空气密度，kg/m^3（ISA 标准大气）
    "Cd": 0.47,      # 球体阻力系数（无量纲，来自流体力学手册）
}

# ==== 3. 量纲自检 ====
# 作用：确认关键组合量的量纲正确，跑不通就说明公式写错了
# 参数：无
# 为什么：物理论文里量纲错误是致命伤，用代码强制检查比人眼可靠
def unit_check():
    """量纲自检：失败会抛异常，等于免费的错误拦截器。"""
    if ureg is None:                         # 降级路径：没装 pint 就跳过
        print("unit check: SKIPPED (pint missing)")
        return
    g = 9.81 * ureg("m/s**2")
    v = PARAMS["v0"] * ureg("m/s")
    t_fall = (v / g).to("s")                 # 特征时间量级
    print(f"unit check: OK, characteristic time ~ {t_fall.magnitude:.2f} s")

# ==== 4. 动力学方程 ====
# 作用：定义 ODE 右端，返回 [dx/dt, dy/dt, dvx/dt, dvy/dt]
# 参数：t 标量时间；y 长度为 4 的状态向量 [x, y, vx, vy]
# 为什么写成这种形式：solve_ivp 要求 f(t, y) -> dy/dt，且 y 必须是一维数组
def rhs(t, y, p=PARAMS):
    x, ypos, vx, vy = y
    v = np.hypot(vx, vy)                     # 速率，用于二次阻力
    # 二次阻力：F = -0.5 * rho * Cd * A * v^2 * (v_hat)
    # 为什么用二次而非 Stokes：450 m/s 属高雷诺数区，惯性阻力主导
    A = np.pi * (p["d"] / 2) ** 2
    k = 0.5 * p["rho_air"] * p["Cd"] * A / p["m"]
    ax = -k * v * vx
    ay = -9.81 - k * v * vy
    return [vx, vy, ax, ay]

# ==== 5. 求解 ====
# 方法签名：solve_ivp(fun, t_span, y0, method=..., rtol=..., atol=...) -> OdeResult
#   作用：用自适应步长 Runge-Kutta 积分 ODE 初值问题
#   参数：t_span=(0, T) 时间区间；y0 初值；method="RK45" 默认 4(5) 阶
#   为什么：自适应步长在高速段自动加密，比固定步长省时间且更准
def integrate(angle_deg, T=120.0):
    th = np.deg2rad(angle_deg)
    v0 = PARAMS["v0"]
    y0 = [0.0, 0.0, v0 * np.cos(th), v0 * np.sin(th)]
    # events：落地（y=0）时停止积分，避免继续算地下轨迹
    # 为什么设 direction=-1：只抓"从上方穿到下方"的那次，避免起点 y=0 被误判为落地
    def hit_ground(t, y, p=None):
        return y[1]
    hit_ground.terminal = True
    hit_ground.direction = -1
    return solve_ivp(rhs, (0, T), y0, method="RK45",
                     rtol=1e-8, atol=1e-10, events=hit_ground,
                     dense_output=True)

# ==== 6. 射程函数 ====
# 作用：给定发射角，返回落地时的水平距离
# 参数：angle_deg 发射角（度）；返回 float（m），若 120 s 内没落地返回 nan
# 为什么单独封一层：后面扫描和求根都要反复调用它
def range_at(angle_deg):
    sol = integrate(angle_deg)
    return float(sol.y_events[0][0][0]) if len(sol.y_events[0]) else np.nan

# ==== 7. 求全部可行发射角 ====
# 作用：给定目标射程 R，返回所有满足的发射角（通常有 2 个解）
# 参数：R 目标水平距离；lo/hi 角度扫描区间；n 扫描点数
# 为什么不能直接用 brentq(lo, hi)：
#   带空气阻力时 range(angle) 是**非单调**的（先增后减），
#   f(1度) 与 f(89度) 可能同号 -> brentq 直接抛
#   ValueError: f(a) and f(b) must have different signs
#   必须先粗扫找符号变化区间，再在区间内用 brentq 精求
#   ——这正是 UPC 题目要考的"求全部可行解"，不是求唯一解
def find_feasible_angles(R, lo=0.5, hi=89.5, n=180):
    grid = np.linspace(lo, hi, n)
    vals = np.array([range_at(a) - R for a in grid])   # 射程与目标的差
    roots = []
    for i in range(n - 1):
        if np.isfinite(vals[i]) and np.isfinite(vals[i + 1]) and vals[i] * vals[i + 1] < 0:
            roots.append(brentq(lambda a: range_at(a) - R, grid[i], grid[i + 1], xtol=1e-6))
    return roots

# ==== 8. 灵敏度分析（银奖门槛项）====
# 作用：改变单个参数，看结果变化多少，用于说明模型的适用范围
# 参数：angle_deg 基准发射角；key 要扫描的参数名；factors 缩放因子元组
# 为什么必须做：评审看的是"结果在多大范围内可信"，不是单点数字
def sensitivity(angle_deg, key, factors=(0.8, 0.9, 1.0, 1.1, 1.2)):
    out = []
    base = PARAMS[key]
    for f_ in factors:
        PARAMS[key] = base * f_
        out.append((f_, range_at(angle_deg)))
    PARAMS[key] = base                            # 复位，避免污染后续计算
    return out

# ==== 9. 出图 ====
# 作用：把轨迹画成论文可用的图
# 为什么这么设置：300 dpi 满足印刷；轴标签带单位是物理评审的硬要求
def plot_trajectory(angle_deg, out="figs/traj.png"):
    sol = integrate(angle_deg)
    plt.figure(figsize=(6, 4))
    plt.plot(sol.y[0], sol.y[1], lw=1.5)
    plt.xlabel("Horizontal distance $x$ (m)")     # 单位必须写
    plt.ylabel("Altitude $y$ (m)")
    plt.title(f"Trajectory at {angle_deg}$^\\circ$")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=300)
    plt.close()

# ==== 10. 主流程 ====
if __name__ == "__main__":
    import os
    os.makedirs("figs", exist_ok=True)            # 输出目录先建，避免 savefig 报错

    unit_check()                                  # 自检 1：量纲必须通过

    roots = find_feasible_angles(1200.0)          # 2025-B 题：目标水平距离 1200 m
    print("feasible angles for 1200 m:",
          ["%.2f deg" % r for r in roots])        # 自检 2：应得到 2.72 deg 与 77.06 deg

    if roots:
        th = roots[0]                             # 取低角度解（弹道平直、飞行时间最短）
        sol = integrate(th)
        print("check: range=%.1f m  flight=%.2f s  maxalt=%.1f m"
              % (range_at(th), sol.t_events[0][0], sol.y[1].max()))   # 自检 3：flight ≈ 3.77 s

        plot_trajectory(th)                       # 出图到 figs/traj.png

        # 灵敏度扫描：看射程对阻力系数 Cd 的依赖有多强
        # 为什么扫 Cd：它是本题最不确定的参数（球体阻力系数随雷诺数在 0.1-0.5 变化）
        for factor, rng in sensitivity(th, "Cd"):
            print("  Cd x%.1f -> range %.1f m" % (factor, rng))       # 自检 4：变化应温和

        # 极限自检：把空气密度设为 0，结果必须回到解析抛物线射程
        PARAMS["rho_air"] = 0.0
        r_vac = range_at(th)
        r_analytic = PARAMS["v0"] ** 2 * np.sin(2 * np.deg2rad(th)) / 9.81
        print("vacuum check: numeric=%.1f m analytic=%.1f m diff=%.3f%%"
              % (r_vac, r_analytic, abs(r_vac - r_analytic) / r_analytic * 100))
        PARAMS["rho_air"] = 1.225                 # 复位
