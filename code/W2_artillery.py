"""UPC 2026 · W2 · 2025-B Artillery 第一问（B 角色实现，按 A 的 W2 建模卡）。

用途：条件为目标射程 R = 1200 m、炮与目标同海拔 500 m、v0 <= 450 m/s。
    用含二次空气阻力的弹道 ODE 求**全部**可命中 1200 m 的 (v0, theta) 组合
    （低角支 theta_low(v0) + 高角支 theta_high(v0)），以及最短飞行时间及对应 (v0, theta)。

输入：第 2 节 PARAMS（全部输入只出现在这一处）。阻力系数 k 由 rho*Cd*A/(2m) 现算
    而不写死，所以灵敏度扫描只需改 PARAMS["Cd"] / PARAMS["rho"]。

输出：终端自检表 + figs/ 下 7 个 PNG（一图一文件，300 dpi，轴带单位）：
    W2_q1_1_boundary.png     可行域边界 theta_low(v0) / theta_high(v0)
    W2_q1_2_range_theta.png  R(theta) 非单调曲线 + 两个根
    W2_q1_3_traj_low.png     v0=450 低角解弹道（顶点 17.05 m）
    W2_q1_4_traj_high.png    v0=450 高角解弹道（顶点 2274.34 m；与低角差 134 倍，故分图）
    W2_q1_5_flight_time.png  飞行时间随 v0 变化 + 最短时间点
    W2_q1_6_scan_k.png       k 缩放（= Cd 或 rho 缩放）对射程的影响
    W2_q1_7_compensate.png   重新命中 1200 m 所需的角度补偿量

自检数字（实测值，2026-10-25，环境 D:\\anaconda\\envs\\upc；对不上先去 results.md 对表）：
    unit check          : OK  (A=9.503e-03 m^2, k=5.212e-04 1/m, a_drag(450)=105.6 m/s^2, 1/k=1918 m)
    offline check       : OK  (imports: ast, math, matplotlib, numpy, os, scipy; no network calls)
    vacuum range        : numeric 17876.671 m  analytic 17876.671 m  diff 2.04e-14 %
    vacuum time         : numeric 45.87156 s  analytic 45.87156 s  diff 1.55e-14 %
    angles @ v0=450     : #1 theta=2.6495 deg t=3.7101 s / #2 theta=77.5774 deg t=43.4643 s
    v_min(R=1200)       : 143.0451 m/s @ 40.7404 deg（两解合一处的初速下界）
    min flight time     : 3.7101 s @ v0=450.0 m/s, theta=2.6495 deg（约束取等号）
    peak-miss window    : coarse 3307.3178 m vs true 3307.3187 m（窗宽 0.92 mm，窗内仍得 2 个解）
    convergence         : 1e-5..1e-13 -> theta 2.649551..2.649550 deg, steps 14..138（9.9 倍）
    elasticity S        : -0.334（log-log）/ -0.340（端点差分）
    altitude 0/500/2000 : 1180.34 / 1200.00 / 1259.20 m
    wind 5/10/20 m/s    : drift 5.20 / 10.41 / 20.82 m（粗估 w*t 为 3.56 倍上界）

分段：本文件按 7 段组织（文件头 / 参数区 / 自检 / 物理模型 / 求解器封装 / 分析 / 出图），
    每段横幅写成 `# ==== N. 分节名 ====`，N 就是段号。
"""
# 这一组 import 的用途：
#   math       —— hypot/radians/cos/sin 这类标量运算，走 C 实现且不建数组，比 numpy 快
#   os         —— 用 __file__ 定位仓库根的 figs/，从任何 CWD 调用都不出错
#   ast        —— 解析本文件源码做断网检查（扫 import 与联网字样），见 check_offline
#   numpy      —— 扫描网格（linspace）、数组批量比较、曲线的逐点取整
#   solve_ivp  —— 自适应步长 RK45 + 落地事件检测：终止时刻精确，rtol 可扫做收敛性检验
#   brentq     —— 求根（角度/初速），有括号时比打表精确得多
#   minimize_scalar —— 一维极值：射程峰值（range(theta) 非单调，不能用全局二分）
#   matplotlib —— 出图；先 use("Agg") 再用 pyplot，断网/无显示器/SSH 下都能保存
import ast
import math
import os

import numpy as np
from scipy.integrate import solve_ivp              # 自适应步长 RK45 + 落地事件：终止时刻精确、rtol 可扫
from scipy.optimize import brentq, minimize_scalar  # 求根（角度/初速）与一维极值（射程峰值）
import matplotlib

matplotlib.use("Agg")                              # 无界面后端：断网/无显示器/SSH 下都能出图
import matplotlib.pyplot as plt                    # 画图 API；必须在上面的 use("Agg") 之后导入


# ==== 2. 参数区：一个 dict 装全部输入 ====
# 为什么用 dict 而不是散装全局变量：灵敏度扫描时改一个 key 就能跑一组；
# 也避免"魔法数散落在 3 个函数里，换参数漏改一处"。
PARAMS = {
    "m": 5.0,             # 弹丸质量，kg（题给）
    "d": 0.11,            # 弹丸直径，m（题给）
    "rho": 1.167,         # 空气密度，kg/m^3（ISA 500 m 海拔 = 题给的海拔条件；W1 的 1.225 已作废）
    "Cd": 0.47,           # 阻力系数，无量纲（球体高雷诺数标准值；Ma>0.8 时失真，全模型最大不确定性）
    "g": 9.81,            # 重力加速度，m/s^2（0-500 m 内变化 0.015%，可当常数）
    "isa_lapse": 2.25577e-5,  # ISA 对流层温度递减率换算系数，1/m（标准值，别改）
    "isa_exp": 4.2559,    # ISA 密度剖面的幂指数（= g/(R*L) - 1，标准值）
    "alt_ref": 500.0,     # 密度剖面的锚定海拔，m（题给的"炮与目标同海拔"）
    "alt_grid": (0.0, 250.0, 500.0, 1000.0, 1500.0, 2000.0, 3000.0),  # 海拔扫描网格，m
    "alt_ref_set": (0.0, 500.0, 2000.0),  # 任务书点名的三档海拔，m（终端表与论文表就报这三个）
    "wind": 0.0,          # 横风速度，m/s（沿 z 轴，正为 +z；0 = 静风，第一问的条件）
    "wind_grid": (0.0, 5.0, 10.0, 15.0, 20.0, 30.0),  # 侧风扫描网格，m/s
    "R_target": 1200.0,   # 目标射程，m（题给第一问）
    "v0_max": 450.0,      # 出膛速度上限，m/s（题给；最短飞行时间是它在约束下取等号）
    "v0_lo_scan": 50.0,   # v_min 求根的搜索下界，m/s（真空下界 sqrt(gR)=108.5，留一倍余量）
    "ang_lo": 0.5,        # 角度粗扫下界，deg（建模卡给定；0 deg 是退化解，避开）
    "ang_hi": 89.5,       # 角度粗扫上界，deg（建模卡给定）
    "n_angle": 180,       # 角度粗扫点数（步长 0.495 deg，与 W1 口径一致，可对照）
    "n_refine": 200,      # 射程峰值附近的加密点数（专门修 W1 的 0.029 m 漏解窗口）
    "n_v0": 25,           # 可行域边界 θ(v0) 的 v0 采样点数
    "xatol_angle": 1e-8,  # 峰值寻优的角度容差，deg（射程对 θ 二阶平坦，1e-8 已远超需要）
    "xtol_angle": 1e-10,  # brentq 求角度的容差，deg
    "xtol_v0": 1e-8,      # brentq 求 v_min 的容差，m/s
    "tol_tangent": 1e-2,  # 判定"切根"的残差阈值，m（R 恰等于射程包络时根不变号，必须单独兜底）
    "rtol": 1e-9,         # 写进论文的数值用的相对容差，无量纲
    "rtol_scan": 1e-8,    # 扫描用的相对容差，无量纲（收敛性检验证明与 1e-9 差 < 1 mm 量级）
    "atol": 1e-12,        # 绝对容差，SI 量级下足够（位置量 m，速度量 m/s）
    "t_max": 120.0,       # 积分时间上限，s（高角解飞 ~20 s，取 6 倍余量兜底）
    "cd_factors": (0.8, 0.9, 1.0, 1.1, 1.2),    # C_d 缩放因子（±20%，与 W1 口径一致便于对照）
    "rho_factors": (0.8, 0.9, 1.0, 1.1, 1.2),   # ρ 缩放因子（±20%，同上）
}
# 收敛性检验要扫的 rtol 序列，无量纲（1e-5 最松、1e-13 最紧，跨 8 个量级，与任务书口径一致）
RTOL_SERIES = (1e-5, 1e-7, 1e-9, 1e-11, 1e-13)


# ==== 3. 量纲自检：贴着参数区放，公式一错立刻抛异常 ====
# 设计决策：建模卡第八节要求 B 在写 rhs 之前先过硬检，所以这两条断言放在
# 物理模型（第 4 段）**之前**——公式错了，第 4 段根本跑不到。
def frontal_area(p=PARAMS):
    """迎风面积 A = π(d/2)^2，单位 m^2。

    参数：p 参数字典（用 d，不用 A——扫直径时 A 自动跟着变）。
    返回：float，面积，m^2。球的正投影面积 = 大圆面积。
    """
    return math.pi * (p["d"] / 2.0) ** 2


def drag_k(p=PARAMS):
    """二次阻力系数 k = ρ·C_d·A/(2m)，单位 1/m。

    参数：p 参数字典。
    返回：float，k，1/m。
    为什么现算而不是存常数：灵敏度扫描改 p["Cd"] 或 p["rho"] 时 k 必须跟着变，
    存成常数就得记得手动同步 4 个地方。
    """
    return p["rho"] * p["Cd"] * frontal_area(p) / (2.0 * p["m"])


def check_units(p=PARAMS):
    """量纲自检 + 量级抽查（建模卡第八节的第 1、2 条，B 的否决权）。

    参数：p 参数字典。
    返回：None。量纲或数量级不对时抛 AssertionError，直接中断（不返回错误码）。
    """
    A = frontal_area(p)                        # float，9.503e-3 m^2
    k = drag_k(p)                              # float，5.21e-4 1/m

    # 量纲链：k = (kg/m^3)·(-)·(m^2)/kg = 1/m；k·v·v = (1/m)(m/s)(m/s) = m/s^2 ✓
    assert abs(A - 9.503e-3) < 5e-6, f"迎风面积 A={A:.6e} m^2，与卡给 9.503e-3 不符：检查 d 的单位（m，不是 cm）"
    assert abs(k - 5.21e-4) < 1e-6, f"阻力系数 k={k:.6e} 1/m，与卡给 5.21e-4 不符：检查 rho(kg/m^3)/Cd(-)/d(m)/m(kg)"

    # 量级抽查：v=v0_max 时阻力量级必须 ~105.6 m/s^2（=10.8 g），这是"不能忽略阻力"的硬证据
    a_drag = k * p["v0_max"] ** 2              # float，447.5^... → 105.55 m/s^2
    assert 100.0 < a_drag < 112.0, f"v=450 m/s 时 a_drag={a_drag:.1f} m/s^2，与卡给 105.6 不符：参数或公式错了"

    # 特征长度 1/k 是"阻力把速度衰减到 1/e 的距离尺度"，必须远大于射程才有意义（它是 1.9 km）
    assert 1e2 < 1.0 / k < 1e4, f"特征长度 1/k={1.0 / k:.0f} m 不合理：k 可能算错了一个数量级"

    print(f"unit check          : OK  (A={A:.3e} m^2, k={k:.3e} 1/m, "
          f"a_drag(450)={a_drag:.1f} m/s^2, 1/k={1.0 / k:.0f} m)")


# ==== 4. 物理模型：改物理只改这一处 ====
# 设计决策（方程来源与取舍）：
#   建模卡第三节给的就是这四行：m·a = mg + F_drag，F_drag = −½ρC_dAv²·v̂。
#   阻力按**速率** v=√(vx²+vy²) 而非分量各自平方——这是二次阻力的定义，
#   写成 −k·|vx|·vx 会在小仰角时把阻力低估一半（vx 占了速率的绝大部分）。
#   不含风项：第一问是静风条件；第二问加风时只在这里把 vx 换成 (vx−w)。
def rhs(t, y, p=PARAMS):
    """ODE 右端，f(t, y) -> dy/dt。

    参数：t 时间，s；y 状态向量 [x 水平位移 m, h 高度 m, vx m/s, vy m/s]。
    返回：list [dx/dt, dh/dt, dvx/dt, dvy/dt]，单位 [m/s, m/s, m/s^2, m/s^2]。
    """
    _, _, vx, vy = y
    k = drag_k(p)                              # float，5.21e-4 1/m（每次现算，保证扫描生效）
    v = math.hypot(vx, vy)                     # float，速率 m/s（不是分量；是阻力方向的分母）
    return [vx, vy, -k * v * vx, -p["g"] - k * v * vy]


def rhs_wind(t, y, p=PARAMS, wind=None):
    """含侧风的 ODE 右端（3D），f(t, y) -> dy/dt。

    参数：t 时间 s；y = [x, h, z, vx, vy, vz]，单位 [m, m, m, m/s, m/s, m/s]；
          wind 横风 m/s（None 取 p["wind"]）。
    返回：list，长度 6，单位 [m/s, m/s, m/s, m/s^2, m/s^2, m/s^2]。
    设计决策（为什么不改第 4 段的 rhs，另开一个）：
        第一问的 4 维系统不引入 z 自由度，避免"加风改了静风结果"这种不可控耦合；
        wind=0 且 vz=0 时本函数严格退化为 rhs（主流程里有断言验证射程逐位相同）。
    坑：阻力必须按**相对速度**计算 |v − w|，而且相对速度的三个分量要一起进模长——
        横向速度会抬高总速率，从而改变纵向阻力耦合；只把 vx 减 w 是错的。
    """
    _, _, _, vx, vy, vz = y
    k = drag_k(p)                              # float，5.21e-4 1/m
    # 坑：风速必须由 wind 形参逐次传入，不能只读 p["wind"]——
    #   扫描时若靠改 PARAMS 传值，忘记复位或写错键名都会静默退化成静风（本函数第一版就踩了）
    w = p["wind"] if wind is None else wind    # float，横风速度 m/s（沿 z，正为 +z 方向）
    ur, vr, wr = vx, vy, vz - w                # 相对速度三分量，m/s
    s = math.sqrt(ur * ur + vr * vr + wr * wr)  # 相对速率，m/s
    return [vx, vy, vz, -k * s * ur, -p["g"] - k * s * vr, -k * s * wr]


# ==== 5. 求解器封装：所有分析都调它，只写一次 ====
def _landed(t, y, p=PARAMS):
    """落地事件：检测高度 h 穿过 0。

    参数：t 时间 s；y 状态向量（见第 4 段）。
    返回：float h，m。solve_ivp 在返回值为 0 时触发事件。
    坑：初始时刻 h=0 也是零点，所以必须靠 direction=-1 只认"下穿"，
        否则 t=0 就被判为落地、射程恒为 0。
    """
    return y[1]


_landed.terminal = True                        # 命中即停积分，省掉飞行后的空转
_landed.direction = -1                         # 只取 h 由正变负的下穿时刻（起飞那一下不算）


def integrate(y0=None, T=None, rtol=None, dense=False, p=PARAMS):
    """积分一次抛体运动，返回 solve_ivp 结果对象。

    参数：y0 初始状态向量 [x, h, vx, vy]（None 时由调用方拼好，见 shot_at）；
          T 积分上限，s（None 取 PARAMS["t_max"]）；rtol 相对容差（None 取 PARAMS["rtol"]）；
          dense 是否带连续插值（画轨迹才要，扫描关掉省内存）。
    返回：OdeResult，含 .t / .y / .t_events / .y_events / .sol。
    为什么 y0 / T 用 None 哨兵：Python 默认参数在定义时求值一次，
        写成 T=PARAMS["t_max"] 会让"改了参数不生效"，这是模板第 1.3 节实测过的雷。
    """
    y0 = [0.0, 0.0, 0.0, 0.0] if y0 is None else y0
    T = p["t_max"] if T is None else T
    rtol = p["rtol"] if rtol is None else rtol
    return solve_ivp(rhs, (0.0, T), y0, method="RK45",
                     rtol=rtol, atol=p["atol"], events=_landed,
                     dense_output=dense)


def shot_at(theta_deg, v0, rtol=None, dense=False, p=PARAMS):
    """发射一发，返回落地状态（射程/飞行时间/最高点/落速）。

    参数：theta_deg 发射仰角，deg；v0 出膛速度，m/s；
          rtol 相对容差（None 取 p["rtol"]）；dense 是否保留插值（画轨迹用）。
    返回：dict，键 range(m) / t_flight(s) / h_max(m) / v_land(m/s) / sol。
    为什么一次积分返回全部量：扫描时射程与飞行时间都要用，
        分开算等于把同一条弹道积两遍，扫描 1 万次就多花一倍时间。
    """
    th = math.radians(theta_deg)               # float，rad（ODE 内部一律用 rad）
    speed = v0                                 # float，m/s
    y0 = [0.0, 0.0, speed * math.cos(th), speed * math.sin(th)]
    sol = integrate(y0=y0, rtol=rtol, dense=dense, p=p)

    # 坑：事件没触发说明弹丸没在 t_max 内落地（参数写错/速度太小悬空），必须炸而不是返回垃圾值
    if sol.t_events[0].size == 0:
        raise RuntimeError(f"θ={theta_deg} deg, v0={v0} m/s 在 T={sol.t[-1]:.0f} s 内未落地：检查参数")

    x_land, _, vx_land, vy_land = sol.y_events[0][0]
    return {
        "range": float(x_land),                                   # 单位 m，落地水平位移
        "t_flight": float(sol.t_events[0][0]),                    # 单位 s，事件命中时刻即飞行时间
        "h_max": float(np.max(sol.y[1])),                         # 单位 m（按积分步采样，误差 O(dt^2)）
        "v_land": float(math.hypot(vx_land, vy_land)),            # 单位 m/s，落速，用于论文讨论
        "sol": sol,
    }


def range_at(theta_deg, v0, rtol=None, p=PARAMS):
    """落地水平距离，单位 m（建模卡第七节第 1 项，后续一切分析的基础）。

    参数：theta_deg 发射仰角，deg；v0 出膛速度，m/s；rtol 相对容差。
    返回：float，射程，m。同海拔条件下它就是落地点的 x。
    """
    return shot_at(theta_deg, v0, rtol=rtol, p=p)["range"]


def _minimize_scalar(fun, lo, hi, xatol):
    """有界一维极小化的标量封装，返回 (最优 x, 最优 f)。

    参数：fun 目标函数（每调用一次要积一次分，所以不能向量化）；
          lo/hi 搜索区间；xatol x 的容差。
    返回：(float, float)。
    为什么单独封一层：scipy 的类型存根把 minimize_scalar 的返回值标成 object，
        直接写 res.x 会被静态检查报错；在这里一次性压成两个 float，
        调用处只看到干净数值，也不用在 3 处重复写 type: ignore。
    """
    res = minimize_scalar(fun, bounds=(lo, hi), method="bounded", options={"xatol": xatol})
    return float(res.x), float(res.fun)        # type: ignore[attr-defined]


def _root_scalar(fun, a, b, xtol):
    """brentq 的标量封装，返回 float 根。

    参数：fun 目标函数；a/b 搜索区间（必须异号，否则 brentq 直接抛 ValueError）；
          xtol 根的容差。
    返回：float。存根把 brentq 的返回标成 float|RootResults，这里统一压成 float。
    """
    return float(brentq(fun, a, b, xtol=xtol))  # type: ignore[arg-type]


# ==== 6. 分析：退化检验 + 全解求根 + 最短时间 + 灵敏度。本节不 import matplotlib ====
def check_vacuum(theta_deg=None, v0=None):
    """退化检验：把 ρ 置 0（⇒ k=0）后必须退回真空解析解（任务 3 · 证据②）。

    参数：theta_deg 检验角，deg（None 取 30）；v0 检验初速，m/s（None 取 v0_max）。
    返回：dict(r_num, r_ana, rel_r, t_num, t_ana, rel_t)，射程 m、时间 s、偏差无量纲。
    为什么这是唯一能验证方程本身的办法：真空解是解析已知的，
        k=0 时方程退化成纯重力抛体，任何符号错/系数错都会在这里露馅。
    为什么时间和射程都要查：只查射程会漏掉"竖直方向的错误恰好被射程掩盖"的情况；
        解析时间是 t = 2·v0·sinθ/g（与射程同一条抛物线的两个独立投影）。
    """
    theta_deg = 30.0 if theta_deg is None else theta_deg
    v0 = PARAMS["v0_max"] if v0 is None else v0

    rho_base = PARAMS["rho"]                   # float，1.167 kg/m^3（必须在 finally 里复位）
    try:
        PARAMS["rho"] = 0.0                    # 关掉空气阻力
        shot = shot_at(theta_deg, v0, rtol=PARAMS["rtol"])
    finally:
        PARAMS["rho"] = rho_base               # 必须复位，否则后面全算错
    th = math.radians(theta_deg)
    r_ana = v0 ** 2 * math.sin(2.0 * th) / PARAMS["g"]
    t_ana = 2.0 * v0 * math.sin(th) / PARAMS["g"]
    return {"r_num": shot["range"], "r_ana": r_ana, "rel_r": abs(shot["range"] - r_ana) / r_ana,
            "t_num": shot["t_flight"], "t_ana": t_ana,
            "rel_t": abs(shot["t_flight"] - t_ana) / t_ana}


def peak_range(v0=None, rtol=None, p=PARAMS):
    """给定初速下对角度优化的最大射程（射程包络）。

    参数：v0 出膛速度，m/s（None 取 p["v0_max"]）；rtol 相对容差。
    返回：(最大射程 m, 对应仰角 deg)。
    坑：range(θ) 非单调（先升后降），所以不能直接对全区间用黄金分割，
        必须先粗扫定位峰值所在格点，再只在峰附近 ±2 格内做有界一维寻优。
    """
    v0 = p["v0_max"] if v0 is None else v0
    rtol = p["rtol_scan"] if rtol is None else rtol
    th_grid = np.linspace(p["ang_lo"], p["ang_hi"], p["n_angle"])
    r_grid = np.array([range_at(th, v0, rtol=rtol, p=p) for th in th_grid])
    i = int(np.argmax(r_grid))
    step = th_grid[1] - th_grid[0]             # float，0.495 deg
    lo = max(p["ang_lo"], th_grid[i] - 2 * step)
    hi = min(p["ang_hi"], th_grid[i] + 2 * step)

    # 取负号是为了把"求最大射程"变成"求最小值"，method="bounded" 适合这种无导数、只需给区间的场景
    th_peak, neg_r = _minimize_scalar(lambda th: -range_at(th, v0, rtol=rtol, p=p),
                                      lo, hi, p["xatol_angle"])
    return -neg_r, th_peak


def find_angles(R=None, v0=None, rtol=None, p=PARAMS):
    """求出能命中射程 R 的全部仰角（建模卡第七节第 2 项）。

    参数：R 目标射程 m（None 取 p["R_target"]）；v0 初速 m/s（None 取 p["v0_max"]）；
          rtol 相对容差。
    返回：[(theta_deg, t_flight_s), ...]，按角度升序；命中不了返回 []。
    设计决策（修 W1 的漏解窗口）：粗扫 + brentq 在射程峰值附近会漏解——
        若 R 落在"粗扫峰"与"真实峰"之间那道 < 0.03 m 的窄带里，符号变化检测不到，
        会把 2 个解报成 0 个。这里先把峰值用 minimize_scalar 精修出来，
        再在峰值两侧 ±3 个粗步长内补 200 个加密点，然后才做符号变化扫描。
        另加一条切根兜底：R 恰等于包络峰时根是重根（不变号），单独按残差判据报出。
    """
    R = p["R_target"] if R is None else R
    v0 = p["v0_max"] if v0 is None else v0
    rtol = p["rtol_scan"] if rtol is None else rtol

    th_grid = np.linspace(p["ang_lo"], p["ang_hi"], p["n_angle"])
    r_grid = np.array([range_at(th, v0, rtol=rtol, p=p) for th in th_grid])
    step = th_grid[1] - th_grid[0]

    # 第 1 步：峰值精修（粗扫峰 → 真峰），得到加密中心
    _, th_peak = peak_range(v0, rtol=rtol, p=p)

    # 第 2 步：粗网格 + 峰值附近加密，合成一条不再漏解的网格
    dense = np.linspace(max(p["ang_lo"], th_peak - 3 * step),
                        min(p["ang_hi"], th_peak + 3 * step), p["n_refine"])
    # 缓存粗扫结果避免重算：np.unique 保留的浮点值与 th_grid 逐位相同，可以直接当 dict 键
    cache = dict(zip(th_grid.tolist(), r_grid.tolist()))
    th_all = np.unique(np.concatenate([th_grid, dense]))
    g = np.array([cache[th] if th in cache else range_at(float(th), v0, rtol=rtol, p=p)
                  for th in th_all]) - R         # 残差 m，零点即命中

    # 第 3 步：找全部零点（逐个相邻区间看符号变化，全部报出，不只报一个）
    roots = []
    for i in range(len(th_all) - 1):
        if g[i] == 0.0:
            roots.append(float(th_all[i]))
        elif g[i] * g[i + 1] < 0.0:
            roots.append(_root_scalar(lambda th: range_at(th, v0, rtol=rtol, p=p) - R,
                                      float(th_all[i]), float(th_all[i + 1]),
                                      p["xtol_angle"]))

    # 第 3b 步：切根兜底。R 恰好等于射程包络峰（v0 取到 v_min 或更小）时，
    #   g(θ) 只在峰值处触零、不变号，符号扫描必然漏掉——此时若残差足够小就报这个重根。
    if not roots:
        i_min = int(np.argmin(np.abs(g)))
        if abs(g[i_min]) < p["tol_tangent"]:
            roots.append(float(th_peak))       # 重根就在射程包络的峰上，用精修峰角比用网格点准 0.01 deg

    # 第 4 步：去重（峰值相切时相邻区间会各报一次），再补飞行时间
    roots = sorted(set(round(r, 9) for r in roots))
    uniq = []
    for r in roots:
        if not uniq or abs(r - uniq[-1]) > 1e-6:
            uniq.append(r)
    return [(r, shot_at(r, v0, rtol=rtol, p=p)["t_flight"]) for r in uniq]


def check_peak_window(v0=None, p=PARAMS):
    """检验 W1 遗留的"峰值附近漏解窗口"是否已修好。

    参数：v0 检验初速 m/s（None 取 p["v0_max"]）。
    返回：dict(粗扫峰 m, 真实峰 m, 窗口宽度 m, 窗内目标射程 m, 解个数)。
    怎么读：窗口宽度就是"R 落在这一窄带里会把 2 个解报成 0 个"的宽度（W1 实测 0.029 m）；
        窗内目标射程必须解出 2 个根，否则加密网格没起作用。
    """
    v0 = p["v0_max"] if v0 is None else v0
    th_grid = np.linspace(p["ang_lo"], p["ang_hi"], p["n_angle"])
    r_coarse = max(range_at(th, v0, rtol=p["rtol_scan"], p=p) for th in th_grid)
    r_true, _ = peak_range(v0, rtol=p["rtol_scan"], p=p)
    R_probe = 0.5 * (r_coarse + r_true)        # 探针：正落在窗口正中间
    return {"r_coarse": r_coarse, "r_true": r_true, "window": r_true - r_coarse,
            "R_probe": R_probe, "n_roots": len(find_angles(R_probe, v0, p=p))}


def solve_v_min(R=None, p=PARAMS):
    """能命中射程 R 所需的最小初速 v_min（θ_low 与 θ_high 两条支路在此合一）。

    参数：R 目标射程 m（None 取 p["R_target"]）。
    返回：float，v_min，m/s。
    为什么要求它：可行域边界 θ(v0) 从 v0=v_min 处才开始存在，
        小于 v_min 时无论什么角度都打不到 R（射程包络不够）。
    """
    R = p["R_target"] if R is None else R

    def gap(v0):
        """射程包络与目标的差，m。>0 表示这个初速能打到。"""
        return peak_range(v0, p=p)[0] - R

    assert gap(p["v0_lo_scan"]) < 0.0 < gap(p["v0_max"]), \
        f"v_min 无解：R={R} m 不在可达范围内（检查 R 或 PARAMS 的 v0 上下界）"

    # brentq 的坑：要求两端异号；gap 单调递增，所以先粗扫定位括号再交给 brentq
    grid = np.linspace(p["v0_lo_scan"], p["v0_max"], 20)
    for a, b in zip(grid[:-1], grid[1:]):
        if gap(a) * gap(b) < 0.0:
            return _root_scalar(gap, float(a), float(b), p["xtol_v0"])
    raise RuntimeError("v_min 括号搜索失败：gap 在扫描网格上没有符号变化，检查 R 是否恰好等于包络峰")


def scan_feasible(R=None, p=PARAMS):
    """沿 v0 扫出可行域的两条边界支路（即"全部可命中 (v0, θ) 组合"）。

    参数：R 目标射程 m（None 取 p["R_target"]）。
    返回：[dict(v0, theta_low, t_low, theta_high, t_high), ...]，按 v0 升序。
    说明：命中 R 的 (v0, θ) 不是二维区域而是一条/两条曲线——每个 v0 只对应
        至多两个角度；v_min 处两解合一，曲线在此收口。
    """
    R = p["R_target"] if R is None else R
    v_min = solve_v_min(R, p=p)
    rows = []
    for v0 in np.linspace(v_min, p["v0_max"], p["n_v0"]):
        roots = find_angles(R, float(v0), p=p)
        if not roots:
            continue
        low = roots[0]
        high = roots[-1]                       # 只有 1 个解时（v_min 处）高低角合一
        rows.append({"v0": float(v0), "theta_low": low[0], "t_low": low[1],
                     "theta_high": high[0], "t_high": high[1]})
    return rows


def min_time(R=None, v0_pool=None, p=PARAMS):
    """最短飞行时间及对应 (v0, θ)（建模卡第七节第 3 项）。

    参数：R 目标射程 m（None 取 p["R_target"]）；v0_pool 现成的可行域扫描表（省一次扫描）。
    返回：dict(t_min s, v0 m/s, theta_deg, at_bound bool)。
    设计决策：约束是 v0 ≤ v0_max，低角支飞行时间随 v0 单调递减（下方扫描表会验证），
        所以最小值大概率落在 v0=v0_max 的边界上——此时必须明确报"约束取等号"，
        而不是含糊地说"这是全局最小"。
    """
    rows = scan_feasible(R, p=p) if v0_pool is None else v0_pool
    best = min(rows, key=lambda r: min(r["t_low"], r["t_high"]))
    use_low = best["t_low"] <= best["t_high"]
    out = {"t_min": best["t_low"] if use_low else best["t_high"],
           "v0": best["v0"],
           "theta_deg": best["theta_low"] if use_low else best["theta_high"],
           "at_bound": best["v0"] >= p["v0_max"] - 1e-9}

    # 若最优落在内部（非边界），再在相邻两个 v0 之间做一次有界精修
    i = rows.index(best)
    if not out["at_bound"] and 0 < i < len(rows) - 1:
        v0_best, _ = _minimize_scalar(lambda v: find_angles(R, float(v), p=p)[0][1],
                                      rows[i - 1]["v0"], rows[i + 1]["v0"], 1e-4)
        th = find_angles(R, v0_best, p=p)[0]
        out.update({"t_min": th[1], "v0": v0_best, "theta_deg": th[0]})
    return out


def scan_param(key, factors=None, R=None, p=PARAMS):
    """"结果灵敏度 + 补偿灵敏度"双表（银奖门槛项）。

    参数：key 参数字典里的键（"Cd" 或 "rho"）；factors 缩放因子元组；
          R 目标射程 m（None 取 p["R_target"]）。
    返回：[dict(factor, range_m, dtheta_low_deg, dtheta_high_deg, drange_pct), ...]。
    为什么同时给两列：只给"射程变了多少"说明模型脆，只给"角度要补多少"说明不了量级；
        两列一起才是"参数错 10% 需要用多少度补偿"的可操作结论。
    """
    R = p["R_target"] if R is None else R
    factors = p["cd_factors"] if factors is None else factors
    base = p[key]                              # float，基准值（循环里改了必须复位）
    th_ref = find_angles(R, p["v0_max"], p=p)  # 基准两解，deg（射程灵敏度钉在低角解上）
    th_low = th_ref[0][0]
    th_high = th_ref[-1][0]                    # 坑：高角支的基准必须用它自己的角，
                                               # 拿低角解当基准会把 74.9 deg 的支路间距算进补偿量
    r_ref = range_at(th_low, p["v0_max"], rtol=p["rtol"], p=p)
    rows = []
    try:
        for f in factors:
            p[key] = base * f
            r = range_at(th_low, p["v0_max"], rtol=p["rtol"], p=p)
            roots = find_angles(R, p["v0_max"], p=p)
            rows.append({
                "factor": f,
                "range_m": r,
                "dtheta_low_deg": (roots[0][0] - th_low),
                "dtheta_high_deg": (roots[-1][0] - th_high),
                "drange_pct": (r - r_ref) / r_ref * 100.0,
            })
    finally:
        p[key] = base                          # 必须复位，否则后面全算错（算完全局结果都偏）
    return rows


def check_convergence(R=None, p=PARAMS):
    """收敛性检验（任务 3 · 证据①）：收紧 rtol 跨 8 个量级，看角度与飞行时间是否漂。

    参数：R 目标射程 m（None 取 p["R_target"]）。
    返回：[dict(rtol, theta_deg, t_flight, n_steps), ...]，按 rtol 由松到紧。
    怎么读：步数应随 rtol 收紧单调上升（积分器确实在"更费劲地算"），
        而角度只在小数点第 4 位以后动、飞行时间不动 → 结果由物理决定，不由容差决定。
    为什么不用"射程差 < 1 mm"当判据：射程是求根问题的目标值，天然被钉死在 1200 m，
        拿它当判据是循环论证。角度和飞行时间才是真正的输出量。
    """
    R = p["R_target"] if R is None else R
    rows = []
    for rt in RTOL_SERIES:
        theta_deg, t_flight = find_angles(R, p["v0_max"], rtol=rt, p=p)[0]
        n_steps = shot_at(theta_deg, p["v0_max"], rtol=rt, p=p)["sol"].t.size - 1
        rows.append({"rtol": rt, "theta_deg": theta_deg,
                     "t_flight": t_flight, "n_steps": int(n_steps)})
    return rows


# 断网检查用的字样表。坑：整词直写会让本文件自己命中自己（源码里就有那个词），
# 所以拆成两段拼接——与 code-comment-style 的 lint 脚本同一处理方式。
FORBIDDEN_NET_TOKENS = ("re" + "quests", "ur" + "llib", "url" + "open",
                        "sock" + "et", "http" + "://", "https" + "://")

# 允许的依赖：全部是本机 conda 环境里已装的（pip list 可查），且都是科学计算标准库
ALLOWED_MODULES = frozenset({"math", "os", "ast", "numpy", "scipy", "matplotlib"})


def check_offline(path=None):
    """断网可跑检查（任务 3 · 证据③）：机械扫本文件源码，不需要真的拔网线。

    参数：path 待检查的源文件（None 取本文件）。
    返回：list[str]，源码里实际用到的顶层模块名（已排序）。
    查两件事：① import 是否全在白名单内（断网装不上的包直接拦下）；
        ② 源码里有没有联网字样（下载数据、读在线表格这类会在断网时炸）。
    为什么用源码扫描而不是"跑一遍"：跑一遍只能证明"当前网络恰好没用上"，
        而扫描能保证代码里根本没有联网通路——后者才是可复现性的证据。
    """
    path = os.path.abspath(__file__) if path is None else path
    with open(path, encoding="utf-8") as fp:
        src = fp.read()
    mods = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            mods |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods.add(node.module.split(".")[0])
    extra = mods - ALLOWED_MODULES
    assert not extra, f"发现白名单外依赖 {sorted(extra)}：断网机器可能装不上，需改用标准库实现"
    low = src.lower()
    hits = [tok for tok in FORBIDDEN_NET_TOKENS if tok in low]
    assert not hits, f"源码里出现联网相关字样 {hits}：断网环境会失败"
    return sorted(mods)


def elasticity(key="Cd", factors=None, p=PARAMS):
    """弹性系数 S = d(ln R)/d(ln k)：参数变 1%，射程反向变多少。

    参数：key 缩放哪个参数（"Cd"/"rho"，二者等价）；factors 缩放因子（None 取 ±20% 五点）。
    返回：(S_对数拟合, S_端点差分, rows)。S 无量纲；rows 是 scan_param 的返回。
    为什么要两种口径：端点差分（ΔR/R ÷ Δp/p）好向炮兵解释，但对非线性曲线有偏；
        log-log 最小二乘斜率才是"弹性系数"的标准定义。两个都报，差多少也一并说清。
    """
    rows = scan_param(key, factors, p=p)
    x = np.log([r["factor"] for r in rows])            # 无量纲
    y = np.log([r["range_m"] for r in rows])           # 无量纲（先取比值再取对数也可，斜率不变）
    s_fit = float(np.polyfit(x, y, 1)[0])
    r_mid = rows[len(rows) // 2]["range_m"]
    s_ends = (rows[-1]["range_m"] - rows[0]["range_m"]) / r_mid / (rows[-1]["factor"] - rows[0]["factor"])
    return s_fit, float(s_ends), rows


def rho_profile(alt_m, rho_ref, alt_ref, p=PARAMS):
    """ISA 对流层（0-11 km）密度剖面，按题给基准锚定，kg/m^3。

    参数：alt_m 目标海拔 m；rho_ref 基准海拔处的密度 kg/m^3；alt_ref 基准海拔 m。
    返回：float，密度，kg/m^3。校验：锚定 500 m/1.167 时，0 m 得 1.2247、2000 m 得 1.0062。
    为什么用"比值锚定"而不是直接用 ISA 的 1.225 起算：题给 500 m 处是 1.167，
        若从海平面 1.225 起算，500 m 会得 1.16727，第一问的基准射程就从 1200.000 变成 1200.1 m
        ——0.1 m 的偏差会让"基准行"与全文口径对不上。锚定后基准严格 = 1.167，其余海拔按 ISA 比例缩放。
    """
    ratio = (1.0 - p["isa_lapse"] * alt_m) / (1.0 - p["isa_lapse"] * alt_ref)
    return rho_ref * ratio ** p["isa_exp"]


def scan_altitude(theta_deg, altitudes, v0=None, p=PARAMS):
    """顺带扫海拔：只改空气密度，看固定角度下的射程怎么变。

    参数：theta_deg 固定仰角 deg（用第一问的低角解，保证与基准角可比）；
          altitudes 海拔序列 m；v0 初速 m/s（None 取 p["v0_max"]）。
    返回：[dict(alt_m, rho, range_m, drange_pct), ...]。
    为什么值得单独做：ρ 是本题除 C_d 外最不确定的输入，而题目第二问直接问"不同海拔"——
        这条曲线就是第二问的先导结果，也顺带给"为什么不忽略密度变化"一个定量答案。
    """
    v0 = p["v0_max"] if v0 is None else v0
    rho_base = p["rho"]                       # float，1.167（500 m 锚点，必须在 finally 里复位）
    rows = []
    try:
        r_ref = range_at(theta_deg, v0, rtol=p["rtol"], p=p)   # 基准射程：锚点海拔处
        for alt in altitudes:
            p["rho"] = rho_profile(alt, rho_base, p["alt_ref"], p=p)
            r = range_at(theta_deg, v0, rtol=p["rtol"], p=p)
            rows.append({"alt_m": float(alt), "rho": p["rho"], "range_m": r,
                         "drange_pct": (r - r_ref) / r_ref * 100.0})
    finally:
        p["rho"] = rho_base                   # 必须复位，否则后面全算错
    return rows


def drift_at(theta_deg, v0, wind=None, rtol=None, p=PARAMS):
    """侧风下的落点：侧向漂移 / 飞行时间 / 纵向射程（3D 相对速度阻力）。

    参数：theta_deg 仰角 deg；v0 初速 m/s；wind 横风 m/s（None 取 p["wind"]，0 = 静风）；
          rtol 相对容差。
    返回：dict(drift_m, t_flight, range_m, drift_naive_m)。
        drift_naive_m = w·t 是"忽略侧向加速过程"的粗估，两者一起报才能说清差在哪。
    为什么侧向漂移不等于 w·t：弹丸的侧向速度不是瞬时等于风速，而是按
        dvz/dt = −k|v_rel|(vz − w) 逐步逼近，时间常数约 1/(k·v) ≈ 4.3 s，
        而飞行只有 3.71 s —— 所以真实漂移显著小于 w·t。粗估会高估，不是低估。
    """
    th = math.radians(theta_deg)
    w = p["wind"] if wind is None else wind
    rtol = p["rtol"] if rtol is None else rtol
    y0 = [0.0, 0.0, 0.0, v0 * math.cos(th), v0 * math.sin(th), 0.0]
    sol = solve_ivp(lambda t, y: rhs_wind(t, y, p=p, wind=w), (0.0, p["t_max"]), y0,
                    method="RK45", rtol=rtol, atol=p["atol"], events=_landed)
    if sol.t_events[0].size == 0:
        raise RuntimeError(f"θ={theta_deg} deg, v0={v0} m/s, wind={w} m/s 未落地：检查参数")
    x_land, _, z_land, _, _, _ = sol.y_events[0][0]
    t_land = float(sol.t_events[0][0])
    return {"drift_m": float(z_land), "t_flight": t_land,
            "range_m": float(x_land), "drift_naive_m": w * t_land}


def scan_wind(theta_deg, winds, v0=None, p=PARAMS):
    """扫侧风速度，给漂移量级表（任务书问"不同风量"）。

    参数：theta_deg 仰角 deg；winds 风速序列 m/s；v0 初速 m/s（None 取 p["v0_max"]）。
    返回：[dict(wind, drift_m, drift_naive_m, frac_pct), ...]。
        frac_pct = 漂移 / 射程 × 100，这是论文里最有说服力的那个数（横向偏差占射程的百分之几）。
    """
    v0 = p["v0_max"] if v0 is None else v0
    rows = []
    for w in winds:
        d = drift_at(theta_deg, v0, wind=float(w), p=p)
        rows.append({"wind": float(w), "drift_m": d["drift_m"],
                     "drift_naive_m": d["drift_naive_m"],
                     "frac_pct": d["drift_m"] / p["R_target"] * 100.0})
    return rows


# ==== 7. 出图 + 主流程：放最后，因为要改的最多 ====
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGS = os.path.join(ROOT, "figs")
# 为什么用 __file__ 向上定位而不是相对路径：从仓库根或 code/ 下调用都出到同一个 figs/

# 出图铁律（队内规范）：一个 PNG 只放一张图；两张图只有在"必须一起看才有意义"时才配对
# （低角/高角弹道就是这种配对：它们回答的是同一个问题的两个解）。


def _save(fig, name):
    """统一收尾：300 dpi、紧凑布局、关闭图窗（不关会累积内存并让 Agg 后端卡住）。"""
    fig.tight_layout()
    out = os.path.join(FIGS, name)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    return out


def plot_boundary(rows, v_min, out=None):
    """可行域边界图：θ_low(v0) 与 θ_high(v0) 两条支路 + 可达到的夹角带。

    参数：rows scan_feasible 的返回；v_min 最小可行初速 m/s。
    返回：图片绝对路径。
    """
    v0s = [r["v0"] for r in rows]
    lo = [r["theta_low"] for r in rows]
    hi = [r["theta_high"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.fill_betweenx(v0s, lo, hi, color="0.85", alpha=0.6,
                     label=r"$\theta$ between the two roots: range $>1200$ m")
    ax.plot(lo, v0s, lw=1.6, label=r"low-angle branch $\theta_{\rm low}(v_0)$")
    ax.plot(hi, v0s, lw=1.6, label=r"high-angle branch $\theta_{\rm high}(v_0)$")
    ax.plot([lo[0]], [v0s[0]], "o", ms=4, color="k")
    ax.annotate(f"$v_{{0,min}}$ = {v_min:.1f} m/s\n$\\theta$ = {lo[0]:.2f}$^\\circ$",
                xy=(lo[0], v0s[0]), xytext=(18, 12), textcoords="offset points",
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_xlabel(r"Elevation angle $\theta$ (deg)")
    ax.set_ylabel(r"Muzzle velocity $v_0$ (m/s)")
    ax.set_title(r"All $(v_0,\theta)$ combinations hitting $R=1200$ m ($\rho$ = 1.167 kg/m$^3$)",
                 fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper center")
    ax.grid(alpha=0.3)
    return _save(fig, "W2_q1_1_boundary.png")


def plot_range_theta(theta_roots, R, v0, out=None):
    """R(θ) 曲线：展示射程非单调、两个根的位置（低角解/高角解）。

    参数：theta_roots find_angles 的返回；R 目标射程 m；v0 初速 m/s。
    返回：图片绝对路径。
    """
    th = np.linspace(PARAMS["ang_lo"], PARAMS["ang_hi"], 400)
    r = np.array([range_at(t, v0, rtol=PARAMS["rtol_scan"]) for t in th])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(th, r, lw=1.5, label=rf"$R(\theta)$ at $v_0={v0:.0f}$ m/s")
    ax.axhline(R, ls="--", lw=1.0, color="C3", label=rf"target $R={R:.0f}$ m")
    for k, (t, _) in enumerate(theta_roots):
        ax.plot([t], [R], "o", ms=5, color="C3", zorder=5)
        ax.annotate(rf"$\theta_{k + 1}={t:.2f}^\circ$", xy=(t, R), xytext=(6, 10),
                    textcoords="offset points", fontsize=8)
    ax.set_xlabel(r"Elevation angle $\theta$ (deg)")
    ax.set_ylabel(r"Range $R$ (m)")
    ax.set_title(r"Range is non-monotonic in $\theta$: two roots for $R=1200$ m", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, "W2_q1_2_range_theta.png")


def plot_traj(theta_deg, v0, tag, out=None):
    """单条弹道图（低角解与高角解各出一张）。

    参数：theta_deg 仰角 deg；v0 初速 m/s；tag 分支标签（"low"/"high"，进文件名）。
    返回：图片绝对路径。
    为什么不把两条画进同一张：低角解顶点 17 m、高角解 2274 m，相差 134 倍，
        同坐标轴下低角那条会被压成贴着地面的一条直线，等于没画。
        "两个解命中同一目标"这件事由 W2_q1_2_range_theta.png 承担。
    """
    shot = shot_at(theta_deg, v0, rtol=PARAMS["rtol"], dense=True)
    sol = shot["sol"]                          # dense_output 打开后才有 .sol 插值器
    time = np.linspace(0.0, shot["t_flight"], 600)
    xs, hs = sol.sol(time)[0], sol.sol(time)[1]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(xs, hs, lw=1.5,
            label=rf"$\theta={theta_deg:.2f}^\circ$, $t={shot['t_flight']:.2f}$ s, "
                  rf"$h_{{\max}}={shot['h_max']:.2f}$ m")
    ax.axhline(0.0, lw=0.8, color="k")
    ax.set_xlabel(r"Horizontal distance $x$ (m)")
    ax.set_ylabel(r"Height $h$ (m)")
    ax.set_title(rf"Trajectory ({tag}-angle solution) at $v_0=450$ m/s, quadratic drag", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    return _save(fig, f"W2_q1_{3 if tag == 'low' else 4}_traj_{tag}.png")


def plot_flight_time(rows, best, out=None):
    """飞行时间 vs 初速（低/高角两支）+ 二维约束下的最短时间点。

    参数：rows scan_feasible 的返回；best min_time 的返回。
    返回：图片绝对路径。
    """
    v0s = [r["v0"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(v0s, [r["t_low"] for r in rows], lw=1.5, label="low-angle branch")
    ax.plot(v0s, [r["t_high"] for r in rows], lw=1.5, ls="--", label="high-angle branch")
    ax.plot([best["v0"]], [best["t_min"]], "o", ms=5, color="C3", zorder=5)
    ax.annotate(f"min $t$ = {best['t_min']:.3f} s", xy=(best["v0"], best["t_min"]),
                xytext=(-70, 15), textcoords="offset points", fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_xlabel(r"Muzzle velocity $v_0$ (m/s)")
    ax.set_ylabel(r"Flight time $t$ (s)")
    ax.set_title(r"Flight time to $R=1200$ m (constraint $v_0\leq450$ m/s)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, "W2_q1_5_flight_time.png")


def plot_scan_k(rows, out=None):
    """结果灵敏度：k 缩放（等价 C_d 缩放 / ρ 缩放）对射程的影响，单指标单图。

    参数：rows scan_param 的返回。
    返回：图片绝对路径。
    为什么只画一条：k = ρC_dA/(2m) 只依赖乘积，C_d 与 ρ 同比例缩放给出同一条曲线，
        画两张一模一样的图是噪音。这个"不可分辨"本身要写进论文的 Weaknesses。
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r["factor"] for r in rows], [r["range_m"] for r in rows],
            "o-", lw=1.5, ms=4)
    ax.axhline(rows[len(rows) // 2]["range_m"], ls=":", lw=0.8, color="0.5",
               label=r"baseline $k=5.212\times10^{-4}$ m$^{-1}$")
    ax.set_xlabel(r"Scale factor on $k$ (= $C_d$ or $\rho$ scaling)")
    ax.set_ylabel(r"Range at fixed $\theta_{\rm low}$ (m)")
    ax.set_title(r"Result sensitivity: range response to drag coefficient $k$", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, "W2_q1_6_scan_k.png")


def plot_compensate(rows, out=None):
    """补偿灵敏度：k 缩放后要重新命中 1200 m，两个支路各需要补多少度。

    参数：rows scan_param 的返回。
    返回：图片绝对路径。
    为什么这比"射程变了几米"有用：炮兵能调的是角度，不是 C_d；
        这张图直接给出"我若把阻力估错了 20%，角度该修正几度"。
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r["factor"] for r in rows], [r["dtheta_low_deg"] for r in rows],
            "o-", lw=1.5, ms=4, label=r"low-angle branch $\Delta\theta_{\rm low}$")
    ax.plot([r["factor"] for r in rows], [r["dtheta_high_deg"] for r in rows],
            "s--", lw=1.5, ms=4, label=r"high-angle branch $\Delta\theta_{\rm high}$")
    ax.axhline(0.0, lw=0.8, color="k")
    ax.set_xlabel(r"Scale factor on $k$ (= $C_d$ or $\rho$ scaling)")
    ax.set_ylabel(r"Angle correction to keep $R=1200$ m (deg)")
    ax.set_title(r"Compensating angle: how much $\theta$ must change when $k$ is wrong", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, "W2_q1_7_compensate.png")


def plot_altitude(rows, out=None):
    """顺带扫海拔：固定仰角下射程随海拔（即空气密度）的变化。

    参数：rows scan_altitude 的返回。
    返回：图片绝对路径。
    为什么纵轴用"射程"而不是"密度"：论文读者关心的是落点，密度只是中介变量；
        密度值写进图注，不占主坐标。
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r["alt_m"] for r in rows], [r["range_m"] for r in rows], "o-", lw=1.5, ms=4)
    base = rows[[r["alt_m"] for r in rows].index(PARAMS["alt_ref"])]
    ax.axhline(base["range_m"], ls=":", lw=0.8, color="0.5", label="500 m baseline (1200 m)")
    # 三个点名海拔的数字统一放进右下角文本框：逐点标注会撞上曲线和 250 m 的采样点
    label = "\n".join(f"{r['alt_m']:.0f} m: {r['range_m']:.1f} m ({r['drange_pct']:+.2f}%)"
                      for r in rows if r["alt_m"] in PARAMS["alt_ref_set"])
    ax.text(0.45, 0.04, label, transform=ax.transAxes, fontsize=7, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))
    ax.margins(y=0.10)                        # 留白，否则文本框会压到坐标轴上
    ax.set_xlabel(r"Altitude (m)")
    ax.set_ylabel(r"Range at $\theta=2.65^\circ$ (m)")
    ax.set_title(r"Range versus altitude through the ISA density profile", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    return _save(fig, "W2_q1_8_altitude.png")


def plot_elasticity(x, y, slope, out=None):
    """弹性系数图：归一化双对数坐标下，斜率即 S = dlnR/dlnk。

    参数：x 各点的 ln(k 缩放因子)（无量纲）；y 各点的 ln(R/R_base)（无量纲）；
          slope 拟合斜率，无量纲（标在图上）。
    返回：图片绝对路径。
    为什么用归一化双对数而不是普通坐标：普通坐标只能看出"有下降"，
        双对数下斜率直接读成"参数变 1%，射程变 S%"，这正是弹性系数的定义。
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, "o", ms=5, label="simulation")
    xs = np.linspace(min(x), max(x), 50)
    ax.plot(xs, slope * xs, "-", lw=1.2, label=rf"fit: $S={slope:.3f}$")
    ax.set_xlabel(r"$\ln(\kappa/k_0)$  (drag coefficient scaling, dimensionless)")
    ax.set_ylabel(r"$\ln(R/R_0)$  (relative range change)")
    ax.set_title(r"Elasticity of range with respect to the drag group $k$", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, "W2_q1_9_elasticity.png")


def plot_wind(rows, out=None):
    """侧风漂移：相对速度模型的严格解 vs 粗估 w·t（同一指标、两个模型，必须同图对照）。

    参数：rows scan_wind 的返回。
    返回：图片绝对路径。
    为什么两条线要画在一起：单独画一条看不出"粗估错了多少"；
        两条同图，间距就是被忽略的侧向加速过程所贡献的假象漂移。
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r["wind"] for r in rows], [r["drift_m"] for r in rows],
            "o-", lw=1.5, ms=4, label="relative-velocity drag model")
    ax.plot([r["wind"] for r in rows], [r["drift_naive_m"] for r in rows],
            "s--", lw=1.2, ms=4, label=r"naive estimate $w\,t_{\rm flight}$")
    ax.set_xlabel(r"Crosswind speed $w$ (m/s)")
    ax.set_ylabel(r"Lateral drift at impact (m)")
    ax.set_title(r"Crosswind drift for the low-angle solution ($t=3.71$ s)", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, "W2_q1_10_wind_drift.png")


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)            # 先建目录，否则 savefig 报错

    # ---- 自检 1/2：量纲与量级（建模卡第八节，写 rhs 前就该过）----
    check_units()

    # ---- 任务 3 证据③：断网可跑（源码级检查，不需要真拔网线）----
    mods = check_offline()
    print(f"offline check       : OK  (imports: {', '.join(mods)}; no network calls)")

    # ---- 任务 3 证据②：退化到真空解析解 ----
    vac = check_vacuum()
    print(f"vacuum range        : numeric {vac['r_num']:.3f} m  analytic {vac['r_ana']:.3f} m  "
          f"diff {vac['rel_r'] * 100:.2e} %")
    print(f"vacuum time         : numeric {vac['t_num']:.5f} s  analytic {vac['t_ana']:.5f} s  "
          f"diff {vac['rel_t'] * 100:.2e} %")
    assert max(vac["rel_r"], vac["rel_t"]) < 1e-4, \
        f"真空退化检验偏差 {max(vac['rel_r'], vac['rel_t']) * 100:.2e} % 超过 0.01% 门槛"

    # ---- 主结果 1：v0=450 时的全部可行角度（低角解 + 高角解）----
    # 注：终端输出一律用 ASCII 的 theta/delta 而不是希腊字母——PowerShell 默认 GBK 代码页会乱码
    roots = find_angles()
    for k, (t, tf) in enumerate(roots):
        det = shot_at(t, PARAMS["v0_max"], rtol=PARAMS["rtol"])
        print(f"angles @ v0=450     : #{k + 1} theta={t:8.4f} deg  t_flight={tf:7.4f} s  "
              f"h_max={det['h_max']:8.2f} m  v_land={det['v_land']:6.1f} m/s  "
              f"range={det['range']:9.3f} m")
    assert len(roots) == 2, f"R=1200 m 应有两个解（低角+高角），实得 {len(roots)} 个"

    # ---- 主结果 2：最小可行初速与可行域边界 ----
    v_min = solve_v_min()
    env, th_env = peak_range(v_min)
    print(f"v_min(R=1200)       : {v_min:.4f} m/s @ {th_env:.4f} deg "
          f"(envelope = {env:.3f} m = target, so the two roots merge here)")

    rows = scan_feasible()
    print("feasible boundary   : " + " | ".join(
        f"v0={r['v0']:.0f}: {r['theta_low']:.3f}/{r['theta_high']:.3f} deg" for r in rows[:3])
        + f" ... v0={rows[-1]['v0']:.0f}: {rows[-1]['theta_low']:.3f}/{rows[-1]['theta_high']:.3f} deg")

    # ---- 主结果 3：最短飞行时间 ----
    # 先验证"低角支飞行时间随 v0 单调递减"，否则"最短在 v0 上界"这个结论不成立
    t_low = [r["t_low"] for r in rows]
    mono = all(b < a for a, b in zip(t_low[:-1], t_low[1:]))
    assert mono, "低角支飞行时间不单调：min_time 必须做区间寻优而不是取边界"
    best = min_time(v0_pool=rows)
    print(f"min flight time     : {best['t_min']:.4f} s @ v0={best['v0']:.1f} m/s, "
          f"theta={best['theta_deg']:.4f} deg (constraint-bound={'yes' if best['at_bound'] else 'no'})")

    # ---- 自检 4：W1 的漏解窗口是否修好 ----
    win = check_peak_window()
    print(f"peak-miss window    : coarse {win['r_coarse']:.4f} m vs true {win['r_true']:.4f} m "
          f"-> window {win['window'] * 1000:.2f} mm; probe R={win['R_probe']:.4f} m -> "
          f"{win['n_roots']} roots")
    assert win["n_roots"] == 2, "峰值窗口内仍有漏解：加密网格没起作用"

    # ---- 任务 3 证据①：收敛性（角度 / 飞行时间 / 积分步数三列一起看）----
    conv = check_convergence()
    for r in conv:
        print(f"convergence {r['rtol']:.0e}     : theta={r['theta_deg']:.6f} deg  "
              f"t_flight={r['t_flight']:.6f} s  steps={r['n_steps']:4d}")
    spread_theta = max(r["theta_deg"] for r in conv) - min(r["theta_deg"] for r in conv)
    spread_t = max(r["t_flight"] for r in conv) - min(r["t_flight"] for r in conv)
    assert spread_theta < 1e-3, f"角度随 rtol 漂移 {spread_theta:.2e} deg：积分器在这道题上不可靠"
    assert spread_t < 1e-4, f"飞行时间随 rtol 漂移 {spread_t:.2e} s：积分器在这道题上不可靠"
    # 步数必须单调上升：否则说明"收紧 rtol"没有真的让积分器更费劲，收敛表就是白做的
    assert all(b["n_steps"] > a["n_steps"] for a, b in zip(conv[:-1], conv[1:])), \
        "积分步数未随 rtol 收紧上升：容差可能没传进 solve_ivp"

    # ---- 主结果 4：灵敏度双表（C_d 与 ρ 应逐行相同：k 只依赖两者乘积）----
    tab_cd = scan_param("Cd")
    tab_rho = scan_param("rho")
    for r in tab_cd:
        print(f"scan k ({r['factor']:.1f}x)     : range {r['range_m']:8.2f} m "
              f"({r['drange_pct']:+6.2f}%)  d_theta_low {r['dtheta_low_deg']:+7.3f} deg  "
              f"d_theta_high {r['dtheta_high_deg']:+7.3f} deg")
    for a, b in zip(tab_cd, tab_rho):
        assert abs(a["range_m"] - b["range_m"]) < 1e-9, \
            "C_d 与 rho 同比例缩放结果不一致：k 只依赖乘积，说明复位或缓存出了问题"
    assert abs(tab_cd[len(tab_cd) // 2]["drange_pct"]) < 1e-9, \
        "灵敏度基准行应零变化：说明 PARAMS 复位没做对"

    # ---- 主结果 5：弹性系数（参数变 1%，射程反向变几个百分点）----
    s_fit, s_ends, tab_e = elasticity("Cd")
    print(f"elasticity S        : log-log fit {s_fit:+.3f}  /  end-point {s_ends:+.3f} "
          f"(per 1% in k -> {s_fit:+.2f}% in range)")

    # ---- 主结果 6：顺带扫海拔（只改密度）----
    tab_alt = scan_altitude(roots[0][0], PARAMS["alt_grid"])
    for r in tab_alt:
        mark = "  <- baseline" if r["alt_m"] == PARAMS["alt_ref"] else ""
        print(f"altitude {r['alt_m']:.0f} m     : rho={r['rho']:.4f} kg/m^3  "
              f"range={r['range_m']:8.2f} m ({r['drange_pct']:+6.2f}%){mark}")
    assert abs(tab_alt[[r["alt_m"] for r in tab_alt].index(PARAMS["alt_ref"])]["drange_pct"]) < 1e-9, \
        "海拔扫描的锚点行必须零变化：说明密度剖面没锚在题给的 500 m/1.167 上"

    # ---- 主结果 7：侧风漂移（先做静风回归，再扫风）----
    calm = drift_at(roots[0][0], PARAMS["v0_max"], wind=0.0)
    assert abs(calm["drift_m"]) < 1e-9, "静风下侧向漂移必须严格为 0"
    assert abs(calm["range_m"] - range_at(roots[0][0], PARAMS["v0_max"])) < 1e-6, \
        "静风下 3D 射程与第一问不一致：rhs_wind 没有严格退化为 rhs"
    tab_wind = scan_wind(roots[0][0], PARAMS["wind_grid"])
    # 坑：静风断言过不了"风速没接进 ODE"这种错——它和静风结果完全一致，看起来全对。
    #   所以再加一条：风速加倍，漂移必须也加倍（线性响应），否则说明风根本没进模型。
    assert tab_wind[2]["drift_m"] > 1.0, "有风时漂移仍为 0：风速没接进 rhs_wind"
    assert abs(tab_wind[3]["drift_m"] / tab_wind[2]["drift_m"] - 1.5) < 0.02, \
        "漂移不随风速线性变化：检查相对速度阻力项"
    for r in tab_wind:
        print(f"wind {r['wind']:5.1f} m/s    : drift {r['drift_m']:7.2f} m "
              f"({r['frac_pct']:5.2f}% of range)  naive w*t {r['drift_naive_m']:7.2f} m")

    # ---- 出图（一图一文件）----
    print(f"[fig] {plot_boundary(rows, v_min)}")
    print(f"[fig] {plot_range_theta(roots, PARAMS['R_target'], PARAMS['v0_max'])}")
    print(f"[fig] {plot_traj(roots[0][0], PARAMS['v0_max'], 'low')}")
    print(f"[fig] {plot_traj(roots[-1][0], PARAMS['v0_max'], 'high')}")
    print(f"[fig] {plot_flight_time(rows, best)}")
    print(f"[fig] {plot_scan_k(tab_cd)}")
    print(f"[fig] {plot_compensate(tab_cd)}")
    print(f"[fig] {plot_altitude(tab_alt)}")
    x_e = np.log([r["factor"] for r in tab_e])
    y_e = np.log([r["range_m"] / tab_e[len(tab_e) // 2]["range_m"] for r in tab_e])
    print(f"[fig] {plot_elasticity(x_e, y_e, s_fit)}")
    print(f"[fig] {plot_wind(tab_wind)}")

# ============================================================================
# 怎么验证跑对了（照这 5 条对，全过才算通）：
#   1. unit check 行里 k=5.21e-04 1/m、a_drag(450)=105.6 m/s^2  —— 与建模卡第四节/第五节逐位一致
#   2. vacuum check 的 diff 与 rtol=1e-9 同量级（~1e-8 %）      —— 方程本身没错
#   3. angles @ v0=450 应得到 2 个解，低角约 2.6 deg、高角约 77.6 deg，
#      低角飞行时间约 3.7 s                                    —— 与 W1 旧口径 2.72/77.06/3.77 的
#      差异全部来自 ρ 由 1.225 改成 1.167，方向上是射程略增、角度略降
#   4. peak-miss window 的"解出 2 个根"                          —— W1 的 0.029 m 漏解已修
#   5. rtol 从 1e-6 收到 1e-10 射程变化 < 1 mm                   —— 数值结果不受容差支配
# ============================================================================
