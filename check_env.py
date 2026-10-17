# -*- coding: utf-8 -*-
"""check_env.py —— UPC 备赛环境自检脚本（数值计算 / 建模优化 / 绘图 全链路）

用途：
    在 conda 环境 upc (Python 3.11.17) 里跑一遍，确认「能算、能解、能画中文图」。
    换机器、重装环境、或比赛前一晚，都先跑这个脚本。

输入：
    无。不需要任何外部数据文件。

输出：
    1) 终端打印 19 个必装库的导入结果与版本号；
    2) 终端打印 8 项数值/换算自检的实际值；
    3) 生成 _env_check/env_report.txt（版本快照，可直接存档）；
    4) 生成 _env_check/fig_check.png（中文标题折线图，用于验证字体不出现方框）。

跑完应看到的自检数字（关键验收点，对不上说明装错了）：
    [1] quad(sin, 0, pi)                 -> 2.0            （绝对误差 < 1e-10）
    [2] minimize(Rosenbrock, 起点 -1.2,1)-> x ≈ [1.000, 1.000]（Nelder-Mead 只到 1e-4 量级，
                                            最优值 ~1e-9，判据是 < 1e-8）
    [3] numpy 特征值 (上三角 [[1,1],[0,2]]) -> 1.0, 2.0     （特征值之和 3.0）
    [4] pint 单位换算 1 mile             -> 1609.344 m
    [5] uncertainties 误差传播 (2±0.1)+(3±0.2) -> 5.0000 ± 0.2236
    [6] cvxpy  min x+y s.t. x+y>=1       -> 1.0
    [7] pulp   min x+y s.t. x+y>=1       -> 1.0
    [8] 中文字体解析                       -> 命中 msyh / simhei 字体文件
    [9] 必装库导入成功数                   -> 19 / 19

退出码：全部通过 0，有任一失败 1（便于写进批处理做门禁）。
"""

# ==== 0. 标准库与产物目录 ====

import sys
from pathlib import Path

# 变量 OUT_DIR：pathlib.Path，自检产物目录
#   示例值 WindowsPath('d:/UPC备赛/W1/_env_check')
#   为什么放脚本同级而不是系统临时目录：报告和 PNG 要能当场交出去/存档，
#     临时目录容易被系统清理，赛前重装机器后翻不到
OUT_DIR = Path(__file__).resolve().parent / "_env_check"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 控制台编码兜底：Windows 默认 GBK，若输出里出现 GBK 无法表示的字符会抛 UnicodeEncodeError
#   用 reconfigure 切 utf-8 并加 errors="replace"（最坏情况只显示成 ? 而不中断）；
#   老解释器没有 reconfigure 方法，故用 try 兜住
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

# 注：不要在 PowerShell 里用 `conda run -n upc python check_env.py` 跑本脚本。
#   conda run 会先按 GBK 解码子进程输出再打印，遇到特殊字符会自己抛 UnicodeEncodeError。
#   正确姿势是先激活环境，再用 python 直跑（见文件末尾"怎么验证跑对了"）。


# ==== 1. 必装库清单与导入检查 ====

import importlib                    # 按字符串名动态导入，实现"缺哪个报哪个"
import importlib.metadata as imeta   # 读已安装发行版的版本号（importlib.metadata）

# 变量 LIB_USAGE：dict[str, str]（键=import 名，值=一句话用途）
#   示例值 {"numpy": "数组/矩阵运算（其余数值库的底座）", ...}
#   为什么是这 19 个：前 13 项 = UPC 2026 计划书指定的核心 7 包 + 补充 A/B/C 里
#     参与计算的部分，一个不能少；后 6 项是额外补的建模/优化/交互绘图库，
#     属于"武器库"，缺了不影响基本盘。jupyter 这类纯工具不参与计算，故不列入
LIB_USAGE = {
    # ---- 计划书核心 7 包 ----
    "numpy": "数组/矩阵运算（其余数值库的底座）",
    "scipy": "数值积分、方程求根、优化、插值、信号处理",
    "sympy": "符号推导：解析解、公式化简、LaTeX 输出",
    "matplotlib": "绘图底座，出论文用图",
    "pandas": "表格清洗与聚合，CSV/Excel 读写",
    "pint": "物理量单位换算与量纲检查",
    "uncertainties": "误差传播：ufloat 自动合成不确定度",
    # ---- 计划书补充 A：单位 / 数据 I-O ----
    "pint_pandas": "pint 单位与 pandas DataFrame 打通",
    "openpyxl": "读写 .xlsx（比赛数据常见格式）",
    # ---- 计划书补充 B：统计 / 拟合 / 绘图 ----
    "statsmodels": "统计检验、回归诊断、时间序列",
    "seaborn": "统计图快捷封装（热力图、分布图）",
    # ---- 计划书补充 C：交互 / 效率 ----
    "tqdm": "长循环进度条，判断程序卡没卡",
    "numba": "JIT 编译热点循环（首次编译有开销，长循环才划算）",
    # ---- 额外补（非计划书要求，属武器库）----
    "sklearn": "回归/分类/聚类/降维，建模主力",
    "networkx": "图论建模：最短路、生成树、网络流",
    "cvxpy": "凸优化建模（LP/QP/SOCP）",
    "pulp": "线性/整数规划建模",
    "sko": "智能优化算法：遗传/粒子群/模拟退火（pip 包名 scikit-opt，模块名 sko）",
    "plotly": "交互式图表，导出 HTML 可缩放",
}

# 变量 CHECK_MODULES：list[str]，参与导入检查的模块名，直接用清单的键
#   示例值 ["numpy", "scipy", ..., "sko", ...]
#   为什么键写成纯模块名：import 必须用 sko 这种模块名，
#     而 pip 发行版名是 scikit-opt，两者不一致的问题交给下面的 dist_name 映射处理
CHECK_MODULES = list(LIB_USAGE)

# 变量 DIST_NAME_FIX：dict[str, str]，模块名 -> pip 发行版名（不一致时才需要）
#   示例值 {"sklearn": "scikit-learn", "sko": "scikit-opt", "pint_pandas": "pint-pandas"}
#   为什么需要：imeta.version("sklearn") 会抛 PackageNotFoundError，必须查发行版名
DIST_NAME_FIX = {"sklearn": "scikit-learn", "sko": "scikit-opt", "pint_pandas": "pint-pandas"}

# 变量 imported：dict[str, str]，模块名 -> 版本号；导入失败的不进字典
#   示例值 {"numpy": "2.4.6", ...}
#   为什么记录版本而不是布尔值：现场排查时"装的是哪个版本"往往就是答案
imported = {}

for module_name in CHECK_MODULES:
    # importlib.import_module(name) -> module 对象
    #   作用：等价于 import numpy，但名字可以是运行时算出来的字符串
    #   关键参数：name 用点分模块名（"scipy.optimize"），不要写文件路径
    #   坑：导入失败抛 ModuleNotFoundError（ImportError 的子类），必须捕获，
    #       否则一个包缺失会让整个自检脚本崩掉、后面的检查全看不到
    try:
        importlib.import_module(module_name)
    except ImportError as exc:
        print(f"[FAIL] {module_name:<12} 导入失败: {exc}")
        continue

    # 版本号：用发行版名查（sklearn -> scikit-learn、sko -> scikit-opt 名字不一致，故做映射）
    dist_name = DIST_NAME_FIX.get(module_name, module_name)
    try:
        version = imeta.version(dist_name)
    except imeta.PackageNotFoundError:
        version = "未知"
    imported[module_name] = version
    print(f"[ OK ] {module_name:<12} {version:<10} {LIB_USAGE.get(module_name, '')}")

print(f"\n必装库导入成功：{len(imported)} / {len(CHECK_MODULES)}")

# 缺包就到此为止：后面的数值/绘图自检都依赖这些库，硬跑只会抛 ModuleNotFoundError，
#   反而把"到底缺哪个"的信息淹掉，所以先给出补装命令再退出
missing_modules = [name for name in CHECK_MODULES if name not in imported]
if missing_modules:
    print("\n以下库缺失，请先补装后重新自检：")
    for module_name in missing_modules:
        print(f"  pip install {DIST_NAME_FIX.get(module_name, module_name)}")
    sys.exit(1)


# ==== 2. 数值计算自检（numpy / scipy）====

# 设计决策：不"随便跑个数"，而是选 3 道有解析解、能一眼判对错的题。
#   这样自检不依赖任何参考输出文件，赛前机器上也能徒手核对。

import numpy as np                                  # 数组运算，本节所有计算的载体
from scipy import integrate, optimize               # 数值积分与无约束优化

# --- 2.1 数值积分：∫₀^π sin(x) dx 解析值为 2 ---
# integrate.quad(func, a, b) -> (积分值 float, 绝对误差估计 float)
#   作用：自适应高斯-克朗罗德积分，适合光滑函数的高精度求值
#   关键参数：func 要能接受 ndarray（写成 np.sin 这种向量化形式）；a、b 为标量边界
#   坑：返回的是元组，第一项才是积分值；无限区间写 np.inf，不要写 1e30
integral_value, integral_error = integrate.quad(np.sin, 0.0, np.pi)

# --- 2.2 无约束优化：Rosenbrock 函数极小点在 (1, 1) ---
# optimize.minimize(fun, x0, method) -> OptimizeResult（.x 解、.fun 目标值、.success 是否收敛）
#   作用：局部无约束/有约束优化，建模调参、最小二乘拟合的通用入口
#   关键参数：x0 是起点 ndarray；method="Nelder-Mead" 无需梯度（最稳），
#             "BFGS" 更快但要求 fun 光滑且最好给 jac
#   坑：不传 method 默认 BFGS；起点给得离极小点太远会不收敛，本例起点 (-1.2, 1) 是经典考验点
rosenbrock = lambda p: 100.0 * (p[1] - p[0] ** 2) ** 2 + (1.0 - p[0]) ** 2
opt_result = optimize.minimize(rosenbrock, x0=np.array([-1.2, 1.0]), method="Nelder-Mead")

# --- 2.3 线性代数：上三角矩阵 [[1,1],[0,2]] 特征值为 1 和 2 ---
# np.linalg.eigvals(A) -> ndarray[complex]，特征值数组（顺序未定义）
#   作用：求特征值，用于稳定性分析、主成分、马尔可夫链稳态判断
#   关键参数：A 必须是方阵；非对称矩阵返回复数，需要用 .real 取实部比较
#   坑：返回顺序不保证，别按位置取第一个；比较时用排序后的结果
eigen_values = np.sort(np.linalg.eigvals(np.array([[1.0, 1.0], [0.0, 2.0]])).real)

print("\n---- 数值计算自检 ----")
print(f"[1] quad(sin, 0, pi)      = {integral_value:.12f}   误差 {integral_error:.2e}")
print(f"[2] Rosenbrock 最优解      = {np.round(opt_result.x, 8)}  最优值 {opt_result.fun:.2e}")
print(f"[3] 特征值                = {np.round(eigen_values, 8)}")


# ==== 3. 单位与误差自检（pint / uncertainties）====

# 设计决策：计划书把 pint 与 uncertainties 列为核心 7 包之二，说明题目里会出现
#   "带物理量单位的实测数据"和"带不确定度的测量值"。这两类计算写错不会报错、
#   只会静默给出错答案（比如把 0.1 m 和 0.1 km 直接相加得到 0.2 "米"），
#   所以必须单独验一遍换算与误差传播。

import pint                                         # 单位换算与量纲检查
from uncertainties import ufloat                    # 带不确定度的数值（自动误差传播）

# 变量 ureg：pint.UnitRegistry，单位注册表（换算关系的唯一来源）
#   示例值 <pint.UnitRegistry object at 0x...>
#   为什么全局只建一个：构造一次约 0.2-0.3 秒，放在循环里会把程序拖垮；
#     自定义单位也必须注册到同一个 registry 才生效
ureg = pint.UnitRegistry()

# 变量 distance：pint.Quantity，带单位的量（数值 + 单位），可参与四则运算
#   示例值 1.0 mile
#   为什么写成 1.0 * ureg.mile：只有"数值 × 单位"才得到 Quantity；
#     ureg.mile 单独拿出来是 Unit 对象，不能直接参与算术
distance = 1.0 * ureg.mile

# Quantity.to(unit) -> Quantity，换成目标单位后的新量（原量不变）
#   作用：按注册表里的换算关系改单位，数值随之缩放
#   关键参数：unit 传单位对象（ureg.meter）最快；传字符串 "meter" 也能用，但要走一遍解析
#   坑：改自身的是 .ito()，想要返回值必须用 .to()；两者名字只差一个字母，很容易写错
meters = distance.to(ureg.meter)

# 变量 a_measured / b_measured：uncertainties.core.Variable，带标准差的名义值
#   示例值 2.0+/-0.1（名义值 2.0，标准差 0.1）
#   为什么不用 (值, 误差) 元组手算：多步运算要手工推偏导数，ufloat 用自动微分算，
#     连"同一变量出现两次"的相关项也能正确合并
a_measured = ufloat(2.0, 0.1)
b_measured = ufloat(3.0, 0.2)

# 变量 sum_measured：Variable，两独立量之和，标准差按平方和开方合成
#   示例值 5.00+/-0.22（sqrt(0.1² + 0.2²) = 0.2236）
#   为什么拿加法做示例：结果能徒手验算，比乘法更有说服力
sum_measured = a_measured + b_measured

print("\n---- 单位与误差自检 ----")
print(f"[4] 1 mile 换算           = {meters.magnitude:.3f} m")
print(f"[5] (2±0.1)+(3±0.2)       = {sum_measured.nominal_value:.4f} ± {sum_measured.std_dev:.4f}")


# ==== 4. 优化求解自检（cvxpy / pulp）====

# 设计决策：同一道最小线性规划题分别用 cvxpy 和 pulp 各解一遍。
#   两个求解器给出同一个 1.0，才能说明"求解器链"是通的，
#   只测 cvxpy 无法排除 CBC/pulp 那条链断掉。

# --- 3.1 cvxpy：min x+y s.t. x+y>=1, x>=0, y>=0 -> 最优值 1.0 ---
import cvxpy as cp                                   # 凸优化建模层，负责把模型转成求解器输入

# 变量 x_cvx / y_cvx：cp.Variable，优化决策变量（默认标量、无界）
#   示例值：求解前是符号对象，求解后 .value 为 float
#   为什么不用 numpy 数组代替：cvxpy 需要符号图来判定 DCP 规则并求导，ndarray 无法做到
x_cvx = cp.Variable()
y_cvx = cp.Variable()

# cp.Problem(objective, constraints).solve() -> 最优值 float（同时把 .value 写回各变量）
#   作用：按 DCP 规则编译模型并调用后端求解器（本环境自带 OSQP / Clarabel / SCS）
#   关键参数：objective 用 cp.Minimize/cp.Maximize 包住；constraints 传约束列表
#   坑：约束写成 x >= 1 - y 也会被接受，但涉及除法的模型必须显式满足 DCP，否则报 DCPError
cvx_problem = cp.Problem(cp.Minimize(x_cvx + y_cvx), [x_cvx + y_cvx >= 1, x_cvx >= 0, y_cvx >= 0])
cvx_value = cvx_problem.solve()

# --- 3.2 pulp：同一道题，验证 CBC 求解器链路 -> 最优值 1.0 ---
import pulp                                         # 线性/整数规划建模，自带 CBC 求解器

# 变量 lb_value：float，pulp 解出的最优目标值
#   示例值 1.0
#   为什么把建模与取值分成两步写：pulp 是"先建模、后 solve、再取值"的范式，
#     中途取值只会得到 None
lp_problem = pulp.LpProblem("env_check", pulp.LpMinimize)
x_lp = pulp.LpVariable("x_lp", lowBound=0)
y_lp = pulp.LpVariable("y_lp", lowBound=0)
lp_problem += x_lp + y_lp                           # 第一句 += 被解释为目标函数
lp_problem += x_lp + y_lp >= 1                      # 之后的 += 才是约束

# lp_problem.solve(solver) -> int 求解状态码（1 = Optimal）
#   作用：调用求解器求解已建模的问题
#   关键参数：用 pulp.PULP_CBC_CMD(msg=False) 指定自带求解器并关掉求解日志
#   坑：pulp.value(变量) 与 pulp.value(表达式) 是模块级函数，不要和同名变量重名
#       （写成 from pulp import * 后尤其容易撞名，故本脚本始终用 pulp.xxx 全名调用）
solve_status = lp_problem.solve(pulp.PULP_CBC_CMD(msg=False))
lp_value = pulp.value(lp_problem.objective)

print("\n---- 优化求解自检 ----")
print(f"[6] cvxpy 最优值 = {cvx_value:.6f}  （状态 {cvx_problem.status}）")
print(f"[7] pulp  最优值 = {lp_value:.6f}  （状态码 {solve_status}，1 表示 Optimal）")


# ==== 5. 绘图自检（matplotlib 中文字体 + 落盘 PNG）====

# 设计决策：中文显示是建模出图最常见的翻车点——字体缺失时 matplotlib 不报错，
#   只是把中文画成方框，很容易到交稿前才发现。所以自检必须"既查字体文件、又真出一张图"。

import matplotlib                                   # 绘图底座
matplotlib.use("Agg")                               # 切到无 GUI 后端，避免无显示器/远程时卡住
import matplotlib.pyplot as plt                     # 面向对象的绘图入口
from matplotlib import font_manager                 # 查询系统实际命中的字体文件

# 变量 CHINESE_FONT：str，首选中文字体名
#   示例值 "Microsoft YaHei"（Windows 自带）
#   为什么首选 YaHei 而不是 SimHei：YaHei 字重更全、观感更接近论文排版；
#     但部分精简版 Windows 只有 SimHei，故在 rcParams 里给了回退链
CHINESE_FONT = "Microsoft YaHei"

# 全局字体设置，作用于之后所有图
#   坑：font.sans-serif 是"候选链"，前面的找不到才往后退，故可多个字体一起写；
#       axes.unicode_minus 必须置 False，否则负号会因字体缺字符而显示成方框
matplotlib.rcParams["font.sans-serif"] = [CHINESE_FONT, "SimHei", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

# font_manager.findfont(prop, fallback_to_default=True) -> str，命中的字体文件绝对路径
#   作用：查"系统里到底有没有这个字体"，找不到时会静默退回 DejaVuSans.ttf
#   关键参数：prop 要传 FontProperties(family=...)，不能直接传字符串
#   坑：返回值不告诉你是否命中，必须自己检查路径里有没有 msyh/simhei 关键字
font_path = font_manager.findfont(font_manager.FontProperties(family=CHINESE_FONT))
font_ok = any(tag in Path(font_path).name.lower() for tag in ("msyh", "simhei", "msyhbd"))

# 变量 x_axis：ndarray[float]，横轴采样点
#   示例值 [-3.1416, ..., 3.1416]，共 300 个点
#   为什么是 300：曲线拐点处要平滑，少于 100 个点 sin 峰值会被画成折线角
x_axis = np.linspace(-np.pi, np.pi, 300)
y_axis = np.sin(x_axis)

# 设计决策：一张 PNG 只画一条曲线，方便"图里有没有方框"一眼看穿；
#   多面板拼图会让字体问题被稀释、看不清。
fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
ax.plot(x_axis, y_axis, color="#2b6cb0", linewidth=2, label="y = sin(x)")
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax.set_title("环境自检：中文标题（负半轴 -1 用来验证负号）")
ax.set_xlabel("横轴 x（弧度）")
ax.set_ylabel("纵轴 y")
ax.legend()
ax.grid(alpha=0.3)

# fig.savefig(path, dpi, bbox_inches) -> None
#   作用：把画布写盘为图片文件
#   关键参数：bbox_inches="tight" 自动裁掉多余白边，贴进论文时不用再裁
#   坑：savefig 之后必须 close，否则循环出图会累积未释放的画布导致内存暴涨
fig_check_path = OUT_DIR / "fig_check.png"
fig.savefig(fig_check_path, dpi=150, bbox_inches="tight")
plt.close(fig)

print("\n---- 绘图自检 ----")
print(f"[8] 中文字体命中文件 = {font_path}")
print(f"    字体判定 = {'OK（非方框）' if font_ok else '警告：退回默认字体，中文可能显示为方框'}")
print(f"    已输出 = {fig_check_path}")


# ==== 6. 汇总与报告落盘 ====

# 设计决策：把"通过 / 失败"集中成短清单，并写一份报告文件。
#   比赛现场需要的是 3 秒内看出哪项挂了，而不是翻一屏日志。

# 变量 CHECKS：list[tuple[str, bool, str]]，逐项 (名称, 是否通过, 实测值描述)
#   示例值 ("scipy 数值积分", True, "2.000000000000")
#   为什么把"实测值"也写进去：失败时不用重跑就能知道偏了多少
CHECKS = [
    ("必装库导入", len(imported) == len(CHECK_MODULES), f"{len(imported)}/{len(CHECK_MODULES)}"),
    ("scipy 数值积分", abs(integral_value - 2.0) < 1e-10, f"{integral_value:.12f}"),
    ("scipy 优化 (Rosenbrock)", float(opt_result.fun) < 1e-8, f"最优值 {opt_result.fun:.2e}"),
    ("numpy 特征值", np.allclose(eigen_values, [1.0, 2.0], atol=1e-8), f"{np.round(eigen_values, 6).tolist()}"),
    ("pint 单位换算", abs(meters.magnitude - 1609.344) < 1e-9, f"{meters.magnitude:.3f} m"),
    ("uncertainties 误差传播", abs(sum_measured.std_dev - 0.2236068) < 1e-6, f"±{sum_measured.std_dev:.4f}"),
    ("cvxpy 线性规划", abs(float(cvx_value) - 1.0) < 1e-6, f"{float(cvx_value):.6f}"),
    ("pulp 线性规划", abs(float(lp_value) - 1.0) < 1e-6, f"{float(lp_value):.6f}"),
    ("matplotlib 中文字体", font_ok, Path(font_path).name),
]

print("\n==== 自检汇总 ====")
for check_name, passed, detail in CHECKS:
    print(f"[{'PASS' if passed else 'FAIL'}] {check_name:<26} 实测 {detail}")

failed_count = sum(1 for _, passed, _ in CHECKS if not passed)

# 变量 report_path：Path，报告文件路径
#   示例值 WindowsPath('d:/UPC备赛/W1/_env_check/env_report.txt')
#   为什么用 utf-8 显式编码：Windows 默认 GBK 写中文没问题，但换到 Linux 打开会乱码
report_path = OUT_DIR / "env_report.txt"
with open(report_path, "w", encoding="utf-8") as report_file:
    report_file.write("UPC 备赛环境自检报告\n")
    report_file.write(f"解释器: {sys.version.split()[0]}\n")
    report_file.write(f"通过 {len(CHECKS) - failed_count} / {len(CHECKS)} 项\n\n")
    report_file.write("[库版本]\n")
    for module_name, version in imported.items():
        report_file.write(f"  {module_name:<14} {version}\n")
    report_file.write("\n[检查项]\n")
    for check_name, passed, detail in CHECKS:
        report_file.write(f"  {'PASS' if passed else 'FAIL'}  {check_name:<26} {detail}\n")

print(f"\n报告已写入 {report_path}")
print(f"结论：{'环境可用，可以开赛' if failed_count == 0 else f'有 {failed_count} 项失败，请先修复再开赛'}")

# sys.exit(code) -> 结束进程并以 code 作为退出码
#   作用：让调用方（批处理、CI、你自己 echo $LASTEXITCODE）能判断成败
#   坑：不要把 sys.exit 写在 with 块里，否则报告可能来不及 flush
sys.exit(0 if failed_count == 0 else 1)


# ---------------------------------------------------------------------------
# 怎么验证跑对了（在 W1 目录下执行）：
#   conda activate upc
#   python check_env.py
#   echo "退出码=$LASTEXITCODE"
#   （若不想激活环境，就直指解释器：& "D:\anaconda\envs\upc\python.exe" check_env.py）
#
# 预期输出关键行：
#   必装库导入成功：19 / 19
#   [1] quad(sin, 0, pi)      = 2.000000000000   误差 2.22e-14 以内
#   [2] Rosenbrock 最优解      = [1.00002202 1.00004222]  最优值 8.18e-10
#   [3] 特征值                = [1. 2.]
#   [4] 1 mile 换算           = 1609.344 m
#   [5] (2±0.1)+(3±0.2)       = 5.0000 ± 0.2236
#   [6] cvxpy 最优值 = 1.000000
#   [7] pulp  最优值 = 1.000000
#   [8] 字体判定 = OK（非方框）
#   [PASS] x 9（全 PASS），结论：环境可用，可以开赛
#   退出码=0
#
# 同时人工打开 _env_check/fig_check.png：标题应为完整中文、负号"−1"不应是方框。
# ---------------------------------------------------------------------------
