"""UPC 论文双对数模板——跨越多个数量级的幂律 / 标度律关系。

用途：判断两个量是否为幂律关系，并直接读出幂指数（斜率）。
输入：X / Y 一维数组（在 §2 数据段替换）。
输出：figs/fig3_loglog.png（300 dpi，英文，图内标注拟合斜率）。

跑完应看到的自检数字（本模板自带演示数据，可直接对照）：
  data points         : 200
  fitted slope        : 2.000  (理论值 2，允许 ±0.01)
  fitted intercept    : log10(1/g) = -0.9917  (g = 9.81 -> -log10(9.81))
  x range             : 1e+00 .. 1e+03  (3 个数量级)
  saved               : figs/fig3_loglog.png  (>10 kB)

正文引用编号：Figure 3。
"""

# import 说明：
#   numpy      -> 造演示数据、做 log10 线性拟合、算残差
#   matplotlib -> 出图；Agg 后端保证无界面环境也能落盘
#   os         -> 建目录、文件大小自检
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # 必须在 use("Agg") 之后导入，顺序反了后端会被重置

# ==== 1. 配置（换题只改这一段）====
# 变量 FIG_PATH：str，输出路径
FIG_PATH = "figs/fig3_loglog.png"

# 变量 DPI / FIGSIZE
#   例：300 / (6.0, 4.5)
#   为什么比单图略高：双对数图右上角常放拟合公式，需要留白
DPI = 300
FIGSIZE = (6.0, 4.5)

# 变量 LABELS：dict，轴标签
#   例：{"x": "Initial speed $v_0$ (m/s)", "y": "Range $R$ (m)"}
#   坑：双对数图的轴标签不需要写"log"，log 由刻度体现，写了反而冗余
LABELS = {"x": "Initial speed $v_0$ (m/s)", "y": "Range $R$ (m)"}

# ==== 2. 数据（比赛时替换成自己的数组）====
# 设计决策：演示数据取真空抛物线射程 R = v0^2 * sin(2θ) / g 的 θ=45° 情形，
#   即 R ∝ v0^2，斜率理论值恰好为 2——这样自检有确定的比对基准。

# 变量 V：np.ndarray, shape (200,), float64，单位 m/s
#   例：np.logspace(0, 3, 200) -> 1 .. 1000 m/s，跨 3 个数量级
#   为什么用 logspace 而不是 linspace：双对数图上前者才让点在视觉上均匀分布
V = np.logspace(0.0, 3.0, 200)

# 变量 G：float，重力加速度，m/s^2
#   例：9.81
#   为什么写常量：下面解析式与自检都要用，避免两次手抄出错
G = 9.81

# 变量 THETA_DEG：float，发射角，deg
#   例：45.0
#   为什么取 45：此时 sin(2θ)=1，射程最大，也让公式最简
THETA_DEG = 45.0

# 变量 R：np.ndarray, shape (200,), float64，单位 m
#   例：V^2 * sin(2*45°) / 9.81
R = V ** 2 * np.sin(2.0 * np.deg2rad(THETA_DEG)) / G

# ==== 3. 对数拟合 ====
# 设计决策：不要用 np.polyfit(V, R, 2) 再去猜次数。双对数图的正确做法是
#   先取 log10 再线性拟合——斜率就是你想要的幂指数，且残差在同权下更合理。

# 方法签名：np.log10(x) -> np.ndarray
#   作用：取以 10 为底的对数
#   关键参数：x 必须全为正，出现 0 或负数会得到 -inf/nan 而不报错
#   坑：这是双对数最常见的静默错误——数据里混一个 0，拟合结果整体失效
log_v = np.log10(V)
log_r = np.log10(R)

# 方法签名：np.polyfit(x, y, deg) -> np.ndarray, shape (deg+1,)
#   作用：最小二乘多项式拟合，deg=1 即线性回归
#   关键参数：返回系数按"高次在前"，斜率是 slope[0]，截距是 slope[1]
#   坑：polyfit 不做异常值剔除；若某点明显偏离，先剔除再拟合，不要让它带偏整条线
coeff = np.polyfit(log_v, log_r, 1)
slope = coeff[0]
intercept = coeff[1]

# ==== 4. 建画布 ====
fig, ax = plt.subplots(figsize=FIGSIZE)

# 方法签名：ax.loglog(x, y, **kwargs) -> list[Line2D]
#   作用：双对数坐标画图（两轴都取 log）
#   关键参数：marker 用 'o' 小点，配合 ms 控制大小，比纯线更容易看数据密度
#   坑：loglog 对非正值会告警并丢弃该点，画出来是空白但不会报错
ax.loglog(V, R, "o", ms=2.5, color="#1f4e79", label="simulation")

# 设计决策：拟合线用"网络"式取点自己画，而不是 plot(log_v, log_r)。
#   原因：loglog 坐标系里直接给 log 后的值会再取一次 log，得到完全错误的图。

# 变量 v_fit：np.ndarray, shape (200,)，与 V 同域
#   为什么用 V 本身：拟合线要和原始数据 x 对齐，否则图例位置会飘
v_fit = V.copy()
r_fit = 10.0 ** (slope * np.log10(v_fit) + intercept)

# 方法签名：ax.plot(x, y, **kwargs) -> list[Line2D]
#   作用：在已有双对数坐标系上叠加拟合线
#   关键参数：ls 线型；lw 线宽要比散点略粗
#   坑：此处 x 必须是原始量（v_fit），不是 log10 后的量
ax.plot(v_fit, r_fit, "-", lw=1.6, color="#c00000", label="power-law fit")

# ==== 5. 标注斜率 ====
# 方法签名：ax.text(x, y, s, **kwargs) -> Text
#   作用：在数据坐标处写文字，用来标斜率或公式
#   关键参数：transform=ax.transAxes 时 xy 改用轴比例（0-1），图例位置更好控
#   坑：用数据坐标标在左上角，换个数据范围文字就跑到坐标轴外，双对数图尤其明显
ax.text(
    0.04, 0.96,
    r"$R \propto v_0^{%.3f}$" % slope + "\n" + r"slope = %.3f" % slope,
    transform=ax.transAxes,
    va="top", ha="left", fontsize=10,
    bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9),
)

ax.set_xlabel(LABELS["x"])
ax.set_ylabel(LABELS["y"])

# 方法签名：ax.legend(**kwargs) -> Legend
#   作用：区分"数据点"和"拟合线"
#   关键参数：loc="lower right" 避开左上角的斜率标注
ax.legend(loc="lower right", fontsize=9)

# 方法签名：ax.grid(**kwargs) -> None
#   作用：双对数图必须开网格，否则读不出数量级
#   关键参数：which="both" 同时画主刻度与次刻度网格
#   坑：只开主网格时，次刻度区间里读不出位置，跨数量级的图会难读
ax.grid(which="both", alpha=0.25)

# ==== 6. 保存与自检 ====
os.makedirs(os.path.dirname(FIG_PATH) or ".", exist_ok=True)
fig.savefig(FIG_PATH, dpi=DPI, bbox_inches="tight")
plt.close(fig)

if __name__ == "__main__":
    size_kb = os.path.getsize(FIG_PATH) / 1024.0
    print("data points         : %d" % V.size)
    print("fitted slope        : %.3f" % slope)
    print("fitted intercept    : %.4f" % intercept)
    print("x range             : %.0e .. %.0e" % (V.min(), V.max()))
    print("saved               : %s  (%.1f kB)" % (FIG_PATH, size_kb))
    # 自检断言：斜率偏离理论值 2 超过 0.01 就说明数据或拟合有问题
    assert abs(slope - 2.0) < 0.01, "斜率不是 2，检查数据是否为正、是否混入 0"
    assert size_kb > 10.0, "输出图过小，检查 FIG_PATH 与 savefig 调用"
    print("[cite] Figure 3 -> %s" % FIG_PATH)
