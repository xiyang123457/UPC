"""UPC 论文多子图模板——同一结论的多个侧面（不同参数下的对比）。

用途：把"同一物理量在不同参数下的形态"排成 2x2，供论文单栏使用。
输入：一组参数（§2 参数集段）+ 一个"参数 -> 曲线"的函数（§3）。
输出：figs/fig2_panels.png（300 dpi，英文，共用轴标签带单位）。

跑完应看到的自检数字（本模板自带演示数据，可直接对照）：
  panels              : 4  (2 x 2)
  points per panel    : 300
  zeta list           : [0.05, 0.1, 0.2, 0.4]
  peak of panel[0]    : 1.000
  saved               : figs/fig2_panels.png  (>10 kB)

正文引用编号：Figure 2。
"""

# import 说明：
#   numpy      -> 生成时间轴与曲线数据
#   matplotlib -> 出图；Agg 后端保证无显示环境也能落盘
#   os         -> 建目录、文件大小自检
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # 必须在 use("Agg") 之后导入，顺序反了后端会被重置

# ==== 1. 布局配置（换题只改这一段）====
# 变量 FIG_PATH：str，输出路径
#   例："figs/fig2_panels.png"
FIG_PATH = "figs/fig2_panels.png"

# 变量 NROWS / NCOLS：int，子图矩阵形状
#   例：2 / 2 -> 4 个子图
#   为什么用 2x2：论文单栏最多放得下 2x2，再多字就小到看不清
NROWS, NCOLS = 2, 2

# 变量 DPI / FIGSIZE：int / tuple
#   例：300 / (7.0, 5.5)
#   为什么多子图要更宽：2x2 里每个子图都要有可读的刻度标签
DPI = 300
FIGSIZE = (7.0, 5.5)

# 变量 SHARE_Y：bool，是否共用 y 轴
#   例：True
#   为什么共享：可以让"幅度差异"一眼看出；代价是内层子图不再画 y 刻度标签
SHARE_Y = True

# 变量 LABELS：dict，共用轴标签
#   例：{"x": "Time $t$ (s)", "y": "Displacement $y$ (m)"}
LABELS = {"x": "Time $t$ (s)", "y": "Displacement $y$ (m)"}

# ==== 2. 参数集（比赛时替换成你要对比的那一维）====
# 设计决策：把"要对比的变量"整理成一个 list，主循环按 list 迭代。
#   好处是改对比维度时只动这个 list，画图代码一行不用碰。

# 变量 ZETAS：list[float]，阻尼比，无量纲
#   例：[0.05, 0.1, 0.2, 0.4]
#   为什么这几个值：跨越欠阻尼到接近临界阻尼，形态差异明显，适合演示
ZETAS = [0.05, 0.1, 0.2, 0.4]

# 变量 T：np.ndarray, shape (300,), float64，单位 s
#   例：np.linspace(0, 20, 300)
#   为什么取 0-20 s：让最小阻尼比的那条也能看到 4-5 个周期
T = np.linspace(0.0, 20.0, 300)

# 变量 OMEGA0：float，固有角频率，rad/s
#   例：2*pi（即 1 Hz）
#   为什么取 1 Hz：周期 1 s，20 s 内点数够密，视觉上不会锯齿
OMEGA0 = 2.0 * np.pi

# ==== 3. 曲线函数 ====
# 作用：给定阻尼比，返回位移序列
# 参数：zeta 阻尼比（无量纲）；返回 np.ndarray，shape (300,)，单位 m
# 为什么单独封函数：主循环只负责排版，物理部分抽出来才好替换成真实模型
def response(zeta, t=T, omega0=OMEGA0):
    """欠阻尼自由振动：y = exp(-zeta*omega0*t) * cos(omega_d * t)。"""
    omega_d = omega0 * np.sqrt(max(0.0, 1.0 - zeta ** 2))   # 有阻尼角频率
    return np.exp(-zeta * omega0 * t) * np.cos(omega_d * t)

# ==== 4. 建画布 ====
# 方法签名：plt.subplots(nrows, ncols, figsize=..., sharey=...) -> (Figure, ndarray[Axes])
#   作用：一次建好整个子图矩阵，axes 是二维 ndarray，可直接索引
#   关键参数：sharey=True 让所有子图共用 y 刻度范围
#   坑：single subplot 时 axes 不是数组；这里固定 2x2，故一定能拿到 ndarray
fig, axes = plt.subplots(NROWS, NCOLS, figsize=FIGSIZE, sharey=SHARE_Y)

# ==== 5. 逐格画 ====
# 设计决策：用 axes.flat 做一维迭代，而不是写双层 for + axes[i][j]。
#   这样参数 list 与子图个数解耦——改了 NROWS/NCOLS 也不用改循环结构。

# 方法签名：np.ndarray.flat -> flatiter
#   作用：把二维 axes 数组摊平成一维迭代器
#   关键参数：无
#   坑：flat 返回的是迭代器不是 list，需要重复遍历时先 list(axes.flat)
for zeta, ax in zip(ZETAS, axes.flat):
    ax.plot(T, response(zeta), lw=1.4, color="#1f4e79")

    # 方法签名：ax.set_title(s, **kwargs) -> Text
    #   作用：给单个子图加标题，说明这一格对应哪个参数值
    #   关键参数：fontsize 默认会偏大，多子图里要压到 9-10
    #   坑：标题里必须写明参数名和数值，否则读者分不清四格差异在哪
    ax.set_title(r"$\zeta$ = %.2f" % zeta, fontsize=10)
    ax.grid(alpha=0.3)
    ax.axhline(0.0, color="gray", ls="--", lw=0.7, alpha=0.6)

# 设计决策：共用轴标签不挂在每个子图上（会重复四遍），改用 figure 级文本。
#   这是多子图最容易做错的地方——每个 ax 都 set_xlabel 会让图显得很脏。

# 方法签名：fig.supxlabel(s) / fig.supylabel(s) -> Text
#   作用：在整张图级别放一个居中轴标签
#   关键参数：fontsize 统一；y 位置由布局自动算
#   坑：supxlabel 与 tight_layout 的兼容性在旧版 matplotlib 上较差，
#       本模板已要求 matplotlib>=3.8；低于此版本请改用 fig.text(0.5, 0.02, ...)
fig.supxlabel(LABELS["x"])
fig.supylabel(LABELS["y"])

# 方法签名：fig.tight_layout() -> None
#   作用：自动调整子图间距，避免标题与上一行刻度重叠
#   关键参数：rect 可手动留白，一般不用
#   坑：必须在 supxlabel/supylabel 之后调用，顺序反了会被再次挤压
fig.tight_layout()

# ==== 6. 保存与自检 ====
os.makedirs(os.path.dirname(FIG_PATH) or ".", exist_ok=True)

# 方法签名：fig.savefig(fname, dpi=..., bbox_inches=...) -> None
#   作用：写盘
#   关键参数：bbox_inches="tight" 会把 supxlabel 一并算进裁剪范围
#   坑：多子图用 bbox_inches="tight" 偶尔会裁掉 supxlabel，
#       如果发现横轴标签被切，改成 plt.savefig(..., bbox_inches=None)
fig.savefig(FIG_PATH, dpi=DPI, bbox_inches="tight")
plt.close(fig)

if __name__ == "__main__":
    size_kb = os.path.getsize(FIG_PATH) / 1024.0
    print("panels              : %d  (%d x %d)" % (NROWS * NCOLS, NROWS, NCOLS))
    print("points per panel    : %d" % T.size)
    print("zeta list           : %s" % ZETAS)
    print("peak of panel[0]    : %.3f" % np.abs(response(ZETAS[0])).max())
    print("saved               : %s  (%.1f kB)" % (FIG_PATH, size_kb))
    assert len(ZETAS) <= NROWS * NCOLS, "参数比子图多，多余的会被静默丢弃"
    assert size_kb > 10.0, "输出图过小，检查 FIG_PATH 与 savefig 调用"
    print("[cite] Figure 2 -> %s" % FIG_PATH)
