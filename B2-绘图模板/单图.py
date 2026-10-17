"""UPC 论文单图模板——一个变量随另一个变化（轨迹、衰减曲线、响应曲线）。

用途：给 C 直接套用的单图样板；只改 §1 配置段与 §2 数据段即可出图。
输入：X / Y 两个一维数组（在 §2 数据段替换）。
输出：figs/fig1_single.png（300 dpi，英文，轴标签带单位）。

跑完应看到的自检数字（本模板自带演示数据，可直接对照）：
  data points         : 400
  peak of |y|         : 1.000
  envelope at t=5.0 s : 0.223   (exp(-0.3*5) = 0.2231)
  saved               : figs/fig1_single.png  (>10 kB)

正文引用编号：Figure 1（改图时同步改这里和论文 caption）。
"""

# import 说明：
#   numpy      -> 造演示数据、求峰值；比赛时换成你的计算结果数组
#   matplotlib -> 出图。必须显式指定 Agg 后端，否则无显示环境（服务器/CI）会卡住
#   os         -> 建输出目录、查文件大小做落盘自检
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")           # 无界面后端：不弹窗，直接写文件
import matplotlib.pyplot as plt  # 必须在 use("Agg") 之后导入，顺序反了后端会被重置

# ==== 1. 全局配置（换题只改这一段）====
# 变量 FIG_PATH：str，输出路径
#   例："figs/fig1_single.png"
#   为什么放常量：论文里图的编号和文件名必须对得上，散落在代码里迟早错位
FIG_PATH = "figs/fig1_single.png"

# 变量 DPI：int，输出分辨率
#   例：300
#   为什么是 300：期刊/竞赛论文的印刷下限，低于 150 会被评审看出锯齿
DPI = 300

# 变量 FIGSIZE：tuple[float, float]，英寸
#   例：(6.0, 4.0)
#   为什么是 6x4：单栏图的常规尺寸，等比缩放到论文里不会糊
FIGSIZE = (6.0, 4.0)

# 变量 LABELS：dict，轴标签与图例文字
#   例：{"x": "Time $t$ (s)", "y": "Displacement $y$ (m)", "legend": "response"}
#   为什么统一放这里：轴标签必须带单位，评审第一眼看的就是这个
LABELS = {
    "x": "Time $t$ (s)",
    "y": "Displacement $y$ (m)",
    "legend": "damped response",
}

# ==== 2. 数据（比赛时替换成自己的数组）====
# 设计决策：模板自带一组可复现的演示数据，让你不接真实数据也能立刻验证
#   链路是通的（出图、存盘、自检）。接真实数据时只替换 x / y 两行即可。

# 变量 t：np.ndarray, shape (400,), float64，单位 s
#   例：np.linspace(0.0, 10.0, 400) -> 步长 0.025 s
#   为什么取 400 点：画曲线 200 点以上就看不出折线，400 是余量
t = np.linspace(0.0, 10.0, 400)

# 变量 y：np.ndarray, shape (400,), float64，单位 m
#   例：exp(-0.3t) * cos(2*pi*0.5*t)
#   为什么用阻尼振荡：它同时含"衰减"和"振荡"两种形态，适合当通用样板
y = np.exp(-0.3 * t) * np.cos(2.0 * np.pi * 0.5 * t)

# ==== 3. 建画布 ====
# 方法签名：plt.subplots(figsize=..., dpi=...) -> (Figure, Axes)
#   作用：一次性拿到 figure 与 axes，比 plt.figure() + gca() 少一层隐式状态
#   关键参数：dpi 此处只影响屏幕预览，真正落盘分辨率由 savefig(dpi=) 决定
#   坑：别在这里设 tight_layout，先画完再调，否则标注会被裁掉
fig, ax = plt.subplots(figsize=FIGSIZE)

# ==== 4. 画曲线 ====
# 方法签名：ax.plot(x, y, **kwargs) -> list[Line2D]
#   作用：画折线/曲线
#   关键参数：lw 线宽；label 供 legend 取用，漏了 label 图例会空
#   坑：label 只在第一次画该数据时设，重复调用会生成多条图例
ax.plot(t, y, lw=1.5, color="#1f4e79", label=LABELS["legend"])

# ==== 5. 装饰（轴标签 / 图例 / 网格）====
# 方法签名：ax.set_xlabel(s) -> Text
#   作用：设置横轴标签
#   关键参数：字符串里必须含单位，用 LaTeX 数学模式写变量名
#   坑：写成 "t(s)" 少个空格会显得挤，推荐 "Time $t$ (s)"
ax.set_xlabel(LABELS["x"])
ax.set_ylabel(LABELS["y"])

# 方法签名：ax.axhline(y=0, **kwargs) -> Line2D
#   作用：画一条水平参考线，标出零点，读图时有个基准
#   关键参数：ls 线型；alpha 透明度，别抢主曲线
#   坑：axhline 的 y 取数据坐标，但 xmin/xmax 取轴比例（0-1）；想让参考线只覆盖
#       数据区间会踩这个混用，跨坐标系的图别用 axhline
ax.axhline(0.0, color="gray", ls="--", lw=0.8, alpha=0.6)

# 方法签名：ax.legend(**kwargs) -> Legend
#   作用：显示图例
#   关键参数：loc 锚点；frameon 是否画边框
#   坑：图例必须有，评审靠它区分多条曲线；单曲线也别省
ax.legend(loc="upper right", frameon=True, fontsize=9)

# 方法签名：ax.grid(**kwargs) -> None
#   作用：加网格，方便读数
#   关键参数：alpha 透明度，0.3 左右最不抢戏
ax.grid(alpha=0.3)

# ==== 6. 保存与自检 ====
# 设计决策：先建目录再存盘，且存完立刻查文件大小——savefig 在路径错误时
#   可能静默失败（尤其 Windows 反斜杠转义），落盘自检是最便宜的兜底。

# 方法签名：os.makedirs(path, exist_ok=True) -> None
#   作用：递归建目录
#   关键参数：exist_ok=True 让目录已存在时不抛 FileExistsError
os.makedirs(os.path.dirname(FIG_PATH) or ".", exist_ok=True)

# 方法签名：fig.savefig(fname, dpi=..., bbox_inches=...) -> None
#   作用：把 figure 写盘
#   关键参数：dpi 落盘分辨率；bbox_inches="tight" 自动裁掉白边
#   坑：必须在 plt.close() 之前调用，关掉后再存会得到空白图
fig.savefig(FIG_PATH, dpi=DPI, bbox_inches="tight")
plt.close(fig)

if __name__ == "__main__":
    size_kb = os.path.getsize(FIG_PATH) / 1024.0
    print("data points         : %d" % t.size)
    print("peak of |y|         : %.3f" % np.abs(y).max())
    print("envelope at t=5.0 s : %.3f" % np.exp(-0.3 * 5.0))
    print("saved               : %s  (%.1f kB)" % (FIG_PATH, size_kb))
    # 自检断言：图没落盘或几乎是空图，直接报错，别等到写论文才发现
    assert size_kb > 10.0, "输出图过小，检查 FIG_PATH 与 savefig 调用"
    print("[cite] Figure 1 -> %s" % FIG_PATH)
