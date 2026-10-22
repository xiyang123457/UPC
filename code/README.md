# code/ — 比赛代码

命名约定（48 小时内靠这个找东西，别用 `final_v2_新.py`）：

| 文件 | 内容 |
|---|---|
| `model.py` | 物理模型：ODE 右端、参数、量纲 |
| `solve.py` | 求解：积分、求根、扫描、灵敏度 |
| `plots.py` | 出图：所有 matplotlib 调用集中在这里 |
| `run_all.py` | 一键复现：`python run_all.py` 产生 figs/ 与终端表格 |

规则：
- 任何写进 `results.md` 的数字，必须能由 `run_all.py` 一次跑出来。
- 脚本内不硬编码数字，参数走顶层 dict。
- 中文注释、英文图，与 `B2-绘图模板/` 保持同一风格。

## 模板文件（不是比赛代码，可随时复制改写）

上表的四个名字是**比赛正式代码**的命名；下面两个是**可复用模板**，不属于上表命名约定。

| 文件 | 内容 |
|---|---|
| `skeleton_template.py` | 七段结构最小可运行骨架（阻尼自由落体示例）。改第 2 段 `PARAMS` 与第 4 段 `rhs` 即可换题，说明见根目录 `W2-B1-代码组织模板.md` |
| `demo_swap_rhs.py` | 把骨架换成阻尼弹簧振子的对照版，用来验证「换题时段 5 逐字不动」 |

两者都从 `code/` 内部向上定位仓库根的 `figs/` 输出（`ROOT = dirname(dirname(__file__))`），从任意 CWD 调用都能正确出图：

```powershell
python code/skeleton_template.py
# 预期：[fig] <仓库根>\figs\skeleton.png
```
