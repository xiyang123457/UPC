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
