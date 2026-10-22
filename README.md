# UPC 2026 备赛 · W1 工作区

UPC 2026 备赛第一周（W1）的工作仓库：一套能**算数、解优化、出论文图**的 Python 环境，加三份团队协作规范。

> 队内按 A / B / C 分工，本仓库以 B 侧交付物为主（环境、代码模板、同步机制）。

## 快速开始

### 1. 建环境（首次，约 2 分钟）

```powershell
conda create -n upc python=3.11 -y
conda activate upc
pip install -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

选 Python 3.11 而不是最新版，是因为科学计算生态（`numba`、各求解器 wheel）对 3.11 的支持最完整。

### 2. 验收环境（必做）

```powershell
python check_env.py
```

预期：`19 / 19` 库导入成功 + `9/9` 项 PASS + 退出码 0。检查项覆盖数值积分、Rosenbrock 优化、特征值、`pint` 单位换算、`uncertainties` 误差传播、`cvxpy`/`pulp` 线性规划、matplotlib 中文字体。

产物写入 `_env_check/`（已被 git 忽略，可随时重跑）。

### 3. 复现 W1 基线数字

```powershell
python test.py
```

预期输出：

```
feasible angles for 1200 m: ['2.72 deg', '77.06 deg']
check: range=1200.0 m  flight=3.77 s  maxalt=17.6 m
  Cd x0.8 -> range 1292.1 m
  ...
vacuum check: numeric=1955.7 m analytic=1955.7 m diff=0.000%
```

每个数字在 `results.md` 里都有对应的一行出处。

### 4. 打开 Jupyter 演示

```powershell
jupyter lab
```

左侧文件树双击 `jupyter_demo.ipynb`，再走 `Kernel → Restart Kernel and Run All Cells`。演示分步执行、`sympy` 公式渲染、内嵌图、`ipywidgets` 交互滑块、`%timeit` 计时。

> Jupyter 服务依赖终端窗口，关掉终端即断连。交互单元静态预览时可能空白，重新执行该单元即可。

## 目录结构

| 路径 | 内容 |
|---|---|
| `test.py` | 抛体动力学起手模板（2025-B Artillery 示例）：ODE + 事件检测 + 多解求根 + 灵敏度 + 量纲自检 |
| `check_env.py` | 环境自检：19 个库导入 + 9 项数值/绘图验证，退出码 0 才算可用 |
| `requirements.txt` | 依赖清单，按「计划书核心 7 包 / 补充 A·B·C / 额外武器库 / 未装」分档 |
| `jupyter_demo.ipynb` | JupyterLab 上手演示（16 单元，已实跑通过） |
| `results.md` | **数值结果唯一出处**，论文里每个数字都要能在这里找到 |
| `B1-环境自检.png` / `.txt` | 环境自检的截图与终端输出留档 |
| `B1-环境快照-lock.txt` | `pip freeze` 全量锁定，用于精确复现环境 |
| `B2-绘图模板/` | 论文用三套出图模板：`单图.py` / `多子图.py` / `双对数.py`（300 dpi、英文图注、带单位） |
| `B3-同步会安排.md` | 队内同步会规范（频率、三问、会后记录模板） |
| `code/` | 比赛代码目录，命名约定见其 README；另含七段结构骨架 `skeleton_template.py` 与换题对照版 `demo_swap_rhs.py` |
| `refs/gold_papers/` | 往届金奖论文存放处，用法见其 README（目前为空） |
| `W2-B1-代码组织模板.md` | **W2 任务 1**：七段结构 + 三条命名规矩 + 自检放置规则 + 完整骨架，配套 `code/skeleton_template.py` |

## 三条硬规则

**1. 数字只有一个出处：`results.md`**
论文正文、图注、摘要里出现的每一个数字，都必须能在 `results.md` 找到一行，且写明怎么算出来的。不许出现本文件之外的新数字；改模型后先改 `results.md`，再改论文，顺序不能反。

**2. 代码命名固定**
`code/` 下只用 `model.py` / `solve.py` / `plots.py` / `run_all.py`，禁止 `final_v2_新.py`。任何写进 `results.md` 的数字，必须能由 `python run_all.py` 一次跑出来。

**3. 同步会不砍**
每周日 21:00–21:15，硬上限 15 分钟。固定三问：谁卡住了 / 现在手上有什么**可验证的实物** / 下个交付物谁负责、什么时候交。W3 模拟周改为每 6 小时一次。详见 `B3-同步会安排.md`。

## W1 完成情况

| 项 | 状态 | 证据 |
|---|---|---|
| conda 环境与依赖清单 | 完成 | `requirements.txt`，19 包全部可导入 |
| 环境自检脚本 | 完成 | `check_env.py`，9 项 PASS、退出码 0 |
| 抛体动力学起手模板 | 完成 | `test.py`，基线数字见 `results.md` §1 |
| 论文绘图三套模板 | 完成 | `B2-绘图模板/`，自带演示数据与自检数字 |
| 同步会机制 | 完成 | `B3-同步会安排.md`（待 A、C 确认日历） |
| Jupyter 上手演示 | 完成 | `jupyter_demo.ipynb`，16 单元全跑通 |

## 已知限制

1. **`find_feasible_angles` 在极限射程处漏解**（`test.py`）
   目标射程落入 `(3197.2183, 3197.2473)` m 这个约 **0.029 m** 宽的窗口时，180 点粗扫检测不到符号变化，返回空列表，而真实存在 2 个解。修法：峰值附近二次加密网格，或拆成两段单调区间各求一次根。**W1 未修，W2 内解决**，详见 `results.md` §3。
2. `code/` 与 `refs/gold_papers/` 目前只有 README 占位，实体内容待补。
3. `B2-绘图模板/` 产出的是**示例数据**，只用于验证模板可用；正式论文的图必须用真实模型数据重画，并替换 `results.md` 图清单里对应的行。

## 关于 git 忽略

`figs/`、`_env_check/`、`.ai-memory/`、`__pycache__/` 不入库 —— 它们都是脚本一键重跑的产物，clone 后看不到属正常。`B1-环境自检.png` 是例外，作为验收证据留在库内。

## 常用命令速查

| 目的 | 命令 |
|---|---|
| 激活环境 | `conda activate upc` |
| 验收环境 | `python check_env.py` |
| 复现基线 | `python test.py` |
| 打开笔记本 | `jupyter lab` |
| 导出笔记本为 HTML | `jupyter nbconvert --to html jupyter_demo.ipynb` |
| 重装依赖 | `pip install -r requirements.txt` |
