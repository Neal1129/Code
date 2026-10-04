# 生科挑战赛代码包

本代码包按“数据—源码—模型—结果”结构整理，包含三个相互独立的功能模块：

- `src/tracker/`：细菌视频追踪 Web 应用。
- `src/abm/`：光响应微纳机器人黏膜给药 ABM 仿真与桌面界面。
- `src/esm2/`：CheZ-LOV 连接肽的 ESM-2 masked-marginal 打分。

## 环境与运行

建议使用 Python 3.11，先安装基础依赖：

```bash
pip install -r requirements.txt
```

ESM-2 打分需要额外安装 `torch` 和 `fair-esm`，首次运行会自动下载
`esm2_t33_650M_UR50D` 权重。

```bash
python design.py             # 启动 ABM 桌面设计/仿真界面
python train.py              # 运行 ABM 仿真，结果保存到 results/abm/
python predict.py            # 计算 ESM-2 打分，结果保存到 results/esm2/
python src/tracker/app.py    # 启动追踪 Web 应用：http://127.0.0.1:5005
```

如需在多次启动后保持追踪网页的登录会话，请先设置环境变量 `SECRET_KEY`。

## 目录说明

```text
Code/
├── data/          输入数据及来源说明
├── models/        模型权重下载说明
├── notebooks/     可复核 Notebook 预留目录
├── src/           核心源代码
├── results/       程序生成结果
├── logs/          运行日志预留目录
├── train.py       ABM 一键仿真入口
├── design.py      ABM 桌面设计界面入口
└── predict.py     ESM-2 一键预测入口
```

## 数据与模型说明

`data/esm2/results-table.xlsx` 是 ESM-2 打分的输入筛选结果表；原始文件随本项目提供。
本代码包未包含受限的原始实验数据。模型权重不随代码包分发，使用 ESM 官方预训练模型按需下载，详见 `models/README.md`。
