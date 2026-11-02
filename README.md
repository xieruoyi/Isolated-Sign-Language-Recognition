# ASL 手语识别入门项目

输入 MediaPipe 关键点序列，预测 250 类孤立 ASL 手语。项目包含数据检查、双手特征预处理、按参与者划分验证集、小型时间卷积模型和单样本预测。

## 当前工作方式：VS Code + Kaggle 远程内核

请打开 `notebooks/01_kaggle_explore.ipynb`，选择已连接的 Kaggle Jupyter Server 内核，按顺序运行。Notebook 已独立化，会自动查找 `/kaggle/input` 下的 ASL 比赛数据，不依赖本地项目文件或本地安装。生成的图片保存在 `/kaggle/working/asl-project`。

### 从 GitHub 使用

克隆本仓库后，用 VS Code 打开文件夹，再将 notebook 连接到自己的 Kaggle Jupyter Server。在 Kaggle notebook 中添加 `asl-signs` 比赛数据。仓库不包含比赛数据、虚拟环境或服务器连接 URL；每位使用者需要使用自己的 Kaggle 会话。

本地 `.venv` 是之前误配置时创建的环境，保留但不用于这条远程流程。以下本地安装和终端训练命令仅供选择本地运行时参考。

## 本地运行（可选）

环境已配置：`.venv` 使用 Python 3.12.1，依赖已安装，TensorFlow 2.19.1 导入和计算检查通过。打开 notebook 后，在右上角选择 **ASL Project (Python 3.12)** 或 `.venv` 的解释器。已安装的版本保存在 `requirements-lock.txt`。

用 VS Code 打开本文件夹。建议安装 Python 和 Jupyter 扩展。

本机已找到 Python 3.12.1，并使用它创建项目的 `.venv`。此项目使用 **Python 3.11 或 3.12**。如果需要重新创建环境，可在 VS Code 的 PowerShell 终端运行：

```powershell
& "$env:LOCALAPPDATA/Programs/Python/Python312/python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

第一行使用本机已有的 Python 3.12 路径。
然后按 Ctrl+Shift+P → Python: Select Interpreter → 选择 `.venv`。下方命令直接指定解释器，不需要更改 PowerShell 执行策略。

## 放入数据

从 Kaggle 的 asl-signs 比赛页面获取数据（需自行登录并接受数据使用规则）。解压后目录为：

```text
data/asl-signs/
  train.csv
  sign_to_prediction_index_map.json
  train_landmark_files/
    ...
```

如果数据在其他地方，修改 `config.json` 的 `data_dir`，Windows 路径建议写成 `D:/datasets/asl-signs`。无需复制已有数据。项目不会读取或保存 Kaggle 密钥。

## 按顺序运行

```powershell
# 检查路径、标签、关键点形状和缺失率
.\.venv\Scripts\python.exe asl.py inspect

# 先用 1000 个样本验证整个流程
.\.venv\Scripts\python.exe asl.py prepare --limit 1000
.\.venv\Scripts\python.exe asl.py train

# 跑通后，用全部数据重新准备并训练
.\.venv\Scripts\python.exe asl.py prepare
.\.venv\Scripts\python.exe asl.py train

# 替换为实际 parquet 文件路径
.\.venv\Scripts\python.exe asl.py predict "data/asl-signs/train_landmark_files/参与者/样本.parquet"
```

`artifacts/` 保存特征缓存、最佳模型、训练曲线数值和验证指标。重复运行会覆盖同名产物；保留重要实验前请备份该目录。
全量特征缓存及训练数组需要数 GB 内存；内存不足先使用 `--limit 1000`。

## Kaggle 运行

VS Code 连接 Kaggle 不代表本地终端自动获得远程 GPU。在 Kaggle Notebook 中使用这些文件时，先把配置的 `data_dir` 设为实际数据目录（通常是 `/kaggle/input/asl-signs`），`output_dir` 设为 `/kaggle/working/artifacts`，再使用 `!python asl.py inspect` 等命令。Notebook 的工作目录需包含项目文件。Kaggle 自带环境如果能导入所需包，可先直接使用，避免无必要地重装 TensorFlow。

## 评估和项目边界

- 按 `participant_id` 划分训练/验证，避免同一人的样本同时进入两边。
- 标签使用官方 JSON 编号，缺失坐标有单独有效性特征。
- 1000 样本只用于检查流程，不代表完整 250 类任务表现。
- 当前是训练基线，还没有生成比赛 `submission.zip`。提交需要将预处理一起实现为 TensorFlow 运算，再导出原始 `(帧数, 543, 3)` 输入的 TFLite 签名，并核验官方运行时、体积和延迟。不能把只接收处理后特征的模型直接当作比赛模型提交。
- 当前不包含摄像头实时识别或 ASL 句子翻译。

数据来源：https://www.kaggle.com/competitions/asl-signs 。使用时遵守原数据 CC-BY 授权并保留归属说明。
