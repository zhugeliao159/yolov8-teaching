# YOLOv8 工程教学项目

用 COCO128 和小火龙图片，体验数据标注、模型训练、图片推理和摄像头推理。项目可以放在 Windows、Linux 或 WSL 的任意目录；下面的命令都在**项目根目录**执行。

## 文件说明

```text
datasets/      数据集、YOLO 标注和配置
weights/       YOLOv8n、YOLOv8s 和训练好的小火龙模型
scripts/       训练、图片推理、摄像头推理
notebooks/     数据集教学 Notebook
lesson_yolov8.html  可离线打开的教学课件
runs/          运行后自动生成的训练与推理结果
```

## 1. 准备环境

推荐 Python 3.11 或 3.12。在项目目录创建独立的虚拟环境：

```bash
python -m venv .venv
```

Windows PowerShell 激活：

```powershell
.\.venv\Scripts\Activate.ps1
```

如果 PowerShell 的策略阻止激活，可以改用命令提示符执行 `.venv\Scripts\activate.bat`，或者直接使用 `.venv\Scripts\python.exe` 替代下方的 `python`。

Linux / WSL 激活：

```bash
source .venv/bin/activate
```

安装 PyTorch：在没有 NVIDIA GPU 的电脑上，可以安装 CPU 版本：

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
```

如果希望用支持 CUDA 的 NVIDIA GPU 训练，可先根据电脑和驱动情况选择对应的 [PyTorch 官方安装方式](https://pytorch.org/get-started/locally/)。例如，适用 CUDA 12.4 版本的配置可以使用：

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

然后安装其他依赖：

```bash
python -m pip install -r requirements.txt
python -c "import torch; print('GPU 可用:', torch.cuda.is_available())"
```

`False` 表示当前环境没有可用的 CUDA GPU，图片和摄像头推理仍可使用 CPU。训练脚本也会自动使用 CPU，只是速度可能较慢。GPU 版本要求电脑有兼容的 NVIDIA 显卡与驱动；本项目不安装或修改驱动。

## 2. 运行图片推理

```bash
python scripts/predict.py
```

脚本默认使用 `weights/charmander.pt` 和小火龙验证集中的一张照片。运行后会显示结果图片的保存位置；修改脚本开头的 `model_path` 和 `image_path` 可以换模型或图片。

**这份课堂模型的置信度偏低**，示例暂时采用 `conf=0.01` 且最多显示一个框。它可能把背景误认成小火龙，不能把示例结果当作可靠的实际应用。

## 3. 运行摄像头推理

```bash
python scripts/camera.py
```

摄像头窗口打开后，按 **Q** 退出。若使用多个摄像头，可把 `camera.py` 中的 `VideoCapture(0)` 改为 `VideoCapture(1)`。通常建议在 Windows 本机 Python 运行摄像头；WSL 是否能访问摄像头取决于设备和系统配置。

## 4. 训练

```bash
python scripts/train_gpu.py
```

脚本开头可修改 `model_name`（`yolov8n.pt` 或 `yolov8s.pt`）、`dataset_name`（`charmander` 或 `coco128`）和 `epochs`。默认只运行两轮，适合演示流程，不代表模型已经训练充分。结束后按照屏幕上的路径查找 `best.pt`。

程序会根据本机项目位置，在 `.runtime/` 下生成带绝对路径的临时数据配置；不用手工修改用户名或指定 `/home/fyz`。

## 5. 数据集 Notebook

```bash
python -m ipykernel install --user --name yolov8-teaching --display-name "Python (YOLOv8 Teaching)"
python -m jupyter lab
```

打开 `notebooks/YOLOv8_Teaching.ipynb`，选择 `Python (YOLOv8 Teaching)` 内核。Notebook 会展示至少 5 张带原始标注框的 COCO128 图片，并将图片保存到 `.runtime/`。

COCO128 中训练和验证使用同一批图片；小火龙只有 14 张训练图片和 3 张验证图片。这些指标适合教学演示，不能可靠代表新场景的表现。COCO128 的预训练模型也可能见过相关数据。

## Git 内容

`.gitignore` 排除了本地虚拟环境、临时配置、标签缓存和训练输出。项目内的 `yolov8n.pt`、`yolov8s.pt` 与 `charmander.pt` 用于课堂启动；更大的官方型号可按需从 Ultralytics 官方来源下载。
