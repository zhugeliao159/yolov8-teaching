from ultralytics import YOLO
from ultralytics.utils import LOGGER
import torch

from dataset import PROJECT, dataset_config

# 修改这里可以选择模型、数据集和训练轮数
model_name = "yolov8n.pt"      # 也可以改为 yolov8s.pt
dataset_name = "charmander"    # 也可以改为 coco128
epochs = 2                     # 课堂演示先跑 2 轮

# 有可用的 NVIDIA GPU 就用第 0 张，否则用 CPU
device = 0 if torch.cuda.is_available() else "cpu"
LOGGER.setLevel("ERROR")

print(f"训练：{model_name} | 数据集：{dataset_name} | 轮数：{epochs} | 设备：{device}")
model = YOLO(str(PROJECT / "weights" / model_name))


def show_epoch(trainer):
    print(f"第 {trainer.epoch + 1}/{trainer.epochs} 轮完成，训练损失：{trainer.tloss.sum().item():.3f}", flush=True)


model.add_callback("on_train_epoch_end", show_epoch)
results = model.train(
    data=str(dataset_config(dataset_name)),
    epochs=epochs,
    imgsz=640,
    batch=4,
    device=device,
    workers=0,
    amp=False,
    verbose=False,
    plots=False,
    project=str(PROJECT / "runs" / "detect"),
    name="classroom_train",
)
print(f"检测得分（mAP@0.5）：{results.box.map50:.1%}")
print(f"训练好的模型：{model.trainer.best}")
print("提示：小数据集的分数仅供课堂实验，不代表真实场景表现。")
