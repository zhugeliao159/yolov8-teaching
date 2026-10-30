from ultralytics import YOLO

from dataset import PROJECT

# 换成自己的模型或图片时，修改下面两行
model_path = PROJECT / "weights" / "charmander.pt"
image_path = PROJECT / "datasets" / "charmander" / "images" / "val" / "dragon0012.jpg"

model = YOLO(str(model_path))

# 这份小火龙模型的置信度较低，暂时调低阈值并只显示一个框。
results = model.predict(
    source=str(image_path),
    device="cpu",
    conf=0.01,
    max_det=1,
    save=True,
    project=str(PROJECT / "runs" / "detect"),
    name="classroom_predict",
    verbose=False,
)
print(f"检测到 {len(results[0].boxes)} 个目标")
print(f"结果图片保存在：{results[0].save_dir}")
