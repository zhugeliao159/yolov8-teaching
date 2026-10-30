import cv2
from ultralytics import YOLO

from dataset import PROJECT

model = YOLO(str(PROJECT / "weights" / "charmander.pt"))
camera = cv2.VideoCapture(0)  # 默认摄像头；有多个摄像头时可尝试 1

if not camera.isOpened():
    raise RuntimeError("无法打开摄像头，请检查权限和是否被其他程序占用。")

print("摄像头已打开，按 Q 退出。")
while True:
    ok, frame = camera.read()
    if not ok:
        break
    # 当前模型分数较低，临时降低阈值，只显示得分最高的一个框。
    result = model.predict(frame, device="cpu", conf=0.01, max_det=1, verbose=False)[0]
    cv2.imshow("Charmander detection - press Q to quit", result.plot())
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()
