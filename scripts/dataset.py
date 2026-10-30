from pathlib import Path

import yaml

# 所有路径都从项目目录开始计算，项目放到其他地方也能使用。
PROJECT = Path(__file__).resolve().parent.parent


def dataset_config(name):
    """为当前电脑生成 YOLO 数据集配置文件。"""
    folder = PROJECT / "datasets" / name
    with (folder / f"{name}.yaml").open(encoding="utf-8") as file:
        data = yaml.safe_load(file)

    # YOLO 需要知道图片实际在哪里；运行时填入本机的绝对路径。
    data["path"] = str(folder)
    output = PROJECT / ".runtime" / f"{name}.yaml"
    output.parent.mkdir(exist_ok=True)
    with output.open("w", encoding="utf-8") as file:
        yaml.safe_dump(data, file, sort_keys=False, allow_unicode=True)
    return output
