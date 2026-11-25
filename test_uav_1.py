import cv2
import numpy as np
from huggingface_hub import hf_hub_download, list_repo_files
from ultralytics import YOLO

# 1. 載入 VisDrone 權重（專攻無人機斜拍/俯拍的車與人）
repo_id = "dronefreak/visdrone-yolo11m"
pt_files = [f for f in list_repo_files(repo_id) if f.endswith(".pt")]
visdrone_path = hf_hub_download(repo_id=repo_id, filename=pt_files[0])
model = YOLO(visdrone_path)

# 讀取測試圖片
image_path = "test.jpg"
img = cv2.imread(image_path)
annotated_img = img.copy()

# 統計字典
counts = {"行人": 0, "機車/自行車": 0, "汽車/大型車": 0}

# 2. 一次偵測所有目標！
# VisDrone 類別: 0:pedestrian, 1:people, 2:bicycle, 3:car, 4:van, 5:truck, 6:tricycle, 7:awning-tricycle, 8:bus, 9:motor
target_classes = [0, 1, 2, 3, 4, 5, 8, 9]
results = model.predict(source=img, classes=target_classes, conf=0.25, imgsz=1280, verbose=False)[0]

if results.boxes is not None:
    boxes = results.boxes.xyxy.cpu().numpy()
    clss = results.boxes.cls.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()

    for box, cls_id, conf in zip(boxes, clss, confs):
        x1, y1, x2, y2 = map(int, box)
        cls_id = int(cls_id)
        name = model.names[cls_id]

        # 依類別分色標示
        if cls_id in [0, 1]:
            counts["行人"] += 1
            color = (255, 100, 0)    # 藍色：行人
        elif cls_id in [2, 9]:
            counts["機車/自行車"] += 1
            color = (0, 165, 255)    # 橘色：機車/自行車
        else:
            counts["汽車/大型車"] += 1
            color = (0, 255, 0)      # 綠色：汽車/貨車/公車

        label = f"{name} {conf:.2f}"
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)

# 3. 儲存並印出統計
cv2.imwrite("merged_obb_result.jpg", annotated_img)
print("偵測完成！統計結果：", counts)