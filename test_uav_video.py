import os
import glob
import cv2
from huggingface_hub import hf_hub_download, list_repo_files
from ultralytics import YOLO

# 1. 載入模型
repo_id = "dronefreak/visdrone-yolo26m"
pt_files = [f for f in list_repo_files(repo_id) if f.endswith(".pt")]
model = YOLO(hf_hub_download(repo_id=repo_id, filename=pt_files[0]))

# 2. 讀取資料夾內所有圖片並按檔名排序（確保時間順序正確）
image_folder = "DataVideo"
image_paths = sorted(glob.glob(os.path.join(image_folder, "*.jpg")))

if not image_paths:
    raise FileNotFoundError(f"在 {image_folder} 找不到任何 .jpg 圖片！")

# 取得第一張圖片尺寸，初始化 MP4 影片寫入器（設定 30 FPS）
first_img = cv2.imread(image_paths[0])
h, w, _ = first_img.shape
out_video = cv2.VideoWriter("tracked_sequence.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))

# 3. 逐幀執行追蹤與自訂繪圖
target_classes = [0, 1, 2, 3, 4, 5, 8, 9]

for idx, img_path in enumerate(image_paths):
    img = cv2.imread(img_path)
    # 必須設定 persist=True 才能跨圖片延續追蹤 ID
    res = model.track(source=img, tracker="botsort.yaml", classes=target_classes, conf=0.25, imgsz=1280, persist=True, verbose=False)[0]

    if res.boxes is not None and res.boxes.id is not None:
        boxes = res.boxes.xyxy.cpu().numpy()
        ids = res.boxes.id.int().cpu().numpy()
        clss = res.boxes.cls.int().cpu().numpy()

        for box, track_id, cls_id in zip(boxes, ids, clss):
            x1, y1, x2, y2 = map(int, box)
            name = model.names[cls_id]

            if cls_id in [0, 1]:
                color = (255, 100, 0)    # 藍色：行人
            elif cls_id in [2, 9]:
                color = (0, 165, 255)    # 橘色：機車/自行車
            else:
                color = (0, 255, 0)      # 綠色：汽車/貨車/公車

            label = f"ID:{track_id} {name}"
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out_video.write(img)
    if (idx + 1) % 30 == 0 or (idx + 1) == len(image_paths):
        print(f"進度：已處理 {idx + 1} / {len(image_paths)} 張圖片")

out_video.release()
print("全部處理完成！已自動合成追蹤影片：tracked_sequence.mp4")