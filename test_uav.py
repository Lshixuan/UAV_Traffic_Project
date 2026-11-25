from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# 1. 自動從 Hugging Face 下載 uavdt-yolo26m 權重檔
# 註：建議先至該模型網頁的 "Files and versions" 確認實際的 .pt 檔名（通常為 best.pt）
model_path = hf_hub_download(
    repo_id="dronefreak/uavdt-yolo26m",
    filename="best.pt"
)

# 2. 載入模型
model = YOLO(model_path)

# 3. 測試對「單張圖片」進行車輛偵測
model.predict(
    source="test.jpg",
    conf=0.25,
    imgsz=640,
    save=True  # 結果會儲存在 runs/detect/predict/ 資料夾內
)
