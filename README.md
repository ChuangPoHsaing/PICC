# PICC 安全消毒與管線辨識系統

本專案是一個基於 YOLOv8 的智慧醫療影像辨識系統，用於識別並監控 PICC（外周中心靜脈導管）導管與消毒過程中棉花棒的接觸行為，避免醫療失誤並提升消毒流程標準化程度。

---

## 🔍 功能特點

* **即時目標偵測**：辨識 PICC 導管與棉花棒物件。
* **碰撞偵測系統**：根據紅色橢圓（模擬棉花棒消毒區域）與四個綠點（模擬消毒位置）是否接觸進行流程判斷。
* **消毒步驟追蹤**：依序偵測棉花棒接觸 top → right → bottom → left 四個點以計算一次完整消毒。
* **異常警告提示**：

  * 未移除 Tegaderm 膜。
  * 管路底座未拆除。
  * 管線拉出超過 5cm 。
* **即時影像儲存與紀錄輸出**：當消毒完成或異常事件發生時，自動儲存影像並記錄到 Excel。

---

## 🛠️ 安裝指南

請先確認已安裝以下套件與環境：

### 系統需求

* Python 3.8+

### 安裝步驟

```bash
git clone https://github.com/your_username/picc-safety-detection.git
cd picc-safety-detection

# 安裝必要套件
pip install -r requirements.txt
```

---

## ▶️ 使用方法

1. 開啟外接鏡頭
2. 執行主程式：
3. 預設輸出儲存在 `results/`，含：

   * 標註後的影片
   * Excel 紀錄檔（含消毒次數與異常提示）
   * 異常情況下擷取之靜態影像

---

## 📁 項目架構

```
picc-safety-detection/
├── detect_picc.py              # 主程式
├── yolov8_utils.py             # YOLOv8 推論與座標處理工具
├── disinfect_tracker.py        # 消毒碰撞邏輯與計數器
├── requirements.txt            # 所需套件
├── videos/                     # 原始輸入影片
├── results/                    # 偵測後影片與報告儲存區
└── weights/
    └── best.pt                 # YOLOv8 預訓練模型
```

---

## 🧠 算法原理

### 1. **物件偵測 YOLOv8**

使用 Ultralytics YOLOv8 進行物件辨識，並透過置信度門檻（預設 0.5）過濾預測。

### 2. **棉花棒紅色橢圓區域識別**

* 根據 ID 與顏色偵測出特定的紅色區域。
* 偵測出該紅色區域中心點與四個綠色控制點（top, right, bottom, left）的距離。

### 3. **碰撞流程控制**

* 當紅色區域依序碰觸四個綠點，即視為完成一次消毒。
* 使用邏輯鎖與 flags 控制執行順序，避免誤判。

### 4. **異常判定邏輯**

* 管路拉出 5cm 以上：以畫面比例 1cm=2pixels 判定距離。
* 若未移除特定物件（Tegaderm 膜或底座），顯示提示訊息。

---

## ⚠️ 注意事項

* **模型格式**：僅支援 YOLOv8 格式之 `.pt` 模型。
* **畫面比例校正**：需依照實際相機設定正確的 1cm 對應 pixel 比例。
* **光線與顏色識別**：請保持影像環境穩定，避免紅色與背景混淆。

---

## 🤝 貢獻指南

歡迎任何形式的貢獻，包括但不限於：

* Bug 回報
* 功能建議
* Pull Request 改進程式碼

請先閱讀 [CONTRIBUTING.md](CONTRIBUTING.md) 了解詳細流程。

---

## 📚 相關資源

* [Ultralytics YOLOv8 官方文件](https://docs.ultralytics.com/)
* [OpenCV](https://opencv.org/)
* [Python 官方網站](https://www.python.org/)
* [PICC 介紹 - 衛福部資料](https://www.mohw.gov.tw/)
