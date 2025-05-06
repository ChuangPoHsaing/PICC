import tkinter as tk
import cv2
import numpy as np
import time
from PIL import Image, ImageTk, ImageFont, ImageDraw
from ultralytics import YOLO
import openpyxl  
import os
from math import sqrt
import queue
import threading
import sys
import torch

# 啟用CUDA加速（如果可用）
if torch.cuda.is_available():
    torch.set_default_device('cuda')
    print("CUDA加速已啟用")
else:
    print("CUDA不可用，使用CPU")

def show_sterile_area_error():
    """
    顯示無菌區未準備好的錯誤視窗
    """
    error_window = tk.Tk()
    error_window.title("錯誤")
    error_window.geometry("400x200")
    
    # 置中顯示
    screen_width = error_window.winfo_screenwidth()
    screen_height = error_window.winfo_screenheight()
    x = (screen_width - 400) // 2
    y = (screen_height - 200) // 2
    error_window.geometry(f"400x200+{x}+{y}")
    
    # 錯誤訊息
    message = tk.Label(
        error_window,
        text="請先準備好無菌區\n\n需要準備：\n- 手套\n- 管路底座\n- 無菌紗布\n- ",
        font=("Arial", 14)
    )
    message.pack(pady=20)
    
    # 確認按鈕
    def close_and_exit():
        error_window.destroy()
        sys.exit()
    
    ok_button = tk.Button(
        error_window,
        text="確認",
        font=("Arial", 12),
        command=close_and_exit
    )
    ok_button.pack(pady=10)
    
    error_window.mainloop()

def check_sterile_area_initial(model, video_path):
    """
    初始檢查無菌區
    返回是否通過檢查
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    
    # 讀取5幀進行檢查，確保檢測穩定
    required_items = {1}  # 需要的物品ID
    success_count = 0
    
    cv2.namedWindow("Sterile Area Check", cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 降低解析度以提高檢測速度
        frame = cv2.resize(frame, (960, 540))
        display_frame = frame.copy()
        
        # 定義無菌區位置
        square_size = 350  # 因為解析度降低，方框大小也需調整
        margin = 50
        top_left_x = frame.shape[1] - square_size - margin
        top_left_y = margin
        
        # 繪製綠色方框
        cv2.rectangle(display_frame, 
                     (top_left_x, top_left_y), 
                     (top_left_x + square_size, top_left_y + square_size), 
                     (0, 255, 0), 
                     3)
        
        # 執行物件偵測
        results = model.predict(frame, conf=0.5)
        
        # 檢查無菌區內的物品
        found_items = set()
        
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                # 獲取座標
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                # 獲取類別ID
                C_id = int(box.cls.cpu().numpy()[0])

                
                # 在畫面上標示檢測到的物品
                cv2.rectangle(display_frame, 
                            (int(x1), int(y1)), 
                            (int(x2), int(y2)), 
                            (255, 0, 0), 
                            2)
                
                # 檢查物體中心點是否在無菌區內
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                if (top_left_x <= center_x <= top_left_x + square_size and
                    top_left_y <= center_y <= top_left_y + square_size and
                    C_id in required_items):
                    found_items.add(C_id)
        
        # 顯示找到的物品數量
        cv2.putText(display_frame, 
                   f"Found Items: {len(found_items)}/{len(required_items)}", 
                   (50, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   1, 
                   (0, 255, 0), 
                   2)
        
        # 顯示畫面
        cv2.imshow("Sterile Area Check", display_frame)
        key = cv2.waitKey(10)
        
        # 檢查是否找到所有物品
        if found_items == required_items:
            success_count += 1
        else:
            success_count = 0
            
        # 如果連續3幀都檢測到所有物品，就算成功
        if success_count >= 3:
            cv2.destroyWindow("Sterile Area Check")
            cap.release()
            return True
            
        # 按ESC鍵退出
        if key == 27:
            cv2.destroyWindow("Sterile Area Check")
            cap.release()
            return False
    
    cv2.destroyWindow("Sterile Area Check")
    cap.release()
    return False

def calculate_distance(point1, point2):
    """
    計算兩點之間的歐氏距離
    :param point1: 第一個點的座標 (x, y)
    :param point2: 第二個點的座標 (x, y)
    :return: 兩點之間的距離
    """
    return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


# 優化紅色區域檢測函數，使用更輕量級的操作
def detect_red_area_optimized(roi):
    """
    檢測紅色區域，使用更快的方法
    """
    # 轉換到HSV色彩空間
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # 紅色的HSV範圍
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([8, 255, 200])
    lower_red2 = np.array([165, 120, 70])
    upper_red2 = np.array([180, 255, 200])
    
    # 創建遮罩 - 使用位元運算加速
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)
    
    # 使用更小的結構元素進行形態學操作
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    
    return red_mask

# 改進區域覆蓋率計算函數，使用更高效的numpy操作
def get_zone_coverage_optimized(red_mask, zone_coords):
    """
    計算紅色區域在指定區域內的覆蓋率，優化版
    """
    try:
        x1, y1, x2, y2 = map(int, zone_coords)
        
        # 提取該區域的遮罩
        zone_mask = red_mask[y1:y2, x1:x2]
        
        # 使用numpy的sum和size方法更高效地計算覆蓋率
        red_pixels = np.sum(zone_mask > 0)
        zone_area = zone_mask.size
        
        if zone_area == 0:
            return 0.0
            
        coverage = (red_pixels / zone_area) * 100
        return min(coverage, 100.0)
        
    except Exception:
        return 0.0

class DisinfectionZones:
    def __init__(self):
        self.zones = {}
        self.coverage = {
            'top': 0,
            'bottom': 0,
            'left': 0,
            'right': 0
        }
        self.threshold = 15
        self.last_update_time = {}
        # 添加緩存，減少重複計算
        self.last_red_mask = None
        self.last_update_frame = 0
        self.update_interval = 5  # 每5幀更新一次覆蓋率

    def update_zones(self, picc_bbox):
        """
        更新四個消毒區域的座標
        """
        x1, y1, x2, y2 = picc_bbox
        width = x2 - x1
        height = y2 - y1
        mid_x = x1 + width // 2
        mid_y = y1 + height // 2

        self.zones = {
            'top': (x1, y1, x2, mid_y),
            'bottom': (x1, mid_y, x2, y2),
            'left': (x1, y1, mid_x, y2),
            'right': (mid_x, y1, x2, y2)
        }

    def update_coverage(self, red_mask, current_frame):
        """
        更新各個區域的覆蓋率，引入幀間隔
        """
        # 每隔幾幀更新一次，減少計算量
        if current_frame % self.update_interval != 0:
            return
            
        self.last_red_mask = red_mask  # 保存遮罩以供繪製使用
        current_time = time.time()
        
        for zone_name, zone_coords in self.zones.items():
            # 使用優化版覆蓋率計算
            new_coverage = get_zone_coverage_optimized(red_mask, zone_coords)
            
            # 更新覆蓋率，使用最大值
            self.coverage[zone_name] = max(self.coverage[zone_name], new_coverage)
            
            # 更新最後更新時間
            self.last_update_time[zone_name] = current_time

    def is_disinfection_complete(self):
        """
        檢查是否所有區域都達到所需的覆蓋率
        """
        complete = all(coverage >= self.threshold for coverage in self.coverage.values())
        if complete:
            # 重置覆蓋率，為下一次消毒做準備
            self.coverage = {k: 0 for k in self.coverage}
        return complete

    def draw_zones(self, track_frame):
        """
        在追蹤畫面上繪製區域和覆蓋率
        """
        if not hasattr(self, 'zones') or not self.zones:
            return
            
        colors = {
            'top': (255, 0, 0),     # 藍色
            'bottom': (0, 255, 0),   # 綠色
            'left': (0, 0, 255),     # 紅色
            'right': (255, 255, 0)   # 青色
        }
        
        # 繪製紅色區域
        if self.last_red_mask is not None:
            red_overlay = np.zeros_like(track_frame)
            for zone_name, zone_coords in self.zones.items():
                x1, y1, x2, y2 = zone_coords
                zone_mask = np.zeros((track_frame.shape[0], track_frame.shape[1]), dtype=np.uint8)
                zone_mask_roi = self.last_red_mask[y1:y2, x1:x2]
                zone_mask[y1:y2, x1:x2] = zone_mask_roi
                red_overlay[zone_mask > 0] = [0, 0, 180]
            
            # 使用更快的圖像混合方法
            alpha = 0.4
            track_frame[:] = cv2.addWeighted(track_frame, 1-alpha, red_overlay, alpha, 0)
        
        # 繪製區域邊框
        for zone_name, zone_coords in self.zones.items():
            x1, y1, x2, y2 = zone_coords
            color = colors[zone_name]
            cv2.rectangle(track_frame, (x1, y1), (x2, y2), color, 2)

        # 在左上角顯示覆蓋率資訊
        start_y = 30
        line_spacing = 30
        
        # 設定背景矩形
        text_bg_width = 200
        text_bg_height = 140
        cv2.rectangle(track_frame, (10, 10), (10 + text_bg_width, 10 + text_bg_height), (255, 255, 255), -1)
        
        # 添加標題
        cv2.putText(
            track_frame,
            "Zone Coverage (15%):",
            (20, start_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )
        
        # 顯示每個區域的覆蓋率
        for i, (zone_name, coverage) in enumerate(self.coverage.items()):
            y_position = start_y + (i + 1) * line_spacing
            text = f"{zone_name}: {coverage:.1f}%"
            color = colors[zone_name]
            
            cv2.putText(
                track_frame,
                text,
                (20, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            # 如果覆蓋率達到閾值，顯示勾勾
            if coverage >= self.threshold:
                cv2.putText(
                    track_frame,
                    "V",
                    (160, y_position),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

# 使用較小的數據更新間隔，減少I/O操作
def save_to_excel(employee_id, employee_name, disinfect_count, glove_worn, tegaderm_film_removed, base_removed, cotton_swab, clear_data=False):
    file_path = "PICC_training_data.xlsx"
    
    if not os.path.exists(file_path) or clear_data:
        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "PICC訓練數據"
        ws.append(["員工ID", "姓名", "消毒次數", "佩戴手套", "移除Tegaderm film", "移除底座", "消毒棉棒準備"])
    else:
        wb = openpyxl.load_workbook(file_path)
        ws = wb.active
    
    ws.append([
        employee_id, 
        employee_name, 
        disinfect_count,
        "是" if glove_worn else "否",
        "是" if tegaderm_film_removed else "否",
        "是" if base_removed else "否",
        "是" if cotton_swab else "否"
    ])
    
    wb.save(file_path) 

# 緩存常用的中文字體
cached_fonts = {}
def put_chinese_text(img, text, position, font_path, font_size, color):
    # 優化字體載入，使用緩存
    font_key = f"{font_path}_{font_size}"
    if font_key not in cached_fonts:
        cached_fonts[font_key] = ImageFont.truetype(font_path, font_size)
    font = cached_fonts[font_key]
    
    # 使用PIL處理文字
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

def draw_square_on_frame(frame, top_left_x, top_left_y, square_size, color=(0, 255, 0), thickness=2):
    """
    在影像上繪製正方形框
    """
    bottom_right_x = top_left_x + square_size
    bottom_right_y = top_left_y + square_size
    cv2.rectangle(frame, (top_left_x, top_left_y), 
                 (bottom_right_x, bottom_right_y), color, thickness)


# 主類別
class PICCTrainingSystem:
    def __init__(self, video_path, model_path, font_path):
        # 載入模型時指定設備，考慮使用半精度浮點數(fp16)加速
        self.model = YOLO(model_path)
        
        # 優化模型推理設定
        self.model.fuse()  # 融合模型層，提高推理速度
        
        # 設置攝像頭
        self.cap = cv2.VideoCapture(video_path)
        # 設定較低的解析度以提高FPS
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        self.font_path = font_path
        self.lock = threading.Lock()
        self.frame_queue = queue.Queue(maxsize=5)  # 減少隊列大小，避免延遲
        self.stop_event = threading.Event()
        self.disinfection_zones = DisinfectionZones()
        
        # 檢測參數
        self.detection_interval = 2  # 每隔幾幀進行一次完整檢測
        self.frame_count = 0
        self.display_frame_count = 0
        
        # 初始化追蹤變數
        self.employee_id = ""
        self.employee_name = ""
        self.glove_worn = False
        self.tegaderm_film_removed = True
        self.base_removed = True
        self.cotton_swab = False
        self.disinfect_count = 0
        self.names = self.model.names
        self.excel_data = []
        self.save_interval = 15  # 增加保存間隔到15秒
        
        # 檢測結果快取
        self.cached_results = None
        self.cached_red_mask = None

    def frame_capture_thread(self):
        while not self.stop_event.is_set():
            success, frame = self.cap.read()
            if not success:
                break
            try:
                # 降低解析度以提高處理速度
                frame = cv2.resize(frame, (1280, 720))
                
                # 只有當隊列未滿時才放入新幀，避免積壓
                if not self.frame_queue.full():
                    self.frame_queue.put(frame, block=False)
            except (queue.Full, RuntimeError):
                continue

    def object_detection_thread(self):
        cv2.namedWindow("YOLOv8 PICC Training", cv2.WINDOW_NORMAL)
        
        # 初始化 FPS 計算
        fps_start_time = time.time()
        fps_counter = 0
        fps = 0
        
        # 預先分配記憶體給顯示圖像
        track_frame = np.zeros((720, 1280, 3), np.uint8)
        track_frame.fill(255)
        
        last_detection_time = time.time()
        
        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=0.1)
                self.frame_count += 1
                self.display_frame_count += 1
                
                current_time = time.time()
                # 計算滾動平均FPS
                fps_counter += 1
                if current_time - fps_start_time > 1.0:
                    fps = fps_counter / (current_time - fps_start_time)
                    fps_counter = 0
                    fps_start_time = current_time
                
                # 減少深拷貝的使用，直接在原始幀上繪製
                display_frame = frame.copy()
                
                # 每隔幾幀才執行一次完整的物體檢測
                run_full_detection = (self.frame_count % self.detection_interval == 0)
                
                # 執行物體檢測，隔幀檢測以提高速度
                if run_full_detection or self.cached_results is None:
                    # 優化YOLO參數，降低IoU閾值會減少計算量
                    results = self.model.track(
                        frame, 
                        tracker="bytetrack.yaml", 
                        persist=True, 
                        verbose=False, 
                        conf=0.5,
                        iou=0.45,  # 降低IoU閾值
                        imgsz=640  # 使用較小的處理尺寸
                    )
                    
                    # 更新快取
                    self.cached_results = results
                    last_detection_time = current_time
                
                # 處理檢測結果
                self.process_detection_optimized(frame, display_frame, track_frame, self.cached_results, self.frame_count)
                
                # 每5秒更新一次追蹤畫面，減少計算量
                if current_time - last_detection_time > 5.0:
                    track_frame.fill(255)  # 重置追蹤畫面
                    if hasattr(self, 'disinfection_zones'):
                        self.disinfection_zones.draw_zones(track_frame)
                    last_detection_time = current_time
                
                # 在每個窗口的幀上顯示FPS
                cv2.putText(display_frame, f'FPS: {fps:.1f}', (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                
                # 顯示狀態資訊
                display_frame = self.display_status(display_frame)
                
                # 使用單一視窗顯示，將三個視圖合併為一個
                combined_frame = np.zeros((720, 1280*2, 3), dtype=np.uint8)
                combined_frame[:, :1280] = display_frame
                combined_frame[:, 1280:] = track_frame
                
                # 顯示合併的畫面
                cv2.imshow("YOLOv8 PICC Training", combined_frame)
                
                if cv2.waitKey(1) & 0xFF == 27:
                    self.stop_event.set()
                    
            except queue.Empty:
                continue
                
    def track_picc_distance(self, frame, results):
        """
        追蹤PICC管線與設置點的距離
        """
        # 如果還未設置基準點，嘗試設置
        if not hasattr(self, 'picc_base_point'):
            for box in results[0].boxes.data:
                C_id = int(box[6])
                # 如果檢測到PICC
                if C_id == 2:  # 假設2是PICC的類別ID
                    x1, y1, x2, y2 = map(int, box[:4])
                    # 使用PICC邊界框的中心作為基準點
                    self.picc_base_point = ((x1 + x2) // 2, (y1 + y2) // 2)
                    self.picc_base_tracking_start_time = time.time()
                    break
        
        # 如果已設置基準點，則檢查距離
        if hasattr(self, 'picc_base_point'):
            for box in results[0].boxes.data:
                C_id = int(box[6])
                if C_id == 2:  # PICC
                    x1, y1, x2, y2 = map(int, box[:4])
                    current_point = ((x1 + x2) // 2, (y1 + y2) // 2)
                    
                    # 計算距離
                    distance = calculate_distance(self.picc_base_point, current_point)
                    
                    # 轉換為實際距離（假設像素與實際距離的比例）
                    # 這個比例需要根據您的實際情況調整
                    actual_distance = distance * 0.1  # 每像素代表0.1cm
                    
                    # 如果距離超過5cm
                    if actual_distance > 5:
                        # 顯示警告
                        cv2.putText(
                            frame, 
                            f"Warning: PICC movement >5cm! ({actual_distance:.1f}cm)", 
                            (50, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.8,  # 縮小字體大小
                            (0, 0, 255), 
                            2
                        )

    def process_detection_optimized(self, frame, display_frame, track_frame, results, frame_count):
        """
        優化版的檢測處理函數
        """
        # 減少鎖的使用範圍，只鎖定關鍵區域
        with self.lock:
            PICC_boundaryX1, PICC_boundaryY1, PICC_boundaryX2, PICC_boundaryY2 = 0, 0, 0, 0
            class_4_count = 0
            
            # 先檢查現有的狀態，避免每幀都重置
            current_status = {
                'glove_worn': False,
                'tegaderm_film_removed': True,
                'base_removed': True,
                'cotton_swab': False
            }
            
            # 檢查PICC距離
            if hasattr(self, 'track_picc_distance') and frame_count % 5 == 0:  # 降低檢查頻率
                self.track_picc_distance(display_frame, results)
            
            # 檢查是否有檢測結果
            if len(results) > 0 and results[0].boxes.id is not None:
                boxes = results[0].boxes.data
                
                # 遍歷所有檢測到的物件
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box[:4])
                    trackid = int(box[4])
                    r = round(float(box[5]), 2)
                    C_id = int(box[6])
                    
                    # 減少顯示資訊，
                    # 減少顯示資訊，只顯示物件名和ID
                    label = f"{self.names[C_id]} {trackid}"
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                 
                    # 記錄PICC ID為2
                    if C_id == 2:
                        PICC_boundaryX1, PICC_boundaryY1, PICC_boundaryX2, PICC_boundaryY2 = x1, y1, x2, y2
                        
                        # 更新消毒區域
                        if hasattr(self, 'disinfection_zones'):
                            self.disinfection_zones.update_zones((x1, y1, x2, y2))
                    
                    # 更新物品狀態檢查
                    if C_id == 1:  # 手套
                        current_status['glove_worn'] = True
                    elif C_id == 6:  # Tegaderm film
                        current_status['tegaderm_film_removed'] = False
                    elif C_id == 3:  # 底座
                        current_status['base_removed'] = False
                    
                    # 檢查類別ID為4的物體計數
                    if C_id == 4:  # 假設4是需要計數的類別ID
                        current_status['cotton_swab'] = True
                        class_4_count += 1
            
                    # 更新狀態
                    self.glove_worn = current_status['glove_worn']
                    self.cotton_swab = current_status['cotton_swab']
                    self.tegaderm_film_removed = current_status['tegaderm_film_removed']
                    self.base_removed = current_status['base_removed']
            
                    # 處理紅色區域檢測（消毒區域）
            if PICC_boundaryX1 > 0 and frame_count % 5 == 0:  # 降低處理頻率
                # 從邊界框擴展區域，獲取ROI
                roi_x1 = max(0, PICC_boundaryX1 - 100)
                roi_y1 = max(0, PICC_boundaryY1 - 100)
                roi_x2 = min(frame.shape[1], PICC_boundaryX2 + 100)
                roi_y2 = min(frame.shape[0], PICC_boundaryY2 + 100)
                
                # 獲取ROI
                roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
                
                if roi.size > 0:
                    # 檢測紅色區域
                    red_mask = detect_red_area_optimized(roi)
                    
                    # 將局部紅色遮罩轉換為全局坐標
                    global_red_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                    global_red_mask[roi_y1:roi_y2, roi_x1:roi_x2] = red_mask
                    
                    # 更新消毒區域覆蓋率
                    if hasattr(self, 'disinfection_zones'):
                        self.disinfection_zones.update_coverage(global_red_mask, frame_count)
                        self.disinfection_zones.draw_zones(track_frame)
                        
                        # 檢查消毒是否完成
                        if self.disinfection_zones.is_disinfection_complete():
                            self.disinfect_count += 1
                            
                            # 顯示完成消毒的訊息
                            cv2.putText(
                                display_frame,
                                "消毒完成！",
                                (int(frame.shape[1]/2) - 100, int(frame.shape[0]/2)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (0, 255, 0),
                                3
                            )
                            
                            # 使用中文顯示
                            display_frame = put_chinese_text(
                                display_frame,
                                "消毒完成！",
                                (int(frame.shape[1]/2) - 150, int(frame.shape[0]/2) + 50),
                                self.font_path,
                                45,
                                (0, 255, 0)
                            )
                            
                            # 定時保存數據到Excel
                            if time.time() - getattr(self, 'last_save_time', 0) > self.save_interval:
                                self.save_data_to_excel()
                                self.last_save_time = time.time()



    def display_status(self, frame):
        """
        顯示當前訓練狀態，使用中文顯示姓名
        """
        # 使用純OpenCV方法，避免創建新物件
        start_y = 80
        line_height = 40
        
        # 設定半透明背景區域
        status_width = 400
        status_height = 240
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, start_y - 30), (10 + status_width, start_y + status_height), (255, 255, 255), -1)
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # 添加標題
        cv2.putText(frame, "Training Status:", (20, start_y), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 2)
        
        # 顯示學員ID (使用英文顯示)
        cv2.putText(frame, f"ID: {self.employee_id}", (20, start_y + line_height), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # 使用中文顯示學員姓名
        frame = put_chinese_text(
            frame,
            f"姓名: {self.employee_name}",
            (20, start_y + 2 * line_height),
            self.font_path,
            24,  # 適合的字體大小
            (0, 0, 0)  # 黑色
        )
        
        # 顯示檢測狀態
        cv2.putText(frame, f"Gloves: {'Yes' if self.glove_worn else 'No'}", (20, start_y + 3 * line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.glove_worn else (0, 0, 255), 2)
        
        cv2.putText(frame, f"Cotton Swab: {'Yes' if self.cotton_swab else 'No'}", (20, start_y + 4 * line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.cotton_swab else (0, 0, 255), 2)
        
        # 顯示Tegaderm Film移除狀態
        cv2.putText(frame, f"Tegaderm Film Removed: {'Yes' if self.tegaderm_film_removed else 'No'}", (20, start_y + 5 * line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.tegaderm_film_removed else (0, 0, 255), 2)
        
        # 顯示底座移除狀態
        cv2.putText(frame, f"Base Removed: {'Yes' if self.base_removed else 'No'}", (20, start_y + 6 * line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if self.base_removed else (0, 0, 255), 2)
        
        # 顯示消毒次數
        cv2.putText(frame, f"Disinfection Count: {self.disinfect_count}", (20, start_y + 7 * line_height), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame

# 修改學員資訊輸入視窗，確保能正確處理中文輸入

    def save_data_to_excel(self):
        """
        保存訓練數據到Excel
        """
        if self.employee_id and self.employee_name:
            save_to_excel(
                self.employee_id,
                self.employee_name,
                self.disinfect_count,
                self.glove_worn,
                self.tegaderm_film_removed,
                self.base_removed,
                self.cotton_swab
            )
            print(f"已保存訓練數據，消毒次數: {self.disinfect_count}")

    def start(self):
        """
        啟動系統
        """
        # 先檢查無菌區準備情況
        if not check_sterile_area_initial(self.model, 1):  # 0代表預設攝像頭
            show_sterile_area_error()
            return

        # 創建學員資訊輸入視窗
        self.create_input_window()

    def create_input_window(self):
        """
        創建學員資訊輸入視窗，支持中文輸入
        """
        input_window = tk.Tk()
        input_window.title("學員資訊")
        input_window.geometry("400x250")
        
        # 正確引入字體模塊
        import tkinter.font as tkfont
        
        # 設置窗口字體為支持中文的字體
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(family="Microsoft JhengHei", size=10)  # 微軟正黑體
        
        # 置中顯示
        screen_width = input_window.winfo_screenwidth()
        screen_height = input_window.winfo_screenheight()
        x = (screen_width - 400) // 2
        y = (screen_height - 250) // 2
        input_window.geometry(f"400x250+{x}+{y}")
        
        # 建立標籤和輸入框
        tk.Label(input_window, text="學員ID:", font=("Microsoft JhengHei", 14)).grid(row=0, column=0, padx=20, pady=20, sticky="w")
        id_entry = tk.Entry(input_window, font=("Microsoft JhengHei", 14), width=15)
        id_entry.grid(row=0, column=1, padx=20, pady=20)
        
        tk.Label(input_window, text="學員姓名:", font=("Microsoft JhengHei", 14)).grid(row=1, column=0, padx=20, pady=20, sticky="w")
        name_entry = tk.Entry(input_window, font=("Microsoft JhengHei", 14), width=15)
        name_entry.grid(row=1, column=1, padx=20, pady=20)
        
        # 啟動按鈕
        def start_training():
            self.employee_id = id_entry.get()
            self.employee_name = name_entry.get()
            
            if not self.employee_id or not self.employee_name:
                tk.messagebox.showerror("錯誤", "請輸入學員ID和姓名")
                return
                
            input_window.destroy()
            
            # 啟動系統線程
            self.last_save_time = time.time()
            self.frame_thread = threading.Thread(target=self.frame_capture_thread)
            self.detection_thread = threading.Thread(target=self.object_detection_thread)
            
            self.frame_thread.daemon = True
            self.detection_thread.daemon = True
            
            self.frame_thread.start()
            self.detection_thread.start()
            
            # 先等待一下，確保其他線程已經啟動
            time.sleep(0.5)
            
            # 顯示提示視窗
            self.show_instructions()
        
        start_button = tk.Button(
            input_window,
            text="開始訓練",
            font=("Microsoft JhengHei", 14),
            command=start_training
        )
        start_button.grid(row=2, column=0, columnspan=2, pady=30)
        
        # 將輸入欄位設置為預設焦點
        id_entry.focus_set()
        
        input_window.mainloop()

# 優化 put_chinese_text 函數，確保能正確處理中文
    def put_chinese_text(img, text, position, font_path, font_size, color):
        # 優化字體載入，使用緩存
        font_key = f"{font_path}_{font_size}"
        if font_key not in cached_fonts:
            try:
                cached_fonts[font_key] = ImageFont.truetype(font_path, font_size)
            except IOError:
                # 如果找不到指定字體，使用系統預設字體
                print(f"無法找到指定字體: {font_path}，使用預設字體")
                cached_fonts[font_key] = ImageFont.load_default()
        font = cached_fonts[font_key]
        
        # 使用PIL處理文字
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text(position, text, font=font, fill=color)
        return np.array(img_pil)

    def show_instructions(self):
        """
        顯示訓練指導視窗
        """
        root = tk.Tk()
        root.withdraw()  # 隱藏根視窗
        
        instruction_window = tk.Toplevel(root)
        instruction_window.title("訓練指導")
        instruction_window.geometry("600x400")
        
        # 置中顯示
        screen_width = instruction_window.winfo_screenwidth()
        screen_height = instruction_window.winfo_screenheight()
        x = (screen_width - 600) // 2
        y = (screen_height - 400) // 2
        instruction_window.geometry(f"600x400+{x}+{y}")
        
        # 指導內容
        instructions = [
            "PICC訓練系統使用指導：",
            "",
            "1. 請確保工作區域光線充足",
            "2. 確保PICC線在鏡頭視野範圍內",
            "3. 必須佩戴手套進行操作",
            "4. 準備消毒棉棒",
            "5. 請按照規範進行PICC消毒",
            "6. 系統會自動追蹤消毒的四個區域",
            "7. 每個區域需達到15%的覆蓋率才算完成",
            "8. 完成所有區域消毒後，計數將自動增加",
            "9. 按ESC鍵結束訓練",
            "",
            "祝訓練順利！"
        ]
        
        # 添加指導文字
        text_widget = tk.Text(instruction_window, font=("Arial", 12), wrap=tk.WORD)
        text_widget.pack(padx=20, pady=20, fill=tk.BOTH, expand=True)
        
        for line in instructions:
            text_widget.insert(tk.END, line + "\n")
        
        text_widget.config(state=tk.DISABLED)
        
        # 確認按鈕
        def close_window():
            instruction_window.destroy()
            root.destroy()  # 同時關閉根視窗
        
        confirm_button = tk.Button(
            instruction_window,
            text="我已了解",
            font=("Arial", 12),
            command=close_window
        )
        confirm_button.pack(pady=10)
        
        # 確保視窗在前面顯示
        instruction_window.lift()
        instruction_window.attributes('-topmost', True)
        instruction_window.after_idle(instruction_window.attributes, '-topmost', False)
        
        # 設定當視窗關閉時的行為
        instruction_window.protocol("WM_DELETE_WINDOW", close_window)
    
        # 主循環
        root.mainloop()
        
    def stop(self):
        """
        停止系統
        """
        self.stop_event.set()
        if hasattr(self, 'frame_thread'):
            self.frame_thread.join(timeout=1.0)
        if hasattr(self, 'detection_thread'):
            self.detection_thread.join(timeout=1.0)
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        # 最後保存數據
        self.save_data_to_excel()
        print("系統已停止")

# 主函數
def main():
    # 設置參數
    video_path = 1  # 使用預設攝像頭
    model_path = "D:\yolov8 final\PICC_1030.pt"  # YOLOv8模型路徑
    font_path = "D:\yolov8 final\PICC-camera\TaipeiSansTCBeta-Bold.ttf"  # 字體路徑，請確保存在
    
    # 創建系統實例
    system = PICCTrainingSystem(video_path, model_path, font_path)
    
    try:
        # 啟動系統
        system.start()
        
        # 等待主執行緒結束
        while not system.stop_event.is_set():
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("程式被使用者中斷")
    finally:
        # 停止系統
        system.stop()

if __name__ == "__main__":
    main()
    