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
        text="請先準備好無菌區\n\n需要準備：\n- 無菌墊\n- 手套\n- 管路底座\n- 無菌紗布\n- 人工皮",
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
    required_items = {0, 1, 3, 5, 6}  # 需要的物品ID
    success_count = 0
    
    cv2.namedWindow("Sterile Area Check", cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, (1920, 1080))
        display_frame = frame.copy()
        
        # 定義無菌區位置
        square_size = 700
        margin = 100
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
        key = cv2.waitKey(1)
        
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


def detect_and_display_red_area(frame, track_frame, bbox):
    """
    檢測紅色區域並創建遮罩，使用更深的紅色
    """
    # 提取 ROI
    x1, y1, x2, y2 = bbox
    roi = frame[y1:y2, x1:x2]

    # 轉換到 HSV 色彩空間
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 調整紅色的 HSV 範圍來檢測更深的紅色
    # H: 降低上限使紅色更純，S: 提高下限使顏色更飽和，V: 降低上限使顏色更深
    lower_red1 = np.array([0, 120, 70])    # 提高飽和度下限
    upper_red1 = np.array([8, 255, 200])   # 降低明度上限
    lower_red2 = np.array([165, 120, 70])  # 提高飽和度下限
    upper_red2 = np.array([180, 255, 200]) # 降低明度上限

    # 創建遮罩
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # 去除小噪點
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    # 創建與原始圖像相同大小的遮罩
    full_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
    full_mask[y1:y2, x1:x2] = red_mask

    # 在track_frame上顯示檢測到的紅色區域，使用更深的紅色
    overlay = track_frame.copy()
    red_area = np.zeros_like(frame)
    red_area[full_mask > 0] = [0, 0, 180]  # 使用更深的紅色 [B, G, R]
    
    # 調整混合比例使紅色更明顯
    cv2.addWeighted(overlay, 0.4, red_area, 0.6, 0, track_frame)  # 增加紅色區域的權重

    return full_mask

def get_zone_coverage(red_mask, zone_coords):
    """
    計算紅色區域在指定區域內的覆蓋率
    """
    try:
        x1, y1, x2, y2 = map(int, zone_coords)
        
        # 提取該區域的遮罩
        zone_mask = red_mask[y1:y2, x1:x2]
        
        # 計算區域面積和紅色像素數量
        zone_area = (x2 - x1) * (y2 - y1)
        red_pixels = np.count_nonzero(zone_mask)
        
        if zone_area == 0:
            return 0.0
            
        # 計算覆蓋率並加入調試信息
        coverage = (red_pixels / zone_area) * 100
        print(f"Zone {zone_coords}: Area = {zone_area}, Red pixels = {red_pixels}, Coverage = {coverage:.2f}%")
        
        return min(coverage, 100.0)
        
    except Exception as e:
        print(f"覆蓋率計算錯誤: {e}")
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
        self.threshold = 15  # 將閾值從25%降低到15%
        self.last_update_time = {}

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

    def update_coverage(self, red_mask):
        """
        更新各個區域的覆蓋率
        """
        current_time = time.time()
        
        for zone_name, zone_coords in self.zones.items():
            # 計算新的覆蓋率
            new_coverage = get_zone_coverage(red_mask, zone_coords)
            
            # 更新覆蓋率，使用最大值
            self.coverage[zone_name] = max(self.coverage[zone_name], new_coverage)
            
            # 更新最後更新時間
            self.last_update_time[zone_name] = current_time
            
            # 列印調試信息
            print(f"{zone_name} zone coverage: {self.coverage[zone_name]:.2f}%")

    def is_disinfection_complete(self):
        """
        檢查是否所有區域都達到所需的覆蓋率（15%）
        """
        complete = all(coverage >= self.threshold for coverage in self.coverage.values())
        if complete:
            print("消毒完成！所有區域都達到15%覆蓋率")
            # 重置覆蓋率，為下一次消毒做準備
            self.coverage = {k: 0 for k in self.coverage}
        return complete

    def draw_zones(self, track_frame):
        """
        在追蹤畫面上繪製區域和覆蓋率
        """
        colors = {
            'top': (255, 0, 0),     # 藍色
            'bottom': (0, 255, 0),   # 綠色
            'left': (0, 0, 255),     # 紅色
            'right': (255, 255, 0)   # 青色
        }
        
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
            "Zone Coverage (15%):",  # 更新標題顯示閾值
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
def save_to_excel(employee_id, employee_name, disinfect_count, glove_worn, tegaderm_film_removed, base_removed, cotton_swab, clear_data=False):
    file_path = "PICC_training_data.xlsx"
    
    if not os.path.exists(file_path)or clear_data:
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

def put_chinese_text(img, text, position, font_path, font_size, color):
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)
    return np.array(img_pil)

def draw_square_on_frame(frame, top_left_x, top_left_y, square_size, color=(0, 255, 0), thickness=2):
    """
    在影像上繪製正方形框
    :param frame: 當前的影像幀
    :param top_left_x: 正方形左上角的 x 座標
    :param top_left_y: 正方形左上角的 y 座標
    :param square_size: 正方形的邊長
    :param color: 正方形顏色 (B, G, R)
    :param thickness: 線條厚度
    """
    bottom_right_x = top_left_x + square_size
    bottom_right_y = top_left_y + square_size
    top_left = (top_left_x, top_left_y)
    bottom_right = (bottom_right_x, bottom_right_y)
    cv2.rectangle(frame, top_left, bottom_right, color, thickness)



# 主類別
class PICCTrainingSystem:
    def __init__(self, video_path, model_path, font_path):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_path)
        self.font_path = font_path
        self.lock = threading.Lock()
        self.frame_queue = queue.Queue(maxsize=10)
        self.stop_event = threading.Event()
        self.disinfection_zones = DisinfectionZones()

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
        



    def frame_capture_thread(self):
        while not self.stop_event.is_set():
            success, frame = self.cap.read()
            if not success:
                break
            try:
                frame = cv2.resize(frame, (1920, 1080))
                self.frame_queue.put(frame, timeout=1)
            except queue.Full:
                continue



    def object_detection_thread(self):
        cv2.namedWindow("YOLOv8-track", cv2.WINDOW_NORMAL)
        cv2.namedWindow("YOLOv8", cv2.WINDOW_NORMAL)
        cv2.namedWindow("track", cv2.WINDOW_NORMAL)   

    # 初始化 FPS 計算
        prev_time = time.time()

        while not self.stop_event.is_set():
            try:
                frame = self.frame_queue.get(timeout=1)
                frame_copy = frame.copy()
                track_frame = np.zeros((1080, 1920, 3), np.uint8)
                track_frame.fill(255)
                
            

                # 計算 FPS
                current_time = time.time()
                fps = 1 / (current_time - prev_time)
                prev_time = current_time

                results = self.model.track(frame, tracker="bytetrack.yaml", persist=True, verbose=False, conf=0.5, iou=0.6 # 非極大值抑制的 IoU 閾值)
                       ) 

                # 偵測處理，確保傳遞正確參數
                self.process_detection(frame, frame_copy, track_frame, results)

                # 顯示 FPS
                cv2.putText(frame, f'FPS: {fps:.2f}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                square_size = 700  # 設定正方形的邊長
                margin = 100  # 正方形與邊界的間距
                top_left_x = frame.shape[1] - square_size - margin
                top_left_y = margin
                draw_square_on_frame(frame, top_left_x, top_left_y, square_size, color=(0, 255, 0), thickness=3)
                frame = self.display_status(frame)

                # 顯示畫面
                cv2.imshow("YOLOv8-track", frame)
                cv2.imshow("YOLOv8", frame_copy)
                cv2.imshow("track", track_frame)

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
                            f"Warning: PICC line movement exceeds 5cm! distance{actual_distance:.2f}cm", 
                            (50, 200), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1, 
                            (0, 0, 255), 
                            2
                        )
    

    def process_detection(self, frame, frame_copy, track_frame, results):
        with self.lock:
            PICC_boundaryX1, PICC_boundaryY1, PICC_boundaryX2, PICC_boundaryY2 = 0, 0, 0, 0
            class_4_count = 0
            
            # 每一幀都重置狀態
            self.glove_worn = False
            self.tegaderm_film_removed = True
            self.base_removed = True
            self.cotton_swab = False
            self.cotton_swab_count = 0
           
            self.track_picc_distance(frame, results)
            # 檢查是否有檢測結果
            if results[0].boxes.id is not None:
                boxes = results[0].boxes.data
                
                # 遍歷所有檢測到的物件
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box[:4])
                    trackid = int(box[4])
                    r = round(float(box[5]), 2)
                    C_id = int(box[6])
                    n = self.names[C_id]
                    
                    # 顯示檢測框和資訊
                    inf_show = f"#{trackid} {r} {n}"
                    cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (255, 128, 128), 3)
                    cv2.putText(frame_copy, inf_show, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    # 根據物件類別更新狀態
                    if C_id == 1:  # 手套
                        self.glove_worn = True
                        
                    elif C_id == 2:  # PICC
                        PICC_boundaryX1, PICC_boundaryY1, PICC_boundaryX2, PICC_boundaryY2 = x1, y1, x2, y2
                        cv2.rectangle(frame, (PICC_boundaryX1, PICC_boundaryY1), 
                                    (PICC_boundaryX2, PICC_boundaryY2), (128, 0, 0), 3)
                        self.disinfection_zones.update_zones((x1, y1, x2, y2))
                        
                    elif C_id == 4:  # 消毒棉棒
                        self.cotton_swab = True
                        class_4_count += 1
                        self.cotton_swab_count = class_4_count
                        
                        # 檢測紅色區域
                        red_mask = detect_and_display_red_area(frame, track_frame, (x1, y1, x2, y2))
                        
                        # 更新消毒區域覆蓋率
                        if hasattr(self, 'disinfection_zones'):
                            self.disinfection_zones.update_coverage(red_mask)
                            
                            # 檢查是否完成消毒
                            if self.disinfection_zones.is_disinfection_complete():
                                self.disinfect_count += 1
                                print(f"消毒次數增加到: {self.disinfect_count}")
                            
                            # 繪製區域和覆蓋率
                            self.disinfection_zones.draw_zones(track_frame)
                            
                    elif C_id == 6:  # Tegaderm film
                        self.tegaderm_film_removed = False
                        
                    elif C_id == 3:  # 底座
                        self.base_removed = False

        # 更新 cotton_swab_count，確保在沒有檢測到消毒棉棒時為0
        if class_4_count == 0:
            self.cotton_swab_count = 0
               

                
            

    def display_status(self, frame):
        """
        Display the current system status on the frame.
        :param frame: The current video frame to overlay status text.
        """
        # Text for various statuses
        status_text_glove = "成功穿戴手套" if self.glove_worn else "未成功穿戴"
        status_text_film = "成功移除第一層tegaderm film" if self.tegaderm_film_removed else "未移除 第一層tegaderm film"
        status_text_base = "成功移除管路底座" if self.base_removed else "未移除 管路底座"
        status_cotton_swab = f"現有: {self.cotton_swab_count}支消毒棉棒" if self.cotton_swab_count > 0 else "未準備消毒棉花棒"
        status_disinfect = f"消毒次數: {self.disinfect_count}" if self.disinfect_count > 0 else "未成功消毒"

        # Display the text on the frame using `put_chinese_text`
        frame = put_chinese_text(frame, status_text_glove, (50, 300), self.font_path, 50, (0, 255, 0))
        frame = put_chinese_text(frame, status_text_film, (50, 400), self.font_path, 50, (255, 0, 0))
        frame = put_chinese_text(frame, status_text_base, (50, 500), self.font_path, 50, (0, 0, 255))
        frame = put_chinese_text(frame, status_cotton_swab, (50, 600), self.font_path, 50, (128, 128, 0))
        frame = put_chinese_text(frame, status_disinfect, (50, 700), self.font_path, 50, (0, 128, 255))

        return frame




    def save_data_thread(self):
        while not self.stop_event.is_set():
            with self.lock:
                save_to_excel(self.employee_id, self.employee_name,
                              self.disinfect_count, self.glove_worn,
                              self.tegaderm_film_removed,
                              self.base_removed, self.cotton_swab)
            time.sleep(5)

    def start_processing(self):
    # 清空上一次的数据
        save_to_excel("", "", 0, False, False, False, False, clear_data=True)
         # 清空前次數據
        self.disinfect_count = 0
        self.glove_worn = False
        self.tegaderm_film_removed = True
        self.base_removed = True
        self.cotton_swab_count = 0
        self.cotton_swab = False
        
    
        threads = [
            threading.Thread(target=self.frame_capture_thread),
            threading.Thread(target=self.object_detection_thread),
            threading.Thread(target=self.save_data_thread)
        ]
        for thread in threads:
            thread.start()
        return threads


    def stop_processing(self):
        self.stop_event.set()
        self.cap.release()
        cv2.destroyAllWindows()



def get_user_input():
    user_data = {}
    input_window = tk.Tk()
    input_window.title("輸入資料")
    input_window.geometry("500x400")

    tk.Label(input_window, text="請輸入您的工號:", font=("Arial", 16)).pack(pady=10)
    employee_id_entry = tk.Entry(input_window, font=("Arial", 14), width=30)
    employee_id_entry.pack(pady=5)

    tk.Label(input_window, text="請輸入您的姓名:", font=("Arial", 16)).pack(pady=10)
    employee_name_entry = tk.Entry(input_window, font=("Arial", 14), width=30)
    employee_name_entry.pack(pady=5)

    def submit():
        employee_id = employee_id_entry.get() if employee_id_entry.get() else "無"
        employee_name = employee_name_entry.get() if employee_name_entry.get() else "匿名"
        user_data['employee_id'] = employee_id
        user_data['employee_name'] = employee_name
        input_window.destroy()

    submit_button = tk.Button(input_window, text="提交", font=("Arial", 14), command=submit)
    submit_button.pack(pady=20)

    input_window.mainloop()
    return user_data

def show_main_window():
    main_window = tk.Tk()
    main_window.title("PICC 訓練系統")
    main_window.geometry("600x500")

    # 加入圖片
    image_path = r"ui2.jpg"  # 替換為您的圖片路徑
    if os.path.exists(image_path):
        img = Image.open(image_path)
        img = img.resize((300, 300))
        photo = ImageTk.PhotoImage(img)
        img_label = tk.Label(main_window, image=photo)
        img_label.image = photo  # 避免圖片被垃圾回收
        img_label.pack(pady=10)
    else:
        tk.Label(main_window, text="找不到圖片", font=("Arial", 14)).pack(pady=10)

    # 加入按鈕
    tk.Label(main_window, text="歡迎使用 PICC 訓練系統", font=("Arial", 18)).pack(pady=10)
    def open_input_window():
        main_window.destroy()


    start_button = tk.Button(main_window, text="開始訓練", font=("Arial", 14), command=open_input_window)
    start_button.pack(pady=20)

    main_window.mainloop()
    

def main():
    font_path = "TaipeiSansTCBeta-Bold.ttf"
    video_path = 1
    model_path = "PICC_1030.pt"
    
    # 1. 顯示主視窗
    show_main_window()
    
    # 2. 獲取使用者輸入
    user_input = get_user_input()
    
    # 3. 載入模型並檢查無菌區
    model = YOLO(model_path)
    if not check_sterile_area_initial(model, video_path):
        show_sterile_area_error()
        return
    
    # 4. 開始主程式
    system = PICCTrainingSystem(video_path, model_path, font_path)
    system.employee_id = user_input['employee_id']
    system.employee_name = user_input['employee_name']

    try:
        threads = system.start_processing()
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        print("停止中...")
    finally:
        system.stop_processing()

if __name__ == "__main__":
    main()

