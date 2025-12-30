import cv2
import time
import threading
import numpy as np
import libcamera
from LOBOROBOT2 import LOBOROBOT
from picamera2 import Picamera2
from flask import Flask, Response, render_template_string, jsonify

# ==================================================================
# 網頁前端 (HTML)
# ==================================================================
INDEX_HTML = """
<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <title>Raspberry Pi Lane Keeper (Blob Detection)</title>
  <style>
    body { font-family: sans-serif; text-align: center; background-color: #f0f0f0; }
    img { border-radius:8px; border:2px solid #333; max-width: 100%; }
    button { padding: 10px 20px; margin: 5px; font-size: 16px; cursor: pointer; border-radius: 5px; }
    .control-group { margin: 15px 0; padding: 10px; background: #fff; display: inline-block; border-radius: 8px; }
  </style>
</head>
<body>
  <h3>Pi 雙線車道維持系統 (白線區塊偵測版)</h3>
  
  <img src="/live" alt="Video Stream" style="width:640px;">
  <br>
  
  <div class="control-group">
      <h4>鏡頭控制</h4>
      <button onclick="fetch('/api/turn_left', {method:'POST'})">◀ 左轉</button>
      <button onclick="fetch('/api/turn_right',{method:'POST'})">右轉 ▶</button>
      <button onclick="fetch('/api/turn_up',   {method:'POST'})">▲ 上轉</button>
      <button onclick="fetch('/api/turn_down', {method:'POST'})">下轉 ▼</button>
  </div>

  <div class="control-group">
      <h4>車體控制</h4>
      <button id="start"  style="background-color:#4CAF50; color:white;" onclick="fetch('/api/start',{method:'POST'})">啟動 (Start)</button>
      <button id="stop"   style="background-color:#f44336; color:white;" onclick="fetch('/api/stop', {method:'POST'})">停止 (Stop)</button>
      <br>
      <button id="accel"  onclick="fetch('/api/accelerate', {method:'POST'})">加速 (W)</button>
      <button id="decel"  onclick="fetch('/api/decelerate', {method:'POST'})">減速 (S)</button>
      <br>
      <button id="leftf"  onclick="fetch('/api/left_forward', {method:'POST'})">左前 (A)</button>
      <button id="rightf" onclick="fetch('/api/right_forward',{method:'POST'})">右前 (D)</button>
      <br>
      <button id="autogo" style="background-color:#2196F3; color:white;" onclick="fetch('/api/autogo',{method:'POST'})">沿線自走 (Auto)</button>
  </div>

  <script>
    // 鍵盤監聽事件，讓電腦鍵盤也能控制小車
    window.addEventListener('keydown', (e) => {
      if (e.repeat) return; // 防止長按重複觸發過快
      if (['w','W','ArrowUp'].includes(e.key)) document.getElementById('accel').click();
      if (['s','S','ArrowDown'].includes(e.key)) document.getElementById('decel').click();
      if (['a','A','ArrowLeft'].includes(e.key)) document.getElementById('leftf').click();
      if (['d','D','ArrowRight'].includes(e.key)) document.getElementById('rightf').click();
      if (['g','G'].includes(e.key)) document.getElementById('autogo').click();
      if (e.key === ' ') document.getElementById('stop').click();
    });
  </script>
</body>
</html>
"""
# ==================================================================
# 全域變數與參數設定
# ==================================================================
ww, hh = 320, 240   # 攝影機解析度 
Cam_X, Cam_Y = 10, 9 # 伺服馬達 (Servo) 的控制腳位 ID

# 鏡頭角度限制
angle, updown = 90, 20
MIN_ANGLE, MAX_ANGLE = 20, 150
MIN_UPDOWN, MAX_UPDOWN = -2.5, 40
STEP_ANGLE = 5  # 每次按鍵轉動的角度

# 速度與狀態變數
speed = 20                  # 基礎速度
l_ofs, r_ofs = 0, 0         
car_go, auto_go = 0, 0      
lost_counter = 0            # 紀錄連續幾幀沒看到線 (用於防呆停止)

SPEED_MIN, SPEED_MAX, SPEED_STEP = 0, 100, 5

WHITE_THRESHOLD = 200       # 亮度門檻 
MIN_CONTOUR_AREA = 100      # 最小區塊面積：過濾雜訊
LANE_WIDTH_PIXELS = 260     # 預估車道寬度 (像素)
ROI_START_ROW = 140         # ROI REGION

# --- Thread Locks ---
# 防止多個執行緒同時讀寫同一個變數導致衝突
frame_lock = threading.Lock()
robot_lock = threading.Lock() 
speed_lock = threading.Lock()
angle_lock = threading.Lock()

# --- 硬體初始化 ---
# Picamera2 設定
picamera = Picamera2()
config = picamera.create_preview_configuration(
    main={"format": "RGB888", "size": (ww, hh)},
    raw={"format": "SRGGB12", "size": (ww, hh)},
)
# 設定影像翻轉 (若鏡頭倒裝需開啟)
config["transform"] = libcamera.Transform(hflip=1, vflip=1)
picamera.configure(config)
picamera.start()

# LOBOROBOT 馬達驅動板設定
clbrobot = LOBOROBOT()
with robot_lock:
    clbrobot.t_stop(0.1) # 初始狀態先停止
# ==================================================================
# 輔助函式
# ==================================================================
def safe_move(base_speed, l_offset, r_offset):
    """
    馬達安全保護函式：
    確保計算後的馬達 PWM 數值落在 0~100 之間
    """
    target_l = base_speed + l_offset
    target_r = base_speed + r_offset
    
    # 限制範圍 (Clamp)
    safe_target_l = max(0, min(100, target_l))
    safe_target_r = max(0, min(100, target_r))
    
    # 反推回 Offset 以便呼叫 move_with_offset
    new_safe_l_ofs = safe_target_l - base_speed
    new_safe_r_ofs = safe_target_r - base_speed
    
    with robot_lock:
        clbrobot.move_with_offset(int(base_speed), int(new_safe_l_ofs), int(new_safe_r_ofs), 0.05)
# ==================================================================
# 視覺演算法 
# ==================================================================
def process_lane_blob(img_rgb):
    global lost_counter, speed, ww, hh
    
    final_view = img_rgb.copy()     # 用於繪圖顯示結果
    
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    roi_gray = blurred[ROI_START_ROW:, :]    # ROI REGION 裁剪  
    _, binary = cv2.threshold(roi_gray, WHITE_THRESHOLD, 255, cv2.THRESH_BINARY) # 二值化
    # 找出所有白色區塊的外框
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 初始化左右線候選清單
    left_contours = []
    right_contours = []
    mid_screen = ww // 2 # 畫面水平中心點 X 座標
    
    # 遍歷所有找到的輪廓
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > MIN_CONTOUR_AREA: # 過濾太小的噪點
            # 計算輪廓的「力矩」(Moments)，用來找重心
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"]) # 重心 X
                cy = int(M["m01"] / M["m00"]) # 重心 Y (相對於 ROI)
                real_cy = cy + ROI_START_ROW
            
                # 依據重心在畫面左側還是右側，分類為左線或右線
                if cx < mid_screen:
                    left_contours.append((area, cx, real_cy, cnt))
                else:
                    right_contours.append((area, cx, real_cy, cnt))
    
    best_left = None
    best_right = None
    # 面積最大的區塊當作車道線
    if left_contours:
        best_left = max(left_contours, key=lambda x: x[0])
    if right_contours:
        best_right = max(right_contours, key=lambda x: x[0])
        
    # 導航運算
    found_line = False
    steering = 0
    current_center_x = mid_screen # 預設當前行進目標為畫面正中央
    
    cv2.line(final_view, (mid_screen, 0), (mid_screen, hh), (100, 100, 100), 1)
    if best_left is not None and best_right is not None:
        # 雙線模式：左右線都看到了
        # 目標：走在兩條線的正中間
        lx, ly = best_left[1], best_left[2]
        rx, ry = best_right[1], best_right[2]
        current_center_x = (lx + rx) // 2
        found_line = True
        cv2.drawContours(final_view, [best_left[3] + [0, ROI_START_ROW]], -1, (0, 255, 0), 2)
        cv2.drawContours(final_view, [best_right[3] + [0, ROI_START_ROW]], -1, (0, 255, 0), 2)
        cv2.putText(final_view, "Double Mode", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    elif best_left is not None:
        # 左線模式：只看到左線
        # 目標：假裝右線存在，走在左線 + 車道寬的一半
        lx, ly = best_left[1], best_left[2]
        current_center_x = lx + (LANE_WIDTH_PIXELS // 2)
        found_line = True
        cv2.drawContours(final_view, [best_left[3] + [0, ROI_START_ROW]], -1, (255, 255, 0), 2)
        cv2.putText(final_view, "Left Track", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    elif best_right is not None:
        # 右線模式：只看到右線
        # 目標：假裝左線存在，走在右線 - 車道寬的一半
        rx, ry = best_right[1], best_right[2]
        current_center_x = rx - (LANE_WIDTH_PIXELS // 2)
        found_line = True
        cv2.drawContours(final_view, [best_right[3] + [0, ROI_START_ROW]], -1, (255, 255, 0), 2)
        cv2.putText(final_view, "Right Track", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    if found_line:
        lost_counter = 0
        # 計算誤差 (Error) = 畫面中心(理想) - 當前路徑中心(實際)
        # 正值代表車偏左(需右轉)，負值代表車偏右(需左轉)
        error = (ww // 2) - current_center_x
        cv2.circle(final_view, (int(current_center_x), hh - 40), 10, (0, 0, 255), -1)
        steering = int(error * 0.30)
        steering = max(min(steering, 45), -45)  # 限制轉向最大幅度
        cv2.putText(final_view, f"Steer: {steering}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        current_l_ofs = -steering
        current_r_ofs = steering
        safe_move(speed, current_l_ofs, current_r_ofs)
        
    else:
        # 完全沒看到線
        lost_counter += 1
        cv2.putText(final_view, "LOST", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        # 如果連續 60 幀都沒看到線，強制停車
        if lost_counter > 60:
            with robot_lock:
                clbrobot.t_stop(0.05)                
    return final_view

# ==================================================================
# 背景執行緒 (核心控制迴圈)
# ==================================================================
running = True
latest_frame = None

def capture_loop():
    """
    獨立於 Web Server 的執行緒。
    負責：持續抓圖 -> 判斷模式 (手動/自動) -> 執行對應邏輯 -> 更新全域影像變數
    """
    global picamera, latest_frame, running
    global car_go, auto_go, speed, l_ofs, r_ofs
    target_fps = 20 # 目標幀率
    interval = 1.0 / target_fps
    
    while running:
        start_time = time.time()
        
        try:
            # 從硬體抓取最新畫面 (Numpy Array 格式)
            frame = picamera.capture_array()
        except Exception as e:
            print("Capture error:", e)
            time.sleep(0.1)
            continue
        
        if auto_go == 1:
            processed_frame = process_lane_blob(frame.copy())
            with frame_lock:
                latest_frame = processed_frame
        elif car_go == 1:
            with speed_lock:
                curr_speed = speed
                curr_l = l_ofs
                curr_r = r_ofs
            safe_move(curr_speed, curr_l, curr_r)
            with frame_lock:
                latest_frame = frame 
        else:
            with frame_lock:
                latest_frame = frame
        elapsed = time.time() - start_time
        if elapsed < interval:
            time.sleep(interval - elapsed)
t = threading.Thread(target=capture_loop, daemon=True)
t.start()
# ==================================================================
# Flask Web Server 設定
# ==================================================================
app = Flask(__name__)
def live_mjpeg():
    """
    MJPEG 串流產生器
    """
    while True:
        with frame_lock:
            if latest_frame is None:
                time.sleep(0.01)
                continue
            frame_to_show = latest_frame.copy()    
        ok, jpeg = cv2.imencode(".jpg", cv2.cvtColor(frame_to_show, cv2.COLOR_RGB2BGR),
                                [int(cv2.IMWRITE_JPEG_QUALITY), 60]) 
        if not ok: continue
        time.sleep(0.05) # 控制串流傳輸頻率       
        # 產生 Multipart 回應格式
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" +
               jpeg.tobytes() +
               b"\r\n")

@app.route("/")
def index():
    """首頁：回傳 HTML 控制介面"""
    return render_template_string(INDEX_HTML)

@app.route("/live")
def video_feed():
    """影像串流路徑"""
    return Response(live_mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

# --- API 實作 (接收前端按鈕請求) ---

@app.route("/api/turn_left", methods=["POST"])
def turn_left():
    global angle
    with angle_lock:
        angle = min(MAX_ANGLE, angle + STEP_ANGLE)
    with robot_lock:
        clbrobot.set_servo_angle(Cam_X, int(angle), 0.1)
    return jsonify({"ok": True, "angle": angle})

@app.route("/api/turn_right", methods=["POST"])
def turn_right():
    global angle
    with angle_lock:
        angle = max(MIN_ANGLE, angle - STEP_ANGLE)
    with robot_lock:
        clbrobot.set_servo_angle(Cam_X, int(angle), 0.1)
    return jsonify({"ok": True, "angle": angle})

@app.route("/api/turn_up", methods=["POST"])
def turn_up():
    global updown
    with angle_lock:
        updown = max(MIN_UPDOWN, updown - STEP_ANGLE/2)
    with robot_lock:
        clbrobot.set_servo_angle(Cam_Y, int(updown), 0.1)
    return jsonify({"ok": True, "updown": updown})

@app.route("/api/turn_down", methods=["POST"])
def turn_down():
    global updown
    with angle_lock:
        updown = min(MAX_UPDOWN, updown + STEP_ANGLE/2)
    with robot_lock:
        clbrobot.set_servo_angle(Cam_Y, int(updown), 0.1)
    return jsonify({"ok": True, "updown": updown})

@app.route("/api/accelerate", methods=["POST"])
def accelerate():
    global car_go, speed, l_ofs, r_ofs
    with speed_lock:
        if car_go == 1:
            speed = min(speed + SPEED_STEP, SPEED_MAX)
            l_ofs, r_ofs = 0, 0 # 加速時重置轉向偏移
    return jsonify({"ok": True, "speed": speed})

@app.route("/api/decelerate", methods=["POST"])
def decelerate():
    global car_go, speed, l_ofs, r_ofs
    with speed_lock:
        if car_go == 1:
            speed = max(speed - SPEED_STEP, SPEED_MIN)
            l_ofs, r_ofs = 0, 0
    return jsonify({"ok": True, "speed": speed})

@app.route("/api/left_forward", methods=["POST"])
def left_forward():
    global car_go, l_ofs
    with speed_lock:
        if car_go == 1:
            l_ofs -= 10 # 調整左輪差速
    return jsonify({"ok": True, "l_ofs": l_ofs})

@app.route("/api/right_forward", methods=["POST"])
def right_forward():
    global car_go, r_ofs
    with speed_lock:
        if car_go == 1:
            r_ofs -= 10 # 調整右輪差速
    return jsonify({"ok": True, "r_ofs": r_ofs})

@app.route("/api/start", methods=["POST"])
def start():
    """手動啟動模式"""
    global car_go, auto_go, l_ofs, r_ofs
    with speed_lock:
        l_ofs, r_ofs = 0, 0
        car_go = 1
        auto_go = 0
    return jsonify({"ok": True, "mode": "manual"})

@app.route("/api/stop", methods=["POST"])
def stop():
    """停止所有動作"""
    global car_go, auto_go
    with speed_lock:
        car_go = 0
        auto_go = 0
    with robot_lock:
        clbrobot.t_stop(0.1)
    return jsonify({"ok": True, "go": False})

@app.route("/api/autogo", methods=["POST"])
def autogo():
    """自動駕駛模式啟動"""
    global car_go, auto_go, l_ofs, r_ofs, lost_counter
    with speed_lock:
        l_ofs, r_ofs = 0, 0
        lost_counter = 0
        car_go = 0
        auto_go = 1
    
    # 切換到自動模式時，自動將鏡頭歸位 (水平居中，垂直向下看)
    with robot_lock:
        clbrobot.set_servo_angle(Cam_X, 90, 0.1)
        clbrobot.set_servo_angle(Cam_Y, 15, 0.1)
        
    return jsonify({"ok": True, "mode": "auto"})

def cleanup():
    """程式結束前的資源釋放"""
    global running
    running = False
    with robot_lock:
        clbrobot.t_stop(0.1)
    try:
        if t.is_alive(): t.join(timeout=1.0)
        picamera.stop()
    except Exception:
        pass

if __name__ == "__main__":
    # 程式啟動時先將雲台復位
    with robot_lock:
        clbrobot.set_servo_angle(Cam_X, angle, 1)
        clbrobot.set_servo_angle(Cam_Y, updown, 1)

    try:
        # 啟動 Flask 伺服器
        # host="0.0.0.0" 允許區網內其他裝置連線
        app.run(host="0.0.0.0", port=5000, threaded=True, debug=False)
    finally:
        cleanup()