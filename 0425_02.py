import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import face_recognition
import numpy as np
import time

# 初始化攝影機
cap = cv2.VideoCapture(0)

# 初始化主視窗
window = tk.Tk()
window.title("臉部偵測與部位標示")
window.geometry("720x580")

# 顯示畫面的標籤
label = tk.Label(window)
label.pack()

# 載入 Haar Cascade 模型（人臉偵測）
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 儲存目前畫面的人臉資料
current_faces = []
current_frame = None

def update_frame():
    global current_faces, current_frame

    ret, frame = cap.read()
    if ret:
        current_frame = frame.copy()

        # 將圖像從 BGR 轉換為 RGB 格式
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 確保圖像是 8 位元的 RGB 圖像
        rgb_frame = np.uint8(rgb_frame)

        # 偵測人臉
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        current_faces = faces

        # 顯示臉部關鍵點
        for (x, y, w, h) in faces:
            # 裁切出臉部區域
            face_img = frame[y:y+h, x:x+w]

            # 偵測臉部關鍵點
            rgb_face = rgb_frame[y:y+h, x:x+w]  # 只處理臉部區域
            face_landmarks_list = face_recognition.face_landmarks(rgb_face)
            if face_landmarks_list:
                face_landmarks = face_landmarks_list[0]

                # 繪製綠色方框並標示各部位
                points = {
                    "眉心": face_landmarks['nose_bridge'][0],       # 27
                    "鼻頭": face_landmarks['nose_tip'][2],          # 30
                    "人中": face_landmarks['top_lip'][3],           # 33
                    "左臉頰": face_landmarks['chin'][2],             # 2
                    "右臉頰": face_landmarks['chin'][14],            # 14
                    "額頭": (face_landmarks['nose_bridge'][0][0], face_landmarks['nose_bridge'][0][1] - 40)  # 推估
                }

                # 畫綠色方框
                for name, point in points.items():
                    x, y = point
                    cv2.circle(frame, (x + x, y + y), 5, (0, 255, 0), -1)  # 調整顯示位置
                    cv2.putText(frame, name, (x + x + 5, y + y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 顯示在 Tkinter
        img = Image.fromarray(rgb_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.configure(image=imgtk)

    # 每 10 毫秒更新畫面
    window.after(10, update_frame)

def capture_faces():
    global current_faces, current_frame

    if current_frame is None or len(current_faces) == 0:
        print("⚠️ 未偵測到人臉，無法擷取")
        return

    for i, (x, y, w, h) in enumerate(current_faces):
        face_img = current_frame[y:y+h, x:x+w]
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"face_{timestamp}_{i+1}.jpg"
        cv2.imwrite(filename, face_img)
        print(f"✅ 臉部已儲存為 {filename}")

        # 顯示擷取的照片
        cv2.imshow("擷取的臉部", face_img)

def close_app():
    cap.release()
    window.destroy()

# 按鈕區塊
btn_frame = ttk.Frame(window)
btn_frame.pack(pady=10)

capture_btn = ttk.Button(btn_frame, text="擷取臉部畫面", command=capture_faces)
capture_btn.pack(side=tk.LEFT, padx=10)

close_btn = ttk.Button(btn_frame, text="關閉視窗", command=close_app)
close_btn.pack(side=tk.LEFT, padx=10)

# 開始畫面更新
update_frame()
window.mainloop()
