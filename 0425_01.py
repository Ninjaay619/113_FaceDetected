import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time

# 初始化攝影機
cap = cv2.VideoCapture(0)

# 初始化主視窗
window = tk.Tk()
window.title("臉部偵測擷取系統")
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

        # 灰階處理
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 偵測人臉
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
        current_faces = faces

        # 畫紅框
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # 顯示在 Tkinter
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
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
