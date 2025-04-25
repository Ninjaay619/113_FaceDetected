import dlib
import cv2
import numpy as np

# 加載預訓練的人臉檢測器
detector = dlib.get_frontal_face_detector()

# 加載人臉特徵點檢測器
predictor = dlib.shape_predictor("C:/Users/jay/Desktop/Python/0425/shape_predictor_68_face_landmarks.dat")

# 讀取圖像
img = cv2.imread('Photo_human_1.jpeg')

# 檢查圖像是否成功加載
if img is None:
    print("無法加載圖像，請檢查路徑。")
else:
    # 確保圖像為 RGB 格式，這對 dlib 來說是必需的
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 將圖像轉為灰度圖（8-bit 灰度圖）
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    # 確認圖像是否成功轉換為 8-bit 灰度圖
    if gray.dtype != np.uint8:
        print("錯誤：圖像轉換為灰度圖後的類型不是 uint8！")
    else:
        # 檢測人臉
        faces = detector(gray)

        # 處理每一張檢測到的人臉
        for face in faces:
            # 偵測面部特徵點
            landmarks = predictor(gray, face)

            # 繪製面部特徵點
            for n in range(0, 68):  # 68 個面部特徵點
                x, y = landmarks.part(n).x, landmarks.part(n).y
                cv2.circle(img, (x, y), 1, (0, 0, 255), -1)  # 用紅色標示特徵點

        # 顯示結果
        cv2.imshow("Face Detection with 68 Landmarks", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
