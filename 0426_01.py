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
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    if gray.dtype != np.uint8:
        print("錯誤：圖像轉換為灰度圖後的類型不是 uint8！")
    else:
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)

            # 繪製68個面部特徵點
            for n in range(68):
                x, y = landmarks.part(n).x, landmarks.part(n).y
                cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

            # 畫三個長方形
            def draw_rect(img, pt1_idx, pt2_idx, color, label):
                x1, y1 = landmarks.part(pt1_idx).x, landmarks.part(pt1_idx).y
                x2, y2 = landmarks.part(pt2_idx).x, landmarks.part(pt2_idx).y
                top_left = (min(x1, x2), min(y1, y2))
                bottom_right = (max(x1, x2), max(y1, y2))
                cv2.rectangle(img, top_left, bottom_right, color, 2)
                cv2.putText(img, label, (top_left[0], top_left[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            draw_rect(img, 40, 4, (0, 0, 255),"") #右臉頰
            draw_rect(img, 47 , 12, (0, 0, 255),"") #左臉頰
            draw_rect(img, 32, 52, (0, 255, 255),"") #人中
            draw_rect(img, 39, 32,(255,0 ,255), "") #右鼻翼
            draw_rect(img, 42, 34,(255,0 ,255), "") #左鼻翼

            def draw_square_on_point(point_index, color, label):
                cx = landmarks.part(point_index).x
                cy = landmarks.part(point_index).y
                box_size = 20
                top_left = (cx - box_size // 2, cy - box_size // 2)
                bottom_right = (cx + box_size // 2, cy + box_size // 2)
                cv2.rectangle(img, top_left, bottom_right, color, 2)
                cv2.putText(img, label, (top_left[0], top_left[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

            draw_square_on_point(30, (0, 0, 0),"") #鼻頭(脾)
            draw_square_on_point(27, (0, 0, 0),"") #鼻根(肺)

        # 顯示結果
        cv2.imshow("Face Detection with Custom Rectangles", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
