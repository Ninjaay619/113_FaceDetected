import dlib
import cv2
import numpy as np
import os

# 設定檔案路徑
img_path = 'Photo_human_2.jpg'
predictor_path = "C:/Users/jay/Desktop/Python/0425/shape_predictor_68_face_landmarks.dat"

# 建立輸出資料夾
output_dir = "output"
os.makedirs(output_dir, exist_ok=True)

# 初始化 dlib 檢測器與特徵點預測器
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


def detect_forehead(landmarks):
    eyebrow_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 27)]
    min_y = min([p[1] for p in eyebrow_points])
    left_x = landmarks.part(17).x
    right_x = landmarks.part(26).x
    forehead_height = int((right_x - left_x) * 0.4)
    hairline_y = min_y - forehead_height

    outline_points = []
    for x in range(left_x, right_x + 1, 3):
        outline_points.append((x, min_y))
    for y in range(min_y, hairline_y - 1, -3):
        outline_points.append((right_x, y))
    for x in range(right_x, left_x - 1, -3):
        outline_points.append((x, hairline_y))
    for y in range(hairline_y, min_y + 1, 3):
        outline_points.append((left_x, y))

    segment_height = forehead_height // 3
    y_coords = [min_y - i * segment_height for i in range(4)]

    boxes = []
    centers = []
    for i in range(3):
        top_left = (left_x, y_coords[i + 1])
        bottom_right = (right_x, y_coords[i])
        center_x = (left_x + right_x) // 2
        center_y = (y_coords[i] + y_coords[i + 1]) // 2
        boxes.append((top_left, bottom_right))
        centers.append((center_x - 10, center_y + 5))
    return outline_points, boxes, centers


def save_box_image(img, top_left, bottom_right, filename):
    x1, y1 = top_left
    x2, y2 = bottom_right
    cropped_img = img[y1:y2, x1:x2]
    if cropped_img.size > 0:
        cv2.imwrite(os.path.join(output_dir, filename), cropped_img)


def draw_rect(img, pt1_idx, pt2_idx, color, label):
    x1, y1 = landmarks.part(pt1_idx).x, landmarks.part(pt1_idx).y
    x2, y2 = landmarks.part(pt2_idx).x, landmarks.part(pt2_idx).y
    top_left = (min(x1, x2), min(y1, y2))
    bottom_right = (max(x1, x2), max(y1, y2))
    cv2.rectangle(img, top_left, bottom_right, color, 2)
    if label:
        cv2.putText(img, label, (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    save_box_image(img, top_left, bottom_right, label + ".jpg")


def draw_square_on_point(img, point_index, color, label):
    cx = landmarks.part(point_index).x
    cy = landmarks.part(point_index).y
    box_size = 20
    top_left = (cx - box_size // 2, cy - box_size // 2)
    bottom_right = (cx + box_size // 2, cy + box_size // 2)
    cv2.rectangle(img, top_left, bottom_right, color, 2)
    if label:
        cv2.putText(img, label, (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    save_box_image(img, top_left, bottom_right, label + ".jpg")


# 讀取與處理圖像
img = cv2.imread(img_path)
if img is None:
    print("無法加載圖像，請檢查路徑。")
else:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if gray.dtype == np.uint8:
        faces = detector(gray)
        for face_id, face in enumerate(faces):
            landmarks = predictor(gray, face)

            # 額頭分區
            outline_points, boxes, centers = detect_forehead(landmarks)
            for i, (box, pos) in enumerate(zip(boxes, centers)):
                cv2.rectangle(img, box[0], box[1], (0, 0, 255), 1)
                cv2.putText(img, f"forehead_{i+1}", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                save_box_image(img, box[0], box[1], f"forehead_{i+1}.jpg")

            # 臉部其他區塊標記與儲存
            draw_rect(img, 40, 4, (0, 255, 0), "")
            draw_rect(img, 47, 12, (0, 255, 0), "")
            draw_rect(img, 32, 52, (0, 255, 255), "")
            draw_rect(img, 39, 32, (255, 0, 255), "")
            draw_rect(img, 42, 34, (255, 0, 255), "")
            draw_square_on_point(img, 30, (0, 0, 0), "")
            draw_square_on_point(img, 27, (0, 0, 0), "")

        cv2.imshow("Combined Facial Marking", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("灰度圖類型錯誤")
