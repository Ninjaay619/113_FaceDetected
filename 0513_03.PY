import dlib
import cv2
import numpy as np
import os

# ---------- 資料夾檢查與清除 ---------- #
folders_to_clear = ["output", "processed"]
for folder in folders_to_clear:
    if os.path.exists(folder):
        for f in os.listdir(folder):
            file_path = os.path.join(folder, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(folder)

# ---------- 臉部偵測與擷取 ---------- #
img_path = 'Photo_human_2.jpg'
predictor_path = "C:/Users/jay/Desktop/Python/0425/shape_predictor_68_face_landmarks.dat"

output_dir = "output"
processed_dir = "processed"

# 初始化 dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

def detect_forehead(landmarks):
    eyebrow_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 27)]
    min_y = min([p[1] for p in eyebrow_points])
    left_x = landmarks.part(17).x
    right_x = landmarks.part(26).x
    forehead_height = int((right_x - left_x) * 0.4)
    hairline_y = min_y - forehead_height

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
    return boxes, centers

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

# 處理圖像
img = cv2.imread(img_path)
if img is None:
    print("無法加載圖像，請檢查路徑。")
else:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face_id, face in enumerate(faces):
        landmarks = predictor(gray, face)

        # 額頭分區
        forehead_boxes, centers = detect_forehead(landmarks)
        for i, (box, pos) in enumerate(zip(forehead_boxes, centers)):
            cv2.rectangle(img, box[0], box[1], (0, 0, 255), 1)
            cv2.putText(img, f" ", pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            save_box_image(img, box[0], box[1], f"forehead_{i+1}.jpg")

        # 額頭以外其他臉部框框
        draw_rect(img, 40, 4, (0, 0, 255), "")     # 右臉頰
        draw_rect(img, 47, 12, (0, 0, 255), "")    # 左臉頰
        draw_rect(img, 32, 52, (0, 255, 255), "")  # 人中
        draw_rect(img, 39, 32, (255, 0, 255), "")  # 右鼻翼
        draw_rect(img, 42, 34, (255, 0, 255), "")  # 左鼻翼
        draw_square_on_point(img, 30, (0, 0, 0), "")  # 鼻頭(脾)
        draw_square_on_point(img, 27, (0, 0, 0), "")  # 鼻根(肺)

    cv2.imshow("Combined Facial Marking", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ---------- 圖像處理與格子分析 ---------- #
imported = False
try:
    import cv2
    import numpy as np
    import os
    imported = True
except:
    print("處理模組載入失敗")

if imported:
    input_folder = 'output'
    output_folder = 'processed'
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            file_path = os.path.join(input_folder, filename)
            image = cv2.imread(file_path)
            if image is None:
                print(f"無法讀取圖片: {file_path}")
                continue

            min_size = 400
            h, w = image.shape[:2]
            if h < min_size or w < min_size:
                scale_x = max(min_size / w, 1)
                scale_y = max(min_size / h, 1)
                image = cv2.resize(image, (int(w * scale_x), int(h * scale_y)), interpolation=cv2.INTER_LINEAR)

            scale_percent = 100
            resized_width = int(image.shape[1] * scale_percent / 100)
            resized_height = int(image.shape[0] * scale_percent / 100)
            image = cv2.resize(image, (resized_width, resized_height))

            h, w, _ = image.shape
            grid_rows = 100
            grid_cols = 100
            cell_height = h // grid_rows
            cell_width = w // grid_cols

            output_image = np.zeros_like(image)
            output2_image = image.copy()
            avg_color_entire_image = image.mean(axis=(0, 1)).astype(np.uint8)

            for row in range(grid_rows):
                for col in range(grid_cols):
                    y1 = row * cell_height
                    y2 = min((row + 1) * cell_height, h)
                    x1 = col * cell_width
                    x2 = min((col + 1) * cell_width, w)

                    cell = image[y1:y2, x1:x2]
                    if cell.size == 0:
                        continue

                    avg_color = cell.mean(axis=(0, 1)).astype(np.uint8)
                    brightness = avg_color.mean()

                    output_image[y1:y2, x1:x2] = avg_color

                    if brightness < 122:
                        cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        output2_image[y1:y2, x1:x2] = avg_color_entire_image

            for row in range(1, grid_rows):
                y = row * cell_height
                cv2.line(output_image, (0, y), (w, y), (0, 0, 0), 1)
            for col in range(1, grid_cols):
                x = col * cell_width
                cv2.line(output_image, (x, 0), (x, h), (0, 0, 0), 1)

            base_name = os.path.splitext(filename)[0]
            cv2.imwrite(os.path.join(output_folder, f'original_{base_name}.jpg'), image)
            cv2.imwrite(os.path.join(output_folder, f'grid_{base_name}.jpg'), output_image)
            cv2.imwrite(os.path.join(output_folder, f'dark_blocks_{base_name}.jpg'), output2_image)
            print(f"已儲存處理結果：{filename}")
