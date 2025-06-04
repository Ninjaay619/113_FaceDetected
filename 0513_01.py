import cv2
import numpy as np
import os

# 資料夾設定
input_folder = 'output'
output_folder = 'processed'

# 建立儲存資料夾（如不存在）
os.makedirs(output_folder, exist_ok=True)

# 遍歷資料夾中的所有圖像檔案
for filename in os.listdir(input_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        file_path = os.path.join(input_folder, filename)
        image = cv2.imread(file_path)
        if image is None:
            print(f"無法讀取圖片: {file_path}")
            continue

        # ========== 自動放大圖片 ========== #
        min_size = 400
        h, w = image.shape[:2]
        if h < min_size or w < min_size:
            scale_x = max(min_size / w, 1)
            scale_y = max(min_size / h, 1)
            image = cv2.resize(image, (int(w * scale_x), int(h * scale_y)), interpolation=cv2.INTER_LINEAR)

        # ========== 縮放比例 ========== #
        scale_percent = 100
        resized_width = int(image.shape[1] * scale_percent / 100)
        resized_height = int(image.shape[0] * scale_percent / 100)
        image = cv2.resize(image, (resized_width, resized_height))

        # 更新尺寸
        h, w, _ = image.shape

        # 網格設定
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
                    # 偏黑區塊畫紅框
                    cv2.rectangle(output_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    output2_image[y1:y2, x1:x2] = avg_color_entire_image

        # 畫格線
        for row in range(1, grid_rows):
            y = row * cell_height
            cv2.line(output_image, (0, y), (w, y), (0, 0, 0), 1)
        for col in range(1, grid_cols):
            x = col * cell_width
            cv2.line(output_image, (x, 0), (x, h), (0, 0, 0), 1)

        # 儲存處理結果
        base_name = os.path.splitext(filename)[0]
        cv2.imwrite(os.path.join(output_folder, f'original_{base_name}.jpg'), image)
        cv2.imwrite(os.path.join(output_folder, f'grid_{base_name}.jpg'), output_image)
        cv2.imwrite(os.path.join(output_folder, f'dark_blocks_{base_name}.jpg'), output2_image)
        print(f"已儲存處理結果：{filename}")
