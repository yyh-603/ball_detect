import json
import cv2
import numpy as np
from PIL import Image, ImageDraw

# 讀取 COCO 標註文件
def load_coco_annotations(json_file):
    with open(json_file, 'r') as f:
        coco_data = json.load(f)
    return coco_data

# 畫分割掩碼
def draw_segmentation(image, segmentation, color):
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    for seg in segmentation:
        points = [(seg[i], seg[i+1]) for i in range(0, len(seg), 2)]
        draw.polygon(points, outline=color, fill=color)
    return np.array(img)

# 讀取圖片並畫出標註
def visualize_annotations(coco_data, images_dir, output_dir):
    for image_info in coco_data['images']:
        image_id = image_info['id']
        image_path = f"{images_dir}/{image_info['file_name']}"
        
        image = cv2.imread(image_path)
        if image is None:
            continue
        
        annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] == image_id]
        # print(annotations)
        for ann in annotations:
            segmentation = ann['segmentation']
            category_name = ann['category_id']
            
            # 隨機顏色
            color = tuple(np.random.randint(0, 255, 3).tolist())
            
            # 畫分割掩碼
            image = draw_segmentation(image, segmentation, color)
            
            # 畫邊界框
            bbox = ann['bbox']
            x, y, w, h = map(int, bbox)
            cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
            
            # 顯示類別標籤
            cv2.putText(image, category_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 保存圖片
        output_path = f"{output_dir}/{image_info['file_name']}"
        cv2.imwrite(output_path, image)

# 設定路徑
json_file = './output_coco.json'
images_dir = './'
output_dir = './output'

# 確保輸出目錄存在
import os
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 讀取標註並畫出
coco_data = load_coco_annotations(json_file)
visualize_annotations(coco_data, images_dir, output_dir)

print("標註圖片已保存至", output_dir)
