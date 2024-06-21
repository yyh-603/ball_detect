import json
import os
import glob

def labelme_to_coco(json_dir, output_file):
    data = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    category_set = set()
    annotation_id = 1
    image_id = 1

    for json_file in glob.glob(os.path.join(json_dir, '*.json')):
        with open(json_file) as f:
            labelme_data = json.load(f)

        image_info = {
            "id": image_id,
            "file_name": labelme_data['imagePath'],
            "width": labelme_data['imageWidth'],
            "height": labelme_data['imageHeight']
        }
        data["images"].append(image_info)

        for shape in labelme_data['shapes']:
            points = shape['points']
            segmentation = [list(sum(points, []))]

            min_x = min([p[0] for p in points])
            max_x = max([p[0] for p in points])
            min_y = min([p[1] for p in points])
            max_y = max([p[1] for p in points])
            bbox = [min_x, min_y, max_x - min_x, max_y - min_y]

            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": shape['label'],
                "segmentation": segmentation,
                "area": (max_x - min_x) * (max_y - min_y),
                "bbox": bbox,
                "iscrowd": 0
            }
            data["annotations"].append(annotation)
            annotation_id += 1

            category_set.add(shape['label'])

        image_id += 1

    for category_id, category in enumerate(category_set):
        data["categories"].append({
            "id": category_id,
            "name": category
        })

    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

# 使用示例
labelme_to_coco('./label', 'output_coco.json')
