from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')

model.train(data='./YOLODataset/dataset.yaml', epochs=10, save_dir="./", batch=1)

# results = model.val()
# print(results)

results = model ("./haha/images/test/test-1.jpg")  # perform inference
results[0].show()  # display results for the first image
results[0].save(filename="./result.jpg")