from ultralytics import YOLO

model = YOLO("yolov8x-seg.pt")  # initialize model
results = model("test.jpg")  # perform inference
results[0].show()  # display results for the first image
results[0].save(filename="result.jpg")