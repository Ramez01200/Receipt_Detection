import ultralytics
from roboflow import Roboflow
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
# rf = Roboflow(api_key="WT7LJlCJAbqS3XRxsfQe")
# project = rf.workspace("ramez-cicie").project("receipt-detection-igzta")
# version = project.version(3)
# dataset = version.download("yolov8")





#yolo train model=yolov8n.pt data="D:/Yolov8/receipt-detection-3/data.yaml" epochs=1 imgsz=640 single_cls=True

#yolo task=detect mode=val model="D:/Yolov8/runs/detect/train6/weights/best.pt" data = "D:/Yolov8/receipt-detection-3/data.yaml"

#yolo task=detect mode=predict model="D:/Yolov8/runs/detect/train6/weights/best.pt" data="D:/Yolov8/receipt-detection-3/data.yaml" source="C:/Users/Ramez/Downloads/WhatsApp Image 2024-04-01 at 19.43.32_2d628c6c.jpg"

model = YOLO("D:/Yolov8/runs/detect/train6/weights/best.pt")

image_path = "C:/Users/Ramez/Downloads/WhatsApp Image 2024-04-01 at 19.43.32_2d628c6c.jpg"

image = cv2.imread(image_path)


results = model.predict(image_path)


for result in results:
    boxes = result.boxes  # Bounding box outputs
    for box in boxes:
        bbox_coordinates = box.xyxy[0]
        print(bbox_coordinates)
  


    masks = result.masks  # Segmentation mask outputs
    keypoints = result.keypoints  # Pose keypoints
    probs = result.probs  # Classification probabilities
    result.show() 


x1, y1, x2, y2 = bbox_coordinates

# Convert the coordinates to integers
x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

# Crop the image
cropped_image = image[y1:y2, x1:x2]


# Display the cropped image
cv2.imshow("Cropped Receipt", cropped_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

