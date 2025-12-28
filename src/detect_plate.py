import cv2
from ultralytics import YOLO

class PlateDetector:
    def __init__(self, model_path: str):
        """
        model_path: path to YOLOv8 plate detection model (.pt)
        """
        self.model = YOLO(model_path)

    def detect(self, image):
        """
        Detect license plates in an image.
        Returns list of cropped plate images.
        """
        results = self.model(image)
        plates = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box)
                plate = image[y1:y2, x1:x2]
                plates.append(plate)

        return plates
