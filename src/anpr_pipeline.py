import cv2
from detect_plate import PlateDetector
from ocr_plate import PlateOCR

def run_anpr(image_path, yolo_model_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found")

    detector = PlateDetector(yolo_model_path)
    ocr = PlateOCR()

    plates = detector.detect(image)
    results = []

    for plate in plates:
        text = ocr.recognize(plate)
        results.append(text)

    return results
if __name__ == "__main__":
    image_path = "sample.jpg"
    yolo_model_path = "yolov8_plate.pt"

    outputs = run_anpr(image_path, yolo_model_path)
    print("Detected plates:", outputs)
