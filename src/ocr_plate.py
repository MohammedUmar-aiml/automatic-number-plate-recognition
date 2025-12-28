from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
from PIL import Image

class PlateOCR:
    def __init__(self, model_name="microsoft/trocr-small-printed"):
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)

    def recognize(self, plate_image):
        """
        plate_image: OpenCV image (BGR)
        Returns recognized text
        """
        image = Image.fromarray(plate_image[..., ::-1])  # BGR → RGB

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)

        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text
