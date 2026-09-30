from transformers import BlipProcessor, BlipForConditionalGeneration
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    device_map="auto"
).to(device)

def caption_image(raw_image):
    inputs = processor(raw_image, return_tensors="pt").to(device)
    out = model.generate(**inputs)

    caption = processor.decode(out, skip_special_tokens=True)
    return caption