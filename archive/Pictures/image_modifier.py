import torch
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from PIL import Image

# Load image
image = Image.open("Female/full-length-portrait-cute-woman-gray-wall.jpg").convert("RGB")

model_id = "timbrooks/instruct-pix2pix"
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.to("cuda")
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
# `image` is an RGB PIL.Image
images = pipe("change her to have short hair", image=image).images
images[0]