import os

import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

flux_kontext_path = os.getenv("FLUX_KONTEXT") if os.getenv("FLUX_KONTEXT") else "black-forest-labs/FLUX.1-Kontext-dev"

import ipdb; ipdb.set_trace()

pipe = FluxKontextPipeline.from_pretrained(flux_kontext_path, torch_dtype=torch.bfloat16)
pipe.to("cuda")

input_image = load_image("mv_custom/assets/20250630_223253_new_image_4.png")


image = pipe(
  image=input_image,
  prompt="Fill and enhance the details of the corrupted rendering image, make it more detailed and complete, then imagine: Add a pair of sunglasses for the soldier, and is intricately laced into a whimsical fairy tale world. keep the pose unchanged and ensure the harmony of the entire scene.",
  guidance_scale=3.5
).images[0]

image.save("mv_custom/output_soldier.png")

