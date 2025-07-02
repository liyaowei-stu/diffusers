import torch
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

pipe = FluxKontextPipeline.from_pretrained("/group/40033/public_models/FLUX.1-Kontext-dev/", torch_dtype=torch.bfloat16)
pipe.to("cuda")

input_image = load_image("ic_custom/20250630_223253_new_image_4.png")

import ipdb; ipdb.set_trace()

image = pipe(
  image=input_image,
  prompt="Fill and enhance the details of the corrupted rendering image, make it more detailed and complete, then imagine: Add a pair of sunglasses for the soldier, and is intricately laced into a whimsical fairy tale world. keep the pose unchanged and ensure the harmony of the entire scene.",
  guidance_scale=3.5
).images[0]

image.save("ic_custom/output_soldier.png")

