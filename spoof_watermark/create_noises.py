import numpy as np
from PIL import Image 
import os 

dir = "images/imagenet/noise"

if not os.path.exists(dir):
    os.makedirs(dir)

noise = np.random.randn(512, 512, 3)
noise = noise - noise.min()
noise = noise / noise.max() * 255
pil_img = Image.fromarray(np.uint8(noise)).convert('RGB')
pil_img.save(os.path.join(dir, 'noise1.png'))

noise = np.random.randn(512, 512, 3)
noise = noise - noise.min()
noise = noise / noise.max() * 192
pil_img = Image.fromarray(np.uint8(noise)).convert('RGB')
pil_img.save(os.path.join(dir, 'noise2.png'))

noise = np.random.randn(512, 512, 3)
noise = noise - noise.min()
noise = noise / noise.max() * 128
pil_img = Image.fromarray(np.uint8(noise)).convert('RGB')
pil_img.save(os.path.join(dir, 'noise3.png'))