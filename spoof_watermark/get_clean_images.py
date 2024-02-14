from utils import CustomImageFolder
from torchvision import transforms
from argparse import ArgumentParser
from tqdm import tqdm

import os
import sys
sys.path.append(os.path.join(os.getcwd(), "TreeRingWatermark"))
from TreeRingWatermark.optim_utils import *
from TreeRingWatermark.io_utils import *

parser = ArgumentParser()
parser.add_argument("--inp", type=str)
parser.add_argument("--out", default="images/imagenet/clean/", type=str)
parser.add_argument("--data-cnt", default=50, type=int)
args = parser.parse_args()

transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.Resize(256),
        transforms.ToTensor(),
    ]
)


if not os.path.exists(args.out):
    os.makedirs(args.out)
        
dataset = CustomImageFolder(args.inp, transform=transform, data_cnt=args.data_cnt)

for i in tqdm(range(len(dataset))):
    pil_img = Image.open(dataset.filenames[i]).convert('RGB')
    img_no_w = transform_img(pil_img)
    filename = dataset.img_ids[i]
    filename = filename.split('.')[0] + ".png"
    img_no_w = ((img_no_w + 1) / 2)
    transforms.ToPILImage()(img_no_w).save(os.path.join(args.out, filename))