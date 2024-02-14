import cv2
import os
from argparse import ArgumentParser
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from time import time
from torchvision.datasets import ImageNet, ImageFolder

from utils.utils import CustomImageFolder
from torchvision.utils import save_image


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset", default="imagenet", type=str)
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--data-cnt", default=100, type=int)
    parser.add_argument("--out-dir", default="images/imagenet/org", type=str)
    args = parser.parse_args()

    assert os.path.exists(args.data_dir)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.dataset == 'imagenet':
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Resize(256),
                transforms.ToTensor(),
            ]
        )

    else:
        raise ModuleNotFoundError(f"Dataset {args.dataset} not implemented")

    s = time()
    print(f"Loading image folder {args.data_dir} ...")
    dataset = CustomImageFolder(args.data_dir, transform=transform, data_cnt=args.data_cnt, shuffle=True)
    for i, (img_tensor, label) in tqdm(enumerate(dataset)):
        save_image(img_tensor, os.path.join(args.out_dir, f'{dataset.img_ids[i]}.png'))
    print(f"Finished. Loading took {time() - s:.2f}s")


if __name__ == "__main__":
    main()
