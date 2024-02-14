import cv2
import os
from argparse import ArgumentParser
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from time import time
from torchvision.datasets import ImageNet, ImageFolder

from imwatermark import WatermarkEncoder, WatermarkDecoder
from utils import CustomImageFolder
from watermark_dm import run_watermark_dm
from tree_ring import run_tree_ring


def main():
    parser = ArgumentParser()
    parser.add_argument("--wm-method", default="dwtDct", type=str,
        choices=["dwtDct", "dwtDctSvd", "rivaGan", "treeRing", 'watermarkDM'])
    parser.add_argument("--dataset", default="imagenet", type=str)
    parser.add_argument("--data-dir", type=str)
    parser.add_argument("--data-cnt", default=-1, type=int)
    parser.add_argument("--out-dir", default="images", type=str)
    args = parser.parse_args()

    out_dir = os.path.join(args.out_dir, args.dataset, args.wm_method)
    assert os.path.exists(args.data_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if args.dataset in ['imagenet', 'noise']:
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Resize(256),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

    elif args.dataset == 'cifar10':  # only works for watermarkDM
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    else:
        raise ModuleNotFoundError(f"Dataset {args.dataset} not implemented")

    s = time()
    print(f"Loading image folder {args.data_dir} ...")
    dataset = CustomImageFolder(args.data_dir, transform=transform, data_cnt=args.data_cnt)
    print(f"Finished. Loading took {time() - s:.2f}s")
    
    print(f"Watermarking images with {args.wm_method} method...")

    if args.wm_method in ["dwtDct", "dwtDctSvd", "rivaGan"]:
        wm_key = '0100010001000010111010111111110011101000001111101101010110000001'
        encoder = WatermarkEncoder()
        if args.wm_method == "rivaGan":
            wm_key = wm_key[:32]
            WatermarkEncoder.loadModel()
        encoder.set_watermark('bits', wm_key)

        
        for i, (img_tensor, label) in tqdm(enumerate(dataset)):
            img_np = img_tensor.numpy().transpose(1, 2, 0) * 255
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            # img_np = cv2.imread(dataset.filenames[i])
            # img_np = cv2.resize(img_np, (256, 256), interpolation=cv2.INTER_AREA)
            wm_img = encoder.encode(img_np, args.wm_method)
            cv2.imwrite(os.path.join(out_dir, dataset.img_ids[i].split('.')[0] + '.png'), wm_img)

        # for img_fn in tqdm(os.listdir(org_img_dir)):
        #     if not img_fn.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
        #         continue
        #     org_img = cv2.imread(os.path.join(org_img_dir, img_fn))
        #     wm_img = encoder.encode(org_img, args.wm_method)
        #     cv2.imwrite(os.path.join(wm_img_dir, img_fn), wm_img)

    elif args.wm_method == "treeRing":
        run_tree_ring(dataset, dataset_name = args.dataset, out_dir = out_dir)

    elif args.wm_method == "watermarkDM":
        run_watermark_dm(dataset, dataset_name = args.dataset, out_dir = out_dir)

    else:
        raise ModuleNotFoundError(f"Method {args.wm_method} not implemented")


if __name__ == "__main__":
    main()
