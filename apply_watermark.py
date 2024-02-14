import cv2
import os
from argparse import ArgumentParser
from tqdm import tqdm
from torchvision import transforms
from PIL import Image
from time import time

from imwatermark import WatermarkEncoder, WatermarkDecoder
from utils.utils import CustomImageFolder


def main():
    parser = ArgumentParser()
    parser.add_argument("--wm-method", default="dwtDct", type=str,
        choices=["dwtDct", "dwtDctSvd", "rivaGan", "treeRing", "watermarkDM", "stegaStamp", "MBRS"])
    parser.add_argument("--dataset", default="imagenet", type=str)
    parser.add_argument("--data-dir", default="images/imagenet/org", type=str)
    parser.add_argument("--data-cnt", default=-1, type=int)
    parser.add_argument("--out-dir", default="images/imagenet/dwtDct", type=str)
    args = parser.parse_args()

    assert os.path.exists(args.data_dir)
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    if args.dataset == 'imagenet':
        transform = transforms.Compose(
            [
                transforms.Resize(256),
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
            wm_img = encoder.encode(img_np, args.wm_method)
            cv2.imwrite(os.path.join(args.out_dir, dataset.img_ids[i].split('.')[0] + '.png'), wm_img)

    elif args.wm_method == "treeRing":
        from utils.tree_ring import run_tree_ring
        run_tree_ring(dataset, dataset_name = args.dataset, out_dir = args.out_dir)

    elif args.wm_method == "watermarkDM":
        from utils.watermark_dm import run_watermark_dm
        run_watermark_dm(dataset, dataset_name = args.dataset, out_dir = args.out_dir)

    elif args.wm_method == "stegaStamp":
        from utils.stega_stamp import run_stega_stamp
        run_stega_stamp(dataset, dataset_name = args.dataset, out_dir = args.out_dir)
    
    elif args.wm_method == "MBRS":
        from utils.MBRS import MBRS
        _MBRS = MBRS()
        _MBRS.run_MBRS(dataset, dataset_name = args.dataset, out_dir = args.out_dir)

    else:
        raise ModuleNotFoundError(f"Method {args.wm_method} not implemented")


if __name__ == "__main__":
    main()
