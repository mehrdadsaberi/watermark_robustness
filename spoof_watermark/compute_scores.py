from utils import CustomImageFolder
from torchvision import transforms
from argparse import ArgumentParser
from six.moves import cPickle as pkl
import numpy as np

parser = ArgumentParser()
parser.add_argument("--inp", type=str)
parser.add_argument("--out", type=str)
parser.add_argument("--method", default="treeRing", type=str)
parser.add_argument("--data-cnt", type=int)
args = parser.parse_args()

transform = []

transform.append(transforms.Compose(
            [
                transforms.Resize(256),
                transforms.ToTensor(),
            ]
        ))

transform.append(transforms.Compose(
            [
                transforms.RandomCrop(200),
                transforms.Resize(256),
                transforms.ToTensor(),
            ]
        ))

transform.append(transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomRotation([-30, 30]),
                transforms.ToTensor(),
            ]
        ))


scores = []

for t in transform:
    dataset = CustomImageFolder(args.inp, transform=t, data_cnt=args.data_cnt)
    
    if args.method == "treeRing":
        from tree_ring import decode_tree_ring
        score = decode_tree_ring(dataset, True)
        scores.extend(list(score))
        
    elif args.method == "watermarkDM":
        from watermark_dm import decode_watermark_dm
        score = decode_watermark_dm(dataset, "imagenet")
        scores.extend(list(score))
        
    elif args.method in ["dwtDct", "dwtDctSvd", "rivaGan"]:
        
        from imwatermark import WatermarkDecoder
        import cv2
        from tqdm import tqdm
        
        wm_key = '0100010001000010111010111111110011101000001111101101010110000001'
        wm_key = [x == '1' for x in wm_key]
        if args.method == "rivaGan":
                wm_key = wm_key[:32]
                
        decoder = WatermarkDecoder('bits', len(wm_key))
        if args.method == "rivaGan":
            WatermarkDecoder.loadModel()
            
        for i, (img_tensor, label) in tqdm(enumerate(dataset)):
            img_np = img_tensor.numpy().transpose(1, 2, 0) * 255
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            watermark = list(decoder.decode(img_np, args.method))
            try:
                if len(watermark) != len(wm_key):
                    continue
                scores.append(sum([wm_key[i] == watermark[i] for i in range(len(wm_key))]) / len(wm_key))
            except:
                scores.append(0)

with open(args.out, 'wb') as f:
    pkl.dump(np.stack(scores), f)