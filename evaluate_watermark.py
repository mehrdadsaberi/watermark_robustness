import cv2
import os
from argparse import ArgumentParser
from tqdm import tqdm
from torchvision import transforms
import json
from time import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from torchvision.utils import save_image
import yaml
import random

from imwatermark import WatermarkEncoder, WatermarkDecoder
from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
from diffusers import StableDiffusionImg2ImgPipeline

from utils.utils import CustomImageFolder, InversableStableDiffusionPipeline, \
    GuidedDiffusion, dict2namespace, ImageCaptioner

class CaptionBasedEdit():
    def __init__(self, img_size=256, save_imgs=False):
        self.image_captioner = ImageCaptioner(image_size=img_size)
        model_id = "timbrooks/instruct-pix2pix"
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
        self.pipe.to("cuda")
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.save_imgs = save_imgs
        
    def __call__(self, img):
        caption = self.image_captioner(img)
        edit_prompt = input(caption + ": ")
        edited_img = self.pipe(edit_prompt, image=img).images[0] # image_guidance_scale=1.5, guidance_scale=7
        if self.save_imgs:
            if not os.path.exists(f'capedit_images/{caption}'):
                os.makedirs(f'capedit_images/{caption}')
            img.save(f'capedit_images/{caption}/org.png')
            edited_img.save(f'capedit_images/{caption}/{edit_prompt}.png')
        return edited_img
    
    def __repr__(self):
        return self.__class__.__name__ + '(prompt={})'.format(self.prompt)


class EditImage():
    def __init__(self, prompt="make it snow", img_guide_scale=1.5, save_imgs=False):
        self.prompt = prompt
        model_id = "timbrooks/instruct-pix2pix"
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
        self.pipe.to("cuda")
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.save_imgs = save_imgs
        self.img_guide_scale = img_guide_scale
        
    def __call__(self, img):
        edited_img = self.pipe(self.prompt, image=img, image_guidance_scale=self.img_guide_scale).images[0] # image_guidance_scale=1.5, guidance_scale=7
        if self.save_imgs:
            save_dir = f'edit_images/{self.prompt}_{self.img_guide_scale}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            rnd_id = random.randint(10**9, 10**10)
            img.save(os.path.join(save_dir, f'{rnd_id}_org.png'))
            edited_img.save(os.path.join(save_dir, f'{rnd_id}_edited.png'))
        return edited_img
    
    def __repr__(self):
        return self.__class__.__name__ + '(prompt={})'.format(self.prompt)

class AddGaussianNoise():
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={}, std={})'.format(self.mean, self.std)

class DiffPure():
    def __init__(self, steps=0.4, save_imgs=False, fname="base"):
        with open('DiffPure/configs/imagenet.yml', 'r') as f:
            config = yaml.safe_load(f)
        self.config = dict2namespace(config)
        self.runner = GuidedDiffusion(self.config, t = int(steps * int(self.config.model.timestep_respacing)), model_dir = 'DiffPure/pretrained/guided_diffusion')
        self.steps = steps
        self.save_imgs = save_imgs
        self.cnt = 0
        self.fname = fname

        if self.save_imgs:
            save_dir = f'./diffpure_images/{self.fname}/{self.steps}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

    def __call__(self, img):
        img_pured, img_noisy = self.runner.image_editing_sample((img.unsqueeze(0) - 0.5) * 2)
        img_noisy = (img_noisy.squeeze(0).to(img.dtype).to("cpu") + 1) / 2
        img_pured = (img_pured.squeeze(0).to(img.dtype).to("cpu") + 1) / 2
        if self.save_imgs:
            save_dir = f'./diffpure_images/{self.fname}/{self.steps}'
            save_image(img, os.path.join(save_dir, f'{self.cnt}.png'))
            save_image(img_noisy, os.path.join(save_dir, f'{self.cnt}_noisy.png'))
            save_image(img_pured, os.path.join(save_dir, f'{self.cnt}_pured.png'))
            self.cnt += 1
        return img_pured
    
    def __repr__(self):
        return self.__class__.__name__ + '(steps={})'.format(self.steps)

class ImageRephrase():
    def __init__(self, img_size=256, strength=0.5, num_passes=1, save_imgs=False, fname="base"):
        self.image_captioner = ImageCaptioner(image_size=img_size)
        model_id = "runwayml/stable-diffusion-v1-5"
        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
        self.pipe = self.pipe.to("cuda")
        self.strength = strength
        self.num_passes = num_passes
        self.img_size = img_size
        self.save_imgs = save_imgs
        self.cnt = 0
        self.fname = fname

        if self.save_imgs:
            save_dir = f'diffpure_latent_images/{self.fname}/{self.strength}'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

    def __call__(self, img):
        caption = self.image_captioner(img)
        save_dir = f'diffpure_latent_images/{self.fname}/{self.strength}'
        if self.save_imgs:
            img.save(os.path.join(save_dir, f'{self.cnt}.png'))
        for i in range(self.num_passes):
            img = self.pipe(prompt=caption, image=img.resize((768, 512)), strength=self.strength, guidance_scale=7.5).images[0].resize((self.img_size, self.img_size))
            if self.save_imgs:
                img.save(os.path.join(save_dir, f'{self.cnt}_pured_{i}.png'))
        self.cnt += 1
        return img

    def __repr__(self):
        return self.__class__.__name__

def get_transforms(aug_str, fname="base", save_images=False):
    aug = aug_str.split(',')[0]
    params = aug_str.split(',')[1:]
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.ToTensor(),
        ]
    )
    org_transform = transform

    if aug == 'no_aug':
        pass
    elif aug == 'rotation':
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomRotation([-int(90 * float(params[0])), int(90 * float(params[0]))]),
                transforms.ToTensor(),
            ]
        )
    elif aug == 'crop':
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(256 - int(128 * float(params[0]))),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )
    elif aug == 'flip':
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomHorizontalFlip(1.0),
                transforms.ToTensor(),
            ]
        )
    elif aug == 'stretch':
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop((256 - int(128 * float(params[0])), 256)),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )
    elif aug == 'blur': # remove
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.GaussianBlur(5),
                transforms.ToTensor(),
            ]
        )
    elif aug == 'combo':
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop((256 - int(128 * float(params[0])), 256)),
                transforms.RandomHorizontalFlip(1.0),
                transforms.RandomRotation([-int(90 * float(params[0])), int(90 * float(params[0]))]),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )
    elif aug == 'edit':
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                EditImage(params[0], float(params[1]), save_imgs=False),
                transforms.ToTensor(),
            ]
        )
    elif aug == 'edit stretch':
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                EditImage(params[0], float(params[1]), save_imgs=False),
                transforms.CenterCrop((256 - int(128 * float(params[2])), 256)),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )
    elif aug == 'edit combo':
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                EditImage("make it snow"),
                transforms.CenterCrop((128, 256)),
                transforms.RandomHorizontalFlip(1.0),
                transforms.RandomRotation([-90, 90]),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ]
        )
    elif aug == 'gaussian':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            AddGaussianNoise(0., float(params[0])) # 0.05
        ])
    elif aug == 'diffpure':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor(),
            DiffPure(steps=float(params[0]), save_imgs=save_images, fname=fname)
        ])
    elif aug == 'diffpure_latent':
        transform = transforms.Compose([
            transforms.Resize(256),
            ImageRephrase(img_size=256, strength=float(params[0]), num_passes=int(params[1]), save_imgs=save_images, fname=fname),
            transforms.ToTensor(),
        ])
    elif aug == 'capedit':
        transform = transforms.Compose([
            transforms.Resize(256),
            CaptionBasedEdit(img_size=256, save_imgs=False),
            transforms.ToTensor(),
        ])
    else:
        print(f"Augmentation {aug} not implemented!!!")
        return None, None
    
    return transform, org_transform


def main():
    parser = ArgumentParser()
    parser.add_argument("--wm-method", default="dwtDct", type=str,
        choices=["dwtDct", "dwtDctSvd", "rivaGan", "treeRing", 'watermarkDM', 'stegaStamp', 'MBRS'])
    parser.add_argument("--attack", default="diffpure", type=str,
        choices=["no_aug", "diffpure", "diffpure_latent", 'common_augs', 'image_edit'])
    parser.add_argument("--data-dir", default="images/imagenet/dwtDct", type=str)
    parser.add_argument("--org-data-dir", default="images/imagenet/org", type=str)
    parser.add_argument("--dataset", default="imagenet", type=str)
    parser.add_argument("--out-fname", default="out", type=str)
    parser.add_argument("--save-images", action='store_true')

    args = parser.parse_args()


    assert os.path.exists(args.data_dir)
    
    if not os.path.exists('results'):
        os.mkdir('results')

    print(f"||  Method: {args.wm_method}")

    if args.attack == 'no_aug':
        aug_list = ['no_aug']
    elif args.attack == 'diffpure':
        aug_list = ['no_aug', 'diffpure,0.1', 'diffpure,0.2', 'diffpure,0.3']
    elif args.attack == 'diffpure_latent':
        aug_list = ['no_aug', 'diffpure_latent,0.2,1', 'diffpure_latent,0.3,1', 'diffpure_latent,0.4,1']
    elif args.attack == 'common_augs':
        aug_list = ['no_aug', 'gaussian,0.02', 'gaussian,0.04', 'gaussian,0.08', 'gaussian,0.16',
            'crop,0.25', 'crop,0.5', 'crop,0.75', 'crop,1.0',
            'rotation,0.25', 'rotation,0.5', 'rotation,0.75', 'rotation,1.0', 'flip',
            'stretch,0.25', 'stretch,0.5', 'stretch,0.75', 'stretch,1.0',
            'combo,0.25', 'combo,0.5', 'combo,0.75', 'combo,1.0']
    elif args.attack == 'image_edit':
        aug_list = ['edit,make it snow,1.3', 'edit,make it at night,1.3', 'edit,change background to desert,1.5',
                'edit,convert into an oil painting,1.2', 'edit,add a tennis ball to the image,2.0']
    else:
        raise ModuleNotFoundError(f"Attack {args.attack} not implemented")


    bit_ths = np.arange(0.0, 1.001, 0.1)

    th_results, labels, preds = {}, {}, {}
    fpr_th = 0.1

    for aug in aug_list:
        print("Augmentation:", aug)

        transform, org_transform = get_transforms(aug, args.wm_method, args.save_images)
        if transform == None:
            continue
        
        dataset = CustomImageFolder(args.data_dir, transform=transform)
        data_cnt = len(dataset)
        org_dataset = CustomImageFolder(args.org_data_dir, transform=org_transform, data_cnt=data_cnt)

        th_results[aug] = {}
        labels[aug], preds[aug] = [], []
       
        
        if args.wm_method in ["dwtDct", "dwtDctSvd", "rivaGan"]:
            wm_key = '0100010001000010111010111111110011101000001111101101010110000001'
            wm_key = [x == '1' for x in wm_key]
            
            if args.wm_method == "rivaGan":
                wm_key = wm_key[:32]
            
            decoder = WatermarkDecoder('bits', len(wm_key))
            if args.wm_method == "rivaGan":
                WatermarkDecoder.loadModel()

            for i, (img_tensor, label) in tqdm(enumerate(dataset)):
                img_np = img_tensor.numpy().transpose(1, 2, 0) * 255
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                # img_np = cv2.imread(dataset.filenames[i])
                watermark = list(decoder.decode(img_np, args.wm_method))
                labels[aug].append(1)
                try:
                    if len(watermark) != len(wm_key):
                        continue
                    preds[aug].append(sum([wm_key[i] == watermark[i] for i in range(len(wm_key))]) / len(wm_key))
                except:
                    preds[aug].append(0.)
                
            for i, (img_tensor, label) in tqdm(enumerate(org_dataset)):
                img_np = img_tensor.numpy().transpose(1, 2, 0) * 255
                img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
                watermark = list(decoder.decode(img_np, args.wm_method))
                labels[aug].append(0)
                try:
                    if len(watermark) != len(wm_key):
                        continue
                    preds[aug].append(sum([wm_key[i] == watermark[i] for i in range(len(wm_key))]) / len(wm_key))
                except:
                    preds[aug].append(1.)

        elif args.wm_method == "treeRing":
            from utils.tree_ring import decode_tree_ring
            preds_org = decode_tree_ring(org_dataset, args.dataset)
            preds_wm = decode_tree_ring(dataset, args.dataset)
            preds[aug] = preds_wm + preds_org
            labels[aug] = [1 for i in range(len(preds_wm))] + [0 for i in range(len(preds_org))]
           
        elif args.wm_method == "watermarkDM":
            from utils.watermark_dm import decode_watermark_dm
            preds_wm = decode_watermark_dm(dataset, args.dataset)
            preds_org = decode_watermark_dm(org_dataset, args.dataset)
            preds[aug] = preds_wm + preds_org
            labels[aug] = [1 for i in range(len(preds_wm))] + [0 for i in range(len(preds_org))]
 
        elif args.wm_method == "MBRS":
            from utils.MBRS import MBRS
            try:
                _MBRS
            except NameError:
                _MBRS = MBRS()
            preds_wm = _MBRS.decode_MBRS(dataset, args.dataset)
            preds_org = _MBRS.decode_MBRS(org_dataset, args.dataset)
            preds[aug] = preds_wm + preds_org
            labels[aug] = [1 for i in range(len(preds_wm))] + [0 for i in range(len(preds_org))]
 
        elif args.wm_method == "stegaStamp":
            from utils.stega_stamp import decode_stega_stamp
            preds_wm = decode_stega_stamp(dataset, args.dataset)
            preds_org = decode_stega_stamp(org_dataset, args.dataset)
            preds[aug] = preds_wm + preds_org
            labels[aug] = [1 for i in range(len(preds_wm))] + [0 for i in range(len(preds_org))] 
            
        else:
            raise ModuleNotFoundError(f"Method {args.wm_method} not implemented")

        for bit_th in bit_ths:
            eval_dict = {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0}
            for i in range(len(labels[aug])):
                pred = preds[aug][i]
                if labels[aug][i] == 1:
                    if pred >= bit_th:
                        eval_dict['tp'] += 1
                    else:
                        eval_dict['fn'] += 1
                else:
                    if pred < bit_th:
                        eval_dict['tn'] += 1
                    else:
                        eval_dict['fp'] += 1
            th_results[aug][bit_th] = eval_dict

    # with open(f'th_results/{args.wm_method}_{args.dataset}_{args.out_fname}.json', 'w') as f:
    #     json.dump(th_results, f, indent=4)


    #### AUROC PLOT
    if not os.path.exists('results/plots'):
        os.mkdir('results/plots')

    results = {}

    sns.set_theme(style="darkgrid")
    plt.rcParams['text.usetex'] = True
    plt.figure(figsize=(16,8))
    for aug in labels.keys():
        results[aug] = {}
        fpr, tpr, thresh = metrics.roc_curve(labels[aug], preds[aug])
        results[aug]["auc"] = metrics.roc_auc_score(labels[aug], preds[aug])
        plt.plot(fpr,tpr,label="{}, auc={:.2f}".format(aug, results[aug]["auc"]))

        for i in range(len(fpr)):
            if fpr[i] > fpr_th:
                results[aug]["th_tpr"] = tpr[max(0, i - 1)]
                break
        results[aug]["labels"] = labels[aug]
        results[aug]["preds"] = preds[aug]
        results[aug]["fpr"] = list(fpr)
        results[aug]["tpr"] = list(tpr)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.savefig(f'results/plots/{args.wm_method}_{args.dataset}_{args.out_fname}_roc.png')
    plt.clf()

    with open(f'results/{args.wm_method}_{args.dataset}_{args.out_fname}.json', 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
