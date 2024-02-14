import argparse
import wandb
import copy
from tqdm import tqdm
from statistics import mean, stdev
from sklearn import metrics
import torch
from torch.utils.data import Dataset, DataLoader


from torchvision import transforms

import sys
import os
sys.path.append(os.path.join(os.getcwd(), "TreeRingWatermark"))

from TreeRingWatermark.inverse_stable_diffusion import InversableStableDiffusionPipeline
from diffusers import DPMSolverMultistepScheduler
import open_clip
from TreeRingWatermark.optim_utils import *
from TreeRingWatermark.io_utils import *
from utils import get_imagenet_class_name


DIR = "DIRECTORY/TO/CACHE/"

def run_tree_ring(input_dataset=None, dataset_name='imagenet', out_dir='images/imagenet/treeRing'):

    parser = argparse.ArgumentParser(description='diffusion watermark')
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--dataset', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=10, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--with_tracking', action='store_true')
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--test_num_inference_steps', default=None, type=int)
    parser.add_argument('--reference_model', default=None)
    parser.add_argument('--reference_model_pretrain', default=None)
    parser.add_argument('--max_num_log_image', default=100, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)

    # watermark
    parser.add_argument('--w_seed', default=999999, type=int)
    parser.add_argument('--w_channel', default=0, type=int)
    parser.add_argument('--w_pattern', default='rand')
    parser.add_argument('--w_mask_shape', default='circle')
    parser.add_argument('--w_radius', default=10, type=int)
    parser.add_argument('--w_measurement', default='l1_complex')
    parser.add_argument('--w_injection', default='complex')
    parser.add_argument('--w_pattern_const', default=0, type=float)
    
    # for image distortion
    parser.add_argument('--r_degree', default=None, type=float)
    parser.add_argument('--jpeg_ratio', default=None, type=int)
    parser.add_argument('--crop_scale', default=None, type=float)
    parser.add_argument('--crop_ratio', default=None, type=float)
    parser.add_argument('--gaussian_blur_r', default=None, type=int)
    parser.add_argument('--gaussian_std', default=None, type=float)
    parser.add_argument('--brightness_factor', default=None, type=float)
    parser.add_argument('--rand_aug', default=0, type=int)

    
    args = parser.parse_args(['--w_channel', '3', 
                '--w_pattern', 'ring',
                '--reference_model', 'ViT-g-14',
                '--reference_model_pretrain', 'laion2b_s12b_b42k'])

    if args.test_num_inference_steps is None:
        args.test_num_inference_steps = args.num_inference_steps

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    table = None
    if args.with_tracking:
        wandb.init(project='diffusion_watermark', name=args.run_name, tags=['tree_ring_watermark'])
        wandb.config.update(args)
        table = wandb.Table(columns=['gen_no_w', 'no_w_clip_score', 'gen_w', 'w_clip_score', 'prompt', 'no_w_metric', 'w_metric'])
    
    # load diffusion model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler', cache_dir=DIR)
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision='fp16',
        cache_dir=DIR
        )
    pipe = pipe.to(device)

    # reference model
    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(args.reference_model, pretrained=args.reference_model_pretrain, device=device, cache_dir=DIR)
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    # dataset
    if not input_dataset:
        raise ModuleNotFoundError(f"Loading the dataset is not implemented!")
    dataset = input_dataset

    tester_prompt = '' # assume at the detection time, the original prompt is unknown
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    # ground-truth patch
    gt_patch = get_watermarking_pattern(pipe, args, device)

    results = []
    clip_scores = []
    clip_scores_w = []
    no_w_metrics = []
    w_metrics = []

    for i in tqdm(range(len(dataset))):
        seed = i + args.gen_seed
        
        pil_img = Image.open(dataset.filenames[i]).convert('RGB')
        if dataset_name == 'imagenet':
            current_prompt = f'a photo of {get_imagenet_class_name(dataset.img_ids[i])}'
        elif dataset_name == 'noise':
            current_prompt = 'a noisy image'
        else :
            current_prompt = ''
        
        ### generation
        # generation without watermarking
        set_random_seed(seed)
        # init_latents_no_w = pipe.get_random_latents()

        img_no_w = transform_img(pil_img).unsqueeze(0).to(text_embeddings.dtype).to(device)
        image_latents_no_w = pipe.get_image_latents(img_no_w, sample=False)

        init_latents_no_w = pipe.forward_diffusion(
            latents=image_latents_no_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )

        # print(init_latents_no_w.min(), init_latents_no_w.max())
        outputs_no_w = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_no_w,
            )
        orig_image_no_w = outputs_no_w.images[0]
        
        # generation with watermarking
        if init_latents_no_w is None:
            set_random_seed(seed)
            init_latents_w = pipe.get_random_latents()
        else:
            init_latents_w = copy.deepcopy(init_latents_no_w)

        # get watermarking mask
        watermarking_mask = get_watermarking_mask(init_latents_w, args, device)

        # inject watermark
        init_latents_w = inject_watermark(init_latents_w, watermarking_mask, gt_patch, args)

        outputs_w = pipe(
            current_prompt,
            num_images_per_prompt=args.num_images,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            height=args.image_length,
            width=args.image_length,
            latents=init_latents_w,
            )
        orig_image_w = outputs_w.images[0]

        # orig_image_w = Image.open(f'../images/tree_ring_exp/w_edit_{i}.png')

        # orig_image_w = orig_image_w.resize((1024, 512))
        # orig_image_w = orig_image_w.crop((256, 0, 768, 512))
        # orig_image_w = orig_image_w.rotate(45, resample=Image.Resampling.BICUBIC)

        # orig_image_w.save(f'../images/tree_ring/w_aug_{i}.png')
        # orig_image_w.save(f'../images/tree_ring_exp/w_edit_aug_{i}.png')
        

        # orig_image_no_w.save(f'../images/tree_ring/no_w_{i}.png')
        filename = dataset.img_ids[i]
        filename = filename.split('.')[0] + ".png"
        orig_image_w.save(os.path.join(out_dir, filename))


        ### test watermark
        # distortion
        orig_image_no_w_auged, orig_image_w_auged = image_distortion(orig_image_no_w, orig_image_w, seed, args)

        # reverse img without watermarking
        img_no_w = transform_img(orig_image_no_w_auged).unsqueeze(0).to(text_embeddings.dtype).to(device)
        # print(img_no_w.min(), img_no_w.max())
        # exit()
        image_latents_no_w = pipe.get_image_latents(img_no_w, sample=False)
        # print(image_latents_no_w.min(), image_latents_no_w.max())

        reversed_latents_no_w = pipe.forward_diffusion(
            latents=image_latents_no_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )

        # reverse img with watermarking
        img_w = transform_img(orig_image_w_auged).unsqueeze(0).to(text_embeddings.dtype).to(device)
        image_latents_w = pipe.get_image_latents(img_w, sample=False)

        reversed_latents_w = pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )

        # eval
        no_w_metric, w_metric = eval_watermark(reversed_latents_no_w, reversed_latents_w, watermarking_mask, gt_patch, args)

        if args.reference_model is not None:
            sims = measure_similarity([orig_image_no_w, orig_image_w], current_prompt, ref_model, ref_clip_preprocess, ref_tokenizer, device)
            w_no_sim = sims[0].item()
            w_sim = sims[1].item()
        else:
            w_no_sim = 0
            w_sim = 0

        results.append({
            'no_w_metric': no_w_metric, 'w_metric': w_metric, 'w_no_sim': w_no_sim, 'w_sim': w_sim,
        })

        no_w_metrics.append(no_w_metric)
        w_metrics.append(w_metric)

        if args.with_tracking:
            if (args.reference_model is not None) and (i < args.max_num_log_image):
                # log images when we use reference_model
                table.add_data(wandb.Image(orig_image_no_w), w_no_sim, wandb.Image(orig_image_w), w_sim, current_prompt, no_w_metric, w_metric)
            else:
                table.add_data(None, w_no_sim, None, w_sim, current_prompt, no_w_metric, w_metric)

        clip_scores.append(w_no_sim)
        clip_scores_w.append(w_sim)

    # roc
    preds = no_w_metrics +  w_metrics
    t_labels = [1] * len(no_w_metrics) + [0] * len(w_metrics)

    print(no_w_metrics)
    print(w_metrics)


    fpr, tpr, thresholds = metrics.roc_curve(t_labels, preds, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr))/2)
    low = tpr[np.where(fpr<.01)[0][-1]]

    if args.with_tracking:
        wandb.log({'Table': table})
        wandb.log({'clip_score_mean': mean(clip_scores), 'clip_score_std': stdev(clip_scores),
                   'w_clip_score_mean': mean(clip_scores_w), 'w_clip_score_std': stdev(clip_scores_w),
                   'auc': auc, 'acc':acc, 'TPR@1%FPR': low})
    
    print(f'clip_score_mean: {mean(clip_scores)}')
    print(f'w_clip_score_mean: {mean(clip_scores_w)}')
    print(f'auc: {auc}, acc: {acc}, TPR@1%FPR: {low}')


def decode_tree_ring(input_dataset=None, dataset_name='imagenet'):

    parser = argparse.ArgumentParser(description='diffusion watermark')
    parser.add_argument('--run_name', default='test')
    parser.add_argument('--dataset', default='Gustavosta/Stable-Diffusion-Prompts')
    parser.add_argument('--start', default=0, type=int)
    parser.add_argument('--end', default=10, type=int)
    parser.add_argument('--image_length', default=512, type=int)
    parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base')
    parser.add_argument('--with_tracking', action='store_true')
    parser.add_argument('--num_images', default=1, type=int)
    parser.add_argument('--guidance_scale', default=7.5, type=float)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--test_num_inference_steps', default=None, type=int)
    parser.add_argument('--reference_model', default=None)
    parser.add_argument('--reference_model_pretrain', default=None)
    parser.add_argument('--max_num_log_image', default=100, type=int)
    parser.add_argument('--gen_seed', default=0, type=int)

    # watermark
    parser.add_argument('--w_seed', default=999999, type=int)
    parser.add_argument('--w_channel', default=0, type=int)
    parser.add_argument('--w_pattern', default='rand')
    parser.add_argument('--w_mask_shape', default='circle')
    parser.add_argument('--w_radius', default=10, type=int)
    parser.add_argument('--w_measurement', default='l1_complex')
    parser.add_argument('--w_injection', default='complex')
    parser.add_argument('--w_pattern_const', default=0, type=float)
    
    # for image distortion
    parser.add_argument('--r_degree', default=None, type=float)
    parser.add_argument('--jpeg_ratio', default=None, type=int)
    parser.add_argument('--crop_scale', default=None, type=float)
    parser.add_argument('--crop_ratio', default=None, type=float)
    parser.add_argument('--gaussian_blur_r', default=None, type=int)
    parser.add_argument('--gaussian_std', default=None, type=float)
    parser.add_argument('--brightness_factor', default=None, type=float)
    parser.add_argument('--rand_aug', default=0, type=int)

    
    args = parser.parse_args(['--w_channel', '3', 
                '--w_pattern', 'ring'])

    if args.test_num_inference_steps is None:
        args.test_num_inference_steps = args.num_inference_steps


    table = None
    if args.with_tracking:
        wandb.init(project='diffusion_watermark', name=args.run_name, tags=['tree_ring_watermark'])
        wandb.config.update(args)
        table = wandb.Table(columns=['gen_no_w', 'no_w_clip_score', 'gen_w', 'w_clip_score', 'prompt', 'no_w_metric', 'w_metric'])
    
    # load diffusion model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    scheduler = DPMSolverMultistepScheduler.from_pretrained(args.model_id, subfolder='scheduler', cache_dir=DIR)
    pipe = InversableStableDiffusionPipeline.from_pretrained(
        args.model_id,
        scheduler=scheduler,
        torch_dtype=torch.float16,
        revision='fp16',
        cache_dir=DIR
        )
    pipe = pipe.to(device)

    # reference model
    if args.reference_model is not None:
        ref_model, _, ref_clip_preprocess = open_clip.create_model_and_transforms(args.reference_model, pretrained=args.reference_model_pretrain, device=device, cache_dir=DIR)
        ref_tokenizer = open_clip.get_tokenizer(args.reference_model)

    # dataset
    if not input_dataset:
        raise ModuleNotFoundError(f"Loading the dataset is not implemented!")
    dataset = input_dataset

    tester_prompt = '' # assume at the detection time, the original prompt is unknown
    text_embeddings = pipe.get_text_embedding(tester_prompt)

    # ground-truth patch
    gt_patch = get_watermarking_pattern(pipe, args, device)

    results = []
    clip_scores = []
    clip_scores_w = []
    no_w_metrics = []
    w_metrics = []

    tensor_to_pil_transform = transforms.ToPILImage()
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for images, _ in tqdm(dataloader):
        pil_image_w = tensor_to_pil_transform(images.to(device).squeeze(0))

        img_w = transform_img(pil_image_w).unsqueeze(0).to(text_embeddings.dtype).to(device)
        image_latents_w = pipe.get_image_latents(img_w, sample=False)

        latents_w = pipe.forward_diffusion(
            latents=image_latents_w,
            text_embeddings=text_embeddings,
            guidance_scale=1,
            num_inference_steps=args.test_num_inference_steps,
        )

        latents_no_w = latents_w

        watermarking_mask = get_watermarking_mask(latents_w, args, device)

        # eval
        no_w_metric, w_metric = eval_watermark(latents_no_w, latents_w, watermarking_mask, gt_patch, args)

        no_w_metrics.append(no_w_metric)
        w_metrics.append(w_metric)


    # roc
    preds = no_w_metrics +  w_metrics
    t_labels = [1] * len(no_w_metrics) + [0] * len(w_metrics)

    # w_metrics = np.array(w_metrics)
    # no_w_metrics = np.array(no_w_metrics)

    # print(no_w_metrics)
    # print(w_metrics)

    # thr = 78.0
    
    return [1 - (x / 100) for x in w_metrics]
