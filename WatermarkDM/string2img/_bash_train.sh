#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 python train_cifar10.py \
# --data_dir /cmlscratch/msaberi/data/cifar/uncompressed \
# --image_resolution 32 \
# --output_dir ./_output/cifar10_64 \
# --bit_length 64 \
# --batch_size 64 \
# --num_epochs 100 \


CUDA_VISIBLE_DEVICES=0 python train_imagenet.py \
--data_dir /fs/cml-datasets/ImageNet/ILSVRC2012 \
--image_resolution 256 \
--output_dir ./_output/imagenet_64_adv_03 \
--bit_length 64 \
--batch_size 64 \
--num_epochs 100 \


# CUDA_VISIBLE_DEVICES=0 python train_cifar10.py \
# --data_dir /cmlscratch/msaberi/data/cifar/uncompressed \
# --image_resolution 32 \
# --output_dir ./_output/cifar10_8 \
# --bit_length 8 \
# --batch_size 64 \
# --num_epochs 100 \