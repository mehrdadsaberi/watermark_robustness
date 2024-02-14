
#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python embed_watermark_cifar10.py \
--encoder_path checkpoints/stegastamp_64_20062023_13:26:38_encoder.pth \
--decoder_path checkpoints/stegastamp_64_20062023_13:26:38_decoder.pth \
--image_resolution 32 \
--identical_fingerprints \
--batch_size 128 \
