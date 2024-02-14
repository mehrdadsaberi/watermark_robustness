# Guided Diffusion Model for DiffPure
mkdir -p ./DiffPure/pretrained/guided_diffusion
wget -O ./DiffPure/pretrained/guided_diffusion/256x256_diffusion_uncond.pt https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt

# watermarkDM
mkdir -p checkpoints/watermarkDM
wget -O checkpoints/watermarkDM/imagenet_decoder.pth "https://watermarkattack.s3.amazonaws.com/watermarkDM_imagenet_decoder.pth"
wget -O checkpoints/watermarkDM/imagenet_encoder.pth "https://watermarkattack.s3.amazonaws.com/watermarkDM_imagenet_encoder.pth"

# stegaStamp
mkdir -p checkpoints/stegaStamp
wget -O checkpoints/stegaStamp/stega.zip "https://watermarkattack.s3.amazonaws.com/stegastamp_pretrained.zip"
unzip checkpoints/stegaStamp/stega.zip -d checkpoints/stegaStamp
rm checkpoints/stegaStamp/stega.zip

# MBRS
wget -O MBRS_repo/results/MBRS_256_m256/models/D_42.pth "https://watermarkattack.s3.amazonaws.com/MBRS_D_42.pth"
wget -O MBRS_repo/results/MBRS_256_m256/models/EC_42.pth "https://watermarkattack.s3.amazonaws.com/MBRS_EC_42.pth"

# Binary Classifiers for Adversarial Attack
mkdir checkpoints/classifiers
wget -O checkpoints/classifiers/treeRing_classifier.pt "https://watermarkattack.s3.amazonaws.com/treeRing_classifier.pt"
wget -O checkpoints/classifiers/stegaStamp_classifier.pt "https://watermarkattack.s3.amazonaws.com/stegaStamp_classifier.pt"

