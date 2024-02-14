#!/bin/bash


# python apply_watermark.py \
# --wm-method dwtDct \
# --dataset imagenet \
# --data-dir ./images/imagenet/org \
# --out-dir ./images/imagenet/dwtDct \

# python apply_watermark.py \
# --wm-method dwtDctSvd \
# --dataset imagenet \
# --data-dir ./images/imagenet/org \
# --out-dir ./images/imagenet/dwtDctSvd \

# python apply_watermark.py \
# --wm-method rivaGan \
# --dataset imagenet \
# --data-dir ./images/imagenet/org \
# --out-dir ./images/imagenet/rivaGan \

# python apply_watermark.py \
# --wm-method watermarkDM \
# --dataset imagenet \
# --data-dir ./images/imagenet/org \
# --out-dir ./images/imagenet/watermarkDM \

# python apply_watermark.py \
# --wm-method MBRS \
# --dataset imagenet \
# --data-dir ./images/imagenet/org \
# --out-dir ./images/imagenet/MBRS \

# python apply_watermark.py \
# --wm-method treeRing \
# --dataset imagenet \
# --data-dir ./images/imagenet/org \
# --out-dir ./images/imagenet/treeRing \

# python apply_watermark.py \
# --wm-method stegaStamp \
# --dataset imagenet \
# --data-dir ./images/imagenet/org \
# --out-dir ./images/imagenet/stegaStamp \



################  for classifier training

## since TreeRing is slow, generation of 10000 images takes a long time
## if you want a quicker way, you can try training the classifier with a lower number of samples
## the strength of the adversarial attack might decrease if lower number of samples is used

# python apply_watermark.py \
# --wm-method treeRing \
# --dataset imagenet \
# --data-cnt 10000 \
# --data-dir ./images/imagenet/org_train \
# --out-dir ./images/imagenet/treeRing_wm_train \


# python apply_watermark.py \
# --wm-method stegaStamp \
# --dataset imagenet \
# --data-cnt 10000 \
# --data-dir ./images/imagenet/org_train \
# --out-dir ./images/imagenet/stegaStamp_wm_train \

