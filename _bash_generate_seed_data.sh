#!/bin/bash

# Specify path to ImageNet data
DATA_DIR= /path/to/imagenet/validation


python generate_seed_data.py \
--data-dir $DATA_DIR \
--data-cnt 100 \
--out-dir images/imagenet/org \


################  for classifier training

# python generate_seed_data.py \
# --data-dir $DATA_DIR \
# --data-cnt 10000 \
# --out-dir images/imagenet/org_train \

