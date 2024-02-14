#!/bin/bash

mkdir -p ./checkpoints/classifiers/


python train_classifier.py \
--wm-dir images/imagenet/treeRing_wm_train \
--org-dir images/imagenet/org_train \
--out-dir checkpoints/classifiers/treeRing_classifier.pt \
--data-cnt 10000 \
--epochs 10 \


python train_classifier.py \
--wm-dir images/imagenet/stegaStamp_wm_train \
--org-dir images/imagenet/org_train \
--out-dir checkpoints/classifiers/stegaStamp_classifier.pt \
--data-cnt 10000 \
--epochs 10 \