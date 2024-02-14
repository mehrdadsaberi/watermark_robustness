#!/bin/bash

python adv_attack.py \
--wm-method treeRing \
--wm-dir ./images/imagenet/treeRing \
--org-dir ./images/imagenet/org \
--model-dir checkpoints/classifiers/treeRing_classifier.pt \


python adv_attack.py \
--wm-method stegaStamp \
--wm-dir ./images/imagenet/stegaStamp \
--org-dir ./images/imagenet/org \
--model-dir checkpoints/classifiers/stegaStamp_classifier.pt \

