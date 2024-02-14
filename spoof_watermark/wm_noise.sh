CUDA_VISIBLE_DEVICES=0 python apply_watermark.py \
--wm-method dwtDct \
--dataset noise \
--data-dir images/imagenet/noise/ \
--data-cnt 50 \
--out-dir ./images \

CUDA_VISIBLE_DEVICES=0 python apply_watermark.py \
--wm-method dwtDctSvd \
--dataset noise \
--data-dir images/imagenet/noise/ \
--data-cnt 50 \
--out-dir ./images \

CUDA_VISIBLE_DEVICES=0 python apply_watermark.py \
--wm-method watermarkDM \
--dataset noise \
--data-dir images/imagenet/noise/ \
--data-cnt 50 \
--out-dir ./images \

CUDA_VISIBLE_DEVICES=0 python apply_watermark.py \
--wm-method rivaGan \
--dataset noise \
--data-dir images/imagenet/noise/ \
--data-cnt 50 \
--out-dir ./images \

CUDA_VISIBLE_DEVICES=0 python apply_watermark.py \
--wm-method treeRing \
--dataset noise \
--data-dir images/imagenet/noise/ \
--data-cnt 50 \
--out-dir ./images \