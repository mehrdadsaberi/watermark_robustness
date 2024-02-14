1. Make folders images/ and results/ for storing outputs.
2. Clone https://github.com/yunqing-me/WatermarkDM and https://github.com/YuxinWenRick/tree-ring-watermark to current directory.
3. Note that some of the codes in this folder are adapted directly from the above two repositories.
4. Run get_clean_images.py to save clean images from ImageNet. Make sure to pass arguments to folder where ImageNet is saved.
5. Run create_noises.py to create noisy images.
6. Run wm_noise.sh to watermark the noisy images using various watermarking tools with just a black-box access.
7. Run mixup_noises.sh to blend watermarked noisy images to clean images.
8. Run compute_scores.sh to plot the results of spoofing.