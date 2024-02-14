Instructions for performing Deepfake experiments
=========================================

1. Get access to dataset by filling form in https://github.com/ondyari/FaceForensics
2. Download faceforensics_download_v4.py from FaceForensics and rename it to download.py in the current directory.
3. Clone https://github.com/deepfakes/faceswap/ and install faceswap software. faceswap/ folder should be in the current directory.
4. Activate the faceswap conda environment.
5. Change file names according to needs in preprocess.sh in the current directory and run it.
6. This downloads the datasets required and preprocesses them.
7. Experiment scripts are in the experiments/ folder in the current directory.
8. Change directory to experiments/
9. Run train_resnet.py or train_vgg.py to train a ResNet-18 for VGG-16-BN on the FaceSwap datatset.
	a. For Deepfake dataset, add argument -task='deepfake'
10. Make folder experiments/results/ to store output results.
11. Run train_noise_resnet.py or train_noise_vgg.py with the "-task" argument to train detectors of varying (\sigma, \alpha)-robustness.
12. Run plot.py to make the plots.
