#!/bin/bash

cd ~/deepfake/
python download.py "/deepfake/" -d "original" -c "c40" --server="EU2"
python download.py "/deepfake/" -d "FaceSwap" -c "c40" --server="EU2"
python download.py "/deepfake/" -d "Deepfakes" -c "c40" --server="EU2"
python extract_images_from_videos.py "/deepfake/manipulated_sequences/Deepfakes/c40/videos/" "/deepfake/manipulated_sequences/Deepfakes_images"
python extract_images_from_videos.py "/deepfake/manipulated_sequences/FaceSwap/c40/videos/" "/deepfake/manipulated_sequences/FaceSwap_images"
python extract_images_from_videos.py "/deepfake/original_sequences/youtube/c40/videos/" "/deepfake/original_sequences/youtube_images"

cd ~/faceswap
conda activate faceswap
python faceswap.py extract -i "/deepfake/manipulated_sequences/Deepfakes_images/" -o "/deepfake/manipulated_sequences/Deepfakes_images_extract/"
python faceswap.py extract -i "/deepfake/manipulated_sequences/FaceSwap_images/" -o "/deepfake/manipulated_sequences/FaceSwap_images_extract/"
python faceswap.py extract -i "/deepfake/original_sequences/youtube_images/" -o "/deepfake/original_sequences/youtube_images_extract/"

cd ~/deepfake
python train_test_split.py "/deepfake/manipulated_sequences/Deepfakes_images_extract" "/deepfake/preprocessed/Deepfakes"
python train_test_split.py "/deepfake/manipulated_sequences/FaceSwap_images_extract" "/deepfake/preprocessed/FaceSwap"
python train_test_split.py "/deepfake/original_sequences/youtube_images_extract" "/deepfake/preprocessed/original"

cd /deepfake/
mkdir datasets
mkdir datasets/FaceSwap
mkdir datasets/Deepfakes
mkdir datasets/FaceSwap/train
mkdir datasets/FaceSwap/test
mkdir datasets/Deepfakes/train
mkdir datasets/Deepfakes/test/
cp -r preprocessed/original/train/ datasets/FaceSwap/train/original
cp -r preprocessed/original/train/ datasets/Deepfakes/train/original
cp -r preprocessed/original/test/ datasets/Deepfakes/test/original
cp -r preprocessed/original/test/ datasets/FaceSwap/test/original
cp -r preprocessed/FaceSwap/train/ datasets/FaceSwap/train/manipulated
cp -r preprocessed/FaceSwap/test/ datasets/FaceSwap/test/manipulatedC
cp -r preprocessed/Deepfakes/test/ datasets/Deepfakes/test/manipulated
cp -r preprocessed/Deepfakes/train/ datasets/Deepfakes/train/manipulated