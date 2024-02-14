import os
import random
import cv2
import argparse

def extract_frames(video_path, output_folder, num_frames=5):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    
    # Get total number of frames in the video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Randomly select frames
    frame_indices = random.sample(range(total_frames), num_frames)
    
    # Read and save selected frames
    for index in frame_indices:
        # Set the frame index
        video.set(cv2.CAP_PROP_POS_FRAMES, index)
        
        # Read the frame
        ret, frame = video.read()
        
        if ret:
            # Generate output file name
            output_file = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(video_path))[0]}_{index}.png")
            
            # Save the frame as a PNG image
            cv2.imwrite(output_file, frame)
    
    # Release the video file
    video.release()

def extract_frames_from_videos(source_folder, destination_folder, num_frames=5):
    # Get a list of video files in the source folder
    video_files = [f for f in os.listdir(source_folder) if f.endswith(".mp4")]
    
    # Extract frames from each video
    for video_file in video_files:
        video_path = os.path.join(source_folder, video_file)
        
        # Extract frames from the video
        extract_frames(video_path, destination_folder, num_frames)
        
        print(f"Extracted frames from {video_file}", flush=True, end="\r")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Extract random frames from MP4 videos.")
    parser.add_argument("source_folder", help="Path to the source folder containing MP4 videos.")
    parser.add_argument("destination_folder", help="Path to the destination folder to save the PNG images.")
    parser.add_argument("--num_frames", type=int, default=5, help="Number of frames to extract per video (default: 5)")
    
    args = parser.parse_args()
    
    source_folder = args.source_folder
    destination_folder = args.destination_folder
    num_frames_per_video = args.num_frames
    
    extract_frames_from_videos(source_folder, destination_folder, num_frames_per_video)