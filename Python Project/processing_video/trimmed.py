import cv2
import os

def extract_frames_from_videos(base_folder, output_folder, frame_rate=5):
    """
    Extract frames from videos in a nested folder structure.
    Each subfolder contains pairs of thermal and RGB videos.

    :param base_folder: Path to the folder containing subfolders of videos.
    :param output_folder: Path to the folder where extracted frames will be saved.
    :param frame_rate: Number of frames to extract per second.
    """
    # Iterate through all subfolders
    for subfolder in os.listdir(base_folder):
        subfolder_path = os.path.join(base_folder, subfolder)
        if not os.path.isdir(subfolder_path):
            continue  # Skip if not a folder

        print(f"Processing subfolder: {subfolder}")

        # Create corresponding output folder for this subfolder
        subfolder_output = os.path.join(output_folder, subfolder)
        os.makedirs(subfolder_output, exist_ok=True)

        # Process each video in the subfolder
        for video_file in os.listdir(subfolder_path):
            video_path = os.path.join(subfolder_path, video_file)
            if not video_file.lower().endswith(('.mp4', '.avi', '.mkv', '.mov')):
                continue  # Skip non-video files

            print(f"  Processing video: {video_file}")

            # Create output folder for frames from this video
            video_name = os.path.splitext(video_file)[0]
            video_output = os.path.join(subfolder_output, video_name)
            os.makedirs(video_output, exist_ok=True)

            # Open the video and extract frames
            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                print(f"    Error: Cannot open video {video_file}")
                continue

            fps = video.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps / frame_rate)  # Calculate frame interval

            frame_count = 0
            saved_frames = 0

            while True:
                ret, frame = video.read()
                if not ret:
                    break  # Exit loop if no more frames

                if frame_count % frame_interval == 0:
                    frame_filename = os.path.join(video_output, f"frame_{saved_frames:04d}.jpg")
                    cv2.imwrite(frame_filename, frame)
                    saved_frames += 1
                    print(f"    Saved: {frame_filename}")

                frame_count += 1

            video.release()
            print(f"    Completed video: {video_file}, Total frames saved: {saved_frames}")

# Example usage
base_folder = "D:\Python Project\processing_video/video"  # Folder containing subfolders of videos
output_folder = "D:\Python Project\processing_video/frames"  # Where extracted frames will be saved
frame_rate = 5  # Extract 5 frames per second

extract_frames_from_videos(base_folder, output_folder, frame_rate)
