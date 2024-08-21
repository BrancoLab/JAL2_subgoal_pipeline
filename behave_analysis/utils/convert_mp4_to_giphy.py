# must pip install moviepy

import os
from moviepy.editor import VideoFileClip


def convert_mp4_to_gif(input_folder, output_folder, max_size=50, resize_factor=0.5, fps=10):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(".mp4"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".gif")
            try:
                print(f"Converting {input_path} to {output_path}...")
                clip = VideoFileClip(input_path)

                # Resize the clip
                clip_resized = clip.resize(resize_factor)

                # Reduce the frame rate
                clip_resized = clip_resized.set_fps(fps)

                clip_resized.write_gif(output_path, program="ffmpeg")

                # Check the file size
                file_size = os.path.getsize(output_path) / (1024 * 1024)  # Convert bytes to MB
                if file_size > max_size:
                    print(f"Warning: {output_path} is {file_size:.2f}MB, larger than the max size of {max_size}MB.")
                else:
                    print(f"Successfully converted {input_path} to {output_path}, file size: {file_size:.2f}MB")
            except Exception as e:
                print(f"Failed to convert {input_path}: {e}")


input_folder = r"Z:\Jasmine_Laurence\Experimental_Data\JAL004\004_flip_2023_09_03T12_04_16\processed_data\trials\flip\escape"
output_folder = input_folder

convert_mp4_to_gif(input_folder, output_folder)
