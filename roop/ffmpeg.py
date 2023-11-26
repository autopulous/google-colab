import glob
from typing import List
import os
import subprocess
import time

import roop.globals

from roop.core import update_status
from roop.file import get_temp_directory_path, get_temp_output_file_path, move_temp_file


def detect_fps(input_path: str) -> float:
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1', input_path]
    output = subprocess.check_output(command).decode().strip().split('/')

    try:
        numerator, denominator = map(int, output)
        return numerator / denominator
    except Exception:
        pass

    return 30


# Example extract command line command
# ffmpeg -hide_banner -hwaccel auto -i ..\?.mp4 -q:v 0 -pix_fmt rgb24 -vf fps=30 %04d.png

def extract_frames(input_file_path: str, fps: float = 30) -> bool:
    temp_directory_path = get_temp_directory_path(input_file_path)
    temp_frame_quality = roop.globals.temp_frame_quality * 31 // 100

    return run_ffmpeg(['-hwaccel', 'auto', '-i', input_file_path, '-q:v', str(temp_frame_quality), '-pix_fmt', 'rgb24', '-vf', 'fps=' + str(fps), os.path.join(temp_directory_path, '%04d.' + roop.globals.temp_frame_format)])


# Example create video command line command
# ffmpeg -hide_banner -hwaccel auto -r 30 -start_number 0001 -i .\%04d.png -c:v libx264 -crf 0 -pix_fmt yuv420p -vf colorspace=bt709:iall=bt601-6-625:fast=1 -y x.mp4

def create_video(input_file_path: str, fps: float = 30) -> bool:
    commands = ['-hwaccel', 'auto', '-r', str(fps)]

    temp_directory_path = get_temp_directory_path(input_file_path)
    first_frame_number = get_first_frame_number(temp_directory_path)

    if 0 < int(first_frame_number):
        commands.extend(['-start_number', first_frame_number])

    commands.extend(['-i', os.path.join(temp_directory_path, '%04d.' + roop.globals.temp_frame_format), '-c:v', roop.globals.output_video_encoder])

    output_video_lossiness = (roop.globals.output_video_lossiness + 1) * 51 // 100

    if roop.globals.output_video_encoder in ['libx264', 'libx265', 'libvpx']:
        commands.extend(['-crf', str(output_video_lossiness)])

    if roop.globals.output_video_encoder in ['h264_nvenc', 'hevc_nvenc']:
        commands.extend(['-cq', str(output_video_lossiness)])

    temp_output_path = get_temp_output_file_path(input_file_path)
    commands.extend(['-pix_fmt', 'yuv420p', '-vf', 'colorspace=bt709:iall=bt601-6-625:fast=1', '-y', temp_output_path])

    return run_ffmpeg(commands)


# Example restore command line command
# ffmpeg -hide_banner -hwaccel auto -i /content/drive/MyDrive/Colab/input/temp/video.mp4 -ss 00:00:50:00 -i /content/drive/MyDrive/Colab/input/original.mp4 -shortest -c:v copy -map 0:v:0 -map 1:a:0 -y /content/drive/MyDrive/Colab/input/temp/merged.mp4

def restore_audio(input_file_path: str, output_path: str, fps: float = 30) -> None:
    temp_output_path = get_temp_output_file_path(input_file_path)
    temp_directory_path = get_temp_directory_path(input_file_path)

    commands = ['-i', temp_output_path]

    if 0 < int(get_first_frame_number(temp_directory_path)):
        commands.extend(['-ss', get_first_frame_time_index(temp_directory_path, fps)])

    commands.extend(['-i', input_file_path, '-shortest', '-c:v', 'copy', '-map', '0:v:0', '-map', '1:a:0', '-y', output_path])

    done = run_ffmpeg(commands)

    if not done:
        move_temp_file(input_file_path, output_path)


def run_ffmpeg(args: List[str]) -> bool:
    commands = ['ffmpeg', '-hide_banner', '-loglevel', roop.globals.log_level]
    commands.extend(args)

    try:
        print()
        update_status('Issuing command', 'FFMPEG')

        print()
        print(" ".join(map(str, commands)))
        print()

        start = time.time()
        subprocess.check_output(commands, stderr=subprocess.STDOUT)
        end = time.time()
        print("Processing duration: ", format_time_index(end - start))
        print()
        return True
    except Exception as exception:
        print("Error: ffmpeg command failed")
        print(exception)
        print()
        return False


def format_time_index(position: float) -> str:
    hours = int(position / 3600)
    position -= hours * 3600
    minutes = int(position / 60)
    position -= minutes * 60
    seconds = int(position)
    position -= seconds
    milliseconds = int(position * 100)
    return f'{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:02}'


def get_first_frame_number(directory_path: str) -> str:
    frame = min(glob.glob(directory_path + '/*.' + roop.globals.temp_frame_format)).split('/')[-1].split('.')[0]
    return frame


def get_first_frame_time_index(directory_path: str, fps: float = 30) -> str:
    first_frame_number = int(get_first_frame_number(directory_path)) / fps
    return format_time_index(first_frame_number)


def get_frame_count(directory_path: str) -> int:
    return len(glob.glob(directory_path + '/*.' + roop.globals.temp_frame_format))


def get_last_frame_number(directory_path: str) -> int:
    frame_count = get_frame_count(directory_path)
    first_frame_number = int(get_first_frame_number(directory_path))
    last_frame_number = first_frame_number + frame_count - 1
    return last_frame_number


def get_last_frame_time_index(directory_path: str, fps: float = 30) -> str:
    position = get_last_frame_number(directory_path) / fps
    return format_time_index(position)
