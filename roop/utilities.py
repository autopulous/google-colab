import glob
import mimetypes
import os
import platform
import shutil
import ssl
import subprocess
import urllib
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

import roop.globals

TEMP_DIRECTORY = 'temp'
TEMP_VIDEO_FILE = 'temp.mp4'

# monkey patch ssl for mac

if platform.system().lower() == 'darwin':
    ssl._create_default_https_context = ssl._create_unverified_context


def run_ffmpeg(args: List[str]) -> bool:
    commands = ['ffmpeg', '-hide_banner', '-loglevel', roop.globals.log_level]
    commands.extend(args)

    try:
        subprocess.check_output(commands, stderr=subprocess.STDOUT)
        return True
    except Exception:
        pass

    return False


def detect_fps(input_path: str) -> float:
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1', input_path]
    output = subprocess.check_output(command).decode().strip().split('/')

    try:
        numerator, denominator = map(int, output)
        return numerator / denominator
    except Exception:
        pass

    return 30


def extract_frames(input_path: str, fps: float = 30) -> bool:
    temp_directory_path = get_temp_directory_path(input_path)
    temp_frame_quality = roop.globals.temp_frame_quality * 31 // 100

    # Example extract command line command
    # ffmpeg -hide_banner -hwaccel auto -i ..\?.mp4 -q:v 0 -pix_fmt rgb24 -vf fps=30 %04d.png

    return run_ffmpeg(['-hwaccel', 'auto', '-i', input_path, '-q:v', str(temp_frame_quality), '-pix_fmt', 'rgb24', '-vf', 'fps=' + str(fps), os.path.join(temp_directory_path, '%04d.' + roop.globals.temp_frame_format)])


def create_video(input_path: str, fps: float = 30) -> bool:
    temp_output_path = get_temp_output_path(input_path)
    temp_directory_path = get_temp_directory_path(input_path)
    start_position = get_start_position(temp_directory_path, fps)

    commands = ['-hwaccel', 'auto', '-r', str(fps), '-ss', start_position, '-i', os.path.join(temp_directory_path, '%04d.' + roop.globals.temp_frame_format), '-c:v', roop.globals.output_video_encoder]

    output_video_lossiness = (roop.globals.output_video_lossiness + 1) * 51 // 100

    if roop.globals.output_video_encoder in ['libx264', 'libx265', 'libvpx']:
        commands.extend(['-crf', str(output_video_lossiness)])

    if roop.globals.output_video_encoder in ['h264_nvenc', 'hevc_nvenc']:
        commands.extend(['-cq', str(output_video_lossiness)])

    commands.extend(['-pix_fmt', 'yuv420p', '-vf', 'colorspace=bt709:iall=bt601-6-625:fast=1', '-y', temp_output_path])

    # Example create video command line command
    # ffmpeg -hide_banner -hwaccel auto -r 30 -start_number 0001 -i .\%04d.png -c:v libx264 -crf 0 -pix_fmt yuv420p -vf colorspace=bt709:iall=bt601-6-625:fast=1 -y x.mp4

    return run_ffmpeg(commands)


def restore_audio(input_path: str, output_path: str) -> None:
    temp_output_path = get_temp_output_path(input_path)
    temp_directory_path = get_temp_directory_path(input_path)
    start_position = get_start_position(temp_directory_path, fps)

    done = run_ffmpeg(['-i', temp_output_path, '-ss', start_position, '-i', input_path, '-shortest' '-c:v', 'copy', '-map', '0:v:0', '-map', '1:a:0', '-y', output_path])

    # Example restore command line command
    # ffmpeg -hide_banner -hwaccel auto -i /content/drive/MyDrive/Colab/input/temp/video.mp4 -ss 50 -i /content/drive/MyDrive/Colab/input/split-4.mp4 -shortest -c:v copy -map 0:v:0 -map 1:a:0 -y /content/drive/MyDrive/Colab/input/temp/merged.mp4

    if not done:
        move_temp(input_path, output_path)


def get_temp_frame_paths(input_path: str) -> List[str]:
    temp_directory_path = get_temp_directory_path(input_path)
    return glob.glob((os.path.join(glob.escape(temp_directory_path), '*.' + roop.globals.temp_frame_format)))


def get_temp_directory_path(input_path: str) -> str:
    input_name, _ = os.path.splitext(os.path.basename(input_path))
    input_directory_path = os.path.dirname(input_path)
    return os.path.join(input_directory_path, TEMP_DIRECTORY, input_name)


def get_temp_output_path(input_path: str) -> str:
    temp_directory_path = get_temp_directory_path(input_path)
    return os.path.join(temp_directory_path, TEMP_VIDEO_FILE)


def get_start_position(directory_path: str, fps: float = 30) -> str:
    frame = min(glob.glob(directory_path + '/*.' + roop.globals.temp_frame_format)).split('/')[-1].split('.')[0]
    position = int(frame) / fps
    hours = int(position / 3600)
    position -= hours * 3600
    minutes = int(position / 60)
    position -= minutes * 60
    seconds = int(position)
    position -= seconds
    milliseconds = int(position * 100)
    return f'{hours:02}:{minutes:02}:{seconds:02}:{milliseconds:02}'


def get_frame_count(directory_path: str) -> int:
    return len(glob.glob(directory_path + '/*.' + roop.globals.temp_frame_format))


def normalize_output_path(replacement_path: str, input_path: str, output_path: str) -> Optional[str]:
    if replacement_path and input_path and output_path:
        replacement_name, _ = os.path.splitext(os.path.basename(replacement_path))
        input_name, input_extension = os.path.splitext(os.path.basename(input_path))

        if os.path.isdir(output_path):
            return os.path.join(output_path, replacement_name + '-' + input_name + input_extension)

    return output_path


def create_temp(input_path: str) -> None:
    temp_directory_path = get_temp_directory_path(input_path)
    Path(temp_directory_path).mkdir(parents=True, exist_ok=True)


def move_temp(input_path: str, output_path: str) -> None:
    temp_output_path = get_temp_output_path(input_path)

    if os.path.isfile(temp_output_path):
        if os.path.isfile(output_path):
            os.remove(output_path)

        shutil.move(temp_output_path, output_path)


def clean_temp(input_path: str) -> None:
    if roop.globals.keep_frames:
        return

    temp_directory = get_temp_directory_path(input_path)
    
    if os.path.isdir(temp_directory):
        shutil.rmtree(temp_directory)
    
    temp_parent_directory = os.path.dirname(temp_directory)
    
    if os.path.isdir(temp_parent_directory) and not os.listdir(temp_parent_directory):
        os.rmdir(temp_parent_directory)


def has_image_extension(image_path: str) -> bool:
    return image_path.lower().endswith(('png', 'jpg', 'jpeg', 'webp'))


def is_image(image_path: str) -> bool:
    if image_path and os.path.isfile(image_path):
        mimetype, _ = mimetypes.guess_type(image_path)
        return bool(mimetype and mimetype.startswith('image/'))

    return False


def is_video(video_path: str) -> bool:
    if video_path and os.path.isfile(video_path):
        mimetype, _ = mimetypes.guess_type(video_path)
        return bool(mimetype and mimetype.startswith('video/'))

    return False


def conditional_download(download_directory_path: str, urls: List[str]) -> None:
    if not os.path.exists(download_directory_path):
        os.makedirs(download_directory_path)

    for url in urls:
        download_file_path = os.path.join(download_directory_path, os.path.basename(url))

        if not os.path.exists(download_file_path):
            request = urllib.request.urlopen(url)  # type: ignore[attr-defined]
            total = int(request.headers.get('Content-Length', 0))

            with tqdm(total=total, desc='Downloading', unit='B', unit_scale=True, unit_divisor=1024) as progress:
                urllib.request.urlretrieve(url, download_file_path, reporthook=lambda count, block_size, total_size: progress.update(block_size))  # type: ignore[attr-defined]


def resolve_relative_path(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))
