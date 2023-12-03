import glob
import mimetypes
import os
import shutil

from pathlib import Path
from typing import List, Optional

import roop.globals

TEMP_DIRECTORY = 'temp'
TEMP_VIDEO_FILE = 'temp.mp4'


def get_absolute_path(path: str) -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), path))


def get_temp_frame_file_paths(input_file_path: str) -> List[str]:
    temp_directory_path = get_temp_directory_path(input_file_path)
    return sorted(glob.glob((os.path.join(glob.escape(temp_directory_path), '*.' + roop.globals.temp_frame_format))))


def get_temp_directory_path(input_file_path: str) -> str:
    input_file_name, _ = os.path.splitext(os.path.basename(input_file_path))
    input_directory_path = os.path.dirname(input_file_path)
    return os.path.join(input_directory_path, TEMP_DIRECTORY, input_file_name)


def get_temp_output_file_path(input_file_path: str) -> str:
    temp_directory_path = get_temp_directory_path(input_file_path)
    return os.path.join(temp_directory_path, TEMP_VIDEO_FILE)


def normalize_output_file_path(replacement_file_path: str, input_file_path: str, output_file_path: str) -> Optional[str]:
    if replacement_file_path and input_file_path and output_file_path:
        replacement_name, _ = os.path.splitext(os.path.basename(replacement_file_path))
        input_name, input_extension = os.path.splitext(os.path.basename(input_file_path))

        if os.path.isdir(output_file_path):
            return os.path.join(output_file_path, replacement_name + '-' + input_name + input_extension)

    return output_file_path


def create_temp_directory(input_file_path: str) -> None:
    temp_directory_path = get_temp_directory_path(input_file_path)
    Path(temp_directory_path).mkdir(parents=True, exist_ok=True)


def move_temp_file(input_file_path: str, output_file_path: str) -> None:
    temp_output_file_path = get_temp_output_file_path(input_file_path)

    if os.path.isfile(temp_output_file_path):
        if os.path.isfile(output_file_path):
            os.remove(output_file_path)

        shutil.move(temp_output_file_path, output_file_path)


def clean_temp_directory(input_file_path: str) -> None:
    if roop.globals.keep_frames:
        return

    temp_directory_path = get_temp_directory_path(input_file_path)

    if os.path.isdir(temp_directory_path):
        shutil.rmtree(temp_directory_path)

    temp_parent_directory_path = os.path.dirname(temp_directory_path)

    if os.path.isdir(temp_parent_directory_path) and not os.listdir(temp_parent_directory_path):
        os.rmdir(temp_parent_directory_path)


def has_image_extension(image_file_path: str) -> bool:
    return image_file_path.lower().endswith(('png', 'jpg', 'jpeg', 'webp'))


def is_image(image_file_path: str) -> bool:
    if image_file_path and os.path.isfile(image_file_path):
        mimetype, _ = mimetypes.guess_type(image_file_path)
        return bool(mimetype and mimetype.startswith('image/'))

    return False


def is_video(video_file_path: str) -> bool:
    if video_file_path and os.path.isfile(video_file_path):
        mimetype, _ = mimetypes.guess_type(video_file_path)
        return bool(mimetype and mimetype.startswith('video/'))

    return False
