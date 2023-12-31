from typing import Any, List, Callable
import cv2
import os
import threading
from gfpgan.utils import GFPGANer

import roop.globals
import roop.processors.frame.core

from roop.download import conditional_download
from roop.face_analyser import get_many_faces
from roop.file import get_absolute_path, get_temp_directory_path, is_image, is_video
from roop.typing import Frame, Face
from roop.progress import update_status

FACE_ENHANCER = None
THREAD_SEMAPHORE = threading.Semaphore()
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-ENHANCER'


def get_face_enhancer() -> Any:
    global FACE_ENHANCER

    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_file_path = get_absolute_path('../models/GFPGANv1.4.pth')
            # todo: set models path -> https://github.com/TencentARC/GFPGAN/issues/399
            FACE_ENHANCER = GFPGANer(model_path=model_file_path, upscale=1, device=get_device())

    return FACE_ENHANCER


def get_device() -> str:
    if 'CUDAExecutionProvider' in roop.globals.execution_providers:
        return 'cuda'

    if 'CoreMLExecutionProvider' in roop.globals.execution_providers:
        return 'mps'

    return 'cpu'


def clear_face_enhancer() -> None:
    global FACE_ENHANCER

    FACE_ENHANCER = None


def pre_check() -> bool:
    download_directory_path = get_absolute_path('../models')
    conditional_download(download_directory_path, ['https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth'])

    return True


def pre_start() -> bool:
    if roop.globals.reprocess_frames:
        temp_frame_file_path = get_temp_directory_path(roop.globals.input_path)
        if not os.path.exists(temp_frame_file_path) or not os.path.isdir(temp_frame_file_path):
            update_status(f'Extracted video frames cannot be found in: {temp_frame_file_path}', NAME)
            return False
    else:
        if not is_image(roop.globals.input_path) and not is_video(roop.globals.input_path):
            update_status('Select an image or video for target path.', NAME)
            return False

    return True


def post_process() -> None:
    clear_face_enhancer()


def enhance_face(target_face: Face, temp_frame: Frame) -> Frame:
    start_x, start_y, end_x, end_y = map(int, target_face['bbox'])
    padding_x = int((end_x - start_x) * 0.5)
    padding_y = int((end_y - start_y) * 0.5)
    start_x = max(0, start_x - padding_x)
    start_y = max(0, start_y - padding_y)
    end_x = max(0, end_x + padding_x)
    end_y = max(0, end_y + padding_y)
    temp_face = temp_frame[start_y:end_y, start_x:end_x]

    if temp_face.size:
        with THREAD_SEMAPHORE:
            _, _, temp_face = get_face_enhancer().enhance(
                temp_face,
                paste_back=True
            )
        temp_frame[start_y:end_y, start_x:end_x] = temp_face

    return temp_frame


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    many_faces = get_many_faces(temp_frame)

    if many_faces:
        for target_face in many_faces:
            temp_frame = enhance_face(target_face, temp_frame)

    return temp_frame


def process_frames(replacement_path: str, sorted_frame_file_paths: List[str], update: Callable[[], None]) -> None:
    for frame_file_path in sorted_frame_file_paths:
        temp_frame = cv2.imread(frame_file_path)
        result = process_frame(None, None, temp_frame)
        cv2.imwrite(frame_file_path, result)

        if update:
            update()


def process_image(replacement_path: str, input_path: str, output_path: str) -> None:
    target_frame = cv2.imread(input_path)
    result = process_frame(None, None, target_frame)
    cv2.imwrite(output_path, result)


def process_video(replacement_path: str, sorted_frame_file_paths: List[str]) -> None:
    roop.processors.frame.core.process_video(None, sorted_frame_file_paths, process_frames)
