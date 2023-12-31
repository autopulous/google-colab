from typing import Any, List, Callable

import cv2
import insightface
import os
import threading

import roop.globals
import roop.processors.frame.core

from roop.download import conditional_download
from roop.face_analyser import get_one_face, get_many_faces, find_similar_face
from roop.face_reference import get_face_reference, set_face_reference, clear_face_reference
from roop.file import get_absolute_path, get_temp_directory_path, is_image, is_video
from roop.typing import Face, Frame
from roop.progress import update_status

FACE_SWAPPER = None
THREAD_LOCK = threading.Lock()
NAME = 'ROOP.FACE-SWAPPER'


def get_face_swapper() -> Any:
    global FACE_SWAPPER

    with THREAD_LOCK:
        if FACE_SWAPPER is None:
            model_file_path = get_absolute_path('../models/inswapper_128.onnx')
            FACE_SWAPPER = insightface.model_zoo.get_model(model_file_path, providers=roop.globals.execution_providers)

    return FACE_SWAPPER


def clear_face_swapper() -> None:
    global FACE_SWAPPER

    FACE_SWAPPER = None


def pre_check() -> bool:
    download_directory_path = get_absolute_path('../models')
    conditional_download(download_directory_path, ['https://huggingface.co/CountFloyd/deepfake/resolve/main/inswapper_128.onnx'])
    return True


def pre_start() -> bool:
    if not is_image(roop.globals.replacement_path):
        update_status('Select an image for replacement path', NAME)
        return False
    elif not get_one_face(cv2.imread(roop.globals.replacement_path)):
        update_status('No face in replacement path detected', NAME)
        return False

    if roop.globals.reprocess_frames:
        temp_directory_path = get_temp_directory_path(roop.globals.input_path)
        if not os.path.exists(temp_directory_path) or not os.path.isdir(temp_directory_path):
            update_status(f'Extracted video frames cannot be found in: {temp_directory_path}', NAME)
            return False
    else:
        if not is_image(roop.globals.input_path) and not is_video(roop.globals.input_path):
            update_status('Select an image or video for target path', NAME)
            return False

    return True


def post_process() -> None:
    clear_face_swapper()
    clear_face_reference()


def swap_face(source_face: Face, target_face: Face, temp_frame: Frame) -> Frame:
    return get_face_swapper().get(temp_frame, target_face, source_face, paste_back=True)


def process_frame(source_face: Face, reference_face: Face, temp_frame: Frame) -> Frame:
    if roop.globals.many_faces:
        many_faces = get_many_faces(temp_frame)
        if many_faces:
            for target_face in many_faces:
                temp_frame = swap_face(source_face, target_face, temp_frame)
    else:
        target_face = find_similar_face(temp_frame, reference_face)
        if target_face:
            temp_frame = swap_face(source_face, target_face, temp_frame)

    return temp_frame


def process_frames(replacement_path: str, sorted_frame_file_paths: List[str], update: Callable[[], None]) -> None:
    source_face = get_one_face(cv2.imread(replacement_path))
    reference_face = None if roop.globals.many_faces else get_face_reference()

    for frame_file_path in sorted_frame_file_paths:
        temp_frame = cv2.imread(frame_file_path)
        result = process_frame(source_face, reference_face, temp_frame)
        cv2.imwrite(frame_file_path, result)

        if update:
            update()


def process_image(replacement_path: str, input_path: str, output_path: str) -> None:
    source_face = get_one_face(cv2.imread(replacement_path))
    target_frame = cv2.imread(input_path)
    reference_face = None if roop.globals.many_faces else get_one_face(target_frame, roop.globals.reference_face_position)
    result = process_frame(source_face, reference_face, target_frame)
    cv2.imwrite(output_path, result)


def process_video(replacement_path: str, sorted_frame_file_paths: List[str]) -> None:
    if not roop.globals.many_faces and not get_face_reference():
        reference_frame = cv2.imread(sorted_frame_file_paths[roop.globals.reference_frame_number])
        reference_face = get_one_face(reference_frame, roop.globals.reference_face_position)
        set_face_reference(reference_face)

    roop.processors.frame.core.process_video(replacement_path, sorted_frame_file_paths, process_frames)
