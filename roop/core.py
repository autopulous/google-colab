import os
import sys

# single thread doubles cuda performance
# this needs to be set before torch import

if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'

# reduce tensorflow log level

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from time import gmtime, strftime
from typing import List

import warnings
import platform
import signal
import shutil
import argparse
import onnxruntime
import tensorflow

import roop.globals
import roop.metadata
import roop.ui as ui

from roop.ffmpeg import detect_fps, extract_frames, create_video, restore_audio
from roop.file import get_temp_directory_path, has_image_extension, is_image, is_video, get_sorted_frame_file_paths, create_temp_directory, move_temp_file, clean_temp_directory, normalize_output_file_path
from roop.predictor import predict_image, predict_video
from roop.processors.frame.core import get_frame_processors_modules
from roop.progress import update_status

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser(formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=100))
    program.add_argument('-i', '--input', help='input image or video file', dest='input_path')
    program.add_argument('-r', '--replacement', help='replacement image file', dest='replacement_path')
    program.add_argument('-o', '--output', help='output file or directory', dest='output_path')
    program.add_argument('--frame-processors', help='frame processors (e.g., face_swapper, face_enhancer, ...)', dest='frame_processors', default=['face_swapper'], nargs='+')
    program.add_argument('--allow-nsfw', help='skip nsfw checks', dest='allow_nsfw', action='store_true')
    program.add_argument('--keep-fps', help='keep target fps', dest='keep_fps', action='store_true')
    program.add_argument('--keep-frames', help='keep temporary frames', dest='keep_frames', action='store_true')
    program.add_argument('--reprocess-frames', help='reprocess temporary frames', dest='reprocess_frames', action='store_true')
    program.add_argument('--render-only', help='only generate a video from the temporary frames', dest='render_only', action='store_true')
    program.add_argument('--skip-video', help='skip video creation', dest='skip_video', action='store_true')
    program.add_argument('--skip-audio', help='skip copying audio to video', dest='skip_audio', action='store_true')
    program.add_argument('--many-faces', help='process every face', dest='many_faces', action='store_true')
    program.add_argument('--reference-face-position', help='position of the reference face', dest='reference_face_position', type=int, default=0)
    program.add_argument('--reference-frame-number', help='number of the reference frame', dest='reference_frame_number', type=int, default=0)
    program.add_argument('--similar-face-distance', help='face distance used for recognition', dest='similar_face_distance', type=float, default=0.85)
    program.add_argument('--temp-frame-format', help='image format used for frame extraction', dest='temp_frame_format', default='png', choices=['jpg', 'png'])
    program.add_argument('--temp-frame-quality', help='image quality used for frame extraction', dest='temp_frame_quality', type=int, default=0, choices=range(101), metavar='[0-100]')
    program.add_argument('--output-video-encoder', help='encoder used for the output video', dest='output_video_encoder', default='libx264', choices=['libx264', 'libx265', 'libvpx-vp9', 'h264_nvenc', 'hevc_nvenc'])
    program.add_argument('--output-video-lossiness', help='the amount of lossiness for the output video', dest='output_video_lossiness', type=int, default=35, choices=range(101), metavar='[0-100]')
    program.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory', type=int)
    program.add_argument('--execution-provider', help='available execution provider (choices: cpu, cuda, mps, ...)', dest='execution_provider', default=['cpu'], choices=suggest_execution_providers(), nargs='+')
    program.add_argument('--execution-threads', help='number of execution threads', dest='execution_threads', type=int, default=suggest_execution_threads())
    program.add_argument('-v', '--version', action='version', version=f'{roop.metadata.name} {roop.metadata.version}')

    args = program.parse_args()

    roop.globals.input_path = args.input_path
    roop.globals.replacement_path = args.replacement_path
    roop.globals.output_path = normalize_output_file_path(roop.globals.replacement_path, roop.globals.input_path, args.output_path)
    roop.globals.headless = roop.globals.replacement_path is not None and roop.globals.input_path is not None and roop.globals.output_path is not None
    roop.globals.frame_processors = args.frame_processors
    roop.globals.allow_nsfw = args.allow_nsfw
    roop.globals.keep_fps = args.keep_fps
    roop.globals.keep_frames = args.keep_frames
    roop.globals.reprocess_frames = args.reprocess_frames
    roop.globals.render_only = args.render_only
    roop.globals.skip_video = args.skip_video
    roop.globals.skip_audio = args.skip_audio
    roop.globals.many_faces = args.many_faces
    roop.globals.reference_face_position = args.reference_face_position
    roop.globals.reference_frame_number = args.reference_frame_number
    roop.globals.similar_face_distance = args.similar_face_distance
    roop.globals.temp_frame_format = args.temp_frame_format
    roop.globals.temp_frame_quality = args.temp_frame_quality
    roop.globals.output_video_encoder = args.output_video_encoder
    roop.globals.output_video_lossiness = args.output_video_lossiness
    roop.globals.max_memory = args.max_memory
    roop.globals.execution_providers = decode_execution_providers(args.execution_provider)
    roop.globals.execution_threads = args.execution_threads


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    if 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
        return 8

    return 1


def limit_resources() -> None:
    # prevent tensorflow memory leak

    gpus = tensorflow.config.experimental.list_physical_devices('GPU')

    for gpu in gpus:
        tensorflow.config.experimental.set_virtual_device_configuration(gpu, [
            tensorflow.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)
        ])

    # limit memory usage

    if roop.globals.max_memory:
        memory = roop.globals.max_memory * 1024 ** 3

        if platform.system().lower() == 'darwin':
            memory = roop.globals.max_memory * 1024 ** 6

        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False

    if not shutil.which('ffmpeg'):
        update_status('ffmpeg is not installed')
        return False

    return True


def process_image() -> None:
    if not roop.globals.allow_nsfw:
        update_status('NSFW check...')
        if predict_image(roop.globals.input_path):
            update_status('Processing image halted: NSFW detected!')
            destroy()

    shutil.copy2(roop.globals.input_path, roop.globals.output_path)

    # process image

    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
        update_status('Processing...', frame_processor.NAME)
        frame_processor.process_image(roop.globals.replacement_path, roop.globals.output_path, roop.globals.output_path)
        frame_processor.post_process()

    # validate image

    if is_image(roop.globals.input_path):
        update_status('Processing to image succeed!')
    else:
        update_status('Processing to image failed!')

    return


def process_video() -> None:
    # not safe for work check

    if not roop.globals.allow_nsfw:
        update_status('NSFW check...')
        if predict_video(roop.globals.input_path):
            update_status('Processing video halted: NSFW detected!')
            destroy()

    if not roop.globals.reprocess_frames and not roop.globals.render_only:
        update_status('Creating temporary directory...')
        create_temp_directory(roop.globals.input_path)

        # extract frames

        if roop.globals.keep_fps:
            fps = detect_fps(roop.globals.input_path)
            update_status(f'Extracting frames with {fps} FPS...')
            extract_frames(roop.globals.input_path, fps)
        else:
            update_status('Extracting frames with 30 FPS...')
            extract_frames(roop.globals.input_path)
    else:
        update_status('Checking for frames to reprocess and/or render...')
        temp_directory_path = get_temp_directory_path(roop.globals.input_path)

        # frames are expected to exist

        if not os.path.isdir(temp_directory_path) or not os.listdir(temp_directory_path):
            update_status('Processing video halted: did not find frames to reprocess and/or render')
            destroy()

    # process frame

    sorted_frame_file_paths = get_sorted_frame_file_paths(roop.globals.input_path)

    update_status(f'Processing frames from: {sorted_frame_file_paths}')

    if not sorted_frame_file_paths:
        update_status('Frames not found...')
        return

    update_status(f'render only: {roop.globals.render_only}')

    if not roop.globals.render_only:
        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
            update_status('Progressing...', frame_processor.NAME)
            frame_processor.process_video(roop.globals.replacement_path, sorted_frame_file_paths)
            frame_processor.post_process()

    # create video

    if not roop.globals.skip_video:
        if roop.globals.keep_fps:
            fps = detect_fps(roop.globals.input_path)
            update_status(f'Creating video with {fps} FPS...')
            create_video(roop.globals.input_path, fps)
        else:
            update_status('Creating video with 30 FPS...')
            create_video(roop.globals.input_path)

        # handle audio

        if roop.globals.skip_audio:
            move_temp_file(roop.globals.input_path, roop.globals.output_path)
            update_status('Skipping audio...')
        else:
            if roop.globals.keep_fps:
                update_status('Restoring audio...')
            else:
                update_status('Restoring audio might cause issues as fps are not kept...')

            restore_audio(roop.globals.input_path, roop.globals.output_path)

    # clean temp

    update_status('Cleaning temporary resources...')

    clean_temp_directory(roop.globals.input_path)

    # validate video

    if is_video(roop.globals.input_path):
        update_status('Processing to video succeed!')
    else:
        update_status('Processing to video failed!')


def start() -> None:
    if not roop.globals.render_only:
        for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
            if not frame_processor.pre_start():
                return

    if has_image_extension(roop.globals.input_path):
        process_image()
        return

    process_video()


def destroy() -> None:
    if roop.globals.input_path:
        clean_temp_directory(roop.globals.input_path)

    print()
    print('Ended: ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    print()

    sys.exit()


def run() -> None:
    print()
    print('Started: ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    print()

    parse_args()

    if not pre_check():
        return

    for frame_processor in get_frame_processors_modules(roop.globals.frame_processors):
        if not frame_processor.pre_check():
            return

    limit_resources()

    if roop.globals.headless:
        start()
    else:
        window = ui.init(start, destroy)
        window.mainloop()

    print()
    print('Ended: ' + strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    print()
