from typing import List, Optional

input_path: Optional[str] = None
replacement_path: Optional[str] = None
output_path: Optional[str] = None
headless: Optional[bool] = None
frame_processors: List[str] = []
allow_nsfw: Optional[bool] = None
keep_fps: Optional[bool] = None
keep_frames: Optional[bool] = None
reprocess_frames: Optional[bool] = None
render_only: Optional[bool] = None
skip_video: Optional[bool] = None
skip_audio: Optional[bool] = None
many_faces: Optional[bool] = None
reference_face_position: Optional[int] = None
reference_frame_number: Optional[int] = None
similar_face_distance: Optional[float] = None
temp_frame_format: Optional[str] = None
temp_frame_quality: Optional[int] = None
output_video_encoder: Optional[str] = None
output_video_lossiness: Optional[int] = None
max_memory: Optional[int] = None
execution_providers: List[str] = []
execution_threads: Optional[int] = None

log_level: str = 'error'
