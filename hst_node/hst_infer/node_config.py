from pathlib import Path

from hst_infer.utils import get_path

# node / topics
HST_INFER_NODE = "hst"
MULTI_HUMAN_SKELETON_TOPIC = 'skeleton/data/multi_human'

SRC_PROJECT_PATH = get_path.get_src_project_dir_path("src/hst_node")

# frame / coordinate
CAMERA_FRAME = "camera_color_optical_frame"
STRETCH_BASE_FRAME = "base_link"

# rviz
RVIZ_HST: bool = True
RVIZ_HST_TOPIC = HST_INFER_NODE + "/vis/human_trajectory"
CURRENT_COLOR = dict(r=1.0, g=0.5, b=0.5, a=1.0)

# hst parameters
WINDOW_LENGTH = 19
HISTORY_LENGTH = 6
PREDICTION_MODES_NUM = 6

KEYPOINT_NUM = 17
DIM_XYZ = 3
MAX_AGENT_NUM = 12

# yolo keypoints to hst keypoints
KEYPOINT_INTERPOLATION = True

# Human Scene Transformer
NETWORK_PARAM_PATH: Path = SRC_PROJECT_PATH / "hst_net_param"
HST_CKPT_PATH: Path = NETWORK_PARAM_PATH / "ckpts/ckpt-30"

# Evaluation
EVALUATION_NODE: bool = True
PICKLE_DIR_PATH: Path = SRC_PROJECT_PATH / "pickle"

## Motion Capture
MOTION_CAPTURE_TF: bool = False
HUMAN_FRAME: str = "human"