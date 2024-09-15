from pathlib import Path

from hst_infer.utils import get_path

# node / topics
HST_INFER_NODE = "hst"
MULTI_HUMAN_SKELETON_TOPIC = 'skeleton/data/multi_human'

SRC_PROJECT_PATH = get_path.get_src_project_dir_path("src/hst_combo/hst_node")

# frame / coordinate
CAMERA_FRAME = "camera_color_optical_frame"
STRETCH_BASE_FRAME = "base_link"

# rviz
RVIZ_HST: bool = True
RVIZ_HST_TOPIC = HST_INFER_NODE + "/vis/human_trajectory"
CURRENT_COLOR = dict(r=1.0, g=0.5, b=0.5, a=1.0)

# social controller
SOCIAL_CONTROLLER: bool = True
PREDICTION_TOPIC = HST_INFER_NODE + '/predictions'

# hst parameters
WINDOW_LENGTH = 20
HISTORY_LENGTH = 7
PREDICTION_MODES_NUM = 20

KEYPOINT_NUM = 17
DIM_XYZ = 3
MAX_AGENT_NUM = 14
ACTIVE_AGENT_NUM = 1

# yolo keypoints to hst keypoints
KEYPOINT_INTERPOLATION = True

# Human Scene Transformer
#NETWORK_PARAM_PATH: Path = SRC_PROJECT_PATH / "hst_net_param_no_keypoints"
NETWORK_PARAM_PATH: Path = SRC_PROJECT_PATH / "pedestrians_zara1"
HST_CKPT_PATH: Path = NETWORK_PARAM_PATH / "ckpts/ckpt-20"
NO_KEYPOINTS: bool = True

# Evaluation
EVALUATION_NODE: bool = True
PICKLE_DIR_PATH: Path = SRC_PROJECT_PATH / "pickle"

## Motion Capture
MOTION_CAPTURE_TF: bool = True
HUMAN_FRAME: str = "human"
EGOCENTRIC: bool = False
MOTION_CAPTURE_HISTORY: bool = True
