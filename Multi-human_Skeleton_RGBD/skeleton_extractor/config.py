import numpy as np
import cv2
from skeleton_extractor.utils import get_project_dir_path

# SKELETAL NODE config #############################################################

# SUB topics
COLOR_FRAME_TOPIC = '/camera/color/image_raw'
COLOR_COMPRESSED_FRAME_TOPIC = '/camera/color/image_raw/compressed'
DEPTH_ALIGNED_TOPIC = '/camera/aligned_depth_to_color/image_raw'
DEPTH_ALIGNED_COMPRESSED_TOPIC = '/camera/aligned_depth_to_color/image_raw/compressed'
CAMERA_INFO_TOPIC = '/camera/color/camera_info'
# PUB topic
## Skeleton Data in skeleton_interfaces
MULTI_HUMAN_SKELETON_TOPIC = 'skeleton/data/multi_human'
## RViz Markers
RVIZ_IMG2D_SKELETON_TOPIC = '/skeleton/vis/keypoints_2d_img'
RVIZ_MARKER3D_SKELETON_TOPIC = '/skeleton/vis/keypoints_3d_makers'
PUB_FREQ:float = 20

CAMERA_FRAME = "camera_color_optical_frame"
STRETCH_BASE_FRAME = "base_link"
# DATA TYPE
ID_TYPE = np.int32

# Node PARAMETERS
SKELETON_NODE = "skeleton"
COMPRESSED_TOPICS = {'rgb': True, 'depth': False}
MAX_MISSING = 15
SKELETAL_LINE_PAIRS_LIST = [(4,2),(2,0),(0,1),(1,3),
                            (10,8),(8,6),(6,5),(5,7),(7,9),
                            (6,12),(12,14),(14,16),(5,11),(11,13),(13,15),(12,11)]
SKELETAL2BODY = np.array(["Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
                          "Left Shoulder", " Right SHoulder", "Left Elbow", "Right Elbow",
                          "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
                          "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"], dtype=str)
REPLACE_TIMESTAMP: bool = True  # use ROS2 current timestamp istead of rosbag timestamp

# Pre-trained Model
POSE_MODEL = 'yolov8m-pose.pt'

# Parameters
CAMERA_INTRINSIC = [906.7041625976562, 0.0, 653.4981689453125, 0.0, 906.7589111328125, 375.4635009765625, 0.0, 0.0, 1.0]
CAMERA_ROTATE = cv2.ROTATE_90_CLOCKWISE

# RViz Visualization (code errors)
RVIZ_VIS = True
HUMAN_MARKER_ID_VOL = 10
KEYPOINT_ID_OFFSET = 0
LINE_ID_OFFSET = 1
GEO_CENTER_ID_OFFSET = 2


# PATH
PROJECT_SRC_PATH = get_project_dir_path()
DATA_DIR_PATH = PROJECT_SRC_PATH / "data"
###################################################################################

# TODO: Cutomized Parameters
# TEST_NAME = "test_Minimal_True"
TEST_NAME = "Multi_Human2"
SAVE_DATA = False
SAVE_YOLO_IMG = False
DEMO_TYPE = 'gif'

# Filters
USE_KALMAN = True
MINIMAL_FILTER = True
OUTLIER_FILTER = True

# Plot and Video
SELECTED_VIDEO_REGION = True
VIDEO_START = 70
VIDEO_END = 170