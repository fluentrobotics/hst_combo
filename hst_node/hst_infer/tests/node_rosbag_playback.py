import rclpy
import rclpy.duration
import rclpy.time
from rclpy.node import Node

import tf2_ros
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import do_transform_point

from std_msgs.msg import Int32MultiArray, Float32MultiArray, Header, ColorRGBA, MultiArrayDimension
from geometry_msgs.msg import PoseArray, Pose, Quaternion, Point, PointStamped, Transform, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, PointCloud2

import math
import cv2
import cv_bridge
import time

class read_rosbag(Node):
    def __init__(self, 
                 moniter_list: list[tuple[str, str]] = None,
                 tf1: str = None,
                 tf2: str = None,
                 ):

        super().__init__("read_rosbag_tmp")

        self._sub_compressedimage = self.create_subscription(
            moniter_list[0][1], moniter_list[0][0], self._read_tf, 1
        )

        # tf2
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)
        self.tf1 = tf1
        self.tf2 = tf2
        self.target_frame = "map"

        self.min_dis = float('inf')
        self.prev_time = 0.0

        self.bridge = cv_bridge.CvBridge()


    def _read_tf(self, msg: CompressedImage):
        print("--------------------------")
        start = time.time()
        transformation1 = self._tf_buffer.lookup_transform(
            target_frame=self.target_frame, 
            source_frame=self.tf1,
            time=rclpy.time.Time(seconds=0, nanoseconds=0),
            timeout=rclpy.duration.Duration(seconds=0, nanoseconds=int(0.5e9)),
        )

        transformation2 = self._tf_buffer.lookup_transform(
            target_frame=self.target_frame, 
            source_frame=self.tf2,
            time=rclpy.time.Time(seconds=0, nanoseconds=0),
            timeout=rclpy.duration.Duration(seconds=0, nanoseconds=int(0.5e9)),
        )
        print(f"TF query latency: {time.time()-start}")

        x1,y1,z1 = transformation1.transform.translation.x, transformation1.transform.translation.y, transformation1.transform.translation.z
        x2,y2,z2 = transformation2.transform.translation.x, transformation2.transform.translation.y, transformation2.transform.translation.z

        curr_rgb_time = float(msg.header.stamp.sec)+1e-9*float(msg.header.stamp.nanosec)
        curr_tf1_time = float(transformation1.header.stamp.sec)+1e-9*float(transformation1.header.stamp.nanosec)
        curr_tf2_time = float(transformation2.header.stamp.sec)+1e-9*float(transformation2.header.stamp.nanosec)

        print(f"\
              \nRGBstamp: {curr_rgb_time}, TF1stamp: {curr_tf1_time}, TF2stamp: {curr_tf2_time}\
              \nRGBstamp diff: {curr_rgb_time-self.prev_time}")
        self.prev_time = curr_rgb_time * 1.0
        print(f"block 1 pos: {x1,y1,z1}, {transformation1.child_frame_id} wrt {transformation1.header.frame_id}\
              \nblock 2 pos: {x2,y2,z2}, {transformation2.child_frame_id} wrt {transformation2.header.frame_id}")

        pos1 = [x1,y1,z1]
        pos2 = [x2,y2,z2]
        update = False
        if math.dist(pos1, pos2) < self.min_dis:
            self.min_dis = math.dist(pos1, pos2)
            update = True
            print(f"relative pos: {x1-x2}, {y1-y2}, {z1-z2}, lower distance: {update}, dist: {self.min_dis}")
        else:
            print(f"relative pos: {x1-x2}, {y1-y2}, {z1-z2}, lower distance: {update}")
        # print(f"Relative Pos between tf1 and tf2 in frame map= {x1-x2}, {y1-y2}, {z1-z2}")

        bgr_frame = self.bridge.compressed_imgmsg_to_cv2(msg)
        cv2.imshow("rgb", bgr_frame)
        cv2.waitKey(3)

def main(args=None):
    moniter_list = [
        ("/camera/color/image_raw/compressed", CompressedImage),
        ("/camera/depth/color/points", PointCloud2),
        ("/tf", )
    ]
    tf1 = "block_1"
    tf2 = "block_2"

    rclpy.init(args=args)
    node = read_rosbag(moniter_list, tf1, tf2)
    rclpy.spin(node)


if __name__ == "__main__":
    main()