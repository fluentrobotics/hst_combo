#! /usr/bin/env python

import os
import threading
import argparse
import cv2
import pickle
import numpy as np
from typing import Optional, TypedDict
from pathlib import Path
import tensorflow as tf

# import rosbag
import rclpy
import rclpy.duration
import rclpy.time
from rclpy.node import Node
# import message_filters

import tf2_ros
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_geometry_msgs import do_transform_point

# Message type
from std_msgs.msg import Int32MultiArray, Float32MultiArray, Header, ColorRGBA, MultiArrayDimension
from geometry_msgs.msg import PoseArray, Pose, Quaternion, Point, PointStamped, Transform, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from skeleton_interfaces.msg import MultiHumanSkeleton, HumanSkeleton
from cv_bridge import CvBridge

# inference pipeline
from hst_infer.data_buffer import skeleton_buffer
from hst_infer.utils.logger import logger, get_hst_infer_latency
from hst_infer.utils.run_once import run_once
from hst_infer.utils.transform import transformstamped_to_tr
from hst_infer.utils.rviz2_marker import add_multihuman_future_pos_markers, add_multihuman_current_pos_markers, delete_multihuman_pos_markers
from hst_infer.node_config import *
from hst_infer.human_scene_transformer.model import model as hst_model
from hst_infer.human_scene_transformer.config import hst_config
# HST model
from hst_infer.human_scene_transformer import infer
from hst_infer.utils import keypoints
# from human_scene_transformer import run

class HST_infer_node(Node):
    def __init__(self,
                 ):
        
        super().__init__(HST_INFER_NODE)
        self.skeleton_databuffer = skeleton_buffer(history_len=hst_config.hst_model_param.num_history_steps)

        # Subscriber #######################################
        ### human skeleton msg 
        self._skeleton_sub = self.create_subscription(
            MultiHumanSkeleton, MULTI_HUMAN_SKELETON_TOPIC, self._skeleton_callback, 5
        )
        ### tf2
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        ### hst checkpoint
        self.model: hst_model.HumanTrajectorySceneTransformer = infer.init_model()
        _checkpoint_path = HST_CKPT_PATH.as_posix()
        _param_path = NETWORK_PARAM_PATH.as_posix()
        checkpoint_mngr = tf.train.Checkpoint(model=self.model)
        checkpoint_mngr.restore(_checkpoint_path).assert_existing_objects_matched()

        #  Publisher ######################################
        if RVIZ_HST:
            self._traj_marker_pub = self.create_publisher(
                MarkerArray, RVIZ_HST_TOPIC, 5
            )
        # logger
        logger.info(
            f"\nNode Name: {HST_INFER_NODE} \n \
            receive message from {MULTI_HUMAN_SKELETON_TOPIC} \n \
            HST checkpoint: {_checkpoint_path} \n \
            GPU: {tf.config.list_physical_devices('GPU')} \
            Tensorflow Eager Mode: {tf.executing_eagerly()} \
        ")
        
        # counter
        self.counter: int = 0

    
    def _skeleton_callback(self, msg: MultiHumanSkeleton):

        self._get_start_time(header=msg.header)
        # t2 = self.get_clock().now()

        ### human position, human skeleton
        self.skeleton_databuffer.receive_msg(msg)
        keypointATKD, human_pos_ATD, keypoint_mask_ATK = self.skeleton_databuffer.get_data_array()
        A,T,K,D = keypointATKD.shape

        current_human_id_set = self.skeleton_databuffer.get_current_multihumanID_set()
        if len(current_human_id_set) == 0:
            # TODO: delete all rviz
            if RVIZ_HST:
                # logger.debug(f"Empty human set{current_human_id_set}")
                markerarray = delete_multihuman_pos_markers(frame_id=STRETCH_BASE_FRAME,)
                self._traj_marker_pub.publish(markerarray)

        else:

            if msg.header.frame_id != STRETCH_BASE_FRAME:
                t, r = self.tf2_array_transformation(source_frame=CAMERA_FRAME, target_frame=STRETCH_BASE_FRAME)
                logger.warning(f"The frame {msg.header.frame_id} is not {STRETCH_BASE_FRAME}, please check the skeleton extractor")
                try:
                    # np dot ##########
                    keypoint_stretch = r @ keypointATKD[...,np.newaxis]
                    human_pos_stretch = r @ human_pos_ATD[...,np.newaxis]
                    keypointATKD_stretch = np.squeeze(keypoint_stretch, axis=-1) + t
                    human_pos_ATD_stretch = np.squeeze(human_pos_stretch, axis=-1) + t
                except:
                    # einsum ##########
                    # [3,3] @ [3,1] = [3,1] but einsum is slower
                    keypointATKD_stretch = np.einsum("ji,...i->...j", r, keypointATKD) + t
                    human_pos_ATD_stretch = np.einsum("ji,...i->...j", r, human_pos_ATD) + t
            
            else:
                human_pos_ATD_stretch = human_pos_ATD
                keypointATKD_stretch = keypointATKD

            if EVALUATION_NODE:
                # get human TF at the very first time
                try:
                    human_t = self._get_human_motion_capture_pose(HUMAN_FRAME, STRETCH_BASE_FRAME)
                except:
                    human_t = np.nan


            ### robot position
            robot_pos_TD = np.zeros((HISTORY_LENGTH, DIM_XYZ))

            # To HST tensor input
            agent_position = tf.convert_to_tensor(human_pos_ATD_stretch[np.newaxis,...,:2])     # 2D position
            agent_keypoint = tf.convert_to_tensor(
                keypoints.human_keypoints.map_yolo_to_hst_batch(
                    keypoints_ATKD= keypointATKD_stretch,
                    keypoint_mask_ATK= keypoint_mask_ATK,
                    keypoint_center_ATD= human_pos_ATD_stretch,)
            )

            # no agent orientation data
            agent_orientation = tf.convert_to_tensor(np.full((1,1,hst_config.hst_dataset_param.num_steps,1),
                                                            np.nan, dtype=float))
            # TODO: robot remains static
            robot_position = tf.convert_to_tensor(np.zeros((1,hst_config.hst_dataset_param.num_steps,3)))

            # input dict
            # agents/keypoints: [B,A,T,33*3]
            # agents/position: [B,A,T,2]
            # agents/orientation: [1,1,T,1] useless
            # robot/position: [B,T,3]
            input_batch = {
                'agents/keypoints': agent_keypoint,
                'agents/position': agent_position,
                'agents/orientation': agent_orientation,
                'robot/position': robot_position,
                }
            # logger.debug(f'shape:\n keypoints {agent_keypoint.shape}, position {agent_position.shape}, {robot_position.shape}')

            full_pred, output_batch = self.model(input_batch, training=False)
            agent_position_pred = full_pred['agents/position']
            agent_position_logits = full_pred['mixture_logits']
            # logger.debug(f"{type(agent_position_pred)}, {agent_position_pred.shape}")

            try:
                agent_position_pred = agent_position_pred.numpy()
                agent_position_logits = np.squeeze(
                    agent_position_logits.numpy()
                )
            except:
                logger.error(f"cannot convert agent position into numpy")

            assert agent_position_logits.shape == (PREDICTION_MODES_NUM,) ,"Incorrect prediction modes"
            agent_position_prob = np.exp(agent_position_logits) / sum(np.exp(agent_position_logits))
            # logger.info(f"\nMode weights: {agent_position_prob}")

            if RVIZ_HST:
                markers_list = list()

                multi_human_pos_ATMD = np.squeeze(agent_position_pred, axis=0)       # remove batch 1
                multi_human_mask_AT = np.any(keypoint_mask_ATK, axis=-1)
                # logger.debug(f"{multi_human_pos_ATMD.shape}")
                assert multi_human_pos_ATMD.shape == (A,T,hst_config.hst_model_param.num_modes,2), \
                    f"human position shape {multi_human_pos_ATMD.shape} should be {(A,T,hst_config.hst_model_param.num_modes,99)}"
                assert multi_human_mask_AT.shape == (A,T)

                markers_list.append(
                    add_multihuman_current_pos_markers(
                        multi_human_history_pos_ATD=human_pos_ATD_stretch,
                        multi_human_mask_AT=multi_human_mask_AT,
                        present_idx=HISTORY_LENGTH,
                        frame_id=STRETCH_BASE_FRAME,
                        ns=HST_INFER_NODE,
                    )
                )

                markers_list.append(
                    add_multihuman_future_pos_markers(
                        multi_human_pos_ATMD=multi_human_pos_ATMD,
                        multi_human_mask_AT=multi_human_mask_AT,
                        modes_prob=agent_position_prob,
                        present_idx=HISTORY_LENGTH,
                        frame_id=STRETCH_BASE_FRAME,
                        ns=HST_INFER_NODE,
                        )
                )

                markerarray = MarkerArray()
                markerarray.markers = markers_list
                self._traj_marker_pub.publish(markerarray)


        if EVALUATION_NODE:
            get_hst_infer_latency(self, msg)
            # save pickles of skeletons, traj prediction, mocap
            pickle_file_path = PICKLE_DIR_PATH / "evaluation_data.pkl"
            data_to_save = {
                "human_pos_ground_true_ATD": human_pos_ATD_stretch,
                "human_pos_mask_AT": multi_human_mask_AT,
                "human_pos_HST_ATMD": multi_human_pos_ATMD,
                "HST_mode_weights": agent_position_prob,
                "human_T": human_t,
                "human_id_set_in_window": self.skeleton_databuffer.humanID_in_window,
                "human_id_2_array_idx": self.skeleton_databuffer.id2idx_in_window,
            }

            if self.counter == 0:
                write_mod = 'wb'
            else:
                write_mod = 'ab'

            with open(pickle_file_path.as_posix(), write_mod) as pickle_hd:
                pickle.dump(data_to_save, pickle_hd)
                logger.success(f"Dump pickle at step {self.counter}")
                self.counter += 1
            # print("then pickle dump")
            # get_hst_infer_latency(self, msg)
            
            # debug ###
            # logger.debug(f"Buffer depth:{len(self.skeleton_databuffer)}\n")
            # logger.debug(f"\nget_image:{msg.header.stamp}\nreceive_skeleton:{t2}\nafter_databuffer:{self.get_clock().now()}")
            # logger.debug(f"keypointATKD nonzero:{np.nonzero(keypointATKD)}\n \
            #       human position nonzero:{np.nonzero(human_pos_ATD)}\n \
            #       mask sparse:{np.nonzero(keypoint_mask_ATK)}\n \
            #       ")
            # exit()
            ##



    def tf2_array_transformation(self, source_frame: str, target_frame: str) -> tuple[np.ndarray, np.ndarray]:
        """
        return translation(3,), rotation(3,3)
        """
        try:
            # P^b(target) = T_a^b @ P^a(source), T_a^b means T of a wrt b frame, ^super: wrt, _sub: processing
            # fun lookup_transform is designated to convert the pose of an object from source frame to target frame
            # so what we get is T_a^b, T of a wrt b, which is also the pose of a(source) relative to b(target)
            
            # NOTE: lookup_transfrom hint is not accurate, to clearify and simplify it
            # target_frame = TransfromStamped.header.frame_id, source_frame = TransformStamped.child_frame_id
            # in conclusion, return T and R of source_frame in target_frame
            transformation = self._tf_buffer.lookup_transform(target_frame=target_frame, 
                                                              source_frame=source_frame,
                                                              time=rclpy.time.Time(seconds=0, nanoseconds=0),
                                                              timeout=rclpy.duration.Duration(seconds=0, nanoseconds=int(0.5e9)),
                                                              )
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            logger.exception(f'Unable to find the transformation from {source_frame} to {target_frame}')

        t,r = transformstamped_to_tr(transformation)
        return t,r


    def _get_human_motion_capture_pose(self, source_frame: str = HUMAN_FRAME, target_frame: str = STRETCH_BASE_FRAME) -> np.ndarray:
        """
        (default) return translation of motion capture w.r.t robot
        """
        t, r = self.tf2_array_transformation(source_frame=source_frame, target_frame=target_frame)
        return t



    @run_once
    def _get_start_time(self, header: Header):
        self._start_time = header.stamp
        logger.info(f"hst started at {self._start_time}")



def main(args=None):
    
    rclpy.init(args=args)
    node = HST_infer_node()
    rclpy.spin(node)


if __name__ == "__main__":
    main()