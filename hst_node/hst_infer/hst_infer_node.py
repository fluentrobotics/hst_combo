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
from std_msgs.msg import Int32MultiArray, Float32MultiArray, Header, ColorRGBA, MultiArrayDimension, Float32, Int32
from geometry_msgs.msg import PoseArray, Pose, Quaternion, Point, PointStamped, Transform, TransformStamped
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from skeleton_interfaces.msg import MultiHumanSkeleton, HumanSkeleton, Predictions
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

import warnings

import time

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
        self.robot_positions = []
        self.human_positions = {}
        self.agent_poses = tf.zeros((1, MAX_AGENT_NUM, WINDOW_LENGTH, 2))
        self.agent_orientations = tf.zeros((1, MAX_AGENT_NUM, WINDOW_LENGTH, 1))

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
        if SOCIAL_CONTROLLER:
            self._prediction_pub = self.create_publisher(
                Predictions, PREDICTION_TOPIC, 5
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

        self.time = time.time()
        self.mocap_time = time.time()
        self.last_mocap_pose = None
        self.prediction_callback_counter = 0

        self.timer = self.create_timer(0.01, self.timer_callback)

    def timer_callback(self) -> None:
        for i in range(ACTIVE_AGENT_NUM):
            try:
                pose = self._get_human_motion_capture_pose(HUMAN_FRAME + "_" + str(i+1), 'map')
                if i in self.human_positions:
                    self.human_positions[i].append(pose)
                else:
                    self.human_positions[i] = [pose]
            except:
                self.human_positions[i] = []
                

    
    def _skeleton_callback(self, msg: MultiHumanSkeleton):
        start_time = time.time()

        self._get_start_time(header=msg.header)
        # t2 = self.get_clock().now()

        ### human position, human skeleton
        self.skeleton_databuffer.receive_msg(msg)
        keypointATKD, human_pos_ATD, keypoint_mask_ATK = self.skeleton_databuffer.get_data_array()
        A,T,K,D = keypointATKD.shape

        current_human_id_set = self.skeleton_databuffer.get_current_multihumanID_set()
        print("CURRENT HUMAN ID SET: ", current_human_id_set)
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

            t, r = self.tf2_array_transformation(source_frame=STRETCH_BASE_FRAME, target_frame='map')
            try:
                # np dot ##########
                keypoint_map = r @ keypointATKD_stretch[...,np.newaxis]
                human_pos_map = r @ human_pos_ATD_stretch[...,np.newaxis]
                keypointATKD_map = np.squeeze(keypoint_map, axis=-1) + t
                human_pos_ATD_map = np.squeeze(human_pos_map, axis=-1) + t
            except:
                # einsum ##########
                # [3,3] @ [3,1] = [3,1] but einsum is slower
                keypointATKD_map = np.einsum("ji,...i->...j", r, keypointATKD_stretch) + t
                human_pos_ATD_map = np.einsum("ji,...i->...j", r, human_pos_ATD_stretch) + t

            #print(human_pos_ATD_stretch.shape, human_pos_ATD_stretch[0])
            #print(human_pos_ATD_map.shape, human_pos_ATD_map[0])

            keypointATKD_map[:,HISTORY_LENGTH+1:,:,:] = np.zeros_like(keypointATKD_map[:,HISTORY_LENGTH+1:,:,:])
            human_pos_ATD_map[:,HISTORY_LENGTH+1:,:] = np.zeros_like(human_pos_ATD_map[:,HISTORY_LENGTH+1:,:])

            if self.prediction_callback_counter < 6:
                keypointATKD_map[:,:HISTORY_LENGTH-self.prediction_callback_counter,:,:] = np.zeros_like(keypointATKD_map[:,:HISTORY_LENGTH-self.prediction_callback_counter,:,:])
                human_pos_ATD_map[:,:HISTORY_LENGTH-self.prediction_callback_counter,:] = np.zeros_like(human_pos_ATD_map[:,:HISTORY_LENGTH-self.prediction_callback_counter,:])

                keypointATKD_stretch[:,:HISTORY_LENGTH-self.prediction_callback_counter,:,:] = np.zeros_like(keypointATKD_stretch[:,:HISTORY_LENGTH-self.prediction_callback_counter,:,:])
                human_pos_ATD_stretch[:,:HISTORY_LENGTH-self.prediction_callback_counter,:] = np.zeros_like(human_pos_ATD_stretch[:,:HISTORY_LENGTH-self.prediction_callback_counter,:])
            self.prediction_callback_counter = self.prediction_callback_counter + 1


            #print("MAP AFTER ZEROING: ", human_pos_ATD_map[0])

            if EVALUATION_NODE:
                human_t = {}
                agent_position_map_np = np.zeros((1, MAX_AGENT_NUM, WINDOW_LENGTH, 2))
                agent_orientation_map_np = np.zeros((1, MAX_AGENT_NUM, WINDOW_LENGTH, 1))
                # get human TF at the very first time
                # for i in current_human_id_set:
                #     #try:
                #     if EGOCENTRIC:
                #         human_t[i] = self._get_human_motion_capture_pose(HUMAN_FRAME + "_" + str(i), 'base_link')
                #     else:
                #         human_t[i] = self._get_human_motion_capture_pose(HUMAN_FRAME + "_" + str(i), 'map')
                #     #print("TIME SINCE LAST MOCAP: ", time.time() - self.mocap_time)
                #     #print("POSE: ", human_t[i])
                #     self.mocap_time = time.time()
                #     if self.last_mocap_pose is not None:
                #         print("LAST MOCAP POSE: ", self.last_mocap_pose)
                #     self.last_mocap_pose = human_t[i]
                #     if i not in self.human_positions:
                #         self.human_positions[i] = [human_t[i]]
                #     else:
                #         self.human_positions[i].append(human_t[i])
                tracked_agents = []
                for i in range(ACTIVE_AGENT_NUM):
                    # agent_position_map_np[:,i,HISTORY_LENGTH,:] = np.array(human_t[i][:2])
                    # agent_orientation_map_np[:,i,HISTORY_LENGTH,:] = np.array(human_t[i][2])
                    # print("REACHED END OF TRY")
                    if len(self.human_positions[i]) > 0:
                        agent_position_map_np[:,i,HISTORY_LENGTH,:] = np.array(self.human_positions[i][-1][:2])
                        agent_orientation_map_np[:,i,HISTORY_LENGTH,:] = np.array(self.human_positions[i][-1][2])
                        tracked_agents.append(i)
                        human_t[i] = self.human_positions[i][-1]
                    #print("AGENT POSITION MAP NP: ", agent_position_map_np)
                    #except:
                    #    human_t[i] = np.nan

            ### robot position
            robot_pos_TD = np.zeros((HISTORY_LENGTH, DIM_XYZ))

            #print("KEPOINT MASK ATK: ", keypoint_mask_ATK.shape)

            # To HST tensor input
            agent_position = tf.convert_to_tensor(human_pos_ATD_stretch[np.newaxis,...,:2])     # 2D position
            print("AGENT POSITION HST SHAPE ", agent_position.shape)
            agent_keypoint = tf.convert_to_tensor(
                keypoints.human_keypoints.map_yolo_to_hst_batch(
                    keypoints_ATKD= keypointATKD_stretch,
                    keypoint_mask_ATK= keypoint_mask_ATK,
                    keypoint_center_ATD= human_pos_ATD_stretch,)
            )

            if MOTION_CAPTURE_HISTORY:
                agent_position_map_np[:,:,:HISTORY_LENGTH,:] = self.agent_poses[:,:,1:HISTORY_LENGTH+1,:] #Could be causing problems
                agent_orientation_map_np[:,:,:HISTORY_LENGTH,:] = self.agent_orientations[:,:,1:HISTORY_LENGTH+1,:]
                #print("AGENT POSITION MAP NP: ", agent_position_map_np)
                self.agent_poses = agent_position_map_np
                self.agent_orientations = agent_orientation_map_np
                agent_position_np_offset = agent_position_map_np.copy()
                #agent_position_np_offset[:,0,:HISTORY_LENGTH+1,0] = agent_position_map_np[:,0,:HISTORY_LENGTH+1,0]
                agent_position_map = tf.convert_to_tensor(agent_position_np_offset)
                agent_orientation_map = tf.convert_to_tensor(agent_orientation_map_np)
            else:
                agent_position_map = tf.convert_to_tensor(human_pos_ATD_map[np.newaxis,...,:2])     # 2D position
            #print("AGENT MAP POS SIZE: ", agent_position_map.shape)
            agent_keypoint_map = tf.convert_to_tensor(
                keypoints.human_keypoints.map_yolo_to_hst_batch(
                    keypoints_ATKD= keypointATKD_map,
                    keypoint_mask_ATK= keypoint_mask_ATK,
                    keypoint_center_ATD= human_pos_ATD_map,)
            )

            #print("HUMAN POS ATD MAP SHAPE: ", human_pos_ATD_map.shape)
            print("AGENT POSITION MAP AFTER: ", agent_position_map)
            #print("AGENT ORIENTATION MAP: ", agent_orientation_map)

            human_pos_ATD_map = agent_position_map_np.squeeze()

            #print("AGENT HISTORY MAP + 5: ", agent_position_map)

            # no agent orientation data
            agent_orientation = tf.convert_to_tensor(np.full((1,1,hst_config.hst_dataset_param.num_steps,1),
                                                            np.nan, dtype=float))
            # TODO: robot remains static
            self.robot_positions.append(t)
            robot_position = np.array(self.robot_positions[len(self.robot_positions)-hst_config.hst_dataset_param.num_steps:])
            robot_position = tf.convert_to_tensor(np.expand_dims(robot_position, 0))
            #print("SELF ROBOT POSITIONS SHAPE: ", robot_position.shape)
            #print("AGENT POSITION: ", agent_position_map[0][0])
            #print("AGENT KEYPOINT: ", agent_keypoint_map[0][0])

            # input dict
            # agents/keypoints: [B,A,T,33*3]
            # agents/position: [B,A,T,2]
            # agents/orientation: [1,1,T,1] useless
            # robot/position: [B,T,3]
            input_batch = {
                'agents/keypoints': agent_keypoint_map,
                'agents/position': agent_position_map,
                'agents/orientation': agent_orientation_map,
                'robot/position': robot_position,
                }
            
            robot_position_stretch = tf.convert_to_tensor(np.zeros((1,hst_config.hst_dataset_param.num_steps,3)))
            input_batch_stretch = {
                'agents/keypoints': agent_keypoint,
                'agents/position': agent_position,
                'agents/orientation': agent_orientation,
                'robot/position': robot_position_stretch,
                }
            #logger.debug(f'shape:\n keypoints {agent_keypoint.shape}, position {agent_position.shape}, {robot_position.shape}')

            inference_start_time = time.time()
            full_pred, output_batch = self.model(input_batch, training=False)
            full_pred_stretch, output_batch_stretch = self.model(input_batch_stretch)
            agent_position_pred = full_pred['agents/position']
            agent_position_logits = full_pred['mixture_logits']
            agent_position_pred_stretch = full_pred_stretch['agents/position']
            agent_position_logits_stretch = full_pred_stretch['mixture_logits']
            # logger.debug(f"{type(agent_position_pred)}, {agent_position_pred.shape}")

            app_np = tf.squeeze(agent_position_pred).numpy()
            apl_np = tf.squeeze(agent_position_logits).numpy()

            #print("APP NP: ", app_np[0,:,0,:])

            app_np_stretch = tf.squeeze(agent_position_pred_stretch).numpy()
            apl_np_stretch = tf.squeeze(agent_position_logits_stretch).numpy()

            try:
                agent_position_pred = agent_position_pred.numpy()
                agent_position_logits = np.squeeze(
                    agent_position_logits_stretch.numpy()
                )
            except:
                logger.error(f"cannot convert agent position into numpy")

            try:
                agent_position_pred_stretch = agent_position_pred_stretch.numpy()
                agent_position_logits_stretch = np.squeeze(
                    agent_position_logits_stretch.numpy()
                )
            except:
                logger.error(f"cannot convert agent position into numpy")

            assert agent_position_logits.shape == (PREDICTION_MODES_NUM,) ,"Incorrect prediction modes"
            agent_position_prob = np.exp(agent_position_logits) / sum(np.exp(agent_position_logits))
            agent_position_prob_stretch = np.exp(agent_position_logits_stretch) / sum(np.exp(agent_position_logits_stretch))
            # logger.info(f"\nMode weights: {agent_position_prob}")

            if RVIZ_HST:
                markers_list = list()
                print("VIS")
                if EGOCENTRIC:
                    multi_human_pos_ATMD = np.squeeze(agent_position_pred_stretch, axis=0)       # remove batch 1
                else:
                    multi_human_pos_ATMD = np.squeeze(agent_position_pred, axis=0)       # remove batch 1
                multi_human_mask_AT = np.any(keypoint_mask_ATK, axis=-1)
                # logger.debug(f"{multi_human_pos_ATMD.shape}")
                assert multi_human_pos_ATMD.shape == (A,T,hst_config.hst_model_param.num_modes,2), \
                    f"human position shape {multi_human_pos_ATMD.shape} should be {(A,T,hst_config.hst_model_param.num_modes,99)}"
                assert multi_human_mask_AT.shape == (A,T)

                markers_list.append(
                    add_multihuman_current_pos_markers(
                        multi_human_history_pos_ATD=human_pos_ATD_map,
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
                predarray = Float32MultiArray()
                #print("LOGITS: ", apl_np)
                #print("PREDICTION 1: ", app_np[0,:,0,:])
                #print("PREDICTION 2: ", app_np[1,:,0,:])
                #print("PREDICTION 1 STRETCH: ", app_np_stretch[0,:,0,:])
                #print("PREDICTION 2 STRETCH: ", app_np_stretch[1,:,0,:])
                logitarray = Float32MultiArray()
                if EGOCENTRIC:
                    predarray.data = app_np_stretch.flatten().tolist()
                    logitarray.data = apl_np_stretch.flatten().tolist()
                else:
                    predarray.data = app_np.flatten().tolist()
                    logitarray.data = apl_np.flatten().tolist()
                prediction_msg = Predictions()
                prediction_msg.predictions = predarray
                prediction_msg.logits = logitarray

                num_agents_int = Int32()
                num_agents_int.data = len(current_human_id_set)
                prediction_msg.num_agents = num_agents_int

                max_agents_int = Int32()
                max_agents_int.data = MAX_AGENT_NUM
                prediction_msg.max_agents = max_agents_int

                history_length_int = Int32()
                history_length_int.data = HISTORY_LENGTH
                prediction_msg.history_length = history_length_int

                window_length_int = Int32()
                window_length_int.data = WINDOW_LENGTH
                prediction_msg.window_length = window_length_int

                timestep_float = Float32()
                timestep_float.data = hst_config.hst_model_param.timestep
                prediction_msg.timestep = timestep_float

                self._prediction_pub.publish(prediction_msg)
                current_time = time.time()
                print("TIME: ", current_time - self.time)
                self.time = current_time
                #print("TOTAL RUNTIME: ", time.time() - start_time)


                if EVALUATION_NODE:
                    get_hst_infer_latency(self, msg)
                    # save pickles of skeletons, traj prediction, mocap
                    pickle_file_path = PICKLE_DIR_PATH / "evaluation_data_multi.pkl"
                    if EGOCENTRIC:
                        human_pos_save = human_pos_ATD_stretch
                    else:
                        human_pos_save = human_pos_ATD_map
                    #print("HUMAN POS SAVE: ", human_pos_save)
                    print("HUMAN T: ", human_t)
                    data_to_save = {
                        "human_pos_ground_true_ATD": human_pos_save,
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

    warnings.simplefilter("ignore")

if __name__ == "__main__":
    main()