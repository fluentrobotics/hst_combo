import pickle
import numpy as np
import matplotlib.pyplot as plt

from hst_infer.node_config import *

pickle_file_path = PICKLE_DIR_PATH / "evaluation_data_multi.pkl"

def main():

    counter = 0

    human_real_translation_list = list()
    human_RGBD_translation_list = list()
    human_history_list = list()
    human_HST_res_list = list()
    human_HST_res_list_all_modes = list()
    for i in range(PREDICTION_MODES_NUM):
        human_HST_res_list_all_modes.append([])

    color1_pred = '#FB575D'
    color2_pred = '#15251B'

    color1_history = "#8A5AC2"
    color2_history = '#15251B'

    color1_gt = "#D4CC47"
    color2_gt = "#7C4D8B"

    mode = 7
    plot = "full predictions"

    with open(pickle_file_path.as_posix(), 'rb') as pickle_hd:
        while True:
            try:
                dict_to_read: dict = pickle.load(pickle_hd)
                counter += 1
                
                # print(dict_to_read.keys())
                human_pos_ATD_stretch = dict_to_read["human_pos_ground_true_ATD"]
                multi_human_mask_AT = dict_to_read["human_pos_mask_AT"]
                multi_human_pos_ATMD = dict_to_read["human_pos_HST_ATMD"]
                agent_position_prob = dict_to_read["HST_mode_weights"]
                for key in dict_to_read["human_T"]:
                    print("KEY: ", key)
                human_t_motion_capture = dict_to_read["human_T"][0]
                humanID_in_window = dict_to_read["human_id_set_in_window"]
                id2idx_in_window = dict_to_read["human_id_2_array_idx"]

                print(f"current human ID: {humanID_in_window}")
                print(f"human T from Motion Capture:    {human_t_motion_capture}")
                # print(f"valid human in mat: {np.nonzero(multi_human_mask_AT)}")
                # print(f"nonzero human pose: {np.nonzero(human_pos_ATD_stretch)}")

                if counter <= HISTORY_LENGTH:
                    continue

                # ######## Single Human Condition ##############################################################################
                #####################################################################################################################
                human_idx = id2idx_in_window[humanID_in_window[0]]
                # if np.isnan(human_pos_ATD_stretch[human_idx,HISTORY_LENGTH,:2]).any():
                #     continue
                
                human_real_translation_list.append(human_t_motion_capture[:2])          # list of np.array (2,), finally get (t,2) array

                print("STEP: ", counter)
                print("HUMAN IDX: ", humanID_in_window[0], human_idx)
                human_RGBD_translation_list.append(human_pos_ATD_stretch[human_idx,HISTORY_LENGTH,:2])
                human_history_list.append(human_pos_ATD_stretch[human_idx,:HISTORY_LENGTH,:2])

                print(f"human T from RGBD:              {human_pos_ATD_stretch[human_idx,HISTORY_LENGTH,:2]}")
                print(f"human T diff:                   {human_t_motion_capture[:2] - human_pos_ATD_stretch[human_idx,HISTORY_LENGTH,:2]}")
                # print(f"REAL human motion: {human_real_translation_list}")

                if mode == 7:
                    best_mode_idx = np.argmax(agent_position_prob)
                else:
                    best_mode_idx = mode
                human_pos_predict = multi_human_pos_ATMD[human_idx,:,best_mode_idx,:2].reshape((WINDOW_LENGTH,2))  # (T,2)
                print(human_pos_predict.shape)
                human_HST_res_list.append(human_pos_predict)

                print("ATMD: ", multi_human_pos_ATMD.shape, human_idx)
                for i in range(PREDICTION_MODES_NUM):
                    human_HST_res_list_all_modes[i].append(multi_human_pos_ATMD[human_idx,:,i,:2].reshape((WINDOW_LENGTH,2)))

                # build the time series data
                human_idx_in_window = [id2idx_in_window[humanID] for humanID in humanID_in_window]
                # print(f"current human idx: {human_idx_in_window}")
                print(f"###################################################################\n")

                # 0-4 is history, 5 is present, 6+ is future
                # skip first 5 steps

                #               human id                idx                positions         desired pos      vs GroundTrue
                # humanID_in_window -> id2idx_in_window -> human_pos_HST_ATMD -> get future pos -> human_t_motion_capture

                # 1. skeleton extractor performance
                
                # 2. HST performance


                # break
                # input()

                # ######## Multi Human Condition ##############################################################################
                #####################################################################################################################
                # for h_idx in human_t_motion_capture:
                #     if np.isnan(human_t_motion_capture[h_idx]):
                #         continue
                #     print("HUMAN T MOTION CAPTURE: ", human_t_motion_capture[h_idx])
                #     human_pos = human_t_motion_capture[h_idx][:2]
                #     human_real_translation_list.append(human_pos)          # list of np.array (2,), finally get (t,2) array

                #     human_idx = id2idx_in_window[humanID_in_window[0]]
                #     human_RGBD_translation_list.append(human_pos_ATD_stretch[human_idx,HISTORY_LENGTH,:2])

                #     print(f"human T from RGBD:              {human_pos_ATD_stretch[human_idx,HISTORY_LENGTH,:2]}")
                #     print(f"human T diff:                   {human_t_motion_capture[:2] - human_pos_ATD_stretch[human_idx,HISTORY_LENGTH,:2]}")
                #     # print(f"REAL human motion: {human_real_translation_list}")

                #     best_mode_idx = np.argmax(agent_position_prob)
                #     human_pos_predict = multi_human_pos_ATMD[human_idx,:,best_mode_idx,:2].reshape((19,2))  # (T,2)
                #     print(human_pos_predict.shape)
                #     human_HST_res_list.append(human_pos_predict)

                #     # build the time series data
                #     human_idx_in_window = [id2idx_in_window[humanID] for humanID in humanID_in_window]
                #     # print(f"current human idx: {human_idx_in_window}")
                #     print(f"###################################################################\n")

                # 0-4 is history, 5 is present, 6+ is future
                # skip first 5 steps

                #               human id                idx                positions         desired pos      vs GroundTrue
                # humanID_in_window -> id2idx_in_window -> human_pos_HST_ATMD -> get future pos -> human_t_motion_capture

                # 1. skeleton extractor performance
                
                # 2. HST performance


                # break
                # input()

            except EOFError:
                break
    
    print(f"steps: {counter}")
    steps = counter - HISTORY_LENGTH
    human_HST_res_list = np.array(human_HST_res_list)
    human_HST_res_list_all_modes = np.array(human_HST_res_list_all_modes)
    # print(f"HST Prediction {human_HST_res_list.shape}\n{human_HST_res_list}")
    human_RGBD_translation_list = np.array(human_RGBD_translation_list)
    human_real_translation_list = np.array(human_real_translation_list)
    human_history_list = np.array(human_history_list)
    human_history_list[:,:,0] = human_history_list[:,:,0]
    print("HISTORY SHAPE: ", human_history_list.shape)
    # print(f"RGBD Projection {human_RGBD_translation_list.shape}\n{human_RGBD_translation_list}")

    # Errors ###############################################################################################################
    ################################################################################################################

    ## Motion Capture v.s. RGBD
    mc_rgbd_diff = human_real_translation_list - human_RGBD_translation_list
    mean_error = np.mean(mc_rgbd_diff, axis=0)
    normalized_error = (human_real_translation_list - mean_error) - human_RGBD_translation_list # We consider it as system error
    print(f"Motion Capture vs RGB-D\nMean errors on axis X and Y: {mean_error}")

    # for i in range(38):
        # print("HST RES LIST: ")
        # print(human_HST_res_list[i])
        # print("HUMAN REAL TRANSLATION: ")
        # print(human_real_translation_list[i])

    # for i in range(WINDOW_LENGTH-HISTORY_LENGTH-1):
    #     print("HST RES LIST SHAPE: ", human_HST_res_list.shape)
    #     pred_step = human_HST_res_list[:,HISTORY_LENGTH + 1 + i,:]
    #     print("PRED STEP SHAPE: ", pred_step[:steps-i-1,:].shape, human_real_translation_list[i+1:,:].shape)
    #     diff = np.linalg.norm(pred_step[:steps-i-1,:] - human_real_translation_list[i+1:,:], axis=1)
    #     print("DIFF SHAPE: ", diff.shape)
    #     mean_diff = np.mean(diff)
    #     print(f"Motion capture vs predicted at timestep {i} is {mean_diff}")

    #     best_mode = []
    #     best_diff = []
    #     for s in range(steps-i-1):
    #         min_diff = 99999.9
    #         best_pred = None
    #         for m in range(PREDICTION_MODES_NUM):
    #             pred_step_single = human_HST_res_list_all_modes[m,s,HISTORY_LENGTH + 1 + i,:]
    #             diff = np.linalg.norm(pred_step_single - human_real_translation_list[i+1+s])
    #             if diff < min_diff:
    #                 best_pred = pred_step_single
    #                 min_diff = diff
    #         best_mode.append(best_pred)
    #         best_diff.append(min_diff)
    #     best_mode = np.array(best_mode)
    #     best_diff = np.array(best_diff)
    #     print(f"Motion capture vs BEST MODE predicted at timestep {i} is {np.mean(best_diff)}")

    #     if plot == 'all timesteps':
    #         # print(mc_rgbd_diff.shape)
    #         print("WHT IS I SET: ", i)
    #         fig = plt.figure(figsize=(16,16))
    #         ax1 = fig.add_subplot(2,2,1)
    #         ax1.set_title('Raw Errors')
    #         ax2 = fig.add_subplot(2,2,2)
    #         ax2.set_title('Performance: L2 Errors')
    #         ax1.scatter(np.arange(mc_rgbd_diff.shape[0]), mc_rgbd_diff[:,0], label='err_X', color='r', s=6)
    #         ax1.scatter(np.arange(mc_rgbd_diff.shape[0]), mc_rgbd_diff[:,1], label='err_Y', color='b', s=6)
    #         ax1.legend()
    #         ax2.scatter(np.arange(mc_rgbd_diff.shape[0]), np.linalg.norm(mc_rgbd_diff, axis=1), label='L2 norm', s=6)
    #         ax2.legend()
    #         ax3 = fig.add_subplot(2,2,3)
    #         ax3.set_title(f'Exclude System Error on X:{mean_error[0]} Y:{mean_error[1]}')
    #         ax3.scatter(np.arange(mc_rgbd_diff.shape[0]), normalized_error[:,0], label='err_X with offset', color='r', s=6)
    #         ax3.scatter(np.arange(mc_rgbd_diff.shape[0]), normalized_error[:,1], label='err_Y with offset', color='b', s=6)
    #         ax3.legend()
    #         ax4 = fig.add_subplot(2,2,4)
    #         #ax4.scatter(np.arange(mc_rgbd_diff.shape[0]), np.linalg.norm(normalized_error, axis=1), label='L2 norm with offset', s=6)
    #         print("PRED STEP SHAPE: ", pred_step.shape)
    #         ax4.scatter(pred_step[:steps-i-1,0], pred_step[:steps-i-1,1], color=get_color_gradient(color1_pred, color2_pred, steps-1-i), label='PRED', s=6)
    #         ax4.scatter(human_real_translation_list[i+1:steps+i+1,0], human_real_translation_list[i+1:steps+i+1,1], color=get_color_gradient(color1_gt, color2_gt, steps-1-i), label='MOCAP', s=6)
    #         ax4.set_xlim(-3, 3)
    #         ax4.set_ylim(-6, 6)
    #         ax4.legend()
    #         plt.show()

    if plot == 'full predictions':
        print("HST RES LIST SHAPE: ", human_HST_res_list.shape)
        pred_step = human_HST_res_list[:,HISTORY_LENGTH+1:,:]
        print("PRED STEP SHAPE: ", pred_step.shape, human_real_translation_list.shape)
        # diff = np.linalg.norm(pred_step[:38-i-1,:] - human_real_translation_list[i+1:,:], axis=1)
        # print("DIFF SHAPE: ", diff.shape)
        # mean_diff = np.mean(diff)
        # print(f"Motion capture vs predicted ADE is {mean_diff}")
        for i in range(steps):
            # print(mc_rgbd_diff.shape)
            print("WHT IS I SET: ", i)
            # fig = plt.figure(figsize=(16,16))
            # ax1 = fig.add_subplot(2,2,1)
            # ax1.set_title('Raw Errors')
            # ax2 = fig.add_subplot(2,2,2)
            # ax2.set_title('Performance: L2 Errors')
            # ax1.scatter(np.arange(mc_rgbd_diff.shape[0]), mc_rgbd_diff[:,0], label='err_X', color='r', s=6)
            # ax1.scatter(np.arange(mc_rgbd_diff.shape[0]), mc_rgbd_diff[:,1], label='err_Y', color='b', s=6)
            # ax1.legend()
            # ax2.scatter(np.arange(mc_rgbd_diff.shape[0]), np.linalg.norm(mc_rgbd_diff, axis=1), label='L2 norm', s=6)
            # ax2.legend()
            # ax3 = fig.add_subplot(2,2,3)
            # ax3.set_title(f'Exclude System Error on X:{mean_error[0]} Y:{mean_error[1]}')
            # ax3.scatter(np.arange(mc_rgbd_diff.shape[0]), normalized_error[:,0], label='err_X with offset', color='r', s=6)
            # ax3.scatter(np.arange(mc_rgbd_diff.shape[0]), normalized_error[:,1], label='err_Y with offset', color='b', s=6)
            # ax3.legend()
            
            
            
            # ax4 = fig.add_subplot(2,2,4)
            # #ax4.scatter(np.arange(mc_rgbd_diff.shape[0]), np.linalg.norm(normalized_error, axis=1), label='L2 norm with offset', s=6)
            # ax4.scatter(pred_step[i,:,0], pred_step[i,:,1], color=get_color_gradient(color1_pred, color2_pred, WINDOW_LENGTH-HISTORY_LENGTH-1), label='PRED', s=6)
            # ax4.scatter(human_history_list[i,:,0], human_history_list[i,:,1], color=get_color_gradient(color1_history, color2_history, HISTORY_LENGTH), label='HIST', s=6)
            # ax4.scatter(human_real_translation_list[i+1:i+WINDOW_LENGTH-HISTORY_LENGTH-1,0], human_real_translation_list[i+1:i+WINDOW_LENGTH-HISTORY_LENGTH-1,1], color=get_color_gradient(color1_gt, color2_gt, WINDOW_LENGTH-HISTORY_LENGTH-2), label='MOCAP', s=6)
            # ax4.set_xlim(-3, 3)
            # ax4.set_ylim(-6, 6)
            # ax4.legend()
            plt.scatter(pred_step[i,:,0], pred_step[i,:,1], color=get_color_gradient(color1_pred, color2_pred, WINDOW_LENGTH-HISTORY_LENGTH-1), label='PRED', s=6)
            plt.scatter(human_history_list[i,:,0], human_history_list[i,:,1], color=get_color_gradient(color1_history, color2_history, HISTORY_LENGTH), label='HIST', s=6)
            plt.scatter(human_real_translation_list[i+1:i+WINDOW_LENGTH-HISTORY_LENGTH-1,0], human_real_translation_list[i+1:i+WINDOW_LENGTH-HISTORY_LENGTH-1,1], color=get_color_gradient(color1_gt, color2_gt, WINDOW_LENGTH-HISTORY_LENGTH-2), label='MOCAP', s=6)
            plt.xlim(-3, 3)
            plt.ylim(-6, 6)
            plt.legend()
            plt.savefig("plot_" + str(i))
            plt.close()

def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

def get_color_gradient(c1, c2, n):
    c1_rgb = np.array(hex_to_RGB(c1))/255
    c2_rgb = np.array(hex_to_RGB(c2))/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors] 

if __name__ == "__main__":
    main()