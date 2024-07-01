import pickle
import numpy as np
import matplotlib.pyplot as plt

from hst_infer.node_config import *

pickle_file_path = PICKLE_DIR_PATH / "evaluation_data.pkl"

def main():

    counter = 0

    human_real_translation_list = list()
    human_RGBD_translation_list = list()
    human_HST_res_list = list()

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
                human_t_motion_capture = dict_to_read["human_T"]
                humanID_in_window = dict_to_read["human_id_set_in_window"]
                id2idx_in_window = dict_to_read["human_id_2_array_idx"]

                print(f"current human ID: {humanID_in_window}")
                print(f"human T from Motion Capture:    {human_t_motion_capture}")
                # print(f"valid human in mat: {np.nonzero(multi_human_mask_AT)}")
                # print(f"nonzero human pose: {np.nonzero(human_pos_ATD_stretch)}")

                if counter <= 6:
                    continue

                # ######## Single Human Condition ##############################################################################
                #####################################################################################################################
                human_real_translation_list.append(human_t_motion_capture[:2])          # list of np.array (2,), finally get (t,2) array

                human_idx = id2idx_in_window[humanID_in_window[0]]
                human_RGBD_translation_list.append(human_pos_ATD_stretch[human_idx,HISTORY_LENGTH,:2])

                print(f"human T from RGBD:              {human_pos_ATD_stretch[human_idx,HISTORY_LENGTH,:2]}")
                print(f"human T diff:                   {human_t_motion_capture[:2] - human_pos_ATD_stretch[human_idx,HISTORY_LENGTH,:2]}")
                # print(f"REAL human motion: {human_real_translation_list}")

                best_mode_idx = np.argmax(agent_position_prob)
                human_pos_predict = multi_human_pos_ATMD[human_idx,:,best_mode_idx,:2].reshape((19,2))  # (T,2)
                print(human_pos_predict.shape)
                human_HST_res_list.append(human_pos_predict)

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

            except EOFError:
                break
    
    print(f"steps: {counter}")
    human_real_translation_list = np.array(human_real_translation_list)
    # print(f"Motion Capture {human_real_translation_list.shape}\n{human_real_translation_list}")
    human_HST_res_list = np.array(human_HST_res_list)
    # print(f"HST Prediction {human_HST_res_list.shape}\n{human_HST_res_list}")
    human_RGBD_translation_list = np.array(human_RGBD_translation_list)
    # print(f"RGBD Projection {human_RGBD_translation_list.shape}\n{human_RGBD_translation_list}")

    # Errors ###############################################################################################################
    ################################################################################################################

    ## Motion Capture v.s. RGBD
    mc_rgbd_diff = human_real_translation_list - human_RGBD_translation_list
    mean_error = np.mean(mc_rgbd_diff, axis=0)
    normalized_error = (human_real_translation_list - mean_error) - human_RGBD_translation_list # We consider it as system error
    print(f"Motion Capture vs RGB-D\nMean errors on axis X and Y: {mean_error}")
    
    # print(mc_rgbd_diff.shape)
    fig = plt.figure(figsize=(16,16))
    ax1 = fig.add_subplot(2,2,1)
    ax1.set_title('Raw Errors')
    ax2 = fig.add_subplot(2,2,2)
    ax2.set_title('Performance: L2 Errors')
    ax1.scatter(np.arange(mc_rgbd_diff.shape[0]), mc_rgbd_diff[:,0], label='err_X', color='r', s=6)
    ax1.scatter(np.arange(mc_rgbd_diff.shape[0]), mc_rgbd_diff[:,1], label='err_Y', color='b', s=6)
    ax1.legend()
    ax2.scatter(np.arange(mc_rgbd_diff.shape[0]), np.linalg.norm(mc_rgbd_diff, axis=1), label='L2 norm', s=6)
    ax2.legend()
    ax3 = fig.add_subplot(2,2,3)
    ax3.set_title(f'Exclude System Error on X:{mean_error[0]} Y:{mean_error[1]}')
    ax3.scatter(np.arange(mc_rgbd_diff.shape[0]), normalized_error[:,0], label='err_X with offset', color='r', s=6)
    ax3.scatter(np.arange(mc_rgbd_diff.shape[0]), normalized_error[:,1], label='err_Y with offset', color='b', s=6)
    ax3.legend()
    ax4 = fig.add_subplot(2,2,4)
    ax4.scatter(np.arange(mc_rgbd_diff.shape[0]), np.linalg.norm(normalized_error, axis=1), label='L2 norm with offset', s=6)
    ax4.legend()
    plt.show()

if __name__ == "__main__":
    main()