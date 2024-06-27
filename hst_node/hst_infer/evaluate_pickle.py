import pickle
import numpy as np
from hst_infer.node_config import *

pickle_file_path = PICKLE_DIR_PATH / "evaluation_data.pkl"

def main():

    counter = 0

    human_real_translation_list = list()
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
                print(f"human_t: {human_t_motion_capture}")
                print(f"valid human in mat: {np.nonzero(multi_human_mask_AT)}")
                print(f"nonzero human pose: {np.nonzero(human_pos_ATD_stretch)}")

                if counter <= 6:
                    continue

                # ######## Single Human Condition #######################################
                human_real_translation_list.append(human_t_motion_capture[:2])          # list of np.array (2,), finally get (t,2) array
                # print(f"REAL human motion: {human_real_translation_list}")
                best_mode_idx = np.argmax(agent_position_prob)
                human_idx = id2idx_in_window[humanID_in_window[0]]
                human_pos_predict = multi_human_pos_ATMD[human_idx,:,best_mode_idx,:2].reshape((19,2))  # (T,2)
                print(human_pos_predict.shape)
                human_HST_res_list.append(human_pos_predict)

                # build the time series data
                human_idx_in_window = [id2idx_in_window[humanID] for humanID in humanID_in_window]
                # print(f"current human idx: {human_idx_in_window}")

                # 0-4 is history, 5 is present, 6+ is future
                # skip first 5 steps

                #               human id                idx                positions         desired pos      vs GroundTrue
                # humanID_in_window -> id2idx_in_window -> human_pos_HST_ATMD -> get future pos -> human_t_motion_capture

                # 1. skeleton extractor performance
                
                # 2. HST performance


                # break
                input()

            except EOFError:
                break
    
    print(counter)
    human_motion_real_T2 = np.array(human_real_translation_list).reshape((counter,2))       # single human

if __name__ == "__main__":
    main()