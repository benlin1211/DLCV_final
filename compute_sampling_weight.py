import pandas as pd
from lib.pc_utils import read_plyfile, save_point_cloud
from lib.utils import read_txt
import os
import numpy as np
from lib.constants.scannet_constants import *
from lib.constants.dataset_sets import HEAD_CATS_SCANNET_200, COMMON_CATS_SCANNET_200, TAIL_CATS_SCANNET_200
from tqdm import tqdm, trange
import pickle

CLASS_LABELS_200 # label name
VALID_CLASS_IDS_200 # id
TAIL_CATS_SCANNET_200 # label name

#print(CLASS_LABELS_200 == HEAD_CATS_SCANNET_200+COMMON_CATS_SCANNET_200+TAIL_CATS_SCANNET_200)
# sets = HEAD_CATS_SCANNET_200+COMMON_CATS_SCANNET_200+TAIL_CATS_SCANNET_200
mapping = {}
for label_name, id_ in zip(CLASS_LABELS_200, VALID_CLASS_IDS_200):
    # print(label_name, id_) 
    mapping[label_name] = id_

def inverse(weight_dict):
    for k, v in weight_dict.items():
        # inverse
        # print(k, v)
        weight_dict[k] = 1/(v)
    return weight_dict

def compute_instance_freq(data_root, data_paths, output_path, dump_head=False, dump_common=True, dump_tail=True):
    
    # init
    weight_head = {}
    for label_name in HEAD_CATS_SCANNET_200:
        label_id = mapping[label_name]
        weight_head[label_id] = 0.0

    weight_commom = {}
    for label_name in COMMON_CATS_SCANNET_200:
        label_id = mapping[label_name]
        weight_commom[label_id] = 0.0

    weight_tail = {}
    for label_name in TAIL_CATS_SCANNET_200:
        label_id = mapping[label_name]
        weight_tail[label_id] = 0.0

    # compute freq
    pbar = tqdm(data_paths)
    tail_class = VALID_CLASS_IDS_200
    for i, data_path in enumerate(pbar):
        fullply_f = os.path.join(data_root, data_path)
        query_pointcloud = read_plyfile(fullply_f)
        # query_xyz = query_pointcloud[:,0:3] 
        # query_rgb = query_pointcloud[:,3:6] 
        query_label = query_pointcloud[:, 6]
        # query_instance = query_pointcloud[:, 7]

        if dump_head:     
            for label_name in HEAD_CATS_SCANNET_200:
                label_id = mapping[label_name]
                ids = np.where(query_label==label_id)
                weight_head[label_id] += len(ids[0])

        if dump_common:
            for label_name in COMMON_CATS_SCANNET_200:
                label_id = mapping[label_name]
                ids = np.where(query_label==label_id)
                weight_commom[label_id] += len(ids[0]) 
               
        if dump_tail:
            for label_name in TAIL_CATS_SCANNET_200:
                label_id = mapping[label_name]
                ids = np.where(query_label==label_id)
                weight_tail[label_id] += len(ids[0])

    # update weight
    if dump_head:     
        weight_head = inverse(weight_head)
    if dump_common:
        weight_commom = inverse(weight_commom)
    if dump_tail:
        weight_tail = inverse(weight_tail)

    weight_final = {}
    weight_final.update(weight_head)
    weight_final.update(weight_commom)
    weight_final.update(weight_tail)

    save_as = os.path.join(output_path, "./common_tail_split_inst_sampling_weights.pkl")
    with open(save_as, 'wb') as f:
        pickle.dump(weight_final, f)

    for k, v in weight_final.items():
        # inverse
        print(k, v)

if __name__=="__main__":

    output_path = "./scannet_200/feature_data/"
    os.makedirs(output_path, exist_ok=True)

    data_root = "./scannet_200"
    data_path_all = read_txt(os.path.join(data_root, 'train.txt'))

    # create_instance_ply(data_root, data_path_train, output_path_train)
    # create_instance_ply(data_root, data_path_val, output_path_val)
    compute_instance_freq(data_root, data_path_all, output_path, 
                          dump_head=False, dump_common=True, dump_tail=True)