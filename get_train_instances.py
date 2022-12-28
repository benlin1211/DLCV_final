import pandas as pd
from lib.pc_utils import read_plyfile, save_point_cloud
from lib.utils import read_txt
import os
import numpy as np
from lib.constants.scannet_constants import *
from lib.constants.dataset_sets import HEAD_CATS_SCANNET_200, COMMON_CATS_SCANNET_200, TAIL_CATS_SCANNET_200
from tqdm import tqdm, trange

CLASS_LABELS_200 # label name
VALID_CLASS_IDS_200 # id
TAIL_CATS_SCANNET_200 # label name

mapping = {}
for label_name, id_ in zip(CLASS_LABELS_200, VALID_CLASS_IDS_200):
    # print(label_name, id_) 
    mapping[label_name] = id_

def create_instance_ply(data_root, data_paths, output_path):

    os.makedirs(output_path, exist_ok=True)
    for label_name in CLASS_LABELS_200:
        os.makedirs(os.path.join(output_path, label_name), exist_ok=True)

    pbar = tqdm(data_paths)
    tail_class = VALID_CLASS_IDS_200
    for i, data_path in enumerate(pbar):
        fullply_f = os.path.join(data_root, data_path)
        query_pointcloud = read_plyfile(fullply_f)
        # query_xyz = query_pointcloud[:,0:3] 
        # query_rgb = query_pointcloud[:,3:6] 
        query_label = query_pointcloud[:, 6]
        # query_instance = query_pointcloud[:, 7]
    
        for j, label_name in enumerate(COMMON_CATS_SCANNET_200):
            # print(label_name)
            label_id = mapping[label_name]
            ids = np.where(query_label==label_id)

            if len(ids[0])!=0: # found instance with corresponding label
                result_pointcloud = np.squeeze(query_pointcloud[ids, :])

                # save different instance as ply
                instance_ids = np.unique(result_pointcloud[:,7])
                for instance_id in instance_ids:
                    save_ply_as = os.path.join(output_path, label_name, f"{i}_{int(instance_id)}.ply")
                    # print("Save", save_ply_as)
                    pbar.set_description("Save", save_ply_as)
                    save_point_cloud(
                        result_pointcloud,
                        save_ply_as,
                        with_label=True,
                        verbose=False)

        for j, label_name in enumerate(TAIL_CATS_SCANNET_200):
            # print(label_name)
            label_id = mapping[label_name]
            ids = np.where(query_label==label_id)

            if len(ids[0])!=0: # found instance with corresponding label
                result_pointcloud = np.squeeze(query_pointcloud[ids, :])

                # save different instance as ply
                instance_ids = np.unique(result_pointcloud[:,7])
                for instance_id in instance_ids:
                    save_ply_as = os.path.join(output_path, label_name, f"{i}_{int(instance_id)}.ply")
                    # print("Save", save_ply_as)
                    pbar.set_description("Save", save_ply_as)
                    save_point_cloud(
                        result_pointcloud,
                        save_ply_as,
                        with_label=True,
                        verbose=False)
                        
if __name__=="__main__":

    output_path_all = "./scannet_200/train/train_instances/"

    data_root = "/home/pywu/final-project-challenge-2-peiyuanwu/scannet_200"
    data_path_all = read_txt(os.path.join(data_root, 'train.txt'))

    # create_instance_ply(data_root, data_path_train, output_path_train)
    # create_instance_ply(data_root, data_path_val, output_path_val)
    create_instance_ply(data_root, data_path_all, output_path_all)
