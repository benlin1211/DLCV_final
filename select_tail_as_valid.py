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
for id_, label_name in zip(CLASS_LABELS_200, VALID_CLASS_IDS_200):
    mapping[id_] = label_name

#print("mapping",mapping)
#print(len(mapping))

def create_instance_ply(data_root, data_paths):

    tail_file = []
    skip_id = []

    #print(VALID_CLASS_IDS_200)

    for i, data_path in enumerate(tqdm(data_paths)):
        fullply_f = os.path.join(data_root, data_path)
        query_pointcloud = read_plyfile(fullply_f)
        # query_xyz = query_pointcloud[:,0:3] 
        # query_rgb = query_pointcloud[:,3:6] 
        query_label = query_pointcloud[:, 6]
        # query_instance = query_pointcloud[:, 7]

        for j, label_name in enumerate(TAIL_CATS_SCANNET_200):
            
            label_id = mapping[label_name]
            ids = np.where(query_label==label_id)
            # print(ids)
            if len(ids[0])!=0 and data_path and label_id not in skip_id:
                tail_file.append(data_path)
                skip_id.append(label_id)
                if len(skip_id)==66:
                    skip_id=[]
                break
    with open('val.txt', 'w') as f:
        for n in tail_file:
            f.write(f"{n}\n")

    # with open('train.txt', 'w') as f:
    #     for n in data_path_train:
    #         if n not in tail_file:
    #             f.write(f"{n}\n")

if __name__=="__main__":


    data_root = "/home/pywu/LanguageGroundedSemseg/scannet_200"
    data_path_train = read_txt(os.path.join(data_root, 'train.txt'))

    create_instance_ply(data_root, data_path_train)



