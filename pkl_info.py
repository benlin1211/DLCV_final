import pickle

files = [
    # "./scannet_200/feature_data/clip_feats_scannet_200.pkl",
    #"./scannet_200/feature_data/dataset_frequencies.pkl",
    #"./scannet_200/feature_data/scannet200_category_weights.pkl", # 190? 
    #"./scannet_200/feature_data/tail_split_inst_sampling_weights.pkl", 
    "./scannet_200/feature_data/common_tail_split_inst_sampling_weights.pkl",
    #"./scannet_200/feature_data/common_double_tail_split_inst_sampling_weights.pkl",
    #"./scannet_200/feature_data/full_train_bbs_with_rels.pkl", 
]

for file_name in files:
    with open(file_name, 'rb') as f:
        new_dict = pickle.load(f)
    print(file_name)
    # c=0
    for k, v in new_dict.items():
        print(k, v, end=" ")
        # c+=1
    print(c)
    print("\n",len(new_dict))
    breakpoint()