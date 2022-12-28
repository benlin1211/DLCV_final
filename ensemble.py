import os
import numpy as np
from tqdm import tqdm, trange
from collections import Counter

def read_path(path):
    filenames = []
    with open(path, 'rb') as f:
        lines = f.readlines()

        for line in lines:
            line = str(line, encoding="utf-8").rstrip('\n')
            line = line.split("/")[1].split(".")[0] + ".txt"
            filenames.append(line)
    return filenames, len(filenames)

from tqdm import tqdm
def read_txt(path):
    filenames = []
    with open(path, 'rb') as f:
        lines = f.readlines()

        for line in lines:
            line = int(str(line, encoding="utf-8").rstrip('\n'))
            filenames.append(line)
    return filenames


def get_pred(txt_path, filenames):
    pred_list = []
    #print(filenames)
    for filename in tqdm(filenames):
        path = os.path.join(txt_path, filename)
        preds = read_txt(path)
        # for p in preds:
        #     print(p, end=" ")
        # print(len(preds))
        pred_list.append(preds)
        # breakpoint()
    # print("num of file:", len(pred_list))
    return pred_list

def voting(numbers):
    counts = Counter(numbers)
    max_count = counts.most_common(1)[0][1]
    out = [value for value, count in counts.most_common() if count == max_count]
    return out

# def most_frequent(List):
# 	counter = 0
# 	num = List[0]
# 	for i in List:
# 		curr_frequency = List.count(i)
# 		if(curr_frequency> counter):
# 			counter = curr_frequency
# 			num = i
# 	return num

# def most_common(lst):
#     return max(set(lst), key=lst.count)

filenames, num_files = read_path("./scannet_200/test.txt")
txt_paths = ["./submit_tail_continue_001/","./submit_tail_continue_001/"]
outpath = "./ensemble"
os.makedirs(outpath, exist_ok=True)

elective = []

for txt_path in txt_paths:
    print(f"Loading txts from file {txt_path}")
    pred_list = get_pred(txt_path, filenames)
    print(len(pred_list))
    elective.append( pred_list )
    print(len(elective))

print("Voting...")
final_pred_list = []
for i in trange(num_files):
    vote_box = []
    for pred_list in elective:
        #print(pred_list)
        vote_box.append(pred_list[i])

    p = voting(vote_box)
    final_pred_list.append(p)

print("Saving result txts...")
for i, filename in enumerate(tqdm(filenames)):
    preds = np.array(final_pred_list[i])
    # print(len(preds))
    save_as = os.path.join(outpath, filename)
    np.savetxt(save_as, preds, fmt='%i')