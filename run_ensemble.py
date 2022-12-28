import os
import numpy as np
from tqdm import tqdm, trange
from collections import Counter
from tqdm import tqdm
import random

random.seed(1211)

# read result for a txt file
def read_txt(path):
    preds = []
    with open(path, 'rb') as f:
        lines = f.readlines()
        # i=0
        for p in lines:
            p = int(str(p, encoding="utf-8").rstrip('\n'))
            preds.append(p)
            # i+=1
            # if i>10:
            #     break

    return preds

# read txt files
def read_path(path):
    filenames = []
    with open(path, 'rb') as f:
        lines = f.readlines()
        # i=0
        for line in lines:
            line = str(line, encoding="utf-8").rstrip('\n')
            line = line.split("/")[1].split(".")[0] + ".txt"
            filenames.append(line)
            # i+=1
            # if i>5:
            #     break
    return filenames, len(filenames)

# read results for a folder
def get_pred(txt_path, filenames):
    pred_list = []
    num_list = []
    #print(filenames)

    for filename in tqdm(filenames):
        path = os.path.join(txt_path, filename)
        preds = read_txt(path)
        # for p in preds:
        #     print(p, end=" ")
        # print(len(preds))
        pred_list.append(preds)
        num_list.append(len(preds))
        
    # print("num of file:", len(pred_list))
    return pred_list, num_list

def voting(numbers):
    counts = Counter(numbers)
    max_count = counts.most_common(1)[0][1]
    out = [value for value, count in counts.most_common() if count == max_count]
    if len(out) > 1:
        out = [random.choice(out)]
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

if __name__=="__main__":
    
    filenames, num_files = read_path("./scannet_200/test.txt")
    txt_paths = ["./submit_tail_continue_001/","./submit_tail6033_002/", "./toy2_003/"]
    outpath = "./ensemble"
    os.makedirs(outpath, exist_ok=True)

    elective = []

    for txt_path in txt_paths:
        print(f"Loading txts from file {txt_path}")
        pred_list, num_list = get_pred(txt_path, filenames)
        elective.append( pred_list )

    print(len(num_list))
    print(num_list[0])
    
    # print(elective)
    # print(elective[0]) # pred_list, final_pred_list
    # print(elective[0][0]) # preds, final_preds
    # print(elective[0][0][0]) # p_i, p

    print("Voting...")
    final_pred_list = []
    for i in trange(num_files):
        final_preds = []
        for j in range(num_list[i]):
            #print(pred_list)
            vote_box = []
            for pred_list in elective:
                vote_box.append(pred_list[i][j]) # 3
                # print(len(vote_box))

            p = voting(vote_box)
            print(vote_box, "=>", p)
            final_preds.append(p)
        final_pred_list.append(final_preds)

    print("Saving result txts...")
    for i, filename in enumerate(tqdm(filenames)):
        final_preds = np.array(final_pred_list[i])
        # print(final_preds)
        # print(len(preds))
        save_as = os.path.join(outpath, filename)
        np.savetxt(save_as, final_preds, fmt='%i')