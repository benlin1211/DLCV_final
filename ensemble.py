import os
import numpy as np
from tqdm import tqdm, trange
from collections import Counter
from tqdm import tqdm

list_1 = [1,2,3]
list_2 = [4,5,6]

# Create an empty list
list_ = []

# Create List of lists
list_.append(list_1)
list_.append(list_2)
print (list_)
print(list_[1])
print(list_[1][1])

# read txt files
def read_path(path):
    filenames = []
    with open(path, 'rb') as f:
        lines = f.readlines()
        for line in lines:
            line = str(line, encoding="utf-8").rstrip('\n')
            line = line.split("/")[1].split(".")[0] + ".txt"
            filenames.append(line)
    return filenames, len(filenames)

# read result for a file
def read_txt(path):
    preds = []
    with open(path, 'rb') as f:
        lines = f.readlines()
        for p in lines:
            p = int(str(p, encoding="utf-8").rstrip('\n'))
            preds.append(p)
    return preds

# read results for a folder
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

if __name__=="__main__":
    
    filenames, num_files = read_path("./scannet_200/test.txt")
    txt_paths = ["./submit_tail_continue_001/","./submit_tail_continue_001/"]
    outpath = "./ensemble"
    os.makedirs(outpath, exist_ok=True)

    elective = []

    for txt_path in txt_paths:
        print(f"Loading txts from file {txt_path}")
        pred_list = get_pred(txt_path, filenames)
        elective.append( pred_list )

    # print(elective)
    # print(elective[0]) # pred_list, final_pred_list
    # print(elective[0][0]) # preds, final_preds
    # print(elective[0][0][0]) # p_i, p

    print("Voting...")
    final_pred_list = []
    for i in trange(num_files):
        for pred_list in elective:
            #print(pred_list)
            final_preds = []
            for preds in pred_list:
                vote_box = []
                for p_i in preds:
                    vote_box.append(p_i)
                p = voting(vote_box)
            final_preds.append(p)
        final_pred_list.append(final_preds)

    print("Saving result txts...")
    for i, filename in enumerate(tqdm(filenames)):
        preds = np.array(final_pred_list[i])
        # print(len(preds))
        save_as = os.path.join(outpath, filename)
        np.savetxt(save_as, preds, fmt='%i')