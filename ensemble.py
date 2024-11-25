from tqdm import tqdm
import pandas as pd
import os


blank = pd.read_csv("Your Path")
root_path = "Your Path"
csv_paths = []
for fold_name in os.listdir(root_path):
    fold_path = os.path.join(root_path, fold_name)
    csv_paths += [os.path.join(fold_path, path) for path in os.listdir(fold_path) if ".csv" in path]
result = {qid: [] for qid in blank.Id}
for cp in csv_paths:
    pred = pd.read_csv(cp)
    for i in tqdm(range(len(pred))):
        qid, ans = pred.iloc[i]
        result[qid].append(ans)
final_anss = []
for qid, anss in result.items():
    filtered_anss = list(filter(lambda x: len(str(x)) < 30, anss))
    count_dict = {}
    for fa in filtered_anss:
        if fa in count_dict:
            count_dict[fa] += 1
        else:
            count_dict[fa] = 1
    sorted_anss = sorted([(k, v) for k, v in count_dict.items()], key=lambda x:x[1])
    ans = sorted_anss[-1][0] if sorted_anss else ""
    final_anss.append(ans)
            
        
blank.Predicted = final_anss
blank.to_csv('Your Path', index=False)
