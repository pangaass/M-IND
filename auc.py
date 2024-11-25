from sklearn.metrics import average_precision_score,roc_auc_score
import argparse
import json
import os
import re
parser = argparse.ArgumentParser(description='Test for argparse')

parser.add_argument('--target', type=str, default="./pred.json", help='prediction file')
parser.add_argument('--src', type=str, default="/workspace/pangyunhe/project/IND-WhoIsWho/ind_valid_author_ground_truth.json", help='prediction file')

args = parser.parse_args()

def cal_auc_map(pred, ground_truth):
    data_dict = pred
    labels_dict = ground_truth

    total_w = 0
    total_auc = 0
    total_ap = 0
    for aid in labels_dict:
        cur_normal_data = labels_dict[aid]["normal_data"]
        cur_outliers = labels_dict[aid]["outliers"]
        cur_labels = []
        cur_preds = []
        cur_w = len(cur_outliers)
        for item in cur_normal_data:
            cur_labels.append(1)
            cur_preds.append(data_dict[aid][item])
            # cur_preds.append(1)
        for item in cur_outliers:
            cur_labels.append(0)
            cur_preds.append(data_dict[aid][item])
            # cur_preds.append(0)
        cur_auc = roc_auc_score(cur_labels, cur_preds)
        cur_map = average_precision_score(cur_labels, cur_preds)
        total_ap += cur_w * cur_map
        total_auc += cur_w * cur_auc
        total_w += cur_w
        
    mAP = total_ap / total_w
    avg_auc = total_auc / total_w
    return avg_auc,mAP 

if __name__ == "__main__":
    #如果args.target是一个文件夹，则循环访问文件夹下所有的json文件并计算auc
    #如果args.target是一个文件，则直接计算auc
    with open(args.src, 'r') as f:
        ground_truth = json.load(f)

    if os.path.isdir(args.target):
        json_files = [f for f in os.listdir(args.target) if f.endswith('.json')]
        try:
            #使用正则表达式取得json_file中的数字转化成int
            json_files_steps = [int(re.findall(r'\d+', f)[0]) for f in json_files]
            #根据json_files_steps 对json_file进行由小到大排序
            json_files = [f for _, f in sorted(zip(json_files_steps, json_files))]
        except:
            pass
        for json_file in json_files:
            try:
                with open(os.path.join(args.target, json_file), 'r') as f:
                    pred = json.load(f)
                auc,mAP = cal_auc_map(pred, ground_truth)
                
                print(json_file,auc)
            except:
                pass
    else:
        with open(args.target, 'r') as f:
            pred = json.load(f)
        auc,mAP = cal_auc_map(pred, ground_truth)
        print("AUC:",auc,"mAP:",mAP)
