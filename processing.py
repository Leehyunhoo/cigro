from collections import Counter
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

'''
def Filtering(args, ann_files):
    c = Counter()
    c.update(list(ann_files['label']))

    if args.trash_conf != -1:
        print('trash_conf: ', args.trash_conf)
        trash_start = [x[1] for x in c.most_common()].index(args.trash_conf)
        trash_label = [x[0] for x in c.most_common()][trash_start:]

        new_ann_files = []
        for i in tqdm(range(len(ann_files))):
            if ann_files.loc[i]['label'] not in trash_label:
                new_ann_files.append(ann_files.loc[i])

        ann_files = pd.concat(new_ann_files, axis = 1).transpose().reset_index(drop = True)
    print('total data length: ', len(ann_files))
    return ann_files

'''

def Filtering(args, ann_files):
    new_ann_files = []

    label_threshold = args.label_threshold
    label_groups = list(ann_files.groupby('label'))

    for i in range(len(label_groups)):
        if len(label_groups[i][1]) >= label_threshold:
            new_ann_files.append(label_groups[i][1])

    return pd.concat(new_ann_files).reset_index(drop = True)


def PreProcessing(args, ann_files):
    ann_files = Filtering(args, ann_files)
    skf = StratifiedKFold(n_splits = args.fold, random_state = 42, shuffle = True)

    folds = []
    labelsss = list(ann_files['label'])
    for idx, (train_idx, val_idx) in enumerate(skf.split(ann_files, labelsss)):
        folds.append((train_idx, val_idx))
        
    train_idx, val_idx = folds[0]
        
    train_ann_files = pd.DataFrame(np.array(ann_files)[train_idx])
    val_ann_files = pd.DataFrame(np.array(ann_files)[val_idx])
    print('train data length: ', len(train_ann_files))
    print('val data length: ', len(val_ann_files))
    
    
    train_ann_files.columns = ["text", "label"]
    val_ann_files.columns = ["text", "label"]
    
    return train_ann_files, val_ann_files
    
