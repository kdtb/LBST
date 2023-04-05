import os
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from lightning.pytorch import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
import random

import config

# Create .csv file



class createCSV:
    def __init__(self, base_dir, all_csv, train_csv, val_csv, test_csv, label_column, test_size):
        super().__init__()
        self.base_dir = base_dir
        self.all_csv = all_csv
        self.train_csv = train_csv
        self.val_csv = val_csv
        self.test_csv = test_csv
        self.label_column = label_column
        self.test_size = test_size
        
    def set_all_seeds(seed):
        os.environ["PL_GLOBAL_SEED"] = str(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        
    def __df__(self):
        return
        base_path = self.all_csv
        target_dirs = os.listdir(base_path)
        print(target_dirs)

        ## Create 2 separate df's: one for Approved images, one for NonApproved, containing file_name and label

        ### Assign label = 0 to Approved images
        approved = pd.DataFrame(
            data=os.listdir(os.path.join(base_path, target_dirs[1])), columns=[self.label_column]
        )
        approved = approved.assign(label=0)

        ### Assign label = 1 to NonApproved images
        nonapproved = pd.DataFrame(
            data=os.listdir(os.path.join(base_path, target_dirs[2])), columns=[self.label_column]
        )
        nonapproved = nonapproved.assign(label=1)

        ## Merge into 1 df
        df = pd.concat([approved, nonapproved])

        ## Add parcel_id column containing character 3-10 from file_name column
        df["parcel_id"] = df[self.label_column].str[3:10]

        ## Write .csv
        df.to_csv(
            self.all_csv,
            sep=",",
            encoding="utf-8",
            index=False,
        )


        ## Split group by

        set_all_seeds(config.SEED)

        from sklearn.model_selection import GroupShuffleSplit

        splitter = GroupShuffleSplit(test_size=self.test_size, n_splits=2, random_state=config.SEED)
        split = splitter.split(df, groups=df.parcel_id)
        train_inds, test_inds = next(split)

        train_set = df.iloc[train_inds]
        test_set = df.iloc[test_inds]


        print(test_set.to_string())
        print("Test set length:", len(test_set), "Train set length:", len(train_set), sep="\n")


        ## Split train into 80/20 train/val

        train_set2 = train_set
        splitter2 = GroupShuffleSplit(test_size=self.test_size, n_splits=1, random_state=config.SEED)
        split2 = splitter2.split(train_set2, groups=train_set2.parcel_id)
        train_inds2, val_inds = next(split2)

        train_set2 = train_set.iloc[train_inds2]
        val_set = train_set.iloc[val_inds]


        ## Save to csv

        train_set2.to_csv(
            self.train_csv,
            sep=",",
            encoding="utf-8",
            index=False,
        )
        val_set.to_csv(
            self.val_csv,
            sep=",",
            encoding="utf-8",
            index=False,
        )
        test_set.to_csv(
            self.test_csv,
            sep=",",
            encoding="utf-8",
            index=False,
        )
        
